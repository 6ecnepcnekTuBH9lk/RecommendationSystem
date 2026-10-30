"""Lossless statistical projections, not a copy of canonical JSON or profiles.

Adapters validate once. Query-dependent adapter errors remain deferred; a bad
unmapped timestamp, for example, must not break an unfiltered calculation.
Only local integer customer/order SIDs survive publication.
"""

from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
import json
from types import SimpleNamespace

from Application import dataset_statistics as reference
from Application.interactions import classify_action_system_name
from Application.mindbox.adapters import adapt_action, adapt_action_system_name, adapt_order
from Application.mindbox.adapters._common import AdapterError, identifier, number, objects, text, timestamp
from Application.mindbox.order_dedup import OrderSnapshots, semantic_fingerprint
from Application.order_statistics import order_for_adapter

SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE customer_ids(sid INTEGER PRIMARY KEY, source TEXT UNIQUE NOT NULL);
CREATE TABLE order_ids(sid INTEGER PRIMARY KEY, source TEXT UNIQUE NOT NULL);
CREATE TABLE items(code TEXT PRIMARY KEY, kind TEXT, collection TEXT);
CREATE TABLE actions(
 id INTEGER PRIMARY KEY, event_us INTEGER, date_bad INTEGER NOT NULL,
 name TEXT, customer INTEGER, has_product INTEGER NOT NULL,
 header_bad INTEGER NOT NULL, adapter_bad INTEGER NOT NULL, details TEXT NOT NULL);
CREATE TABLE action_products(
 event INTEGER NOT NULL REFERENCES actions(id), ordinal INTEGER NOT NULL,
 namespace TEXT NOT NULL, value TEXT NOT NULL, item TEXT,
 PRIMARY KEY(event,ordinal)) WITHOUT ROWID;
CREATE TABLE orders(
 id INTEGER PRIMARY KEY, event_us INTEGER, date_bad INTEGER NOT NULL,
 outside INTEGER NOT NULL, customer INTEGER, order_sid INTEGER, fingerprint BLOB,
 header_bad INTEGER NOT NULL, adapter_bad INTEGER NOT NULL,
 aggregate_bad INTEGER NOT NULL, details TEXT NOT NULL);
CREATE TABLE order_lines(
 event INTEGER NOT NULL REFERENCES orders(id), ordinal INTEGER NOT NULL,
 quantity TEXT NOT NULL, status TEXT, namespace TEXT, value TEXT,
 price TEXT, item TEXT, eligible INTEGER NOT NULL,
 PRIMARY KEY(event,ordinal)) WITHOUT ROWID;
"""

# Sensitive minimal snapshot lives in a separate file in the private generation.
# No original IDs; ordinal and SID bind it to the facts from the same build.
DEMOGRAPHY_SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE profiles(
 ordinal INTEGER PRIMARY KEY, customer INTEGER NOT NULL, sex TEXT, born TEXT);
CREATE INDEX profile_cohort ON profiles(customer,ordinal);
"""

INDEXES = (
    "CREATE INDEX action_dates ON actions(event_us)",
    "CREATE INDEX action_bad_dates ON actions(id) WHERE date_bad=1",
    "CREATE INDEX order_dates ON orders(event_us)",
    "CREATE INDEX order_global ON orders(id) WHERE outside=1 OR date_bad=1",
    "CREATE INDEX item_selection ON items(kind,collection,code)",
)


def microseconds(value):
    delta = value - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return delta.days * 86400000000 + delta.seconds * 1000000 + delta.microseconds


def event_date(raw, field, required):
    try:
        value = timestamp(raw, field, required=required)
        return microseconds(value) if value is not None else None, 0
    except AdapterError:
        return None, 1


def dump(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


class Projector:
    def __init__(self, connection, identities, selection, products, period):
        self.db, self.identities, self.selection = connection, identities, selection
        self.products, self.period = products, period

    def sid(self, value, table="customer_ids"):
        # Table name is internal, never input SQL. Dictionaries stay disk-backed.
        self.db.execute(f"INSERT OR IGNORE INTO {table}(source) VALUES(?)", (value,))
        return self.db.execute(f"SELECT sid FROM {table} WHERE source=?", (value,)).fetchone()[0]

    def customer(self, value):
        return self.sid(self.identities.resolve(value))

    def action(self, raw, ordinal):
        event, bad_date = event_date(raw, "dateTimeUtc", False)
        header_bad = adapter_bad = has_product = 0
        name = customer = None
        details, product_rows = {}, []
        try:
            name = adapt_action_system_name(raw)
            has_product = int(bool(objects(raw, "products")))
            source = identifier(raw, "customer.ids.mindboxId", required=False)
            customer = self.customer(source) if source is not None else None
        except AdapterError:
            header_bad = 1
        if not header_bad and classify_action_system_name(name, self.selection.interaction_rules()) is not None:
            try:
                action = adapt_action(raw, self.identities, product_namespaces=self.selection.action_product_namespaces)
                details = dict(channel_system_name=action.channel_system_name,
                               channel_external_id=action.channel_external_id, channel_name=action.channel_name,
                               price=str(action.product_view_price) if action.product_view_price is not None else None,
                               available=action.product_view_is_available)
                for index, product in enumerate(action.products):
                    item = self.products.resolve(product, strict=False).item_id
                    product_rows.append((ordinal, index, product.namespace, product.value, item))
            except AdapterError:
                adapter_bad = 1
        self.db.execute("INSERT INTO actions VALUES(?,?,?,?,?,?,?,?,?)",
                        (ordinal, event, bad_date, name, customer, has_product, header_bad, adapter_bad, dump(details)))
        self.db.executemany("INSERT INTO action_products VALUES(?,?,?,?,?)", product_rows)

    def order(self, raw, ordinal):
        event, bad_date = event_date(raw, "firstAction.dateTimeUtc", True)
        outside = int(not bad_date and not any(microseconds(a) <= event < microseconds(b) for a, b in self.period))
        customer = order_sid = fingerprint = None
        header_bad = adapter_bad = aggregate_bad = 0
        details, rows, eligible = {}, [], []
        if not bad_date:
            try:
                for index, line in enumerate(objects(raw, "lines", required=True)):
                    quantity = number(line, "quantity")
                    integral = quantity == quantity.to_integral_value()
                    # An invalid status is read before dedup only WITHOUT a filter.
                    try:
                        status = identifier(line, "status.ids.externalId") if integral else None
                    except AdapterError:
                        status = None
                    rows.append([ordinal, index, str(quantity), status, None, None, None, None, int(integral)])
                    if integral:
                        eligible.append(line)
                customer = self.customer(identifier(raw, "customer.ids.mindboxId"))
                order_sid = self.sid(identifier(raw, "ids.mindboxId"), "order_ids")
                fingerprint = semantic_fingerprint(raw)
            except AdapterError:
                header_bad = 1
            if not header_bad:
                try:
                    adapted = adapt_order(order_for_adapter({**raw, "lines": eligible}), self.identities,
                                          product_namespaces=self.selection.order_product_namespaces)
                    for row, line in zip((r for r in rows if r[-1]), adapted):
                        row[4:8] = [line.product.namespace, line.product.value, str(line.price_of_line),
                                    self.products.resolve(line.product, strict=False).item_id]
                except AdapterError:
                    adapter_bad = 1
                try:
                    key = identifier(raw, "firstAction.channel.ids.externalId", required=False)
                    # Read only the fields OrderAggregates actually consumes.
                    channel_name = text(raw, "firstAction.channel.name") if key is not None else None
                    method = text(raw, "customFields.orderingMethod")
                    delivery_type = text(raw, "customFields.deliveryType")
                    payments = [{"type": text(p, "type")} for p in objects(raw, "payments")]
                    delivery = number(raw, "deliveryCost", required=False)
                    details = dict(firstAction={"channel": {"ids": {"externalId": key}, "name": channel_name}},
                                   customFields={"orderingMethod": method, "deliveryType": delivery_type},
                                   payments=payments, deliveryCost=str(delivery) if delivery is not None else None)
                except AdapterError:
                    aggregate_bad = 1
        self.db.execute("INSERT INTO orders VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (ordinal, event, bad_date, outside, customer, order_sid, fingerprint,
                         header_bad, adapter_bad, aggregate_bad, dump(details)))
        self.db.executemany("INSERT INTO order_lines VALUES(?,?,?,?,?,?,?,?,?)", rows)

    def profiles(self, root, demographics, check):
        from Application.mindbox.adapters._common import birth_date
        path = reference.database(root)
        if not path.exists():
            return
        import sqlite3
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            for ordinal, (source_id, birthday, sex) in enumerate(db.execute(
                    "SELECT id,json_extract(raw,'$.birthDate'),json_extract(raw,'$.sex') FROM profiles ORDER BY id")):
                check()
                try:
                    born = birth_date({"birthDate": birthday})
                except AdapterError:
                    born = None
                demographics.execute("INSERT INTO profiles VALUES(?,?,?,?)", (
                    ordinal, self.customer(source_id), sex if sex in ("male", "female") else None,
                    born.isoformat() if born is not None else None))
                if ordinal % 1000 == 0:
                    self.db.commit()
                    demographics.commit()
        finally:
            db.close()


class IndexedSnapshots(OrderSnapshots):
    def accept(self, raw, *, diagnose=False):
        # Same first-eligible comparison as OrderSnapshots; hashes were computed
        # on the ORIGINAL full snapshot, before projection and integer SID mapping.
        self.raw += 1
        key, fingerprint = raw["ids"]["mindboxId"], raw["_fingerprint"]
        previous = self.fingerprints.get(key)
        if previous is None:
            self.fingerprints[key] = fingerprint
            return True
        if previous == fingerprint:
            self.identical += 1
        else:
            self.conflicting += 1
        return False


def action_record(raw, products):
    if raw["_adapter_bad"]:
        raise AdapterError("Indexed action failed source adapter validation")
    details = raw["_details"]
    return SimpleNamespace(
        action_id=str(raw["_ordinal"]), action_system_name=raw["actionTemplate"]["ids"]["systemName"],
        event_datetime_utc=raw["_date"], source_customer_id=raw["customer"]["ids"]["mindboxId"],
        customer_id=raw["customer"]["ids"]["mindboxId"], products=tuple(products),
        channel_system_name=details.get("channel_system_name"), channel_external_id=details.get("channel_external_id"),
        channel_name=details.get("channel_name"), product_view_price=Decimal(details["price"]) if details.get("price") is not None else None,
        product_view_is_available=details.get("available"))


def profile_result(db, provenance, root, as_of, check, progress, cohort=None, identities=None, scope=None, buyers=None):
    from Application.customer_demographics import AGE_LABELS, age_group, calendar_age
    from Application.dataset_statistics import Coverage, _distribution, _histogram_median
    genders, ages, groups = Counter(), Counter(), Counter()
    customers = provenance["customers"]
    count = 0 if customers["present"] else None
    if scope is not None:
        scope.profiles_available = customers['present']
    if buyers is not None:
        buyers.profiles_available = customers['present']
    summary = json.loads(customers.get("summary") or "{}")
    seen = set()
    if cohort is not None and scope is None and buyers is None:
        db.execute("CREATE TEMP TABLE cohort(customer INTEGER PRIMARY KEY)")
        db.executemany("INSERT INTO cohort VALUES(?)", ((int(c),) for c in cohort))
        rows = db.execute("SELECT p.customer,p.sex,p.born FROM cohort c CROSS JOIN demographics.profiles p INDEXED BY profile_cohort ON p.customer=c.customer ORDER BY p.ordinal")
    else:
        rows = db.execute("SELECT customer,sex,born FROM demographics.profiles ORDER BY ordinal")
    try:
        for sid, sex, birthday in rows:
            check()
            if scope is not None:
                scope.profile(str(sid))  # Same local SID representation as indexed events.
            if buyers is not None:
                buyers.profile(str(sid), sex, birthday, as_of)
            if cohort is not None:
                if ((scope is not None or buyers is not None) and str(sid) not in cohort) or sid in seen:
                    continue
                seen.add(sid)
            count += 1
            age = calendar_age(date.fromisoformat(birthday), as_of) if birthday is not None else None
            group = age_group(age) if age is not None else "Возраст не определён"
            genders["Мужчины" if sex == "male" else "Женщины" if sex == "female" else "Не указан"] += 1
            groups[group] += 1
            if age is not None:
                ages[age] += 1
    finally:
        rows.close()
        if cohort is not None and scope is None and buyers is None:
            db.execute("DROP TABLE cohort")
    kinds = summary.get("source_kinds", [summary["source_kind"]] if summary.get("source_kind") else [])
    coverage = Coverage("Customers", tuple(tuple(v) for v in summary.get("intervals", ())), tuple(kinds), summary.get("updated"))
    return (count, coverage, _distribution(genders, ("Мужчины", "Женщины", "Не указан")),
            _distribution(groups, AGE_LABELS), sum(a*n for a,n in ages.items()) / sum(ages.values()) if ages else None,
            _histogram_median(ages))
