"""Private SQLite codec for normalized UNIFIED_PURCHASE_V1 order observations."""

from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import groupby
import json
import sqlite3

from Application.mindbox.selection import DEFAULT_ORDER_NAMESPACES
from Application.statistics_facts import SCHEMA as LEGACY_SCHEMA, Projector, dump, microseconds
from Application.statistics_storage_safety import StatisticsIndexError
from Application.statistics_unified_facts import OrderLineState, OrderObservation, reconcile_order
from Application.statistics_unified_policy import EvidenceQuality, Measured, classify_status_code

SCHEMA_VERSION = 4
PROJECTION_ID = "STATISTICS_OBSERVATIONS_V1"
OBSERVATION_VERSION = 1
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Reuse the v3 action, item and private dictionary tables. Orders have their
# own schema and never go through the lossy v3 order projector.
SCHEMA = LEGACY_SCHEMA.split("CREATE TABLE orders(", 1)[0] + """
CREATE TABLE line_ids(
 sid INTEGER PRIMARY KEY, order_scope INTEGER NOT NULL, source TEXT NOT NULL,
 UNIQUE(order_scope, source));
CREATE TABLE order_observations(
 occurrence_id INTEGER PRIMARY KEY, source_entry INTEGER NOT NULL, part INTEGER NOT NULL,
 source_ordinal INTEGER NOT NULL, order_sid INTEGER, customer_sid INTEGER, order_rank INTEGER,
 created_us INTEGER, raw_fingerprint BLOB, lines_quality TEXT NOT NULL,
 raw_line_count INTEGER NOT NULL, fields TEXT NOT NULL,
 UNIQUE(source_entry, part, source_ordinal));
CREATE TABLE order_line_observations(
 occurrence_id INTEGER NOT NULL REFERENCES order_observations(occurrence_id),
 ordinal INTEGER NOT NULL, line_sid INTEGER, line_number TEXT, namespace_mask INTEGER NOT NULL,
 input_quantity TEXT, fields TEXT NOT NULL, PRIMARY KEY(occurrence_id, ordinal)) WITHOUT ROWID;
CREATE TABLE item_metadata(
 code TEXT PRIMARY KEY REFERENCES items(code), name TEXT NOT NULL, category TEXT,
 sex TEXT, style_group TEXT);
CREATE TABLE category_metadata(code TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL);
"""

PROFILE_SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE profiles(ordinal INTEGER PRIMARY KEY, customer INTEGER NOT NULL);
CREATE INDEX profile_cohort ON profiles(customer,ordinal);
"""

INDEXES = (
    "CREATE INDEX observation_order ON order_observations(order_sid,occurrence_id)",
    "CREATE INDEX observation_date ON order_observations(created_us)",
    "CREATE INDEX action_dates ON actions(event_us)",
    "CREATE INDEX action_bad_dates ON actions(id) WHERE date_bad=1",
    "CREATE INDEX item_selection ON items(kind,collection,code)",
)

_LAYOUT = {
    "metadata": "key value",
    "items": "code kind collection",
    "actions": "id event_us date_bad name customer has_product header_bad adapter_bad details",
    "action_products": "event ordinal namespace value item",
    "order_observations": ("occurrence_id source_entry part source_ordinal order_sid customer_sid order_rank "
                           "created_us raw_fingerprint lines_quality raw_line_count fields"),
    "order_line_observations": "occurrence_id ordinal line_sid line_number namespace_mask input_quantity fields",
    "item_metadata": "code name category sex style_group",
    "category_metadata": "code name path",
}
_PROFILE_LAYOUT = {"metadata": "key value", "profiles": "ordinal customer"}


def validate_layout(db, *, profile=False):
    layout = _PROFILE_LAYOUT if profile else _LAYOUT
    actual = {name for name, in db.execute("SELECT name FROM sqlite_schema WHERE type='table' "
                                          "AND name NOT LIKE 'sqlite_%'")}
    if actual != layout.keys():
        raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")
    for name, columns in layout.items():
        if [row[1] for row in db.execute("PRAGMA table_info(" + name + ")")] != columns.split():
            raise StatisticsIndexError("INCOMPATIBLE_SCHEMA")


def _state(value):
    return value.quality.value, value.reasons


def _pack(value, encode=lambda v: v):
    return (None if value.value is None else encode(value.value), *_state(value))


def _text(value):
    if not isinstance(value, str):
        raise ValueError("Expected projected text")
    return value


def _number(value):
    value = Decimal(_text(value))
    if not value.is_finite():
        raise ValueError("Expected finite projected decimal")
    return value


def _sid(value):
    if type(value) is not int or value < 1:
        raise ValueError("Invalid projected SID")
    return str(value)


def _time(value):
    if type(value) is not int:
        raise ValueError("Invalid projected timestamp")
    return EPOCH + timedelta(microseconds=value)


def _measurement(payload, decode=lambda v: v):
    value, quality, reasons = payload
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise ValueError("Invalid projected quality reasons")
    return Measured(None if value is None else decode(value), EvidenceQuality(quality), tuple(reasons))


def _channel(value):
    code, name = value
    return _text(code), None if name is None else _text(name)


_ATTRIBUTES = {"delivery_cost": _number, "delivery_type": _text,
               "ordering_method": _text, "payment_types": lambda value: tuple(map(_text, value))}


def _attributes(values):
    if [name for name, _ in values] != list(_ATTRIBUTES):
        raise ValueError("Invalid order attributes projection")
    return tuple((name, _measurement(value, _ATTRIBUTES[name])) for name, value in values)


class ObservationProjector:
    def __init__(self, db, auxiliary: Projector, check: Callable[[], None]):
        self.db, self.auxiliary, self.check = db, auxiliary, check

    def _line_sid(self, order_scope, key):
        if key is None:
            return None
        args = order_scope, key
        self.db.execute("INSERT OR IGNORE INTO line_ids(order_scope,source) VALUES(?,?)", args)
        return self.db.execute("SELECT sid FROM line_ids WHERE order_scope=? AND source=?", args).fetchone()[0]

    def order(self, occurrence, observation, raw_lines):
        order, customer = observation.order_key, observation.customer_key
        order_sid = None if order.value is None else self.auxiliary.sid(order.value, "order_ids")
        customer_sid = None if customer.value is None else self.auxiliary.sid(customer.value)
        created = observation.order_created_at
        fields = dump([_state(order), _state(customer), _state(created), _pack(observation.channel),
                       [(name, _pack(value, str if name == "delivery_cost" else lambda v: v))
                        for name, value in observation.order_attributes]])
        self.db.execute("INSERT INTO order_observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (occurrence, *observation.source_ref, order_sid, customer_sid, None,
                         None if created.value is None else microseconds(created.value),
                         observation.raw_fingerprint, observation.lines_quality.value,
                         observation.raw_line_count, fields))

        def rows():
            for line in observation.lines:
                self.check()
                raw = raw_lines[line.ordinal]
                number = raw.get("number")
                product = raw.get("product")
                ids = product.get("ids") if isinstance(product, dict) else None
                mask = sum(1 << i for i, ns in enumerate(DEFAULT_ORDER_NAMESPACES)
                           if isinstance(ids, dict) and ns in ids)
                yield (occurrence, line.ordinal, self._line_sid(order_sid or -occurrence, line.stable_line_key.value),
                       str(number) if type(number) is int else None, mask,
                       None if line.input_quantity is None else str(line.input_quantity),
                       dump([_state(line.stable_line_key), _pack(line.status_code),
                             _pack(line.nominal_units, str), _pack(line.source_price, str),
                             _pack(line.currency), _pack(line.product_key), line.product_token]))
        self.db.executemany("INSERT INTO order_line_observations VALUES(?,?,?,?,?,?,?)", rows())

    def prepare_read(self):
        self.db.executescript("""
        CREATE TABLE order_ranks(sid INTEGER PRIMARY KEY, rank INTEGER NOT NULL);
        INSERT INTO order_ranks SELECT sid,row_number() OVER(ORDER BY source COLLATE BINARY) FROM order_ids;
        UPDATE order_observations SET order_rank=(SELECT rank FROM order_ranks WHERE sid=order_sid);
        DROP TABLE order_ranks;
        """)
        for command in INDEXES:
            self.check()
            self.db.execute(command)
        self.db.commit()

    def finish(self):
        self.prepare_read()
        self.db.execute("PRAGMA secure_delete=ON")
        self.db.executescript("DROP TABLE customer_ids; DROP TABLE order_ids; DROP TABLE line_ids;")
        self.db.commit()


@dataclass(frozen=True, slots=True, repr=False)
class IndexedOrder:
    observation: OrderObservation
    order_sort_rank: int | None
    line_numbers: tuple[tuple[int, int], ...]


def _line(row):
    ordinal, line_sid, number, mask, quantity, fields = row
    if (type(ordinal) is not int or ordinal < 0
            or type(mask) is not int or not 0 <= mask < (1 << len(DEFAULT_ORDER_NAMESPACES))):
        raise ValueError("Invalid projected line")
    identity, status, units, price, currency, product, token = json.loads(fields)
    code = _measurement(status, _text)
    classified, group = classify_status_code(code.value)
    if code.value is not None and classified != code:
        raise ValueError("Invalid projected status")
    stable_key = _measurement([line_sid, *identity], _sid)
    product_token = None if token is None else tuple((_text(ns), _text(key)) for ns, key in token)
    line = OrderLineState(stable_key, ordinal, code, group, None if quantity is None else _number(quantity),
                          _measurement(units, _number),
                          Measured(None, EvidenceQuality.UNKNOWN, ("FULFILLMENT_UNVERIFIED",)),
                          _measurement(price, _number), _measurement(currency, _text),
                          _measurement(product, _text), product_token)
    return line, None if number is None else int(_text(number))


def _order(header, rows, check):
    (_, entry, part, ordinal, order_sid, customer_sid, rank, created, fingerprint,
     lines_quality, raw_line_count, fields) = header
    order, customer, date, channel, attributes = json.loads(fields)
    key = _measurement([order_sid, *order], _sid)
    if (type(raw_line_count) is not int or raw_line_count < 0
            or (rank is None) != (key.value is None)
            or rank is not None and (type(rank) is not int or rank < 1)
            or fingerprint is not None and (not isinstance(fingerprint, bytes) or len(fingerprint) != 32)):
        raise ValueError("Invalid projected order")
    lines, line_numbers = [], []
    for row in rows:
        check()
        if row[12] is not None:
            line, number = _line(row[12:])
            if line.ordinal >= raw_line_count:
                raise ValueError("Invalid projected ordinal")
            lines.append(line)
            if number is not None:
                line_numbers.append((line.ordinal, number))
    quality = EvidenceQuality(lines_quality)
    if quality is EvidenceQuality.KNOWN and len(lines) != raw_line_count:
        raise ValueError("Missing projected line")
    return IndexedOrder(
        OrderObservation((entry, part, ordinal), key, _measurement([customer_sid, *customer], _sid),
                         _measurement([created, *date], _time), fingerprint, quality, tuple(lines),
                         _measurement(channel, _channel), _attributes(attributes), raw_line_count),
        rank, tuple(line_numbers))


class ObservationReader:
    def __init__(self, db, manifest, check, *, restore=None, profile_values=None):
        self.db, self.manifest, self.check = db, manifest, check
        self._restore = restore
        self._profile_values = profile_values
        self.identity_space = "INDEXED:" + manifest["binding"]["generation"]
        self._sql_failure = None
        db.set_progress_handler(self._sql_check, 10000)

    def _sql_check(self):
        try:
            self.check()
            return 0
        except Exception as exc:
            self._sql_failure = exc
            return 1

    def orders(self):
        try:
            rows = self.db.execute("""
                SELECT o.*,l.ordinal,l.line_sid,l.line_number,l.namespace_mask,l.input_quantity,l.fields
                FROM order_observations o LEFT JOIN order_line_observations l USING(occurrence_id)
                ORDER BY o.order_sid,o.occurrence_id,l.ordinal
            """)
            with closing(rows):
                for _, group in groupby(rows, key=lambda row: row[0]):
                    self.check()
                    header = next(group)
                    row = _order(header[:12], _prepend(header, group), self.check)
                    yield row if self._restore is None else self._restore(row)
        except (sqlite3.Error, ValueError, TypeError, ArithmeticError):
            if self._sql_failure is not None:
                failure, self._sql_failure = self._sql_failure, None
                raise failure from None
            raise StatisticsIndexError("INVALID_OBSERVATION_PROJECTION") from None

    def facts(self, *, product_predicate=None):
        """Feed one complete SID group into CORE, independent of the selection."""
        for key, group in groupby(self.orders(), key=lambda row: row.observation.order_key.value):
            self.check()
            if key is None:
                for _ in group:
                    pass
                continue
            first = next(group)

            def observations():
                yield first.observation
                for row in group:
                    if row.order_sort_rank != first.order_sort_rank:
                        raise StatisticsIndexError("INVALID_OBSERVATION_PROJECTION")
                    yield row.observation

            yield reconcile_order(observations(), order_sort_rank=first.order_sort_rank,
                                  product_predicate=product_predicate, check=self.check)

    @contextmanager
    def profile_demographics(self):
        """Read pinned demographics on demand without storing personal attributes in v4."""
        if self._profile_values is None:
            raise StatisticsIndexError("PROFILE_SOURCE_UNAVAILABLE")
        with closing(self.profile_links()) as links, closing(self._profile_values(links)) as values:
            yield values

    def profile_links(self):
        """Private ordinal-to-customer SID mapping into the pinned Customers source."""
        return self.db.execute("SELECT ordinal,customer FROM demographics.profiles ORDER BY ordinal")


def _prepend(first, rest):
    yield first
    yield from rest
