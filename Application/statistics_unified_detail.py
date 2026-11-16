"""Version 2 detail rows from a validated, pinned Unified report scope."""

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
import json
import sqlite3

from Application import statistics_unified_rfm as rfm
from Application.analysis_filter import AnalysisFilter
from Application.customer_demographics import calendar_age
from Application.statistics_detail_backend import _csv_entry
from Application.statistics_detail_contract import CSV_DELIMITER, CSV_ENCODING, CSV_TEXT_PREFIX, csv_cell
from Application.statistics_detail_storage import DetailExportError
from Application.statistics_index import BuildGuard, GIB, manifest_digest
from Application.statistics_storage_safety import StatisticsIndexError, validate_storage_root
from Application.statistics_unified_aggregate_input import _contexts, _indexed_metadata, _orders, _period
from Application.statistics_unified_aggregates import _Money, _in_period, order_in_detail_scope
from Application.statistics_unified_dto import DETAIL_FORMAT, RESULT_SCHEMA, _wire
from Application.statistics_unified_facts import _eligible, _sum, evaluate_money
from Application.statistics_unified_index import open_unified_index
from Application.statistics_unified_policy import StatusGroup
from Application.statistics_unified_service import _binding, _index_integrity, algorithm_version

CURRENCIES = ("RUB", "KZT")
MEASURES = ("source_nominal", "known_purchase_subtotal", "recognized_purchase", "partial_nominal",
            "cancelled_nominal", "returned_nominal", "confirmed_refund", "net_after_refunds")
MONEY_COLUMNS = tuple(f"{currency}_{field}_{suffix}" for currency in CURRENCIES
                      for field in MEASURES for suffix in ("value", "known", "quality", "reasons"))
FILES = ("orders.csv", "order_lines.csv", "clients.csv", "manifest.json")
ORDER_COLUMNS = (
    "order_key", "client_key", "owner_quality", "owner_reasons", "created_utc", "channel", "channel_quality",
    "whole_state", "slice_state", "state_quality", "has_current_cancellation", "has_current_return",
    "has_current_cp", "observed_purchase", "purchase_quality", "purchase_reasons",
    "positions_observed", "purchase_positions_observed", "nominal_units_known",
    "fulfilled_units", "fulfilled_quality", "fulfillment_reasons",
    "RUB_valued_positions", "RUB_unvalued_positions", "KZT_valued_positions", "KZT_unvalued_positions", *MONEY_COLUMNS,
)
LINE_COLUMNS = (
    "order_key", "line_key", "item", "category", "product_quality", "product_reasons", "current_status",
    "current_group", "status_quality", "status_reasons", "historical_statuses", "had_cancellation",
    "had_return", "had_cp", "observed_purchase", "currency", "currency_quality",
    "nominal_units", "nominal_quality", "nominal_reasons", "fulfilled_units", "fulfilled_quality",
    "fulfillment_reasons", *MONEY_COLUMNS,
)
RFM_COLUMNS = ("in_base", "recency", "recency_quality", "recency_reasons",
               "observed_frequency", "frequency", "frequency_quality", "frequency_reasons",
               "monetary", "monetary_quality", "monetary_reasons", "known_subtotal",
               "r_score", "f_score", "m_score", "segment")
RFM_WINDOWS = ("LAST_12M", "LAST_24M", "LAST_48M")
CLIENT_COLUMNS = (
    "client_key", "profile_sex", "profile_age", "selected_orders", "selected_purchase_orders",
    "buyer_period_orders", "buyer_period_purchase_positions", "first_buyer_period_purchase",
    "last_buyer_period_purchase", "RUB_known_subtotal", "KZT_known_subtotal",
    "RUB_valued_orders", "KZT_valued_orders", "unvalued_buyer_period_orders",
    *(f"{window}_{currency}_{key}" for window in RFM_WINDOWS for currency in CURRENCIES for key in RFM_COLUMNS),
)
_SCHEMA = """
CREATE TABLE clients (
    sid TEXT UNIQUE NOT NULL, selected INTEGER NOT NULL DEFAULT 0,
    selected_buys INTEGER NOT NULL DEFAULT 0, buys INTEGER NOT NULL DEFAULT 0,
    positions INTEGER NOT NULL DEFAULT 0, first TEXT, last TEXT,
    rub TEXT, kzt TEXT, rub_valued INTEGER NOT NULL DEFAULT 0, kzt_valued INTEGER NOT NULL DEFAULT 0,
    unvalued INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE profiles (sid TEXT PRIMARY KEY, sex TEXT, age INTEGER) WITHOUT ROWID;
CREATE TABLE lines (ordinal INTEGER PRIMARY KEY, payload TEXT NOT NULL);
"""


def item_matches(selection, product, category, code, item):
    return (selection.matches(item) and (not product or code == product)
            and (not category or item.category == category))


def filter_snapshot(selection, product, category):
    return dict(selection=selection.to_dict(), product=product, category=category)


def _selected_items(db, report, request, check):
    if type(request) is not dict or set(request) != {"selection", "product", "category"}:
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    selection = AnalysisFilter.from_dict(request["selection"])
    product, category = request["product"], request["category"]
    if any(type(code) is not str or len(code) > 1024 for code in (product, category)):
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    params = report.metadata["parameters"]
    if ((params["selected_start"], params["selected_end"]) != (selection.start_date, selection.end_date)
            or any(params[field] != _wire(selection.interval)
                   for field in ("order_period", "buyer_period", "customer_period"))):
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    catalog = _indexed_metadata(db)
    selected = None
    if selection.product_restricted or product or category:
        selected = set()
        for code, item in catalog.items():
            check()
            if item_matches(selection, product, category, code, item):
                selected.add(code)
    digest = None if selected is None else manifest_digest(sorted(selected))
    if digest != params["product_filter_key"]:
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    return selected, catalog


def _measured(value):
    return value.value, value.quality.value, "|".join(value.reasons)


def _money_cells(rows):
    for row in rows:
        known = dict(row.known_nominals)
        for field in MEASURES:
            reading = getattr(row, field)
            partial = known.get(field)
            if field in ("recognized_purchase", "known_purchase_subtotal"):
                partial = row.known_purchase_subtotal.value
            yield reading.value
            yield reading.value if reading.value is not None else partial
            yield reading.quality.value
            yield "|".join(reading.reasons)


def _encoded(values):
    return tuple("" if value is None else csv_cell(value) for value in values)


def _client(db, owner):
    if owner is None:
        return None
    db.execute("INSERT OR IGNORE INTO clients(sid) VALUES (?)", (owner,))
    number = db.execute("SELECT rowid FROM clients WHERE sid=?", (owner,)).fetchone()[0]
    return f"C{number}"


def _add_amount(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return format(_sum([Decimal(a), Decimal(b)]), "f")


def _buyer_history(db, fact):
    owner = fact.customer_key.value
    if owner is None or fact.purchase.observed.value is not True:
        return
    db.execute("INSERT OR IGNORE INTO clients(sid) VALUES (?)", (owner,))
    day = fact.order_created_at.value.isoformat()
    subtotals = [row.recognized_purchase.value if row.valued_purchase_positions else None for row in fact.money]
    db.execute("""UPDATE clients SET buys=buys+1,positions=positions+?,
                  first=MIN(COALESCE(first,?),?),last=MAX(COALESCE(last,?),?),
                  rub=amount_add(rub,?),kzt=amount_add(kzt,?),rub_valued=rub_valued+?,
                  kzt_valued=kzt_valued+?,unvalued=unvalued+? WHERE sid=?""",
               (fact.purchase.known_positions, day, day, day, day,
                *(format(a, "f") if a is not None else None for a in subtotals),
                *(int(a is not None) for a in subtotals), int(all(a is None for a in subtotals)), owner))


def _line_rows(db, key, fact, catalog, check):
    for ordinal, (line, versions) in enumerate(zip(fact.resolved_lines, fact.line_history, strict=True), 1):
        check()
        codes = "|".join(sorted({entry.status_code.value for entry in versions if entry.status_code.value}))
        history = {entry.status_group for entry in versions}
        product = line.product_key.value
        row = _encoded((key, f"{key}-{ordinal}", product, catalog[product].category if product else None,
                        line.product_key.quality.value, "|".join(line.product_key.reasons),
                        *_measured(line.status_code)[:1], line.status_group.value, *_measured(line.status_code)[1:],
                        codes, StatusGroup.CANCELLED in history, StatusGroup.RETURNED in history,
                        StatusGroup.PARTIALLY_COMPLETED in history, _eligible(line),
                        line.currency.value, line.currency.quality.value, *_measured(line.nominal_units),
                        *_measured(line.fulfilled_units), *_money_cells(evaluate_money((line,), (versions,), completeness=()))))
        db.execute("INSERT INTO lines(payload) VALUES (?)", (json.dumps(row, ensure_ascii=False),))


def _order_row(key, client, whole, fact):
    owner = fact.customer_key
    channel = fact.channel.value
    return _encoded((key, client, owner.quality.value, "|".join(owner.reasons), fact.order_created_at.value,
                     channel[1] or channel[0] if channel else None, fact.channel.quality.value,
                     whole.current_kind.value, fact.current_kind.value, fact.state_quality.value,
                     fact.has_cancelled, fact.has_returned, fact.has_partial, *_measured(fact.purchase.observed),
                     len(fact.resolved_lines), fact.purchase.known_positions, fact.purchase.known_nominal_units,
                     *_measured(fact.purchase.fulfilled_units),
                     *(n for value in fact.money for n in (value.valued_purchase_positions, value.unvalued_purchase_positions)),
                     *_money_cells(fact.money)))


def _verify_counts(db, report, counts, amounts):
    orders = report.orders
    expected = (orders["unique_orders"]["known"], orders["positions"]["known"],
                orders["purchase_orders"]["known"], orders["purchase_positions"]["known"],
                orders["cancellations"]["orders"]["known"], orders["return_states"]["orders"]["known"],
                orders["buyers"]["known"])
    buyers = db.execute("SELECT COUNT(*) FROM clients WHERE selected_buys>0").fetchone()[0]
    if (*[counts[name] for name in ("orders", "lines", "buys", "positions", "cancelled", "returned")], buyers) != expected:
        raise DetailExportError("REPORT_DETAIL_COUNTS_MISMATCH")
    history = report.customers["data"]["purchases"]["counts"]
    identified = db.execute("SELECT COUNT(*),COALESCE(SUM(buys),0) FROM clients WHERE buys>0").fetchone()
    if identified != (history["buyers"]["known"], history["buyer_purchase_orders"]["known"]):
        raise DetailExportError("REPORT_DETAIL_COUNTS_MISMATCH")
    for row in amounts.result(set()):
        saved = report.money["currencies"][row.currency]
        for key in ("valued_orders", "unvalued_orders", "valued_positions", "unvalued_positions"):
            if getattr(row, key) != saved[key]:
                raise DetailExportError("REPORT_DETAIL_COUNTS_MISMATCH")
        for field in MEASURES:
            known = row.amounts[field].known
            wire = saved["amounts"][field]["known"]
            if known != (Decimal(wire) if wire is not None else None):
                raise DetailExportError("REPORT_DETAIL_COUNTS_MISMATCH")


def _write(reader, archive, workspace, control, report, request, guard):
    def check():
        control.check()
        guard.check()

    items, catalog = _selected_items(reader.db, report, request, check)
    params = report.metadata["parameters"]
    stamp = datetime.fromisoformat(report.calculated_at)
    periods = {name: _period(tuple(map(datetime.fromisoformat, value)), ()) if value else None
               for name in ("order_period", "buyer_period", "customer_period") for value in (params[name],)}
    contexts = _contexts(reader, calculated_at=stamp, selected_items=items, **periods)
    scope, buyer_scope = contexts[1], contexts[3]
    for context in (scope, buyer_scope):
        context_wire = dict(_wire(context), identity_space=report.source["provenance_digest"])
        if context_wire != report.contexts[context.context_id]:
            raise DetailExportError("REPORT_CONTEXT_MISMATCH")

    rfm_context, bounds = rfm._context(reader, stamp, params["selected_end"])
    if dict(_wire(rfm_context), identity_space=report.source["provenance_digest"]) != report.contexts["rfm_48m"]:
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    rfm_quality = Counter()

    detail_db = workspace.folder / "details.sqlite"
    with workspace.create(detail_db.name):
        pass
    validate_storage_root(workspace.folder, children=(detail_db,))
    with closing(sqlite3.connect(detail_db)) as db:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA temp_store=MEMORY")
        db.execute("PRAGMA cache_size=-32768")
        db.set_progress_handler(guard.sql_check, 10000)
        db.create_function("amount_add", 2, _add_amount, deterministic=True)
        db.create_collation("RFM_DECIMAL", rfm.decimal_compare)
        db.executescript(_SCHEMA + rfm._SCHEMA)
        cutoff = rfm_context.as_of_date or stamp.date()
        profile_age_at = stamp.astimezone().date()
        counts, amounts = Counter(), _Money()

        def orders():
            predicate = None if items is None else items.__contains__
            with closing(_orders(reader, predicate, scope, check, include_line_history=True)) as rows:
                for entry in rows:
                    rfm._spool_order(db, entry, bounds, rfm_context.orders_intervals, rfm_quality)
                    whole, fact = entry.whole, entry.selected
                    if whole is None:
                        continue
                    created = whole.order_created_at.value
                    if _in_period(created, buyer_scope.orders_intervals, buyer_scope.period):
                        _buyer_history(db, whole)
                    if not order_in_detail_scope(whole, fact, scope):
                        continue
                    counts["orders"] += 1
                    counts["lines"] += len(fact.resolved_lines)
                    purchase = fact.purchase.observed.value is True
                    counts["buys"] += purchase
                    counts["positions"] += fact.purchase.known_positions
                    counts["cancelled"] += fact.has_cancelled
                    counts["returned"] += fact.has_returned
                    currency = fact.order_currency.value
                    amounts.add(fact.money, purchase=purchase, currency=currency)
                    owner = fact.customer_key.value
                    client = _client(db, owner)
                    if client is not None:
                        db.execute("UPDATE clients SET selected=selected+1,selected_buys=selected_buys+? WHERE sid=?",
                                   (int(purchase), owner))
                    key = f"O{counts['orders']}"
                    _line_rows(db, key, fact, catalog, check)
                    yield _order_row(key, client, whole, fact)

        control.stage("orders")
        order_count = _csv_entry(archive, "orders.csv", ORDER_COLUMNS, orders(), control)
        _verify_counts(db, report, counts, amounts)
        control.stage("order_lines")
        def lines():
            for (payload,) in db.execute("SELECT payload FROM lines ORDER BY ordinal"):
                check()
                yield json.loads(payload)

        line_count = _csv_entry(archive, "order_lines.csv", LINE_COLUMNS, lines(), control)
        if (order_count, line_count) != (counts["orders"], counts["lines"]):
            raise DetailExportError("BROKEN_DETAIL_RELATIONS")
        with reader.profile_demographics() as profiles:
            for sid, birthday, sex in profiles:
                check()
                if db.execute("SELECT 1 FROM clients WHERE sid=?", (sid,)).fetchone():
                    db.execute("INSERT OR IGNORE INTO profiles VALUES (?,?,?)",
                               (sid, sex if sex in ("male", "female") else None, calendar_age(birthday, profile_age_at)))

        control.stage("clients")
        reliable_since = None
        if bounds:
            end = bounds["LAST_48M"][1]
            reliable_since = next((a for a, b in rfm_context.orders_intervals if a < end <= b), None)

        score_config = {}
        if bounds:
            for window in RFM_WINDOWS:
                saved = report.rfm["windows"][window]
                complete = saved["coverage"]["status"] == "FULL"
                for currency in CURRENCIES:
                    thresholds = saved["currencies"][currency]["thresholds"]
                    cuts = dict(thresholds, M=dict(thresholds["M"], values=tuple(map(Decimal, thresholds["M"]["values"]))))
                    score_config[window, currency] = complete, cuts

        def personal(sid):
            if not bounds:
                return (None,) * (len(RFM_WINDOWS) * len(CURRENCIES) * len(RFM_COLUMNS))
            buyer = rfm._Buyer(bounds)
            for (payload,) in db.execute("SELECT payload FROM rfm_orders WHERE customer=? ORDER BY rank", (sid,)):
                check()
                buyer.add(json.loads(payload))
            values = []
            for window in RFM_WINDOWS:
                for currency in CURRENCIES:
                    complete, thresholds = score_config[window, currency]
                    row = buyer.features(window, currency, cutoff, reliable_since, complete)
                    scores, segment = (None, None, None), None
                    if row.in_base:
                        scores, segment = rfm._score(*rfm._scorable(row, complete), thresholds,
                                                     prior_orders=buyer.prior_orders[window, currency])
                    values.extend((row.in_base, *_measured(row.recency), row.observed_frequency,
                                   *_measured(row.frequency), *_measured(row.monetary),
                                   row.known_subtotal, *scores, segment))
            return values

        def clients():
            for row in db.execute("""SELECT c.rowid,c.sid,p.sex,p.age,c.selected,c.selected_buys,c.buys,c.positions,
                                            c.first,c.last,c.rub,c.kzt,c.rub_valued,c.kzt_valued,c.unvalued
                                     FROM clients c LEFT JOIN profiles p USING(sid) ORDER BY c.rowid"""):
                check()
                n, sid, sex, age, selected, selected_buys, buys, positions, first, last, rub, kzt, rv, kv, unvalued = row
                yield _encoded((f"C{n}", sex, age, selected, selected_buys, buys, positions,
                                datetime.fromisoformat(first) if first else None, datetime.fromisoformat(last) if last else None,
                                Decimal(rub) if rub is not None else None, Decimal(kzt) if kzt is not None else None,
                                rv, kv, unvalued, *personal(sid)))

        clients_count = _csv_entry(archive, "clients.csv", CLIENT_COLUMNS, clients(), control)
        if db.execute("SELECT COUNT(*) FROM clients").fetchone()[0] != clients_count:
            raise DetailExportError("BROKEN_DETAIL_RELATIONS")
        guard.check(force=True)
    manifest = dict(format_version=DETAIL_FORMAT, dto_version=RESULT_SCHEMA, policy=report.policy,
                    generated_at=datetime.now(timezone.utc).isoformat(), calculated_at=report.calculated_at,
                    cutoff=report.rfm["as_of_date"], profile_age_at=profile_age_at.isoformat(),
                    context_id=scope.context_id, contexts={c.context_id: report.contexts[c.context_id] for c in (scope, buyer_scope, rfm_context)},
                    provenance_digest=report.source["provenance_digest"], source_binding=report.metadata["source_binding"],
                    source_fingerprint=report.metadata["fingerprint"], algorithm=report.metadata["calculation_algorithm"],
                    rows=dict(orders=order_count, order_lines=line_count, clients=clients_count),
                    csv=dict(encoding=CSV_ENCODING, delimiter=CSV_DELIMITER, newline="CRLF",
                             null="", text_prefix=CSV_TEXT_PREFIX))
    archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True, separators=(",", ":")))


def write_detail(archive, workspace, control, report, request, *, raw_root, catalog_path, index_dir):
    control.stage("source_validation")
    if report.source["reader"] != "INDEXED" or not report.metadata["detail"]["available"]:
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    if report.metadata["calculation_algorithm"] != algorithm_version():
        raise DetailExportError("REPORT_CONTEXT_MISMATCH")
    guard = BuildGuard(workspace.folder, control.cancelled, None, 20 * GIB)
    rejected = None
    try:
        guard.check(force=True)
        with open_unified_index(raw_root=raw_root, catalog_path=catalog_path, index_dir=index_dir, check=control.check) as reader:
            try:
                provenance = reader.manifest["binding"]["provenance"]
                source, generation = _binding("INDEXED", provenance, reader.manifest)
                if source != report.source or generation != report.metadata["source_binding"]:
                    raise DetailExportError("SOURCE_CHANGED")
                _index_integrity(reader, control.check)
                _write(reader, archive, workspace, control, report, request, guard)
                if report.metadata["calculation_algorithm"] != algorithm_version():
                    raise DetailExportError("REPORT_CONTEXT_MISMATCH")
            except DetailExportError as error:
                # The reader checks provenance on exit and translates ValueError
                # to an index error. Keep detail failures in their own domain.
                rejected = error
            except sqlite3.Error:
                if guard.failure is not None:
                    raise guard.failure from None
                rejected = DetailExportError("DETAIL_EXPORT_FAILED")
        if rejected is not None:
            raise rejected
    except StatisticsIndexError as error:
        code = {"INSUFFICIENT_DISK": "INSUFFICIENT_SPACE", "INSUFFICIENT_MEMORY": "INSUFFICIENT_MEMORY",
                "STALE_SOURCE": "SOURCE_CHANGED", "GENERATION_CHANGED": "SOURCE_CHANGED"}.get(str(error), "SOURCE_NOT_READY")
        raise DetailExportError(code) from None
