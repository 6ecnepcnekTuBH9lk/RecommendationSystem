"""Indexed input to the reference driver. No canonical JSON/profile row reads.

Date predicates prune through B-trees. Broader diagnostics intentionally retain
source headers on product-only filters: narrowing those counters would be wrong.
Line occurrences and selected catalog item membership are fetched from SQLite.
"""

from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from time import perf_counter
from types import SimpleNamespace

from Application import dataset_statistics as reference
from Application.analysis_filter import AnalysisFilter
from Application.mindbox.adapters._common import AdapterError
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
from Application.mindbox.records import ProductKey
from Application.product_resolution import DEFAULT_CATALOG_PATH
from Application.statistics_cache import validate_result
from Application.statistics_facts import IndexedSnapshots, action_record, microseconds, profile_result
from Application.statistics_index import DEFAULT_INDEX_DIR, StatisticsIndexError, open_generation
from Application.statistics_slices import capture_sources

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class TimedCursor(sqlite3.Cursor):
    def _timed(self, name, *args, **kwargs):
        started = perf_counter()
        try:
            return getattr(super(), name)(*args, **kwargs)
        finally:
            self.connection.sql_seconds += perf_counter()-started

    def execute(self, *args, **kwargs):
        return self._timed("execute", *args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._timed("executemany", *args, **kwargs)

    def fetchone(self):
        return self._timed("fetchone")

    def fetchmany(self, size=1):
        return self._timed("fetchmany", size)

    def fetchall(self):
        return self._timed("fetchall")

    def __next__(self):
        return self._timed("__next__")


class TimedConnection(sqlite3.Connection):
    sql_seconds = 0.

    def cursor(self, factory=TimedCursor):
        return super().cursor(factory)

    def execute(self, *args, **kwargs):
        return self.cursor().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self.cursor().executemany(*args, **kwargs)


class IndexedInput:
    def __init__(self, db, provenance, selection, check, metrics):
        self.db, self.provenance, self.selection = db, provenance, selection
        self.check, self.metrics = check, metrics
        self.identities = SimpleNamespace(resolve=lambda value: value)
        self.snapshots = IndexedSnapshots
        self.selected_items = set()
        if selection.product_restricted:
            clauses, parameters = [], []
            for column, values in (("kind", selection.nomenclature_types), ("collection", selection.collections)):
                if values is not None:
                    known = [v for v in values if v is not None]
                    parts = [f"{column} IN ({','.join('?' for _ in known)})"] if known else []
                    parameters.extend(known)
                    if None in values:
                        parts.append(f"{column} IS NULL")
                    clauses.append("("+" OR ".join(parts)+")")
            self.selected_items = {r[0] for r in db.execute("SELECT code FROM items WHERE "+" AND ".join(clauses), parameters)}
        self._batch_products, self._batch_lines = {}, {}
        self.rfm_interval = None

    def records(self, name, sources):
        # IDs are global raw occurrence positions, already in effective source order.
        if name not in ("actions", "orders"):
            raise StatisticsIndexError("UNEXPECTED_INDEX_SOURCE")
        if not sources:
            return
        where, args = "", []
        if self.selection.interval is not None:
            first, last = map(microseconds, self.selection.interval)
            # UNION lets SQLite use the date and sparse exceptional-row indexes.
            exceptional = "date_bad=1" if name == "actions" else "outside=1 OR date_bad=1"
            where = (f" WHERE id IN (SELECT id FROM {name} WHERE event_us>=? AND event_us<? "
                     f"UNION SELECT id FROM {name} WHERE {exceptional})")
            args = [first, last]
            if name == 'orders' and self.rfm_interval is not None:
                where = where[:-1] + ' UNION SELECT id FROM orders WHERE event_us>=? AND event_us<?)'
                args.extend(map(microseconds, self.rfm_interval))
        cursor = self.db.execute(f"SELECT * FROM {name}{where} ORDER BY id", args)
        try:
            while True:
                started = perf_counter()
                rows = cursor.fetchmany(512)
                self.metrics["sqlite_seconds"] += perf_counter()-started
                if not rows:
                    break
                identifiers = [row[0] for row in rows]
                slots = ",".join("?" for _ in identifiers)
                if name == "actions":
                    self._batch_products = {}
                    for event, namespace, value in self.db.execute(
                            f"SELECT event,namespace,value FROM action_products WHERE event IN ({slots}) ORDER BY event,ordinal", identifiers):
                        self._batch_products.setdefault(event, []).append(ProductKey(namespace, value))
                else:
                    self._batch_lines = {}
                    for event, *line in self.db.execute(
                            f"SELECT event,ordinal,quantity,status,namespace,value,price FROM order_lines WHERE event IN ({slots}) ORDER BY event,ordinal", identifiers):
                        self._batch_lines.setdefault(event, []).append(line)
                for row in rows:
                    self.check()
                    self.metrics["source_headers"] += 1
                    yield self._action(row) if name == "actions" else self._order(row)
        finally:
            cursor.close()

    def _action(self, row):
        ordinal, event, bad_date, name, customer, has_product, header_bad, adapter_bad, encoded = row
        if header_bad:
            # SQL selection already applied the reference date gate. Invalid dates
            # remain present so an explicit date filter still raises AdapterError.
            raise AdapterError("Indexed action header failed source validation")
        date = EPOCH+timedelta(microseconds=event) if event is not None else None
        return {"_ordinal": ordinal, "_date": date, "_adapter_bad": adapter_bad, "_details": json.loads(encoded),
                "dateTimeUtc": "invalid" if bad_date else date.isoformat() if date else None,
                "actionTemplate": {"ids": {"systemName": name}}, "products": [{}] if has_product else [],
                "customer": {"ids": {"mindboxId": str(customer) if customer is not None else None}}}

    def action(self, raw):
        return action_record(raw, self._batch_products.get(raw["_ordinal"], ()))

    def _order(self, row):
        ordinal, event, bad_date, outside, customer, order_sid, fingerprint, header_bad, adapter_bad, aggregate_bad, encoded = row
        date = EPOCH+timedelta(microseconds=event) if event is not None else None
        # Outside-global records never reach header validation in the reference.
        if header_bad and not outside and not bad_date:
            raise AdapterError("Indexed order header failed source validation")
        details = json.loads(encoded)
        channel = details.get("firstAction", {}).get("channel")
        lines = [] if bad_date else self._batch_lines.get(ordinal, ())
        return dict(details, _ordinal=ordinal, _date=date, _adapter_bad=adapter_bad, _aggregate_bad=aggregate_bad,
                    _fingerprint=fingerprint, _lines=lines, _header_bad=header_bad, ids={"mindboxId": str(order_sid)},
                    customer={"ids": {"mindboxId": str(customer)}},
                    firstAction={"dateTimeUtc": "invalid" if bad_date else date.isoformat(), "channel": channel},
                    lines=[{"quantity": Decimal(quantity), "status": {"ids": {"externalId": status}}}
                           for _, quantity, status, _, _, _ in lines])

    def order(self, raw):
        if raw["_adapter_bad"] or raw['_header_bad']:
            raise AdapterError("Indexed order failed source adapter validation")
        customer = raw["customer"]["ids"]["mindboxId"]
        return tuple(SimpleNamespace(
            product=ProductKey(namespace, value), quantity=Decimal(quantity), price_of_line=Decimal(price),
            line_status=status, customer_id=customer, source_customer_id=customer,
            order_datetime_utc=raw["_date"], line_id=str(ordinal))
            for ordinal, quantity, status, namespace, value, price in raw["_lines"]
            if Decimal(quantity) == Decimal(quantity).to_integral_value())

    def add_order(self, aggregates, raw, purchases):
        if not purchases:
            return
        if raw["_aggregate_bad"]:
            raise AdapterError("Indexed order aggregate fields failed source validation")
        if raw.get("deliveryCost") is not None:
            raw = dict(raw, deliveryCost=Decimal(raw["deliveryCost"]))
        aggregates.add(raw, purchases)

    def customer_snapshot(self, *args, **kwargs):
        started = perf_counter()
        try:
            return profile_result(self.db, self.provenance, *args, **kwargs)
        finally:
            self.metrics["demography_seconds"] = perf_counter()-started


def calculate_indexed_statistics(*, raw_root=DEFAULT_RAW_ROOT, catalog_path=DEFAULT_CATALOG_PATH,
                                 index_dir=None, analysis_filter=AnalysisFilter(),
                                 progress=None, cancelled=None, metrics=None, calculated_at=None):
    calculated_at = calculated_at or datetime.now(timezone.utc)
    metrics = metrics if metrics is not None else {}
    metrics.update(sqlite_seconds=0., validation_seconds=0., source_headers=0)
    started = perf_counter()

    def check():
        if cancelled and cancelled():
            raise InterruptedError("Расчёт статистики отменён")

    def capture():
        check()
        try:
            return capture_sources(raw_root, catalog_path, analysis_filter, calculated_at=calculated_at)
        except InterruptedError:
            raise
        except Exception:
            # AUTO can now use the reference's established source error handling;
            # strict INDEXED remains fail-closed without exposing file contents.
            raise StatisticsIndexError("SOURCE_UNAVAILABLE") from None

    check()
    provenance, effective = capture()
    metrics["attestation_seconds"] = perf_counter()-started
    load_started = perf_counter()
    db, manifest = open_generation(index_dir or DEFAULT_INDEX_DIR, provenance,
                                   connection_factory=TimedConnection, check=check)
    metrics["load_seconds"] = perf_counter()-load_started
    with closing(db):
        failure = []

        def sql_check():
            try:
                check()
                return 0
            except InterruptedError as exc:
                failure.append(exc)
                return 1

        db.set_progress_handler(sql_check, 10000)
        if progress:
            progress("INDEXED: расчёт по актуальному аналитическому индексу")
        source = IndexedInput(db, provenance, effective, check, metrics)
        aggregate_started = perf_counter()
        sql_before = db.sql_seconds
        try:
            result = reference._calculate_statistics(Path(raw_root), catalog_path, progress, check, effective, source,
                                                     calculated_at=calculated_at, customer_filter=analysis_filter)
        except sqlite3.OperationalError:
            if failure:
                raise failure[0] from None
            raise StatisticsIndexError("SQLITE_QUERY_ERROR") from None
        metrics["driver_seconds"] = perf_counter()-aggregate_started
        metrics["sqlite_seconds"] = db.sql_seconds-sql_before
        before_validation = perf_counter()
        validate_result(asdict(result))
        metrics["validation_seconds"] = perf_counter()-before_validation
        check()
        attestation_started = perf_counter()
        current, _ = capture()
        metrics["attestation_seconds"] += perf_counter()-attestation_started
        if current != provenance:
            raise StatisticsIndexError("STALE_SOURCE")
    metrics["generation"] = manifest["generation"]
    metrics["total_seconds"] = perf_counter()-started
    metrics["aggregate_seconds_excluding_sql"] = metrics["driver_seconds"]-metrics["sqlite_seconds"]
    return result
