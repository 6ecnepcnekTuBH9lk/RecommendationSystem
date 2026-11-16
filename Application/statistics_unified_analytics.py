"""Persistent reconciled facts for the schema 4 observation generation."""

from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import fields, is_dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from itertools import groupby
import json
import os
from pathlib import Path
import re
import sqlite3
import zlib

from Application.mindbox.canonical_storage import atomic_json, storage_lock
from Application.mindbox.selection import MindboxSelectionConfig
from Application import statistics_unified_cohorts as cohorts
from Application import statistics_unified_rfm as rfm
from Application.statistics_cohorts_categories import category_reference
from Application.statistics_facts import microseconds
from Application.statistics_index import BuildGuard, GIB, manifest_digest
from Application.statistics_slices import _hash_file, _stat
from Application.statistics_storage_safety import StatisticsIndexError, reject_reparse, validate_storage_root
from Application.statistics_unified_aggregates import AggregateMetric, AggregateOrder, _bucket
from Application.statistics_unified_aggregate_input import _effective, _indexed_actions, _indexed_metadata
from Application.statistics_unified_facts import (
    AnalyticsContext, MonetaryEvidence, OrderBusinessFact, OrderKind, OrderLineState, PreparedOrder,
    ProductOrderEvidence, PurchaseEvidence, SnapshotCounts, _issues, prepare_order, scope_order,
)
from Application.statistics_unified_index import (
    _capture, _configure, _estimate, _inputs, _matches_projection, _readonly, _read_json, _separate, _staging,
    _writer_lock, open_unified_index,
)
from Application.statistics_unified_policy import EvidenceQuality, Measured, PurchasePolicy, StatusGroup
from Application.statistics_unified_compat import prepared_component_hash


POINTER = "current-analytics-v1.json"
SCHEMA = """
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE facts(rank INTEGER PRIMARY KEY, stamp INTEGER, whole BLOB NOT NULL, prepared BLOB NOT NULL,
                   unresolved_product INTEGER NOT NULL);
CREATE INDEX facts_date ON facts(stamp,rank);
CREATE TABLE fact_items(item TEXT, rank INTEGER, PRIMARY KEY(item,rank)) WITHOUT ROWID;
CREATE TABLE profiles(ordinal INTEGER PRIMARY KEY, owner TEXT NOT NULL, born TEXT, sex TEXT);
"""
_TYPES = {cls.__name__: cls for cls in (
    date, Decimal, datetime, EvidenceQuality, Measured, MonetaryEvidence, OrderBusinessFact,
    OrderKind, OrderLineState, PreparedOrder, ProductOrderEvidence, PurchaseEvidence,
    PurchasePolicy, SnapshotCounts, StatusGroup, frozenset, AnalyticsContext,
    AggregateMetric, rfm.UnifiedRfmResult, cohorts.UnifiedCohortsResult,
)}
_DATACLASSES = frozenset(name for name, cls in _TYPES.items() if is_dataclass(cls))
_CONSTRUCTORS = {name: cls.fromisoformat if cls in (date, datetime) else cls
                 for name, cls in _TYPES.items()}
for _name in ("Measured", "Decimal", "PurchasePolicy", "SnapshotCounts", "EvidenceQuality", "StatusGroup", "OrderKind"):
    _CONSTRUCTORS[_name] = lru_cache(maxsize=16384, typed=True)(_CONSTRUCTORS[_name])

# The analytical readers consume order evidence, not the original snapshot
# lineage. Keep the full wire format for existing generations and preparation.
_OMITTED_FIELDS = {
    OrderBusinessFact: frozenset(("snapshots", "diagnostics", "observation_groups", "policy",
                                 "whole_order_kind", "line_history")),
    OrderLineState: frozenset(("stable_line_key", "ordinal", "input_quantity", "nominal_units",
                               "fulfilled_units", "source_price", "currency")),
    PurchaseEvidence: frozenset(("evidence_sources", "executed_at", "fulfilled_units")),
    MonetaryEvidence: frozenset(("confirmed_refund", "net_after_refunds")),
    ProductOrderEvidence: frozenset(("known_positions",)),
}
_ORDER_FIELDS = {cls.__name__: tuple(field.name not in omitted for field in fields(cls))
                 for cls, omitted in _OMITTED_FIELDS.items()}
_BUYER_FIELDS = tuple(
    field.name not in _OMITTED_FIELDS[OrderBusinessFact] | {"resolved_lines", "historical_statuses",
                                                          "unassigned_product_issues"}
    for field in fields(OrderBusinessFact))


def _encode(value):
    if isinstance(value, Enum):
        return {"@": type(value).__name__, "v": value.value}
    if isinstance(value, (Decimal, date)):
        return {"@": type(value).__name__, "v": str(value)}
    if is_dataclass(value):
        return {"@": type(value).__name__, "v": [_encode(getattr(value, f.name)) for f in fields(value)]}
    if isinstance(value, frozenset):
        return {"@": "frozenset", "v": [_encode(item) for item in sorted(value)]}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value):
    if isinstance(value, list):
        return tuple(map(_decode, value))
    if isinstance(value, dict):
        if "@" not in value:
            return {key: _decode(item) for key, item in value.items()}
        cls = _TYPES[value["@"]]
        payload = _decode(value["v"])
        if is_dataclass(cls):
            return cls(*payload)
        return cls.fromisoformat(payload) if cls in (date, datetime) else cls(payload)
    return value


def _dump(value):
    return json.dumps(_encode(value), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _load(value):
    if isinstance(value, bytes):
        value = zlib.decompress(value)
    return _decode(json.loads(value))


def _order_value(value, buyer_only=False):
    if type(value) is list:
        return tuple(map(_order_value, value))
    if type(value) is not dict:
        return value
    name = value["@"]
    payload = value["v"]
    create = _CONSTRUCTORS[name]
    if name == "frozenset":
        return frozenset(map(_order_value, payload))
    if name in _DATACLASSES:
        keep = _BUYER_FIELDS if buyer_only else _ORDER_FIELDS.get(name)
        if keep is not None:
            return create(*(_order_value(part) if needed else None
                            for part, needed in zip(payload, keep, strict=True)))
        return create(*map(_order_value, payload))
    return create(payload)


def _binding(reader):
    package = Path(__file__).parent
    modules = ("statistics_unified_analytics.py", "statistics_unified_facts.py", "statistics_unified_policy.py",
               "statistics_unified_projection.py", "statistics_unified_aggregates.py", "statistics_unified_aggregate_input.py",
               "statistics_unified_rfm.py", "statistics_unified_cohorts.py", "statistics_rfm_core.py",
               "statistics_rfm_validation.py", "statistics_cohorts_core.py", "statistics_cohorts_channels.py",
               "statistics_cohorts_categories.py", "statistics_cohorts_money.py", "statistics_period.py")
    return {"schema": 1, "observations": manifest_digest(reader.manifest),
            "algorithm": manifest_digest({name: prepared_component_hash(name, _hash_file(package / name)) for name in modules})}


def _populate(db, reader, check, history_db, phase):
    raw_orders = raw_lines = projected_lines = invalid = 0
    statuses, issues = Counter(), set()
    diagnostics, totals, purchase_issues, position_issues = Counter(), Counter(), set(), set()
    coverage = _effective(reader.manifest["binding"]["provenance"], "orders")
    build_time = datetime.now(timezone.utc)
    rfm_context, bounds = rfm._context(reader, build_time, None)
    cohort_context, core = cohorts._context(reader, build_time, None, None, check)
    rfm_diagnostics, cohort_diagnostics = Counter(), Counter()
    item_categories = {code: item.category for code, item in _indexed_metadata(reader.db).items()}
    categories = category_reference(*cohorts._indexed_categories(reader.db, check))
    history_db.create_collation("RFM_DECIMAL", rfm.decimal_compare)
    history_db.executescript(rfm._SCHEMA + cohorts._SCHEMA)
    phase("Согласование заказов…")
    with closing(reader.orders()) as rows:
        for key, group in groupby(rows, key=lambda row: row.observation.order_key.value):
            snapshots = []
            for row in group:
                check()
                observation = row.observation
                raw_orders += 1
                raw_lines += observation.raw_line_count
                projected_lines += len(observation.lines)
                if observation.lines_quality is not EvidenceQuality.KNOWN:
                    issues.add((observation.lines_quality, "LINES_INCOMPLETE"))
                    statuses[None, "UNRESOLVED"] += observation.raw_line_count - len(observation.lines)
                statuses.update(_bucket(line.status_code) for line in observation.lines)
                snapshots.append(observation)
            if key is None:
                invalid += len(snapshots)
                continue
            prepared = prepare_order(snapshots, order_sort_rank=row.order_sort_rank, check=check)
            whole = scope_order(prepared, check=check)
            diagnostics.update(whole.diagnostics)
            diagnostics["excluded_conflicting_positions"] += whole.purchase.excluded_positions
            for name in ("exact_duplicates", "business_equivalent", "conflicting_snapshots"):
                diagnostics[name] += getattr(whole.snapshots, name)
            totals["purchase_orders"] += whole.purchase.observed.value is True
            totals["purchase_positions"] += whole.purchase.known_positions
            purchase_issues.update(_issues(whole.purchase.observed))
            position_issues.update(_issues(whole.purchase.positions))
            entry = AggregateOrder((), whole, whole)
            rfm._spool_order(history_db, entry, bounds, rfm_context.orders_intervals, rfm_diagnostics)
            cohorts._spool_order(history_db, entry, core, item_categories, categories, cohort_diagnostics, check)
            stamp = whole.order_created_at.value
            if stamp is not None and not any(start <= stamp < end for start, end in coverage):
                diagnostics["orders_outside_coverage"] += 1
            candidates = {line.product_key.value for resolved, group in prepared.pairs for line in (resolved, *group)}
            candidates.update(line.product_key.value for line in prepared.omitted)
            candidates.update(item for _, item, _ in prepared.positives)
            unresolved = None in candidates or any(reason == "LINES_INCOMPLETE" for _, reason in prepared.composition)
            db.execute("INSERT INTO facts VALUES(?,?,?,?,?)", (
                whole.order_sort_rank, microseconds(stamp) if stamp is not None else None,
                zlib.compress(_dump(whole).encode("utf-8"), 1),
                zlib.compress(_dump(prepared).encode("utf-8"), 1), unresolved))
            db.executemany("INSERT INTO fact_items VALUES(?,?)",
                           ((item, whole.order_sort_rank) for item in candidates if item is not None))
            if whole.order_sort_rank % 2000 == 0:
                db.commit()
                history_db.commit()
            if whole.order_sort_rank % 100000 == 0:
                phase(f"Согласование заказов · обработано {raw_orders:,} снимков".replace(",", " "))
    db.execute("INSERT INTO metadata VALUES('raw',?)",
               (_dump((raw_orders, raw_lines, tuple(statuses.items()), tuple(sorted(issues)), invalid)),))
    db.execute("INSERT INTO metadata VALUES('order_history',?)", (_dump((
        tuple(diagnostics.items()), tuple(totals.items()), tuple(sorted(purchase_issues)),
        tuple(sorted(position_issues)))),))
    phase("Подготовка клиентских данных и действий…")
    with reader.profile_demographics() as profiles:
        for ordinal, (owner, born, sex) in enumerate(profiles):
            check()
            db.execute("INSERT INTO profiles VALUES(?,?,?,?)", (ordinal, owner, born.isoformat() if born else None, sex))
    selection = MindboxSelectionConfig(**reader.manifest["binding"]["provenance"]["canonical"]["selection"])
    action_diagnostics = Counter()
    invalid_unclassified = False
    for event in _indexed_actions(reader, selection, check):
        rfm._member(history_db, event.customer_key)
        if event.reason:
            action_diagnostics[event.reason] += 1
        if event.kind is None and event.reason == "ACTION_INVALID":
            invalid_unclassified = True
    db.execute("INSERT INTO metadata VALUES('action_diagnostics',?)",
               (_dump((tuple(action_diagnostics.items()), invalid_unclassified)),))
    for (_, owner) in reader.profile_links():
        check()
        rfm._member(history_db, str(owner))
    if invalid:
        rfm_diagnostics["ORDER_ID_INVALID"] = invalid
        cohort_diagnostics["ORDER_ID_INVALID"] = invalid
    phase("Подготовка RFM и когорт…")
    db.execute("INSERT INTO metadata VALUES('cohort_input',?)",
               (_dump((cohort_context.as_of_date, dict(cohort_diagnostics))),))
    results = (rfm._summarize(history_db, rfm_context, bounds, rfm_diagnostics, None, check),
               cohorts._summarize(history_db, cohort_context, core, categories, cohort_diagnostics, None, check))
    db.execute("INSERT INTO metadata VALUES('history',?)", (_dump(results),))
    db.execute("INSERT INTO metadata VALUES('rfm_segmentation',?)", (rfm.SEGMENTATION_POLICY,))
    # Keep the exact ordered cohort inputs for other cohort bounds at the same cutoff.
    for name in ("cohort_population", "cohort_orders"):
        schema = history_db.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()[0]
        db.execute(schema)
        rows = history_db.execute(f"SELECT * FROM {name}")
        placeholders = ",".join("?" for _ in rows.description)
        def checked_rows():
            for row in rows:
                check()
                yield row
        db.executemany(f"INSERT INTO {name} VALUES({placeholders})", checked_rows())
    db.commit()
    return raw_orders, projected_lines


def prepare_analytics(*, raw_root, catalog_path, index_dir, expected_bytes, cancelled=None, observer=None,
                      phase=lambda message: None):
    """Build in a new private directory and publish a pointer only after validation."""
    _estimate(expected_bytes)
    root, catalog = _inputs(raw_root, catalog_path)
    directory = reject_reparse(index_dir)
    _separate(directory, root)
    pointer = directory / POINTER
    root_id = validate_storage_root(directory, children=(pointer,) if os.path.lexists(pointer) else ())
    with _writer_lock(directory), _staging(directory, root_id) as (folder, _, retain):
        guard = BuildGuard(folder, cancelled, observer, 20 * GIB)
        guard.check(force=True, extra=2 * expected_bytes)
        path = folder / "statistics.sqlite"
        with open_unified_index(raw_root=root, catalog_path=catalog, index_dir=directory, check=guard.check) as reader:
            binding = _binding(reader)
            history_path = folder / "demographics.sqlite"
            with closing(sqlite3.connect(path)) as db, closing(sqlite3.connect(history_path)) as history_db:
                _configure(db, guard, cache_kib=32768)
                _configure(history_db, guard, cache_kib=32768)
                db.executescript(SCHEMA)
                raw = _populate(db, reader, guard.check, history_db, phase)
                counts = reader.manifest["binding"]["counts"]
                if raw != (counts["orders"], counts["order_lines"]):
                    raise StatisticsIndexError("INCOMPLETE_ANALYTICAL_INDEX")
                db.execute("INSERT INTO metadata VALUES('binding',?)", (json.dumps(binding),))
                db.commit()
                phase("Проверка целостности аналитического индекса…")
                if db.execute("PRAGMA quick_check(1)").fetchall() != [("ok",)]:
                    raise StatisticsIndexError("INVALID_ANALYTICAL_INDEX")
                if _binding(reader) != binding:
                    raise StatisticsIndexError("ANALYTICAL_ALGORITHM_CHANGED")
            history_path.unlink()
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
        manifest = {"generation": folder.name, "binding": binding, "file": list(_stat(path))}
        atomic_json(folder / "manifest.json", manifest)
        validate_storage_root(directory, children=(folder, path, folder / "manifest.json"))
        with storage_lock(root):
            if not _matches_projection(_capture(root, catalog), reader.manifest["binding"]["provenance"]):
                raise StatisticsIndexError("STALE_SOURCE")
            if _binding(reader) != binding:
                raise StatisticsIndexError("ANALYTICAL_ALGORITHM_CHANGED")
            guard.check(force=True)
            atomic_json(pointer, manifest)
        retain()
        return manifest


class PreparedAnalytics:
    def __init__(self, db, generation):
        self.db = db
        self.generation = generation
        self.raw = _load(db.execute("SELECT value FROM metadata WHERE key='raw'").fetchone()[0])
        self.action_diagnostics = _load(db.execute(
            "SELECT value FROM metadata WHERE key='action_diagnostics'").fetchone()[0])
        self.order_history = _load(db.execute("SELECT value FROM metadata WHERE key='order_history'").fetchone()[0])

    def cohort_input(self, context):
        cutoff, diagnostics = _load(self.db.execute(
            "SELECT value FROM metadata WHERE key='cohort_input'").fetchone()[0])
        return Counter(diagnostics) if cutoff == context.as_of_date else None

    def history(self, rfm_context, cohort_context, core):
        saved_rfm, saved_cohorts = _load(self.db.execute("SELECT value FROM metadata WHERE key='history'").fetchone()[0])
        segmentation = self.db.execute("SELECT value FROM metadata WHERE key='rfm_segmentation'").fetchone()
        reusable_rfm = (segmentation == (rfm.SEGMENTATION_POLICY,)
                        and saved_rfm.context.as_of_date == rfm_context.as_of_date)
        reusable_cohorts = (saved_cohorts.context.as_of_date == cohort_context.as_of_date
                            and (core is None or core.cohort_start is None and core.cohort_end is None))
        return (replace(saved_rfm, context=rfm_context) if reusable_rfm else None,
                replace(saved_cohorts, context=cohort_context) if reusable_cohorts else None)

    def orders(self, selected_items, scope, check, *, period=None):
        for _ in range(self.raw[-1]):
            check()
            yield AggregateOrder((), None, None)
        selected_range = None if selected_items is None or scope.period is None else tuple(map(microseconds, scope.period))
        start, end = selected_range or (0, 0)
        where = "WHERE stamp IS NULL OR (stamp>=? AND stamp<?)" if period is not None else ""
        if selected_range is None:
            projection, values = "0,NULL", ()
        else:
            projection = """stamp>=? AND stamp<?,
                CASE WHEN stamp>=? AND stamp<? AND (unresolved_product OR rank IN (
                    SELECT item.rank FROM json_each(?) AS wanted JOIN fact_items AS item ON item.item=wanted.value
                )) THEN prepared END"""
            values = start, end, start, end, json.dumps(sorted(selected_items))
        if period is not None:
            values += tuple(map(microseconds, period))
        query = f"SELECT whole,{projection} FROM facts {where} ORDER BY rank"
        for packed, in_product_period, packed_scope in self.db.execute(query, values):
            check()
            # No scoped order can consume line or status detail if the item
            # index proves the slice empty. History still needs buyer evidence.
            buyer_only = in_product_period and packed_scope is None
            whole = _order_value(json.loads(zlib.decompress(packed)), buyer_only=buyer_only)
            selected = whole
            date_value = whole.order_created_at.value
            if in_product_period and any(a <= date_value < b for a, b in scope.orders_intervals):
                # No candidate product, no unobserved lines and no unknown item.
                # Such an order has an exactly empty product slice.
                selected = (scope_order(_load(packed_scope), product_predicate=selected_items.__contains__, check=check)
                            if packed_scope is not None else None)
            yield AggregateOrder((), whole, selected)

    def profile_details(self):
        for owner, born, sex in self.db.execute("SELECT owner,born,sex FROM profiles ORDER BY ordinal"):
            yield owner, date.fromisoformat(born) if born is not None else None, sex


@contextmanager
def open_analytics(directory, reader, check):
    pointer = directory / POINTER
    if not pointer.is_file():
        raise StatisticsIndexError("ANALYTICAL_INDEX_MISSING")
    validate_storage_root(directory, children=(pointer,))
    manifest = _read_json(pointer, 1024 * 1024)
    if (not isinstance(manifest, dict) or set(manifest) != {"generation", "binding", "file"}
            or not isinstance(manifest["generation"], str) or not re.fullmatch(r"[0-9a-f]{32}", manifest["generation"])):
        raise StatisticsIndexError("INVALID_ANALYTICAL_INDEX")
    folder = directory / manifest["generation"]
    path = folder / "statistics.sqlite"
    validate_storage_root(directory, children=(folder, path, folder / "manifest.json"))
    if manifest["binding"] != _binding(reader):
        raise StatisticsIndexError("ANALYTICAL_INDEX_STALE")
    if _read_json(folder / "manifest.json", 1024 * 1024) != manifest or list(_stat(path)) != manifest["file"]:
        raise StatisticsIndexError("INVALID_ANALYTICAL_INDEX")
    interrupted = None
    try:
        with closing(_readonly(path)) as db:
            def sql_check():
                nonlocal interrupted
                try:
                    check()
                    return 0
                except InterruptedError as error:
                    interrupted = error
                    return 1

            db.set_progress_handler(sql_check, 10000)
            saved = db.execute("SELECT value FROM metadata WHERE key='binding'").fetchone()
            if saved is None or json.loads(saved[0]) != manifest["binding"]:
                raise StatisticsIndexError("INVALID_ANALYTICAL_INDEX")
            yield PreparedAnalytics(db, manifest["generation"])
            check()
            if _read_json(pointer, 1024 * 1024) != manifest or list(_stat(path)) != manifest["file"]:
                raise StatisticsIndexError("ANALYTICAL_INDEX_STALE")
    except (sqlite3.Error, ValueError, KeyError, TypeError, zlib.error) as error:
        if interrupted is not None:
            raise interrupted from None
        if isinstance(error, StatisticsIndexError):
            raise
        raise StatisticsIndexError("INVALID_ANALYTICAL_INDEX") from None
