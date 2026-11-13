"""Single-source UNIFIED_PURCHASE_V1 report calculation and optional cache."""

from collections import Counter
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
import sqlite3

from Application import dataset_statistics as reference
from Application import statistics_unified_cohorts as cohorts
from Application import statistics_unified_rfm as rfm
from Application.mindbox.canonical_storage import storage_lock
from Application.mindbox.selection import MindboxSelectionConfig
from Application.product_resolution import ProductResolver
from Application.statistics_cohorts_categories import category_reference
from Application.statistics_index import BuildGuard, GIB, manifest_digest
from Application.statistics_rfm_validation import rfm_cutoff
from Application.statistics_slices import _hash_file
from Application.statistics_storage_safety import StatisticsIndexError, reject_reparse, validate_storage_root
from Application.statistics_unified_aggregate_input import (
    _contexts, _effective, _indexed_actions, _indexed_metadata, _indexed_profiles, _orders, _period,
    _reference_actions, _reference_demographics, _reference_profiles,
)
from Application.statistics_unified_aggregates import accumulate_aggregates
from Application.statistics_unified_cache import UnifiedCache
from Application.statistics_unified_dto import _wire, assemble_result, fingerprint
from Application.statistics_unified_index import (
    _capture, _configure, _estimate, _identities, _inputs, _separate, _staging, open_unified_index, reference_order_reader,
)
from Application.statistics_unified_projection import SCHEMA_VERSION as INDEX_SCHEMA


_ALGORITHM_MODULES = (
    "statistics_unified_service.py", "statistics_unified_dto.py", "statistics_unified_cache.py",
    "statistics_unified_aggregate_input.py", "statistics_unified_aggregates.py",
    "statistics_unified_policy.py", "statistics_unified_facts.py", "statistics_unified_projection.py",
    "statistics_unified_index.py", "statistics_unified_rfm.py", "statistics_unified_cohorts.py",
    "statistics_unified_breakdowns.py", "statistics_unified_detail.py", "action_statistics.py", "order_statistics.py",
    "statistics_rfm_core.py", "statistics_rfm_validation.py", "statistics_cohorts_core.py",
    "statistics_cohorts_channels.py", "statistics_cohorts_categories.py", "statistics_cohorts_money.py",
    "statistics_period.py", "statistics_detail_rfm.py",
)


def algorithm_version():
    package = Path(__file__).parent
    return manifest_digest({name: _hash_file(package / name) for name in _ALGORITHM_MODULES})


def _check_cancelled(cancelled):
    if cancelled and cancelled():
        raise InterruptedError("Unified statistics cancelled")


def _binding(kind, provenance, manifest=None):
    return (dict(reader=kind, projection_schema=INDEX_SCHEMA if kind == "INDEXED" else None,
                 provenance_digest=manifest_digest(provenance)), manifest_digest(manifest) if manifest is not None else None)


def _fingerprint(source, generation, provenance, calculated_at, parameters, algorithm):
    cutoff = rfm_cutoff(_effective(provenance, "orders"), parameters["selected_end"], calculated_at)
    return fingerprint(source, generation, calculated_at.isoformat(), parameters, _wire(cutoff), algorithm)


def _cached_result(cache, cache_only, source, generation, provenance, calculated_at, parameters, algorithm, check):
    if not cache_only:
        key = _fingerprint(source, generation, provenance, calculated_at, parameters, algorithm)
        return cache.load(key, check=check)
    candidate = cache.latest_candidate(check=check)
    if candidate is None:
        return None
    cutoff = rfm_cutoff(_effective(provenance, "orders"), parameters["selected_end"], calculated_at)
    if candidate.rfm["as_of_date"] != _wire(cutoff):
        return None
    previous_time = datetime.fromisoformat(candidate.calculated_at)
    key = _fingerprint(source, generation, provenance, previous_time, parameters, algorithm)
    return candidate if candidate.metadata["fingerprint"] == key else None


def _paths_disjoint(a, b):
    if a.is_relative_to(b) or b.is_relative_to(a):
        raise StatisticsIndexError("UNIFIED_STORAGE_PATHS_MUST_BE_SEPARATE")


def _index_integrity(reader, check):
    failure = None

    def progress():
        nonlocal failure
        try:
            check()
        except InterruptedError as exc:
            failure = exc
            return 1
        return 0

    reader.db.set_progress_handler(progress, 10000)
    try:
        if reader.db.execute("PRAGMA main.quick_check(1)").fetchall() != [("ok",)]:
            raise StatisticsIndexError("INVALID_UNIFIED_INDEX_INTEGRITY")
    except sqlite3.Error:
        if failure is not None:
            raise failure from None
        raise StatisticsIndexError("INVALID_UNIFIED_INDEX_INTEGRITY") from None
    finally:
        reader.db.set_progress_handler(reader._sql_check, 10000)
    check()


def _calculate(reader, *, scratch, expected_bytes, cancelled, observer, calculated_at, periods,
               selected_items, parameters, metadata, categories, actions, profiles, profile_details):
    contexts = _contexts(reader, calculated_at=calculated_at, **periods, selected_items=selected_items)
    predicate = None if selected_items is None else selected_items.__contains__
    item_categories = {code: item.category for code, item in metadata.items()}
    with _staging(scratch, validate_storage_root(scratch)) as (folder, _, _):
        guard = BuildGuard(folder, cancelled, observer, 20 * GIB)
        guard.check(force=True, extra=expected_bytes)
        check = guard.check
        rfm_context, bounds = rfm._context(reader, calculated_at, parameters["selected_end"])
        cohort_context, core = cohorts._context(
            reader, calculated_at, parameters["selected_start"], parameters["selected_end"], check,
        )
        rfm_diagnostics, cohort_diagnostics = Counter(), Counter()
        try:
            with closing(sqlite3.connect(folder / "statistics.sqlite")) as db:
                _configure(db, guard, cache_kib=32768)
                db.create_collation("RFM_DECIMAL", rfm.decimal_compare)
                db.executescript(rfm._SCHEMA + cohorts._SCHEMA)

                def feed_orders(orders):
                    for entry in orders:
                        check()
                        rfm._spool_order(db, entry, bounds, rfm_context.orders_intervals, rfm_diagnostics)
                        cohorts._spool_order(db, entry, core, item_categories, categories, cohort_diagnostics, check)
                        yield entry

                def feed_actions():
                    for event in actions:
                        check()
                        rfm._member(db, event.customer_key)
                        yield event

                def feed_profiles():
                    for key in profiles:
                        check()
                        rfm._member(db, key)
                        yield key

                if contexts[4].period is not None:
                    # The date-restricted general population omits undated profiles.
                    # RFM's all-observed population still includes them.
                    for _ in feed_profiles():
                        pass
                with closing(_orders(reader, predicate, contexts[1], check)) as orders, closing(
                        feed_orders(orders)) as grouped, closing(feed_actions()) as events, closing(feed_profiles()) as customers:
                    aggregate = accumulate_aggregates(db, contexts=contexts, orders=grouped, actions=events,
                        profile_keys=customers, metadata=metadata, product_predicate=predicate,
                        action_period=periods["order_period"], profile_details=profile_details,
                        profiles_available=reader.manifest["binding"]["provenance"]["customers"]["present"],
                        category_reference=categories, check=check)
                rfm_result = rfm._summarize(db, rfm_context, bounds, rfm_diagnostics, None, check)
                cohort_result = cohorts._summarize(db, cohort_context, core, categories, cohort_diagnostics, None, check)
                guard.check(force=True)
                return aggregate, rfm_result, cohort_result
        except sqlite3.Error:
            if guard.failure is not None:
                raise guard.failure from None
            raise StatisticsIndexError("UNIFIED_CALCULATION_FAILED") from None


def calculate_unified_statistics(*, raw_root, catalog_path, scratch_dir, expected_bytes, calculated_at,
                                 reader="REFERENCE", index_dir=None, cache_dir=None, cache_mode="latest",
                                 order_period=None, buyer_period=None, customer_period=None, selected_items=None,
                                 selected_start=None, selected_end=None, cancelled=None, observer=None,
                                 product_filter=None, cache_only=False):
    """Return a validated report. The caller supplies a private scratch/cache root."""
    if reader not in ("REFERENCE", "INDEXED") or (reader == "INDEXED") != (index_dir is not None):
        raise ValueError("An explicit source and matching index root are required")
    if not isinstance(calculated_at, datetime) or calculated_at.utcoffset() is None:
        raise ValueError("An aware calculation timestamp is required")
    _estimate(expected_bytes)
    calculated_at = calculated_at.astimezone(timezone.utc)
    if cache_only and (cache_dir is None or cache_mode != "latest"):
        raise ValueError("Opening a cached report requires the latest cache")
    if product_filter is not None and (not callable(product_filter) or selected_items is not None):
        raise ValueError("Use either a product predicate or selected item codes")
    if isinstance(selected_items, (str, bytes)):
        raise ValueError("A collection of product codes is required")
    selected_items = None if selected_items is None else frozenset(selected_items)
    if selected_items is not None and any(not isinstance(code, str) or not code or code.strip() != code
                                          for code in selected_items):
        raise ValueError("Invalid product filter")
    start = date.fromisoformat(selected_start) if selected_start is not None else None
    end = date.fromisoformat(selected_end) if selected_end is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("Reversed customer date bounds")
    periods = {key: _period(value, ()) for key, value in dict(order_period=order_period, buyer_period=buyer_period,
                                                            customer_period=customer_period).items()}
    def request(metadata=None):
        items = (selected_items if product_filter is None else
                 frozenset(code for code, item in metadata.items() if product_filter(code, item)))
        return items, dict(_wire(periods), product_filter_key=None if items is None else manifest_digest(sorted(items)),
                           selected_start=_wire(start), selected_end=_wire(end))
    root, catalog = _inputs(raw_root, catalog_path)
    scratch = reject_reparse(scratch_dir)
    _separate(scratch, root)
    index = reject_reparse(index_dir) if index_dir is not None else None
    if index is not None:
        _paths_disjoint(index, scratch)
    cache = None
    if cache_dir is not None:
        target = reject_reparse(cache_dir)
        _separate(target, root)
        _paths_disjoint(target, scratch)
        if index is not None:
            _paths_disjoint(target, index)
        cache = UnifiedCache(target, mode=cache_mode)

    def check():
        _check_cancelled(cancelled)

    check()
    algorithm = algorithm_version()
    if cache is not None and reader == "REFERENCE":
        # A reusable result is discovered before building the canonical Orders spool.
        with storage_lock(root):
            provenance = _capture(root, catalog)
            source, generation = _binding(reader, provenance)
            _, parameters = request(reference._catalog_snapshot(catalog)[1] if product_filter is not None else None)
            hit = _cached_result(cache, cache_only, source, generation, provenance, calculated_at, parameters, algorithm, check)
            if hit is not None:
                check()
                if algorithm_version() != algorithm:
                    raise StatisticsIndexError("UNIFIED_ALGORITHM_CHANGED")
                if product_filter is not None and _capture(root, catalog) != provenance:
                    raise StatisticsIndexError("STALE_SOURCE")
                return hit
        if cache_only:
            raise StatisticsIndexError("UNIFIED_CACHE_MISS")

    reader_scope = (reference_order_reader(raw_root=root, catalog_path=catalog, scratch_dir=scratch,
                    expected_bytes=expected_bytes, cancelled=cancelled, observer=observer) if reader == "REFERENCE"
                    else open_unified_index(raw_root=root, catalog_path=catalog, index_dir=index, check=check))
    hit = None
    with reader_scope as input_reader:
        provenance = input_reader.manifest["binding"]["provenance"]
        source, generation = _binding(reader, provenance, input_reader.manifest if reader == "INDEXED" else None)
        metadata = _indexed_metadata(input_reader.db) if reader == "INDEXED" and product_filter is not None else None
        if reader == "INDEXED" and cache is not None:
            _, parameters = request(metadata)
            hit = _cached_result(cache, cache_only, source, generation, provenance, calculated_at, parameters, algorithm, check)
        if hit is None and not cache_only:
            selection = MindboxSelectionConfig(**provenance["canonical"]["selection"])
            if reader == "REFERENCE":
                identities = _identities(root, provenance, input_reader.check)
                items, metadata, names, paths = reference._catalog_snapshot(catalog)
                actions = _reference_actions(root, provenance, identities, ProductResolver(items), selection, input_reader.check)
                profiles = _reference_profiles(root, identities, input_reader.check)
                demographics = closing(_reference_demographics(root, identities, input_reader.check))
            else:
                _index_integrity(input_reader, check)
                if metadata is None:
                    metadata = _indexed_metadata(input_reader.db)
                names, paths = cohorts._indexed_categories(input_reader.db, check)
                actions = _indexed_actions(input_reader, selection, check)
                profiles = _indexed_profiles(input_reader)
                demographics = input_reader.profile_demographics()
            items_in_scope, parameters = request(metadata)
            with closing(actions), closing(profiles), demographics as profile_details:
                parts = _calculate(input_reader, scratch=scratch, expected_bytes=expected_bytes, cancelled=cancelled,
                    observer=observer, calculated_at=calculated_at, periods=periods, selected_items=items_in_scope,
                    parameters=parameters, metadata=metadata, categories=category_reference(names, paths),
                    actions=actions, profiles=profiles, profile_details=profile_details)
    check()
    if algorithm_version() != algorithm:
        raise StatisticsIndexError("UNIFIED_ALGORITHM_CHANGED")
    if hit is not None:
        return hit
    if cache_only:
        raise StatisticsIndexError("UNIFIED_CACHE_MISS")
    result = assemble_result(*parts, item_metadata=metadata, source=source, source_binding=generation,
                             parameters=parameters, algorithm=algorithm, check=check)
    if cache is not None:
        with storage_lock(root):
            if _capture(root, catalog) != provenance:
                raise StatisticsIndexError("STALE_SOURCE")
            if algorithm_version() != algorithm:
                raise StatisticsIndexError("UNIFIED_ALGORITHM_CHANGED")
            cache.save(result, check=check)
    return result
