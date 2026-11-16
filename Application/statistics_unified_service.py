"""Single-source UNIFIED_PURCHASE_V1 report calculation and optional cache."""

from collections import Counter
from contextlib import closing, nullcontext
from dataclasses import replace
from datetime import date, datetime, timezone
from functools import partial
from pathlib import Path
import sqlite3
from time import perf_counter

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
    _contexts, _effective, _indexed_action_headers, _indexed_actions, _indexed_metadata, _indexed_profiles, _orders, _period,
    _reference_actions, _reference_demographics, _reference_profiles,
)
from Application.statistics_unified_aggregates import accumulate_aggregates
from Application.statistics_unified_analytics import POINTER as ANALYTICS_POINTER, _load as load_analytics_value, open_analytics
from Application.statistics_unified_cache import LATEST_RESULT_NAME, UnifiedCache
from Application.statistics_unified_compat import equivalent_hash
from Application.statistics_unified_dto import _wire, assemble_result, fingerprint
from Application.statistics_unified_index import (
    _capture, _configure, _estimate, _identities, _inputs, _matches_projection, _read_json, _separate, _staging,
    open_unified_index, reference_order_reader,
)
from Application.statistics_unified_projection import SCHEMA_VERSION as INDEX_SCHEMA


_ALGORITHM_MODULES = (
    "statistics_unified_service.py", "statistics_unified_dto.py", "statistics_unified_cache.py",
    "statistics_unified_aggregate_input.py", "statistics_unified_aggregates.py",
    "statistics_unified_policy.py", "statistics_unified_facts.py", "statistics_unified_projection.py",
    "statistics_unified_index.py", "statistics_unified_rfm.py", "statistics_unified_cohorts.py",
    "statistics_unified_breakdowns.py", "statistics_unified_detail.py", "statistics_unified_analytics.py",
    "action_statistics.py", "order_statistics.py",
    "statistics_rfm_core.py", "statistics_rfm_validation.py", "statistics_cohorts_core.py",
    "statistics_cohorts_channels.py", "statistics_cohorts_categories.py", "statistics_cohorts_money.py",
    "statistics_period.py", "statistics_detail_rfm.py",
)


def algorithm_version():
    package = Path(__file__).parent
    return manifest_digest({name: equivalent_hash(name, _hash_file(package / name)) for name in _ALGORITHM_MODULES})


def _check_cancelled(cancelled):
    if cancelled and cancelled():
        raise InterruptedError("Unified statistics cancelled")


def _binding(kind, provenance, manifest=None):
    return (dict(reader=kind, projection_schema=INDEX_SCHEMA if kind == "INDEXED" else None,
                 provenance_digest=manifest_digest(provenance)), manifest_digest(manifest) if manifest is not None else None)


def _compatibility_key(source, generation, provenance, calculated_at, parameters, algorithm, analytics_generation=None):
    cutoff = rfm_cutoff(_effective(provenance, "orders"), parameters["selected_end"], calculated_at)
    age_date = calculated_at.astimezone().date()
    key = fingerprint(source, generation, age_date.isoformat(), parameters, _wire(cutoff), algorithm)
    return manifest_digest([key, analytics_generation]) if analytics_generation is not None else key


def _cached_result(cache, source, generation, provenance, calculated_at, parameters, algorithm, check,
                   analytics_generation=None):
    key = _compatibility_key(source, generation, provenance, calculated_at, parameters, algorithm, analytics_generation)
    candidate = cache.load(key, check=check, semantic=True)
    if candidate is None:
        return None
    previous_time = datetime.fromisoformat(candidate.calculated_at)
    previous_key = fingerprint(candidate.source, candidate.metadata["source_binding"],
                               previous_time.astimezone().date().isoformat(),
                               candidate.metadata["parameters"], candidate.rfm["as_of_date"],
                               candidate.metadata["calculation_algorithm"])
    if analytics_generation is not None:
        previous_key = manifest_digest([previous_key, analytics_generation])
    return candidate if previous_key == key else None


def open_latest_statistics(*, raw_root, catalog_path, index_dir, cache_dir, check=lambda: None):
    """Load one fully validated saved DTO without entering the calculation workflow."""
    from Application.analysis_filter import AnalysisFilter
    from Application.statistics_unified_detail import item_matches
    if not Path(cache_dir).is_dir():
        raise StatisticsIndexError("UNIFIED_CACHE_MISS")
    cache = UnifiedCache(cache_dir, mode="slice")
    latest = cache.last_selection()
    if latest is None:
        code = "SAVED_REPORT_STALE" if (Path(cache_dir) / LATEST_RESULT_NAME).exists() else "UNIFIED_CACHE_MISS"
        raise StatisticsIndexError(code)
    try:
        selection, product, category = latest["selection"]
        selection = AnalysisFilter.from_dict(selection)
        if not isinstance(product, str) or not isinstance(category, str):
            raise ValueError("Invalid product selection")
        result = cache.load(latest["key"], check=check, semantic=True)
        if result is None or latest["fingerprint"] != result.metadata["fingerprint"]:
            raise ValueError("Invalid saved result")
        algorithm = algorithm_version()
        with open_unified_index(raw_root=raw_root, catalog_path=catalog_path, index_dir=index_dir,
                                check=check, verify_profile_digest=False) as reader:
            with open_analytics(Path(index_dir).resolve(), reader, check) as prepared:
                provenance = reader.manifest["binding"]["provenance"]
                source, generation = _binding("INDEXED", provenance, reader.manifest)
                params = result.metadata["parameters"]
                period = _wire(selection.interval)
                restricted = selection.product_restricted or product or category
                filter_key = (manifest_digest(sorted(code for code, item in _indexed_metadata(reader.db).items()
                             if item_matches(selection, product, category, code, item))) if restricted else None)
                if (source != result.source or generation != result.metadata["source_binding"]
                        or algorithm != result.metadata["calculation_algorithm"]
                        or prepared.generation != latest["analytics_generation"]
                        or params != dict(order_period=period, buyer_period=period, customer_period=period,
                                          selected_start=selection.start_date, selected_end=selection.end_date,
                                          product_filter_key=filter_key)
                        or _compatibility_key(source, generation, provenance, datetime.fromisoformat(result.calculated_at),
                                              params, algorithm, prepared.generation) != latest["key"]):
                    raise ValueError("Incompatible saved result")
        return result, ("unified", "INDEXED", selection, product, category)
    except (ValueError, KeyError, TypeError, OSError):
        raise StatisticsIndexError("SAVED_REPORT_STALE") from None


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
               selected_items, parameters, metadata, categories, actions, profiles, profile_details, prepared=None,
               measure=None, action_headers=None):
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
        rfm_result, cohort_result = (prepared.history(rfm_context, cohort_context, core)
                                     if prepared is not None else (None, None))
        rfm_population = None
        if prepared is not None and rfm_result is None:
            saved_history = prepared.db.execute("SELECT value FROM metadata WHERE key='history'").fetchone()[0]
            rfm_population = load_analytics_value(saved_history)[0].population
        cohort_input = prepared.cohort_input(cohort_context) if prepared is not None and cohort_result is None else None
        rfm_diagnostics = Counter()
        cohort_diagnostics = Counter() if cohort_input is None else cohort_input
        try:
            with closing(sqlite3.connect(folder / "statistics.sqlite")) as db:
                _configure(db, guard, cache_kib=32768)
                # This database is disposable and never read outside this request.
                db.execute("PRAGMA synchronous=OFF")
                db.create_collation("RFM_DECIMAL", rfm.decimal_compare)
                if rfm_result is None:
                    db.executescript(rfm._SCHEMA)
                if cohort_result is None:
                    db.executescript(cohorts._SCHEMA)

                def feed_orders(orders):
                    for entry in orders:
                        check()
                        if rfm_result is None:
                            rfm._spool_order(db, entry, bounds, rfm_context.orders_intervals, rfm_diagnostics)
                        if cohort_result is None and cohort_input is None:
                            cohorts._spool_order(db, entry, core, item_categories, categories, cohort_diagnostics, check)
                        yield entry

                def feed_actions():
                    for event in actions:
                        check()
                        if rfm_result is None and rfm_population is None:
                            rfm._member(db, event.customer_key)
                        yield event

                def feed_profiles():
                    for key in profiles:
                        check()
                        if rfm_result is None and rfm_population is None:
                            rfm._member(db, key)
                        yield key

                if contexts[4].period is not None and rfm_result is None and rfm_population is None:
                    # The date-restricted general population omits undated profiles.
                    # RFM's all-observed population still includes them.
                    for _ in feed_profiles():
                        pass
                ranges = tuple(context.period for context in (contexts[1], contexts[3], contexts[4]))
                fact_range = None
                if rfm_result is not None and (cohort_result is not None or cohort_input is not None) and all(ranges):
                    fact_range = min(p[0] for p in ranges), max(p[1] for p in ranges)
                orders = (_orders(reader, predicate, contexts[1], check) if prepared is None
                          else prepared.orders(selected_items, contexts[1], check, period=fact_range))
                started = perf_counter()
                with closing(orders), closing(
                        feed_orders(orders)) as grouped, closing(feed_actions()) as events, closing(feed_profiles()) as customers:
                    aggregate = accumulate_aggregates(db, contexts=contexts, orders=grouped, actions=events,
                        profile_keys=customers, metadata=metadata, product_predicate=predicate,
                        action_period=periods["order_period"], profile_details=profile_details,
                        profiles_available=reader.manifest["binding"]["provenance"]["customers"]["present"],
                        category_reference=categories, check=check,
                        raw_summary=prepared.raw[:-1] if prepared is not None else None,
                        action_diagnostics=prepared.action_diagnostics if prepared is not None else None,
                        order_history=prepared.order_history if prepared is not None else None,
                        action_headers=action_headers)
                if measure is not None:
                    measure("aggregate_inputs_and_result", perf_counter() - started)
                started = perf_counter()
                if rfm_result is None:
                    rfm_result = rfm._summarize(db, rfm_context, bounds, rfm_diagnostics, None, check)
                    if rfm_population is not None:
                        rfm_result = replace(rfm_result, population=rfm_population)
                if measure is not None:
                    measure("rfm_summary", perf_counter() - started)
                started = perf_counter()
                if cohort_result is None:
                    cohort_result = cohorts._summarize(
                        db, cohort_context, core, categories, cohort_diagnostics, None, check,
                        source_db=prepared.db if cohort_input is not None else None)
                if measure is not None:
                    measure("cohorts_summary", perf_counter() - started)
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
                                 product_filter=None, cache_only=False, prepared_index=False, remember_selection=None,
                                 phase=None, measure=None):
    """Return a validated report. The caller supplies a private scratch/cache root."""
    if reader not in ("REFERENCE", "INDEXED") or (reader == "INDEXED") != (index_dir is not None):
        raise ValueError("An explicit source and matching index root are required")
    if not isinstance(calculated_at, datetime) or calculated_at.utcoffset() is None:
        raise ValueError("An aware calculation timestamp is required")
    _estimate(expected_bytes)
    calculated_at = calculated_at.astimezone(timezone.utc)
    if cache_only and cache_dir is None:
        raise ValueError("Opening a cached report requires a cache")
    if prepared_index and reader != "INDEXED":
        raise ValueError("Prepared analytics requires an indexed source")
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
            hit = _cached_result(cache, source, generation, provenance, calculated_at, parameters, algorithm, check)
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
    started = perf_counter()
    with reader_scope as input_reader, (
            open_analytics(index, input_reader, check) if prepared_index else nullcontext()) as prepared:
        provenance = input_reader.manifest["binding"]["provenance"]
        source, generation = _binding(reader, provenance, input_reader.manifest if reader == "INDEXED" else None)
        analytics_generation = prepared.generation if prepared is not None else None
        if measure is not None:
            measure("open_indexes", perf_counter() - started)
        metadata = _indexed_metadata(input_reader.db) if reader == "INDEXED" and product_filter is not None else None
        if reader == "INDEXED" and cache is not None:
            started = perf_counter()
            _, parameters = request(metadata)
            hit = _cached_result(cache, source, generation, provenance, calculated_at, parameters, algorithm, check,
                                 analytics_generation)
            if measure is not None:
                measure("cache_hit" if hit is not None else "cache_miss", perf_counter() - started)
        if hit is None and not cache_only:
            if phase is not None:
                phase("Расчёт статистики…")
            selection = MindboxSelectionConfig(**provenance["canonical"]["selection"])
            action_headers = None
            if reader == "REFERENCE":
                identities = _identities(root, provenance, input_reader.check)
                items, metadata, names, paths = reference._catalog_snapshot(catalog)
                actions = _reference_actions(root, provenance, identities, ProductResolver(items), selection, input_reader.check)
                profiles = _reference_profiles(root, identities, input_reader.check)
                demographics = closing(_reference_demographics(root, identities, input_reader.check))
            else:
                if prepared is None:
                    _index_integrity(input_reader, check)
                if metadata is None:
                    metadata = _indexed_metadata(input_reader.db)
                names, paths = cohorts._indexed_categories(input_reader.db, check)
                # RFM population is source-wide and already known from preparation.
                # Undated events still carry scope and quality information.
                action_range = None
                if prepared is not None and periods["order_period"] is not None and periods["customer_period"] is not None:
                    order_range, customer_range = periods["order_period"], periods["customer_period"]
                    action_range = min(order_range[0], customer_range[0]), max(order_range[1], customer_range[1])
                if prepared is not None:
                    action_headers = partial(_indexed_action_headers, reader=input_reader, selection=selection,
                                             period=periods["order_period"], customer_period=periods["customer_period"],
                                             check=check)
                profiles = _indexed_profiles(input_reader)
                demographics = (input_reader.profile_demographics() if prepared is None
                                else closing(prepared.profile_details()))
            items_in_scope, parameters = request(metadata)
            if reader == "INDEXED":
                actions = _indexed_actions(input_reader, selection, check, period=action_range,
                                           selected_items=items_in_scope if action_headers is not None else None)
            with closing(actions), closing(profiles), demographics as profile_details:
                parts = _calculate(input_reader, scratch=scratch, expected_bytes=expected_bytes, cancelled=cancelled,
                    observer=observer, calculated_at=calculated_at, periods=periods, selected_items=items_in_scope,
                    parameters=parameters, metadata=metadata, categories=category_reference(names, paths),
                    actions=actions, profiles=profiles, profile_details=profile_details, prepared=prepared, measure=measure,
                    action_headers=action_headers)
    check()
    if algorithm_version() != algorithm:
        raise StatisticsIndexError("UNIFIED_ALGORITHM_CHANGED")
    if hit is None and cache_only:
        raise StatisticsIndexError("UNIFIED_CACHE_MISS")
    if phase is not None:
        phase("Открытие сохранённого отчёта…" if hit is not None else "Подготовка и сохранение отчёта…")
    started = perf_counter()
    result = hit if hit is not None else assemble_result(
        *parts, item_metadata=metadata, source=source, source_binding=generation,
        parameters=parameters, algorithm=algorithm, check=check)
    if measure is not None and hit is None:
        measure("assemble_dto", perf_counter() - started)
    if cache is not None and (hit is None or remember_selection is not None):
        started = perf_counter()
        with storage_lock(root):
            if not _matches_projection(_capture(root, catalog), provenance):
                raise StatisticsIndexError("STALE_SOURCE")
            if algorithm_version() != algorithm:
                raise StatisticsIndexError("UNIFIED_ALGORITHM_CHANGED")
            if analytics_generation is not None:
                active = _read_json(index / ANALYTICS_POINTER, 1024 * 1024)
                if active.get("generation") != analytics_generation:
                    raise StatisticsIndexError("ANALYTICAL_INDEX_STALE")
            key = _compatibility_key(source, generation, provenance, calculated_at, parameters, algorithm,
                                     analytics_generation)
            if hit is None:
                cache.save(result, check=check, key=key)
            if remember_selection is not None:
                cache.publish_selection(key=key, selection=remember_selection,
                                        fingerprint=result.metadata["fingerprint"],
                                        analytics_generation=analytics_generation, check=check)
        if measure is not None:
            measure("save_report", perf_counter() - started)
    return result
