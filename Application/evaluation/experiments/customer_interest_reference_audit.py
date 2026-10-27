"""Aggregate-only diagnostics of unchanged indexed references; never guess an ID."""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
import random
from time import monotonic

from Application import customer_history_index as index
from Application.customer_history_records import HistoryKind
from Application.evaluation.experiments.customer_interest import Background, ResearchContext, build_profile
from Application.evaluation.experiments.customer_interest_index_pilot import (
    Bounds, Budget, PilotError, convert_events, describe, feature_configs, open_reader, read_customer,
    reference_background, select_customers,
)
from Application.mindbox.adapters._common import AdapterError, product_key
from Application.mindbox.selection import DEFAULT_FAVORITE_ACTIONS, DEFAULT_VIEW_ACTIONS
from Application.model.item_feature_snapshot import FEATURE_FIELDS
from Application.product_resolution import (
    CatalogDiagnostics, ProductCatalog, ProductResolver, ResolutionStatus, normalize_product_key,
)
from Application.recommendation_catalog import RecommendationCatalog


ACTIVE = ("ВидНоменклатуры", "ВидАссортимента", "КатегорияНаСайте")
STAGES = ("VIEW", "PRODUCT_OBJECT", "ANY_ID", "SUPPORTED_NAMESPACE", "UNAMBIGUOUS", "NORMALIZED", "CATALOG", "ACTIVE_ATTRIBUTES", "PROFILE_SIGNAL")


def reference_diagnostic(reference, catalog, namespaces):
    """One occurrence. Candidate values stay private; diagnostics are fixed labels."""
    supported = tuple(k for k in reference.keys if k.namespace in namespaces)
    selected = reference.product
    code = None
    flags = [True, bool(reference.keys), bool(supported), len(supported) == 1]
    if not reference.keys:
        reason = "NO_ID"
    elif not supported:
        reason = "UNSUPPORTED_NAMESPACE"
    elif len(supported) != 1:
        reason = "MULTIPLE_SUPPORTED_NAMESPACES"
    elif selected is None or selected != supported[0]:
        reason = "SELECTED_KEY_INCONSISTENT"
    else:
        normalized = normalize_product_key(selected)
        code = normalized.item_id if normalized.status is ResolutionStatus.RESOLVED else None
        reason = ("INVALID_NORMALIZED_ID" if code is None else "UNKNOWN_CATALOG" if code not in catalog.all_codes else
                  "CATALOG_ATTRIBUTE_CONFLICT" if code not in catalog.items else "RESOLVED")
    normalized_ok = code is not None
    catalog_ok = normalized_ok and code in catalog.all_codes
    attrs_ok = catalog_ok and code in catalog.items and any(catalog.items[code].get(f) for f in ACTIVE)
    if catalog_ok and code in catalog.items and not attrs_ok:
        reason = "NO_ACTIVE_ATTRIBUTES"
    flags += [normalized_ok, catalog_ok, attrs_ok]
    # Comparison uses the same original scalar namespace-choice helper as BPR.
    try:
        bpr = product_key({"ids": {k.namespace: k.value for k in reference.keys}}, namespaces)
        bpr_normalized = normalize_product_key(bpr)
        bpr_code = bpr_normalized.item_id if bpr_normalized.status is ResolutionStatus.RESOLVED else None
    except AdapterError:
        bpr_code = None
    convergence = set()
    if len(supported) > 1:
        for key in supported:
            candidate = normalize_product_key(key)
            if candidate.status is ResolutionStatus.RESOLVED:
                convergence.add(candidate.item_id)
    return {"flags": tuple(flags), "reason": reason, "normalized_code": code,
            "bpr_normalized_code": bpr_code, "supported_count": len(supported),
            "ambiguous_convergent_prefix": len(supported) > 1 and len(convergence) == 1}


def view_funnel(events, catalog, namespaces, resolver, customer, *, kind=HistoryKind.VIEW):
    """Nonempty ALL/ANY funnels, occurrences and unique normalized items kept apart."""
    all_counts, any_counts, occurrences, reasons, combinations, qualities = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    empty_details, event_quality, namespaces_count = Counter(), Counter(), Counter()
    unique, source_keys = defaultdict(set), set()
    collateral, rejected = 0, 0
    bpr_disagreements, convergent = 0, 0
    for event in events:
        if event.kind is not kind:
            continue
        all_counts[kind.value] += 1
        any_counts[kind.value] += 1
        refs = [o.reference for variant in event.variants for o in variant.event.details.products]
        rows = [reference_diagnostic(ref, catalog, namespaces) for ref in refs]
        for flag in event.quality:
            event_quality[flag.value] += 1
        adapted, _ = convert_events(customer, (event,), resolver)
        baseline_rejected = not adapted
        rejected += int(baseline_rejected)
        event_reasons = set()
        if not rows:
            event_reasons.add("NO_PRODUCT_OBJECT")
            for variant in event.variants:
                detail = variant.event.details
                empty_details["with_category_objects" if detail.categories else "without_category_objects"] += 1
                empty_details["with_view_price" if detail.product_view_price is not None else "without_view_price"] += 1
                system = detail.system_name if detail.system_name in (*DEFAULT_VIEW_ACTIONS, *DEFAULT_FAVORITE_ACTIONS) else "OTHER"
                empty_details["action_" + system] += 1
                for source_kind in {o.source_kind for o in variant.origins}:
                    empty_details["source_" + source_kind] += 1
        if baseline_rejected and any(row["flags"][5] for row in rows):
            collateral += 1
        for reference, row in zip(refs, rows):
            for key in reference.keys:
                source_keys.add((key.namespace, key.value))
                namespaces_count[key.namespace if key.namespace in namespaces else "unsupported"] += 1
            for quality in reference.quality:
                qualities[quality.value] += 1
            reasons["OCCURRENCE_" + row["reason"]] += 1
            event_reasons.add(row["reason"])
            bpr_disagreements += int(row["normalized_code"] != row["bpr_normalized_code"])
            convergent += int(row["ambiguous_convergent_prefix"])
            for stage, passed in zip(STAGES[1:-1], row["flags"]):
                if passed:
                    occurrences[stage] += 1
                    if row["normalized_code"] is not None:
                        unique[stage].add(row["normalized_code"])
        for step, stage in enumerate(STAGES[1:-1]):
            all_counts[stage] += int(bool(rows) and all(row["flags"][step] for row in rows))
            any_counts[stage] += int(any(row["flags"][step] for row in rows))
        # Admission here means positive resolved VIEW mass, not a READY user.
        signal = bool(adapted) and any(row["flags"][-1] for row in rows)
        all_counts["PROFILE_SIGNAL"] += int(signal)
        any_counts["PROFILE_SIGNAL"] += int(signal)
        for reason in event_reasons:
            reasons["EVENT_" + reason] += 1
        combinations["+".join(sorted(event_reasons))] += 1
    return {"events_all": dict(all_counts), "events_any": dict(any_counts), "product_occurrences": dict(occurrences),
            "unique_normalized_codes": {k: len(v) for k, v in unique.items()}, "unique_source_keys": len(source_keys),
            "reasons": dict(reasons), "event_reason_combinations": dict(combinations), "quality_occurrences": dict(qualities),
            "baseline_rejected_views" if kind is HistoryKind.VIEW else "baseline_rejected_favorites": rejected,
            "collateral_events_with_resolved_occurrence": collateral,
            "bpr_choice_normalization_disagreements": bpr_disagreements,
            "ambiguous_convergent_occurrences": convergent, "empty_product_details": dict(empty_details),
            "event_quality": dict(event_quality), "namespace_occurrences": dict(namespaces_count)}, unique, source_keys


def scope_matches(prior, activity, buyers, scalar, totals):
    """Use original typed channel counters, never the old prose's conflated denominator."""
    expected_view_count = prior["channels"].get("VIEW", 0)
    expected_rejected_views = expected_view_count - prior["diagnostics"].get("ADAPTED_VIEW", 0)
    read_keys = {"pages", "partial_conflict_pages", "CUSTOMER_HISTORY_LIMIT", "UNUSABLE_CUSTOMER_HISTORY"}
    expected_conversion = Counter({k: v for k, v in prior["diagnostics"].items() if k not in read_keys})
    return (dict(activity) == prior["source_activity_strata"] and buyers == prior["buyers_probed"]
            and scalar["baseline_rejected_views"] == expected_rejected_views
            and totals["events_all"]["VIEW"] == expected_view_count and totals["conversion"] == expected_conversion)


def background_reliability(reference_profiles, cutoff):
    """Descriptive user support and jackknife; no reweighting/smoothing changes."""
    full, support = reference_background(reference_profiles, cutoff)
    result = {}
    for name in ACTIVE:
        q = full.distributions.get(name)
        if not q:
            result[name] = {"support_users": support.get(name, 0), "available": False}
            continue
        contributors = Counter()
        shares = defaultdict(list)
        for profile in reference_profiles:
            field = next(f for f in profile.fields if f.name == name)
            if field.mass:
                for value, mass in field.counts.items():
                    if mass > 1e-12:
                        contributors[value] += 1
                        shares[value].append(mass / field.mass)
        tv, lost = [], []
        for omitted in range(len(reference_profiles)):
            loo, _ = reference_background(reference_profiles[:omitted] + reference_profiles[omitted + 1:], cutoff)
            if name not in loo.distributions:
                continue
            other = loo.distributions[name]
            tv.append(.5 * math.fsum(abs(q.get(v, 0) - other.get(v, 0)) for v in set(q) | set(other)))
            lost.append(sum(p > 1e-12 and other.get(v, 0) <= 1e-12 for v, p in q.items()))
        histogram = Counter("0" if not contributors[v] else "1" if contributors[v] == 1 else "2" if contributors[v] == 2 else "3-4" if contributors[v] <= 4 else "5+" for v in q)
        effective = [(math.fsum(s)**2 / math.fsum(x*x for x in s)) for s in shares.values()]
        result[name] = {"available": True, "support_users": support[name], "vocabulary_size": len(q),
                        "value_user_support_histogram": dict(histogram), "effective_contributors": describe(effective),
                        "raw_q_zero_values": sum(p <= 1e-12 for p in q.values()),
                        "raw_q_positive_min": min(p for p in q.values() if p > 1e-12),
                        "smoothed_floor": .05 / len(q), "loo_total_variation": describe(tv),
                        "loo_values_losing_all_evidence": describe(lost)}
    return result


def audit_worker(output):
    """Re-read the fixed bounded index scope only for diagnosis, no comparative pilot."""
    from Application.model.propensity_ml.resources import ResourceGuard
    from Application.model.propensity_ml.contracts import MLDataError, MLCancelled
    from scripts.run_customer_interest_index_pilot import ROOT, PRIVATE_LIMIT, index_preflight
    started = monotonic()
    prior = json.loads((ROOT / "filtered_data/.customer-interest-profile-03-20261026/loo-confirmed/PILOT.json").read_text(encoding="utf-8"))
    limits = Bounds()
    resources = ResourceGuard((output,), cancelled=lambda: (output / "CANCEL").exists(), private_limit=PRIVATE_LIMIT)
    def poll():
        try:
            resources.poll(force=True)
        except (MLDataError, MLCancelled) as exc:
            raise PilotError("CANCELLED" if isinstance(exc, MLCancelled) else str(exc)) from None
        if resources.peak_working_set > PRIVATE_LIMIT or sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > limits.output_bytes:
            raise PilotError("MEMORY_OR_OUTPUT_LIMIT")
    budget = Budget(limits, poll=poll, cancelled=lambda: (output / "CANCEL").exists())
    budget.check()
    service = index.CustomerHistoryIndexService()
    ready, preflight = index_preflight(service, index.DEFAULT_RAW_ROOT)
    paths = [ROOT / "input_data" / name for name in ("nomenclature.csv", "site_categories.csv")]
    if any(p.stat().st_size > 16 * 2**20 for p in paths):
        raise PilotError("CATALOG_INPUT_LIMIT")
    catalog = RecommendationCatalog(*paths).load()
    if len(catalog.all_codes) > 50_000:
        raise PilotError("CATALOG_CODES_LIMIT")
    resolver = ProductResolver(ProductCatalog(frozenset(catalog.items), CatalogDiagnostics()))
    cutoff = datetime.fromisoformat(prior["window"][1])
    since = cutoff - timedelta(days=365)
    context = ResearchContext(cutoff, datetime.fromtimestamp(paths[0].stat().st_mtime, timezone.utc),
        mode="EXPLORATORY", dataset_kind="INDEX_EXPLORATORY", knowledge_cutoff=datetime.fromisoformat(prior["knowledge_cutoff"]))
    namespaces = index._capture_sources(index.DEFAULT_RAW_ROOT).selection.action_product_namespaces
    totals, unique, source_keys = defaultdict(Counter), defaultdict(set), set()
    scalar = Counter()
    cohort, observed = [], set()
    panel = tuple(random.Random(prior["seed"]).sample(sorted(catalog.items), 256))
    activity = Counter()
    with open_reader(service, index.DEFAULT_RAW_ROOT, budget) as (connection, same):
        for customer in select_customers(connection, budget, seed=prior["seed"]):
            events, read_diag = read_customer(connection, same, customer, since, cutoff, budget)
            if events is None:
                scalar.update(read_diag)
                continue
            n = len(events)
            activity["0" if n == 0 else "1-5" if n <= 5 else "6-30" if n <= 30 else "31-200"] += 1
            funnel, codes, keys = view_funnel(events, catalog, namespaces, resolver, customer)
            source_keys.update(keys)
            for key, value in funnel.items():
                if isinstance(value, dict):
                    if key != "unique_normalized_codes":
                        totals[key].update(value)
                else:
                    if key != "unique_source_keys":
                        scalar[key] += value
            for stage, values in codes.items():
                unique[stage].update(values)
            favorite, _, _ = view_funnel(events, catalog, namespaces, resolver, customer, kind=HistoryKind.FAVORITE)
            for key, value in favorite.items():
                if isinstance(value, dict):
                    if key != "unique_normalized_codes":
                        totals["favorite_" + key].update(value)
                elif key == "baseline_rejected_favorites":
                    scalar[key] += value
            adapted, conversion = convert_events(customer, events, resolver)
            totals["conversion"].update(conversion)
            cohort.append((customer, adapted, n))
            for event in events:
                for variant in event.variants:
                    detail = variant.event.details
                    refs = detail.products if event.source.value == "ACTION" else detail.lines
                    for occurrence in refs:
                        product = occurrence.reference.product
                        if product is not None:
                            resolved = resolver.resolve(product, strict=False)
                            if resolved.status is ResolutionStatus.RESOLVED:
                                observed.add(resolved.item_id)
    if not scope_matches(prior, activity, budget.customers, scalar, totals):
        return {"status": "BLOCKED", "reason": "FIXED_COHORT_OR_BASELINE_MISMATCH",
                "buyers_probed": budget.customers, "activity": dict(activity), "scalars": dict(scalar),
                "funnel": {k: dict(v) for k,v in totals.items()}, "event_rows_read_including_lookahead": budget.events,
                "resources": resources.payload()}
    selected = tuple(catalog.items[c] for c in sorted(observed | set(panel)))
    if len(selected) > limits.max_catalog_items:
        raise PilotError("PROFILE_CATALOG_LIMIT")
    vocabulary = {f: sorted({i.get(f) for i in selected if i.get(f)}) for f in FEATURE_FIELDS}
    uniform = Background({f: {v: 1/len(vs) for v in vs} for f, vs in vocabulary.items() if vs}, 0, cutoff, False)
    config = feature_configs()["WITH_CATEGORY"]
    initial = []
    for customer, events, count in cohort:
        budget.check()
        initial.append((customer, events, count, build_profile(customer, events, selected, uniform, context, config)))
    references = [row for row in initial if row[3].diagnostics["total_mass"] > 0][:24]
    background, field_support = reference_background([row[3] for row in references], cutoff)
    statuses, causes, thresholds = Counter(), Counter(), Counter()
    for customer, events, count, _ in initial:
        budget.check()
        bg = reference_background([row[3] for row in references if row[0] != customer], cutoff)[0] if any(row[0] == customer for row in references) else background
        profile = build_profile(customer, events, selected, bg, context, config)
        statuses[profile.status] += 1
        d = profile.diagnostics
        if not count:
            causes["NO_EVENTS_IN_WINDOW"] += 1
        elif profile.status == "EMPTY":
            causes["EVENTS_WITHOUT_POSITIVE_USABLE_MASS"] += 1
        elif profile.status == "INSUFFICIENT_HISTORY":
            causes["POSITIVE_BUT_INSUFFICIENT_HISTORY"] += 1
            for name, fail in {"DISTINCT_ITEMS_LT3": d["distinct_items"] < 3, "EVIDENCE_EVENTS_LT3": d["evidence_events"] < 3,
                               "MASS_LT1": d["total_mass"] < 1, "EFFECTIVE_ITEMS_LT2": d["effective_items"] < 2}.items():
                thresholds[name] += int(fail)
        else:
            causes[profile.status] += 1
    if dict(statuses) != prior["evaluation"]["statuses"]["WITH_CATEGORY"]:
        raise PilotError("STATUS_RECONSTRUCTION_MISMATCH")
    reliability = background_reliability([row[3] for row in references], cutoff)
    after = service.get_status(raw_root=index.DEFAULT_RAW_ROOT, verify_integrity=False)
    if after.generation != ready.generation or after.source_version != ready.source_version or after.status is not index.HistoryIndexStatus.READY:
        raise PilotError("SOURCES_CHANGED")
    budget.check()
    return {"status": "COMPLETED", "mode": "EXPLORATORY_DIAGNOSTIC_ONLY", "preflight": preflight,
            "buyers_probed": budget.customers, "complete_bounded_histories": len(cohort), "event_rows_read_including_lookahead": budget.events,
            "sampling_unchanged": True, "activity": dict(activity), "scalars": dict(scalar),
            "funnel": {k: dict(v) for k,v in totals.items()}, "unique_normalized_codes": {k: len(v) for k,v in unique.items()},
            "unique_source_keys": len(source_keys), "baseline_statuses": dict(statuses), "status_causes": dict(causes),
            "insufficient_thresholds_overlapping": dict(thresholds), "field_reference_support": field_support,
            "background_reliability": reliability, "comparative_pilot_executed": False, "correction_applied": False,
            "resources": resources.payload(), "diagnostic_seconds": monotonic() - started}
