"""Isolated synthetic evaluation and opt-in bounded, read-only real diagnostics.

No real quality backtest: source availability, past catalog and independent
future labels are not established. Never loads model tensors or builds indexes.
"""

import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import timedelta
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Application.evaluation.experiments.customer_interest import InterestConfig, build_profile
from Application.evaluation.experiments.customer_interest_evaluation import (
    ACTIVE_FIELDS, PROFILE_WEIGHTS, InterestEvaluationError, PilotLimits, categorical_association, rerank_top10,
)
from Application.hybrid_recommendations import compose
from Application.interactions import InteractionType
from Application.model.raw_bpr_inference import RawBprRecommendation, RawBprResult, RawBprStatus
from Application.purchased_products import PurchaseEvidence
from Application.recommendation_catalog import RecommendationCatalog
from Application.recommendation_rules import RecommendationRules
from scripts.demo_customer_interest import action, fixtures, order


def rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Memory(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("peak", "working", "quota_peak_paged", "quota_paged",
                "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile")]
        memory = Memory()
        memory.cb = ctypes.sizeof(memory)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
            raise InterestEvaluationError("MEMORY_MEASUREMENT_UNAVAILABLE")
        return memory.peak  # Peak working set: conservative bound on this Windows process.
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def ablations():
    base = InterestConfig()
    result = {"FULL": base}
    for name in ACTIVE_FIELDS:
        result["ONLY_" + name] = replace(base, field_weights=tuple((f, float(f == name)) for f, _ in base.field_weights))
        result["WITHOUT_" + name] = replace(base, field_weights=tuple((f, w if f != name else 0.) for f, w in base.field_weights))
    result["NO_DECAY_APPROX"] = replace(base, half_lives=tuple((kind, 1e12) for kind, _ in base.half_lives))
    result["EQUAL_CHANNELS"] = replace(base, action_weights=tuple((kind, 1.) for kind, _ in base.action_weights))
    result["UNIFORM_REFERENCE"] = base
    return result


def synthetic_baseline(items, excluded=()):
    """Call the actual unchanged compose; all data and purchase evidence synthetic."""
    complete = [replace(item, stock=Decimal(5)) for item in items if item.code[0] in ("1", "2")]
    catalog = SimpleNamespace(items={i.code: i for i in complete},
                              kinds=frozenset(i.get("ВидНоменклатуры") for i in complete),
                              collections=frozenset(i.get("Коллекция") for i in complete),
                              category_paths={i.get("КатегорияНаСайте"): frozenset((i.get("КатегорияНаСайте"),)) for i in complete})
    # More sources than ten, but production compose still claims only ten targets.
    sources = [complete[(n % 2) * 6 + n // 2 % 6] for n in range(40)]
    metadata = {f"source-{n}": dict(i.attributes) for n, i in enumerate(sources)}
    raw = RawBprResult("synthetic", "synthetic", RawBprStatus.OK, tuple(
        RawBprRecommendation(n + 1, f"source-{n}", float(100 - n)) for n in range(len(sources))))
    evidence = PurchaseEvidence("READY", frozenset(excluded), historical=PurchaseEvidence(
        "READY", checked_codes=frozenset(metadata)))
    rules = RecommendationRules(tuple(sorted(catalog.kinds)), tuple(sorted(catalog.collections)), 1)
    return compose(raw, metadata, catalog, rules, evidence)


def synthetic_evaluation():
    items, reference, context = fixtures()
    fav = [action(f"10000{i}", i, InteractionType.FAVORITE) for i in range(1, 4)]
    cases = {
        "empty": ([], "100005", ()),
        "one_view": ([action("100001", 1)], "200005", ()),
        "short_views": ([action(f"10000{i}", i) for i in range(1, 4)], "100005", ()),
        "favorites_aligned": (fav, "100005", ()),
        "favorites_opposite_target": (fav, "200005", ()),
        "balanced": (fav + [action(f"20000{i}", i, InteractionType.FAVORITE) for i in range(1, 4)], "100005", ()),
        "recent_shift": (fav + [action(f"20000{i}", i + 200, InteractionType.FAVORITE) for i in range(1, 4)], "100005", ()),
        "purchases": ([order(f"10000{i}", i) for i in range(1, 4)], "100005", ("100001", "100002", "100003")),
        "cancelled": ([order(f"10000{i}", i, ("otmenen",)) for i in range(1, 4)], "200005", ()),
        "conflicting": ([order(f"10000{i}", i, ("CP", "otmenen")) for i in range(1, 4)], "200005", ()),
        "long_repetition": ([action(f"10000{i}", day, identifier=f"repeat-{i}-{day}")
                             for i in range(1, 4) for day in range(1, 101)], "100005", ()),
        "outside_pool": (fav, "999999", ()),
    }
    # Non-uniform, artificial reference makes the uniform-prior ablation meaningful.
    distributions = {f: dict(q) for f, q in reference.distributions.items()}
    for f in ACTIVE_FIELDS:
        first, second = distributions[f]
        distributions[f] = {first: .8, second: .2}
    reference = replace(reference, distributions=distributions)
    rows = []
    for name, config in ablations().items():
        bg = reference
        if name == "UNIFORM_REFERENCE":
            bg = replace(reference, distributions={f: {v: 1 / len(q) for v in q} for f, q in distributions.items()})
        for case, (events, target, excluded) in cases.items():
            baseline = synthetic_baseline(items, excluded)
            profile = build_profile("synthetic", events, items, bg, context, config)
            for weight in PROFILE_WEIGHTS:
                result = rerank_top10(baseline, profile, weight=weight, target=target)
                rows.append({"ablation": name, "case": case, "events": len(events), "weight": weight,
                             "status": profile.status, "candidate_count": len(baseline), "changed": result.changed,
                             "recall": result.recall, "ndcg": result.ndcg, "hit_rate": result.hit_rate,
                             "nonzero": sum(abs(v) > 1e-12 for v in result.profile_scores),
                             "profile_scores": result.profile_scores})
    return {"kind": "SYNTHETIC_MECHANICAL_VALIDATION", "independent_customers": 0,
            "selected_weight": None, "grid": PROFILE_WEIGHTS, "rows": rows}


def catalog_diagnostics():
    paths = [ROOT / "input_data" / name for name in ("nomenclature.csv", "site_categories.csv")]
    if any(p.stat().st_size > 16 * 2**20 for p in paths):
        raise InterestEvaluationError("CATALOG_FILE_BOUND")
    catalog = RecommendationCatalog(*paths).load()
    if len(catalog.all_codes) > 50_000:
        raise InterestEvaluationError("CATALOG_ROW_BOUND")
    rows = tuple(catalog.items.values())
    fills = {f: sum(bool(i.get(f)) for i in rows) for f in ACTIVE_FIELDS}
    coverage = Counter(round(sum(w for f, w in InterestConfig().field_weights if i.get(f)), 2) for i in rows)
    pairs = {f"{a} / {b}": categorical_association((i.get(a), i.get(b)) for i in rows)
             for n, a in enumerate(ACTIVE_FIELDS) for b in ACTIVE_FIELDS[n + 1:]}
    return {"trusted_items": len(rows), "all_codes": len(catalog.all_codes), "diagnostics": catalog.diagnostics,
            "filled": fills, "weighted_coverage_counts": dict(coverage), "associations": pairs,
            "source_hashes": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}


def real_diagnostics(output):
    """Sample current BPR identity scope; aggregate metadata only, no profiles/ranking.

    Uses indexed seeks and cap+1, never estimates a complete profile from a
    truncated customer. Source availability guard deliberately blocks scoring.
    """
    from Application import customer_history_index as index
    from Application.customer_history_service import _coverage_complete
    from Application.model.customer_identity_bridge import CustomerIdentityBridge, _capture_sources
    limits = PilotLimits()
    started, seen, events_read = perf_counter(), set(), 0
    strata, kinds, statuses = Counter(), Counter(), Counter()
    peak = 0
    def guard():
        nonlocal peak
        peak = max(peak, rss_bytes())
        limits.check(customers=len(seen), events=events_read, rss_bytes=peak,
                     output_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
                     free_bytes=shutil.disk_usage(output).free, elapsed_seconds=perf_counter() - started)
    guard()
    service = index.CustomerHistoryIndexService()
    ready = service.get_status(verify_integrity=False)
    if ready.status.value != "READY":
        return {"status": "BLOCKED_INDEX_" + ready.status.value, "limits": asdict(limits)}
    sources = _capture_sources(ROOT / "model", index.DEFAULT_RAW_ROOT)
    from Application.mindbox.raw_reader import part_files
    merge_files = part_files(sources.merges_directory, "customer_merges")
    input_sizes = {"mapping_bytes": sources.mappings_path.stat().st_size,
                   "merge_bytes": sum(p.stat().st_size for p in merge_files)}
    if (input_sizes["mapping_bytes"] > 8 * 2**20 or not merge_files or input_sizes["merge_bytes"] > 32 * 2**20):
        return {"status": "BLOCKED_IDENTITY_INPUT_BOUND", "limits": asdict(limits),
                "input_bytes": input_sizes, "mapping_limit_bytes": 8 * 2**20, "merges_limit_bytes": 32 * 2**20,
                "real_profiles_built": 0, "history_rows_read": 0, "peak_rss_mib": rss_bytes() / 2**20,
                "elapsed_seconds": perf_counter() - started}
    users = json.loads(sources.mappings_path.read_text(encoding="utf-8"))["idx2user"]
    if len(users) > 250_000:
        raise InterestEvaluationError("AUDIENCE_BOUND")
    sample = random.Random(20261026).sample(users, min(limits.max_customers, len(users)))
    bridge = CustomerIdentityBridge()
    cutoff = max(c.until_utc for c in ready.coverage)
    since = cutoff - timedelta(days=365)
    complete = _coverage_complete(ready.coverage, ("scope", ("ORDER", "VIEW", "FAVORITE", "UNCLASSIFIED"),
                                                index._micros(since), index._micros(cutoff), 50))
    if not complete:
        return {"status": "BLOCKED_SOURCE_COVERAGE", "limits": asdict(limits)}
    sql = ("SELECT e.kind,e.variant_count,e.owner_conflict,e.time_conflict,e.kind_conflict "
           "FROM event_customers ec INDEXED BY event_customers_customer JOIN events e USING(event_key) "
           "WHERE ec.canonical_customer_id=? AND ec.event_us>=? AND ec.event_us<? "
           "ORDER BY ec.event_us DESC,ec.event_key DESC LIMIT ?")
    flags = Counter()
    with index._reader_lock(index.DEFAULT_RAW_ROOT):
        captured = index._capture_sources(index.DEFAULT_RAW_ROOT)
        with index._read_index(service.index_path()) as con:
            con.set_progress_handler(lambda: int(perf_counter() - started > limits.max_seconds), 10_000)
            inspected = index._inspect_connection(con, captured)
            source_dates = con.execute("SELECT component,observed_us FROM sources").fetchall()
            plan = [row[-1] for row in con.execute("EXPLAIN QUERY PLAN " + sql, ("probe", index._micros(since), index._micros(cutoff), 401))]
            if any("SCAN " in detail or "TEMP B-TREE" in detail for detail in plan):
                raise InterestEvaluationError("UNBOUNDED_QUERY_PLAN")
            for user in sample:
                guard()
                identity = bridge.resolve(user, model_dir=ROOT / "model", raw_root=index.DEFAULT_RAW_ROOT)
                statuses[identity.status.value] += 1
                if not identity.is_matched or identity.canonical_customer_id in seen:
                    continue
                seen.add(identity.canonical_customer_id)
                if events_read + limits.max_events_per_customer + 1 > limits.max_events:
                    raise InterestEvaluationError("TOTAL_EVENT_PREFLIGHT_BOUND")
                rows = con.execute(sql, (identity.canonical_customer_id, index._micros(since), index._micros(cutoff),
                                         limits.max_events_per_customer + 1)).fetchall()
                events_read += len(rows)
                n = len(rows)
                group = "0" if not n else "1-5" if n <= 5 else "6-30" if n <= 30 else "31-400" if n <= 400 else ">400_SKIPPED"
                strata[group] += 1
                if n > limits.max_events_per_customer:
                    continue  # No partial-customer statistics masquerading as complete history.
                for kind, variants, owner, time, kind_conflict in rows:
                    kinds[kind or "UNKNOWN"] += 1
                    flags["multiple_variants"] += int(variants > 1)
                    flags["owner_conflict"] += owner
                    flags["time_conflict"] += time
                    flags["kind_conflict"] += kind_conflict
            if (index._capture_sources(index.DEFAULT_RAW_ROOT).version != captured.version
                    or not bridge.is_current(identity, model_dir=ROOT / "model", raw_root=index.DEFAULT_RAW_ROOT)):
                raise InterestEvaluationError("REAL_SOURCES_CHANGED")
            if inspected.generation != ready.generation:
                raise InterestEvaluationError("HISTORY_GENERATION_CHANGED")
    guard()
    return {"status": "DIAGNOSTICS_ONLY_PROFILE_SCORING_BLOCKED", "audience": "current BPR model users",
            "seed": 20261026, "sample_size": len(sample), "unique_matched": len(seen), "identity_statuses": dict(statuses),
            "window": [since.isoformat(), cutoff.isoformat()], "coverage_complete": complete,
            "strata": dict(strata), "kinds_complete_customers": dict(kinds), "flags": dict(flags),
            "source_publications_after_cutoff": sum(date > index._micros(cutoff) for _, date in source_dates),
            "source_components": len(source_dates), "query_plan": plan, "rows_read_including_cap_probes": events_read,
            "peak_rss_mib": peak / 2**20, "elapsed_seconds": perf_counter() - started,
            "free_gib": shutil.disk_usage(output).free / 2**30, "limits": asdict(limits),
            "profile_metrics": None, "quality_metrics": None, "real_profiles_built": 0,
            "reason": "FIRST_AVAILABILITY_UNVERIFIED; LATE_PUBLICATIONS; NO_INDEPENDENT_FUTURE_LABELS"}


def main(argv=None):
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    research = ROOT / "filtered_data"
    if research not in output.parents or output == research:
        parser.error("Use a separate folder under filtered_data")
    output.mkdir(parents=True, exist_ok=True)
    result = {"synthetic": synthetic_evaluation()}
    if args.real_diagnostics:
        result["catalog"] = catalog_diagnostics()
        result["real"] = real_diagnostics(output)
    result["peak_rss_mib"] = rss_bytes() / 2**20
    result["computation_seconds"] = perf_counter() - started
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > PilotLimits().max_output_bytes // 2:
        raise InterestEvaluationError("OUTPUT_SERIALIZATION_BOUND")
    with (output / "EVALUATION.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"synthetic_rows": len(result["synthetic"]["rows"]), "real": result.get("real"),
                      "catalog": result.get("catalog")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
