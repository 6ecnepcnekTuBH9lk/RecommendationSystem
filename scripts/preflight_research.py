"""Read-only Research ingestion and temporal preparation; never train or score."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Application.evaluation.audit import load_canonical_events, summarize_snapshot
from Application.evaluation.experiments.bpr_weights import benchmark_config
from Application.evaluation.temporal import build_temporal_protocol
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
from Application.model.training_quality import TrainingQualityConfig
from Application.product_resolution import DEFAULT_CATALOG_PATH


def file_state(paths):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()}


def prepare_report(events, source, config, progress):
    """Use actual Research snapshots; count aggregation keys without BPR arrays.

    All frozen weights are positive, internal holdout is disabled: one BPR
    positive pair per distinct user/item in seen_at_cutoff, regardless of repeats.
    Drain the original list while the standard builder normalizes records, to
    avoid retaining two complete millions-of-events representations at once.
    """
    lower, upper = (datetime.fromisoformat(source[k]) for k in ("declared_start", "declared_end"))
    if any(not lower <= e.interaction.event_datetime_utc < upper for e in events):
        raise ValueError("Interaction outside declared dataset window")
    types = Counter(e.interaction.interaction_type.value for e in events)
    users = {e.interaction.customer_id for e in events}
    items = {e.item_id for e in events}
    stamps = [e.interaction.event_datetime_utc for e in events]
    history = {"events": len(events), "by_type": dict(types), "users": len(users), "items": len(items),
               "earliest": min(stamps).isoformat() if stamps else None,
               "latest": max(stamps).isoformat() if stamps else None}
    del users, items, stamps
    windows = {}
    for name, start, end in (("validation", config.validation_start, config.test_start),
                             ("test", config.test_start, config.test_end)):
        selected = [e for e in events if e.interaction.event_datetime_utc >= start
                    and (end is None or e.interaction.event_datetime_utc < end)]
        windows[name] = {"start": start.isoformat(), "end_exclusive": end.isoformat() if end else None,
                         "interactions": len(selected),
                         "users": len({e.interaction.customer_id for e in selected}),
                         "items": len({e.item_id for e in selected}),
                         "unique_positive_pairs": len({(e.interaction.customer_id, e.item_id) for e in selected})}
        del selected

    def consume():
        for index, event in enumerate(events):
            yield event
            events[index] = None

    progress("Building existing temporal protocol; no training or scoring")
    protocol = build_temporal_protocol(consume(), config)
    events.clear()
    snapshots = {}
    for name, snapshot in (("validation", protocol.validation), ("test", protocol.test)):
        progress(f"Checking {name} snapshot invariants")
        result = summarize_snapshot(snapshot)
        counts = Counter(e.interaction.customer_id for e in snapshot.history)
        result["invariants"]["history_threshold_violation"] = sum(
            counts[c.customer_id] < config.min_history_events for c in snapshot.cases)
        if any(result["invariants"].values()):
            raise ValueError("Research temporal invariants failed")
        result["positive_training_pairs"] = sum(len(seen) for seen in snapshot.seen_at_cutoff.values())
        result["history_start_inclusive"] = source["declared_start"]
        snapshots[name] = result
    train = {"start_inclusive": source["declared_start"], "end_exclusive": config.validation_start.isoformat(),
             "interactions": protocol.validation.diagnostics.training_events,
             "users": protocol.validation.diagnostics.training_users,
             "items": protocol.validation.diagnostics.training_items,
             "positive_training_pairs": snapshots["validation"]["positive_training_pairs"]}
    return {"history": history, "train": train, "future_windows": windows, "snapshots": snapshots,
            "temporal_config": {k: v.isoformat() if isinstance(v, datetime) else v for k, v in asdict(config).items()},
            "bpr_arrays_materialized": False,
            "pair_count_method": "sum unique history items per user; exact existing whole-history aggregation keys"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics/data_research_preflight01")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    # Do not allow a diagnostic result to target working inputs, models or settings.
    if not output.is_relative_to(PROJECT_ROOT / "diagnostics"):
        parser.error("Preflight output must be inside project diagnostics")
    if output.is_relative_to(args.raw_root.resolve()) or output.is_relative_to(args.catalog.resolve().parent):
        parser.error("Preflight output cannot target source data")
    raw_root = args.raw_root.resolve()
    paths = list((raw_root / "canonical").rglob("*"))
    paths += list((PROJECT_ROOT / "user_settings").rglob("*"))
    paths += list((PROJECT_ROOT / "model").rglob("*"))
    paths.append(args.catalog.resolve())
    before = file_state(paths)
    started = time.monotonic()

    def progress(message):
        print(f"{time.monotonic() - started:.0f}s: {message}", flush=True)

    try:
        events, source = load_canonical_events(raw_root, args.catalog, progress=progress, read_only=True)
        progress("Canonical ingestion complete")
        report = prepare_report(events, source, benchmark_config(), progress)
        if not report["history"]["events"] or not report["snapshots"]["validation"]["diagnostics"]["evaluation_targets"]:
            raise ValueError("Validation benchmark unavailable")
        if before != file_state(paths):
            raise ValueError("Protected inputs changed during preflight")
        detail = source["ingestion"]["resolution_by_type"].values()
        attempts = sum(c["interactions_total"] for c in detail)
        unknown = sum(c["unknown_candidate"] for c in detail)
        policy = TrainingQualityConfig()
        reasons = []
        if source["ingestion"]["orders_duplicate_conflicting"]:
            reasons.append("CONFLICTING_ORDER_SNAPSHOTS")
        if any(c["invalid_id"] or c["unsupported_namespace"] for c in detail):
            reasons.append("INVALID_ID_OR_UNSUPPORTED_NAMESPACE")
        if unknown > policy.unknown_candidate_max_count or (attempts and unknown / attempts > policy.unknown_candidate_max_rate):
            reasons.append("UNRESOLVED_PRODUCT")
        report.update(source=source, protected_files_unchanged=True, protected_file_count=len(before),
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      seconds=time.monotonic() - started, model_training=False, model_scoring=False,
                      model_publication=False, research_status="READY WITH WARNINGS",
                      frozen_reference={"seed": 42, "epochs_reference_only": 50, "VIEW": 0.5,
                                        "FAVORITE": 2.0, "PURCHASE": 10.0, "early_stop": False,
                                        "item_features": False},
                      production_policy_projection={"status": "BLOCK" if reasons else "NOT VERIFIED",
                                                    "block_reasons": reasons, "executed_preflight": False,
                                                    "note": "Research-scoped counters; production has its own preparation scope"})
        output.mkdir(parents=True, exist_ok=True)
        (output / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        progress("Preflight complete; aggregate report saved")
        return 0
    except Exception as exc:
        print(f"Research preflight failed ({type(exc).__name__}); no training or publication performed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
