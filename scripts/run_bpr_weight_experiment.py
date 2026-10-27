"""Run the fixed BPR-EXP-01 controlled validation benchmark, without publication."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from Application.evaluation.audit import load_canonical_events
from Application.evaluation.experiments.bpr_weights import (
    benchmark_config, freeze_plan, frozen_config, git_provenance, run_experiment, run_single_baseline,
)
from Application.evaluation.temporal import build_temporal_protocol
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
from Application.product_resolution import DEFAULT_CATALOG_PATH


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "user_settings/train_config.json")
    parser.add_argument("--mode", choices=("baseline", "sweep"), default="baseline")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / ("diagnostics/bpr_exp01_baseline_cost" if args.mode == "baseline"
                                          else "diagnostics/bpr_exp01_weights")
    output = args.output_dir.resolve()
    protected = (args.raw_root, args.catalog.parent, args.settings.parent,
                 PROJECT_ROOT / "input_data", PROJECT_ROOT / "model", PROJECT_ROOT / "user_settings")
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        parser.error("Output must be separate from source/model/settings trees")
    if (output / "results.json").exists():
        parser.error("Experiment results already exist; choose a new output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = frozen_config(args.settings)
    temporal = benchmark_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = freeze_plan(cfg, temporal, device, git_provenance(PROJECT_ROOT))
    progress = lambda message: print(message, flush=True)
    events, source = load_canonical_events(args.raw_root, args.catalog, progress)
    protocol = build_temporal_protocol(events, temporal)
    snapshot = protocol.validation
    del events, protocol  # Test snapshot is never passed to training/evaluation.
    plan["canonical_source"] = {key: source[key] for key in ("canonical_revision",) if key in source}
    try:
        execute = run_single_baseline if args.mode == "baseline" else run_experiment
        report = execute(snapshot, temporal, cfg, device, plan, args.output_dir, progress=progress)
    except Exception as exc:
        print("BPR_EXPERIMENT_FAILED: " + type(exc).__name__, file=sys.stderr, flush=True)
        return 1
    print("Completed: " + report["status"] + "; no automatic follow-up.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
