"""Reproducible canonical read-only temporal audit, aggregate Markdown and JSON."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Application.evaluation.audit import audit_scenarios, load_canonical_events, markdown_summary
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
from Application.product_resolution import DEFAULT_CATALOG_PATH


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "diagnostics/temporal_train03a")
    parser.add_argument("--horizons", type=int, nargs="+", default=[14, 30, 60])
    parser.add_argument("--thresholds", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--sensitivity-horizon", type=int, default=30)
    parser.add_argument("--test-end", type=datetime.fromisoformat)
    args = parser.parse_args(argv)
    if any(n < 1 for n in (*args.horizons, *args.thresholds, args.sensitivity_horizon)):
        parser.error("Horizons and history thresholds must be positive")
    if args.test_end is not None and (args.test_end.tzinfo is None or args.test_end.utcoffset() is None):
        parser.error("--test-end must include timezone")
    output = args.output_dir.resolve()
    protected = [args.raw_root.resolve(), args.catalog.resolve().parent,
                 PROJECT_ROOT / "input_data", PROJECT_ROOT / "model", PROJECT_ROOT / "user_settings"]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        parser.error("Output must be a separate diagnostic directory outside source/model/settings trees")
    return args


def main(argv=None):
    args = parse_args(argv)
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    try:
        events, source = load_canonical_events(args.raw_root, args.catalog, progress)
        report = audit_scenarios(events, source, horizons=args.horizons, thresholds=args.thresholds,
                                 sensitivity_horizon=args.sensitivity_horizon, test_end=args.test_end, progress=progress)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "report.md").write_text(markdown_summary(report), encoding="utf-8")
        print(json.dumps({"resolved_events": len(events), "scenarios": len(report["scenarios"]),
                          "unavailable": len(report["unavailable"])}, sort_keys=True))
        return 0
    except Exception as exc:
        # Unexpected adapter/filesystem errors must not expose raw values or customer IDs.
        print(json.dumps({"error": "TEMPORAL_AUDIT_FAILED", "exception_type": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
