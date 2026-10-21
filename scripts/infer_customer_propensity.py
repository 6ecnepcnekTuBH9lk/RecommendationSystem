"""Explicit, opt-in research batch inference. Never trains or changes model pointers."""
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
from threading import Event
from time import monotonic

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "2")

import numpy as np

from Application.model.customer_propensity_history import PropensityCancelled, PropensityDataError
from Application.model.customer_propensity_storage import iter_propensity_rows
from Application.model.propensity_ml.contracts import MLCancelled, MLDataError, write_json
from Application.model.propensity_ml.predictions import PredictionStore, build_predictions, validate_inputs, validate_payload
from Application.model.propensity_ml.resources import ResourceGuard, sample_resources


def emit(value):
    print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true", help="Full validated input pass; no scoring")
    modes.add_argument("--smoke", action="store_true", help="At most 128 real input rows; no prediction publication")
    modes.add_argument("--run", action="store_true", help="Authorize one complete inference attempt")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("filtered_data/customer_propensity_predictions"))
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args(argv)
    if not args.report_dir.is_dir():
        parser.error("Supply an existing owned report directory")
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    cancelled = lambda: stop.is_set() or (args.cancel_file is not None and args.cancel_file.exists())
    samples = []
    reserve_path = args.output_root if args.output_root.exists() else args.output_root.parent
    guard = ResourceGuard([args.report_dir, reserve_path], cancelled=cancelled, observer=samples.append)
    started = monotonic()
    mode = "inference" if args.run else "preflight" if args.preflight else "smoke"
    try:
        guard.poll(force=True)
        if args.preflight:
            result = validate_payload(args.model, args.dataset, guard)
            write_json(args.report_dir / "inference_manifest_validation.json", result)
        elif args.smoke:
            model, m = validate_inputs(args.model, args.dataset)
            rows = []
            reader = iter_propensity_rows(args.dataset, "inference", purpose="inference",
                                         source_version=m["source_version"], generation=m["generation"],
                                         cancelled=lambda: (guard.poll() or False))
            try:
                for row in reader:
                    rows.append(row)
                    if len(rows) == 128:
                        break
            finally:
                reader.close()
            p = model.predict(rows, key_id=m["identity"]["key_id"])
            split = np.concatenate([model.predict(rows[:37]), model.predict(rows[37:])])
            if not np.array_equal(p, split) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise MLDataError("INFERENCE_SMOKE_FAILED")
            result = {"rows": len(rows), "batch_equal": True, "finite_range_valid": True,
                      "model_version": model.metadata["version"], "pipeline": "calibrated"}
            write_json(args.report_dir / "smoke.json", result)
        else:
            final = build_predictions(args.model, args.dataset, args.output_root, guard,
                                      batch_size=args.batch_size, progress=emit)
            store = PredictionStore(final)
            result = {"version": store.manifest["version"], "rows": store.manifest["rows"],
                      "model_version": store.manifest["contract"]["model_version"],
                      "seconds": monotonic() - started, "summary": store.manifest["summary"],
                      "artifact_bytes": sum(p.stat().st_size for p in final.iterdir()),
                      "status": "RESEARCH_ONLY", "pipeline": "calibrated"}
            write_json(args.report_dir / "publication_validation.json", result)
        if not args.run:
            guard.poll(force=True)
        write_json(args.report_dir / (mode + "_resources.json"), guard.payload() | {"native_lifetime": sample_resources()})
        write_json(args.report_dir / (mode + "_resource_samples.json"), samples)
        emit({"status": "COMPLETE", "mode": mode, **{k: v for k, v in result.items() if k not in ("summary",)}})
        return 0
    except (MLCancelled, PropensityCancelled):
        emit({"status": "CANCELLED", "mode": mode, "seconds": monotonic() - started})
        return 2
    except (MLDataError, PropensityDataError, OSError, sqlite3.DatabaseError, ValueError, KeyError, TypeError) as exc:
        # SQLite/OS exception text can contain keys/paths; emit only static categories.
        reason = str(exc) if type(exc) in (MLDataError, PropensityDataError) else type(exc).__name__
        emit({"status": "FAILED", "mode": mode, "reason": reason, "seconds": monotonic() - started})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
