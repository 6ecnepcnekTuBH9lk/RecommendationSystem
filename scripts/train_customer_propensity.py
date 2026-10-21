"""Opt-in independent order-placement SGD research; never builds data or invokes GUI/BPR."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
from threading import Event

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Set before numerical imports; this CLI owns its process, not the application's environment.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np

from Application.model.propensity_ml.artifacts import load_artifact
from Application.model.propensity_ml.contracts import FEATURES, MLCancelled, MLDataError, dataset_manifest, load_json, temporal_roles, write_json
from Application.model.propensity_ml.resources import ResourceGuard, sample_resources
from Application.model.propensity_ml.training import TrainingConfig, dependencies, run_experiment


def emit(value):
    print(json.dumps(value, ensure_ascii=True, allow_nan=False), flush=True)


def synthetic_rows(count=128):
    rows = []
    for j in range(count):
        values = {n: 0 for n in FEATURES}
        for name in ("days_since_last_order_365d", "days_since_view_180d", "days_since_favorite_90d",
                     "average_days_between_orders_365d"):
            values[name] = None
        if j % 2:
            values["views_30d"] = 2
            values["views_90d"] = values["views_180d"] = 2
            values["views_active_days_30d"] = 1
            values["has_view_180d"] = 1
            values["days_since_view_180d"] = 15
        rows.append({"features": values, "available": {n: True for n in FEATURES}, "feature_issues": {}})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Explicitly authorize one experiment in this process")
    parser.add_argument("--allow-retrospective", action="store_true")
    parser.add_argument("--allow-test", action="store_true", help="Evaluate final frozen pipeline once")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--models-root", type=Path, default=Path("filtered_data/customer_propensity_models"))
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validate-artifact", type=Path, help="New-process synthetic inference only; no real customer rows")
    args = parser.parse_args(argv)
    if args.validate_artifact:
        if args.train:
            parser.error("Artifact validation cannot start training")
        model = load_artifact(args.validate_artifact)
        rows = synthetic_rows()
        a = model.predict(rows)
        b = np.concatenate([model.predict(rows[:31]), model.predict(rows[31:])])
        if not np.array_equal(a, b) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
            raise MLDataError("SYNTHETIC_INFERENCE_FAILED")
        emit({"status": "SYNTHETIC_ARTIFACT_VALIDATION_PASSED", "rows": len(a), "version": model.metadata["version"],
              "batch_equal": True, "probability_min": float(a.min()), "probability_max": float(a.max())})
        return 0
    if args.dataset is None:
        parser.error("--dataset is required; no dataset or HMAC key is generated automatically")
    config = TrainingConfig(batch_size=args.batch_size, epochs=args.epochs)
    manifest = dataset_manifest(args.dataset)
    roles = temporal_roles(manifest)
    parent = args.dataset.parent
    for pointer_name in ("CURRENT.json", "CURRENT_INFERENCE.json"):
        pointer = load_json(parent / pointer_name)
        purpose = "inference" if "INFERENCE" in pointer_name else "training"
        other = dataset_manifest(parent / pointer["version"], purpose=purpose)
        if other["generation"] != manifest["generation"] or other["source_version"] != manifest["source_version"]:
            raise MLDataError("SOURCE_PAIR_MISMATCH")
        from Application.model.propensity_ml.contracts import checksum
        if checksum(parent / pointer["version"] / "manifest.json") != pointer["manifest_sha256"]:
            raise MLDataError("POINTER_MANIFEST_CHECKSUM_MISMATCH")
    emit({"status": "PREFLIGHT", "dataset_version": manifest["version"], "roles": roles,
          "dependencies": dependencies(), "resources": sample_resources(), "configuration": asdict(config)})
    if not args.train:
        return 0
    if not args.allow_retrospective or args.run_dir is None:
        parser.error("Training requires --allow-retrospective and a new --run-dir")
    args.models_root.mkdir(exist_ok=True, parents=True)
    args.run_dir.parent.mkdir(exist_ok=True, parents=True)
    if args.run_dir.exists():
        parser.error("Run directory already exists; automatic retries/reuse are forbidden")
    stop = Event()
    previous = signal.signal(signal.SIGINT, lambda *unused: stop.set())
    signal_break = getattr(signal, "SIGBREAK", None)
    previous_break = signal.signal(signal_break, lambda *unused: stop.set()) if signal_break else None
    last_sample = -float("inf")

    def observed(value):
        nonlocal last_sample
        if value["elapsed_seconds"] - last_sample >= 5:
            with (args.run_dir.parent / "resources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")
            last_sample = value["elapsed_seconds"]

    guard = ResourceGuard((args.run_dir.parent, args.models_root),
                          lambda: stop.is_set() or (args.cancel_file is not None and args.cancel_file.exists()), observer=observed)
    try:
        # Future cache growth for train+validation+test and metadata; no raw JSON copy.
        guard.poll(force=True, extra_disk_bytes=manifest["counts"]["rows"] * 194 + 16 * 1024**2)
        result = run_experiment(args.dataset, args.run_dir, args.models_root, guard, config=config,
                                allow_test=args.allow_test, progress=emit)
        emit(result)
        return 0
    except MLCancelled:
        result = {"status": "TRAINING PARTIAL", "reason": "MANUAL_CANCELLATION", "resources": guard.payload()}
        if args.run_dir.exists():
            write_json(args.run_dir / "stopped.json", result)
        emit(result)
        return 2
    except Exception as exc:
        # No raw source values / rows / customer identifiers in errors or diagnostics.
        result = {"status": "TRAINING BLOCKED", "reason": str(exc) if isinstance(exc, MLDataError) else type(exc).__name__,
                  "resources": guard.payload()}
        if args.run_dir.exists():
            write_json(args.run_dir / "stopped.json", result)
        emit(result)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
        if signal_break:
            signal.signal(signal_break, previous_break)


if __name__ == "__main__":
    sys.exit(main())
