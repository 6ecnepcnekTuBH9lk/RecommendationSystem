"""Dedicated QProcess single-run entry point. No child training process or sweep."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from dataclasses import asdict
    import torch
    from Application.evaluation.audit import load_canonical_events
    from Application.evaluation.temporal import build_temporal_protocol
    from Application.evaluation.experiments.bpr_weights import benchmark_config, git_provenance
    from Application.evaluation.experiments.gui_history import clean_record, now
    from Application.evaluation.experiments.gui_run import gui_config, run_validation, BenchmarkUnavailable
    from Application.mindbox.canonical_storage import atomic_json
    from Application.paths import USER_SETTINGS_DIR
    from scripts.mindbox_production_train import EVENT_PREFIX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cancel-file", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    args = parser.parse_args(argv)
    record = clean_record({"run_id": args.run_id, "status": "running", "started_at": now(), "pid": os.getpid()})
    artifact = USER_SETTINGS_DIR / "research_experiments" / record["artifact"]
    terminal = sys.stdout
    start = time.monotonic()

    def emit(event):
        print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), file=terminal, flush=True)

    def check_cancel():
        if args.cancel_file.exists():
            raise KeyboardInterrupt

    try:
        cfg = gui_config(json.loads(args.config.read_text(encoding="utf-8")))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        record.update(hyperparameters=asdict(cfg), epochs_requested=cfg.epochs, epochs_completed=0,
                      device=str(device), torch_version=str(torch.__version__), cuda_build=torch.version.cuda,
                      provenance=git_provenance(PROJECT_ROOT))
        atomic_json(artifact, clean_record(record))
        emit({"stage": "research_started", "result": clean_record(record)})
        check_cancel()
        emit({"stage": "loading", "device": str(device)})
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            events, _ = load_canonical_events(args.raw_root, args.catalog, progress=lambda _: check_cancel())
            check_cancel()
            temporal = benchmark_config()
            protocol = build_temporal_protocol(events, temporal)
            snapshot = protocol.validation
            del protocol, events
        check_cancel()

        def progress(event):
            if event["stage"] == "epoch":
                record["epochs_completed"] = event["epoch"]
            if event["stage"] == "training":
                record.update({k: event[k] for k in ("training_users", "training_items", "training_pairs")})
            emit(event)

        record.update(run_validation(snapshot, temporal, cfg, device, progress, check_cancel))
        record["status"] = "completed"
    except KeyboardInterrupt:
        record.update(status="cancelled", error_summary="CANCELLED")
    except BenchmarkUnavailable:
        record.update(status="failed", error_summary="BENCHMARK_UNAVAILABLE")
    except Exception:
        record.update(status="failed", error_summary="RUN_FAILED")
    record.update(finished_at=now(), total_seconds=time.monotonic() - start)
    record = clean_record(record)
    try:
        atomic_json(artifact, record)
    except OSError:
        record.update(status="failed", error_summary="RUN_FAILED")
    emit({"stage": "research_finished", "result": record})
    return 0 if record["status"] == "completed" else 130 if record["status"] == "cancelled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
