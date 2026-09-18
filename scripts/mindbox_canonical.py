"""Canonical API jobs. Technical paths stay in machine-readable output only."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.mindbox import MindboxClient, MindboxConfig
    from Application.mindbox.canonical_jobs import create_job, resume_job, load_job
    from Application.mindbox.selection import MindboxSelectionConfig, SELECTION_OPTIONS
    from Application.mindbox.storage import DEFAULT_RAW_ROOT
    from Application.mindbox.training_batch import TrainingBatchWindow
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export-daily", "customers", "resume", "status"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--merge-since")
    parser.add_argument("--timeout", type=float, default=14400.)
    parser.add_argument("--poll-interval", type=float, default=5.)
    for field, option in SELECTION_OPTIONS.items():
        parser.add_argument(option, dest=field, action="append")
    args = parser.parse_args(argv)
    active = {}
    def progress(name, ready, total):
        if ready == -1:
            active.update(source=name, since=total)
            return
        print("Event: " + json.dumps({"source": name, "ready": ready, "total": total}), flush=True)
    try:
        if args.command == "status":
            job = load_job(args.state, args.raw_root)
            for name in ("customer_merges", "actions", "orders", "customers"):
                items = [c for c in job["components"] if c["name"] == name]
                if items:
                    progress(name, sum(c["done"] for c in items), len(items))
            return 0
        selection = MindboxSelectionConfig(**{field: getattr(args, field) for field in SELECTION_OPTIONS if getattr(args, field) is not None})
        kwargs = dict(raw_root=args.raw_root, timeout=args.timeout, poll_interval=args.poll_interval, progress=progress)
        with MindboxClient(MindboxConfig.from_env(args.env_file)) as client:
            if args.command == "resume":
                if args.state is None:
                    raise ValueError("State required")
                path = resume_job(client, state_path=args.state, **kwargs)
            else:
                since, until = (datetime.fromisoformat(value).replace(tzinfo=timezone.utc) for value in (args.since, args.until))
                if args.command == "customers":
                    kwargs.update(customers=True, since=since, until=until)
                else:
                    kwargs.update(window=TrainingBatchWindow(since, until, datetime.fromisoformat(args.merge_since).replace(tzinfo=timezone.utc)), selection=selection)
                path = create_job(client, on_state_created=lambda p: print(f"State: {p}", flush=True), **kwargs)
        print(f"Manifest: {path}", flush=True)
        return 0
    except (KeyboardInterrupt, InterruptedError):
        return 130
    except Exception as exc:
        from Application.mindbox.exceptions import MindboxExportTimeoutError
        category = "timeout" if isinstance(exc, (TimeoutError, MindboxExportTimeoutError)) else "failed"
        print("Error: " + json.dumps({"category": category, **active}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
