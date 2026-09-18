"""Monthly canonical Customers export; offline canonical/legacy profile inspection."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.mindbox.customer_profile_snapshot import (
        load_customer_profile_snapshot, load_customer_contact_index,
    )
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="LIVE: sequential monthly Customers upsert")
    validate = commands.add_parser("validate", help="Offline metadata validation; no profile reads")
    inspect = commands.add_parser("inspect", help="Offline safe profile counts")
    for cmd in (export, validate, inspect):
        cmd.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    export.add_argument("--since", required=True)
    export.add_argument("--until", required=True)
    export.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    export.add_argument("--timeout", type=float, default=14400.)
    export.add_argument("--poll-interval", type=float, default=5.)
    for cmd in (validate, inspect):
        cmd.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--model-dir", help="Optional model artifacts for aligned coverage")
    inspect.add_argument("--progress-every", type=int, default=100000, help="Safe record progress interval; 0 disables")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            from scripts.mindbox_canonical import main as canonical_main
            return canonical_main(["customers", "--since", args.since, "--until", args.until,
                "--raw-root", str(args.raw_root), "--env-file", str(args.env_file),
                "--timeout", str(args.timeout), "--poll-interval", str(args.poll_interval)])
        elif args.command == "validate":
            load_customer_profile_snapshot(args.manifest, raw_root=args.raw_root)
            print("Snapshot validation: OK")
        else:
            mappings = None
            if args.model_dir is not None:
                from Application.model.BPRMF import _load_artifacts
                mappings, _ = _load_artifacts(args.model_dir)
            if args.progress_every < 0:
                parser.error("--progress-every must be non-negative")
            index = load_customer_contact_index(args.manifest, mappings, raw_root=args.raw_root,
                progress=lambda count: print(f"Customers processed: {count}", flush=True), progress_every=args.progress_every)
            print("Coverage scope: model users" if mappings is not None else "Coverage scope: snapshot canonical profiles")
            for name, count in asdict(index.diagnostics).items():
                print(f"{name}: {count}")
        return 0
    except KeyboardInterrupt:
        print("Customer profile operation cancelled", file=sys.stderr, flush=True)
        return 130
    except Exception:
        print("Customer profile operation failed; invalid source, transport or metadata", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
