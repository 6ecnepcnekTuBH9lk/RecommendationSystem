"""Customers snapshots: explicit live export, offline metadata validation/inspection."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.mindbox.customer_profile_snapshot import (
        create_customer_profile_snapshot, load_customer_profile_snapshot, load_customer_contact_index, snapshot_manifest_path,
    )
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="LIVE: export Customers once; reuse training merges")
    validate = commands.add_parser("validate", help="Offline metadata validation; no profile reads")
    inspect = commands.add_parser("inspect", help="Offline safe profile counts")
    for cmd in (export, validate, inspect):
        cmd.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    export.add_argument("--training-manifest", type=Path, required=True)
    export.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    export.add_argument("--timeout", type=float, default=3600.)
    export.add_argument("--poll-interval", type=float, default=5.)
    for cmd in (validate, inspect):
        cmd.add_argument("--manifest", type=Path, required=True)
    inspect.add_argument("--model-dir", help="Optional model artifacts for aligned coverage")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            from Application.mindbox import MindboxConfig, MindboxClient
            with MindboxClient(MindboxConfig.from_env(args.env_file)) as client:
                snapshot = create_customer_profile_snapshot(client, training_manifest=args.training_manifest,
                    raw_root=args.raw_root, timeout=args.timeout, poll_interval=args.poll_interval)
            print(f"Snapshot ID: {snapshot.snapshot_id}")
            print(f"Manifest: {snapshot_manifest_path(args.raw_root, snapshot.snapshot_id)}")
            print(f"Customers parts: {snapshot.customers_parts}")
            print(f"Training batch reference: {snapshot.originating_training_batch_id}")
        elif args.command == "validate":
            load_customer_profile_snapshot(args.manifest, raw_root=args.raw_root)
            print("Snapshot validation: OK")
        else:
            mappings = None
            if args.model_dir is not None:
                from Application.model.BPRMF import _load_artifacts
                mappings, _ = _load_artifacts(args.model_dir)
            index = load_customer_contact_index(args.manifest, mappings, raw_root=args.raw_root)
            print("Coverage scope: model users" if mappings is not None else "Coverage scope: snapshot canonical profiles")
            for name, count in asdict(index.diagnostics).items():
                print(f"{name}: {count}")
        return 0
    except Exception:
        print("Customer profile operation failed; invalid source, transport or metadata", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
