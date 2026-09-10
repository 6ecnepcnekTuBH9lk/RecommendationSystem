"""Explicit live export command; validate/prepare are offline and never train."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.mindbox.training_batch import (
        TrainingBatchWindow, create_training_batch, load_training_batch, prepare_training_data_from_batch,
    )
    from Application.mindbox import MindboxClient, MindboxConfig, MindboxError
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError
    from Application.mindbox.adapters import AdapterError
    from Application.mindbox.identity import CustomerIdentityError
    from Application.interactions import InteractionBuildError
    from Application.product_resolution import DEFAULT_CATALOG_PATH, CatalogError, ProductResolutionError

    parser = argparse.ArgumentParser(description="Coordinated Mindbox batch; no model training")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="LIVE API: export merges/actions/orders sequentially")
    for flag in ("since", "until", "merge-since"):
        export.add_argument("--" + flag, required=True, help="UTC YYYY-MM-DD HH:MM")
    export.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    export.add_argument("--timeout", type=float, default=600)
    export.add_argument("--poll-interval", type=float, default=5)
    validate = commands.add_parser("validate", help="Offline manifest and part-reference validation")
    prepare = commands.add_parser("prepare", help="Offline batch -> prepared input; no training")
    for command in (export, validate, prepare):
        command.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    for command in (validate, prepare):
        command.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    prepare.add_argument("--diagnose", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            window = TrainingBatchWindow(*[datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                                           for value in (args.since, args.until, args.merge_since)])
            config = MindboxConfig.from_env(args.env_file)
            with MindboxClient(config) as client:
                batch = create_training_batch(client, raw_root=args.raw_root, window=window,
                                               timeout=args.timeout, poll_interval=args.poll_interval)
            print(f"Batch ID: {batch.batch_id}")
            print(f"Manifest: {args.raw_root / 'training_batches' / batch.batch_id / 'manifest.json'}")
            print(f"Exports: {len(batch.exports)}; Parts: {sum(entry.parts_count for entry in batch.exports)}")
            return 0
        batch = load_training_batch(args.manifest, raw_root=args.raw_root)
        print("Batch validation: OK")
        if args.command == "validate":
            print(f"Exports: {len(batch.exports)}; Parts: {sum(entry.parts_count for entry in batch.exports)}")
            return 0
        from Application.model.BPRMF import TrainConfig
        result = prepare_training_data_from_batch(batch, raw_root=args.raw_root, catalog_path=args.catalog,
                                                  train_config=TrainConfig(), diagnose=args.diagnose)
        stats = result.diagnostics.bpr
        print(f"Users: {stats.unique_users}; Items: {stats.unique_items}")
        print(f"BPR events: {stats.events_total}; Train pairs: {stats.train_pairs_after_aggregation}")
        print(f"Eval events: {stats.eval_events}; Train weight: {stats.total_train_weight:.12g}")
        print(f"Malformed actions: {result.diagnostics.malformed_actions}")
        print(f"Unresolved: {result.diagnostics.resolution.total.unresolved}")
        print(f"Training data complete: {result.complete}")
        return 0 if result.complete else 1
    except (MindboxError, RawExportError, AdapterError, CustomerIdentityError, InteractionBuildError,
            CatalogError, ProductResolutionError, ValueError, OSError, KeyError) as exc:
        print(f"Batch command failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
