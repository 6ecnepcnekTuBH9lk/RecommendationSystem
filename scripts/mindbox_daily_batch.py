"""Daily batch CLI. Only export-daily/resume contact Mindbox; never train."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def print_status(batch, state_path):
    print(f"Batch: {batch.batch_id}")
    print(f"State: {state_path}")
    print(f"Window: {batch.window.interaction_since.isoformat()} -> {batch.window.interaction_until.isoformat()}")
    counts = batch.diagnostics
    print(f"Days: {counts['days_ready']} / {counts['days_total']} complete")
    print(f"Merges: {batch.components[0].status}")
    print(f"Completed components: {counts['components_ready']}")
    print(f"Failed components: {counts['components_failed']}")
    ready = [c for c in batch.components if c.status == "READY"]
    if ready:
        last = ready[-1]
        print(f"Last completed: {last.since.date()} {last.name}")
    else:
        print("Last completed: none")
    for index in range(1, len(batch.components), 2):
        actions, orders = batch.components[index:index + 2]
        if actions.status != "READY" or orders.status != "READY":
            print(f"Current: {actions.since.date()}; Actions: {actions.status}; Orders: {orders.status}")
            break
    print(f"Transport complete: {batch.transport_complete}")


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.mindbox import MindboxConfig, MindboxClient, MindboxError
    from Application.mindbox.daily_training_batch import (
        ChunkedBatchError, create_chunked_training_batch, resume_chunked_training_batch,
        load_chunked_training_batch, prepare_training_data_from_chunked_batch,
    )
    from Application.mindbox.training_batch import TrainingBatchWindow
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError
    from Application.mindbox.adapters import AdapterError
    from Application.mindbox.identity import CustomerIdentityError
    from Application.interactions import InteractionBuildError
    from Application.product_resolution import DEFAULT_CATALOG_PATH, CatalogError, ProductResolutionError

    parser = argparse.ArgumentParser(description="Daily resumable local training batches; no training")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-daily", help="LIVE: start daily exports")
    resume = commands.add_parser("resume", help="LIVE: continue unfinished components")
    status = commands.add_parser("status", help="Offline state inspection")
    validate = commands.add_parser("validate", help="Offline final manifest validation")
    prepare = commands.add_parser("prepare", help="Offline preparation from final manifest")
    for command in (export, resume, status, validate, prepare):
        command.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    for command in (export, resume):
        command.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
        command.add_argument("--timeout", type=float, default=3600)
        command.add_argument("--poll-interval", type=float, default=5)
    for name in ("since", "until", "merge-since"):
        export.add_argument("--" + name, required=True)
    for command in (resume, status):
        command.add_argument("--state", type=Path, required=True)
    for command in (validate, prepare):
        command.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    prepare.add_argument("--diagnose", action="store_true")
    args = parser.parse_args(argv)
    state = getattr(args, "state", None)
    try:
        if args.command in ("export-daily", "resume"):
            if args.command == "export-daily":
                def date(value, format):
                    return datetime.strptime(value, format).replace(tzinfo=timezone.utc)
                window = TrainingBatchWindow(date(args.since, "%Y-%m-%d"), date(args.until, "%Y-%m-%d"),
                                             date(args.merge_since, "%Y-%m-%d %H:%M"))
            config = MindboxConfig.from_env(args.env_file)
            with MindboxClient(config) as client:
                if args.command == "export-daily":
                    batch = create_chunked_training_batch(client, raw_root=args.raw_root, window=window,
                                                           timeout=args.timeout, poll_interval=args.poll_interval)
                else:
                    batch = resume_chunked_training_batch(client, state_path=state, raw_root=args.raw_root,
                                                           timeout=args.timeout, poll_interval=args.poll_interval)
            state = args.raw_root / "training_batches" / batch.batch_id / "state.json"
            print_status(batch, state)
            print(f"Manifest: {state.parent / 'manifest.json'}")
            return 0
        if args.command == "status":
            batch = load_chunked_training_batch(state, raw_root=args.raw_root)
            manifest = state.parent / "manifest.json"
            if manifest.exists():
                batch = load_chunked_training_batch(manifest, raw_root=args.raw_root, require_complete=True)
            print_status(batch, state)
            return 0
        batch = load_chunked_training_batch(args.manifest, raw_root=args.raw_root, require_complete=True)
        print("Transport complete: True; Validation: OK")
        if args.command == "validate":
            return 0
        from Application.model.BPRMF import TrainConfig
        from Application.model.training_quality import TrainingQualityDiagnostics, evaluate_training_quality
        result = prepare_training_data_from_chunked_batch(batch, raw_root=args.raw_root,
                    catalog_path=args.catalog, train_config=TrainConfig(), diagnose=args.diagnose)
        d = result.diagnostics
        for label, count in (("Actions", d.actions), ("Orders", d.orders), ("Order lines", d.order_lines),
                            ("VIEW", d.view_interactions), ("FAVORITE", d.favorite_interactions),
                            ("PURCHASE", d.purchase_interactions), ("Malformed", d.malformed_actions),
                            ("Unmapped", d.unmapped_actions), ("Resolved", d.resolution.total.resolved),
                            ("Unresolved", d.resolution.total.unresolved),
                            ("Unsupported", d.resolution.total.unsupported_namespace)):
            print(f"{label}: {count}")
        for name, count in sorted(d.malformed_action_system_names.items()):
            print(f"  {name}: {count}")
        for name, value in vars(d.bpr).items():
            formatted = f"{value:.12g}" if isinstance(value, float) else str(value)
            print(f"{name}: {formatted}")
        print(f"Training data complete: {result.complete}")
        quality = evaluate_training_quality(result.prepared_data, TrainingQualityDiagnostics(
            actions_view=d.actions_view, actions_favorite=d.actions_favorite,
            malformed_mapped_actions=d.malformed_actions, unresolved_products=d.resolution.total.unresolved,
            unsupported_products=d.resolution.total.unsupported_namespace, bpr_events=d.bpr.events_total,
            unmapped_actions=d.unmapped_actions, malformed_action_system_names=d.malformed_action_system_names))
        print(f"Training quality: {quality.level.value}")
        print(f"Training allowed: {quality.training_allowed}")
        print(f"Mapped actions: {quality.metrics['mapped_actions']}")
        print(f"Malformed mapped actions: {quality.metrics['malformed_mapped_actions']}")
        print(f"Malformed rate: {quality.metrics['malformed_rate']:.4%}")
        print("Issues:")
        for issue in quality.issues:
            print(f"{issue.level.value} {issue.code}: {issue.count}")
            for name, count in sorted(issue.breakdown.items()):
                print(f"  {name}: {count}")
        return 0 if quality.training_allowed else 1
    except (MindboxError, RawExportError, AdapterError, CustomerIdentityError, InteractionBuildError,
            CatalogError, ProductResolutionError, ValueError, OSError, KeyError) as exc:
        print(f"Daily batch failed: {type(exc).__name__}", file=sys.stderr)
        if isinstance(exc, ChunkedBatchError):
            state = exc.state_path
        if state is not None:
            print(f"Batch: {state.parent.name}; State: {state}")
            try:
                print_status(load_chunked_training_batch(state, raw_root=args.raw_root), state)
            except (ValueError, OSError, RawExportError):
                print("State unavailable or invalid; last completed component unknown.")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
