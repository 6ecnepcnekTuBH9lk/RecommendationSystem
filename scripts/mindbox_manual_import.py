"""Offline manual JSON transport; no Mindbox client or credentials."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Application.loading_errors import emit_error
    from Application.mindbox.manual_import import import_interactions, ManualImportError
    from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
    from Application.mindbox.selection import MindboxSelectionConfig, SELECTION_OPTIONS
    from Application.mindbox.training_batch import TrainingBatchWindow

    parser = argparse.ArgumentParser(description="Offline manual Mindbox JSON import")
    commands = parser.add_subparsers(dest="command", required=True)
    interactions = commands.add_parser("interactions")
    customers = commands.add_parser("customers")
    for sub in (interactions, customers):
        sub.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    for name in ("actions", "orders"):
        interactions.add_argument("--" + name, required=True, action="append", type=Path)
    for name in ("since", "until"):
        interactions.add_argument("--" + name, required=True)
    for field, option in SELECTION_OPTIONS.items():
        interactions.add_argument(option, dest=field, action="append")
    customers.add_argument(
        "--customers",
        action="append",
        required=True,
        type=Path,
    )
    args = parser.parse_args(argv)
    try:
        progress = lambda text: print(text, flush=True)
        if args.command == "interactions":
            def date(value):
                return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            window = TrainingBatchWindow(date(args.since), date(args.until), date(args.since))
            selection = MindboxSelectionConfig(**{field: getattr(args, field) for field in SELECTION_OPTIONS
                                                 if getattr(args, field) is not None})
            path = import_interactions(args.actions, args.orders, raw_root=args.raw_root, window=window,
                                       selection=selection, progress=progress)
        else:
            from Application.mindbox.canonical_customers import import_full
            path = import_full(args.raw_root, args.customers, progress=progress)
        print(f"Manifest: {path}", flush=True)
        return 0
    except ManualImportError as exc:
        emit_error(exc, source="customers" if args.command == "customers" else "actions/orders")
        return 1
    except (KeyboardInterrupt, InterruptedError):
        return 130
    except Exception as exc:
        emit_error(exc, source="customers" if args.command == "customers" else "actions/orders")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
