"""Explicit offline Mindbox production preflight/publish commands."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _show(result, stream):
    def emit(message):
        print(message, file=stream, flush=True)
    if result.batch_id:
        emit(f"Batch: {result.batch_id}")
        emit(f"Window: {result.interaction_window['since']} -> {result.interaction_window['until']}")
    if result.quality_report:
        emit(f"Quality: {result.quality_report.level.value}")
        emit(f"Training allowed: {'YES' if result.quality_report.training_allowed else 'NO'}")
        for label, key in (("Users", "users"), ("Items", "items"), ("Train pairs", "train_pairs"),
            ("Eval events", "eval_events"), ("BPR events", "bpr_events"), ("Complete", "complete"),
            ("Malformed mapped actions", "malformed_mapped_actions"), ("Unresolved products", "unresolved_products"),
            ("Unsupported products", "unsupported_products")):
            emit(f"{label}: {result.dataset[key]}")


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from Application.model.BPRMF import TrainConfig
    from Application.model.mindbox_production_training import preflight_production_training, train_and_publish_production_model

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "publish"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "ВходныеДанные" / "Номенклатура.csv")
        sub.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "ВходныеДанные" / "MindboxRaw")
        sub.add_argument("--device", choices=("cpu",), default="cpu")
        if command == "publish":
            sub.add_argument("--allow-warn", action="store_true")
    args = parser.parse_args(argv)
    terminal = sys.stdout
    try:
        cfg = TrainConfig(data_dir=str(args.catalog.resolve().parent))
        kwargs = dict(raw_root=args.raw_root, catalog_path=args.catalog, cfg=cfg, device=torch.device(args.device))
        if args.command == "preflight":
            result = preflight_production_training(args.manifest, **kwargs)
            _show(result, terminal)
        else:
            result = train_and_publish_production_model(args.manifest, **kwargs, allow_warn=args.allow_warn,
                progress=lambda value: _show(value, terminal))
            print(f"Production model published: {'YES' if result.published else 'NO'}")
            print(f"Generation: {result.published_generation}")
            print(f"Disk validation: {result.prepublish_disk_validation}; Post-publish validation: {result.postpublish_validation}")
            print(f"Rolled back: {result.rolled_back}; Cleanup failed: {result.cleanup_failed}")
            print(f"Audit report: {'OK' if result.report_path else 'FAILED'}")
            if result.report_path:
                print(f"Report: production_training/{Path(result.report_path).parent.name}/report.json")
        if result.error_code:
            print(f"Error code: {result.error_code}", file=sys.stderr)
        if result.cancelled:
            return 130
        return 1 if result.error_code else 0
    except KeyboardInterrupt:
        print("Error code: CANCELLED", file=sys.stderr)
        return 130
    except Exception:
        print("Error code: PREPARATION_FAILED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
