"""Explicit offline shadow training; never publish model artifacts."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from Application.model.BPRMF import TrainConfig
    from Application.model.mindbox_shadow_training import shadow_train_daily_manifest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "ВходныеДанные" / "Номенклатура.csv")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "ВходныеДанные" / "MindboxRaw")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args(argv)
    cfg = TrainConfig(data_dir=str(args.catalog.resolve().parent))
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        cfg.epochs = args.epochs

    def show_quality(report):
        print(f"Quality: {report.level.value}")
        print(f"Training allowed: {'YES' if report.training_allowed else 'NO'}")

    try:
        result = shadow_train_daily_manifest(args.manifest, raw_root=args.raw_root, catalog_path=args.catalog,
                                            cfg=cfg, device=torch.device(args.device), on_quality=show_quality)
        print(f"Training data complete: {result.preparation_summary['complete']}")
        print(f"Shadow training: {'COMPLETED' if result.training_completed else result.error_code}")
        if result.training_metrics:
            m = result.training_metrics
            print(f"Epochs completed: {m.epochs_completed}; Best epoch: {m.best_epoch}; Metric: {m.best_metric_name}")
            print(f"Best Recall@{cfg.topk}: {m.best_recall:.6g}; Best NDCG@{cfg.topk}: {m.best_ndcg:.6g}")
            print(f"Early stopped: {m.early_stopped}")
        print(f"Report: {result.report_path}")
        print("Production model published: NO")
        return 0 if result.training_completed else 1
    except Exception:
        # External preparation/training exceptions can contain raw identities.
        print("Shadow training: FAILED; preparation, training or report error", file=sys.stderr)
        print("Production model published: NO")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
