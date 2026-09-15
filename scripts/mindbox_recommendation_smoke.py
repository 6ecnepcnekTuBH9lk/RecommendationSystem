"""Offline temporary training, checkpoint reload and recommendation acceptance."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from Application.model.BPRMF import TrainConfig
    from Application.model.mindbox_recommendation_smoke import run_recommendation_smoke

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--profile-manifest", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "ВходныеДанные" / "Номенклатура.csv")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "ВходныеДанные" / "MindboxRaw")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-export-users", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.max_export_users < 1:
        parser.error("--epochs and --max-export-users must be positive")
    # Capture the terminal stream before orchestration suppresses legacy logging.
    terminal = sys.stdout
    try:
        report, path = run_recommendation_smoke(args.training_manifest, args.profile_manifest,
            raw_root=args.raw_root, catalog_path=args.catalog,
            cfg=TrainConfig(data_dir=str(args.catalog.resolve().parent), epochs=args.epochs),
            device=torch.device(args.device), max_export_users=args.max_export_users,
            settings_source=PROJECT_ROOT / "Настройки",
            progress=lambda message: print(message, file=terminal, flush=True))
        print(f"Smoke: {report['status']}; error_code={report['error_code']}")
        for section in ("quality", "dataset", "training", "artifact", "contacts", "export", "cleanup"):
            print(f"{section}: {report.get(section, {})}")
        print(f"Report: recommendation_smoke/{path.parent.name}/report.json")
        return 130 if report["status"] == "CANCELLED" else (0 if report["status"] == "COMPLETED" else 1)
    except KeyboardInterrupt:
        print("Smoke: CANCELLED", file=sys.stderr)
        return 130
    except Exception:
        print("Smoke: FAILED; error_code=REPORT_OR_SETUP_FAILED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
