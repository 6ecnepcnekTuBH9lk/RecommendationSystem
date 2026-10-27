"""Explicit offline Mindbox production preflight/publish commands."""

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_PREFIX = "TRAINING_EVENT "


def _event(result, stage, stream):
    quality = result.quality_report
    payload = {
        "stage": stage, "batch_id": result.batch_id,
        "interaction_window": dict(result.interaction_window), "dataset": dict(result.dataset),
        "quality": None if quality is None else {
            "level": quality.level.value, "training_allowed": quality.training_allowed,
            "metrics": dict(quality.metrics), "issues": [
                {"code": issue.code, "level": issue.level.value, "count": issue.count,
                 "message": issue.message, "rate": issue.rate, "breakdown": dict(issue.breakdown)}
                for issue in quality.issues]},
        **{name: getattr(result, name) for name in (
            "error_code", "cancelled", "training_started", "training_completed", "publication_started", "published",
            "previous_generation", "published_generation", "prepublish_disk_validation",
            "postpublish_validation", "rolled_back", "cleanup_failed")},
        "training_metrics": None if result.training_metrics is None else {
            name: getattr(result.training_metrics, name) for name in (
                "epochs_completed", "best_epoch", "best_recall", "best_ndcg", "best_metric_name", "early_stopped")},
    }
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), file=stream, flush=True)


def _gui_training(args, kwargs, stream):
    from Application.model.mindbox_production_training import (
        preflight_production_training, train_and_publish_production_model,
    )
    started = time.monotonic()
    managed = bool(args.cancel_file)

    def emit(payload):
        print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), file=stream, flush=True)

    def check_cancel():
        if args.cancel_file and args.cancel_file.exists():
            raise KeyboardInterrupt

    def observer(model, epoch):
        emit({"stage": "epoch", "epoch": epoch.epoch, "epochs": kwargs["cfg"].epochs,
              "loss": epoch.loss, "device": str(kwargs["device"])})
        check_cancel()

    def before_publication():
        check_cancel()
        if args.cancel_file:
            # GUI stops accepting cancellation BEFORE granting publication.
            # No artifact/current write starts until this explicit handshake.
            emit({"stage": "publication_ready"})
            if sys.stdin.readline().strip() != "PUBLISH":
                raise KeyboardInterrupt
        check_cancel()

    def progress(value):
        check_cancel()
        emit({"stage": "publication" if value.publication_started else "training" if value.training_started else "preparation",
              "device": str(kwargs["device"]), "dataset": dict(value.dataset)})

    check_cancel()
    if managed:
        emit({"stage": "loading", "device": str(kwargs["device"])})
    result = preflight_production_training(args.manifest, **kwargs)
    check_cancel()
    _event(result, "preflight", stream)
    if result.error_code or result.quality_report is None:
        return 130 if result.cancelled else 1
    allow_warn = False
    if result.quality_report.level.value == "WARN":
        # stdin stays open in QProcess; EOF/rejection never grants permission.
        allow_warn = sys.stdin.readline().strip() == "YES"
        if not allow_warn:
            print("Обучение отменено до запуска тренера.", file=stream, flush=True)
            return 130
    print("Обучение BPR-MF и безопасная публикация модели...", file=stream, flush=True)
    result = train_and_publish_production_model(
        args.manifest, **kwargs, allow_warn=allow_warn, expected_batch_id=result.batch_id,
        **({"epoch_observer": observer, "check_cancel": check_cancel,
            "before_publication": before_publication, "progress": progress} if managed else {}),
    )
    if managed:
        emit({"stage": "timing", "device": str(kwargs["device"]), "total_seconds": time.monotonic() - started})
    _event(result, "finished", stream)
    if result.report_path:
        print(f"Report: production_training/{Path(result.report_path).parent.name}/report.json",
              file=stream, flush=True)
    return 130 if result.cancelled else 1 if result.error_code or not result.published else 0


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
    from Application.model.BPRMF import TrainConfig, _load_train_config_from_json
    from Application.model.mindbox_production_training import preflight_production_training, train_and_publish_production_model

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "publish", "gui"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "input_data" / "nomenclature.csv")
        sub.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "input_data" / "MindboxRaw")
        sub.add_argument("--device", choices=("cpu", "auto"), default="cpu")
        sub.add_argument("--config", type=Path)
        if command == "gui":
            sub.add_argument("--cancel-file", type=Path)
        if command == "publish":
            sub.add_argument("--allow-warn", action="store_true")
    args = parser.parse_args(argv)
    terminal = sys.stdout
    try:
        cfg = (_load_train_config_from_json(str(args.config)) if args.config
               else TrainConfig(data_dir=str(args.catalog.resolve().parent)))
        if Path(cfg.data_dir).resolve() != args.catalog.resolve().parent:
            raise ValueError("Config and product catalog must use the same directory")
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        kwargs = dict(raw_root=args.raw_root, catalog_path=args.catalog, cfg=cfg, device=torch.device(device))
        if args.command == "gui":
            return _gui_training(args, kwargs, terminal)
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
        if args.command == "gui":
            print(EVENT_PREFIX + json.dumps({"stage": "preflight", "error_code": "PREPARATION_FAILED", "quality": None}),
                  file=terminal, flush=True)
        print("Error code: PREPARATION_FAILED", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
