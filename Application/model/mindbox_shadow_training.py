"""Offline daily-manifest orchestration and allowlisted aggregate JSON report."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import uuid

from Application.mindbox.daily_training_batch import load_chunked_training_batch, prepare_training_data_from_chunked_batch
from .shadow_training import run_shadow_training
from .training_quality import TrainingQualityDiagnostics

REPORT_ROOT = Path(__file__).resolve().parents[2] / "ВходныеДанные" / "MindboxReports" / "shadow_training"


def _atomic_report(path, payload):
    # Serialize before touching the filesystem; reject non-finite JSON numbers.
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _safe_config(cfg):
    # Explicit numeric/boolean allowlist: data paths, feature values and arbitrary
    # attributes are never serialized. Enum-like fields use normalized values.
    names = ("w_view_item", "w_favorite", "w_purchase", "embedding_dim", "epochs", "batch_size", "lr",
             "weight_decay", "bpr_reg", "n_neg", "seed", "topk", "min_user_interactions_for_eval",
             "early_stop", "early_stop_patience", "early_stop_min_delta", "early_stop_min_epochs",
             "use_item_features", "max_item_features", "feature_dropout", "feature_scale", "feat_reg_mult")
    result = {name: getattr(cfg, name) for name in names}
    if any(type(value) not in (int, float, bool) for value in result.values()):
        raise ValueError("Invalid numeric training config")
    result["early_stop_metric"] = cfg.early_stop_metric if cfg.early_stop_metric in ("recall", "ndcg") else "ndcg"
    result["feature_norm"] = cfg.feature_norm if cfg.feature_norm in ("sum", "mean", "sqrt") else "mean"
    known_columns = {"ВидНоменклатуры", "ВидАссортимента", "Марка", "Коллекция", "СезонНоски",
                     "ПолНоменклатуры", "ГруппаСоставов", "КатегорияНаСайте", "СтилеваяГруппа"}
    result["item_feature_cols"] = [name for name in cfg.item_feature_cols if name in known_columns]
    result["other_feature_columns_count"] = len(cfg.item_feature_cols) - len(result["item_feature_cols"])
    return result


def shadow_train_daily_manifest(manifest, *, raw_root, catalog_path, cfg, device, on_quality=None):
    catalog = Path(catalog_path).resolve()
    # The existing feature loader reads this exact basename from cfg.data_dir.
    if catalog.name != "Номенклатура.csv" or catalog.parent != Path(cfg.data_dir).resolve():
        raise ValueError("Preparation and feature catalog must match TrainConfig.data_dir")
    config_summary = _safe_config(cfg)
    batch = load_chunked_training_batch(manifest, raw_root=raw_root, require_complete=True)
    prepared = prepare_training_data_from_chunked_batch(batch, raw_root=raw_root, catalog_path=catalog,
                                                       train_config=cfg, diagnose=True)
    d = prepared.diagnostics
    diagnostics = TrainingQualityDiagnostics(
        actions_view=d.actions_view, actions_favorite=d.actions_favorite, malformed_mapped_actions=d.malformed_actions,
        unresolved_products=d.resolution.total.unresolved, unsupported_products=d.resolution.total.unsupported_namespace,
        bpr_events=d.bpr.events_total, unmapped_actions=d.unmapped_actions,
        malformed_action_system_names=d.malformed_action_system_names)
    result = run_shadow_training(cfg, prepared.prepared_data, diagnostics, device,
                                 complete=prepared.complete, on_quality=on_quality)
    quality = result.quality_report
    run_id = uuid.uuid4().hex
    payload = {
        "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "batch_id": batch.batch_id,
        "interaction_window": {"since": batch.window.interaction_since.isoformat(),
                               "until": batch.window.interaction_until.isoformat()},
        "quality": {"level": quality.level.value, "training_allowed": quality.training_allowed,
                    "metrics": dict(quality.metrics),
                    "issues": [{"code": issue.code, "level": issue.level.value, "count": issue.count,
                                "rate": issue.rate, "breakdown": dict(issue.breakdown)} for issue in quality.issues]},
        "dataset": dict(result.preparation_summary), "config": config_summary,
        "training": {"started": result.training_started, "completed": result.training_completed,
                     "error_code": result.error_code,
                     "metrics": asdict(result.training_metrics) if result.training_metrics else None},
        "production_model_published": False,
    }
    # Fixed report store; callers cannot redirect this writer into Модель.
    report_path = REPORT_ROOT / run_id / "report.json"
    _atomic_report(report_path, payload)
    return replace(result, report_path=str(report_path))
