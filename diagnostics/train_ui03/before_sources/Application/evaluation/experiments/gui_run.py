"""One GUI configuration, one training and final validation; no publication API."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import math
import os
import time

from Application.evaluation.bpr import BprTemporalData, evaluate_bpr_snapshot, prepare_bpr_snapshot
from Application.interactions import InteractionType
from Application.model import BPRMF as core
from Application.model.bpr_preparation import BprWeightConfig
from .bpr_weights import KS, require_validation
from .gui_history import RESEARCH_FIXED


class BenchmarkUnavailable(ValueError):
    """Safe readiness failure discovered after actual canonical preparation."""


def gui_config(values):
    cfg = core.TrainConfig(**RESEARCH_FIXED)
    for key in ("w_view_item", "w_favorite", "w_purchase"):
        value = values[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (key == "w_purchase" and value == 0):
            raise ValueError("Invalid weights")
        setattr(cfg, key, float(value))
    epochs = values["epochs"]
    if type(epochs) is not int or not 1 <= epochs <= 10000:
        raise ValueError("Invalid epoch budget")
    cfg.epochs = epochs
    return cfg


def run_validation(snapshot, temporal, cfg, device, emit, check_cancel):
    require_validation(snapshot, temporal)
    if cfg.early_stop or cfg.use_item_features or cfg.seed != 42:
        raise ValueError("Research protection failed")
    check_cancel()
    if not snapshot.history or not snapshot.cases:
        raise BenchmarkUnavailable("Validation benchmark unavailable")
    emit({"stage": "preparation", "device": str(device)})
    weights = BprWeightConfig(view_weight=cfg.w_view_item, favorite_weight=cfg.w_favorite, purchase_weight=cfg.w_purchase)
    data = prepare_bpr_snapshot(snapshot, weights)
    if len(data.training.splits.eval_users):
        raise ValueError("Validation benchmark unavailable")
    dataset = {"training_users": len(data.training.mappings.idx2user),
               "training_items": len(data.training.mappings.idx2item), "training_pairs": len(data.training.splits.train_pairs)}
    # Presentation aggregate of this exact prepared history; history JSON v1 stays unchanged.
    emit({"stage": "training", "device": str(device), **dataset, "training_events": len(snapshot.history), "hyperparameters": asdict(cfg)})
    check_cancel()
    core._set_seed(cfg.seed)
    started = time.monotonic()

    def observer(model, epoch):
        emit({"stage": "epoch", "epoch": epoch.epoch, "epochs": cfg.epochs, "loss": epoch.loss,
              "device": str(device), "training_seconds": time.monotonic() - started})
        check_cancel()

    # Trainer diagnostics are process-wide and may contain raw identifiers.
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        model, _, observed = core.train_prepared_data_with_metrics(cfg, data.training, device, epoch_observer=observer)
    check_cancel()
    if observed.epochs_completed != cfg.epochs or observed.early_stopped:
        raise ValueError("Incomplete epoch budget")
    training_seconds = time.monotonic() - started
    emit({"stage": "validation", "device": str(device)})
    metrics = {}
    for label, kind in (("overall", None), *((kind.value, kind) for kind in InteractionType)):
        selected = snapshot if kind is None else replace(snapshot, cases=tuple(c for c in snapshot.cases if c.interaction_type is kind))
        for k in KS:
            check_cancel()
            score = evaluate_bpr_snapshot(model, BprTemporalData(selected, data.training), k, device)
            metrics.setdefault(label, {})[str(k)] = {"cases": score.evaluated_users, "ndcg": score.ndcg, "recall": score.recall}
    check_cancel()
    return {**dataset, "epochs_completed": observed.epochs_completed, "training_seconds": training_seconds, "metrics": metrics}
