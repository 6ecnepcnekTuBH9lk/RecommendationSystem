"""One supervised SGD experiment with disjoint temporal calibration and final test."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits

from .artifacts import decision, linear_state, load_artifact, save_artifact
from .contracts import COLUMNS, FEATURES, MLDataError, canonical, checksum, dataset_manifest, temporal_roles, write_json
from .data import build_cache, writer_lock
from .metrics import apply_sigmoid, fit_sigmoid, metrics
from .preprocessing import Preprocessor


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 50000
    seed: int = 20261020
    epochs: int = 10
    patience: int = 2
    min_delta: float = 1e-5
    alpha: float = 1e-4
    threads: int = 2

    def __post_init__(self):
        if (type(self.batch_size) is not int or not 1 <= self.batch_size <= 50000
                or type(self.epochs) is not int or not 1 <= self.epochs <= 10
                or self.patience < 1 or self.alpha <= 0 or self.threads < 1 or self.min_delta < 0):
            raise ValueError("Invalid bounded experiment configuration")


def dependencies():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}


def score_cache(cache, pp, model, selected_codes, guard, batch_size):
    selected = np.flatnonzero(np.isin(cache.dates, selected_codes))
    scores = np.empty(len(selected), dtype=np.float64)
    y = np.empty(len(selected), dtype=np.uint8)
    dates = np.empty(len(selected), dtype=np.uint8)
    for low in range(0, len(selected), batch_size):
        guard.poll(force=True)
        idx = selected[low:low + batch_size]
        x = pp.transform(cache.x[idx])
        scores[low:low + len(idx)] = decision(x, model)
        y[low:low + len(idx)], dates[low:low + len(idx)] = cache.y[idx], cache.dates[idx]
    return scores, y, dates


def train_epochs(cache, selection, pp, config, prevalence, run, guard, progress):
    if cache.manifest["role"] != "train" or selection.manifest["role"] != "validation":
        raise MLDataError("MODEL_FIT_ROLE_FORBIDDEN")
    estimator = SGDClassifier(loss="log_loss", penalty="l2", alpha=config.alpha, learning_rate="optimal",
                              average=True, class_weight=None, early_stopping=False, shuffle=False,
                              random_state=config.seed)
    rng = np.random.default_rng(config.seed)
    history = []
    best, best_loss, stale = None, float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        started = monotonic()
        guard.poll(force=True)
        order = np.arange(cache.manifest["rows"], dtype=np.uint32)
        rng.shuffle(order)
        rows = 0
        for low in range(0, len(order), config.batch_size):
            guard.poll(force=True)
            idx = order[low:low + config.batch_size]
            x = pp.transform(cache.x[idx])
            estimator.partial_fit(x, cache.y[idx], classes=np.array([0, 1]) if epoch == 1 and low == 0 else None)
            if not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
                raise MLDataError("SGD_NUMERICAL_DIVERGENCE")
            rows += len(idx)
            progress({"stage": "train", "epoch": epoch, "rows": rows, "total": len(order)})
        del order
        state = linear_state(estimator, epoch=epoch, selection_log_loss=None)
        scores, y, dates = score_cache(selection, pp, state, range(5), guard, config.batch_size)
        guard.poll(force=True)
        result = metrics(y, expit(scores), prevalence=prevalence, date_codes=dates, dates=selection.manifest["dates"])
        del scores, y, dates
        loss = result["log_loss"]
        state["selection_log_loss"] = loss
        write_json(run / f"checkpoint-epoch-{epoch:02d}.json", state)
        epoch_info = {"epoch": epoch, "train_rows": rows, "seconds": monotonic() - started,
                      "selection": result, "resources": guard.payload()}
        history.append(epoch_info)
        write_json(run / f"epoch-{epoch:02d}-metrics.json", epoch_info)
        progress({"stage": "epoch_complete", "epoch": epoch, "selection_log_loss": loss,
                  "selection_ap": result["average_precision"], "seconds": epoch_info["seconds"]})
        # Keep the actual minimum; min_delta controls patience, not whether a lower checkpoint is preserved.
        significant = loss < best_loss - config.min_delta
        if loss < best_loss:
            best, best_loss = state, loss
        stale = 0 if significant else stale + 1
        if stale >= config.patience:
            break
    if best is None:
        raise MLDataError("NO_VALID_CHECKPOINT")
    return best, {"epochs": history, "best_epoch": best["epoch"], "best_selection_log_loss": best_loss,
                  "stop_reason": "SELECTION_PATIENCE" if stale >= config.patience else "MAX_EPOCHS",
                  "hyperparameters": estimator.get_params(), "prevalence": prevalence,
                  "optimizer_resume": "NOT_SUPPORTED; numeric checkpoints are scoring-only"}


def run_experiment(dataset, run, models_root, guard, *, config=TrainingConfig(), allow_test=False,
                   progress=lambda value: None):
    """No retries. Validation cache contains 11 dates; no test payload access until freeze."""
    dataset, run, models_root = Path(dataset), Path(run), Path(models_root)
    m = dataset_manifest(dataset)
    roles = temporal_roles(m)
    run.mkdir(exist_ok=False)
    from .resources import protect_directory
    protect_directory(run)
    caches = []
    started = monotonic()
    metadata = {"dataset_version": m["version"], "source_version": m["source_version"],
                "generation": m["generation"], "key_id": m["identity"]["key_id"],
                "feature_schema_version": 2, "input_features": list(FEATURES), "columns": list(COLUMNS),
                "target": m["target"], "horizon_days": m["horizon_days"], "as_of_level": m["as_of_level"],
                "business_validation": m["business_validation"], "limitations": m["limitations"],
                "temporal_roles": roles, "dependencies": dependencies(), "configuration": asdict(config),
                "dataset_manifest_sha256": checksum(dataset / "manifest.json")}
    frozen = {**metadata, "test_authorized": allow_test, "calibration_method": "sigmoid_platt_smoothed",
              "probability_bins": [0., .02, .05, .1, .2, .4, .6, .8, 1.],
              "metrics": ["log_loss", "brier", "average_precision_noninterpolated", "roc_auc", "reliability", "top1_5_10pct"],
              "resource_policy": {"private_commit_bytes": 3 * 1024**3, "available_physical_floor_bytes": 4 * 1024**3,
                                  "available_commit_floor_bytes": 3 * 1024**3, "disk_floor_bytes": 20 * 1024**3,
                                  "enforcement": "COOPERATIVE_PERIODIC", "automatic_wallclock_timeout": None},
              "prediction_reduction": "per-row float64 einsum; batch-size invariant",
              "tie_policy": "stable cache row order; no post-test changes",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "code_sha256": {p.name: checksum(p) for p in Path(__file__).parent.glob("*.py")}}
    write_json(run / "frozen_config.json", frozen)
    try:
        with writer_lock(models_root), threadpool_limits(limits=config.threads):
            pp = Preprocessor()
            train = build_cache(run / "cache-train", dataset, m, "train", roles["train"], config.batch_size, guard,
                                preprocessor=pp, progress=progress)
            caches.append(train)
            pp.freeze()
            write_json(run / "preprocessing_report.json", pp.payload())
            prevalence = train.manifest["positive"] / train.manifest["rows"]
            validation_dates = roles["selection"] + roles["calibration"] + roles["holdout"]
            validation = build_cache(run / "cache-validation", dataset, m, "validation", validation_dates,
                                     config.batch_size, guard, progress=progress)
            caches.append(validation)
            linear, training = train_epochs(train, validation, pp, config, prevalence, run, guard, progress)
            training["elapsed_before_calibration_seconds"] = monotonic() - started
            write_json(run / "training_metrics.json", training)
            scores, y, codes = score_cache(validation, pp, linear, range(5, 8), guard, config.batch_size)
            calibration = fit_sigmoid(scores, y, role="calibration", guard=guard)
            calibration["prediction_dates"] = roles["calibration"]
            cal_report = {"parameters": calibration,
                          "calibration_raw": metrics(y, expit(scores), prevalence=prevalence, date_codes=codes, dates=validation_dates),
                          "calibration_calibrated": metrics(y, apply_sigmoid(scores, calibration), prevalence=prevalence,
                                                            date_codes=codes, dates=validation_dates)}
            del scores, y, codes
            scores, y, codes = score_cache(validation, pp, linear, range(8, 11), guard, config.batch_size)
            cal_report["holdout_raw"] = metrics(y, expit(scores), prevalence=prevalence, date_codes=codes, dates=validation_dates)
            cal_report["holdout_calibrated"] = metrics(y, apply_sigmoid(scores, calibration), prevalence=prevalence,
                                                        date_codes=codes, dates=validation_dates)
            del scores, y, codes
            cal_report["quality_warning"] = (cal_report["holdout_calibrated"]["log_loss"] >= cal_report["holdout_raw"]["log_loss"]
                                               or cal_report["holdout_calibrated"]["brier"] >= cal_report["holdout_raw"]["brier"])
            write_json(run / "calibration_report.json", cal_report)
            pipeline = {"preprocessing": pp.payload(), "linear": linear, "calibration": calibration}
            pipeline_hash = hashlib.sha256(canonical(pipeline)).hexdigest()
            write_json(run / "pipeline_frozen.json", {"sha256": pipeline_hash, "pipeline": pipeline,
                                                       "test_not_opened": True, "rules": frozen})
            progress({"stage": "pipeline_frozen", "best_epoch": linear["epoch"], "sha256": pipeline_hash})
            test_result = None
            if allow_test:
                test = build_cache(run / "cache-test", dataset, m, "test", roles["test"], config.batch_size, guard,
                                   progress=progress)
                caches.append(test)
                scores, y, codes = score_cache(test, pp, linear, range(11), guard, config.batch_size)
                test_result = {"raw": metrics(y, expit(scores), prevalence=prevalence, date_codes=codes, dates=roles["test"]),
                               "calibrated": metrics(y, apply_sigmoid(scores, calibration), prevalence=prevalence,
                                                     date_codes=codes, dates=roles["test"]),
                               "pipeline_frozen_sha256": pipeline_hash, "test_evaluations": 1,
                               "test_is_now_consumed": True}
                del scores, y, codes
                write_json(run / "test_metrics.json", test_result)
            if hashlib.sha256(canonical(pipeline)).hexdigest() != pipeline_hash:
                raise MLDataError("PIPELINE_CHANGED_AFTER_FREEZE")
            quality_warnings = []
            if cal_report["quality_warning"]:
                quality_warnings.append("CALIBRATION_DID_NOT_IMPROVE_BOTH_HOLDOUT_PROPER_SCORES")
            for label, result in (("HOLDOUT", cal_report["holdout_calibrated"]),
                                  ("TEST", test_result["calibrated"] if test_result else None)):
                if result is not None and (result["log_loss"] >= result["constant"]["log_loss"]
                                           or result["brier"] >= result["constant"]["brier"]
                                           or result["average_precision"] <= result["constant"]["average_precision"]):
                    quality_warnings.append(label + "_NOT_BETTER_THAN_CONSTANT")
            metadata["pipeline_sha256"] = pipeline_hash
            metadata["quality_warnings"] = quality_warnings
            metadata["elapsed_seconds"] = monotonic() - started
            metadata["resource_metrics"] = guard.payload()
            artifact = save_artifact(models_root, metadata, pp.payload(), linear, calibration,
                                     {"training": training, "calibration": cal_report, "test": test_result}, guard)
            model = load_artifact(artifact)
            # Export parity checked on bounded transformed data, never used to select/refit anything.
            x = pp.transform(train.x[:min(1024, len(train.x))])
            parity = np.max(np.abs(decision(x, model.linear) - decision(x, linear)))
            if parity > 1e-6:
                raise MLDataError("MODEL_EXPORT_PARITY_FAILED")
            result = {"status": "BASELINE TRAINED WITH QUALITY WARNINGS" if quality_warnings else "BASELINE READY FOR REVIEW",
                      "artifact": str(artifact.resolve()), "version": model.metadata["version"],
                      "best_epoch": linear["epoch"], "export_max_score_delta": float(parity),
                      "elapsed_seconds": monotonic() - started, "test_completed": allow_test,
                      "limitations": m["limitations"], "quality_warnings": quality_warnings, "production_ready": False}
            if not allow_test:
                result["status"] = "TRAINING PARTIAL"
            write_json(run / "artifact_validation.json", result)
            return result
    finally:
        for cache in caches:
            cache.close()
        write_json(run / "resource_metrics.json", guard.payload())
