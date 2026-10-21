"""Probability/ranking aggregates, time-specific reliability, fixed Platt mapping."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from .contracts import MLDataError

BINS = (0., .02, .05, .1, .2, .4, .6, .8, 1.)
TOP_FRACTIONS = (.01, .05, .1)


def metrics(y, p, *, prevalence, date_codes=None, dates=None):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if (not len(y) or y.shape != p.shape or not np.isin(y, (0, 1)).all()
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not 0 < prevalence < 1):
        raise MLDataError("INVALID_PROBABILITY_METRIC_INPUT")
    positive, n = int(y.sum(dtype=np.int64)), len(y)
    observed = positive / n
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    result = {"rows": n, "positive": positive, "actual_positive_rate": observed,
              "mean_probability": float(p.mean()),
              "average_precision": float(average_precision_score(y, p)) if positive else None,
              "roc_auc": float(roc_auc_score(y, p)) if 0 < positive < n else None,
              "log_loss": float((-y * np.log(clipped) - (1 - y) * np.log1p(-clipped)).mean()),
              "brier": float(np.square(p - y).mean()),
              "constant": {"probability": prevalence, "mean_probability": prevalence,
                           "actual_positive_rate": observed, "average_precision": observed if positive else None,
                           "roc_auc": .5 if 0 < positive < n else None,
                           "log_loss": -observed * np.log(prevalence) - (1 - observed) * np.log1p(-prevalence),
                           "brier": observed * (1 - prevalence) ** 2 + (1 - observed) * prevalence ** 2},
              "reliability": [], "top_fractions": [], "ece": 0.}
    for j, (low, high) in enumerate(zip(BINS, BINS[1:])):
        mask = (p >= low) & ((p <= high) if j == len(BINS) - 2 else (p < high))
        count = int(mask.sum())
        mean_p = float(p[mask].mean()) if count else None
        fraction = float(y[mask].mean()) if count else None
        result["reliability"].append({"low": low, "high": high, "rows": count,
                                      "positive": int(y[mask].sum()), "mean_probability": mean_p,
                                      "actual_positive_rate": fraction})
        if count:
            result["ece"] += count / n * abs(mean_p - fraction)
    # Stable cache row order resolves score ties; no raw/customer identifier is exported.
    order = np.argsort(-p, kind="stable")
    for fraction in TOP_FRACTIONS:
        k = max(1, int(np.ceil(n * fraction)))
        hits = int(y[order[:k]].sum(dtype=np.int64))
        precision = hits / k
        result["top_fractions"].append({"fraction": fraction, "k": k, "positive": hits,
                                        "precision": precision, "recall": hits / positive if positive else None,
                                        "lift": precision / observed if observed else None})
    if date_codes is not None:
        result["by_date"] = {}
        for code in np.unique(date_codes):
            mask = date_codes == code
            result["by_date"][dates[int(code)]] = metrics(y[mask], p[mask], prevalence=prevalence)
        result["macro_date"] = {key: float(np.mean([v[key] for v in result["by_date"].values() if v[key] is not None]))
                                for key in ("average_precision", "roc_auc", "log_loss", "brier")}
    return result


def fit_sigmoid(scores, y, *, role, guard):
    """Two-parameter Platt sigmoid, with Platt target smoothing; no base-model refit."""
    if role != "calibration":
        raise MLDataError("CALIBRATION_FIT_FORBIDDEN")
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y, dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if (not positives or not negatives or scores.shape != y.shape or not np.isfinite(scores).all()
            or not np.isin(y, (0, 1)).all()):
        raise MLDataError("INVALID_CALIBRATION_DATA")
    mean, scale = float(scores.mean()), float(scores.std())
    if scale <= np.finfo(float).eps:
        raise MLDataError("CONSTANT_CALIBRATION_SCORE")
    z = (scores - mean) / scale
    target = np.where(y == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2))

    def objective(ab):
        guard.poll()
        logits = ab[0] * z + ab[1]
        residual = expit(logits) - target
        return float(np.mean(np.logaddexp(0, logits) - target * logits)), np.array((
            np.dot(residual, z) / len(y), residual.mean()))

    optimum = minimize(objective, np.array((1., np.log(positives / negatives))), jac=True,
                       method="L-BFGS-B", bounds=((0., None), (None, None)),
                       options={"maxiter": 100, "ftol": 1e-13, "gtol": 1e-8})
    if not optimum.success or not np.isfinite(optimum.x).all() or optimum.x[0] <= 0:
        raise MLDataError("CALIBRATION_OPTIMIZATION_FAILED")
    a = float(optimum.x[0] / scale)
    b = float(optimum.x[1] - a * mean)
    return {"method": "sigmoid_platt_smoothed", "formula": "expit(a * raw_logit + b)", "a": a, "b": b,
            "role": role, "rows": len(y), "positive": positives, "actual_positive_rate": positives / len(y),
            "iterations": int(optimum.nit), "score_mean": mean, "score_std": scale,
            "target_smoothing": "positive=(Np+1)/(Np+2); negative=1/(Nn+2)"}


def apply_sigmoid(scores, calibration):
    if calibration.get("method") != "sigmoid_platt_smoothed":
        raise MLDataError("INCOMPATIBLE_CALIBRATION")
    a, b = calibration["a"], calibration["b"]
    if not np.isfinite([a, b]).all() or a <= 0:
        raise MLDataError("INVALID_CALIBRATION_PARAMETERS")
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise MLDataError("NONFINITE_DECISION_SCORE")
    return expit(a * scores + b)
