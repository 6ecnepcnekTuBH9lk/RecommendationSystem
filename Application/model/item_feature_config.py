"""Lightweight feature configuration contract shared by GUI and ML entry points."""


def normalize_feature_norm(value, *, legacy=False):
    if not isinstance(value, str):
        raise ValueError("feature_norm: допустимы mean, sum, sqrt")
    norm = value.strip().lower()
    if legacy:
        norm = {"avg": "mean", "sqrt_mean": "sqrt", "sqrt-norm": "sqrt"}.get(norm, norm)
    if norm not in ("mean", "sum", "sqrt"):
        raise ValueError("feature_norm: допустимы mean, sum, sqrt")
    return norm
