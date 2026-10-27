"""Train-only streaming scaling with distinct observed-absence and unavailable flags."""
from decimal import Decimal, InvalidOperation

import numpy as np
from sklearn.preprocessing import StandardScaler

from Application.model.customer_propensity_dataset import feature_schema
from .contracts import COLUMNS, CONTINUOUS, FEATURES, MLDataError, PREPROCESSING_VERSION, SEMANTIC_NULL

SCHEMA = feature_schema()


def numeric_batch(rows):
    """Deterministic pre-scaling values. Exact source Decimal is never rewritten."""
    x = np.zeros((len(rows), len(COLUMNS)), dtype=np.float32)
    for row in rows:
        if (tuple(row["features"]) != FEATURES or tuple(row["available"]) != FEATURES
                or set(row["feature_issues"]) != {n for n, v in row["available"].items() if v is False}):
            raise MLDataError("FEATURE_ORDER_OR_AVAILABILITY_MISMATCH")
    for j, name in enumerate(FEATURES):
        spec = SCHEMA[name]
        values = np.empty(len(rows), dtype=np.float64)
        for k, row in enumerate(rows):
            value, available = row["features"][name], row["available"][name]
            if type(available) is not bool or (not available and value is not None):
                raise MLDataError("INVALID_AVAILABILITY")
            x[k, 22 + j] = not available
            if not available:
                issues = row["feature_issues"][name]
                if not isinstance(issues, (list, tuple)) or not issues or any(not isinstance(v, str) or not v for v in issues):
                    raise MLDataError("INVALID_AVAILABILITY_REASON")
                values[k] = 0
                continue
            if value is None:
                if name not in SEMANTIC_NULL:
                    raise MLDataError("INVALID_SEMANTIC_NULL")
                x[k, 44 + SEMANTIC_NULL.index(name)] = 1
                values[k] = spec["window_days"] if name.startswith("days_since_") else 0
            elif spec["type"] == "nullable_integer":
                if type(value) is not int or value < 0:
                    raise MLDataError("INVALID_FEATURE_INTEGER")
                if ((name.startswith("has_") and value not in (0, 1))
                        or ("active_days" in name and value > spec["window_days"])
                        or (name.startswith("days_since_") and value >= spec["window_days"])):
                    raise MLDataError("FEATURE_RANGE_ERROR")
                values[k] = value
            else:
                try:
                    if not isinstance(value, str):
                        raise ValueError
                    dec = Decimal(value)
                    if not dec.is_finite() or not 0 <= dec < 365:
                        raise ValueError
                    values[k] = float(dec)
                except (InvalidOperation, ValueError, OverflowError):
                    raise MLDataError("INVALID_FEATURE_DECIMAL") from None
        if name.startswith("days_since_"):
            values /= spec["window_days"]
        elif not name.startswith("has_"):
            np.log1p(values, out=values)
        if not np.isfinite(values).all():
            raise MLDataError("NONFINITE_FEATURES")
        x[:, j] = values
    if not np.isfinite(x).all():
        raise MLDataError("NONFINITE_FLOAT32_FEATURES")
    return x


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mean = None
        self.scale = None
        self.train_rows = 0

    def partial_fit(self, x, *, role):
        if role != "train" or self.mean is not None:
            raise MLDataError("PREPROCESSING_FIT_FORBIDDEN")
        self.scaler.partial_fit(x[:, CONTINUOUS].astype(np.float64))
        self.train_rows += len(x)

    def freeze(self):
        if not self.train_rows or self.mean is not None:
            raise MLDataError("INVALID_PREPROCESSING_STATE")
        self.mean = np.array(self.scaler.mean_, dtype=np.float64)
        self.scale = np.array(self.scaler.scale_, dtype=np.float64)

    def transform(self, x):
        if self.mean is None or x.ndim != 2 or x.shape[1] != 48 or not np.isfinite(x).all():
            raise MLDataError("INVALID_TRANSFORM_INPUT")
        out = np.array(x, dtype=np.float32, order="C", copy=True)
        out[:, CONTINUOUS] = (out[:, CONTINUOUS].astype(np.float64) - self.mean) / self.scale
        if not np.isfinite(out).all():
            raise MLDataError("NONFINITE_SCALED_FEATURES")
        return out

    def payload(self):
        if self.mean is None:
            raise MLDataError("PREPROCESSING_NOT_FROZEN")
        return {"version": PREPROCESSING_VERSION, "input_features": list(FEATURES), "columns": list(COLUMNS),
                "continuous": list(CONTINUOUS), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "train_rows": self.train_rows}

    @classmethod
    def from_payload(cls, value):
        if (value.get("version") != PREPROCESSING_VERSION or value.get("input_features") != list(FEATURES)
                or value.get("columns") != list(COLUMNS) or value.get("continuous") != list(CONTINUOUS)):
            raise MLDataError("INCOMPATIBLE_PREPROCESSING")
        result = cls()
        result.mean = np.asarray(value["mean"], dtype=np.float64)
        result.scale = np.asarray(value["scale"], dtype=np.float64)
        result.train_rows = value["train_rows"]
        if (result.mean.shape != (19,) or result.scale.shape != (19,) or result.train_rows <= 0
                or not np.isfinite(result.mean).all() or not np.isfinite(result.scale).all() or (result.scale <= 0).any()):
            raise MLDataError("INVALID_SCALING_PARAMETERS")
        return result
