"""Strict numeric JSON research models. No pickle or executable deserialization."""
import hashlib
from pathlib import Path
import shutil
import uuid

import numpy as np
from scipy.special import expit

from .contracts import COLUMNS, FEATURES, MLDataError, canonical, checksum, load_json, write_json
from .metrics import apply_sigmoid
from .preprocessing import Preprocessor, numeric_batch
from .resources import protect_directory


def linear_state(estimator, *, epoch, selection_log_loss):
    return {"type": "SGDClassifier_log_loss_L2", "classes": [0, 1], "dtype": str(estimator.coef_.dtype),
            "columns": list(COLUMNS), "coef": estimator.coef_[0].tolist(),
            "intercept": float(estimator.intercept_[0]), "epoch": epoch,
            "selection_log_loss": selection_log_loss,
            "checkpoint_semantics": "SCORING_ONLY; averaged SGD optimizer resume is not implemented"}


def validate_linear(state):
    if (state.get("type") != "SGDClassifier_log_loss_L2" or state.get("classes") != [0, 1]
            or state.get("columns") != list(COLUMNS) or state.get("dtype") not in ("float32", "float64")):
        raise MLDataError("INVALID_LINEAR_SCHEMA")
    coef = np.asarray(state["coef"], dtype=state["dtype"])
    intercept = np.asarray(state["intercept"], dtype=state["dtype"])
    if coef.shape != (48,) or intercept.shape != () or not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise MLDataError("INVALID_LINEAR_PARAMETERS")
    return coef, intercept


def decision(x, state):
    coef, intercept = validate_linear(state)
    if x.ndim != 2 or x.shape[1] != 48 or not np.isfinite(x).all():
        raise MLDataError("INVALID_LINEAR_INPUT")
    # Explicit per-row float64 reduction: BLAS may choose a different float32 kernel
    # when batch size changes, making otherwise identical predictions differ.
    scores = np.einsum("ij,j->i", x, coef, dtype=np.float64, optimize=False) + float(intercept)
    if not np.isfinite(scores).all():
        raise MLDataError("NONFINITE_DECISION_SCORE")
    return scores


class ResearchModel:
    def __init__(self, metadata, preprocessing, linear, calibration):
        self.metadata = metadata
        self.preprocessing = Preprocessor.from_payload(preprocessing)
        validate_linear(linear)
        apply_sigmoid(np.zeros(1), calibration)
        self.linear, self.calibration = linear, calibration

    def predict(self, rows, *, schema_version=2, feature_order=FEATURES, key_id=None):
        if schema_version != 2 or tuple(feature_order) != FEATURES:
            raise MLDataError("INCOMPATIBLE_INFERENCE_SCHEMA")
        if key_id is not None and key_id != self.metadata["key_id"]:
            raise MLDataError("INCOMPATIBLE_PSEUDONYM_KEY")
        x = self.preprocessing.transform(numeric_batch(rows))
        return apply_sigmoid(decision(x, self.linear), self.calibration)

    def raw_probability(self, rows):
        return expit(decision(self.preprocessing.transform(numeric_batch(rows)), self.linear))


def save_artifact(root, metadata, preprocessing, linear, calibration, metrics, guard):
    root = Path(root)
    staging = root / (".model-building-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        protect_directory(staging)
        for name, value in (("preprocessing.json", preprocessing), ("linear.json", linear),
                            ("calibration.json", calibration), ("metrics.json", metrics)):
            write_json(staging / name, value)
        hashes = {p.name: checksum(p) for p in staging.iterdir()}
        body = {**metadata, "artifact_schema": 1, "status": "RESEARCH_ONLY", "checksums": hashes}
        body["version"] = hashlib.sha256(canonical(body)).hexdigest()
        write_json(staging / "manifest.json", body)
        load_artifact(staging)
        guard.poll(force=True)
        final = root / body["version"]
        if final.exists():
            raise MLDataError("MODEL_VERSION_ALREADY_EXISTS")
        staging.rename(final)
        return final
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_artifact(directory):
    directory = Path(directory)
    m = load_json(directory / "manifest.json")
    expected_names = {"preprocessing.json", "linear.json", "calibration.json", "metrics.json"}
    base = {k: v for k, v in m.items() if k != "version"}
    if (m.get("artifact_schema") != 1 or m.get("status") != "RESEARCH_ONLY"
            or m.get("target") != "order_placed_within_30_days" or m.get("horizon_days") != 30
            or m.get("as_of_level") != "RETROSPECTIVE_EVENT_TIME"
            or m.get("business_validation") != "REQUIRES_BUSINESS_VALIDATION"
            or m.get("feature_schema_version") != 2 or m.get("input_features") != list(FEATURES)
            or m.get("columns") != list(COLUMNS) or set(m.get("checksums", {})) != expected_names
            or hashlib.sha256(canonical(base)).hexdigest() != m.get("version")):
        raise MLDataError("INCOMPATIBLE_MODEL_MANIFEST")
    values = {}
    for name in expected_names:
        p = directory / name
        if p.is_symlink() or p.stat().st_size > 4 * 1024 * 1024 or checksum(p) != m["checksums"][name]:
            raise MLDataError("CORRUPT_MODEL_FILE")
        values[name] = load_json(p)
    return ResearchModel(m, values["preprocessing.json"], values["linear.json"], values["calibration.json"])
