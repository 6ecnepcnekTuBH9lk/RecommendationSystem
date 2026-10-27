"""Synthetic frozen artifacts only: no fit, real data, secrets, GUI or training."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from Application.model.customer_propensity_dataset import feature_schema
from Application.model.customer_propensity_history import PropensityDataError
from Application.model.propensity_ml import predictions as pred
from Application.model.propensity_ml.artifacts import load_artifact
from Application.model.propensity_ml.contracts import COLUMNS, CONTINUOUS, FEATURES, MLCancelled, MLDataError, canonical, checksum
from Application.model.propensity_ml.preprocessing import PREPROCESSING_VERSION


class Guard:
    def __init__(self, cancel_at=None):
        self.calls = 0
        self.cancel_at = cancel_at

    def poll(self, **kwargs):
        self.calls += 1
        if self.calls == self.cancel_at:
            raise MLCancelled


def write(path, value):
    path.write_bytes(canonical(value))


@pytest.fixture(autouse=True)
def forbid_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("NO_FIT_IN_PREDICTION_TESTS")
    for cls in (SGDClassifier, StandardScaler):
        monkeypatch.setattr(cls, "fit", forbidden)
        monkeypatch.setattr(cls, "partial_fit", forbidden)
    monkeypatch.setattr(pred, "protect_directory", lambda path: None)


@pytest.fixture
def inputs(tmp_path):
    modeldir, data = tmp_path / "m", tmp_path / "d"
    modeldir.mkdir()
    data.mkdir()
    values = {n: 0 for n in FEATURES}
    for n in pred.FEATURES:
        if n.startswith(("days_since_", "average_")):
            values[n] = None
    rows = []
    for i in range(17):
        row = {"customer_key": f"{i+1:064x}", "prediction_utc": "2026-09-01T00:00:00Z",
               "target_until_utc": "2026-10-01T00:00:00Z", "split": "inference",
               "features": deepcopy(values), "available": {n: True for n in FEATURES}, "feature_issues": {}}
        if i % 3 == 1:
            for n in ("favorites_7d", "favorites_30d", "favorites_90d", "favorites_active_days_30d", "has_favorite_90d"):
                row["features"][n] = 1
            row["features"]["days_since_favorite_90d"] = 3
        elif i % 3 == 2:
            for n in ("views_7d", "views_30d", "views_90d", "views_180d"):
                row["features"][n] = 2
            row["features"]["views_active_days_30d"] = 1
            row["features"]["has_view_180d"] = 1
            row["features"]["days_since_view_180d"] = 10
        rows.append(row)
    filenames = {n: n + ".jsonl.gz" for n in ("train", "validation", "test", "unknown", "embargo", "inference")}
    for name, filename in filenames.items():
        content = b"".join(canonical(r) + b"\n" for r in rows) if name == "inference" else b""
        (data / filename).write_bytes(gzip.compress(content, compresslevel=6, mtime=0))
    checksums = {n: checksum(data / n) for n in filenames.values()}
    storage = {"storage_contract_version": 1, "storage_format": "jsonl.gz",
               "compression": {"codec": "gzip", "level": 6, "mtime": 0, "filename": ""}, "file_names": filenames}
    identity = {"checksums": checksums, "source": "a"*64, "generation": "synthetic-generation",
                "key_id": "b"*64, "schema": 2, "storage": storage}
    version = hashlib.sha256(canonical(identity)).hexdigest()
    m = {"schema_version": 2, "features": feature_schema(), "version": version, "identity": identity,
         "source_version": "a"*64, "generation": "synthetic-generation", "target": "order_placed_within_30_days",
         "horizon_days": 30, "as_of_level": "RETROSPECTIVE_EVENT_TIME", "business_validation": "REQUIRES_BUSINESS_VALIDATION",
         "purpose": "inference", "status": "RESEARCH_DATASET_ONLY", "probe": None, "label_fields_present": False,
         "checksums": checksums, "file_sizes": {n: (data/n).stat().st_size for n in filenames.values()},
         "prediction_dates": ["2026-09-01T00:00:00Z"], "validation_start": "2026-09-01T00:00:00Z",
         "test_start": "2026-09-01T00:00:00Z", "counts": {"rows": len(rows), "files": {n: len(rows) if n == "inference" else 0 for n in filenames}}, **storage}
    write(data / "manifest.json", m)
    write(data.parent / "CURRENT_INFERENCE.json", {"version": version, "status": m["status"], "manifest_sha256": checksum(data / "manifest.json")})
    pp = {"version": PREPROCESSING_VERSION, "input_features": list(FEATURES), "columns": list(COLUMNS),
          "continuous": list(CONTINUOUS), "mean": [0.]*19, "scale": [1.]*19, "train_rows": 3}
    coef = [0.]*48
    coef[FEATURES.index("favorites_7d")] = -20.
    coef[FEATURES.index("views_7d")] = .1
    linear = {"type": "SGDClassifier_log_loss_L2", "classes": [0, 1], "dtype": "float32",
              "columns": list(COLUMNS), "coef": coef, "intercept": -2.}
    calibration = {"method": "sigmoid_platt_smoothed", "a": 1.01, "b": -.07}
    for name, obj in (("preprocessing", pp), ("linear", linear), ("calibration", calibration), ("metrics", {})):
        write(modeldir / (name + ".json"), obj)
    meta = {"artifact_schema": 1, "status": "RESEARCH_ONLY", "feature_schema_version": 2,
            "input_features": list(FEATURES), "columns": list(COLUMNS), "target": m["target"], "horizon_days": 30,
            "as_of_level": m["as_of_level"], "business_validation": m["business_validation"],
            "source_version": m["source_version"], "generation": m["generation"], "key_id": identity["key_id"],
            "limitations": ["Synthetic only"], "checksums": {n: checksum(modeldir/n) for n in ("preprocessing.json", "linear.json", "calibration.json", "metrics.json")}}
    meta["version"] = hashlib.sha256(canonical(meta)).hexdigest()
    write(modeldir / "manifest.json", meta)
    return modeldir, data, rows


def build(inputs, root, **kwargs):
    model, data, _ = inputs
    return pred.build_predictions(model, data, root, Guard(), **kwargs)


def test_validated_reader(inputs):
    model, data, rows = inputs
    result = pred.validate_payload(model, data, Guard())
    assert result["rows"] == result["unique_pseudonyms"] == len(rows)


@pytest.mark.parametrize("batch", [1, 3, 5, 50])
def test_batch_alignment(inputs, tmp_path, batch):
    modeldir, _, rows = inputs
    frozen = load_artifact(modeldir)
    expected = frozen.predict(rows)
    path = build(inputs, tmp_path/"p", batch_size=batch)
    store = pred.PredictionStore(path, verify_integrity=True)
    for row, probability in zip(rows, expected):
        result = store.get_prediction(row["customer_key"], frozen.metadata["version"], row["prediction_utc"])
        assert result.status is pred.PredictionStatus.READY
        assert result.prediction.probability == probability
        assert result.prediction.target_until == row["target_until_utc"]
    assert store.manifest["rows"] == 17
    assert not list(path.parent.glob(".predictions-building-*"))
    assert not list(path.glob("*-journal"))


def test_reproducible(inputs, tmp_path):
    a = pred.PredictionStore(build(inputs, tmp_path/"a", batch_size=3))
    b = pred.PredictionStore(build(inputs, tmp_path/"b", batch_size=8))
    assert a.manifest["summary"] == b.manifest["summary"]
    assert a.manifest["contract"] == b.manifest["contract"]


def test_lookup_states(inputs, tmp_path):
    store = pred.PredictionStore(build(inputs, tmp_path/"p"))
    c = store.manifest["contract"]
    assert store.get_prediction("f"*64, c["model_version"], c["prediction_date"]).status is pred.PredictionStatus.NOT_FOUND
    assert store.get_prediction("invalid", c["model_version"], c["prediction_date"]).status is pred.PredictionStatus.INVALID_INPUT
    assert store.get_prediction("f"*64, "a"*64, c["prediction_date"]).status is pred.PredictionStatus.INCOMPATIBLE_VERSION


def test_flags(inputs, tmp_path):
    _, _, rows = inputs
    assert pred.warning_mask(rows[0], .1) == pred.LOW_INDIVIDUALIZATION
    assert pred.warning_mask(rows[1], 1e-8) == pred.NEAR_ZERO_WITH_RECENT_FAVORITE
    missing = deepcopy(rows[0])
    missing["available"]["days_since_view_180d"] = False
    missing["feature_issues"]["days_since_view_180d"] = ["SYNTHETIC"]
    assert pred.warning_mask(missing, .1) == pred.HISTORICAL_FEATURES_UNAVAILABLE
    result = pred.PredictionStore(build(inputs, tmp_path/"p")).manifest["summary"]["anomalies"]
    assert result["largest_vector_group"] == result["largest_vector_rows_with_inactivity"] == 6
    assert result["near_zero"] == result["near_zero_with_recent_favorite"] == 6


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-.1], [1.01], [[.1]], []])
def test_bad_probabilities(values):
    with pytest.raises(MLDataError):
        pred._probabilities(values, 1)


def test_model_checksum(inputs):
    model, data, _ = inputs
    (model/"linear.json").write_text("{}")
    with pytest.raises(MLDataError):
        pred.validate_inputs(model, data)


def test_pointer_mismatch(inputs):
    model, data, _ = inputs
    write(data.parent/"CURRENT_INFERENCE.json", {"version": "c"*64})
    with pytest.raises(MLDataError):
        pred.validate_inputs(model, data)


def test_corrupt_gzip(inputs, tmp_path):
    _, data, _ = inputs
    with (data/"inference.jsonl.gz").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises((MLDataError, PropensityDataError)):
        build(inputs, tmp_path/"p")
    assert not list((tmp_path/"p").glob("*/manifest.json"))


def test_duplicate_cleanup(inputs, tmp_path, monkeypatch):
    rows = inputs[2]
    monkeypatch.setattr(pred, "iter_propensity_rows", lambda *a, **k: iter(rows[:-1] + [rows[0]]))
    with pytest.raises(MLDataError, match="DUPLICATE"):
        build(inputs, tmp_path/"p", batch_size=3)
    assert not list((tmp_path/"p").glob(".predictions-building-*"))
    assert not list((tmp_path/"p").glob("*/manifest.json"))


def test_cancel_preserves_previous(inputs, tmp_path):
    prior = build(inputs, tmp_path/"p")
    digest = checksum(prior/"predictions.sqlite")
    with pytest.raises(MLCancelled):
        pred.build_predictions(inputs[0], inputs[1], tmp_path/"p", Guard(cancel_at=3), batch_size=1)
    assert checksum(prior/"predictions.sqlite") == digest
    assert not list((tmp_path/"p").glob(".predictions-building-*"))


def test_write_failure(inputs, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("synthetic output failure")
    monkeypatch.setattr(pred, "write_json", fail)
    with pytest.raises(OSError):
        build(inputs, tmp_path/"p")
    assert not list((tmp_path/"p").glob(".predictions-building-*"))
    assert not list((tmp_path/"p").glob("*/manifest.json"))


def test_atomic_visibility(inputs, tmp_path, monkeypatch):
    original = Path.rename
    def rename(source, target):
        assert not Path(target).exists()
        assert (source/"manifest.json").is_file()
        assert pred.PredictionStore(source, verify_integrity=True).manifest["rows"] == 17
        return original(source, target)
    monkeypatch.setattr(Path, "rename", rename)
    build(inputs, tmp_path/"p")


def test_corrupt_publication(inputs, tmp_path):
    final = build(inputs, tmp_path/"p")
    with (final/"predictions.sqlite").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(MLDataError):
        pred.PredictionStore(final)


def test_readonly_lifecycle(inputs, tmp_path):
    final = build(inputs, tmp_path/"p")
    before = checksum(final/"predictions.sqlite")
    store = pred.PredictionStore(final)
    c = store.manifest["contract"]
    for _ in range(20):
        assert store.get_prediction(inputs[2][0]["customer_key"], c["model_version"], c["prediction_date"]).status is pred.PredictionStatus.READY
    with pred._reader(final/"predictions.sqlite") as db:
        with pytest.raises(sqlite3.OperationalError):
            db.execute("DELETE FROM predictions")
    db.close()
    assert checksum(final/"predictions.sqlite") == before
    assert not list(final.glob("*-wal"))
    assert not list(final.glob("*-shm"))


def test_schema_mismatch(inputs):
    model, data, _ = inputs
    m = json.loads((data/"manifest.json").read_text())
    m["features"].pop("favorites_7d")
    write(data/"manifest.json", m)
    with pytest.raises(MLDataError):
        pred.validate_inputs(model, data)


def test_source_error_cleanup(inputs, tmp_path, monkeypatch):
    def bad(*args, **kwargs):
        yield inputs[2][0]
        raise OSError("synthetic read error")
    monkeypatch.setattr(pred, "iter_propensity_rows", bad)
    with pytest.raises(OSError):
        build(inputs, tmp_path/"p", batch_size=1)
    assert not list((tmp_path/"p").glob(".predictions-building-*"))


def test_input_change_cleanup(inputs, tmp_path):
    def change(_):
        (inputs[0]/"metrics.json").write_text('{"changed":true}')
    with pytest.raises(MLDataError, match="INPUTS_CHANGED"):
        build(inputs, tmp_path/"p", progress=change)
    assert not list((tmp_path/"p").glob("*/manifest.json"))


def test_nan_cleanup(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(pred, "apply_sigmoid", lambda scores, calibration: np.full(len(scores), np.nan))
    with pytest.raises(MLDataError, match="INVALID_PREDICTIONS"):
        build(inputs, tmp_path/"p")
    assert not list((tmp_path/"p").glob(".predictions-building-*"))


def test_staging_conflict(inputs, tmp_path):
    root = tmp_path/"p"
    root.mkdir()
    (root/".predictions-building-existing").mkdir()
    with pytest.raises(MLDataError, match="EXISTING_PREDICTION_STAGING"):
        build(inputs, root)
    assert (root/".predictions-building-existing").is_dir()


def test_lookup_detects_change(inputs, tmp_path):
    final = build(inputs, tmp_path/"p")
    store = pred.PredictionStore(final)
    c = store.manifest["contract"]
    with (final/"predictions.sqlite").open("ab") as stream:
        stream.write(b"changed")
    assert store.get_prediction(inputs[2][0]["customer_key"], c["model_version"], c["prediction_date"]).status is pred.PredictionStatus.ARTIFACT_INCONSISTENT


@pytest.mark.parametrize("size", [0, 50001, True])
def test_invalid_batch(inputs, tmp_path, size):
    with pytest.raises(MLDataError, match="INVALID_BATCH_SIZE"):
        build(inputs, tmp_path/"p", batch_size=size)
