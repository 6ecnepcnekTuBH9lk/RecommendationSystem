"""Immutable research predictions. Reuses the frozen model and validated reader.

No training, pseudonym generation, GUI integration, or model pointer changes.
One SQLite database is the prediction payload; metadata is shared by its rows.
"""
from collections import Counter
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
from time import monotonic
import uuid

import numpy as np

from Application.model.customer_propensity_storage import iter_propensity_rows
from .artifacts import decision, load_artifact
from .contracts import COLUMNS, FEATURES, MLDataError, canonical, checksum, dataset_manifest, load_json, write_json
from .data import writer_lock
from .metrics import apply_sigmoid
from .preprocessing import numeric_batch
from .resources import protect_directory

HEX = re.compile(r"[0-9a-f]{64}\Z")
NEAR_ZERO_WITH_RECENT_FAVORITE = 1
LOW_INDIVIDUALIZATION = 2
HISTORICAL_FEATURES_UNAVAILABLE = 4
WARNING_NAMES = {
    NEAR_ZERO_WITH_RECENT_FAVORITE: "NEAR_ZERO_WITH_RECENT_FAVORITE",
    LOW_INDIVIDUALIZATION: "LOW_INDIVIDUALIZATION",
    HISTORICAL_FEATURES_UNAVAILABLE: "HISTORICAL_FEATURES_UNAVAILABLE",
}


class PredictionStatus(str, Enum):
    READY = "READY"
    NOT_FOUND = "NOT_FOUND"
    INVALID_INPUT = "INVALID_INPUT"
    INCOMPATIBLE_VERSION = "INCOMPATIBLE_VERSION"
    ARTIFACT_UNAVAILABLE = "ARTIFACT_UNAVAILABLE"
    ARTIFACT_INCONSISTENT = "ARTIFACT_INCONSISTENT"


@dataclass(frozen=True, slots=True)
class Prediction:
    customer_pseudonym: str = field(repr=False)
    probability: float
    prediction_date: str
    target_until: str
    model_version: str
    warnings: tuple[str, ...]
    status: str = "RESEARCH_ONLY"
    pipeline: str = "calibrated"


@dataclass(frozen=True, slots=True)
class PredictionResult:
    status: PredictionStatus
    prediction: Prediction | None = None


def _date(value):
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None or dt.utcoffset() != timedelta(0) or dt.isoformat().replace("+00:00", "Z") != value:
            raise ValueError
        return dt
    except (ValueError, TypeError, AttributeError):
        raise MLDataError("INVALID_PREDICTION_DATE") from None


def validate_inputs(model_directory, dataset_directory):
    """Metadata preflight; payload validation is performed by the stock reader."""
    model_directory, dataset_directory = Path(model_directory), Path(dataset_directory)
    model = load_artifact(model_directory)
    m = dataset_manifest(dataset_directory, purpose="inference")
    pointer = load_json(dataset_directory.parent / "CURRENT_INFERENCE.json")
    if (pointer.get("version") != m["version"] or pointer.get("status") != m["status"]
            or pointer.get("manifest_sha256") != checksum(dataset_directory / "manifest.json")
            or hashlib.sha256(canonical(m["identity"])).hexdigest() != m["version"]
            or any(m[k] != model.metadata[k] for k in ("source_version", "generation", "target", "horizon_days"))
            or m["identity"]["key_id"] != model.metadata["key_id"]
            or m.get("label_fields_present") is not False or len(m["prediction_dates"]) != 1
            or m["counts"]["rows"] != m["counts"]["files"]["inference"]
            or m["counts"]["rows"] <= 0):
        raise MLDataError("INCOMPATIBLE_INFERENCE_INPUTS")
    _date(m["prediction_dates"][0])
    return model, m


def _input_hashes(model_directory, dataset_directory):
    return {str(p): checksum(p) for p in (
        *(Path(model_directory) / n for n in ("manifest.json", "preprocessing.json", "linear.json", "calibration.json", "metrics.json")),
        Path(dataset_directory) / "manifest.json", Path(dataset_directory).parent / "CURRENT_INFERENCE.json")}


def validate_payload(model_directory, dataset_directory, guard):
    """Full read-only preflight, no scoring; SHA, gzip CRC, schema, uniqueness."""
    model, m = validate_inputs(model_directory, dataset_directory)
    hashes = _input_hashes(model_directory, dataset_directory)
    keys = set()
    rows = 0
    reader = iter_propensity_rows(dataset_directory, "inference", purpose="inference",
                                 source_version=m["source_version"], generation=m["generation"],
                                 cancelled=lambda: (guard.poll() or False))
    try:
        for row in reader:
            key = row["customer_key"]
            if key in keys:
                raise MLDataError("DUPLICATE_INFERENCE_PSEUDONYM")
            keys.add(key)
            rows += 1
            if rows % 50000 == 0:
                guard.poll(force=True)
    finally:
        reader.close()
    if hashes != _input_hashes(model_directory, dataset_directory):
        raise MLDataError("INPUTS_CHANGED_DURING_PREFLIGHT")
    return {"model_version": model.metadata["version"], "dataset_version": m["version"],
            "rows": rows, "unique_pseudonyms": len(keys), "sha256_gzip_schema_count_valid": True,
            "prediction_date": m["prediction_dates"][0], "feature_schema_version": 2,
            "input_features": len(FEATURES), "model_columns": len(COLUMNS), "pipeline": "calibrated"}


def warning_mask(row, probability):
    """Descriptive research flags, never probability correction or business rules."""
    features, available = row["features"], row["available"]
    unavailable = not all(available.values())
    favorite = available["has_favorite_90d"] and features["has_favorite_90d"] == 1
    inactive = (not unavailable and all(
        features[n] is None if n.startswith(("days_since_", "average_")) else features[n] == 0
        for n in FEATURES))
    return ((NEAR_ZERO_WITH_RECENT_FAVORITE if probability <= 1e-6 and favorite else 0)
            | (LOW_INDIVIDUALIZATION if inactive else 0)
            | (HISTORICAL_FEATURES_UNAVAILABLE if unavailable else 0))


def _probabilities(values, expected):
    values = np.asarray(values)
    if values.shape != (expected,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise MLDataError("INVALID_PREDICTIONS")
    return values


def _distribution(p):
    return {"rows": len(p), "minimum": float(p.min()), "maximum": float(p.max()), "mean": float(p.mean()),
            "quantiles": dict(zip(("p1", "p5", "p10", "p25", "median", "p75", "p90", "p95", "p99"),
                                  map(float, np.quantile(p, [.01, .05, .1, .25, .5, .75, .9, .95, .99])))),
            "descriptive_counts": {"below_0_1_percent": int((p < .001).sum()), "below_1_percent": int((p < .01).sum()),
                                   "from_1_to_5_percent": int(((p >= .01) & (p < .05)).sum()),
                                   "from_5_to_10_percent": int(((p >= .05) & (p <= .1)).sum()),
                                   "above_10_percent": int((p > .1).sum())},
            "thresholds_are_business_segments": False}


def _reader(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
    except BaseException:
        connection.close()
        raise
    return connection


class PredictionStore:
    """Validate once off GUI thread; bounded primary-key lookups open/close RO handles.

    Full SHA256 is checked at construction, never on each get_prediction call.
    Version directories are immutable. A size/mtime change invalidates this handle.
    """

    def __init__(self, directory, *, verify_integrity=False):
        self.directory = Path(directory)
        self.manifest = load_json(self.directory / "manifest.json")
        m = self.manifest
        body = {k: v for k, v in m.items() if k != "version"}
        try:
            c = m["contract"]
            valid = (m["artifact_schema"] == 1 and m["status"] == "RESEARCH_ONLY" and c["pipeline"] == "calibrated"
                     and c["feature_schema_version"] == 2 and c["input_features"] == list(FEATURES)
                     and c["model_columns"] == list(COLUMNS) and c["horizon_days"] == 30
                     and c["target"] == "order_placed_within_30_days"
                     and HEX.fullmatch(c["model_version"]) and HEX.fullmatch(c["dataset_version"])
                     and _date(c["target_until"]) - _date(c["prediction_date"]) == timedelta(days=30)
                     and hashlib.sha256(canonical(body)).hexdigest() == m["version"])
        except (KeyError, TypeError):
            valid = False
        if not valid:
            raise MLDataError("INVALID_PREDICTION_MANIFEST")
        self.path = self.directory / "predictions.sqlite"
        spec = m["files"]["predictions.sqlite"]
        if (self.path.is_symlink() or self.path.stat().st_size != spec["bytes"] or checksum(self.path) != spec["sha256"]):
            raise MLDataError("CORRUPT_PREDICTION_PAYLOAD")
        with closing(_reader(self.path)) as db:
            if (db.execute("PRAGMA user_version").fetchone()[0] != 1
                    or json.loads(db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()[0]) != c
                    or db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] != m["rows"]):
                raise MLDataError("INCONSISTENT_PREDICTION_PAYLOAD")
            if verify_integrity and db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise MLDataError("PREDICTION_INTEGRITY_FAILED")
        self._stat = self._fingerprint()

    def _fingerprint(self):
        s = self.path.stat()
        return s.st_size, s.st_mtime_ns

    def get_prediction(self, customer_pseudonym, model_version, prediction_date):
        if not isinstance(customer_pseudonym, str) or not HEX.fullmatch(customer_pseudonym):
            return PredictionResult(PredictionStatus.INVALID_INPUT)
        c = self.manifest["contract"]
        if model_version != c["model_version"] or prediction_date != c["prediction_date"]:
            return PredictionResult(PredictionStatus.INCOMPATIBLE_VERSION)
        try:
            if self._fingerprint() != self._stat:
                return PredictionResult(PredictionStatus.ARTIFACT_INCONSISTENT)
            with closing(_reader(self.path)) as db:
                row = db.execute("SELECT probability,warning_mask FROM predictions WHERE customer_pseudonym=?",
                                 (customer_pseudonym,)).fetchone()
            if row is None:
                return PredictionResult(PredictionStatus.NOT_FOUND)
            probability, mask = row
            if (not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1
                    or type(mask) is not int or mask < 0 or mask & ~7):
                return PredictionResult(PredictionStatus.ARTIFACT_INCONSISTENT)
            record = Prediction(customer_pseudonym, probability, c["prediction_date"], c["target_until"],
                                c["model_version"], tuple(name for bit, name in WARNING_NAMES.items() if mask & bit))
            return PredictionResult(PredictionStatus.READY, record)
        except FileNotFoundError:
            return PredictionResult(PredictionStatus.ARTIFACT_UNAVAILABLE)
        except (OSError, sqlite3.DatabaseError):
            return PredictionResult(PredictionStatus.ARTIFACT_INCONSISTENT)


def build_predictions(model_directory, dataset_directory, output_root, guard, *, batch_size=50000,
                      progress=lambda value: None):
    """One fail-closed attempt. Staging stays private and is removed on any failure."""
    if type(batch_size) is not int or not 1 <= batch_size <= 50000:
        raise MLDataError("INVALID_BATCH_SIZE")
    model, m = validate_inputs(model_directory, dataset_directory)
    input_hashes = _input_hashes(model_directory, dataset_directory)
    if Path(output_root).is_symlink():
        raise MLDataError("INVALID_PREDICTION_ROOT")
    root = Path(output_root).resolve()
    if not root.exists():
        root.mkdir(parents=True)
        protect_directory(root)
    with writer_lock(root):
        if any(root.glob(".predictions-building-*")):
            raise MLDataError("EXISTING_PREDICTION_STAGING_REQUIRES_REVIEW")
        return _build_locked(model, m, model_directory, dataset_directory, root, guard,
                             input_hashes, batch_size, progress)


def _build_locked(model, m, model_directory, dataset_directory, root, guard, input_hashes, batch_size, progress):
    stage = root / (".predictions-building-" + uuid.uuid4().hex)
    started = monotonic()
    expected = m["counts"]["files"]["inference"]
    guard.poll(force=True, extra_disk_bytes=expected * 512 + 1024**2)
    stage.mkdir()
    source = db = None
    try:
        protect_directory(stage)
        contract = {"model_version": model.metadata["version"], "dataset_version": m["version"],
                    "model_manifest_sha256": checksum(Path(model_directory) / "manifest.json"),
                    "dataset_manifest_sha256": checksum(Path(dataset_directory) / "manifest.json"),
                    "source_version": m["source_version"], "generation": m["generation"], "key_id": m["identity"]["key_id"],
                    "feature_schema_version": 2, "input_features": list(FEATURES), "model_columns": list(COLUMNS),
                    "prediction_date": m["prediction_dates"][0], "horizon_days": 30,
                    "target_until": (_date(m["prediction_dates"][0]) + timedelta(days=30)).isoformat().replace("+00:00", "Z"),
                    "target": m["target"], "pipeline": "calibrated", "as_of_level": "RETROSPECTIVE_EVENT_TIME"}
        db = sqlite3.connect(stage / "predictions.sqlite")
        db.executescript("""
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA user_version=1;
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE predictions(
                customer_pseudonym TEXT PRIMARY KEY,
                probability REAL NOT NULL CHECK(probability>=0 AND probability<=1),
                warning_mask INTEGER NOT NULL CHECK(warning_mask>=0 AND warning_mask<=7)
            ) WITHOUT ROWID;
        """)
        db.execute("INSERT INTO metadata VALUES('contract',?)", (canonical(contract).decode(),))
        p = np.empty(expected, dtype=np.float64)
        vectors = Counter()
        inactive_vectors = Counter()
        inactive_probabilities = Counter()
        flags = Counter()
        near_zero = near_favorite = total = 0
        batch = []
        source = iter_propensity_rows(dataset_directory, "inference", purpose="inference",
                                     source_version=m["source_version"], generation=m["generation"],
                                     cancelled=lambda: (guard.poll() or False))

        def consume(rows):
            nonlocal total, near_zero, near_favorite
            guard.poll(force=True)
            x = model.preprocessing.transform(numeric_batch(rows))
            probabilities = _probabilities(apply_sigmoid(decision(x, model.linear), model.calibration), len(rows))
            if total + len(rows) > expected:
                raise MLDataError("INFERENCE_COUNT_MISMATCH")
            records = []
            for row, probability, vector in zip(rows, probabilities, x):
                if row["prediction_utc"] != contract["prediction_date"] or row["target_until_utc"] != contract["target_until"]:
                    raise MLDataError("INFERENCE_DATE_MISMATCH")
                key = row["customer_key"]
                if not isinstance(key, str) or not HEX.fullmatch(key):
                    raise MLDataError("INVALID_PREDICTION_PSEUDONYM")
                mask = warning_mask(row, probability)
                flags.update(name for bit, name in WARNING_NAMES.items() if mask & bit)
                near_zero += int(probability <= 1e-6)
                near_favorite += int(bool(mask & NEAR_ZERO_WITH_RECENT_FAVORITE))
                # Signed zero is equal numerically; normalize it only for vector counts.
                signature = vector.copy()
                signature[signature == 0] = 0
                vector_bytes = signature.tobytes()
                vectors[vector_bytes] += 1
                if mask & LOW_INDIVIDUALIZATION:
                    inactive_vectors[vector_bytes] += 1
                    inactive_probabilities[float(probability)] += 1
                records.append((key, float(probability), mask))
            try:
                db.executemany("INSERT INTO predictions VALUES(?,?,?)", records)
            except sqlite3.IntegrityError:
                raise MLDataError("DUPLICATE_OR_INVALID_PREDICTION") from None
            p[total:total + len(rows)] = probabilities
            total += len(rows)
            progress({"stage": "inference", "rows": total, "total": expected,
                      "seconds": monotonic() - started})

        for row in source:
            batch.append(row)
            if len(batch) == batch_size:
                consume(batch)
                batch.clear()
        if batch:
            consume(batch)
        if total != expected:
            raise MLDataError("INFERENCE_COUNT_MISMATCH")
        db.commit()
        db.close()
        db = None
        if hasattr(source, "close"):
            source.close()
        source = None
        if input_hashes != _input_hashes(model_directory, dataset_directory):
            raise MLDataError("INPUTS_CHANGED_DURING_INFERENCE")
        with closing(_reader(stage / "predictions.sqlite")) as check:
            if check.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise MLDataError("PREDICTION_INTEGRITY_FAILED")
            if check.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] != expected:
                raise MLDataError("PREDICTION_COUNT_MISMATCH")
        guard.poll(force=True)
        unique_probability, probability_counts = np.unique(p, return_counts=True)
        largest_p = int(probability_counts.argmax())
        largest_vector, largest_vector_count = vectors.most_common(1)[0]
        largest_vector_array = np.frombuffer(largest_vector, dtype=np.float32).reshape(1, 48)
        largest_vector_probability = float(apply_sigmoid(decision(largest_vector_array, model.linear), model.calibration)[0])
        summary = {"processed": total, "error_rows": 0, "distribution": _distribution(p),
                   "anomalies": {"near_zero": near_zero, "near_zero_with_recent_favorite": near_favorite,
                                 "warning_counts": dict(flags), "unique_vectors": len(vectors),
                                 "repeated_vector_groups": sum(c > 1 for c in vectors.values()),
                                 "rows_in_repeated_vectors": sum(c for c in vectors.values() if c > 1),
                                 "largest_vector_group": largest_vector_count,
                                 "largest_vector_rows_with_inactivity": inactive_vectors[largest_vector],
                                 "largest_vector_probability": largest_vector_probability,
                                 "unique_probabilities": len(unique_probability),
                                 "rows_in_repeated_probabilities": int(probability_counts[probability_counts > 1].sum()),
                                 "largest_probability_group": int(probability_counts[largest_p]),
                                 "largest_probability": float(unique_probability[largest_p]),
                                 "largest_probability_rows_with_inactivity": inactive_probabilities[float(unique_probability[largest_p])],
                                 "no_target_available": True}}
        payload = stage / "predictions.sqlite"
        with payload.open("r+b") as stream:
            os.fsync(stream.fileno())
        body = {"artifact_schema": 1, "status": "RESEARCH_ONLY", "contract": contract, "rows": total,
                "created_utc": datetime.now(timezone.utc).isoformat(), "summary": summary,
                "warning_definitions": WARNING_NAMES,
                "limitations": model.metadata["limitations"] + [
                    "Individual probabilities are unreliable in known favorite/near-zero groups; not for automated marketing.",
                    "Identical inactive input vectors imply limited individualization; no business thresholds validated."],
                "files": {"predictions.sqlite": {"sha256": checksum(payload), "bytes": payload.stat().st_size}}}
        body["version"] = hashlib.sha256(canonical(body)).hexdigest()
        write_json(stage / "manifest.json", body)
        with (stage / "manifest.json").open("r+b") as stream:
            os.fsync(stream.fileno())
        PredictionStore(stage)
        guard.poll(force=True)
        final = root / body["version"]
        if final.exists():
            raise MLDataError("PREDICTION_VERSION_ALREADY_EXISTS")
        stage.rename(final)
        return final
    finally:
        if source is not None and hasattr(source, "close"):
            source.close()
        if db is not None:
            db.close()
        if stage.exists():
            if stage.resolve().parent != root or not stage.name.startswith(".predictions-building-"):
                raise MLDataError("UNSAFE_STAGING_PATH")
            shutil.rmtree(stage)
