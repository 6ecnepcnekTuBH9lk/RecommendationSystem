"""Frozen temporal roles and schemas for the independent propensity baseline."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from Application.model.customer_propensity_dataset import feature_schema

FEATURES = tuple(sorted(feature_schema()))
SEMANTIC_NULL = tuple(sorted((
    "average_days_between_orders_365d", "days_since_favorite_90d",
    "days_since_last_order_365d", "days_since_view_180d",
)))
COLUMNS = FEATURES + tuple(f"unavailable__{n}" for n in FEATURES) + tuple(
    f"semantic_null__{n}" for n in SEMANTIC_NULL)
CONTINUOUS = tuple(i for i, n in enumerate(FEATURES) if not n.startswith("has_"))
PREPROCESSING_VERSION = "propensity-22-to-48-v1"


class MLDataError(ValueError):
    """Static, anonymous failure category; never include a row or customer key."""


class MLCancelled(Exception):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_json(path, max_bytes=4 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > max_bytes:
        raise MLDataError("INVALID_JSON_FILE")
    try:
        return json.loads(path.read_text("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeError, ValueError):
        raise MLDataError("INVALID_JSON_FILE") from None


def write_json(path, value):
    """Exclusive creation: existing user artifacts are never overwritten."""
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def dataset_manifest(directory, purpose="training"):
    directory = Path(directory)
    m = load_json(directory / "manifest.json")
    if (m.get("schema_version") != 2 or m.get("features") != feature_schema()
            or m.get("target") != "order_placed_within_30_days" or m.get("horizon_days") != 30
            or m.get("purpose") != purpose or m.get("status") != "RESEARCH_DATASET_ONLY"
            or m.get("as_of_level") != "RETROSPECTIVE_EVENT_TIME"
            or m.get("business_validation") != "REQUIRES_BUSINESS_VALIDATION"
            or m.get("probe") is not None or m.get("storage_format") != "jsonl.gz"):
        raise MLDataError("INCOMPATIBLE_DATASET")
    for name, size in m["file_sizes"].items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or path.is_symlink() or path.stat().st_size != size:
            raise MLDataError("DATASET_FILE_SIZE_MISMATCH")
    return m


def temporal_roles(manifest):
    """No silent adaptation of the agreed 33/11/11 grid or embargo boundaries."""
    parse = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))
    dates = manifest["prediction_dates"]
    if len(dates) != 55 or dates != sorted(set(dates)):
        raise MLDataError("TEMPORAL_GRID_MISMATCH")
    groups = {"train": dates[:33], "selection": dates[33:38], "calibration": dates[38:41],
              "holdout": dates[41:44], "test": dates[44:]}
    if (dates[0] != "2022-01-31T00:00:00Z" or dates[-1] != "2026-07-09T00:00:00Z"
            or groups["selection"][0] != manifest["validation_start"]
            or groups["test"][0] != manifest["test_start"]
            or manifest["counts"]["embargo"] != 0 or manifest["counts"]["unknown"] != 0
            or any(parse(b) - parse(a) != timedelta(days=30) for a, b in zip(dates, dates[1:]))):
        raise MLDataError("TEMPORAL_GRID_MISMATCH")
    return groups
