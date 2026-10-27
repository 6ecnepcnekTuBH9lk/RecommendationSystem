"""Versioned streaming JSONL storage; ML schema v2 is independent of compression."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import zlib

from Application.model.customer_propensity_dataset import (
    AS_OF_LEVEL, HORIZON, SCHEMA_VERSION, TARGET, PropensityFeatureRow, PropensityInferenceRow,
    PropensityPlan, feature_schema, iso, utc_day,
)
from Application.model.customer_propensity_history import PropensityCancelled, PropensityDataError

FILES = ("train", "validation", "test", "unknown", "embargo", "inference")


@dataclass(frozen=True, slots=True)
class PropensityStorage:
    format: str = "jsonl.gz"
    level: int = 6

    def __post_init__(self):
        if self.format not in ("jsonl", "jsonl.gz") or type(self.level) is not int or not 0 <= self.level <= 9:
            raise ValueError("Invalid propensity storage configuration")

    def filename(self, partition):
        if partition not in FILES:
            raise ValueError("Invalid propensity partition")
        return partition + "." + self.format

    def metadata(self):
        return {"storage_contract_version": 1, "storage_format": self.format,
                "compression": ({"codec": "gzip", "level": self.level, "mtime": 0, "filename": ""}
                                if self.format == "jsonl.gz" else {"codec": "none"}),
                "file_names": {name: self.filename(name) for name in FILES}}


def storage_from_manifest(manifest):
    # Existing v2 manifests predate storage metadata and always mean plain JSONL.
    keys = {"storage_contract_version", "storage_format", "compression", "file_names", "file_sizes"}
    if not keys.intersection(manifest):
        return PropensityStorage("jsonl")
    try:
        storage = PropensityStorage(manifest["storage_format"], manifest["compression"].get("level", 6))
        expected = storage.metadata()
        if (not keys <= manifest.keys() or type(manifest["storage_contract_version"]) is not int
                or any(manifest[k] != v for k, v in expected.items())):
            raise ValueError
        sizes = manifest["file_sizes"]
        if set(sizes) != set(expected["file_names"].values()) or any(type(v) is not int or v < 0 for v in sizes.values()):
            raise ValueError
        return storage
    except (KeyError, TypeError, AttributeError, ValueError):
        raise PropensityDataError("INVALID_STORAGE_MANIFEST") from None


class _CheckedFile:
    def __init__(self, stream, before_write):
        self.stream, self.before_write = stream, before_write

    def write(self, data):
        self.before_write(len(data))
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def open_writer(path, storage, before_write=lambda size: None):
    """Close gzip trailer before fsync of the owned physical file. No raw copy."""
    with Path(path).open("wb") as physical:
        checked = _CheckedFile(physical, before_write)
        compressed = (gzip.GzipFile(filename="", fileobj=checked, mode="wb", compresslevel=storage.level, mtime=0)
                      if storage.format == "jsonl.gz" else checked)
        text = io.TextIOWrapper(compressed, encoding="utf-8", newline="\n")
        try:
            yield text
        finally:
            try:
                text.flush()
            finally:
                if storage.format == "jsonl.gz":
                    text.close()
                else:
                    text.detach()
            physical.flush()
            os.fsync(physical.fileno())


@contextmanager
def open_reader(path, storage):
    try:
        with (gzip.open(path, "rt", encoding="utf-8", newline="") if storage.format == "jsonl.gz"
              else Path(path).open(encoding="utf-8", newline="")) as stream:
            yield stream
    except (EOFError, gzip.BadGzipFile, zlib.error, UnicodeError, json.JSONDecodeError):
        raise PropensityDataError("INVALID_OUTPUT_ENCODING") from None


def digest(path, cancelled=lambda: False):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if cancelled():
                raise PropensityCancelled
            result.update(block)
    return result.hexdigest()


def _validate_row(row, name, plan, features, dates):
    fields = set((PropensityInferenceRow if name == "inference" else PropensityFeatureRow).__dataclass_fields__)
    if (not isinstance(row, dict) or set(row) != fields
            or not isinstance(row.get("features"), dict) or set(row["features"]) != set(features)
            or not isinstance(row.get("available"), dict) or set(row["available"]) != set(features)
            or not isinstance(row.get("feature_issues"), dict)
            or not isinstance(row.get("customer_key"), str) or len(row["customer_key"]) != 64
            or any(c not in "0123456789abcdef" for c in row["customer_key"])):
        raise PropensityDataError("INVALID_OUTPUT_SCHEMA")
    if name in ("train", "validation", "test") and (
            row["split"] != name or type(row["target"]) is not int or row["target"] not in (0, 1)):
        raise PropensityDataError("INVALID_TRAINING_LABEL")
    if not isinstance(row["prediction_utc"], str):
        raise PropensityDataError("INVALID_OUTPUT_WINDOW")
    date = dates.get(row["prediction_utc"])
    if (date is None or row["split"] != plan.split(date)
            or row["target_until_utc"] != iso(date + timedelta(days=HORIZON))):
        raise PropensityDataError("INVALID_OUTPUT_WINDOW")
    if (set(row["feature_issues"]) != {k for k, v in row["available"].items() if not v}
            or any(not isinstance(v, list) or not v or any(not isinstance(x, str) or not x for x in v)
                   for v in row["feature_issues"].values())):
        raise PropensityDataError("INVALID_OUTPUT_STATE")
    if name == "inference":
        if plan.purpose != "inference" or row["split"] != "inference":
            raise PropensityDataError("INVALID_OUTPUT_PARTITION")
    else:
        target = row["target"]
        state = "UNKNOWN_LABEL" if target is None else "POSITIVE" if target == 1 else "NEGATIVE"
        if (not isinstance(row["label_reasons"], list)
                or any(not isinstance(v, str) or not v for v in row["label_reasons"])
                or (target is not None and (type(target) is not int or target not in (0, 1)))
                or row["label_state"] != state or bool(row["label_reasons"]) != (target is None)):
            raise PropensityDataError("INVALID_OUTPUT_STATE")
        if ((name == "unknown" and target is not None) or (name == "embargo" and row["split"] != "embargo")
                or plan.purpose != "training"):
            raise PropensityDataError("INVALID_OUTPUT_PARTITION")
    for feature, spec in features.items():
        value, available = row["features"][feature], row["available"][feature]
        if type(available) is not bool or (not available and value is not None):
            raise PropensityDataError("INVALID_OUTPUT_AVAILABILITY")
        if value is None:
            if available and not (feature.startswith("days_since_") or feature == "average_days_between_orders_365d"):
                raise PropensityDataError("INVALID_OUTPUT_NULL")
            continue
        if spec["type"] == "nullable_integer":
            if type(value) is not int or value < 0 or (feature.startswith("has_") and value not in (0, 1)):
                raise PropensityDataError("INVALID_OUTPUT_NUMBER")
        else:
            try:
                valid = isinstance(value, str) and Decimal(value).is_finite() and Decimal(value) >= 0
            except InvalidOperation:
                valid = False
            if not valid:
                raise PropensityDataError("INVALID_OUTPUT_DECIMAL")


def iter_validated_file(path, name, plan, expected, storage, cancelled=lambda: False):
    features = feature_schema()
    dates = {iso(date): date for date in plan.dates}
    count = 0
    with open_reader(path, storage) as stream:
        for line in stream:
            if cancelled():
                raise PropensityCancelled
            if not line.endswith("\n"):
                raise PropensityDataError("INVALID_OUTPUT_LINE")
            row = json.loads(line)
            _validate_row(row, name, plan, features, dates)
            count += 1
            if count > expected:
                raise PropensityDataError("OUTPUT_COUNT_MISMATCH")
            yield row
    if count != expected:
        raise PropensityDataError("OUTPUT_COUNT_MISMATCH")


def iter_propensity_rows(directory, partition, *, purpose, allow_probe=False, cancelled=lambda: False,
                         source_version=None, generation=None):
    """Read a v2 partition with physical SHA256 and streaming semantic validation.

    Consume/exhaust the iterator (or close it) to release the file. Full consumption
    also checks gzip CRC/trailer and final row count. Probe artifacts require opt-in.
    Legacy v2 JSONL manifests are accepted; schema v1 is never reinterpreted as v2.
    """
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION
                or manifest["target"] != TARGET or type(manifest["horizon_days"]) is not int
                or manifest["horizon_days"] != HORIZON or purpose not in ("training", "inference")
                or manifest["purpose"] != purpose or manifest["as_of_level"] != AS_OF_LEVEL
                or manifest["features"] != feature_schema()
                or type(manifest["label_fields_present"]) is not bool
                or manifest["label_fields_present"] != (purpose == "training")
                or manifest["status"] not in ("RESEARCH_DATASET_ONLY", "BLOCKED_BY_DATA_QUALITY", "RESEARCH_PROBE_ONLY")
                or ((manifest["status"] == "RESEARCH_PROBE_ONLY" or manifest.get("probe")) and not allow_probe)
                or (source_version is not None and manifest["source_version"] != source_version)
                or (generation is not None and manifest["generation"] != generation)):
            raise ValueError
        storage = storage_from_manifest(manifest)
        filename = storage.filename(partition)
        checksums = manifest["checksums"]
        if (set(checksums) != {storage.filename(n) for n in FILES}
                or any(not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                       for v in checksums.values())):
            raise ValueError
        file_counts = manifest["counts"]["files"]
        expected = file_counts[partition]
        if (set(file_counts) != set(FILES) or any(type(v) is not int or v < 0 for v in file_counts.values())
                or type(manifest["counts"]["rows"]) is not int or sum(file_counts.values()) != manifest["counts"]["rows"]):
            raise ValueError
        parse = lambda s: utc_day(datetime.fromisoformat(s.replace("Z", "+00:00")))
        dates = tuple(parse(v) for v in manifest["prediction_dates"])
        if not dates or tuple(sorted(set(dates))) != dates:
            raise ValueError
        plan = PropensityPlan(dates, parse(manifest["validation_start"]), parse(manifest["test_start"]), purpose)
    except (KeyError, TypeError, ValueError, AttributeError):
        raise PropensityDataError("INVALID_DATASET_MANIFEST") from None
    path = directory / filename
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise PropensityDataError("INVALID_DATASET_FILE")
    if "file_sizes" in manifest and path.stat().st_size != manifest["file_sizes"][filename]:
        raise PropensityDataError("OUTPUT_SIZE_MISMATCH")
    if digest(path, cancelled) != checksums[filename]:
        raise PropensityDataError("OUTPUT_CHECKSUM_MISMATCH")
    yield from iter_validated_file(path, partition, plan, expected, storage, cancelled)
