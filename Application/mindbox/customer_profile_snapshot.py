"""Metadata-only snapshots; Customers raw remains in the M01 raw store."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid

from Application.customer_profiles import build_customer_contact_index
from .adapters import adapt_customer, adapt_customer_merge
from .daily_training_batch import load_chunked_training_batch
from .identity import CustomerIdResolver
from .raw_reader import part_files, iter_export
from .storage import RawExportStorage
from .training_batch import TrainingBatchExport, _directory


class ProfileSnapshotError(ValueError):
    """Safe snapshot error without payload, identities or paths."""


@dataclass(frozen=True, repr=False)
class CustomerProfileSnapshot:
    snapshot_id: str
    created_at: str
    customers_directory: str
    customers_parts: int
    customer_merges_directory: str
    customer_merges_parts: int
    originating_training_batch_id: str | None = None
    transport_complete: bool = True
    schema_version: int = 1


def _uuid(value):
    try:
        valid = isinstance(value, str) and uuid.UUID(value).hex == value
    except ValueError:
        valid = False
    if not valid:
        raise ProfileSnapshotError("Invalid snapshot/batch identity")


def _export_directory(root, name, directory, count):
    if type(count) is not int or count < 1:
        raise ProfileSnapshotError("Invalid snapshot parts count")
    result = _directory(root, TrainingBatchExport(name, "metadata", "metadata", directory, count))
    if len(part_files(result, name)) != count:
        raise ProfileSnapshotError("Snapshot parts mismatch")
    return result


def snapshot_manifest_path(raw_root, snapshot_id):
    _uuid(snapshot_id)
    root = Path(raw_root).resolve()
    path = (root / "customer_profile_snapshots" / snapshot_id / "manifest.json").resolve()
    if not path.is_relative_to(root):
        raise ProfileSnapshotError("Snapshot path escapes raw root")
    return path


def validate_customer_profile_snapshot(snapshot, *, raw_root):
    """Read directory listings and manifests only, never Customers raw contents."""
    try:
        if (not isinstance(snapshot, CustomerProfileSnapshot) or type(snapshot.schema_version) is not int
                or snapshot.schema_version != 1 or snapshot.transport_complete is not True):
            raise ProfileSnapshotError("Invalid snapshot schema/completeness")
        _uuid(snapshot.snapshot_id)
        stamp = datetime.fromisoformat(snapshot.created_at)
        if stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ProfileSnapshotError("Snapshot timestamp requires timezone")
        root = Path(raw_root).resolve()
        _export_directory(root, "customers", snapshot.customers_directory, snapshot.customers_parts)
        _export_directory(root, "customer_merges", snapshot.customer_merges_directory, snapshot.customer_merges_parts)
        if snapshot.originating_training_batch_id is not None:
            _uuid(snapshot.originating_training_batch_id)
            batch = load_chunked_training_batch(root / "training_batches" / snapshot.originating_training_batch_id / "manifest.json",
                                               raw_root=root, require_complete=True)
            merges = next(c.export for c in batch.components if c.name == "customer_merges")
            if (merges.relative_directory, merges.parts_count) != (snapshot.customer_merges_directory, snapshot.customer_merges_parts):
                raise ProfileSnapshotError("Snapshot merge source differs from originating batch")
    except Exception:
        raise ProfileSnapshotError("Invalid customer profile snapshot metadata") from None


def _publish_snapshot(snapshot, raw_root):
    validate_customer_profile_snapshot(snapshot, raw_root=raw_root)
    path = snapshot_manifest_path(raw_root, snapshot.snapshot_id)
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name("manifest.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(asdict(snapshot), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def create_customer_profile_snapshot(client, *, training_manifest, raw_root, timeout=3600., poll_interval=5.):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (timeout, poll_interval)):
        raise ProfileSnapshotError("Positive finite polling settings required")
    root = Path(raw_root).resolve()
    batch = load_chunked_training_batch(training_manifest, raw_root=root, require_complete=True)
    merges = next(c.export for c in batch.components if c.name == "customer_merges")
    parts = client.export("customers", storage=RawExportStorage(root), timeout=timeout, poll_interval=poll_interval)
    if not parts or len({Path(p).resolve().parent for p in parts}) != 1:
        raise ProfileSnapshotError("Invalid published Customers parts")
    try:
        directory = Path(parts[0]).resolve().parent.relative_to(root).as_posix()
    except ValueError:
        raise ProfileSnapshotError("Customers export outside raw root") from None
    snapshot = CustomerProfileSnapshot(uuid.uuid4().hex, datetime.now(timezone.utc).isoformat(), directory, len(parts),
                                       merges.relative_directory, merges.parts_count, batch.batch_id)
    _publish_snapshot(snapshot, root)
    return snapshot


def load_customer_profile_snapshot(manifest, *, raw_root):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ProfileSnapshotError("Duplicate snapshot metadata field")
            result[key] = value
        return result
    try:
        with Path(manifest).open(encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique)
        if not isinstance(data, dict) or set(data) != set(CustomerProfileSnapshot.__dataclass_fields__):
            raise ProfileSnapshotError("Invalid snapshot fields")
        snapshot = CustomerProfileSnapshot(**data)
        if Path(manifest).resolve() != snapshot_manifest_path(raw_root, snapshot.snapshot_id):
            raise ProfileSnapshotError("Snapshot location does not match identity")
        validate_customer_profile_snapshot(snapshot, raw_root=raw_root)
        return snapshot
    except Exception:
        raise ProfileSnapshotError("Cannot load valid customer profile snapshot") from None


def load_customer_contact_index(profile_manifest, model_mappings=None, *, raw_root):
    snapshot = load_customer_profile_snapshot(profile_manifest, raw_root=raw_root)
    root = Path(raw_root).resolve()
    resolver = CustomerIdResolver(adapt_customer_merge(raw) for raw in
        iter_export("customer_merges", input_dir=root / snapshot.customer_merges_directory))
    if model_mappings is None:
        user_ids = None
    elif isinstance(model_mappings, dict):
        user_ids = model_mappings["idx2user"]
    else:
        user_ids = model_mappings.idx2user
    records = (adapt_customer(raw, resolver) for raw in iter_export("customers", input_dir=root / snapshot.customers_directory))
    return build_customer_contact_index(records, user_ids)
