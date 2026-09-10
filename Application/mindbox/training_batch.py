"""Coordinated export metadata. Raw publication remains owned by M01 storage."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import uuid

from .raw_reader import part_files, TIMESTAMP_PATTERN
from .storage import RawExportStorage


EXPORTS = ("customer_merges", "actions", "orders")


class TrainingBatchError(ValueError):
    """Safe batch metadata error; no transport response or raw values."""


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TrainingBatchError("Timezone-aware UTC timestamps required")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class TrainingBatchWindow:
    interaction_since: datetime
    interaction_until: datetime
    merge_since: datetime

    def __post_init__(self):
        for name in ("interaction_since", "interaction_until", "merge_since"):
            value = _utc(getattr(self, name))
            # M01 CLI transport contract uses minute precision; never silently truncate.
            if value.second or value.microsecond:
                raise TrainingBatchError("Window timestamps require minute precision")
            object.__setattr__(self, name, value)
        if not self.merge_since <= self.interaction_since < self.interaction_until:
            raise TrainingBatchError("Require merge_since <= interaction_since < interaction_until")


@dataclass(frozen=True, repr=False)
class TrainingBatchExport:
    name: str
    export_id: str
    operation: str
    relative_directory: str
    parts_count: int


@dataclass(frozen=True, repr=False)
class MindboxTrainingBatch:
    batch_id: str
    created_at_utc: datetime
    window: TrainingBatchWindow
    exports: tuple[TrainingBatchExport, ...]
    schema_version: int = 1


def _directory(root: Path, entry: TrainingBatchExport) -> Path:
    value = entry.relative_directory
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise TrainingBatchError("Invalid relative export path")
    relative = PurePosixPath(value)
    if (relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != entry.name
            or not TIMESTAMP_PATTERN.fullmatch(relative.parts[1]) or relative.as_posix() != value):
        raise TrainingBatchError("Expected export-type/timestamp relative directory")
    directory = (root / value).resolve()
    if not directory.is_relative_to(root) or not directory.is_relative_to((root / entry.name).resolve()):
        raise TrainingBatchError("Export path escapes raw root")
    return directory


def validate_training_batch(batch: MindboxTrainingBatch, raw_root: str | Path) -> None:
    if not isinstance(batch, MindboxTrainingBatch) or type(batch.schema_version) is not int or batch.schema_version != 1:
        raise TrainingBatchError("Unsupported batch schema version")
    if not isinstance(batch.batch_id, str) or len(batch.batch_id) != 32:
        raise TrainingBatchError("Invalid batch ID")
    try:
        if uuid.UUID(batch.batch_id).hex != batch.batch_id:
            raise ValueError
    except ValueError:
        raise TrainingBatchError("Invalid batch ID") from None
    _utc(batch.created_at_utc)
    if not isinstance(batch.window, TrainingBatchWindow):
        raise TrainingBatchError("Invalid window")
    TrainingBatchWindow(**asdict(batch.window))
    if (not isinstance(batch.exports, tuple) or len(batch.exports) != 3
            or any(not isinstance(entry, TrainingBatchExport) or not isinstance(entry.name, str) for entry in batch.exports)
            or {entry.name for entry in batch.exports} != set(EXPORTS)):
        raise TrainingBatchError("Exactly three distinct export references required")
    root = Path(raw_root).resolve()
    paths = set()
    for entry in batch.exports:
        if any(not isinstance(value, str) or not value.strip() for value in (entry.export_id, entry.operation)):
            raise TrainingBatchError("Missing export metadata")
        if type(entry.parts_count) is not int or entry.parts_count < 1:
            raise TrainingBatchError("Invalid parts count")
        directory = _directory(root, entry)
        if directory in paths:
            raise TrainingBatchError("Duplicate export directory")
        paths.add(directory)
        if len(part_files(directory, entry.name)) != entry.parts_count:
            raise TrainingBatchError("Part count does not match manifest")


def _manifest(batch: MindboxTrainingBatch) -> dict:
    return {"schema_version": batch.schema_version, "batch_id": batch.batch_id,
            "created_at_utc": _utc(batch.created_at_utc).isoformat(),
            "window": {key: value.isoformat() for key, value in asdict(batch.window).items()},
            "exports": [asdict(entry) for entry in batch.exports]}


def create_training_batch(client, *, raw_root: str | Path, window: TrainingBatchWindow,
                          poll_interval: float = 5.0, timeout: float = 600.0) -> MindboxTrainingBatch:
    if not isinstance(window, TrainingBatchWindow):
        raise TrainingBatchError("TrainingBatchWindow required")
    if any(not math.isfinite(value) or value <= 0 for value in (poll_interval, timeout)):
        raise TrainingBatchError("Positive finite polling settings required")
    operations = {name: client.config.operations[name] for name in EXPORTS}
    root = Path(raw_root).resolve()
    storage = RawExportStorage(root)
    entries = []
    for name in EXPORTS:
        since = window.merge_since if name == "customer_merges" else window.interaction_since
        payload = {"sinceDateTimeUtc": since.strftime("%Y-%m-%d %H:%M"),
                   "tillDateTimeUtc": window.interaction_until.strftime("%Y-%m-%d %H:%M")}
        export_id = client.start_export(operations[name], payload)
        urls = client.wait_for_export(operations[name], export_id, poll_interval=poll_interval, timeout=timeout)
        parts = client.download_export(name, urls, storage=storage)
        if not parts or len({Path(path).resolve().parent for path in parts}) != 1:
            raise TrainingBatchError("Invalid published export parts")
        directory = Path(parts[0]).resolve().parent
        if not directory.is_relative_to(root):
            raise TrainingBatchError("Published export outside raw root")
        entries.append(TrainingBatchExport(name, export_id, operations[name], directory.relative_to(root).as_posix(), len(parts)))
    batch = MindboxTrainingBatch(uuid.uuid4().hex, datetime.now(timezone.utc), window, tuple(entries))
    validate_training_batch(batch, root)
    parent = (root / "training_batches").resolve()
    if not parent.is_relative_to(root):
        raise TrainingBatchError("Batch directory escapes raw root")
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / batch.batch_id
    directory.mkdir()  # exclusive batch ID; never overwrite an existing batch
    temporary = directory / "manifest.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(_manifest(batch), stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / "manifest.json")
    finally:
        if temporary.exists():
            temporary.unlink()
    return batch


def load_training_batch(manifest_path: str | Path, *, raw_root: str | Path) -> MindboxTrainingBatch:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TrainingBatchError("Duplicate manifest key")
            result[key] = value
        return result

    try:
        with Path(manifest_path).open(encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique)
        if set(data) != {"schema_version", "batch_id", "created_at_utc", "window", "exports"}:
            raise TrainingBatchError("Invalid manifest fields")
        batch = MindboxTrainingBatch(data["batch_id"], datetime.fromisoformat(data["created_at_utc"]),
            TrainingBatchWindow(**{key: datetime.fromisoformat(value) for key, value in data["window"].items()}),
            tuple(TrainingBatchExport(**entry) for entry in data["exports"]), data["schema_version"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise TrainingBatchError("Cannot load valid batch manifest") from None
    validate_training_batch(batch, raw_root)
    return batch


def prepare_training_data_from_batch(batch: MindboxTrainingBatch, *, raw_root: str | Path,
                                     catalog_path, train_config, diagnose: bool = False):
    from Application.model.mindbox_training_preparation import prepare_training_data_from_mindbox

    validate_training_batch(batch, raw_root)
    root = Path(raw_root).resolve()
    directories = {entry.name: _directory(root, entry) for entry in batch.exports}
    return prepare_training_data_from_mindbox(
        actions_export_dir=directories["actions"], orders_export_dir=directories["orders"],
        customer_merges_export_dir=directories["customer_merges"], catalog_path=catalog_path,
        train_config=train_config, diagnose=diagnose)
