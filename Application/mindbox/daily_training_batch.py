"""Daily sequential exports with durable component checkpoints; no training."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from .raw_reader import part_files
from .storage import RawExportStorage
from .training_batch import TrainingBatchError, TrainingBatchExport, TrainingBatchWindow, _directory, _utc


@dataclass(frozen=True)
class DailyWindow:
    since: datetime
    until: datetime


def split_daily_windows(interaction_since: datetime, interaction_until: datetime) -> tuple[DailyWindow, ...]:
    since, until = _utc(interaction_since), _utc(interaction_until)
    if since >= until or any((value.hour, value.minute, value.second, value.microsecond) != (0, 0, 0, 0)
                             for value in (since, until)):
        raise TrainingBatchError("Daily boundaries must be ordered UTC midnights")
    return tuple(DailyWindow(since + timedelta(days=index), since + timedelta(days=index + 1))
                 for index in range((until - since).days))


@dataclass(frozen=True, repr=False)
class BatchComponent:
    name: str
    since: datetime
    until: datetime
    operation: str
    status: str = "PENDING"
    export: TrainingBatchExport | None = None


@dataclass(frozen=True, repr=False)
class ChunkedTrainingBatch:
    batch_id: str
    created_at_utc: datetime
    window: TrainingBatchWindow
    config_fingerprint: str
    components: tuple[BatchComponent, ...]
    transport_complete: bool = False

    @property
    def diagnostics(self):
        return {
            "days_total": (len(self.components) - 1) // 2,
            "days_ready": sum(all(c.status == "READY" for c in self.components[i:i + 2])
                              for i in range(1, len(self.components), 2)),
            "components_ready": sum(c.status == "READY" for c in self.components),
            "components_failed": sum(c.status == "FAILED" for c in self.components),
        }


class ChunkedBatchError(TrainingBatchError):
    def __init__(self, state_path):
        self.state_path = Path(state_path)
        super().__init__("Daily export interrupted; resume using state path")


def _fingerprint(config):
    # Bind endpoint + operation config, never credentials. Secret rotation is allowed.
    text = json.dumps([config.api_url, config.endpoint_id,
                       {name: config.operations[name] for name in ("customer_merges", "actions", "orders")}], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _batch_directory(root, batch_id):
    if not isinstance(batch_id, str) or uuid.UUID(batch_id).hex != batch_id:
        raise TrainingBatchError("Invalid batch ID")
    path = (root / "training_batches" / batch_id).resolve()
    if not path.is_relative_to(root):
        raise TrainingBatchError("Batch path escapes raw root")
    return path


def _document(batch):
    return {"schema_version": 2, "batch_id": batch.batch_id,
            "created_at_utc": batch.created_at_utc.isoformat(),
            "window": {k: v.isoformat() for k, v in asdict(batch.window).items()},
            "config_fingerprint": batch.config_fingerprint, "transport_complete": batch.transport_complete,
            "components": [{"name": c.name, "since": c.since.isoformat(), "until": c.until.isoformat(),
                            "operation": c.operation, "status": c.status,
                            "export": asdict(c.export) if c.export else None} for c in batch.components]}


def _atomic_write(path, batch):
    fd, temporary = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_document(batch), stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _writer_lock(directory):
    # OS lock is automatically released on process death; file existence is not a lock.
    with (directory / ".writer.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise TrainingBatchError("Another writer owns this batch") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_chunked_training_batch(batch, *, raw_root, require_complete=False):
    root = Path(raw_root).resolve()
    _batch_directory(root, batch.batch_id)
    _utc(batch.created_at_utc)
    days = split_daily_windows(batch.window.interaction_since, batch.window.interaction_until)
    TrainingBatchWindow(**asdict(batch.window))
    expected = [("customer_merges", batch.window.merge_since, batch.window.interaction_until)]
    expected += [(name, day.since, day.until) for day in days for name in ("actions", "orders")]
    if (len(batch.components) != len(expected) or type(batch.transport_complete) is not bool
            or not isinstance(batch.config_fingerprint, str) or len(batch.config_fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in batch.config_fingerprint)):
        raise TrainingBatchError("Invalid daily batch metadata")
    paths = set()
    incomplete_seen = False
    for component, identity in zip(batch.components, expected):
        if ((component.name, component.since, component.until) != identity
                or component.status not in ("PENDING", "FAILED", "READY")
                or not isinstance(component.operation, str) or not component.operation.strip()):
            raise TrainingBatchError("Invalid component order/window/status")
        if component.status != "READY":
            incomplete_seen = True
            if component.export is not None:
                raise TrainingBatchError("Unfinished component must not reference an export")
            continue
        if incomplete_seen:
            raise TrainingBatchError("READY components must form a sequential prefix")
        entry = component.export
        if (not isinstance(entry, TrainingBatchExport) or entry.name != component.name or entry.operation != component.operation
                or not isinstance(entry.export_id, str) or not entry.export_id.strip()
                or type(entry.parts_count) is not int or entry.parts_count < 1):
            raise TrainingBatchError("Invalid READY export metadata")
        directory = _directory(root, entry)
        if directory in paths or len(part_files(directory, entry.name)) != entry.parts_count:
            raise TrainingBatchError("Duplicate/missing export parts")
        paths.add(directory)
    if (batch.transport_complete or require_complete) and incomplete_seen:
        raise TrainingBatchError("Incomplete transport batch")
    if require_complete and not batch.transport_complete:
        raise TrainingBatchError("Final manifest required")


def load_chunked_training_batch(path, *, raw_root, require_complete=False):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TrainingBatchError("Duplicate state key")
            result[key] = value
        return result

    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=unique)
        if (set(data) != {"schema_version", "batch_id", "created_at_utc", "window", "config_fingerprint",
                          "components", "transport_complete"} or type(data["schema_version"]) is not int
                or data["schema_version"] != 2):
            raise TrainingBatchError("Invalid daily state schema")
        components = []
        for c in data["components"]:
            if set(c) != {"name", "since", "until", "operation", "status", "export"}:
                raise TrainingBatchError("Invalid component fields")
            components.append(BatchComponent(c["name"], datetime.fromisoformat(c["since"]),
                datetime.fromisoformat(c["until"]), c["operation"], c["status"],
                TrainingBatchExport(**c["export"]) if c["export"] is not None else None))
        batch = ChunkedTrainingBatch(data["batch_id"], datetime.fromisoformat(data["created_at_utc"]),
            TrainingBatchWindow(**{k: datetime.fromisoformat(v) for k, v in data["window"].items()}),
            data["config_fingerprint"], tuple(components), data["transport_complete"])
        directory = _batch_directory(Path(raw_root).resolve(), batch.batch_id)
        if Path(path).resolve() not in (directory / "state.json", directory / "manifest.json"):
            raise TrainingBatchError("State path does not match batch identity")
        validate_chunked_training_batch(batch, raw_root=raw_root, require_complete=require_complete)
        if require_complete and Path(path).name != "manifest.json":
            raise TrainingBatchError("Final manifest required")
        return batch
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise TrainingBatchError("Cannot load valid daily state/manifest") from None


def create_chunked_training_batch(client, *, raw_root, window, poll_interval=5.0, timeout=600.0):
    days = split_daily_windows(window.interaction_since, window.interaction_until)
    # Validate the full window and polling config before writing state or starting API calls.
    TrainingBatchWindow(**asdict(window))
    _poll_settings(poll_interval, timeout)
    components = [BatchComponent("customer_merges", window.merge_since, window.interaction_until,
                                 client.config.operations["customer_merges"])]
    components += [BatchComponent(name, day.since, day.until, client.config.operations[name])
                   for day in days for name in ("actions", "orders")]
    batch = ChunkedTrainingBatch(uuid.uuid4().hex, datetime.now(timezone.utc), window,
                                _fingerprint(client.config), tuple(components))
    validate_chunked_training_batch(batch, raw_root=raw_root)
    directory = _batch_directory(Path(raw_root).resolve(), batch.batch_id)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    state = directory / "state.json"
    try:
        _atomic_write(state, batch)
        return resume_chunked_training_batch(client, state_path=state, raw_root=raw_root,
                                             poll_interval=poll_interval, timeout=timeout)
    except (Exception, KeyboardInterrupt):
        raise ChunkedBatchError(state) from None


def _poll_settings(poll_interval, timeout):
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (poll_interval, timeout)):
        raise TrainingBatchError("Positive finite polling settings required")


def resume_chunked_training_batch(client, *, state_path, raw_root, poll_interval=5.0, timeout=600.0):
    state = Path(state_path).resolve()
    _poll_settings(poll_interval, timeout)
    batch = load_chunked_training_batch(state, raw_root=raw_root)
    if state.name != "state.json":
        raise TrainingBatchError("Resume requires state.json")
    with _writer_lock(state.parent):
        batch = load_chunked_training_batch(state, raw_root=raw_root)
        if (_fingerprint(client.config) != batch.config_fingerprint or any(
                c.operation != client.config.operations[c.name] for c in batch.components)):
            raise TrainingBatchError("Endpoint/operation config differs from saved batch")
        manifest = state.parent / "manifest.json"
        if manifest.exists():
            final = load_chunked_training_batch(manifest, raw_root=raw_root, require_complete=True)
            if replace(batch, transport_complete=True) != final:
                raise TrainingBatchError("State differs from final manifest")
            return final
        root = Path(raw_root).resolve()
        storage = RawExportStorage(root)
        try:
            for index, component in enumerate(batch.components):
                if component.status == "READY":
                    continue
                try:
                    payload = {"sinceDateTimeUtc": component.since.strftime("%Y-%m-%d %H:%M"),
                               "tillDateTimeUtc": component.until.strftime("%Y-%m-%d %H:%M")}
                    export_id = client.start_export(component.operation, payload)
                    urls = client.wait_for_export(component.operation, export_id, poll_interval=poll_interval, timeout=timeout)
                    parts = client.download_export(component.name, urls, storage=storage)
                    if not parts or len({Path(p).resolve().parent for p in parts}) != 1:
                        raise TrainingBatchError("Invalid published parts")
                    relative = Path(parts[0]).resolve().parent.relative_to(root).as_posix()
                    entry = TrainingBatchExport(component.name, export_id, component.operation, relative, len(parts))
                    updated = replace(component, status="READY", export=entry)
                    candidate = replace(batch, components=batch.components[:index] + (updated,) + batch.components[index + 1:])
                    validate_chunked_training_batch(candidate, raw_root=root)
                    _atomic_write(state, candidate)
                    batch = candidate
                except (Exception, KeyboardInterrupt):
                    failed = replace(component, status="FAILED", export=None)
                    batch = replace(batch, components=batch.components[:index] + (failed,) + batch.components[index + 1:])
                    _atomic_write(state, batch)
                    raise
            final = replace(batch, transport_complete=True)
            validate_chunked_training_batch(final, raw_root=root, require_complete=True)
            _atomic_write(manifest, final)
            # Keep READY state as audit; manifest is the final commit marker.
            return final
        except (Exception, KeyboardInterrupt):
            raise ChunkedBatchError(state) from None


def prepare_training_data_from_chunked_batch(batch, *, raw_root, catalog_path, train_config, diagnose=False):
    from Application.model.mindbox_training_preparation import _prepare_training_data_from_mindbox_sources

    validate_chunked_training_batch(batch, raw_root=raw_root, require_complete=True)
    root = Path(raw_root).resolve()
    def directories(name):
        return tuple(_directory(root, c.export) for c in batch.components if c.name == name)
    return _prepare_training_data_from_mindbox_sources(actions_export_dirs=directories("actions"),
        orders_export_dirs=directories("orders"), customer_merges_export_dir=directories("customer_merges")[0],
        catalog_path=catalog_path, train_config=train_config, diagnose=diagnose)
