"""Offline transport. One directory rename commits raw parts and their manifest."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import uuid

from .adapters import adapt_customer_merge
from .adapters.customer_contacts import adapt_customer_contact_candidate
from .customers_stream import iter_json_records
from .customer_profile_snapshot import CustomerProfileSnapshot
from .daily_training_batch import (
    BatchComponent, ChunkedTrainingBatch, _atomic_write, load_chunked_training_batch,
)
from .identity import CustomerIdResolver
from .raw_reader import EXPORT_ROOTS, iter_export, _unique_object
from .selection import DEFAULT_SELECTION, MindboxSelectionConfig
from .training_batch import TrainingBatchExport, TrainingBatchWindow


class ManualImportError(ValueError):
    """Only static categories, never source values or identities."""


def check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Manual import cancelled")


def select_merges_source(raw_root, window=None):
    """Latest compatible complete API batch; broken final manifests block selection."""
    root = Path(raw_root).resolve()
    candidates = []
    try:
        for path in (root / "training_batches").glob("*/manifest.json"):
            if path.parent.name.startswith("."):
                continue
            # Header only; the loader performs strict metadata and part validation.
            with path.open(encoding="utf-8") as stream:
                header = json.load(stream, object_pairs_hook=_unique_object)
            if header.get("source_kind", "API") == "MANUAL":
                continue
            batch = load_chunked_training_batch(path, raw_root=root, require_complete=True, api_only=True)
            if window is None or (batch.window.merge_since <= window.merge_since
                                  and batch.window.interaction_until >= window.interaction_until):
                candidates.append(batch)
    except Exception:
        raise ManualImportError("Сохранённый API manifest повреждён; проверьте историю объединений клиентов.") from None
    if not candidates:
        raise ManualImportError("Нет завершённого API snapshot объединений клиентов с подходящим покрытием.")
    return max(candidates, key=lambda batch: (batch.created_at_utc, batch.batch_id))


def merges_description(source):
    return (f"Объединения клиентов: API snapshot: batch создан {source.created_at_utc:%d.%m.%Y %H:%M UTC}; "
            f"покрытие {source.window.merge_since:%d.%m.%Y %H:%M} — "
            f"{source.window.interaction_until:%d.%m.%Y %H:%M UTC}")


def _resolver(source, root, cancelled=None):
    def records():
        for raw in iter_export("customer_merges", input_dir=root / source.components[0].export.relative_directory):
            check_cancel(cancelled)
            yield adapt_customer_merge(raw)
    check_cancel(cancelled)
    return CustomerIdResolver(records())


def _copy_validate(source, staging, name, *, cancelled=None, progress=None, resolver=None):
    """Copy bytes in bounded chunks, validate one record at a time before commit."""
    directory = staging / name
    directory.mkdir()
    target = directory / f"{name}_part_001.json"
    try:
        with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
            total = os.fstat(incoming.fileno()).st_size
            copied = 0
            while True:
                check_cancel(cancelled)
                chunk = incoming.read(1024 * 1024)
                if not chunk:
                    break
                outgoing.write(chunk)
                copied += len(chunk)
                if progress:
                    progress(f"Копирование {name}: {copied} / {total} bytes")
            outgoing.flush()
            os.fsync(outgoing.fileno())
        count = 0
        for raw in iter_json_records(target, EXPORT_ROOTS[name]):
            check_cancel(cancelled)
            if resolver is not None:
                adapt_customer_contact_candidate(raw, resolver)
            count += 1
            if progress and count % 10000 == 0:
                progress(f"Проверка {name}: {count} записей")
        if progress:
            progress(f"Проверка {name} завершена: {count} записей")
    except InterruptedError:
        raise
    except Exception:
        raise ManualImportError(f"Не удалось импортировать {name}: проверьте JSON envelope и структуру записей.") from None


def import_interactions(actions, orders, *, raw_root, window, selection=DEFAULT_SELECTION,
                        cancelled=None, progress=None):
    root = Path(raw_root).resolve()
    if not isinstance(selection, MindboxSelectionConfig) or not isinstance(window, TrainingBatchWindow):
        raise ManualImportError("Требуются корректные период и правила отбора.")
    source = select_merges_source(root, window)
    _resolver(source, root, cancelled)  # Validate the saved raw merges, not just their metadata.
    if progress:
        progress(merges_description(source))
    batch_id = uuid.uuid4().hex
    parent = root / "training_batches"
    parent.mkdir(parents=True, exist_ok=True)
    components = [source.components[0]]
    with tempfile.TemporaryDirectory(prefix=".manual-", dir=parent) as temporary:
        staging = Path(temporary)
        for name, path in (("actions", actions), ("orders", orders)):
            _copy_validate(path, staging, name, cancelled=cancelled, progress=progress)
            entry = TrainingBatchExport(name, None, "MANUAL", f"training_batches/{batch_id}/{name}", 1, "MANUAL")
            components.append(BatchComponent(name, window.interaction_since, window.interaction_until, "MANUAL", "READY", entry))
        batch = ChunkedTrainingBatch(batch_id, datetime.now(timezone.utc), window, source.config_fingerprint,
                                     tuple(components), True, selection, "MANUAL", source.batch_id)
        _atomic_write(staging / "state.json", batch)
        _atomic_write(staging / "manifest.json", batch)
        check_cancel(cancelled)
        os.replace(staging, parent / batch_id)
    return batch


def import_customers(customers, *, raw_root, cancelled=None, progress=None):
    root = Path(raw_root).resolve()
    source = select_merges_source(root)
    resolver = _resolver(source, root, cancelled)
    if progress:
        progress(merges_description(source))
    snapshot_id = uuid.uuid4().hex
    parent = root / "customer_profile_snapshots"
    parent.mkdir(parents=True, exist_ok=True)
    merges = source.components[0].export
    with tempfile.TemporaryDirectory(prefix=".manual-", dir=parent) as temporary:
        staging = Path(temporary)
        _copy_validate(customers, staging, "customers", cancelled=cancelled, progress=progress, resolver=resolver)
        snapshot = CustomerProfileSnapshot(
            snapshot_id, datetime.now(timezone.utc).isoformat(), f"customer_profile_snapshots/{snapshot_id}/customers", 1,
            merges.relative_directory, merges.parts_count, schema_version=2, source_kind="MANUAL",
            merge_source_training_batch_id=source.batch_id,
        )
        with (staging / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(asdict(snapshot), stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        check_cancel(cancelled)
        os.replace(staging, parent / snapshot_id)
    return snapshot
