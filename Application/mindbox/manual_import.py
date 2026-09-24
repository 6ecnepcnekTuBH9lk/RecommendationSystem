"""Offline canonical pair import and explicit legacy snapshot compatibility helpers."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .adapters import adapt_customer_merge
from .adapters.customer_contacts import adapt_customer_contact_candidate
from .customer_profile_snapshot import CustomerProfileSnapshot
from .daily_training_batch import load_chunked_training_batch
from .identity import CustomerIdResolver
from .raw_reader import iter_export, part_files, _unique_object
from .manual_sources import ManualImportError, normalize_sources
from .selection import DEFAULT_SELECTION, MindboxSelectionConfig
from .training_batch import TrainingBatchWindow, TrainingBatchError

VALIDATION_PROGRESS_EVERY = 50_000


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


def _copy_validate(source, staging, name, *, cancelled=None, progress=None, resolver=None, validate=None):
    """Copy bytes in bounded chunks, validate one record at a time before commit."""
    sources = normalize_sources(source)
    directory = staging / name
    directory.mkdir()
    try:
        for number, path in enumerate(sources, 1):
            check_cancel(cancelled)
            if progress:
                progress(
                    f"Копирование {name}: файл {number} из {len(sources)}"
                )
            target = directory / f"{name}_part_{number:03d}.json"
            with path.open("rb") as incoming, target.open("xb") as outgoing:
                while True:
                    check_cancel(cancelled)
                    chunk = incoming.read(1024 * 1024)
                    if not chunk:
                        break
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        check_cancel(cancelled)
        if progress:
            progress(f"Проверка {name} ...")
        if validate is not None:
            validate(directory)
            return len(sources)
        count = 0
        for raw in iter_export(name, input_dir=directory):
            check_cancel(cancelled)
            if resolver is not None:
                adapt_customer_contact_candidate(raw, resolver)
            count += 1
            if (
                    progress
                    and count % VALIDATION_PROGRESS_EVERY == 0
            ):
                progress(
                    f"Проверка {name}: {count} записей"
                )
        if progress:
            progress(
                f"Проверка {name} завершена: {count} записей"
            )
        check_cancel(cancelled)
        return len(sources)
    except InterruptedError:
        raise
    except Exception:
        raise ManualImportError(f"Не удалось импортировать {name}: проверьте JSON envelope и структуру записей.") from None


def import_interactions(actions, orders, *, raw_root, window, selection=DEFAULT_SELECTION,
                        cancelled=None, progress=None):
    """Publish one canonical manual pair. Return the stable training entry path.

    window.merge_since is retained for call compatibility; coverage is checked
    against the current canonical merges using the explicit interaction interval.
    """
    from . import canonical_storage as store
    root = Path(raw_root).resolve()
    if not isinstance(selection, MindboxSelectionConfig) or not isinstance(window, TrainingBatchWindow):
        raise ManualImportError("Требуются корректные период и правила отбора.")
    actions, orders = normalize_sources(actions), normalize_sources(orders)
    since, until = window.interaction_since, window.interaction_until
    with store.storage_lock(root):
        check_cancel(cancelled)
        data = store.catalog(root)
        try:
            merges = store.require_manual_merges(data, since, until)
        except TrainingBatchError as exc:
            raise ManualImportError(str(exc)) from None
        directory = store.checked_directory(root, merges["directory"], "customer_merges")
        def records():
            for raw in iter_export("customer_merges", input_dir=directory):
                check_cancel(cancelled)
                yield adapt_customer_merge(raw)
        try:
            if len(part_files(directory, "customer_merges")) != merges["parts"]:
                raise ManualImportError("Canonical CustomerMerges parts mismatch")
            resolver = CustomerIdResolver(records())
        except InterruptedError:
            raise
        except Exception:
            raise ManualImportError("Не удалось проверить сохранённые объединения клиентов. Обновите их через API Mindbox.") from None
        store.collect_unreferenced(root)
        target = root / "canonical/objects" / uuid.uuid4().hex
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".manual-interactions-", dir=root / "canonical") as temporary:
            staging = Path(temporary)
            counts = {}
            for name, path in (("actions", actions), ("orders", orders)):
                counts[name] = _copy_validate(path, staging, name, cancelled=cancelled, progress=progress,
                               validate=lambda directory: store.validate_interactions(
                                   name, directory, resolver, selection, cancelled=cancelled, progress=progress))
            check_cancel(cancelled)
            store.require_manual_merges(data, since, until)
            old = data["manual_interactions"]
            pair = {"since": since.isoformat(), "until": until.isoformat(), "updated": store.stamp()}
            for name in ("actions", "orders"):
                pair[name] = {**{key: pair[key] for key in ("since", "until", "updated")},
                              "directory": (target / name).relative_to(root).as_posix(), "parts": counts[name],
                              "source_kind": "MANUAL", "export_id": None, "operation": "MANUAL"}
            data.update(manual_interactions=pair, schema_version=2, revision=uuid.uuid4().hex, selection=asdict(selection))
            os.replace(staging, target)
            committed = False
            try:
                store.atomic_json(root / "canonical/training.json", {"schema_version": 5, "storage": "canonical"})
                check_cancel(cancelled)
                store.atomic_json(root / "canonical/catalog.json", data)
                committed = True
            finally:
                if not committed:
                    # An interrupt may arrive after os.replace(catalog), before it
                    # returns. Never remove raw that is already referenced on disk.
                    try:
                        committed = store.catalog(root)["revision"] == data["revision"]
                    except Exception:
                        committed = True  # Uncertain outcome: leave cleanup to a later safe operation.
                    if not committed:
                        shutil.rmtree(store.checked_directory(root, pair["actions"]["directory"], "actions").parent)
            if old:
                try:
                    shutil.rmtree(store.checked_directory(root, old["actions"]["directory"], "actions").parent)
                except OSError:
                    pass  # Collected by the next operation under the store lock.
    return root / "canonical/training.json"


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
        count = _copy_validate(customers, staging, "customers", cancelled=cancelled, progress=progress, resolver=resolver)
        snapshot = CustomerProfileSnapshot(
            snapshot_id, datetime.now(timezone.utc).isoformat(), f"customer_profile_snapshots/{snapshot_id}/customers", count,
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
