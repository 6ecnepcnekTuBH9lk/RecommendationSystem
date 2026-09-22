"""Current daily raw data. A single atomic catalog switch publishes each component.

Legacy exports are never collected. Readers preparing training hold the same lock
as writers, so replacing a partition cannot remove files underneath a reader.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .daily_training_batch import BatchComponent, ChunkedTrainingBatch, _writer_lock, split_daily_windows
from .raw_reader import iter_export, part_files, _unique_object
from .selection import DEFAULT_SELECTION, MindboxSelectionConfig
from .training_batch import TrainingBatchExport, TrainingBatchWindow, TrainingBatchError


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def storage_lock(root):
    directory = Path(root).resolve() / "canonical"
    directory.mkdir(parents=True, exist_ok=True)
    with _writer_lock(directory):
        yield


def catalog(root):
    path = Path(root) / "canonical/catalog.json"
    if not path.exists():
        return {"schema_version": 1, "revision": uuid.uuid4().hex, "actions": {}, "orders": {},
                "customer_merges": None, "manual_interactions": None, "selection": asdict(DEFAULT_SELECTION)}
    try:
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream, object_pairs_hook=_unique_object)
        fields = {"schema_version", "revision", "actions", "orders", "customer_merges", "selection"}
        version = result["schema_version"]
        if version == 2:
            fields.add("manual_interactions")
        if version not in (1, 2) or set(result) != fields:
            raise ValueError
        result.setdefault("manual_interactions", None)
        MindboxSelectionConfig(**result["selection"])
        for name in ("actions", "orders"):
            for day, entry in result[name].items():
                if datetime.fromisoformat(entry["since"]).date().isoformat() != day:
                    raise ValueError
                if len(split_daily_windows(datetime.fromisoformat(entry["since"]), datetime.fromisoformat(entry["until"]))) != 1:
                    raise ValueError
                checked_directory(root, entry["directory"], name)
        if result["customer_merges"]:
            checked_directory(root, result["customer_merges"]["directory"], "customer_merges")
        manual = result["manual_interactions"]
        if manual is not None:
            validate_manual_period(datetime.fromisoformat(manual["since"]), datetime.fromisoformat(manual["until"]))
            parents = set()
            for name in ("actions", "orders"):
                entry = manual[name]
                if (any(entry[key] != manual[key] for key in ("since", "until", "updated"))
                        or entry["source_kind"] != "MANUAL" or entry["export_id"] is not None
                        or entry["operation"] != "MANUAL" or type(entry["parts"]) is not int or entry["parts"] != 1):
                    raise ValueError
                parents.add(checked_directory(root, entry["directory"], name).parent)
            if len(parents) != 1:
                raise ValueError
        return result
    except (ValueError, TypeError, KeyError, OSError):
        raise TrainingBatchError("Invalid canonical catalog; legacy fallback is disabled") from None


def checked_directory(root, value, name):
    root = Path(root).resolve()
    path = (root / value).resolve()
    parts = Path(value).parts
    if (len(parts) != 4 or parts[:2] != ("canonical", "objects") or parts[3] != name
            or len(parts[2]) != 32 or any(c not in "0123456789abcdef" for c in parts[2])
            or not path.is_relative_to(root / "canonical/objects")):
        raise TrainingBatchError("Invalid canonical object path")
    return path


def validate_manual_period(since, until):
    if (since.tzinfo is None or until.tzinfo is None or since >= until
            or any(value.utcoffset().total_seconds() != 0 or any((value.hour, value.minute, value.second, value.microsecond))
                   for value in (since, until))):
        raise TrainingBatchError("Период ручной выгрузки должен идти от ранней до поздней даты, в полночь UTC.")


def require_manual_merges(data, since, until):
    validate_manual_period(since, until)
    entry = data["customer_merges"]
    if not entry or datetime.fromisoformat(entry["since"]) > since or datetime.fromisoformat(entry["until"]) < until:
        raise TrainingBatchError("Для выбранного периода недостаточно сохранённой истории объединений клиентов. "
                                 "Сначала обновите объединения клиентов через API Mindbox.")
    return entry


def continuous_range(data):
    """Return source pairs in the longest covered run, latest on ties; no virtual days."""
    merges = data["customer_merges"]
    if not merges:
        return ()
    def covered(entry):
        return (datetime.fromisoformat(merges["since"]) <= datetime.fromisoformat(entry["since"])
                and datetime.fromisoformat(entry["until"]) <= datetime.fromisoformat(merges["until"]))
    manual = data.get("manual_interactions")
    pairs = []
    if manual and covered(manual):
        pairs.append({name: manual[name] for name in ("actions", "orders")})
    for day in set(data["actions"]) & set(data["orders"]):
        pair = {name: data[name][day] for name in ("actions", "orders")}
        entry = pair["actions"]
        if manual and (datetime.fromisoformat(entry["since"]) < datetime.fromisoformat(manual["until"])
                       and datetime.fromisoformat(entry["until"]) > datetime.fromisoformat(manual["since"])):
            continue
        if all(covered(value) for value in pair.values()):
            pairs.append(pair)
    pairs.sort(key=lambda pair: datetime.fromisoformat(pair["actions"]["since"]))
    runs = []
    for pair in pairs:
        if not runs or datetime.fromisoformat(runs[-1][-1]["actions"]["until"]) != datetime.fromisoformat(pair["actions"]["since"]):
            runs.append([])
        runs[-1].append(pair)
    def rank(run):
        end = datetime.fromisoformat(run[-1]["actions"]["until"])
        return end - datetime.fromisoformat(run[0]["actions"]["since"]), end
    return tuple(max(runs, key=rank)) if runs else ()


def current_batch(root):
    data = catalog(root)
    pairs = continuous_range(data)
    if not pairs:
        raise TrainingBatchError("No continuous common Actions/Orders coverage")
    def component(name, entry):
        directory = checked_directory(root, entry["directory"], name)
        if len(part_files(directory, name)) != entry["parts"]:
            raise TrainingBatchError("Canonical parts mismatch")
        export = TrainingBatchExport(name, entry["export_id"], entry["operation"], entry["directory"],
                                     entry["parts"], entry["source_kind"])
        return BatchComponent(name, datetime.fromisoformat(entry["since"]), datetime.fromisoformat(entry["until"]),
                              entry["operation"], "READY", export)
    merges = component("customer_merges", data["customer_merges"])
    components = (merges, *(component(name, pair[name]) for pair in pairs for name in ("actions", "orders")))
    window = TrainingBatchWindow(components[1].since, components[-1].until, merges.since)
    return ChunkedTrainingBatch(data["revision"], datetime.fromisoformat(max(
        pair[name]["updated"] for pair in pairs for name in ("actions", "orders"))),
        window, "0" * 64, components, True, MindboxSelectionConfig(**data["selection"]), "CANONICAL")


def validate_interactions(name, directory, resolver, selection, *, cancelled=None, progress=None):
    """Shared streaming validation for API and manual raw; never filter the stored file."""
    from .adapters import adapt_order, adapt_action
    from .adapters.actions import adapt_action_system_name
    from .manual_import import check_cancel
    from Application.interactions import classify_action_system_name
    rules = selection.interaction_rules()
    count = 0
    for raw in iter_export(name, input_dir=directory):
        check_cancel(cancelled)
        if name == "orders":
            adapt_order(raw, resolver, product_namespaces=selection.order_product_namespaces)
        elif classify_action_system_name(adapt_action_system_name(raw), rules) is not None:
            adapt_action(raw, resolver, product_namespaces=selection.action_product_namespaces)
        count += 1
        if progress and count % 10000 == 0:
            progress(f"Проверка {name}: {count} записей")
    if progress:
        progress(f"Проверка {name} завершена: {count} записей")


def publish(root, name, since, until, directory, *, source_kind="API", export_id=None,
            operation="MANUAL", selection=DEFAULT_SELECTION, job=None):
    """Caller holds storage_lock. Validate staged raw before catalog publication."""
    root = Path(root).resolve()
    directory = Path(directory).resolve()
    parts = part_files(directory, name)
    if name in ("actions", "orders"):
        from .canonical_customers import resolver_for
        validate_interactions(name, directory, resolver_for(root), selection)
    if name == "customer_merges":
        from .adapters import adapt_customer_merge
        from .identity import CustomerIdResolver
        CustomerIdResolver(adapt_customer_merge(raw) for raw in iter_export(name, input_dir=directory))
    elif len(split_daily_windows(since, until)) != 1:
        raise TrainingBatchError("Canonical interactions require exactly one UTC day")
    target = root / "canonical/objects" / uuid.uuid4().hex / name
    target.parent.mkdir(parents=True)
    # All callers stage on the raw root filesystem. Never move a user's source file.
    os.replace(directory, target)
    data = catalog(root)
    old = data[name] if name == "customer_merges" else data[name].get(since.date().isoformat())
    entry = {"since": since.isoformat(), "until": until.isoformat(), "updated": stamp(),
             "directory": target.relative_to(root).as_posix(), "parts": len(parts), "source_kind": source_kind,
             "export_id": export_id, "operation": operation, "job": job}
    if name == "customer_merges":
        if old and (since > datetime.fromisoformat(old["since"]) or until < datetime.fromisoformat(old["until"])):
            raise TrainingBatchError("CustomerMerges coverage cannot shrink")
        data[name] = entry
    else:
        data[name][since.date().isoformat()] = entry
    data["revision"] = uuid.uuid4().hex
    data["schema_version"] = 2
    data["selection"] = asdict(selection)
    atomic_json(root / "canonical/training.json", {"schema_version": 5, "storage": "canonical"})
    atomic_json(root / "canonical/catalog.json", data)
    if old:
        obsolete = checked_directory(root, old["directory"], name)
        # Confined to a generated canonical object, never legacy/user paths.
        try:
            shutil.rmtree(obsolete.parent)
        except OSError:
            pass  # Interrupted cleanup is retried by collect_unreferenced.
    return entry


def collect_unreferenced(root):
    """Caller holds the store lock. Only UUID objects absent from current catalog."""
    data = catalog(root)
    live = {entry["directory"].split("/")[2] for name in ("actions", "orders") for entry in data[name].values()}
    if data["customer_merges"]:
        live.add(data["customer_merges"]["directory"].split("/")[2])
    if data["manual_interactions"]:
        live.add(data["manual_interactions"]["actions"]["directory"].split("/")[2])
    parent = Path(root).resolve() / "canonical/objects"
    for path in parent.glob("*"):
        if (path.name not in live and len(path.name) == 32 and all(c in "0123456789abcdef" for c in path.name)
                and not path.is_symlink() and path.resolve().parent == parent):
            shutil.rmtree(path)
    for path in (*parent.parent.glob(".transport-*"), *parent.parent.glob(".manual-interactions-*")):
        if path.is_dir() and not path.is_symlink() and path.resolve().parent == parent.parent:
            shutil.rmtree(path)
    for path in parent.parent.glob(".customers-*.sqlite*"):
        if path.is_file() and not path.is_symlink() and path.resolve().parent == parent.parent:
            path.unlink()


def merge_target(root, since, until):
    old = catalog(root)["customer_merges"]
    return (min(since, datetime.fromisoformat(old["since"])), max(until, datetime.fromisoformat(old["until"]))) if old else (since, until)


def summary(root):
    data = catalog(root)
    pairs = continuous_range(data)
    result = {name: None for name in ("actions", "orders", "customer_merges", "customers")}
    for name in ("actions", "orders"):
        if pairs:
            result[name] = {"since": pairs[0][name]["since"], "until": pairs[-1][name]["until"],
                            "updated": max(pair[name]["updated"] for pair in pairs)}
    if data["customer_merges"]:
        result["customer_merges"] = {key: data["customer_merges"][key] for key in ("since", "until", "updated")}
    return result
