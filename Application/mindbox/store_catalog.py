"""Rebuildable channel summaries. No order, customer or product identities retained."""

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re

from .adapters._common import identifier, text
from . import canonical_storage as storage
from .raw_reader import DEFAULT_RAW_ROOT, iter_export, part_files, _unique_object, RawExportError

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 1


class StoreSourceCollector:
    def __init__(self):
        self.names = defaultdict(Counter)

    def add(self, raw_order):
        key = identifier(raw_order, "firstAction.channel.ids.externalId", required=False)
        if key is not None:
            name = (text(raw_order, "firstAction.channel.name") or "").strip() or "Не указано"
            self.names[key][name] += 1

    def summary(self, entry):
        return {**_identity(entry), "names": {key: dict(sorted(names.items())) for key, names in sorted(self.names.items())}}


def _identity(entry):
    return {key: entry[key] for key in ("since", "until", "parts")}


def _references(data):
    entries = list(data["orders"].values())
    if data.get("manual_interactions"):
        entries.append(data["manual_interactions"]["orders"])
    return {entry["directory"]: entry for entry in sorted(entries, key=lambda e: e["directory"])}


def orders_signature(data):
    def identity(entry):
        return {"directory": entry["directory"], **_identity(entry)}
    manual = data.get("manual_interactions")
    payload = {"daily": [identity(entry) for _, entry in sorted(data["orders"].items())],
               "manual": identity(manual["orders"]) if manual else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _stores(data, sources):
    names = defaultdict(Counter)
    for entry in storage.effective_entries(data, "orders"):
        source = sources.get(entry["directory"])
        if source is not None:
            for key, counts in source["names"].items():
                names[key].update(counts)
    rows = [{"external_id": key, "display_name": min(counts, key=lambda name: (-counts[name], name))}
            for key, counts in names.items()]
    return sorted(rows, key=lambda row: (row["display_name"], row["external_id"]))


def _string(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def validate_catalog(value):
    """Strict boundary; partial publication summaries are usable only for recovery."""
    if (not isinstance(value, dict)
            or set(value) != {"schema_version", "orders_signature", "complete", "sources", "stores"}
            or type(value["schema_version"]) is not int or value["schema_version"] != SCHEMA_VERSION
            or type(value["complete"]) is not bool or not isinstance(value["orders_signature"], str)
            or re.fullmatch(r"[0-9a-f]{64}", value["orders_signature"]) is None
            or not isinstance(value["sources"], dict) or not isinstance(value["stores"], list)):
        raise ValueError("Invalid store catalog schema")
    for directory, source in value["sources"].items():
        if (re.fullmatch(r"canonical/objects/[0-9a-f]{32}/orders", directory) is None
                or not isinstance(source, dict) or set(source) != {"since", "until", "parts", "names"}
                or type(source["parts"]) is not int or source["parts"] < 1 or not isinstance(source["names"], dict)):
            raise ValueError("Invalid store source summary")
        since, until = (datetime.fromisoformat(source[key]) for key in ("since", "until"))
        if since.tzinfo is None or until.tzinfo is None or since >= until:
            raise ValueError("Invalid store source period")
        for key, counts in source["names"].items():
            if not _string(key) or not isinstance(counts, dict) or not counts:
                raise ValueError("Invalid store name counters")
            if any(not _string(name) or type(n) is not int or n <= 0 for name, n in counts.items()):
                raise ValueError("Invalid store name count")
    ids = set()
    for row in value["stores"]:
        if (not isinstance(row, dict) or set(row) != {"external_id", "display_name"}
                or not _string(row["external_id"]) or not _string(row["display_name"])
                or row["external_id"] in ids):
            raise ValueError("Invalid aggregate store")
        ids.add(row["external_id"])
    if value["stores"] != sorted(value["stores"], key=lambda row: (row["display_name"], row["external_id"])):
        raise ValueError("Unsorted stores")
    return value


def _load(root):
    try:
        with (root / "canonical/store_catalog.json").open(encoding="utf-8") as stream:
            return validate_catalog(json.load(stream, object_pairs_hook=_unique_object))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError, RawExportError):
        logger.warning("Каталог магазинов повреждён или недоступен; требуется восстановление.")
        return None


def _reusable(data, previous):
    old = previous["sources"] if previous else {}
    return {key: old[key] for key, entry in _references(data).items()
            if key in old and _identity(old[key]) == _identity(entry)}


def _write(root, data, sources):
    payload = {"schema_version": SCHEMA_VERSION, "orders_signature": orders_signature(data),
               "complete": set(sources) == set(_references(data)),
               "sources": dict(sorted(sources.items())), "stores": _stores(data, sources)}
    validate_catalog(payload)
    storage.atomic_json(root / "canonical/store_catalog.json", payload)
    return payload


def update_store_catalog(root, data, entry, collector):
    """Caller holds storage lock, after canonical commit. Never reread incoming raw.

    On older installations publish a partial artifact; ensure fills missing old
    summaries later. A derived failure must never fail canonical publication.
    """
    try:
        root = Path(root)
        sources = _reusable(data, _load(root))
        sources[entry["directory"]] = collector.summary(entry)
        _write(root, data, sources)
    except Exception:
        logger.warning("Не удалось обновить производный каталог магазинов; canonical Orders сохранены. Каталог будет восстановлен.")


def ensure_store_catalog(raw_root=DEFAULT_RAW_ROOT):
    root = Path(raw_root).resolve()
    if not (root / "canonical/catalog.json").is_file():
        raise ValueError("Для распределения загрузите Orders.")
    with storage.storage_lock(root):
        data = storage.catalog(root)
        references = _references(data)
        if not references:
            raise ValueError("Для распределения загрузите Orders.")
        previous = _load(root)
        sources = _reusable(data, previous)
        if (previous and previous["complete"] and previous["orders_signature"] == orders_signature(data)
                and set(sources) == set(references) and set(previous["sources"]) == set(references)
                and previous["stores"] == _stores(data, sources)):
            return tuple((row["external_id"], row["display_name"]) for row in previous["stores"])
        for key, entry in references.items():
            if key in sources:
                continue
            directory = storage.checked_directory(root, key, "orders")
            if len(part_files(directory, "orders")) != entry["parts"]:
                raise ValueError("Canonical Orders parts mismatch")
            collector = StoreSourceCollector()
            for raw in iter_export("orders", input_dir=directory):
                collector.add(raw)
            sources[key] = collector.summary(entry)
        result = _write(root, data, sources)
        return tuple((row["external_id"], row["display_name"]) for row in result["stores"])
