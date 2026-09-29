"""Independent channel-to-city settings; no statistics or weather side effects."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
import logging
import os
from pathlib import Path
import tempfile

from Application.analysis_filter_settings import _unique
from Application.dataset_statistics import _entries
from Application.mindbox.adapters._common import identifier, text
from Application.mindbox.canonical_storage import catalog, checked_directory, storage_lock
from Application.mindbox.order_dedup import OrderSnapshots
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, iter_export, part_files
from Application.paths import INPUT_DATA_DIR, USER_SETTINGS_DIR
from Application.product_resolution import CatalogError

SETTINGS_PATH = USER_SETTINGS_DIR / "store_city_mapping.json"
CITIES_PATH = INPUT_DATA_DIR / "city_coordinates.csv"
logger = logging.getLogger(__name__)


def discover_stores(records):
    names = defaultdict(Counter)
    snapshots = OrderSnapshots()
    for raw in records:
        if not snapshots.accept(raw, diagnose=True):
            continue
        key = identifier(raw, "firstAction.channel.ids.externalId", required=False)
        if key is None:
            continue
        name = (text(raw, "firstAction.channel.name") or "").strip() or "Не указано"
        names[key][name] += 1
    # Same frequency/lexical tie-break as OrderAggregates.result.
    stores = [(key, min(counts, key=lambda name: (-counts[name], name))) for key, counts in names.items()]
    return tuple(sorted(stores, key=lambda item: (item[1], item[0])))


def load_stores(raw_root=DEFAULT_RAW_ROOT):
    root = Path(raw_root)
    if not (root / "canonical/catalog.json").is_file():
        raise ValueError("Для распределения загрузите Orders.")
    with storage_lock(root):
        entries = _entries(catalog(root), "orders")
        if not entries:
            raise ValueError("Для распределения загрузите Orders.")

        def records():
            for entry in entries:
                directory = checked_directory(root, entry["directory"], "orders")
                if len(part_files(directory, "orders")) != entry["parts"]:
                    raise ValueError("Некорректные части canonical Orders.")
                yield from iter_export("orders", input_dir=directory)

        return discover_stores(records())


def load_cities(path=CITIES_PATH):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="|", strict=True)
            header = next(reader, [])
            if (not {"Город", "Широта", "Долгота"}.issubset(header)
                    or len(set(header)) != len(header) or any(not h.strip() for h in header)):
                raise CatalogError("Некорректные колонки справочника городов.")
            index = header.index("Город")
            cities = set()
            for row in reader:
                if len(row) != len(header) or not row[index].strip():
                    raise CatalogError("Некорректная строка справочника городов.")
                cities.add(row[index].strip())
            return tuple(sorted(cities))
    except (OSError, UnicodeError, csv.Error):
        raise CatalogError("Не удалось прочитать справочник городов.") from None


@dataclass(frozen=True)
class MappingSources:
    stores: tuple = ()
    cities: tuple | None = None
    error: str = ""


def load_sources():
    try:
        stores = load_stores()
    except Exception:
        logger.warning("Не удалось прочитать canonical Orders для распределения магазинов.")
        return MappingSources(error="Для распределения загрузите корректные Orders.")
    try:
        cities = load_cities()
    except CatalogError:
        return MappingSources(stores=stores, error="Для выбора города загрузите корректный city_coordinates.csv.")
    return MappingSources(stores, cities, "" if stores else "В Orders нет каналов со стабильным идентификатором.")


def _validate(stores):
    if not isinstance(stores, dict) or any(
        not isinstance(key, str) or not key or key != key.strip()
        or (city is not None and (not isinstance(city, str) or not city or city != city.strip()))
        for key, city in stores.items()
    ):
        raise ValueError("Некорректное распределение магазинов.")
    return stores


def load_mapping(path=SETTINGS_PATH):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=_unique)
        if (not isinstance(data, dict) or set(data) != {"schema_version", "stores"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 1):
            raise ValueError
        return _validate(data["stores"])
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        logger.warning("Не удалось загрузить распределение магазинов; файл не изменён.")
        return {}


def reconcile(stores, cities, saved):
    available = set(cities or ())
    return {key: saved.get(key) if saved.get(key) in available else None for key, _ in stores}


def save_mapping(stores, path=SETTINGS_PATH):
    payload = json.dumps({"schema_version": 1, "stores": _validate(stores)}, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".store-city-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            logger.warning("Не удалось удалить временный файл распределения.")
