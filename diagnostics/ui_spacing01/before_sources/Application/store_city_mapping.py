"""Independent channel-to-city settings; no statistics or weather side effects."""

from dataclasses import dataclass
import csv
import json
import logging
import os
from pathlib import Path
import tempfile

from Application.analysis_filter_settings import _unique
from Application.mindbox.store_catalog import ensure_store_catalog
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
from Application.paths import INPUT_DATA_DIR, USER_SETTINGS_DIR
from Application.product_resolution import CatalogError

SETTINGS_PATH = USER_SETTINGS_DIR / "store_city_mapping.json"
CITIES_PATH = INPUT_DATA_DIR / "city_coordinates.csv"
logger = logging.getLogger(__name__)


def load_stores(raw_root=DEFAULT_RAW_ROOT):
    return ensure_store_catalog(raw_root)


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
