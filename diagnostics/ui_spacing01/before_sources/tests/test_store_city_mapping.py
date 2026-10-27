import json
from pathlib import Path

import pytest

from Application import store_city_mapping as mapping
from Application.product_resolution import CatalogError
import test_dataset_statistics as baseline

dataset = baseline.dataset


def test_load_stores_delegates_to_catalog(monkeypatch, tmp_path):
    calls = []
    def ensure(root):
        calls.append(root)
        return (("A", "Магазин A"),)
    monkeypatch.setattr(mapping, "ensure_store_catalog", ensure)
    assert mapping.load_stores(tmp_path) == (("A", "Магазин A"),)
    assert calls == [tmp_path]


def test_orders_only_canonical_source(dataset):
    root, _, data = dataset
    data["actions"] = {}
    data["customer_merges"] = None
    baseline.save(root, data)
    (root / "canonical/customers.sqlite").unlink()
    assert mapping.load_stores(root) == (("shop", "Shop"),)


def test_missing_orders(tmp_path):
    with pytest.raises(ValueError):
        mapping.load_stores(tmp_path)


def test_cities_unique_sorted(tmp_path):
    path = tmp_path / "cities.csv"
    path.write_text("Город|Широта|Долгота\n Москва |55|37\nКазань|55|49\nМосква|55|37\n", encoding="utf-8-sig")
    assert mapping.load_cities(path) == ("Казань", "Москва")


@pytest.mark.parametrize("content", ["Город\nМосква\n", "Город|Широта|Долгота\nМосква|55\n",
                                    "Город|Широта|Долгота\n |55|37\n", "Город|Город|Широта|Долгота\n"])
def test_bad_reference_strict(tmp_path, content):
    path = tmp_path / "cities.csv"
    path.write_text(content, encoding="utf-8-sig")
    with pytest.raises(CatalogError):
        mapping.load_cities(path)


@pytest.mark.parametrize("payload", ["broken", "[]", '{"schema_version":true,"stores":{}}',
    '{"schema_version":2,"stores":{}}', '{"schema_version":1,"stores":{"A":12}}',
    '{"schema_version":1,"stores":{"A":null,"A":"Москва"}}'])
def test_bad_settings_safe_unchanged(tmp_path, payload, caplog):
    path = tmp_path / "mapping.json"
    path.write_text(payload, encoding="utf-8")
    assert mapping.load_mapping(path) == {}
    assert path.read_text(encoding="utf-8") == payload
    assert caplog.records


def test_roundtrip_atomic_and_old_store_retained(tmp_path, monkeypatch):
    path = tmp_path / "mapping.json"
    saved = {"old": "Казань", "A": "Москва", "B": "Исчезнувший город"}
    current = mapping.reconcile((("A", "Переименован"), ("B", "B"), ("new", "Новый")), ("Москва", "Казань"), saved)
    assert current == {"A": "Москва", "B": None, "new": None}
    events = []
    real_fsync, real_replace = mapping.os.fsync, mapping.os.replace
    def fsync(fd):
        events.append("fsync")
        real_fsync(fd)
    def replace(source, target):
        events.append("replace")
        assert json.loads(Path(source).read_text(encoding="utf-8"))["schema_version"] == 1
        real_replace(source, target)
    monkeypatch.setattr(mapping.os, "fsync", fsync)
    monkeypatch.setattr(mapping.os, "replace", replace)
    mapping.save_mapping({**saved, **current}, path)
    assert events == ["fsync", "replace"]
    assert mapping.load_mapping(path) == {"old": "Казань", "A": "Москва", "B": None, "new": None}
    assert list(tmp_path.iterdir()) == [path]


def test_failed_atomic_save_preserves_file(tmp_path, monkeypatch):
    path = tmp_path / "mapping.json"
    mapping.save_mapping({"A": None}, path)
    before = path.read_bytes()
    def failure(*args):
        raise OSError("synthetic")
    monkeypatch.setattr(mapping.os, "replace", failure)
    with pytest.raises(OSError):
        mapping.save_mapping({"A": "Москва"}, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_sources_keep_stores_without_cities(monkeypatch):
    monkeypatch.setattr(mapping, "load_stores", lambda: (("A", "Магазин"),))
    def missing():
        raise CatalogError
    monkeypatch.setattr(mapping, "load_cities", missing)
    sources = mapping.load_sources()
    assert sources.stores == (("A", "Магазин"),) and sources.cities is None
    assert sources.error
