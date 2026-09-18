"""Reference publication/cache regression tests, replacing removed interaction CSV routes."""
from types import SimpleNamespace

import pandas as pd
import pytest

from Application.files import reference_import as api
from Application.tabs import create_results_tab, data_processing_tab as ui


def _nomenclature_frame(code, name, collection, stock):
    return pd.DataFrame(
        [
            {
                "КодНоменклатуры": code,
                "Номенклатура": name,
                "НазваниеНаСайте": name,
                "ВидНоменклатуры": "Synthetic kind",
                "ВидАссортимента": "Synthetic assortment",
                "Марка": "Synthetic brand",
                "Коллекция": collection,
                "СезонНоски": "Synthetic season",
                "ПолНоменклатуры": "Synthetic gender",
                "ГруппаСоставов": "Synthetic composition",
                "КатегорияНаСайте": "Synthetic category",
                "СтилеваяГруппа": "Synthetic style",
                "ТитульнаяФотография": "https://example.com/synthetic.jpg",
                "Остаток": stock,
            }
        ]
    )



@pytest.mark.parametrize("failure", [None, "schema", "replace", "write"])
def test_nomenclature_publication_and_cache_invalidation(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "ВходныеДанные"
    directory.mkdir()
    target = directory / "Номенклатура.csv"
    _nomenclature_frame("old-code", "OLD name", "OLD collection", "1").to_csv(target, sep="|", index=False)
    before = target.read_bytes()
    source = tmp_path / "source.csv"
    _nomenclature_frame("new-code", "NEW name", "NEW collection", "42").to_csv(source, sep="|", index=False)
    window = SimpleNamespace(_name_by_code={"old-code": "OLD name"},
                             _collection_by_code={"old-code": "OLD collection"}, _stock_by_code={"old-code": "1"})
    for name in ("update_file_status", "update_filter_controls_availability"):
        monkeypatch.setattr(ui, name, lambda *a: None)
    if failure == "schema":
        source.write_text("wrong\nvalue\n", encoding="utf-8")
    if failure == "replace":
        def fail(*a):
            raise PermissionError("synthetic")
        monkeypatch.setattr(api.os, "replace", fail)
    if failure == "write":
        def fail_write(self, stream, *a, **kw):
            stream.write("partial")
            raise OSError("synthetic")
        monkeypatch.setattr(api.pd.DataFrame, "to_csv", fail_write)
    if failure:
        with pytest.raises((ValueError, OSError)):
            api.import_reference(source, "Номенклатура из 1С", output_dir=directory)
        assert target.read_bytes() == before
        assert window._name_by_code == {"old-code": "OLD name"}
    else:
        result = api.import_reference(source, "Номенклатура из 1С", output_dir=directory)
        ui.apply_reference_result(window, result)
        assert window._name_by_code is window._collection_by_code is window._stock_by_code is None
        create_results_tab._ensure_item_name_map(window)
        create_results_tab._ensure_item_collection_map(window)
        create_results_tab._ensure_item_stock_map(window)
        assert window._name_by_code == {"new-code": "NEW name"}
        assert window._collection_by_code == {"new-code": "NEW collection"}
        assert window._stock_by_code == {"new-code": "42"}
        assert result["seasons"] == ["NEW collection"]
    assert list(directory.iterdir()) == [target]


def test_cleanup_failure_preserves_primary_error(tmp_path, monkeypatch, caplog):
    source = tmp_path / "source.csv"
    source.write_text("Город,Широта,Долгота\nSynthetic,55,37\n", encoding="utf-8-sig")
    def fail_write(self, stream, *a, **kw):
        raise OSError("primary write failure")
    monkeypatch.setattr(api.pd.DataFrame, "to_csv", fail_write)
    original = api.Path.unlink
    def fail_unlink(path, *a, **kw):
        if path.suffix == ".tmp":
            raise PermissionError("cleanup failure")
        return original(path, *a, **kw)
    monkeypatch.setattr(api.Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="primary write failure"):
        api.import_reference(source, "Координаты городов и погода", output_dir=tmp_path / "out")
    assert "временный CSV" in caplog.text
