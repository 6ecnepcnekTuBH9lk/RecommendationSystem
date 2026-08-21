import codecs
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from Application.tabs import data_processing_tab


class _Choice:
    def __init__(self, value):
        self._value = value

    def currentText(self):
        return self._value


class _FileDialog:
    selected_path = ""

    @classmethod
    def getOpenFileName(cls, *args, **kwargs):
        return cls.selected_path, "CSV files (*.csv)"


def _configure_loader(monkeypatch, tmp_path, mode, new_frame):
    source = tmp_path / "synthetic_source.csv"
    source.write_text("synthetic", encoding="utf-8")
    _FileDialog.selected_path = str(source)

    window = SimpleNamespace(
        combo_box_types=_Choice("Заказы клиентов из Mindbox"),
        combo_box_add_or_not=_Choice(mode),
        errors=[],
        messages=[],
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_processing_tab, "QFileDialog", _FileDialog)
    monkeypatch.setattr(
        data_processing_tab,
        "read_csv_auto_encoding",
        lambda *args, **kwargs: new_frame.copy(),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "process_orders_file",
        lambda aboba, frame: frame,
    )

    for name in (
        "analyze_favorites_full_dataset",
        "analyze_orders_full_dataset",
        "analyze_views_full_dataset",
        "refresh_export_kind_values_from_nomenclature_file",
        "refresh_kind_values_from_loaded_files",
        "refresh_season_values_from_nomenclature_file",
        "schedule_status_reset",
        "set_status_ok",
        "set_status_processing",
        "update_file_status",
        "update_filter_controls_availability",
        "update_filter_summary",
    ):
        monkeypatch.setattr(data_processing_tab, name, lambda *args, **kwargs: None)

    monkeypatch.setattr(
        data_processing_tab,
        "set_status_error",
        lambda aboba, text: window.errors.append(text),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "show_custom_message",
        lambda aboba, **kwargs: window.messages.append(kwargs),
    )

    return window


def test_load_csv_overwrite_replaces_existing_orders(tmp_path, monkeypatch):
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target = target_dir / "Заказы.csv"
    pd.DataFrame([{"MindboxID": "old", "Магазин": "Старый"}]).to_csv(
        target, sep="|", index=False
    )
    new_frame = pd.DataFrame([{"MindboxID": "new", "Магазин": "Новый"}])
    window = _configure_loader(
        monkeypatch,
        tmp_path,
        "Добавить новый / Обновить существующий",
        new_frame,
    )

    data_processing_tab.load_csv_file(window)

    raw_result = target.read_bytes()
    assert raw_result.startswith(codecs.BOM_UTF8)
    header = raw_result[len(codecs.BOM_UTF8):].decode("utf-8").splitlines()[0]
    assert header == "MindboxID|Магазин"

    result = pd.read_csv(target, sep="|", encoding="utf-8-sig", dtype=str)
    assert result.columns.tolist() == ["MindboxID", "Магазин"]
    assert result.to_dict(orient="records") == [
        {"MindboxID": "new", "Магазин": "Новый"}
    ]
    assert window.errors == []
    assert list(target_dir.glob(f".{target.name}.*.tmp")) == []


def test_load_csv_append_preserves_existing_rows(tmp_path, monkeypatch):
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target = target_dir / "Заказы.csv"
    pd.DataFrame([{"MindboxID": "old", "Магазин": "Старый"}]).to_csv(
        target, sep="|", index=False
    )
    new_frame = pd.DataFrame([{"MindboxID": "new", "Магазин": "Новый"}])
    window = _configure_loader(
        monkeypatch,
        tmp_path,
        "Добавить данные к существующему",
        new_frame,
    )

    data_processing_tab.load_csv_file(window)

    result = pd.read_csv(target, sep="|", dtype=str)
    assert result.to_dict(orient="records") == [
        {"MindboxID": "old", "Магазин": "Старый"},
        {"MindboxID": "new", "Магазин": "Новый"},
    ]
    assert window.errors == []
    assert list(target_dir.glob(f".{target.name}.*.tmp")) == []


def test_load_csv_append_preserves_existing_file_when_read_fails(tmp_path, monkeypatch):
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target = target_dir / "Заказы.csv"
    target.write_text("old file that cannot be read", encoding="utf-8")
    original_bytes = target.read_bytes()
    new_frame = pd.DataFrame([{"MindboxID": "new", "Магазин": "Новый"}])
    window = _configure_loader(
        monkeypatch,
        tmp_path,
        "Добавить данные к существующему",
        new_frame,
    )

    monkeypatch.setattr(
        data_processing_tab.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("synthetic read error")),
    )

    data_processing_tab.load_csv_file(window)

    assert target.read_bytes() == original_bytes
    assert window.errors == ["Ошибка обработки"]
    assert "synthetic read error" in window.messages[0]["text"]
    assert list(target_dir.glob(f".{target.name}.*.tmp")) == []


def test_load_csv_preserves_existing_file_and_cleans_temp_file_when_write_fails(
    tmp_path, monkeypatch
):
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target = target_dir / "Заказы.csv"
    pd.DataFrame([{"MindboxID": "old", "Магазин": "Старый"}]).to_csv(
        target, sep="|", index=False
    )
    original_bytes = target.read_bytes()
    new_frame = pd.DataFrame([{"MindboxID": "new", "Магазин": "Новый"}])
    window = _configure_loader(
        monkeypatch,
        tmp_path,
        "Добавить новый / Обновить существующий",
        new_frame,
    )

    def write_partially_then_fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b"synthetic partial csv data")
        raise OSError("synthetic write error")

    monkeypatch.setattr(data_processing_tab.pd.DataFrame, "to_csv", write_partially_then_fail)

    data_processing_tab.load_csv_file(window)

    assert target.read_bytes() == original_bytes
    assert window.errors == ["Ошибка обработки"]
    assert "synthetic write error" in window.messages[0]["text"]
    assert list(target_dir.glob(f".{target.name}.*.tmp")) == []


def test_load_csv_preserves_existing_file_when_replace_fails(tmp_path, monkeypatch):
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target = target_dir / "Заказы.csv"
    pd.DataFrame([{"MindboxID": "old", "Магазин": "Старый"}]).to_csv(
        target, sep="|", index=False
    )
    original_bytes = target.read_bytes()
    new_frame = pd.DataFrame([{"MindboxID": "new", "Магазин": "Новый"}])
    window = _configure_loader(
        monkeypatch,
        tmp_path,
        "Добавить новый / Обновить существующий",
        new_frame,
    )
    written_temp_bytes = []

    def fail_replace(source, destination):
        written_temp_bytes.append(Path(source).read_bytes())
        raise PermissionError("synthetic replace error")

    monkeypatch.setattr(data_processing_tab.os, "replace", fail_replace)

    data_processing_tab.load_csv_file(window)

    assert written_temp_bytes
    assert written_temp_bytes[0].startswith(codecs.BOM_UTF8)
    assert written_temp_bytes[0][len(codecs.BOM_UTF8):].decode("utf-8").splitlines() == [
        "MindboxID|Магазин",
        "new|Новый",
    ]
    assert target.read_bytes() == original_bytes
    assert window.errors == ["Ошибка обработки"]
    assert "synthetic replace error" in window.messages[0]["text"]
    assert list(target_dir.glob(f".{target.name}.*.tmp")) == []


def test_cleanup_error_does_not_hide_original_write_error(
    tmp_path, monkeypatch, capsys
):
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target = target_dir / "Заказы.csv"
    pd.DataFrame([{"MindboxID": "old", "Магазин": "Старый"}]).to_csv(
        target, sep="|", index=False
    )
    original_bytes = target.read_bytes()
    new_frame = pd.DataFrame([{"MindboxID": "new", "Магазин": "Новый"}])
    window = _configure_loader(
        monkeypatch,
        tmp_path,
        "Добавить новый / Обновить существующий",
        new_frame,
    )

    def write_partially_then_fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b"synthetic partial csv data")
        raise OSError("primary synthetic write error")

    real_remove = os.remove
    monkeypatch.setattr(data_processing_tab.pd.DataFrame, "to_csv", write_partially_then_fail)
    monkeypatch.setattr(
        data_processing_tab.os,
        "remove",
        lambda path: (_ for _ in ()).throw(PermissionError("synthetic cleanup error")),
    )

    data_processing_tab.load_csv_file(window)

    leftovers = list(target_dir.glob(f".{target.name}.*.tmp"))
    try:
        assert target.read_bytes() == original_bytes
        assert window.errors == ["Ошибка обработки"]
        assert "primary synthetic write error" in window.messages[0]["text"]
        assert "synthetic cleanup error" in capsys.readouterr().err
        assert len(leftovers) == 1
    finally:
        for leftover in leftovers:
            real_remove(leftover)


def test_temp_descriptor_is_not_closed_twice_when_close_fails(tmp_path, monkeypatch):
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target = target_dir / "Заказы.csv"
    pd.DataFrame([{"MindboxID": "old", "Магазин": "Старый"}]).to_csv(
        target, sep="|", index=False
    )
    original_bytes = target.read_bytes()
    new_frame = pd.DataFrame([{"MindboxID": "new", "Магазин": "Новый"}])
    window = _configure_loader(
        monkeypatch,
        tmp_path,
        "Добавить новый / Обновить существующий",
        new_frame,
    )
    real_close = os.close
    closed_descriptors = []

    def close_then_fail(fd):
        closed_descriptors.append(fd)
        real_close(fd)
        raise OSError("synthetic close error")

    monkeypatch.setattr(data_processing_tab.os, "close", close_then_fail)

    data_processing_tab.load_csv_file(window)

    assert len(closed_descriptors) == 1
    assert target.read_bytes() == original_bytes
    assert window.errors == ["Ошибка обработки"]
    assert "synthetic close error" in window.messages[0]["text"]
    assert list(target_dir.glob(f".{target.name}.*.tmp")) == []
