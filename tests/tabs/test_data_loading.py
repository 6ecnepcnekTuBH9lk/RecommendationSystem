import csv
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
        "set_status_error",
        "set_status_ok",
        "set_status_processing",
        "show_custom_message",
        "update_file_status",
        "update_filter_controls_availability",
        "update_filter_summary",
    ):
        monkeypatch.setattr(data_processing_tab, name, lambda *args, **kwargs: None)

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

    result = pd.read_csv(target, sep="|", dtype=str)
    assert result.to_dict(orient="records") == [
        {"MindboxID": "new", "Магазин": "Новый"}
    ]


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


def test_load_csv_currently_overwrites_when_existing_file_cannot_be_read(
    tmp_path, monkeypatch
):
    """Characterize the known STABILITY-01 behavior without fixing it."""
    target_dir = tmp_path / "ВходныеДанные"
    target_dir.mkdir()
    target = target_dir / "Заказы.csv"
    target.write_text("old file that cannot be read", encoding="utf-8")
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

    with target.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    assert rows == [{"MindboxID": "new", "Магазин": "Новый"}]
