from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from Application.tabs import data_processing_tab


class _ListItem:
    def __init__(self, text):
        self._text = str(text)
        self.selected = False

    def text(self):
        return self._text

    def setSelected(self, selected):
        self.selected = selected


class _ListWidget:
    def __init__(self, values):
        self.items = [_ListItem(value) for value in values]

    def blockSignals(self, blocked):
        pass

    def clear(self):
        self.items.clear()

    def addItem(self, value):
        self.items.append(_ListItem(value))

    def count(self):
        return len(self.items)

    def item(self, index):
        return self.items[index]

    def selectedItems(self):
        return [item for item in self.items if item.selected]

    def values(self):
        return [item.text() for item in self.items]


class _TableItem:
    def __init__(self, text):
        self._text = str(text)
        self._flags = data_processing_tab.Qt.ItemFlag.ItemIsEditable

    def text(self):
        return self._text

    def flags(self):
        return self._flags

    def setFlags(self, flags):
        self._flags = flags


class _ComboBox:
    def __init__(self):
        self.items = []
        self.enabled = True
        self.current_index = 0

    def setEditable(self, editable):
        pass

    def addItem(self, value):
        self.items.append(value)

    def addItems(self, values):
        self.items.extend(values)

    def findText(self, value):
        try:
            return self.items.index(value)
        except ValueError:
            return -1

    def setCurrentIndex(self, index):
        self.current_index = index

    def setEnabled(self, enabled):
        self.enabled = enabled


class _Table:
    def __init__(self, stores):
        self.row_count = len(stores)
        self.items = {
            (row, 0): _TableItem(store) for row, store in enumerate(stores)
        }
        self.widgets = {}

    def setRowCount(self, count):
        self.row_count = count
        self.items = {
            key: value for key, value in self.items.items() if key[0] < count
        }
        self.widgets = {
            key: value for key, value in self.widgets.items() if key[0] < count
        }

    def setItem(self, row, column, item):
        self.items[(row, column)] = item

    def setCellWidget(self, row, column, widget):
        self.widgets[(row, column)] = widget

    def stores(self):
        return [self.items[(row, 0)].text() for row in range(self.row_count)]


def _write_pipe_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )


def _store_window(stores):
    return SimpleNamespace(
        store_city_table=_Table(stores),
        _cities=["Москва"],
        _store_city_map={},
    )


def test_store_city_successfully_replaces_old_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "ВходныеДанные" / "СписокМагазинов.csv"
    _write_pipe_csv(source, [{"Магазин": "NEW"}], ["Магазин"])
    window = _store_window(["OLD"])

    monkeypatch.setattr(data_processing_tab, "QTableWidgetItem", _TableItem)
    monkeypatch.setattr(data_processing_tab, "QComboBox", _ComboBox)

    data_processing_tab.refresh_store_city_table(window)

    assert window.store_city_table.stores() == ["NEW"]


def test_store_city_missing_source_keeps_empty_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    window = _store_window(["OLD"])

    monkeypatch.setattr(data_processing_tab, "QTableWidgetItem", _TableItem)
    monkeypatch.setattr(data_processing_tab, "QComboBox", _ComboBox)

    data_processing_tab.refresh_store_city_table(window)

    assert window.store_city_table.stores() == []


def test_store_city_valid_empty_source_replaces_old_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "ВходныеДанные" / "СписокМагазинов.csv"
    _write_pipe_csv(source, [], ["Магазин"])
    window = _store_window(["OLD"])

    monkeypatch.setattr(data_processing_tab, "QTableWidgetItem", _TableItem)
    monkeypatch.setattr(data_processing_tab, "QComboBox", _ComboBox)

    data_processing_tab.refresh_store_city_table(window)

    assert window.store_city_table.row_count == 0
    assert window.store_city_table.stores() == []


def test_store_city_unreadable_source_preserves_old_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "ВходныеДанные" / "СписокМагазинов.csv"
    source.parent.mkdir()
    source.write_text("synthetic source exists", encoding="utf-8")
    window = _store_window(["OLD"])

    monkeypatch.setattr(data_processing_tab, "QTableWidgetItem", _TableItem)
    monkeypatch.setattr(data_processing_tab, "QComboBox", _ComboBox)
    monkeypatch.setattr(
        data_processing_tab.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("synthetic unreadable store source")
        ),
    )

    with pytest.raises(PermissionError, match="synthetic unreadable store source"):
        data_processing_tab.refresh_store_city_table(window)

    assert window.store_city_table.stores() == ["OLD"]


def _kind_window(values):
    return SimpleNamespace(filter_kind=_ListWidget(values))


def test_multi_source_kind_successfully_replaces_old_values(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    for filename, value in (
        ("Заказы.csv", "A"),
        ("Просмотры.csv", "B"),
        ("Избранное.csv", "C"),
    ):
        _write_pipe_csv(
            input_dir / filename,
            [{"ВидНоменклатуры": value}],
            ["ВидНоменклатуры"],
        )
    window = _kind_window(["OLD"])

    data_processing_tab.refresh_kind_values_from_loaded_files(window)

    assert window.filter_kind.values() == ["A", "B", "C"]


def test_multi_source_kind_allows_missing_optional_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    _write_pipe_csv(
        input_dir / "Заказы.csv",
        [{"ВидНоменклатуры": "A"}],
        ["ВидНоменклатуры"],
    )
    _write_pipe_csv(
        input_dir / "Избранное.csv",
        [{"ВидНоменклатуры": "C"}],
        ["ВидНоменклатуры"],
    )
    window = _kind_window(["OLD"])

    data_processing_tab.refresh_kind_values_from_loaded_files(window)

    assert window.filter_kind.values() == ["A", "C"]


def test_multi_source_kind_read_error_does_not_publish_partial_values(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    for filename in ("Заказы.csv", "Просмотры.csv", "Избранное.csv"):
        _write_pipe_csv(
            input_dir / filename,
            [{"ВидНоменклатуры": filename}],
            ["ВидНоменклатуры"],
        )
    window = _kind_window(["old-value"])
    real_read_csv = data_processing_tab.pd.read_csv

    def read_with_failure(path, *args, **kwargs):
        if Path(path).name == "Избранное.csv":
            raise PermissionError("synthetic unreadable kind source")
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(data_processing_tab.pd, "read_csv", read_with_failure)

    with pytest.raises(PermissionError, match="synthetic unreadable kind source"):
        data_processing_tab.refresh_kind_values_from_loaded_files(window)

    assert window.filter_kind.values() == ["old-value"]


def test_startup_reference_failure_is_reported_and_does_not_stop_other_refreshes(
    tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "ВходныеДанные" / "Заказы.csv"
    source.parent.mkdir()
    source.write_text("synthetic source exists", encoding="utf-8")

    calls = []
    statuses = []
    window = _kind_window(["OLD"])
    real_kind_refresh = data_processing_tab.refresh_kind_values_from_loaded_files

    def fail_kind_refresh(current_window):
        calls.append("kind")
        return real_kind_refresh(current_window)

    def refresh_seasons(current_window):
        calls.append("season")

    def refresh_export_kinds(current_window):
        calls.append("export-kind")

    monkeypatch.setattr(
        data_processing_tab,
        "refresh_kind_values_from_loaded_files",
        fail_kind_refresh,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "refresh_season_values_from_nomenclature_file",
        refresh_seasons,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "refresh_export_kind_values_from_nomenclature_file",
        refresh_export_kinds,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_error",
        lambda current_window, text: statuses.append(("error", text)),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_ok",
        lambda current_window, text: statuses.append(("success", text)),
    )
    monkeypatch.setattr(
        data_processing_tab.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("synthetic startup reference failure")
        ),
    )

    data_processing_tab._refresh_filter_references_on_startup(window)

    assert calls == ["kind", "season", "export-kind"]
    assert window.filter_kind.values() == ["OLD"]
    assert statuses == [
        (
            "error",
            "Не удалось обновить справочник фильтров: "
            "synthetic startup reference failure",
        )
    ]


@pytest.mark.parametrize(
    ("function_name", "widget_name", "column", "values"),
    [
        (
            "refresh_export_kind_values_from_nomenclature_file",
            "export_kind_filter",
            "ВидНоменклатуры",
            ["Брюки", "Рубашки"],
        ),
        (
            "refresh_season_values_from_nomenclature_file",
            "filter_season",
            "Коллекция",
            ["Весна-Лето 2026", "Осень-Зима 2026"],
        ),
    ],
)
def test_nomenclature_reference_successfully_replaces_old_values(
    tmp_path, monkeypatch, function_name, widget_name, column, values
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "ВходныеДанные" / "Номенклатура.csv"
    _write_pipe_csv(source, [{column: value} for value in reversed(values)], [column])
    widget = _ListWidget(["OLD"])
    window = SimpleNamespace(**{widget_name: widget})

    getattr(data_processing_tab, function_name)(window)

    assert widget.values() == values


@pytest.mark.parametrize(
    ("function_name", "widget_name"),
    [
        (
            "refresh_export_kind_values_from_nomenclature_file",
            "export_kind_filter",
        ),
        ("refresh_season_values_from_nomenclature_file", "filter_season"),
    ],
)
def test_nomenclature_reference_missing_source_keeps_empty_contract(
    tmp_path, monkeypatch, function_name, widget_name
):
    monkeypatch.chdir(tmp_path)
    widget = _ListWidget(["OLD"])
    window = SimpleNamespace(**{widget_name: widget})

    getattr(data_processing_tab, function_name)(window)

    assert widget.values() == []


@pytest.mark.parametrize(
    ("function_name", "widget_name"),
    [
        (
            "refresh_export_kind_values_from_nomenclature_file",
            "export_kind_filter",
        ),
        ("refresh_season_values_from_nomenclature_file", "filter_season"),
    ],
)
def test_nomenclature_reference_unreadable_source_preserves_old_values(
    tmp_path, monkeypatch, function_name, widget_name
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "ВходныеДанные" / "Номенклатура.csv"
    source.parent.mkdir()
    source.write_text("synthetic source exists", encoding="utf-8")
    widget = _ListWidget(["OLD"])
    window = SimpleNamespace(**{widget_name: widget})

    monkeypatch.setattr(
        data_processing_tab.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pd.errors.ParserError("synthetic unreadable nomenclature")
        ),
    )

    with pytest.raises(
        pd.errors.ParserError, match="synthetic unreadable nomenclature"
    ):
        getattr(data_processing_tab, function_name)(window)

    assert widget.values() == ["OLD"]


@pytest.mark.parametrize(
    ("function_name", "widget_name", "column"),
    [
        (
            "refresh_export_kind_values_from_nomenclature_file",
            "export_kind_filter",
            "ВидНоменклатуры",
        ),
        (
            "refresh_season_values_from_nomenclature_file",
            "filter_season",
            "Коллекция",
        ),
    ],
)
def test_valid_empty_nomenclature_reference_may_publish_empty_values(
    tmp_path, monkeypatch, function_name, widget_name, column
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "ВходныеДанные" / "Номенклатура.csv"
    _write_pipe_csv(source, [], [column])
    widget = _ListWidget(["OLD"])
    window = SimpleNamespace(**{widget_name: widget})

    getattr(data_processing_tab, function_name)(window)

    assert widget.values() == []
