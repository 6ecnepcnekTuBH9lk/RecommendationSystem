from types import SimpleNamespace

import pandas as pd

from Application.tabs import data_processing_tab, train_model_tab


class _TextField:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _Choice:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class _EmptySelection:
    def selectedItems(self):
        return []


class _Layout:
    def addWidget(self, widget):
        self.widget = widget


class _Label:
    def __init__(self, text=""):
        self._text = text

    def setTextInteractionFlags(self, flags):
        pass

    def setAlignment(self, alignment):
        pass

    def setText(self, text):
        self._text = text

    def setStyleSheet(self, stylesheet):
        pass

    def text(self):
        return self._text


def _filter_window():
    return SimpleNamespace(
        filter_date_from=_TextField("01.01.2024"),
        filter_date_to=_TextField("31.01.2024"),
        filter_kind=_EmptySelection(),
        filter_store=_EmptySelection(),
        kind_mode=_Choice("В группе"),
        store_mode=_Choice("В группе"),
        order_full_output_layout=_Layout(),
    )


def test_order_statistics_currently_excludes_rows_with_invalid_dates(
    tmp_path, monkeypatch
):
    """Characterize one side of CORRECTNESS-03 without changing it."""
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()

    common = {
        "НомерЗаказа": "order",
        "MindboxID": "client",
        "КодНоменклатуры": "item",
        "ПолКлиента": "Мужской",
        "Возраст": "30",
        "ВозрастнаяГруппа": "26-35",
        "НазваниеНаСайте": "Синтетический товар",
        "Валюта": "RUB",
        "КонечнаяСтоимость": "90",
        "НачальнаяСтоимость": "100",
        "ПроцентСкидки": "10",
        "Магазин": "Тестовый магазин",
        "ВидНоменклатуры": "Рубашки",
    }
    rows = [
        {**common, "Дата": "2024-01-15", "Количество": "1"},
        {**common, "Дата": "2023-12-31", "Количество": "10"},
        {**common, "Дата": "invalid-date", "Количество": "100"},
    ]
    pd.DataFrame(rows).to_csv(input_dir / "Заказы.csv", sep="|", index=False)

    monkeypatch.setattr(data_processing_tab, "QLabel", _Label)
    for name in (
        "clear_layout",
        "refresh_export_kind_values_from_nomenclature_file",
        "refresh_kind_values_from_loaded_files",
        "refresh_season_values_from_nomenclature_file",
        "set_list_widget_items",
        "set_order_filters_enabled",
        "update_filter_controls_availability",
        "update_filter_summary",
    ):
        monkeypatch.setattr(data_processing_tab, name, lambda *args, **kwargs: None)

    window = _filter_window()
    data_processing_tab.analyze_orders_full_dataset(window)

    assert "Количество продаж: 1" in window.order_full_stats_label.text().splitlines()


def test_training_data_currently_keeps_rows_with_invalid_dates(tmp_path, monkeypatch):
    """Characterize the other side of CORRECTNESS-03 without choosing a new rule."""
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()

    source = pd.DataFrame(
        {
            "Дата": ["2024-01-15", "2023-12-31", "invalid-date"],
            "MindboxID": ["in-range", "out-of-range", "invalid-date"],
        }
    )
    for name in ("Заказы.csv", "Просмотры.csv", "Избранное.csv"):
        source.to_csv(input_dir / name, sep="|", index=False)

    window = _filter_window()
    output_dir = train_model_tab._prepare_training_data_dir(window)

    assert output_dir == "ФильтрованныеДанные"
    for name in ("Заказы.csv", "Просмотры.csv", "Избранное.csv"):
        result = pd.read_csv(tmp_path / output_dir / name, sep="|", dtype=str)
        assert result["MindboxID"].tolist() == ["in-range", "invalid-date"]
