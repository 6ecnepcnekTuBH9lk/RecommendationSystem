from types import SimpleNamespace

import pandas as pd
import pytest

from Application.tabs import data_processing_tab, train_model_tab


_MISSING_DATE = object()


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


def _filter_window(date_from="01.01.2024", date_to="31.01.2024"):
    return SimpleNamespace(
        filter_date_from=_TextField(date_from),
        filter_date_to=_TextField(date_to),
        filter_kind=_EmptySelection(),
        filter_store=_EmptySelection(),
        kind_mode=_Choice("В группе"),
        store_mode=_Choice("В группе"),
        order_full_output_layout=_Layout(),
        favorites_full_output_layout=_Layout(),
    )


def _favorites_row(client, item, date_marker=_MISSING_DATE):
    row = {
        "MindboxID": client,
        "КодНоменклатуры": item,
        "ПолКлиента": "Мужской",
        "Возраст": "30",
        "ВозрастнаяГруппа": "26-35",
        "НазваниеНаСайте": f"Товар {item}",
    }
    if date_marker is not _MISSING_DATE:
        row["Дата"] = date_marker
    return row


def _patch_favorites_statistics_ui(monkeypatch):
    page_messages = []
    monkeypatch.setattr(data_processing_tab, "QLabel", _Label)
    monkeypatch.setattr(data_processing_tab, "clear_layout", lambda *args: None)
    monkeypatch.setattr(
        data_processing_tab,
        "vyvod_zaglyschek",
        lambda **kwargs: page_messages.append(kwargs["text"]),
    )
    return page_messages


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


def test_favorites_statistics_handles_missing_date_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    pd.DataFrame(
        [
            _favorites_row("client-1", "item-1"),
            _favorites_row("client-2", "item-2"),
        ]
    ).to_csv(input_dir / "Избранное.csv", sep="|", index=False)
    page_messages = _patch_favorites_statistics_ui(monkeypatch)

    window = _filter_window(date_from="", date_to="")
    result = data_processing_tab.analyze_favorites_full_dataset(window)

    assert result is True, page_messages
    assert page_messages == []
    output_lines = window.favorites_full_stats_label.text().splitlines()
    assert "Количество добавлений: 2" in output_lines
    assert "Период: Не определён (нет корректных дат)" in output_lines
    assert "Месяц с наибольшим количеством добавлений: Не определён" in output_lines


def test_favorites_statistics_handles_all_invalid_dates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    pd.DataFrame(
        [
            _favorites_row("client-1", "item-1", ""),
            _favorites_row("client-2", "item-2", "   "),
            _favorites_row("client-3", "item-3", "not-a-date"),
            _favorites_row("client-4", "item-4", pd.NA),
        ]
    ).to_csv(input_dir / "Избранное.csv", sep="|", index=False)
    page_messages = _patch_favorites_statistics_ui(monkeypatch)

    window = _filter_window(date_from="", date_to="")
    result = data_processing_tab.analyze_favorites_full_dataset(window)

    assert result is True, page_messages
    assert page_messages == []
    output_lines = window.favorites_full_stats_label.text().splitlines()
    assert "Количество добавлений: 4" in output_lines
    assert "Период: Не определён (нет корректных дат)" in output_lines
    assert "Месяц с наибольшим количеством добавлений: Не определён" in output_lines


def test_favorites_statistics_uses_only_valid_dates_for_period(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    pd.DataFrame(
        [
            _favorites_row("client-1", "item-1", "2024-01-05"),
            _favorites_row("client-2", "item-2", "not-a-date"),
            _favorites_row("client-3", "item-3", "2024-02-15"),
        ]
    ).to_csv(input_dir / "Избранное.csv", sep="|", index=False)
    page_messages = _patch_favorites_statistics_ui(monkeypatch)

    window = _filter_window(date_from="", date_to="")
    result = data_processing_tab.analyze_favorites_full_dataset(window)

    assert result is True
    assert page_messages == []
    output_lines = window.favorites_full_stats_label.text().splitlines()
    assert "Количество добавлений: 3" in output_lines
    assert "Период: 05.01.2024 — 15.02.2024" in output_lines


def test_favorites_period_filter_excludes_all_invalid_dates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    pd.DataFrame(
        [
            _favorites_row("client-1", "item-1", ""),
            _favorites_row("client-2", "item-2", "not-a-date"),
        ]
    ).to_csv(input_dir / "Избранное.csv", sep="|", index=False)
    page_messages = _patch_favorites_statistics_ui(monkeypatch)

    result = data_processing_tab.analyze_favorites_full_dataset(_filter_window())

    assert result is True
    assert page_messages == ["Нет данных по выбранным фильтрам"]


def test_training_date_filter_excludes_rows_with_invalid_dates(tmp_path, monkeypatch):
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
        assert result["MindboxID"].tolist() == ["in-range"]


@pytest.mark.parametrize("invalid_date", ["", "   ", pd.NA])
def test_training_date_filter_excludes_rows_with_empty_dates(
    tmp_path,
    monkeypatch,
    invalid_date,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    source = pd.DataFrame(
        {
            "Дата": [invalid_date],
            "MindboxID": ["undated"],
        }
    )
    for name in ("Заказы.csv", "Просмотры.csv", "Избранное.csv"):
        source.to_csv(input_dir / name, sep="|", index=False)

    output_dir = train_model_tab._prepare_training_data_dir(_filter_window())

    for name in ("Заказы.csv", "Просмотры.csv", "Избранное.csv"):
        result = pd.read_csv(tmp_path / output_dir / name, sep="|", dtype=str)
        assert result.empty


def test_training_date_filter_excludes_rows_when_date_column_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    source = pd.DataFrame({"MindboxID": ["undated"]})
    for name in ("Заказы.csv", "Просмотры.csv", "Избранное.csv"):
        source.to_csv(input_dir / name, sep="|", index=False)

    output_dir = train_model_tab._prepare_training_data_dir(_filter_window())

    for name in ("Заказы.csv", "Просмотры.csv", "Избранное.csv"):
        result = pd.read_csv(tmp_path / output_dir / name, sep="|", dtype=str)
        assert result.empty


def test_training_without_date_filter_keeps_undated_interactions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    pd.DataFrame(
        {
            "MindboxID": ["missing-column"],
        }
    ).to_csv(input_dir / "Заказы.csv", sep="|", index=False)
    pd.DataFrame(
        {
            "MindboxID": ["malformed"],
            "Дата": ["not-a-date"],
        }
    ).to_csv(input_dir / "Просмотры.csv", sep="|", index=False)
    pd.DataFrame(
        {
            "MindboxID": ["empty"],
            "Дата": [""],
        }
    ).to_csv(input_dir / "Избранное.csv", sep="|", index=False)

    output_dir = train_model_tab._prepare_training_data_dir(
        _filter_window(date_from="", date_to="")
    )

    assert output_dir == "ВходныеДанные"
    for name in ("Заказы.csv", "Просмотры.csv", "Избранное.csv"):
        result = pd.read_csv(tmp_path / output_dir / name, sep="|", dtype=str)
        assert len(result) == 1
