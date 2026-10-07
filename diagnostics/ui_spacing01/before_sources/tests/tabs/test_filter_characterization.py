from types import SimpleNamespace

import pandas as pd
import pytest

from Application.tabs import train_model_tab


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


def test_training_date_filter_excludes_rows_with_invalid_dates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input_data"
    input_dir.mkdir()

    source = pd.DataFrame(
        {
            "Дата": ["2024-01-15", "2023-12-31", "invalid-date"],
            "MindboxID": ["in-range", "out-of-range", "invalid-date"],
        }
    )
    for name in ("orders.csv", "views.csv", "favorites.csv"):
        source.to_csv(input_dir / name, sep="|", index=False)

    window = _filter_window()
    output_dir = train_model_tab._prepare_training_data_dir(window)

    assert output_dir == "filtered_data"
    for name in ("orders.csv", "views.csv", "favorites.csv"):
        result = pd.read_csv(tmp_path / output_dir / name, sep="|", dtype=str)
        assert result["MindboxID"].tolist() == ["in-range"]


@pytest.mark.parametrize("invalid_date", ["", "   ", pd.NA])
def test_training_date_filter_excludes_rows_with_empty_dates(
    tmp_path,
    monkeypatch,
    invalid_date,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input_data"
    input_dir.mkdir()
    source = pd.DataFrame(
        {
            "Дата": [invalid_date],
            "MindboxID": ["undated"],
        }
    )
    for name in ("orders.csv", "views.csv", "favorites.csv"):
        source.to_csv(input_dir / name, sep="|", index=False)

    output_dir = train_model_tab._prepare_training_data_dir(_filter_window())

    for name in ("orders.csv", "views.csv", "favorites.csv"):
        result = pd.read_csv(tmp_path / output_dir / name, sep="|", dtype=str)
        assert result.empty


def test_training_date_filter_excludes_rows_when_date_column_is_missing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input_data"
    input_dir.mkdir()
    source = pd.DataFrame({"MindboxID": ["undated"]})
    for name in ("orders.csv", "views.csv", "favorites.csv"):
        source.to_csv(input_dir / name, sep="|", index=False)

    output_dir = train_model_tab._prepare_training_data_dir(_filter_window())

    for name in ("orders.csv", "views.csv", "favorites.csv"):
        result = pd.read_csv(tmp_path / output_dir / name, sep="|", dtype=str)
        assert result.empty


def test_training_without_date_filter_keeps_undated_interactions(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "input_data"
    input_dir.mkdir()
    pd.DataFrame(
        {
            "MindboxID": ["missing-column"],
        }
    ).to_csv(input_dir / "orders.csv", sep="|", index=False)
    pd.DataFrame(
        {
            "MindboxID": ["malformed"],
            "Дата": ["not-a-date"],
        }
    ).to_csv(input_dir / "views.csv", sep="|", index=False)
    pd.DataFrame(
        {
            "MindboxID": ["empty"],
            "Дата": [""],
        }
    ).to_csv(input_dir / "favorites.csv", sep="|", index=False)

    output_dir = train_model_tab._prepare_training_data_dir(
        _filter_window(date_from="", date_to="")
    )

    assert output_dir == "input_data"
    for name in ("orders.csv", "views.csv", "favorites.csv"):
        result = pd.read_csv(tmp_path / output_dir / name, sep="|", dtype=str)
        assert len(result) == 1
