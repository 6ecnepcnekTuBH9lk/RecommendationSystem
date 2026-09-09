from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from Application.tabs import train_model_tab


WEATHER_COLUMNS = (
    "ПогодныеУсловия",
    "СредняяТемпература",
    "КоличествоОсадков",
)


def _orders():
    return pd.DataFrame(
        [
            {
                "MindboxID": "client-1",
                "КодНоменклатуры": "item-1",
                "Количество": "2",
                "Магазин": "STORE-1",
                "Дата": "2025-01-15",
            }
        ]
    )


def _window():
    return SimpleNamespace(_store_city_map={"STORE-1": "Москва"})


def _write_weather(path, rows):
    pd.DataFrame(rows).to_csv(
        path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )


@pytest.mark.parametrize(
    ("duplicate_name", "duplicate_header"),
    [
        ("Дата", " Дата "),
        ("Город", "Город "),
        ("ПогодныеУсловия", " ПогодныеУсловия"),
        ("Дата", "\ufeffДата"),
    ],
    ids=["date", "city", "weather-value", "bom-date"],
)
def test_duplicate_normalized_weather_headers_preserve_orders_and_report_warning(
    tmp_path,
    duplicate_name,
    duplicate_header,
):
    weather_path = tmp_path / "Погода.csv"
    canonical_headers = [
        "Дата",
        "Город",
        "ПогодныеУсловия",
        "СредняяТемпература",
        "КоличествоОсадков",
    ]
    duplicate_index = canonical_headers.index(duplicate_name) + 1
    headers = canonical_headers.copy()
    headers.insert(duplicate_index, duplicate_header)
    values = ["2025-01-15", "Москва", "Ясно", "1.5", "0"]
    values.insert(duplicate_index, values[duplicate_index - 1])
    weather_path.write_text(
        "|".join(headers) + "\n" + "|".join(values) + "\n",
        encoding="utf-8",
    )
    source = _orders()
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), source, str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == len(source) == 1
    for column in (
        "MindboxID",
        "КодНоменклатуры",
        "Количество",
        "Магазин",
        "Дата",
    ):
        assert result[column].tolist() == source[column].tolist()
    assert result.loc[0, "Город"] == "Москва"
    assert all(pd.isna(result.loc[0, column]) for column in WEATHER_COLUMNS)
    assert any(
        "повтор" in message.lower() and duplicate_name in message
        for message in diagnostics
    )


def test_duplicate_weather_keys_do_not_multiply_orders(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    _write_weather(
        weather_path,
        [
            {
                "Дата": "2025-01-15",
                "Город": "Москва",
                "ПогодныеУсловия": "Ясно",
                "СредняяТемпература": "1",
                "КоличествоОсадков": "0",
            },
            {
                "Дата": "2025-01-15",
                "Город": "Москва",
                "ПогодныеУсловия": "Снег",
                "СредняяТемпература": "-1",
                "КоличествоОсадков": "2",
            },
        ],
    )
    source = _orders()

    diagnostics = []
    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), source, str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == len(source) == 1
    assert result.loc[0, "MindboxID"] == "client-1"
    assert result.loc[0, "КодНоменклатуры"] == "item-1"
    assert result.loc[0, "Количество"] == "2"
    assert result.loc[0, "Магазин"] == "STORE-1"
    assert result.loc[0, "Город"] == "Москва"
    assert result.loc[0, "Дата"] == pd.Timestamp("2025-01-15")
    assert all(pd.isna(result.loc[0, column]) for column in WEATHER_COLUMNS)
    assert any("дубл" in message.lower() for message in diagnostics)


def test_weather_success_populates_canonical_columns_without_suffixes(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    _write_weather(
        weather_path,
        [
            {
                "Дата": "2025-01-15",
                "Город": "Москва",
                "ПогодныеУсловия": "Ясно",
                "СредняяТемпература": "1.5",
                "КоличествоОсадков": "0",
            }
        ],
    )

    source = _orders()
    source["ПогодныеУсловия"] = "Старое значение"
    source["СредняяТемпература"] = "99"
    source["КоличествоОсадков"] = "99"

    diagnostics = []
    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), source, str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 1
    assert result.loc[0, "ПогодныеУсловия"] == "Ясно"
    assert result.loc[0, "СредняяТемпература"] == "1.5"
    assert result.loc[0, "КоличествоОсадков"] == "0"
    assert not any(column.endswith("_wx") for column in result.columns)
    assert all(column in result.columns for column in WEATHER_COLUMNS)
    assert diagnostics == []


def test_weather_success_preserves_order_count(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    source = pd.concat([_orders(), _orders().assign(MindboxID="client-2")], ignore_index=True)
    _write_weather(
        weather_path,
        [
            {
                "Дата": "2025-01-15",
                "Город": "Москва",
                "ПогодныеУсловия": "Ясно",
                "СредняяТемпература": "1.5",
                "КоличествоОсадков": "0",
            }
        ],
    )

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), source, str(weather_path)
    )

    assert len(result) == len(source) == 2
    assert result["MindboxID"].tolist() == ["client-1", "client-2"]


def test_weather_no_match_preserves_order_and_reports_coverage(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    _write_weather(
        weather_path,
        [
            {
                "Дата": "2025-01-16",
                "Город": "Москва",
                "ПогодныеУсловия": "Ясно",
                "СредняяТемпература": "1.5",
                "КоличествоОсадков": "0",
            }
        ],
    )
    source = _orders()
    source["ПогодныеУсловия"] = "Сохранить"
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), source, str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 1
    assert result.loc[0, "ПогодныеУсловия"] == "Сохранить"
    assert any("0/1" in message for message in diagnostics)


def test_partial_weather_coverage_is_reported_once(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    _write_weather(
        weather_path,
        [
            {
                "Дата": "2025-01-15",
                "Город": "Москва",
                "ПогодныеУсловия": "Ясно",
                "СредняяТемпература": "1.5",
                "КоличествоОсадков": "0",
            }
        ],
    )
    source = pd.concat(
        [
            _orders(),
            _orders().assign(MindboxID="client-2", Дата="2025-01-16"),
        ],
        ignore_index=True,
    )
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), source, str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 2
    assert result["ПогодныеУсловия"].notna().tolist() == [True, False]
    assert len(diagnostics) == 1
    assert "1/2" in diagnostics[0]


def test_missing_weather_source_preserves_orders_and_reports_warning(tmp_path):
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(),
        _orders(),
        str(tmp_path / "missing-weather.csv"),
        diagnostics=diagnostics,
    )

    assert len(result) == 1
    assert result.loc[0, "Город"] == "Москва"
    assert all(pd.isna(result.loc[0, column]) for column in WEATHER_COLUMNS)
    assert any("отсутств" in message.lower() for message in diagnostics)


@pytest.mark.parametrize(
    "read_error",
    [
        PermissionError("synthetic weather permission error"),
        pd.errors.ParserError("synthetic weather parser error"),
    ],
    ids=["permission", "parser"],
)
def test_weather_read_error_preserves_orders_and_reports_warning(
    tmp_path,
    monkeypatch,
    read_error,
):
    weather_path = tmp_path / "Погода.csv"
    weather_path.write_text("existing", encoding="utf-8")
    diagnostics = []
    monkeypatch.setattr(train_model_tab.pd, "read_csv", Mock(side_effect=read_error))

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), _orders(), str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 1
    assert result.loc[0, "Город"] == "Москва"
    assert any(str(read_error) in message for message in diagnostics)


@pytest.mark.parametrize("missing_column", ["Дата", "Город"])
def test_weather_schema_error_preserves_orders_and_reports_warning(
    tmp_path,
    missing_column,
):
    weather_path = tmp_path / "Погода.csv"
    weather = {
        "Дата": "2025-01-15",
        "Город": "Москва",
        "ПогодныеУсловия": "Ясно",
    }
    weather.pop(missing_column)
    _write_weather(weather_path, [weather])
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), _orders(), str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 1
    assert any("колон" in message.lower() for message in diagnostics)


def test_header_only_weather_preserves_orders_and_reports_warning(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    pd.DataFrame(columns=["Дата", "Город", *WEATHER_COLUMNS]).to_csv(
        weather_path,
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), _orders(), str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 1
    assert any("пуст" in message.lower() for message in diagnostics)


def test_zero_byte_weather_preserves_orders_and_reports_warning(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    weather_path.write_bytes(b"")
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), _orders(), str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 1
    assert result.loc[0, "Город"] == "Москва"
    assert all(pd.isna(result.loc[0, column]) for column in WEATHER_COLUMNS)
    assert any("прочитать" in message.lower() for message in diagnostics)


def test_all_missing_weather_values_are_reported_as_degraded(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    _write_weather(
        weather_path,
        [
            {
                "Дата": "2025-01-15",
                "Город": "Москва",
                "ПогодныеУсловия": pd.NA,
                "СредняяТемпература": pd.NA,
                "КоличествоОсадков": pd.NA,
            }
        ],
    )
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), _orders(), str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 1
    assert all(pd.isna(result.loc[0, column]) for column in WEATHER_COLUMNS)
    assert any("не содержит погодных значений" in message for message in diagnostics)


def test_missing_city_and_date_preserve_interactions(tmp_path):
    weather_path = tmp_path / "Погода.csv"
    _write_weather(
        weather_path,
        [
            {
                "Дата": "2025-01-15",
                "Город": "Москва",
                "ПогодныеУсловия": "Ясно",
                "СредняяТемпература": "1.5",
                "КоличествоОсадков": "0",
            }
        ],
    )
    source = pd.concat(
        [
            _orders().assign(Магазин="UNKNOWN"),
            _orders().assign(MindboxID="client-2", Дата=pd.NA),
        ],
        ignore_index=True,
    )
    diagnostics = []

    result = train_model_tab._enrich_orders_with_city_and_weather(
        _window(), source, str(weather_path), diagnostics=diagnostics
    )

    assert len(result) == 2
    assert result["MindboxID"].tolist() == ["client-1", "client-2"]
    assert any("0/2" in message for message in diagnostics)
