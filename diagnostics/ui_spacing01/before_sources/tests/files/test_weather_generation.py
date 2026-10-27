from types import SimpleNamespace

import pandas as pd

from Application.files import files_processing


def _weather_frame(city, status):
    frame = pd.DataFrame(
        [
            {
                "Дата": "2025-01-01",
                "Город": city,
                "Широта": 55.0,
                "Долгота": 37.0,
                "СредняяТемпература": 1.0,
                "КоличествоОсадков": 0.0,
                "КодПогоды": 0,
                "ПогодныеУсловия": "Ясно",
            }
        ]
    )
    frame.attrs["weather_request_status"] = status
    return frame


def test_network_failure_is_exposed_in_daily_weather_result(monkeypatch):
    monkeypatch.setattr(files_processing, "_get_json_with_retry", lambda *args, **kwargs: None)

    result = files_processing._download_daily_weather_by_coordinates(
        city="Москва",
        latitude=55.75,
        longitude=37.61,
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    assert result.attrs["weather_request_status"] == "failed"
    assert result[
        [
            "ПогодныеУсловия",
            "СредняяТемпература",
            "КоличествоОсадков",
            "КодПогоды",
        ]
    ].isna().all().all()


def test_successful_daily_weather_response_is_exposed(monkeypatch):
    monkeypatch.setattr(
        files_processing,
        "_get_json_with_retry",
        lambda *args, **kwargs: {
            "daily": {
                "time": ["2025-01-01", "2025-01-02"],
                "temperature_2m_mean": [1.5, 2.5],
                "precipitation_sum": [0.0, 1.0],
                "weather_code": [0, 61],
            }
        },
    )

    result = files_processing._download_daily_weather_by_coordinates(
        city="Москва",
        latitude=55.75,
        longitude=37.61,
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    assert result.attrs["weather_request_status"] == "success"
    assert result["Дата"].tolist() == ["2025-01-01", "2025-01-02"]
    assert result["СредняяТемпература"].tolist() == [1.5, 2.5]
    assert result["КоличествоОсадков"].tolist() == [0.0, 1.0]
    assert result["ПогодныеУсловия"].tolist() == ["Ясно", "Дождь (слабый)"]


def test_daily_weather_response_without_dates_is_empty(monkeypatch):
    monkeypatch.setattr(
        files_processing,
        "_get_json_with_retry",
        lambda *args, **kwargs: {"daily": {"time": []}},
    )

    result = files_processing._download_daily_weather_by_coordinates(
        city="Москва",
        latitude=55.75,
        longitude=37.61,
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    assert result.attrs["weather_request_status"] == "empty"
    assert len(result) == 2
    assert result[
        [
            "ПогодныеУсловия",
            "СредняяТемпература",
            "КоличествоОсадков",
            "КодПогоды",
        ]
    ].isna().all().all()


def test_weather_generation_aggregates_partial_api_outcome(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(files_processing, "set_status_processing", lambda *args: None)
    statuses = {"Москва": "success", "Омск": "failed"}
    monkeypatch.setattr(
        files_processing,
        "_download_daily_weather_by_coordinates",
        lambda city, **kwargs: _weather_frame(city, statuses[city]),
    )
    coordinates = pd.DataFrame(
        [
            {"Город": "Москва", "Широта": 55.75, "Долгота": 37.61},
            {"Город": "Омск", "Широта": 54.99, "Долгота": 73.37},
        ]
    )

    result = files_processing._download_weather_for_coordinates_file(
        SimpleNamespace(), coordinates, "2025-01-01", "2025-01-01"
    )

    assert result.attrs["weather_total_cities"] == 2
    assert result.attrs["weather_successful_cities"] == 1
    assert result.attrs["weather_failed_cities"] == 1
    assert result.attrs["weather_empty_cities"] == 0
