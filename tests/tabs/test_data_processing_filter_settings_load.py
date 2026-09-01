import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from Application.tabs import data_processing_tab


class _TextField:
    def __init__(self, text):
        self._text = str(text)

    def setText(self, text):
        self._text = str(text)

    def text(self):
        return self._text


class _SpinBox:
    def __init__(self, value):
        self._value = int(value)

    def setValue(self, value):
        self._value = int(value)

    def value(self):
        return self._value


class _ComboBox:
    def __init__(self, values, current):
        self._values = list(values)
        self._index = self._values.index(current)

    def findText(self, text):
        try:
            return self._values.index(text)
        except ValueError:
            return -1

    def setCurrentIndex(self, index):
        self._index = index

    def currentText(self):
        return self._values[self._index]


def _window(*, with_store_city_table=False):
    values = {
        "filter_date_from": _TextField("01.01.2020"),
        "filter_date_to": _TextField("31.01.2020"),
        "max_export_users_input": _SpinBox(111),
        "store_mode": _ComboBox(["В группе", "Вне группы"], "В группе"),
        "kind_mode": _ComboBox(["В группе", "Вне группы"], "В группе"),
        "_pending_store_selection": ["OLD STORE"],
        "_pending_kind_selection": ["OLD KIND"],
        "_pending_season_selection": ["OLD SEASON"],
        "_pending_export_kind_selection": ["OLD EXPORT KIND"],
        "_pending_store_city_map": {"OLD STORE": "OLD CITY"},
        "_store_city_map": {"OLD STORE": "OLD CITY"},
    }
    if with_store_city_table:
        values["store_city_table"] = object()
    return SimpleNamespace(**values)


def _snapshot(window):
    return {
        "date_from": window.filter_date_from.text(),
        "date_to": window.filter_date_to.text(),
        "max_export_users": window.max_export_users_input.value(),
        "store_mode": window.store_mode.currentText(),
        "kind_mode": window.kind_mode.currentText(),
        "stores_selected": list(window._pending_store_selection),
        "kinds_selected": list(window._pending_kind_selection),
        "seasons_selected": list(window._pending_season_selection),
        "export_kinds_selected": list(window._pending_export_kind_selection),
        "pending_store_city_map": dict(window._pending_store_city_map),
        "store_city_map": dict(window._store_city_map),
    }


def _new_settings(**overrides):
    settings = {
        "date_from": "01.02.2025",
        "date_to": "28.02.2025",
        "max_export_users": "250",
        "store_mode": "Вне группы",
        "kind_mode": "Вне группы",
        "stores_selected": ["NEW STORE"],
        "kinds_selected": ["NEW KIND"],
        "seasons_selected": ["NEW SEASON"],
        "export_kinds_selected": ["NEW EXPORT KIND"],
        "store_city_map": {"NEW STORE": "NEW CITY"},
    }
    settings.update(overrides)
    return settings


def _new_snapshot():
    settings = _new_settings(max_export_users=250)
    return {
        **settings,
        "pending_store_city_map": dict(settings["store_city_map"]),
    }


def _write_settings(path: Path, settings):
    path.write_text(
        json.dumps(settings, ensure_ascii=False),
        encoding="utf-8",
    )


def _prepare_error_capture(monkeypatch):
    errors = []
    resets = []
    successes = []
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_error",
        lambda window, message: errors.append(message),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "schedule_status_reset",
        lambda window, seconds: resets.append(seconds),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_ok",
        lambda window, message: successes.append(message),
    )
    return errors, resets, successes


def test_missing_filter_settings_keeps_existing_state(tmp_path, monkeypatch):
    settings_path = tmp_path / "missing-filter-settings.json"
    monkeypatch.setattr(
        data_processing_tab,
        "order_filters_settings_path",
        lambda: str(settings_path),
    )
    errors, resets, successes = _prepare_error_capture(monkeypatch)
    window = _window()
    old_state = _snapshot(window)

    data_processing_tab.load_order_filter_settings(window)

    assert _snapshot(window) == old_state
    assert errors == []
    assert resets == []
    assert successes == []


def test_valid_filter_settings_replace_complete_state(tmp_path, monkeypatch):
    settings_path = tmp_path / "filter-settings.json"
    _write_settings(settings_path, _new_settings())
    monkeypatch.setattr(
        data_processing_tab,
        "order_filters_settings_path",
        lambda: str(settings_path),
    )
    errors, resets, successes = _prepare_error_capture(monkeypatch)
    load_cities = Mock(
        side_effect=lambda window: setattr(window, "_cities", ["NEW CITY"])
    )
    refresh_table = Mock()
    monkeypatch.setattr(
        data_processing_tab,
        "load_cities_from_coordinates_file",
        load_cities,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "refresh_store_city_table",
        refresh_table,
    )
    window = _window(with_store_city_table=True)

    data_processing_tab.load_order_filter_settings(window)

    assert _snapshot(window) == _new_snapshot()
    assert window._cities == ["NEW CITY"]
    load_cities.assert_called_once_with(window)
    refresh_table.assert_called_once_with(window)
    assert errors == []
    assert resets == []
    assert successes == []


def test_late_settings_conversion_failure_keeps_complete_old_state(
    tmp_path, monkeypatch
):
    settings_path = tmp_path / "filter-settings.json"
    _write_settings(
        settings_path,
        _new_settings(store_city_map=[["invalid dictionary entry"]]),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "order_filters_settings_path",
        lambda: str(settings_path),
    )
    errors, resets, successes = _prepare_error_capture(monkeypatch)
    window = _window()
    old_state = _snapshot(window)

    data_processing_tab.load_order_filter_settings(window)

    assert _snapshot(window) == old_state
    assert errors == ["Настройки фильтров повреждены"]
    assert resets == [5]
    assert successes == []


def test_filter_settings_permission_error_keeps_complete_old_state(
    tmp_path, monkeypatch
):
    settings_path = tmp_path / "filter-settings.json"
    settings_path.write_text("synthetic existing settings", encoding="utf-8")
    monkeypatch.setattr(
        data_processing_tab,
        "order_filters_settings_path",
        lambda: str(settings_path),
    )
    errors, resets, successes = _prepare_error_capture(monkeypatch)
    open_mock = Mock(side_effect=PermissionError("synthetic settings read failure"))
    monkeypatch.setattr(data_processing_tab, "open", open_mock, raising=False)
    window = _window()
    old_state = _snapshot(window)

    data_processing_tab.load_order_filter_settings(window)

    assert _snapshot(window) == old_state
    open_mock.assert_called_once_with(str(settings_path), "r", encoding="utf-8")
    assert errors == ["Настройки фильтров повреждены"]
    assert resets == [5]
    assert successes == []


def test_malformed_filter_settings_json_keeps_complete_old_state(
    tmp_path, monkeypatch
):
    settings_path = tmp_path / "filter-settings.json"
    settings_path.write_text('{"date_from": "01.02.2025",', encoding="utf-8")
    monkeypatch.setattr(
        data_processing_tab,
        "order_filters_settings_path",
        lambda: str(settings_path),
    )
    errors, resets, successes = _prepare_error_capture(monkeypatch)
    window = _window()
    old_state = _snapshot(window)

    data_processing_tab.load_order_filter_settings(window)

    assert _snapshot(window) == old_state
    assert errors == ["Настройки фильтров повреждены"]
    assert resets == [5]
    assert successes == []
