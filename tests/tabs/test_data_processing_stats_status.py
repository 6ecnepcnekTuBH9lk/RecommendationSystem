from types import SimpleNamespace

import pytest

from Application.tabs import data_processing_tab


class _Application:
    @staticmethod
    def processEvents():
        pass


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


def _patch_status_flow(monkeypatch, outcomes):
    calls = []
    statuses = []

    for name in ("orders", "views", "favorites"):
        function_name = f"analyze_{name}_full_dataset"

        def analyze(window, current_name=name):
            calls.append(current_name)
            return outcomes[current_name]

        monkeypatch.setattr(data_processing_tab, function_name, analyze)

    monkeypatch.setattr(data_processing_tab, "QApplication", _Application)
    monkeypatch.setattr(
        data_processing_tab,
        "save_order_filter_settings",
        lambda window: True,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_processing",
        lambda window, text: statuses.append(("processing", text)),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_ok",
        lambda window, text: statuses.append(("success", text)),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_error",
        lambda window, text: statuses.append(("error", text)),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "schedule_status_reset",
        lambda window, seconds: None,
    )

    return SimpleNamespace(), calls, statuses


def test_all_statistics_success_allows_filters_applied_status(monkeypatch):
    window, calls, statuses = _patch_status_flow(
        monkeypatch,
        {"orders": True, "views": True, "favorites": True},
    )

    assert data_processing_tab.apply_filters_all_stats(window) is True
    assert calls == ["orders", "views", "favorites"]

    calls.clear()
    data_processing_tab.save_and_apply_filters(window)

    assert calls == ["orders", "views", "favorites"]
    assert ("success", "Фильтры применены") in statuses


@pytest.mark.parametrize("failed_analysis", ["orders", "views", "favorites"])
def test_statistics_failure_prevents_filters_applied_status(
    monkeypatch, failed_analysis
):
    outcomes = {"orders": True, "views": True, "favorites": True}
    outcomes[failed_analysis] = False
    window, calls, statuses = _patch_status_flow(monkeypatch, outcomes)

    assert data_processing_tab.apply_filters_all_stats(window) is False
    assert calls == ["orders", "views", "favorites"]

    calls.clear()
    data_processing_tab.save_and_apply_filters(window)

    assert calls == ["orders", "views", "favorites"]
    assert ("success", "Фильтры применены") not in statuses
    assert any(kind == "error" for kind, _text in statuses)


def test_real_orders_read_error_does_not_end_with_false_success(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    (input_dir / "Заказы.csv").write_text("synthetic", encoding="utf-8")

    window = SimpleNamespace(order_full_output_layout=_Layout())
    statuses = []
    page_errors = []

    monkeypatch.setattr(data_processing_tab, "QApplication", _Application)
    monkeypatch.setattr(data_processing_tab, "QLabel", _Label)
    monkeypatch.setattr(data_processing_tab, "clear_layout", lambda *args: None)
    monkeypatch.setattr(
        data_processing_tab,
        "set_order_filters_enabled",
        lambda *args: None,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "vyvod_zaglyschek",
        lambda **kwargs: page_errors.append(kwargs["text"]),
    )
    monkeypatch.setattr(
        data_processing_tab.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("synthetic statistics read error")
        ),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "analyze_views_full_dataset",
        lambda aboba: True,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "analyze_favorites_full_dataset",
        lambda aboba: True,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "save_order_filter_settings",
        lambda aboba: True,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_processing",
        lambda aboba, text: statuses.append(("processing", text)),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_ok",
        lambda aboba, text: statuses.append(("success", text)),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_error",
        lambda aboba, text: statuses.append(("error", text)),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "schedule_status_reset",
        lambda *args: None,
    )

    data_processing_tab.save_and_apply_filters(window)

    assert any("synthetic statistics read error" in text for text in page_errors)
    assert ("success", "Фильтры применены") not in statuses
    assert any(kind == "error" for kind, _text in statuses)


@pytest.mark.parametrize(
    ("analyze", "layout_name"),
    [
        (data_processing_tab.analyze_orders_full_dataset, "order_full_output_layout"),
        (data_processing_tab.analyze_views_full_dataset, "views_full_output_layout"),
        (
            data_processing_tab.analyze_favorites_full_dataset,
            "favorites_full_output_layout",
        ),
    ],
)
def test_missing_statistics_file_remains_normal_state(
    tmp_path, monkeypatch, analyze, layout_name
):
    monkeypatch.chdir(tmp_path)
    window = SimpleNamespace(**{layout_name: _Layout()})

    monkeypatch.setattr(data_processing_tab, "QLabel", _Label)
    monkeypatch.setattr(data_processing_tab, "clear_layout", lambda *args: None)
    monkeypatch.setattr(
        data_processing_tab,
        "vyvod_zaglyschek",
        lambda **kwargs: None,
    )
    for name in (
        "refresh_export_kind_values_from_nomenclature_file",
        "refresh_kind_values_from_loaded_files",
        "refresh_season_values_from_nomenclature_file",
        "update_filter_controls_availability",
        "update_filter_summary",
    ):
        monkeypatch.setattr(data_processing_tab, name, lambda *args: None)

    assert analyze(window) is True
