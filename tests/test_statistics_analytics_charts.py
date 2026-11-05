"""Saved-aggregate chart data, native rendering and isolated chart controls."""
from copy import deepcopy
from decimal import Decimal
import subprocess
import sys

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, Qt
from PyQt6.QtCharts import QBarSeries
from PyQt6.QtWidgets import QLabel

from Application import statistics_charts as charts
from Application.statistics_chart_data import (
    buyer_frequency_chart_data, purchase_policy_comparison_chart_data,
    buyer_turnover_chart_data, rfm_segments_chart_data, cohort_repeat_chart_data,
    cohort_retention_heatmap_data, RFM_WINDOWS,
)
from Application.statistics_analytics_charts import RfmSegmentsWidget, CohortRetentionHeatmap
from Application.statistics_purchase_policy import STRICT_COMPLETED, EXTENDED_COMPLETED
from Application.statistics_rfm_presentation import SEGMENT_LABELS
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import app, window

__all__ = ["app", "window"]
pytestmark = pytest.mark.usefixtures("window_settings")


@pytest.fixture
def saved():
    frequency = [["1", 2, 0., 2], ["2", 0, 0., 0], ["3–5", 3, 0., 12],
                 ["6–10", 0, 0., 0], ["11+", 1, 0., 12]]
    strict = dict(buyers=6, repeat_buyers=4, orders=26, frequency=frequency,
                  money={"RUB": {"quintiles": [["0", "0", 2, 0., "0", 0, 0., "0", None],
                                               ["10.125", "10.125", 1, 0., "10.125", 0, 0., "0", None]]},
                         "KZT": {"quintiles": [["999999999999999999999999.123456789",
                                                "999999999999999999999999.123456789", 1, 0.,
                                                "999999999999999999999999.123456789", 0, 0., "0", None]]}})
    extended = dict(buyers=9, repeat_buyers=5, orders=40)
    windows = {}
    for number, code in enumerate(RFM_WINDOWS):
        windows[code] = {"currencies": {}}
        for currency in ("RUB", "KZT"):
            counts = [0, number+1, 0, 2, 0, 0, 3 if currency=="RUB" else 7]
            windows[code]["currencies"][currency] = {
                "segments": [[key, counts[i], None] for i, key in reversed(list(enumerate(SEGMENT_LABELS)))]}
    horizons = {str(day): dict(eligible=eligible, repeated=repeated, repeat_rate=value)
                for day, eligible, repeated, value in ((30, 2, 0, 0.), (60, 1, 1, 100.),
                                                       (90, 0, 0, None), (180, 2, 1, 50.), (365, 5, 2, 40.))}
    cells = [dict(month_offset=0, buyers=7)]
    for offset in range(1, 13):
        cells.append(dict(month_offset=offset, retention_rate=0. if offset==1 else 8.98 if offset==2 else
                          50. if offset==4 else None,
                          eligible=7 if offset<3 else 2 if offset==4 else 0,
                          active=0 if offset==1 else 1, status="FULL" if offset<3 else
                          "PARTIAL" if offset==4 else "UNAVAILABLE",
                          censored=0 if offset<3 else 3, coverage_incomplete=2 if offset==4 else 0))
    return {"buyer_statistics": {"policies": {STRICT_COMPLETED: strict, EXTENDED_COMPLETED: extended}},
            "rfm_statistics": {"policies": {STRICT_COMPLETED: {"windows": windows}}},
            "cohort_statistics": {"policies": {STRICT_COMPLETED: {
                "summary": {"horizons": horizons}, "cohorts": {
                    "2026-02": {"cohort_size": 7, "retention": deepcopy(cells)},
                    "2026-01": {"cohort_size": 7, "retention": cells}}}}}}


def test_new_data_module_import_needs_no_qt_widgets():
    code = ("import sys;import Application.statistics_chart_data;"
            "assert 'PyQt6.QtWidgets' not in sys.modules")
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_frequency_uses_strict_five_saved_counts_keeps_zeros_and_ignores_legacy(saved):
    before = deepcopy(saved)
    saved["purchase_order_distribution"] = [["legacy", 999999, 100]]
    data = buyer_frequency_chart_data(saved)
    assert data.labels == ("1 заказ", "2 заказа", "3–5 заказов", "6–10 заказов", "11+ заказов")
    assert data.series == (("Покупателей", (2, 0, 3, 0, 1)),)
    assert sum(data.series[0][1]) == 6
    assert saved["buyer_statistics"] == before["buyer_statistics"]
    assert buyer_frequency_chart_data({}).empty
    for row in saved["buyer_statistics"]["policies"][STRICT_COMPLETED]["frequency"]:
        row[1] = 0
    data = buyer_frequency_chart_data(saved)
    assert len(data.labels) == 5 and data.empty


def test_policy_comparison_preserves_separate_absolute_bases(saved):
    before = deepcopy(saved)
    data = purchase_policy_comparison_chart_data(saved)
    assert data.labels == ("Покупатели", "Повторные покупатели", "Заказы")
    assert data.series == (("Подтверждённые покупки", (6, 4, 26)),
                           ("Расширенный учёт покупок", (9, 5, 40)))
    assert saved == before and purchase_policy_comparison_chart_data({}).empty


def test_quintiles_preserve_decimal_ranges_fewer_groups_and_known_zero(saved):
    before = deepcopy(saved)
    rub = buyer_turnover_chart_data(saved, "RUB")
    kzt = buyer_turnover_chart_data(saved, "KZT")
    assert rub.labels == ("Группа 1", "Группа 2")
    assert rub.series[0][1] == (Decimal(0), Decimal("10.125"))
    assert rub.ranges == ((Decimal(0), Decimal(0)), (Decimal("10.125"), Decimal("10.125")))
    assert len(kzt.labels) == 1
    assert kzt.series[0][1][0] == Decimal("999999999999999999999999.123456789")
    assert ".123456789" not in kzt.tooltips[0][0] and ",123456789" in kzt.tooltips[0][0]
    saved["buyer_statistics"]["policies"][STRICT_COMPLETED]["money"]["RUB"]["quintiles"] = [
        ["0", "0", 5, 100., "0", 0, 0., "0", None]]
    zero = buyer_turnover_chart_data(saved, "RUB")
    assert zero.labels == ("Группа 1",) and not zero.empty and zero.series[0][1] == (Decimal(0),)
    assert buyer_turnover_chart_data({}, "RUB").empty
    assert before["buyer_statistics"]["policies"][STRICT_COMPLETED]["money"]["KZT"] == saved["buyer_statistics"]["policies"][STRICT_COMPLETED]["money"]["KZT"]


@pytest.mark.parametrize("code", RFM_WINDOWS)
@pytest.mark.parametrize("currency", ["RUB", "KZT"])
def test_rfm_all_windows_currencies_fixed_seven_segments_without_mutation(saved, code, currency):
    before = deepcopy(saved)
    data = rfm_segments_chart_data(saved, code, currency)
    rows = saved["rfm_statistics"]["policies"][STRICT_COMPLETED]["windows"][code]["currencies"][currency]["segments"]
    counts = {key: count for key, count, _ in rows}
    assert data.labels == tuple(label for label, _ in SEGMENT_LABELS.values())
    assert data.series[0][1] == tuple(counts[key] for key in SEGMENT_LABELS)
    assert len(data.labels) == 7 and data.series[0][1][0] == 0
    assert data.labels[-1] == "Оценка недоступна"
    assert saved == before and rfm_segments_chart_data({}).empty


def test_repeat_keeps_saved_rates_denominators_and_unavailable_separate_from_zero(saved):
    before = deepcopy(saved)
    data = cohort_repeat_chart_data(saved)
    assert data.labels == ("30 дней", "60 дней", "90 дней", "180 дней", "365 дней")
    assert data.series[0][1] == (0., 100., None, 50., 40.)
    assert data.bases == ((2, 0), (1, 1), (0, 0), (2, 1), (5, 2))
    assert "Нет данных" in data.tooltips[0][2]
    assert not data.empty and saved == before
    assert cohort_repeat_chart_data({}).empty


def test_heatmap_months_columns_zero_none_partial_future_and_no_mutation(saved):
    before = deepcopy(saved)
    data = cohort_retention_heatmap_data(saved)
    assert [row.month for row in data.rows] == ["2026-01", "2026-02"]
    assert data.columns == tuple(f"M{i}" for i in range(1, 13))
    cells = data.rows[0].cells
    assert cells[0].value == 0. and cells[0].status == "FULL"
    assert cells[1].value == 8.98
    assert cells[2].value is None and cells[2].censored == 3
    assert cells[3].value == 50. and cells[3].status == "PARTIAL"
    assert cells[3].eligible == 2 and cells[3].coverage_incomplete == 2
    assert cells[-1].value is None
    assert saved == before and cohort_retention_heatmap_data({}).rows == ()


@pytest.mark.parametrize("dark", [False, True])
def test_native_policy_bars_legend_hover_distinct_colors_and_original_series(saved, app, dark):
    apply_app_theme(app, dark)
    data = purchase_policy_comparison_chart_data(saved)
    chart = charts.create_chart(data)
    try:
        chart.resize(900, 340);chart.show();app.processEvents()
        series = chart.chart().series()[0]
        assert isinstance(series, QBarSeries)
        bars = series.barSets()
        assert len(bars) == 2 and chart.chart().legend().isVisible()
        assert [[bar.at(i) for i in range(3)] for bar in bars] == [[6,4,26],[9,5,40]]
        assert bars[0].color() != bars[1].color()
        series.hovered.emit(True, 1, bars[1])
        assert "Расширенный учёт покупок" in chart.tooltip.text() and "5" in chart.tooltip.text()
    finally:
        chart.close();chart.deleteLater()


def test_native_rate_marks_true_zero_and_missing_without_claiming_zero(app, saved):
    data = cohort_repeat_chart_data(saved)
    chart = charts.create_chart(data)
    try:
        assert chart.count_axis.max() == 100
        assert [item.text() for _, item in chart._point_labels] == ["0 %", "—"]
        assert chart.data.series[0][1][2] is None
        assert chart.count_axis.titleText() == "Повторная доля, %"
    finally:
        chart.deleteLater()


@pytest.mark.parametrize("dark", [False, True])
def test_heatmap_visual_states_theme_long_lists_and_headers(app, saved, dark):
    apply_app_theme(app, dark)
    before = deepcopy(saved)
    row = saved["cohort_statistics"]["policies"][STRICT_COMPLETED]["cohorts"]["2026-01"]
    saved["cohort_statistics"]["policies"][STRICT_COMPLETED]["cohorts"] = {
        f"{2000+i//12:04d}-{i%12+1:02d}": row for i in range(1000)}
    view = CohortRetentionHeatmap(saved)
    try:
        view.resize(850,420);view.show();app.processEvents()
        model = view.model()
        assert model.rowCount() == 1000 and model.columnCount() == 14
        assert [model.headerData(i,Qt.Orientation.Horizontal) for i in (11,12,13)] == ["M10","M11","M12"]
        assert model.data(model.index(0,2)) == "0,00 %"
        assert model.data(model.index(0,3)) == "8,98 %"
        assert model.data(model.index(0,4)) == "—"
        assert model.data(model.index(0,5)) == "50,00 % (частично)"
        brush = lambda column:model.data(model.index(0,column),Qt.ItemDataRole.BackgroundRole)
        assert brush(2).color() != brush(3).color() != brush(4).color()
        assert brush(5).style() == Qt.BrushStyle.Dense4Pattern
        # Test actual themed painting, not just roles ignored by Material QSS.
        image = view.viewport().grab().toImage()
        for column in (2, 3, 4):
            rect = view.visualRect(model.index(0, column))
            assert image.pixelColor(rect.left()+5, rect.top()+5) == brush(column).color()
        rect = view.visualRect(model.index(0, 5))
        colors = {image.pixelColor(rect.left()+x, rect.top()+y).name()
                  for x in range(4, 12) for y in range(4, 12)}
        assert len(colors) > 1, "Partial coverage must have a visible hatch"
        assert view.horizontalScrollBar().maximum()>0 and view.verticalScrollBar().maximum()>0
        assert view.height()<=420 and model.data(model.index(999,13)) == "—"
        assert all(model.data(model.index(0,c),Qt.ItemDataRole.ToolTipRole) is None for c in range(14))
        assert before["cohort_statistics"]["policies"][STRICT_COMPLETED]["cohorts"]["2026-01"] == row
    finally:
        view.close();view.deleteLater()


def test_rfm_selectors_only_replace_the_plot_without_io(saved, app, monkeypatch):
    import sqlite3
    from Application import dataset_statistics, statistics_slices
    def forbidden(*a, **k):pytest.fail("RFM chart must use saved aggregates only")
    monkeypatch.setattr(sqlite3,"connect",forbidden)
    monkeypatch.setattr(dataset_statistics,"calculate_statistics",forbidden)
    monkeypatch.setattr(statistics_slices,"calculate_cached_statistics",forbidden)
    before=deepcopy(saved)
    widget=RfmSegmentsWidget(saved)
    try:
        assert widget.window_selector.currentData()=="LAST_12M"
        assert widget.currency_selector.currentData()=="RUB"
        window_id=id(widget.window_selector);currency_id=id(widget.currency_selector)
        for w in range(3):
            for c in range(2):
                widget.window_selector.setCurrentIndex(w);widget.currency_selector.setCurrentIndex(c)
                QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents()
                assert id(widget.window_selector)==window_id and id(widget.currency_selector)==currency_id
                assert widget.plot.data==rfm_segments_chart_data(saved,RFM_WINDOWS[w],("RUB","KZT")[c])
                assert len(widget.findChildren(charts.StatisticsChart))==1
        assert saved==before
    finally:
        widget.close();widget.deleteLater()


def test_legacy_and_empty_chart_widgets_do_not_query_data(app):
    widget=RfmSegmentsWidget({})
    heatmap=CohortRetentionHeatmap({})
    try:
        assert widget.plot.objectName()=="statisticsChartEmpty"
        assert any("Нет данных" in label.text() for label in widget.plot.findChildren(QLabel))
        assert heatmap.model().rowCount()==0 and heatmap.empty_label.isVisibleTo(heatmap)
    finally:
        widget.deleteLater();heatmap.deleteLater()


def test_valid_all_zero_rfm_keeps_seven_visible_categories(saved, app):
    for row in saved["rfm_statistics"]["policies"][STRICT_COMPLETED]["windows"]["LAST_12M"]["currencies"]["RUB"]["segments"]:
        row[1] = 0
    data = rfm_segments_chart_data(saved)
    assert not data.empty and len(data.labels) == 7 and data.series[0][1] == (0,)*7
    chart = charts.create_chart(data)
    try:
        assert isinstance(chart, charts.StatisticsChart) and len(chart._product_labels) == 7
    finally:
        chart.deleteLater()
