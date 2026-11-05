"""Synthetic geometry and content regressions for statistics UI polish."""
from copy import deepcopy

import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QTableWidget, QVBoxLayout, QWidget

from Application.tabs import dataset_statistics_tab as ui
from Application.statistics_analytics_charts import analytics_chart, RfmSegmentsWidget
from Application.statistics_chart_data import (
    buyer_frequency_chart_data, purchase_policy_comparison_chart_data,
    buyer_turnover_chart_data, rfm_segments_chart_data, cohort_repeat_chart_data,
)
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import app, window, rich_customer_result

__all__ = ["app", "window", "rich_customer_result"]
pytestmark = pytest.mark.usefixtures("window_settings")


@pytest.mark.parametrize("dark", [False, True])
def test_table_rows_headers_widths_alignment_exact_numeric_text_and_last_column(app, dark):
    apply_app_theme(app, dark)
    host = QWidget()
    layout = QVBoxLayout(host)
    amount = ui._decimal_display("15606214493.67")
    huge = ui._decimal_display("1234567890"*18 + ".67")
    rows = [("Количество", 499111), ("Сумма", amount), ("Большая сумма", huge),
            ("Доля, %", 43.36), ("Период", "2022-01"), ("Код", "000123"),
            ("Доля без смены точности, %", "12.3400"), ("Доля со знаком, %", "43,36 %")]
    table = ui._table(layout, ["Показатель", "Значение"], rows)
    wide = ui._table(layout, ["Показатель", "Сумма, RUB", "Доля, %", "Последняя колонка"],
                     [("Длинный показатель " * 8, amount, 0., "Конец")])
    host.resize(1100, 800)
    host.show()
    try:
        QTest.qWait(30)
        assert table.item(0, 1).text() == "499 111"
        assert table.item(1, 1).text() == "15 606 214 493.67"
        assert table.item(2, 1).text() == huge
        assert table.item(3, 1).text() == "43,36 %"
        assert table.item(4, 1).text() == "2022-01"
        assert table.item(5, 1).text() == "000123"
        assert table.item(6, 1).text() == "12,3400 %"
        assert table.item(7, 1).text() == "43,36 %"
        assert ui._cell_text("50.00%", percentage=True) == "50,00 %"
        for row in (0, 1, 2, 3, 6, 7):
            assert table.item(row, 1).textAlignment() & Qt.AlignmentFlag.AlignRight
        for row in (4, 5):
            assert table.item(row, 1).textAlignment() & Qt.AlignmentFlag.AlignLeft
        assert all(table.item(r, 0).textAlignment() & Qt.AlignmentFlag.AlignLeft for r in range(6))
        assert table.columnWidth(1) > table.fontMetrics().horizontalAdvance(huge) + 12
        assert table.horizontalScrollBar().maximum() > 0
        assert table.verticalScrollBar().maximum() == 0
        assert all(table.rowHeight(r) >= table.sizeHintForRow(r) for r in range(6))
        assert table.rowHeight(0) >= table.fontMetrics().height() + 2 * ui.CONTROL_SPACING
        assert table.horizontalHeader().height() == wide.horizontalHeader().height()
        assert wide.item(0, 2).text() == "0,00 %"
        for widget in (table, wide):
            widget.horizontalScrollBar().setValue(widget.horizontalScrollBar().maximum())
            QTest.qWait(10)
            rect = widget.visualItemRect(widget.item(0, widget.columnCount()-1))
            assert rect.right() < widget.viewport().width()
            assert rect.right() >= 0
        host.resize(600, 800)
        QTest.qWait(20)
        assert table.item(2, 1).text() == huge
    finally:
        host.close()
        host.deleteLater()


@pytest.mark.parametrize("dark", [False, True])
def test_short_tables_fill_width_and_rows_react_to_font_without_header_padding(app, dark):
    apply_app_theme(app, dark)
    host = QWidget()
    layout = QVBoxLayout(host)
    first = ui._table(layout, ["Показатель", "Значение"], [("Короткий", 1), ("Другой", 2)])
    second = ui._table(layout, ["Показатель", "Значение"], [("Короткий", 3)])
    first.resizeRowsToContents()
    host.resize(1200, 500)
    host.show()
    try:
        QTest.qWait(30)
        assert first.rowHeight(0) == second.rowHeight(0)
        for table in (first, second):
            assert abs(sum(table.columnWidth(c) for c in range(2)) - table.viewport().width()) <= 2
            assert table.horizontalHeader().sectionViewportPosition(1) + table.columnWidth(1) == table.viewport().width()
            assert table.verticalScrollBar().maximum() == 0
            assert abs(table.viewport().height()-sum(table.rowHeight(r) for r in range(table.rowCount()))) <= 2
        original = first.rowHeight(0)
        font = first.font()
        font.setPointSizeF(font.pointSizeF()*1.6)
        first.setFont(font)
        QTest.qWait(30)
        assert first.rowHeight(0) > original
        assert first.verticalScrollBar().maximum() == 0
    finally:
        host.close()
        host.deleteLater()


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
def test_kpi_rows_are_balanced_adaptive_and_keep_all_values(app, count):
    host = QWidget()
    layout = QVBoxLayout(host)
    labels = ui._cards(layout, [(f"Показатель {i}", i) for i in range(count)])
    group = host.findChild(ui._StatisticsCards)
    host.resize(1400, 600)
    host.show()
    try:
        QTest.qWait(30)
        expected = 3 if count in (5, 6) else count
        assert group.columns == expected, (group.width(), group.fontMetrics().horizontalAdvance("0"), group.font().pointSizeF(), host.width())
        rows = {}
        for card in group.cards:
            rows.setdefault(card.y(), []).append(card)
        assert [len(row) for row in rows.values()] == ([3, 3] if count == 6 else [3, 2] if count == 5 else [count])
        for row in rows.values():
            assert max(card.width() for card in row)-min(card.width() for card in row) <= 1
            assert row[0].x() == 0
            assert row[-1].geometry().right() == group.width()-1
        host.resize(420, 700)
        QTest.qWait(30)
        assert group.columns < expected
        assert [label.text() for label in labels] == [str(i) for i in range(count)]
        assert len(group.cards) == count
        assert all(card.isVisible() for card in group.cards)
    finally:
        host.close()
        host.deleteLater()


@pytest.mark.parametrize("dark", [False, True])
def test_four_real_six_card_blocks_and_all_39_tables_keep_sources(window, rich_customer_result, dark):
    host, tab = window
    apply_app_theme(ui.QApplication.instance(), dark)
    before = deepcopy(rich_customer_result)
    host.resize(1400, 950)
    tab.render(rich_customer_result)
    tab.sections.setCurrentIndex(3)
    groups = []
    for index in range(5):
        tab.customer_sections.setCurrentIndex(index)
        QTest.qWait(25)
        groups.extend(group for group in tab.customer_sections.widget(index).findChildren(ui._StatisticsCards)
                      if len(group.cards) == 6)
    assert len(groups) == 4
    for group in groups:
        assert group.columns == 3
        assert sorted({card.y() for card in group.cards}) == [group.cards[0].y(), group.cards[3].y()]
        assert abs(group.cards[0].width()-group.cards[-1].width()) <= 1
    assert len(tab.customer_sections.findChildren(QTableWidget)) == 39
    assert rich_customer_result == before


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [620, 1366])
def test_only_requested_axis_titles_hidden_native_legend_gap_and_all_series(app, rich_customer_result, dark, width):
    apply_app_theme(app, dark)
    builders = (buyer_frequency_chart_data, purchase_policy_comparison_chart_data,
                lambda r: buyer_turnover_chart_data(r, "RUB"),
                lambda r: buyer_turnover_chart_data(r, "KZT"), rfm_segments_chart_data)
    for builder in builders:
        data = builder(rich_customer_result)
        widget = analytics_chart(data)
        widget.resize(width, 340)
        widget.show()
        try:
            QTest.qWait(20)
            assert widget.count_axis.titleText() == ""
            assert widget.count_axis.max() > 0
            assert widget.data == data
            assert len(widget.chart().series()[0].barSets()) == len(data.series)
            if len(data.series) == 2:
                legend = widget.chart().legend()
                assert legend.isVisible() and legend.isAttachedToChart()
                assert legend.getContentsMargins()[1] == 16
                assert len(legend.markers()) == 2
                bars = widget.chart().series()[0].barSets()
                assert bars[0].color() != bars[1].color()
                labels_bottom = max(item.mapRectToParent(item.boundingRect()).bottom() for item in widget._bar_labels)
                # Native marker group starts below our fully rendered category labels.
                top = min(legend.mapRectToParent(child.mapRectToParent(child.boundingRect())).top()
                          for child in legend.childItems())
                assert top-labels_bottom >= 10
                assert widget.chart().plotArea().height() >= 150
        finally:
            widget.close()
            widget.deleteLater()
    rate = analytics_chart(cohort_repeat_chart_data(rich_customer_result))
    try:
        assert rate.count_axis.titleText() == "Повторная доля, %"
    finally:
        rate.deleteLater()


def test_wheel_short_table_bubbles_to_page_long_table_scrolls_and_edges_bubble(app):
    host = QWidget()
    outer = QVBoxLayout(host)
    page, layout = ui._scroll_page()
    outer.addWidget(page)
    short = ui._table(layout, ["Показатель", "Значение"], [("A", 1)])
    long = ui._table(layout, ["Показатель", "Значение"], [(str(i), i) for i in range(100)])
    spacer = QWidget()
    spacer.setFixedHeight(900)
    layout.addWidget(spacer)
    host.resize(900, 800)
    host.show()

    def wheel(table, delta):
        center = table.viewport().rect().center()
        event = QWheelEvent(QPointF(center), QPointF(table.viewport().mapToGlobal(center)),
                            QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                            Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
        ui.QApplication.sendEvent(table.viewport(), event)
        QTest.qWait(20)

    try:
        QTest.qWait(30)
        page.verticalScrollBar().setValue(0)
        wheel(short, -120)
        assert page.verticalScrollBar().value() > 0
        position = page.verticalScrollBar().value()
        wheel(long, -120)
        assert long.verticalScrollBar().value() > 0
        assert page.verticalScrollBar().value() == position
        long.scrollToBottom()
        wheel(long, -120)
        assert page.verticalScrollBar().value() > position
        assert long.height() == 450
    finally:
        host.close()
        host.deleteLater()


def test_rfm_controls_keep_empty_axis_title_after_every_selection(app, rich_customer_result):
    widget = RfmSegmentsWidget(rich_customer_result)
    try:
        for window in range(3):
            for currency in range(2):
                widget.window_selector.setCurrentIndex(window)
                widget.currency_selector.setCurrentIndex(currency)
                assert widget.plot.count_axis.titleText() == ""
    finally:
        widget.deleteLater()
