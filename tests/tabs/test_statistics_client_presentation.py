"""Customer presentation uses the saved counts and a shared export model."""

from copy import deepcopy
from io import BytesIO
import os

import pytest
from openpyxl import load_workbook
from PyQt6.QtWidgets import QTableView

from Application import statistics_report_presentation as report
from Application.analysis_filter import AnalysisFilter
from Application.statistics_export import write_statistics_xlsx
from Application.statistics_html_export import build_statistics_html, _specs
from Application.statistics_unified_detail import filter_snapshot
from Application.statistics_unified_presentation import _bucket_counts, _copurchase_blocks, _purchase_categories, _share
from Application.tabs import dataset_statistics_tab as ui
from tests import test_statistics_unified_integration as integration
from tests.tabs.test_statistics_status_ui import app, tab, source, _use_unified, assert_shared_unified_content

__all__ = ["app", "tab", "source"]


@pytest.mark.parametrize(
    "limits, values, expected",
    [
        ((1, 3, 5, 10, 14), (1, 2, 3, 4, 5, 6, 10, 11, 14, 15, 99), [1, 2, 2, 2, 2, 2]),
        ((0, 7, 30, 60, 90), (0, 1, 7, 8, 30, 31, 60, 61, 90, 91, 999), [1, 2, 2, 2, 2, 2]),
        ((1, 3, 5, 10, 14), (), [0, 0, 0, 0, 0, 0]),
    ],
)
def test_buckets_include_each_boundary_once(limits, values, expected):
    assert _bucket_counts(((value, 1) for value in values), limits) == expected


def test_copurchase_percentages_keep_distinct_denominators():
    def count(known, complete=False):
        return dict(known=known, value=known if complete else None, quality="INSUFFICIENT_COVERAGE", reasons=[])

    evidence = dict(orders=count(20, complete=True), multicategory_orders=count(4),
                    multicategory_buyers=count(3), unresolved_category_orders=2)
    a = dict(categories=("A",), orders=count(10), buyers=count(7))
    b = dict(categories=("B",), orders=count(8), buyers=count(6))
    pair = dict(categories=("A", "B"), orders=count(4), buyers=count(3))
    data = dict(categories=[a, b], pairs=[pair])

    def render():
        return {block.table_id: block.rows for block in _copurchase_blocks("buyer_period", data, evidence, {})
                if block.kind == "table"}

    tables = render()
    assert tables["unified.customers.copurchase_base"][3][1] == 20
    assert [row[3] for row in tables["unified.customers.copurchase_categories"]] == [50, 40]
    assert tables["unified.customers.copurchase_pairs"][0][4:7] == (20, 40, 50)
    for total in (count(0, complete=True), count(20)):
        evidence["orders"] = total
        tables = render()
        assert tables["unified.customers.copurchase_base"][3][1] is None
        assert [row[3] for row in tables["unified.customers.copurchase_categories"]] == [None, None]
        assert tables["unified.customers.copurchase_pairs"][0][4:7] == (None, 40, 50)
    a["orders"] = count(0, complete=True)
    pair["orders"] = count(0, complete=True)
    assert render()["unified.customers.copurchase_pairs"][0][5:7] == (None, 0)


def test_category_ranking_limits_the_chart_and_keeps_full_table():
    raw = [(f"{index:03}", index + 1) for index in range(12)]
    labels = {code: (f"Категория {code}", ("Одежда", f"Категория {code}")) for code, _ in raw[:-1]}
    _, chart, table = _purchase_categories("buyer_period", "category", "Категории первой покупки", raw, labels)
    assert len(table.rows) == 12
    assert table.rows[-1] == ("011", "Категория не определена", "", 12)
    assert chart.data.labels[0] == "Категория не определена (011)"
    assert chart.data.series[0][1] == tuple(range(12, 2, -1))


def test_small_nonzero_pair_share_stays_visible_as_a_number():
    cell = report.cell_presentation(_share(1, 170355), percentage=True)
    assert cell.kind == "number" and cell.value > 0
    assert cell.text == "0,00059 %"
    assert cell.number_format == '#,##0.00000" %"'


def test_customer_pages_use_same_saved_view_in_qt_html_and_excel(tab, source, monkeypatch):
    _, widget = tab
    _use_unified(widget, monkeypatch)
    selection = AnalysisFilter("2025-10-01", "2025-11-30", ("Type A",), ("SS26",))
    widget.set_analysis_filter(selection)
    dto = integration._report(source, selected_items=["000001"], selected_start=selection.start_date,
                              selected_end=selection.end_date, order_period=selection.interval,
                              buyer_period=selection.interval, customer_period=selection.interval)
    original = deepcopy(dto.envelope())
    widget.render(dto)
    widget.set_analysis_filter(AnalysisFilter(nomenclature_types=("Новый отбор",)))
    assert widget.displayed_result is dto
    assert "Type A" in widget.report.summary and "Новый отбор" not in widget.report.summary
    calls = []
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *args: ("D:/report.zip", ""))
    monkeypatch.setattr(ui, "validate_report_destination", lambda target: target)
    monkeypatch.setattr(widget, "_start_details", lambda options, **kwargs: calls.append(kwargs))
    widget.export()
    assert len(calls) == 1 and calls[0]["snapshot"].envelope() == original
    assert calls[0]["applied_filter"] == filter_snapshot(selection, "", "")

    overview, purchases = widget.report.pages[3].groups[:2]
    cards = report.kpi_cells(overview.blocks[1])
    assert [label for label, _ in cards] == [
        "Клиенты", "Покупатели периода", "Повторные покупатели",
        "Клиенты с просмотрами", "Клиенты с избранным", "Клиенты с просмотрами и покупками",
    ]
    assert cards[-1][1].displayed_value == dto.actions["data"]["VIEW_PURCHASE"]["known"]
    assert all(block.title not in ("Покупатели периода", "Состав клиентской базы") for block in overview.blocks)
    assert all(block.table_id != "unified.customers.buyer_identity" for block in overview.blocks)
    population = overview.blocks[2]
    assert population.table_id == "unified.customers.population" and not population.options.get("details")
    activity_index = next(i for i, block in enumerate(overview.blocks) if block.title == "Взаимодействия клиентов")
    activity = overview.blocks[activity_index + 1]
    assert activity.kind == "table" and len(activity.rows) == 9
    assert report.table_cells(activity)[3][1].displayed_value == cards[-1][1].displayed_value

    chart = {b.data.title: b.data for b in purchases.blocks if b.kind == "chart"}
    tables = {b.table_id.removeprefix("unified.customers."): b for b in purchases.blocks if b.kind == "table"}
    source_data = dto.customers["data"]["purchases"]
    frequencies = tables["frequency"].rows
    assert len(frequencies) == 6
    assert sum(row[1] for row in frequencies) == sum(row[1] for row in source_data["frequency"])
    assert sum(row[2] for row in frequencies) == pytest.approx(100)
    assert chart["Распределение покупателей по количеству заказов"].series[0][1] == tuple(row[1] for row in frequencies)
    assert chart["Первые покупки по месяцам"].kind == "line"
    assert chart["Первые покупки по месяцам"].series[0][1] == tuple(row[1] for row in source_data["first_months"])
    assert sum(chart["Интервалы до второй покупки"].series[0][1]) == sum(row[1] for row in source_data["second_intervals"])
    assert tables["second_intervals"].rows == tuple(map(tuple, source_data["second_intervals"]))
    assert tables["item_order_distribution"].options["context_id"] == "item_interactions"
    assert tables["frequency"].options["context_id"] == "buyer_period"
    for key in ("first_categories", "second_categories"):
        assert [(row[0], row[-1]) for row in tables[key].rows] == list(map(tuple, source_data[key]))
    assert len(tables["copurchase_pairs"].rows) == len(source_data["pairs"])
    assert len(tables["copurchase_categories"].rows) == len(source_data["categories"])
    frequency_section = next(i for i, block in enumerate(purchases.blocks) if block.title == "Частота и интервалы покупок")
    assert len(purchases.blocks[frequency_section + 1].values) == 2
    expected_channels = dto.cohorts["channels"]["channels"]
    assert tables["purchase_channels"].rows == tuple(
        (row["name"], row["first_buyers"], row["second_buyers"]) for row in expected_channels)
    assert len(chart["Каналы первой и второй покупки — TOP-10"].series) == 2

    widget.sections.setCurrentIndex(3)
    overview_widget = widget.customer_sections.widget(0)
    population_widget = next(t for t in overview_widget.findChildren(QTableView)
                             if t.property("statisticsTableId") == population.table_id)
    assert not population_widget.isHidden()
    grid = overview_widget.findChildren(ui._StatisticsCards)[0]
    grid.resize(1050, 200)
    grid.reflow()
    assert [grid.layout().getItemPosition(i)[:2] for i in range(6)] == [(i // 3, i % 3) for i in range(6)]
    widget.customer_sections.setCurrentIndex(1)
    channels_chart = next(plot for plot in widget.customer_sections.widget(1).findChildren(ui.charts.StatisticsChart)
                          if plot.data.title == "Каналы первой и второй покупки — TOP-10")
    bars = channels_chart.chart().series()[0]
    assert bars.barWidth() == pytest.approx(.8)
    assert channels_chart.height() == 520 and channels_chart.chart().legend().isVisible()
    primary, secondary = bars.barSets()
    assert primary.color().name() == os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#a65cf2").lower()
    assert secondary.color() != primary.color()
    assert primary.pen().width() == secondary.pen().width() == 2
    assert tuple(label.toPlainText() for label in channels_chart._product_labels) == channels_chart.data.display_labels

    document = build_statistics_html(dto, presentation=widget.report)
    stream = BytesIO()
    write_statistics_xlsx(dto, stream, presentation=widget.report)
    book = load_workbook(stream)
    try:
        assert_shared_unified_content(widget, document, book)
        html_charts = {b["title"]: b for b in _specs(widget.report) if b["group"] == purchases.key}
        html_channels = html_charts[channels_chart.data.title]
        assert (html_channels["height"], html_channels["bar_width"]) == (channels_chart.height(), bars.barWidth())
        for title, series in chart.items():
            if series.empty:
                continue
            assert [trace["exact"] for trace in html_charts[title]["data"]] == [
                [None if value is None else str(value) for value in values] for _, values in series.series]
    finally:
        book.close()
    assert dto.envelope() == original
