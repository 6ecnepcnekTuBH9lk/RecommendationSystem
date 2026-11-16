"""Unified statistics presentation and subprocess integration through real Qt widgets."""

from datetime import datetime, timezone
from copy import deepcopy
import json

import pytest
from PyQt6.QtCore import QProcess, QThreadPool, Qt
from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QTableWidget, QTableView, QToolButton

from Application import statistics_unified_index as index
from Application.analysis_filter import AnalysisFilter
from Application.statistics_rfm_presentation import WINDOW_LABELS
from Application.statistics_unified_cache import LATEST_RESULT_NAME
from Application.tabs import dataset_statistics_tab as ui
from Application.statistics_unified_dto import content_digest
from tests import test_statistics_unified_integration as integration
from tests.tabs import test_dataset_statistics_tab as legacy_ui


@pytest.fixture
def app():
    yield from legacy_ui.app.__wrapped__()


@pytest.fixture
def tab(app, tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "CACHE_PATH", tmp_path / "legacy.json")
    monkeypatch.setattr(ui, "load_filter", AnalysisFilter)
    yield from legacy_ui.window.__wrapped__(app)


@pytest.fixture
def source(tmp_path, monkeypatch):
    directory = tmp_path / "source"
    directory.mkdir()
    result = integration.source.__wrapped__(directory, monkeypatch)
    integration._write(result)
    return result


def _use_unified(tab, monkeypatch, source="REFERENCE"):
    monkeypatch.setattr(tab, "_calculation_source", lambda: ("unified", source))
    tab._update_export_button()


def _table(tab, key):
    for widget in tab.findChildren(QTableWidget):
        if widget.property("statisticsTableId") == "unified." + key:
            return widget
    raise AssertionError(key)


def _row(table, title):
    return next(index for index in range(table.rowCount()) if table.item(index, 0).text() == title)


def _real_process(source, monkeypatch):
    root, catalog, _, _, target = source
    work, cache = root / "worker-scratch", root / "worker-cache"

    class Process(QProcess):
        def start(self, program, args):
            super().start(program, args + ["--raw-root", str(root), "--catalog", str(catalog),
                "--unified-workspace", str(work), "--unified-cache", str(cache),
                "--index-dir", str(target), "--expected-bytes", "1048576"])

    monkeypatch.setattr(ui, "QProcess", Process)
    return work, cache


def _wait(tab, timeout=60000):
    legacy_ui.wait_until(lambda: not tab._busy(), timeout=timeout)


def test_real_reference_qprocess_reuses_same_filter_cache(tab, source, monkeypatch):
    _, widget = tab
    work, cache = _real_process(source, monkeypatch)
    _use_unified(widget, monkeypatch)
    assert widget.displayed_result is None and widget.refresh.isEnabled()
    widget.start()
    assert widget.process is not None and not widget.refresh.isEnabled()
    _wait(widget)
    result = widget.displayed_result
    assert result is not None, widget.status.text()
    assert result.source["reader"] == "REFERENCE"
    assert result.orders["purchase_orders"]["known"] == 3
    path = cache / LATEST_RESULT_NAME
    saved = path.read_bytes()
    assert not list(work.iterdir())
    widget.refresh.click()
    _wait(widget)
    assert widget.displayed_result == result and path.read_bytes() == saved
    widget.set_analysis_filter(AnalysisFilter(nomenclature_types=("Type B",)))
    assert "Показан отчет с прежним отбором" in widget.description.text()
    widget.refresh.click()
    _wait(widget)
    assert widget.displayed_result.contexts["order_product_slice"]["product_filter_key"]
    assert path.read_bytes() != saved


def test_qt_widgets_cover_dto_status_money_rfm_cohorts_and_navigation(tab, source, monkeypatch):
    main_window, widget = tab
    _use_unified(widget, monkeypatch)
    dto = integration._report(source, selected_end=None, calculated_at=datetime(2026, 11, 12, tzinfo=timezone.utc))
    widget.render(dto)
    catalog = _table(widget, "orders.catalog")
    assert catalog.rowCount() == 43
    codes = {catalog.item(i, 1).text() for i in range(43)}
    assert {"F", "sold", "CP", "delivering", "Return", "Mindbox-Tech-Cancellation"} <= codes
    tech = next(i for i in range(43) if catalog.item(i, 1).text() == "Mindbox-Tech-Cancellation")
    assert catalog.item(tech, 2).text() == "—"
    assert any(r["bucket"] == "UNKNOWN_CODE" for r in dto.status_catalog["rows"])
    orders = widget.report.pages[1].groups[0].blocks
    cards = next(block for block in orders if block.kind == "cards")
    assert dict(ui.report.kpi_cells(cards))["Заказы с покупкой"].value == 3
    order_page = widget.sections.widget(1)
    label = next(label for label in order_page.findChildren(QLabel) if label.text() == "Заказы с покупкой")
    assert any(child.text() == "3" for child in label.parentWidget().findChildren(QLabel))
    assert orders.index(cards) < next(i for i, block in enumerate(orders) if block.kind == "table")
    assert dto.orders["indicators"]["partial_orders"]["known"] == 1
    assert _table(widget, "orders.cancellations").item(0, 1).data(Qt.ItemDataRole.UserRole).known == 1
    assert _table(widget, "orders.return_states").item(0, 1).data(Qt.ItemDataRole.UserRole).known == 1
    rub, kzt = (_table(widget, "orders.money." + code) for code in ("RUB", "KZT"))
    rub_recognized = rub.item(_row(rub, "Сумма покупок"), 1)
    assert rub_recognized.text() == "1 500,00"
    assert rub_recognized.data(Qt.ItemDataRole.UserRole).value is not None
    assert not rub_recognized.toolTip()
    assert kzt.item(_row(kzt, "Сумма покупок"), 1).text() == "0,00"
    assert _table(widget, "actions.intersections").rowCount() == 4
    assert _table(widget, "products.top.purchased").rowCount() == 1
    overview, purchases = widget.report.pages[3].groups[:2]
    overview_cards = next(block for block in overview.blocks if block.kind == "cards")
    purchase_cards = next(block for block in purchases.blocks if block.kind == "cards")
    assert dict(ui.report.kpi_cells(overview_cards))["Клиенты"].value == 3
    assert dict(ui.report.kpi_cells(purchase_cards))["Повторные покупатели"].value == 1
    segments = _table(widget, "rfm.LAST_12M.RUB.segments")
    assert segments.item(_row(segments, "Оценка недоступна"), 1).text() == "1"
    rfm_blocks = widget.report.pages[3].groups[3].blocks
    tables = {block.table_id: (i, block) for i, block in enumerate(rfm_blocks) if block.kind == "table"}
    assert not any(key.endswith(".thresholds") for key in tables)
    for window, payload in dto.rfm["windows"].items():
        for currency, row in payload["currencies"].items():
            prefix = f"unified.rfm.{window}.{currency}"
            heading = rfm_blocks[tables[prefix + ".metrics"][0] - 1]
            assert heading.kind == "heading" and heading.title == f"{WINDOW_LABELS[window]} — {currency}"
            for suffix in ("metrics", "segments", "segment_money"):
                assert prefix + "." + suffix in tables
            i, scores = tables[prefix + ".scores"]
            assert rfm_blocks[i-2].title == f"Распределение баллов R/F/M — {currency}"
            assert rfm_blocks[i-2].kind == "heading" and not scores.options.get("details")
            assert rfm_blocks[i-1].kind == "chart_row"
            assert scores.rows == tuple(map(tuple, row["score_counts"]))
            assert len(row["thresholds"]) == 3
        i, distributions = tables[f"unified.rfm.{window}.distributions"]
        assert rfm_blocks[i-2].title == "Распределения давности и частоты"
        assert rfm_blocks[i-2].kind == "heading" and not distributions.options.get("details")
        assert rfm_blocks[i-1].kind == "chart_row"
        assert distributions.rows == tuple(
            (currency, label, bucket, count)
            for currency, row in payload["currencies"].items()
            for field, label in (("recency", "Давность, дни"), ("frequency", "Частота, заказы"))
            for bucket, count in row[field]["distribution"])
    assert _table(widget, "cohorts.repeat").rowCount() == 15
    assert _table(widget, "cohorts.retention").rowCount() == 36
    assert _table(widget, "cohorts.channels").rowCount() >= 2
    assert _table(widget, "cohorts.transitions").rowCount() == len(dto.cohorts["categories"]["transitions"])
    assert [widget.sections.tabText(i) for i in range(widget.sections.count())] == [
        "Действия", "Заказы", "Товары", "Клиенты"]
    assert dto.policy["time_basis_id"] == "ORDER_CREATED_PURCHASE_PROXY_V1"
    assert dto.envelope()["schema_version"] == 13
    assert "unassigned_currency_buyers" in dto.rfm["quality"]
    rfm_cards = next(block for block in rfm_blocks if block.kind == "cards")
    assert rfm_cards.values == (
        ("Идентифицированные клиенты", dto.rfm["quality"]["population"]),
        ("Покупатели базы 48 месяцев", dto.rfm["base_buyers"]),
        ("Покупатели обеих валют", dto.rfm["quality"]["dual_currency_buyers"]),
    )
    for page in range(widget.sections.count()):
        widget.sections.setCurrentIndex(page)
        assert widget.sections.widget(page).isVisible()
    for group in range(widget.customer_sections.count()):
        widget.customer_sections.setCurrentIndex(group)
    widget.customer_sections.setCurrentIndex(3)
    rfm_widget = widget.customer_sections.widget(3)
    card_grid, = rfm_widget.findChildren(ui._StatisticsCards)
    assert len(card_grid.cards) == 3
    for width in (1200, 800):
        main_window.resize(width, 700)
        QApplication.processEvents()
        geometry = [card.geometry() for card in card_grid.cards]
        assert len({rect.top() for rect in geometry}) == 1
        assert max(rect.width() for rect in geometry) - min(rect.width() for rect in geometry) <= 1
        assert geometry[0].left() == 0 and geometry[-1].right() + 1 == card_grid.width()
    assert widget.displayed_result is dto
    assert widget.export_button.isEnabled()


def test_missing_source_does_not_start_index_build(tab, source, monkeypatch):
    _, widget = tab
    _real_process(source, monkeypatch)
    sentinel = source[-1] / "current.json"
    sentinel.write_text("legacy", encoding="utf-8")
    catalog = source[0] / "canonical" / "catalog.json"
    catalog.rename(catalog.with_name("catalog-unavailable.json"))
    _use_unified(widget, monkeypatch, "INDEXED")
    widget.start()
    _wait(widget)
    assert ui.UNIFIED_ERRORS["SOURCE_UNAVAILABLE"] in widget.status.text()
    assert widget.displayed_result is None and sentinel.read_text(encoding="utf-8") == "legacy"
    assert not (source[-1] / index.POINTER_NAME).exists()


def test_actual_indexed_qprocess_filter_and_stale_snapshot(tab, source, monkeypatch):
    from Application.statistics_unified_analytics import POINTER
    from scripts.dataset_statistics import _private_directory
    _, widget = tab
    target = source[-1]
    target.rmdir()
    _private_directory(target, create=True)
    _real_process(source, monkeypatch)
    _use_unified(widget, monkeypatch, "INDEXED")
    widget.set_analysis_filter(AnalysisFilter(nomenclature_types=("Type B",)))
    widget.start()
    _wait(widget)
    dto = widget.displayed_result
    assert dto is not None, widget.status.text()
    assert dto.orders["purchase_orders"]["known"] == 0
    assert dto.contexts["order_product_slice"]["product_filter_key"]
    assert dto.contexts["rfm_48m"]["product_filter_key"] is None
    assert dto.rfm["base_buyers"] == dto.cohorts["quality"]["base_buyers"] == 2
    assert "Отбор по виду номенклатуры: Type B" in widget.description.text()
    pointer = (target / index.POINTER_NAME).read_bytes()
    analytics = (target / POINTER).read_bytes()
    widget.start()
    _wait(widget)
    assert widget.displayed_result == dto
    assert (target / index.POINTER_NAME).read_bytes() == pointer
    assert (target / POINTER).read_bytes() == analytics
    rows = integration._write(source)
    integration.rfm_tests._write_orders(source, rows[:-1])
    widget.start()
    _wait(widget)
    assert widget.displayed_result is not None, widget.status.text()
    assert widget.displayed_result.metadata["source_binding"] != dto.metadata["source_binding"]
    assert (target / index.POINTER_NAME).read_bytes() != pointer


def _report_for_ui(source):
    return integration._report(source, selected_end=None, calculated_at=datetime(2026, 11, 12, tzinfo=timezone.utc))


def _scripted_process(monkeypatch, tmp_path, envelope=None, *, error=None, wait_cancel=False, category=""):
    if envelope is not None:
        path = tmp_path / "payload.json"
        payload = dict(report=envelope, analysis_filter=AnalysisFilter().to_dict(), product_code="", category_code=category)
        path.write_text(json.dumps({"event": "result", "value": payload}, ensure_ascii=True), encoding="ascii")
    commands = ["import os,sys,time", 'print(\'{"event":"progress","value":"worker ready"}\',flush=True)']
    if wait_cancel:
        commands.extend(['sys.exit(0 if os.read(sys.stdin.fileno(),32).strip()==b"cancel" else 1)'])
    elif error is not None:
        commands.extend([f'print(\'{json.dumps({"event": "unified_error", "value": error})}\',flush=True)', "sys.exit(1)"])
    else:
        commands.extend(["time.sleep(.3)", f"print(open({str(path)!r},encoding='ascii').read(),flush=True)"])

    class Process(QProcess):
        def start(self, program, args):
            super().start(program, ["-u", "-c", "\n".join(commands)])

    monkeypatch.setattr(ui, "QProcess", Process)


def test_no_model_or_index_picker_and_current_report_remains_usable(tab):
    _, widget = tab
    previous = legacy_ui.sample_result.__wrapped__()
    widget.render(previous)
    assert widget.findChild(QComboBox, "statisticsMode") is None
    assert widget.findChild(QComboBox, "statisticsSource") is None
    assert widget.displayed_result is previous
    assert widget.export_button.isEnabled()


def test_repeated_start_keeps_only_one_worker(tab, source, monkeypatch, tmp_path):
    _, widget = tab
    dto = _report_for_ui(source)
    _scripted_process(monkeypatch, tmp_path, dto.envelope())
    _use_unified(widget, monkeypatch)
    widget.start()
    process = widget.process
    widget.start()
    assert widget.process is process
    _wait(widget)
    assert widget.displayed_result == dto
    assert widget.export_button.isEnabled()


@pytest.mark.parametrize("error", ["CACHE_WRITE_FAILED", "INVALID_UNIFIED_DTO", "INSUFFICIENT_DISK", "UNSAFE_STORAGE_ACL"])
def test_failed_worker_preserves_last_valid_result(tab, source, monkeypatch, tmp_path, error):
    _, widget = tab
    dto = _report_for_ui(source)
    _use_unified(widget, monkeypatch)
    widget.render(dto)
    _scripted_process(monkeypatch, tmp_path, error=error)
    widget.start()
    _wait(widget)
    assert ui.UNIFIED_ERRORS[error] in widget.status.text()
    assert widget.displayed_result is dto


def test_wrong_filter_result_is_discarded_by_background_preparation(tab, source, monkeypatch, tmp_path):
    _, widget = tab
    dto = _report_for_ui(source)
    _use_unified(widget, monkeypatch)
    widget.render(dto)
    malformed = deepcopy(dto.envelope())
    malformed["result"]["contexts"]["rfm_48m"]["product_filter_key"] = "a"*64
    malformed["result"]["metadata"]["content_digest"] = content_digest(malformed["result"])
    _scripted_process(monkeypatch, tmp_path, malformed)
    widget.start()
    _wait(widget)
    assert ui.UNIFIED_ERRORS["INVALID_UNIFIED_DTO"] in widget.status.text()
    assert widget.displayed_result is dto


@pytest.mark.parametrize("close_tab", [False, True])
def test_cooperative_cancel_and_close_of_unified_process(tab, source, monkeypatch, tmp_path, close_tab):
    _, widget = tab
    dto = _report_for_ui(source)
    _use_unified(widget, monkeypatch)
    widget.render(dto)
    _scripted_process(monkeypatch, tmp_path, wait_cancel=True)
    widget.start()
    legacy_ui.wait_until(lambda: "worker ready" in widget.status.text())
    if close_tab:
        widget.close()
    else:
        widget.cancel_button.click()
    _wait(widget)
    assert "отменен" in widget.status.text()
    assert widget.displayed_result is dto
    if close_tab:
        legacy_ui.wait_until(lambda: not widget.isVisible())
    assert QThreadPool.globalInstance().waitForDone(1000)


def test_large_unified_table_keeps_all_rows_without_allocating_items(tab):
    _, widget = tab
    layouts = widget._pages()
    path = "Одежда / Рубашки и сорочки / Рубашки с длинным рукавом"
    block = ui.report.ReportBlock("table", headers=("Значение", "Путь"), rows=tuple((i, path) for i in range(3000)),
                                  table_id="unified.large", options={"context_id": "buyer_period"})
    ui._render_report_blocks(layouts[0], [block])
    table = next(t for t in widget.findChildren(QTableView) if t.property("statisticsTableId") == "unified.large")
    assert not isinstance(table, QTableWidget)
    assert table.model().rowCount() == 3000
    assert table.model().index(2999, 0).data() == "2 999"
    assert table.model().index(2999, 1).data(Qt.ItemDataRole.ToolTipRole) == path


def test_unified_charts_keep_sections_and_hide_duplicate_actions(tab, source, monkeypatch):
    _, widget = tab
    _use_unified(widget, monkeypatch)
    dto = _report_for_ui(source)
    widget.render(dto)
    assert not any(label.property("class") == "statisticsContext" for label in widget.findChildren(QLabel))
    actions = widget.report.pages[0].groups[0].blocks
    action = next(block.data for block in actions if block.kind == "chart")
    assert action.title == "Пересечения действий"
    table = _table(widget, "actions.intersections")
    assert tuple(table.item(row, 1).data(Qt.ItemDataRole.UserRole).value for row in range(table.rowCount())) == action.series[0][1]
    rfm_page = widget.customer_sections.widget(3)
    rfm_charts = set(rfm_page.findChildren(ui.charts.StatisticsChart))
    for chart in widget.findChildren(ui.charts.StatisticsChart):
        title = chart.chart().title()
        if chart in rfm_charts and chart.data.inner_title:
            assert title == chart.data.title.upper()
        else:
            assert not title
    assert all(table.property("statisticsTableId") not in ("unified.actions.totals", "unified.actions.event_names")
               for table in widget.sections.widget(0).findChildren(QTableView))
    assert widget.findChild(ui.RfmSegmentsWidget).choices
    assert widget.findChild(ui.CohortRetentionHeatmap).model().rowCount() == len(dto.cohorts["months"])


def test_order_and_customer_money_in_qt_html_and_xlsx(tab, source, monkeypatch):
    from decimal import Decimal
    import os
    from io import BytesIO
    from openpyxl import load_workbook
    from Application.statistics_export import write_statistics_xlsx
    from Application.statistics_html_export import build_statistics_html, _specs
    from PyQt6.QtCharts import QLineSeries

    _, widget = tab
    _use_unified(widget, monkeypatch)
    dto = _report_for_ui(source)
    widget.render(dto)
    charts = widget.sections.widget(1).findChildren(ui.charts.StatisticsChart)
    specs = {spec["title"]: spec for spec in _specs(widget.report) if spec["section"] == 1}
    count = next(chart for chart in charts if chart.data.title == "Заказы с покупкой — RUB")
    primary = os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#a65cf2").lower()
    assert not specs[count.data.title]["secondary"]
    assert all(series.color().name() == primary for series in count.chart().series())
    money_page = widget.customer_sections.widget(2)
    customer_charts = {chart.data.title: chart for chart in money_page.findChildren(ui.charts.StatisticsChart)}
    for currency in ("RUB", "KZT"):
        title = "Сумма покупок — " + currency
        monthly = next(chart for chart in charts if chart.data.title == title)
        assert monthly.data.kind == "line" and not monthly.chart().legend().isVisible()
        assert not monthly.tooltip.wordWrap()
        assert [label for label, _ in monthly.data.series] == ["Сумма"]
        assert all(isinstance(series, QLineSeries) for series in monthly.chart().series())
        spec = specs[title]
        assert spec["secondary"] and monthly.data.secondary
        assert {trace.get("line", {}).get("color") for trace in spec["data"]} == {
            ui.charts.SECONDARY[False] if currency == "RUB" else None}
        client_chart = customer_charts[title]
        assert not client_chart.data.secondary
        assert all(series.color().name() == primary for series in client_chart.chart().series())
        assert not spec["show_legend"] and all(trace["name"] == "Сумма" for trace in spec["data"])
        assert any(f" {currency}" in tip and "Сумма: " in tip for trace in spec["data"] for tip in trace["hovertext"])
        for i, row in enumerate(r for r in dto.orders["breakdowns"]["months"] if r["currency"] == currency):
            full = row["recognized"]["value"]
            assert monthly.data.series[0][1][i] == (None if full is None else Decimal(full))
        client_months = dto.customers["data"]["buyer_breakdowns"]["monthly_money"][currency]
        client_table = _table(widget, "customers.money.monthly." + currency)
        assert [client_table.item(i, 0).text() for i in range(client_table.rowCount())] == [r[0] for r in client_months]
        assert _table(widget, "customers.money.quintiles." + currency).rowCount() == len(
            dto.customers["data"]["buyer_breakdowns"]["money"][currency]["quintiles"])
    stores = next(chart for chart in charts if chart.data.title == "Магазины и каналы — KZT")
    assert not specs[stores.data.title]["secondary"]
    assert stores.chart().series()[0].barSets()[0].color().name() == os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#a65cf2").lower()
    recognized = _table(widget, "orders.money.RUB")
    qt_cell = recognized.item(_row(recognized, "Сумма покупок"), 1)
    assert qt_cell.text() == "1 500,00"
    assert not qt_cell.toolTip()
    assert qt_cell.data(Qt.ItemDataRole.UserRole).value is not None
    html = build_statistics_html(dto, presentation=widget.report)
    for removed in ('Техническая информация', 'id="section-4"', 'id="tab-4"',
                    'Покупатели без определенной валютной базы'):
        assert removed not in html
    rfm_html = html.split('id="customer-section-3"', 1)[1].split('</section>', 1)[0]
    assert rfm_html.count('class="kpi"') == 3
    assert 'class="kpis" data-count="3"' in rfm_html
    assert qt_cell.text() + "</td>" in html
    assert 'data-note=' not in html and "Известная часть показателя" not in html
    assert '"name": "Подтвержденный подытог"' not in html and '"name": "Полный итог"' not in html
    header = html.split("<header>", 1)[1].split("</header>", 1)[0]
    assert "Дата расчета:" in header and "Mindbox" not in header and "UNIFIED_PURCHASE_V2" not in header
    assert "<h1>" not in header
    outside_rfm = html.split('id="customer-section-3"', 1)
    assert '<article class="chart-card"><h3>' not in outside_rfm[0] + outside_rfm[1].split("</section>", 1)[1]
    assert "Системные имена наблюдаемых действий" not in html
    assert 'class="report-details"' in html
    assert "Пороги R/F/M" not in html
    assert "<summary>Распределение баллов R/F/M" not in html
    assert "<summary>Распределения давности и частоты" not in html
    assert '"type": "scatter"' in html
    stream = BytesIO()
    write_statistics_xlsx(dto, stream, presentation=widget.report)
    stream.seek(0)
    with_book = load_workbook(stream)
    try:
        assert_shared_unified_content(widget, html, with_book)
        assert with_book.sheetnames == ["Действия", "Заказы", "Товары", "Клиенты"]
        labels = [cell for row in with_book["Клиенты"].values for cell in row if isinstance(cell, str)]
        assert "Покупатели без определенной валютной базы" not in labels
        assert not any(label.startswith("Пороги R/F/M") for label in labels)
        for label in ("Распределение баллов R/F/M — RUB", "Распределение баллов R/F/M — KZT",
                      "Распределения давности и частоты"):
            assert labels.count(label) == len(dto.rfm["windows"])
        assert "Тип действия" not in {cell.value for row in with_book["Действия"] for cell in row}
        sheet = with_book["Заказы"]
        excel_cell = next(values[1] for values in sheet.iter_rows()
                          if values[0].value == "Сумма покупок")
        assert excel_cell.data_type == "n"
        assert "*" not in excel_cell.number_format
        assert excel_cell.comment is None
    finally:
        with_book.close()


def test_rfm_chart_projections_preserve_saved_values_and_missing_scores(app):
    from decimal import Decimal

    from Application.statistics_unified_presentation import (
        _rfm_distributions_chart_row, _rfm_scores_chart_row, _rfm_segment_money_chart,
    )

    scores = _rfm_scores_chart_row(((1, 2, None, 8), (1, 5, None, 3), (None, 2, None, 2)))
    assert [plot.labels for plot in scores.data] == [("1", "2", "3", "4", "5")] * 3
    assert [plot.series[0][1] for plot in scores.data] == [
        (11, 0, 0, 0, 0), (0, 10, 0, 0, 3), (None,) * 5,
    ]
    widget = ui.StatisticsChartRow(scores.data)
    widget.show()
    for width, columns in ((1200, 3), (800, 2), (560, 1), (1200, 3)):
        widget.resize(width, 900)
        app.processEvents()
        assert [widget.grid.getItemPosition(widget.grid.indexOf(plot))[:2] for plot in widget.plots] == [
            divmod(i, columns) for i in range(3)]
    widget.close()

    currencies = {
        "RUB": {"recency": {"distribution": ((0, 3), (30, 4), (31, 5), (730, 2), (731, 1))},
                "frequency": {"distribution": ((0, 9), (1, 2), (3, 5), (4, 6), (11, 3))}},
        "KZT": {"recency": {"distribution": ()}, "frequency": {"distribution": ((1, 1), (5, 3), (10, 2))}},
    }
    recency, frequency = _rfm_distributions_chart_row(currencies).data
    assert recency.series == (("RUB", (7, 5, None, None, 2, 1)), ("KZT", (None,) * 6))
    assert frequency.series == (("RUB", (9, 2, 5, 6, None, 3)), ("KZT", (None, 1, None, 3, 2, None)))
    assert _rfm_distributions_chart_row({"KZT": currencies["KZT"]}).data[1].labels[0] == "1"

    business = {code: {"known_subtotal": amount, "share_of_known_subtotal": share}
                for code, amount, share in (("OTHER", "1.25", "50"), ("AT_RISK", "1.25", "50"),
                                            ("LOYAL", "0", "0"), ("UNSCORABLE", None, None))}
    money = _rfm_segment_money_chart("RUB", tuple((code, 0, None) for code in reversed(business)), business).data
    assert money.labels == ("Риск снижения активности", "Прочие", "Лояльные", "Оценка недоступна")
    assert money.series[0][1] == (Decimal("1.25"), Decimal("1.25"), Decimal(0), None)
    assert "50,00 %" in money.tooltips[0][0] and "Доля суммы: —" in money.tooltips[0][-1]


def test_cohort_charts_follow_saved_tables(tab, source, monkeypatch):
    from bisect import bisect_left
    from decimal import Decimal
    from Application.statistics_cohorts_core import HORIZONS
    from Application.statistics_html_export import _specs

    _, widget = tab
    _use_unified(widget, monkeypatch)
    dto = _report_for_ui(source)
    widget.render(dto)
    data = dto.cohorts
    group = widget.report.pages[3].groups[4]
    charts = {block.data.title: block.data for block in group.blocks if block.kind == "chart"}
    assert len(charts) == 10
    months = sorted(data["months"])
    retention = charts["Динамика удержания когорт"]
    assert retention.labels == tuple(f"M{i}" for i in range(1, 13))
    assert retention.series == tuple((m, tuple(cell["retention_rate"] for cell in data["months"][m]["retention"][1:]))
                                     for m in months)
    sizes = charts["Первые и вторые покупки по когортам"]
    assert sizes.labels == tuple(months)
    assert tuple(values for _, values in sizes.series) == tuple(tuple(data["months"][m][key] for m in months)
                                                               for key in ("cohort_size", "observed_repeat_buyers"))
    days = data["summary"]["second_purchase_days"]["distribution"]
    if days:
        groups = [0] * 7
        for day, buyers in days:
            groups[bisect_left((0, 7, 30, 60, 90, 180), day)] += buyers
        assert charts["Интервалы между покупками"].series[0][1] == tuple(groups)
        assert sum(groups) == sum(n for _, n in days)
    rates = [data["summary"]["horizons"][str(h)]["rate_quality"]["value"] for h in HORIZONS]
    assert charts["Повторные покупки по периодам"].series[0][1] == tuple(
        None if rate is None else Decimal(str(rate)) for rate in rates)
    for currency in ("RUB", "KZT"):
        finance = data["money"]["cohorts"]
        values = charts[f"Суммы первых и повторных покупок — {currency}"].series
        for kind, (_, actual) in zip(("first", "repeat"), values, strict=True):
            wire = [finance[m][kind]["currencies"][currency]["recognized"]["value"] for m in months]
            assert actual == tuple(None if v is None else Decimal(v) for v in wire)
    for source_key, title in (("channels", "каналами"), ("categories", "категориями")):
        transitions = data[source_key]["transitions"]
        chart = charts[f"Популярные переходы между {title} — TOP-10"]
        assert list(chart.series[0][1]) == sorted((row["buyers"] for row in transitions), reverse=True)[:10]
    specs = {spec["title"]: spec for spec in _specs(widget.report) if spec["group"] == "cohorts"}
    assert set(specs) == set(charts)
    assert all(trace["connectgaps"] is False for trace in specs[retention.title]["data"])


def test_compact_retention_width_and_legend_preserve_gaps(app):
    from PyQt6.QtCore import QCoreApplication, QPoint
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QHeaderView
    from Application.statistics_chart_data import RetentionCell, RetentionHeatmapData, RetentionRow
    from Application.statistics_unified_presentation import _cohort_retention_chart

    def row(month, values):
        return RetentionRow(month, 17, tuple(RetentionCell(v, "FULL" if v is not None else "UNAVAILABLE",
                                                           None, None, None, None) for v in values))

    data = RetentionHeatmapData((row("2026-01", (0, None, 10, *([None] * 9))),
                                 row("2026-02", (8, 12, *([None] * 10)))))
    heatmap = ui.CohortRetentionHeatmap(data=data, compact=True)
    heatmap.resize(1440, 220)
    heatmap.show()
    for _ in range(4):
        QCoreApplication.processEvents()
    assert sum(heatmap.columnWidth(i) for i in range(14)) == heatmap.viewport().width()
    assert max(heatmap.columnWidth(i) for i in range(2, 14)) - min(heatmap.columnWidth(i) for i in range(2, 14)) <= 1
    header = heatmap.horizontalHeader()
    assert header.sectionResizeMode(13) == QHeaderView.ResizeMode.Stretch
    assert header.sectionResizeMode(12) == QHeaderView.ResizeMode.Interactive
    for width in (1280, 1600):
        heatmap.resize(width, 220)
        QCoreApplication.processEvents()
        edge = header.sectionViewportPosition(13) + header.sectionSize(13)
        assert edge == heatmap.viewport().width() and heatmap.horizontalScrollBar().maximum() == 0
        start = QPoint(edge - 1, header.height() // 2)
        QTest.mousePress(header.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(header.viewport(), start - QPoint(65, 0))
        QTest.mouseRelease(header.viewport(), Qt.MouseButton.LeftButton, pos=start - QPoint(65, 0))
        assert header.sectionViewportPosition(13) + header.sectionSize(13) == edge
    heatmap.resize(820, 220)
    for _ in range(4):
        QCoreApplication.processEvents()
    assert heatmap.horizontalScrollBar().maximum() > 0
    heatmap.horizontalScrollBar().setValue(heatmap.horizontalScrollBar().maximum())
    assert header.sectionViewportPosition(13) + header.sectionSize(13) == heatmap.viewport().width()
    assert min(heatmap.columnWidth(i) for i in range(2, 14)) >= 78
    assert heatmap.model().index(0, 2).data(Qt.ItemDataRole.UserRole).value == 0
    assert heatmap.model().index(0, 3).data(Qt.ItemDataRole.UserRole).value is None
    chart = ui.charts.StatisticsChart(_cohort_retention_chart(data)[1].data)
    first, last, other = chart.chart().series()
    assert list((p.x(), p.y()) for p in first.points()) == [(0, 0)]
    assert list((p.x(), p.y()) for p in last.points()) == [(2, 10)]
    assert first.pen().color() == last.pen().color() != other.pen().color()
    assert 12 <= chart.count_axis.max() < 100
    marker, = chart.chart().legend().markers(first)
    marker.clicked.emit()
    assert not first.isVisible() and not last.isVisible() and other.isVisible()
    marker.clicked.emit()
    assert first.isVisible() and last.isVisible()
    heatmap.close()


def test_cohort_transitions_rank_ties_with_unknown_source_code():
    from Application.statistics_unified_presentation import _cohort_transitions_chart

    transitions = [{"first": "A", "second": "A", "buyers": 3}, {"first": None, "second": None, "buyers": 3}]
    _, chart = _cohort_transitions_chart("Переходы", transitions, {"A": "Магазин", None: "Не указан"},
                                         "first", "second", "тот же канал")
    assert chart.data.labels == ("Не указан → Не указан", "Магазин → Магазин (тот же канал)")
    assert chart.data.series == (("Покупатели", (3, 3)),)


def assert_shared_unified_content(widget, document, book):
    from decimal import Decimal
    from html import escape, unescape
    import re
    from openpyxl.utils.cell import range_boundaries
    from Application.statistics_export import SECTION_FILL

    model = widget.report
    assert widget.description.text().splitlines()[0] == model.summary
    assert '<p class="report-summary">' + escape(model.summary) + '</p>' in document
    checked_tables = checked_numbers = 0
    for page_index, page in enumerate(model.pages):
        sheet = book[page.title]
        assert sheet["A4"].value == model.summary
        excel_tables = iter(sheet.tables.values())
        last_excel_row = 4

        def check_excel(headers, rows):
            nonlocal checked_numbers, last_excel_row
            if not rows:
                last_excel_row = next(row + 1 for row in range(last_excel_row + 1, sheet.max_row)
                                      if sheet.cell(row + 1, 1).value == "Нет данных для таблицы."
                                      and tuple(sheet.cell(row, c).value for c in range(1, len(headers) + 1)) == headers)
                return
            left, top, right, bottom = range_boundaries(next(excel_tables).ref)
            assert top > last_excel_row
            last_excel_row = bottom
            assert tuple(sheet.cell(top, c).value for c in range(left, right + 1)) == headers
            assert bottom - top == len(rows)
            for row_index, row in enumerate(rows, top + 1):
                for column, expected in enumerate(row, left):
                    cell = sheet.cell(row_index, column)
                    if expected.kind == "number":
                        checked_numbers += 1
                        assert cell.data_type == "n"
                        assert cell.value == pytest.approx(float(expected.displayed_value), rel=1e-14)
                        assert cell.number_format == expected.number_format
                    else:
                        assert ("" if cell.value is None else cell.value) == expected.text

        globals_block = ui.report.ReportBlock("cards", values=model.global_cards)
        check_excel(("Показатель", "Значение"), [(ui.report.cell_presentation(label), cell)
                                                for label, cell in ui.report.kpi_cells(globals_block)])
        page_html = document.split(f'id="section-{page_index}"', 1)[1]
        if page_index + 1 < len(model.pages):
            page_html = page_html.split(f'id="section-{page_index + 1}"', 1)[0]
        excel_headings = iter(cell.value for cell in sheet["A"] if cell.fill.fgColor.rgb == "00" + SECTION_FILL)
        for group_index, group in enumerate(page.groups):
            target = (widget.customer_sections.widget(group_index) if page.key == "customers"
                      else widget.sections.widget(page_index))
            group_html = page_html
            if page.key == "customers":
                group_html = page_html.split(f'id="customer-section-{group_index}"', 1)[1].split("</section>", 1)[0]
            tables = {t.property("statisticsTableId"): t.model() for t in target.findChildren(QTableView)
                      if t.property("statisticsTableId")}
            assert set(tables) == {block.table_id for block in group.blocks if block.kind == "table"}
            headings = iter(label.text() for label in target.findChildren(QLabel)
                            if label.property("class") == "statisticsSection")
            html_headings = iter(map(unescape, re.findall(r'<h3 class="section-title">([^<]*)</h3>', group_html)))
            cards = iter(target.findChildren(ui._StatisticsCards))
            for block_index, block in enumerate(group.blocks):
                if block.kind == "heading":
                    assert next(headings) == block.title.upper()
                    assert next(html_headings) == block.title
                    assert block.title in excel_headings
                elif block.kind == "cards":
                    expected = ui.report.kpi_cells(block)
                    rendered = next(cards)
                    assert [tuple(label.text() for label in card.findChildren(QLabel)) for card in rendered.cards] == [
                        (label, cell.text) for label, cell in expected]
                    for label, cell in expected:
                        assert re.search(r'<span>' + re.escape(escape(label)) + '</span><strong[^>]*>'
                                         + re.escape(escape(cell.text)) + '</strong>', group_html)
                    check_excel(("Показатель", "Значение"),
                                [(ui.report.cell_presentation(label), cell) for label, cell in expected])
                elif block.kind == "heatmap" and block.options.get("compact"):
                    qt = target.findChild(ui.CohortRetentionHeatmap).model()
                    headers = ("Когорта", "Покупатели (M0)", *block.data.columns)
                    assert tuple(qt.headerData(i, Qt.Orientation.Horizontal) for i in range(qt.columnCount())) == headers
                    assert qt.rowCount() == len(block.data.rows)
                    fragment = group_html.split('table-scroll heatmap heatmap-compact"><table>', 1)[1].split("</table>", 1)[0]
                    assert tuple(map(unescape, re.findall(r"<th>([^<]*)</th>", fragment))) == headers
                    html_rows = re.findall(r"<tr>(.*?)</tr>", fragment.split("<tbody>", 1)[1], re.S)
                    rows = []
                    for i, row in enumerate(block.data.rows):
                        cells = (ui.report.cell_presentation(row.month), ui.report.cell_presentation(row.buyers),
                                 *(ui.report.cell_presentation(c.value, percentage=True) for c in row.cells))
                        rows.append(cells)
                        texts = list(map(unescape, re.findall(r"<td[^>]*>([^<]*)</td>", html_rows[i])))
                        assert [texts[0], texts[1].replace(" ", ""), *texts[2:]] == [
                            row.month, str(row.buyers), *(c.text for c in cells[2:])]
                        assert [qt.index(i, j).data(Qt.ItemDataRole.UserRole).value for j in range(2, 14)] == [
                            c.value for c in row.cells]
                    assert len(html_rows) == max(1, len(rows))
                    check_excel(headers, rows)
                    checked_tables += 1
                elif block.kind == "table":
                    qt = tables[block.table_id]
                    identifier = f"table-{page_index}-{group.key}-{block_index}"
                    html_table = page_html.split(f'<table id="{identifier}">', 1)[1].split("</table>", 1)[0]
                    assert tuple(unescape(s) for s in re.findall(r'data-column="\d+">([^<]*)<span', html_table)) == block.headers
                    assert tuple(qt.headerData(i, Qt.Orientation.Horizontal) for i in range(qt.columnCount())) == block.headers
                    rows = ui.report.table_cells(block)
                    assert qt.rowCount() == len(rows)
                    html_rows = re.findall(r'<tr data-order="\d+">(.*?)</tr>', html_table, re.S)
                    assert len(html_rows) == len(rows)
                    for row_index, (html_row, row) in enumerate(zip(html_rows, rows)):
                        html_cells = re.findall(r'<td data-sort="([^"]*)" data-kind="([^"]*)"[^>]*>(.*?)</td>', html_row, re.S)
                        assert len(html_cells) == len(row)
                        for column, ((sort, kind, text), expected) in enumerate(zip(html_cells, row)):
                            assert unescape(text) == qt.index(row_index, column).data() == expected.text
                            assert kind == expected.kind
                            if kind == "number":
                                assert Decimal(sort) == Decimal(str(expected.displayed_value))
                    check_excel(block.headers, rows)
                    checked_tables += 1
            assert next(headings, None) is None
            assert next(html_headings, None) is None
            assert next(cards, None) is None
        assert next(excel_tables, None) is None
    return checked_tables, checked_numbers


def test_cleaned_unified_report_agrees_in_qt_html_and_excel(tab, source, monkeypatch):
    from html import escape
    from io import BytesIO
    from openpyxl import load_workbook
    from PyQt6.QtCharts import QHorizontalBarSeries
    from Application.statistics_export import write_statistics_xlsx
    from Application.statistics_html_export import build_statistics_html, _specs

    _, widget = tab
    _use_unified(widget, monkeypatch)
    filters = AnalysisFilter("2025-10-01", "2025-11-30", ("Пиджаки Ёлочка",), ("SS26",))
    widget.set_analysis_filter(filters)
    dto = integration._report(source, selected_items=["000001", "000002"], selected_start=filters.start_date,
                              selected_end=filters.end_date, order_period=filters.interval,
                              buyer_period=filters.interval, customer_period=filters.interval)
    original = deepcopy(dto.envelope())
    widget.render(dto)
    document = build_statistics_html(dto, presentation=widget.report)
    stream = BytesIO()
    write_statistics_xlsx(dto, stream, presentation=widget.report)
    with_file = load_workbook(stream)
    try:
        assert_shared_unified_content(widget, document, with_file)
        kpi = "Заказы без подтвержденных покупок (отмены, возвраты и т.д.)"
        assert widget.description.text().endswith("Отбор по сезону: SS26.")
        assert kpi in document and any(cell.value == kpi for cell in with_file["Заказы"]["A"])
        labels = {label.text().casefold() for label in widget.findChildren(QLabel)}
        sheet_strings = {cell.value for sheet in with_file for row in sheet for cell in row if cell.data_type == "s"}
        removed = ("Способ оформления", "Доставка", "Чек и стоимость доставки", "Дополнительные сведения о статусах",
                   "Идентификаторы товаров", "Все товары", "Покупки без сопоставленного товара",
                   "Системные имена наблюдаемых действий", "Статусы и полнота данных")
        for title in removed:
            assert title.casefold() not in labels
            assert f'class="section-title">{escape(title)}' not in document and f'<summary>{escape(title)}' not in document
            assert not any(value == title or value.startswith(title + " —") for value in sheet_strings)
        ids = {b.table_id for p in widget.report.pages for g in p.groups for b in g.blocks if b.kind == "table"}
        assert ids.isdisjoint("unified." + key for key in (
            "orders.basket_summary", "orders.ordering_methods", "orders.delivery_types", "orders.check_delivery.RUB",
            "orders.check_delivery.KZT", "orders.status_quality", "orders.main", "products.items", "products.namespaces",
            "products.unresolved_purchases", "actions.totals", "actions.event_names"))
        assert "Фактически возвращенные единицы" not in sheet_strings
        for page_index, title, table_id in (
                (1, "Статусы заказов Mindbox", "orders.catalog"),
                (2, "Сводка сопоставления товаров", "products.resolution")):
            widget.sections.setCurrentIndex(page_index)
            assert not _table(widget, table_id).isHidden()
            assert not any(button.text() == title for button in widget.findChildren(QToolButton))
            assert f'<summary>{title}' not in document
        product_charts = {chart.data.title: chart for chart in widget.sections.widget(2).findChildren(ui.charts.StatisticsChart)}
        assert "Категории товаров" not in product_charts
        assert "Пол товара" not in product_charts
        assert not any(b.options.get("context") for p in widget.report.pages for g in p.groups for b in g.blocks)
        assert 'class="report-context"' not in document
        assert "По заказам с покупкой и известным числом позиций." not in sheet_strings
        season = product_charts["Сезон"]
        assert isinstance(season.chart().series()[0], QHorizontalBarSeries)
        expected_groups = dto.products["breakdowns"]["groups"]
        assert _table(widget, "products.group.gender").rowCount() == len(expected_groups["gender"])
        ordered = sorted(expected_groups["collection"], key=lambda row: (-row[4], row[0].casefold(), row[0]))
        assert season.data.series[0][1] == tuple(row[4] for row in ordered)
        assert season.data.labels == tuple(row[0] for row in ordered)
        season_table = _table(widget, "products.group.collection")
        assert [season_table.item(row, 0).text() for row in range(season_table.rowCount())] == list(season.data.labels)
        specs = {spec["title"]: spec for spec in _specs(widget.report)}
        assert "Категории товаров" not in specs and "Просмотры — TOP-10" in specs
        assert "Пол товара" not in specs
        assert specs["Сезон"]["data"][0]["orientation"] == "h"
    finally:
        with_file.close()
    assert dto.envelope() == original


def test_status_shares_use_unique_orders_and_preserve_zero(tab, source, monkeypatch):
    from Application.statistics_unified_presentation import _order_status_rows

    _, widget = tab
    _use_unified(widget, monkeypatch)
    rows = integration._write(source)
    duplicate_states = [r for r in rows if r["lines"][0]["status"]["ids"]["externalId"] in ("CD", "Return")]
    integration.rfm_tests._write_orders(source, rows + duplicate_states)
    dto = _report_for_ui(source)
    widget.render(dto)
    assert dto.diagnostics["counters"]["exact_duplicates"] == 1 + len(duplicate_states)
    assert dto.orders["unique_orders"]["value"] == 8
    for key, label in (("cancellations", "Доля заказов с признаками отмены, %"),
                       ("return_states", "Доля заказов со статусами возврата, %")):
        table = _table(widget, "orders." + key)
        assert table.item(_row(table, label), 1).text() == "12,50 %"

    absent = dict(value=None, known=None, quality="UNKNOWN", reasons=[])
    zero = dict(value=0, known=0, quality="KNOWN", reasons=[])
    metrics = dict(orders=zero, returned_units=absent)
    labels = dict(label="Заказы", only="Только возвраты", share_label="Доля, %")
    rows = _order_status_rows(metrics, 8, **labels)
    assert rows[0][1].value == rows[1][1] == 0
    for denominator in (0, None):
        assert [name for name, _ in _order_status_rows(metrics, denominator, **labels)] == ["Заказы"]
    assert _order_status_rows(dict(orders=absent), 8, **labels) == []


def test_pie_renderer_keeps_zero_categories_and_ignores_unknown_values(app):
    from Application.statistics_html_export import _chart_spec
    from Application.statistics_unified_presentation import _chart

    data = _chart("Группы", ("Есть", "Ноль", "Нет данных"), [("Позиции покупок", (7, 0, None))], pie=True).data
    qt = ui.charts.create_chart(data)
    slices = qt.chart().series()[0].slices()
    html = _chart_spec(data, section=2, group="products", block=0)
    assert [(s.label(), s.value()) for s in slices] == [("Есть", 7), ("Ноль", 0)]
    assert html["data"][0]["labels"] == ["Есть", "Ноль"]
    assert html["data"][0]["values"] == ["7", "0"]
    assert _chart("Группы", ("Ноль",), [("Позиции покупок", (0,))], pie=True).data.empty


@pytest.mark.parametrize("busy", (True, False))
@pytest.mark.parametrize("foreground", ("#eeeeee", "#222222"))
@pytest.mark.parametrize("scale", (1, 1.5))
def test_progress_state_matches_single_native_text_draw(app, busy, foreground, scale):
    from PyQt6.QtGui import QColor, QImage, QPainter, QRegion
    from PyQt6.QtCore import QPoint
    from PyQt6.QtWidgets import QWidget

    progress = ui.QProgressBar()
    progress.setTextVisible(False)
    progress.setRange(0, 0 if busy else 100)
    if not busy:
        progress.setValue(42)
    status = ui._ProgressStatus(progress, "Проверка аналитических структур…")
    status.resize(500, 36)
    palette = status.palette()
    palette.setColor(status.foregroundRole(), QColor(foreground))
    status.setPalette(palette)
    bar_color = QColor("#a65cf2")
    canvas = QImage(round(status.width() * scale), round(status.height() * scale), QImage.Format.Format_ARGB32)
    canvas.setDevicePixelRatio(scale)
    canvas.fill(bar_color)
    reference = canvas.copy()
    painter = QPainter(reference)
    painter.setFont(status.font())
    painter.setPen(QColor(foreground))
    text = status.text() if busy else "42% · " + status.text()
    painter.drawText(status.rect().adjusted(6, 0, -6, 0), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    status.render(canvas, QPoint(), QRegion(), QWidget.RenderFlag.DrawChildren)
    assert canvas == reference
    assert status.toolTip() == progress.accessibleDescription() == status.text()


def test_unified_summary_keeps_displayed_filters_and_progress_status(tab, source, monkeypatch):
    _, widget = tab
    _use_unified(widget, monkeypatch)
    periods = AnalysisFilter("2025-10-02", "2025-11-01", ("Пиджаки Ёлочка",), ("SS26",))
    widget.set_analysis_filter(periods)
    dto = integration._report(source, selected_items=["000001"], selected_start=periods.start_date,
                              selected_end=periods.end_date, order_period=periods.interval,
                              buyer_period=periods.interval, customer_period=periods.interval)
    widget.render(dto)
    widget.set_analysis_filter(AnalysisFilter(nomenclature_types=("Новый отбор",)))
    summary = widget.description.text()
    assert summary.startswith("Статистика рассчитана за период: 02.10.2025 — 01.11.2025. Дата расчета:")
    assert "Отбор по виду номенклатуры: Пиджаки Ёлочка. Отбор по сезону: SS26." in summary
    assert "Новый отбор" not in summary and widget.findChild(QLabel, "statisticsTitle") is None
    assert widget.status.parentWidget() is widget.progress.parentWidget() is widget.progress_container
    detail = "Ошибка расчета. " * 100
    widget.status.setText(detail)
    assert widget.status.toolTip() == detail
    assert widget.progress.accessibleDescription() == detail


def test_reply_for_another_product_filter_is_discarded(tab, source, monkeypatch, tmp_path):
    _, widget = tab
    previous = _report_for_ui(source)
    _use_unified(widget, monkeypatch)
    widget.render(previous)
    widget.set_analysis_filter(AnalysisFilter(nomenclature_types=("Type B",)))
    filtered = integration._report(source, selected_end=None, selected_items=["000001"])
    _scripted_process(monkeypatch, tmp_path, filtered.envelope(), category="cat-A")
    widget.start()
    _wait(widget)
    assert ui.UNIFIED_ERRORS["INVALID_UNIFIED_DTO"] in widget.status.text()
    assert widget.displayed_result is previous


def test_opening_tab_loads_saved_result_without_calculation(app, tmp_path, monkeypatch, source):
    from Application import statistics_unified_service as service
    from dataclasses import replace

    result = _report_for_ui(source)
    result = replace(result, source={**result.source, "reader": "INDEXED"})
    opened = []
    request = ("unified", "INDEXED", AnalysisFilter(), "", "")

    def load(**kwargs):
        opened.append(True)
        return result, request

    monkeypatch.setattr(service, "open_latest_statistics", load)
    monkeypatch.setattr(ui, "load_filter", AnalysisFilter)
    monkeypatch.setattr(ui.DatasetStatisticsTab, "_calculation_source", lambda _: ("unified", "INDEXED"))
    calls = []
    monkeypatch.setattr(ui.DatasetStatisticsTab, "_start_calculation", lambda self, **options: calls.append(options))

    window = legacy_ui.QWidget()
    widget = ui.DatasetStatisticsTab(window)
    try:
        assert not calls
        window.show()
        widget.show()
        legacy_ui.wait_until(lambda: widget.displayed_result is result)
        assert not calls
        assert widget.process is None and widget.export_button.isEnabled()
        assert widget.status.text() == "Готово"
        widget.hide()
        widget.show()
        app.processEvents()
        assert not calls
        assert opened == [True]
        for object_name in ("statisticsOpenUnifiedCache", "statisticsProductCode", "statisticsCategoryCode"):
            assert widget.findChild(legacy_ui.QWidget, object_name) is None
        widget.refresh.click()
        assert calls == [{}]
    finally:
        window.deleteLater()
        app.processEvents()


def test_production_default_has_one_source_and_does_not_load_schema3(app, tmp_path, monkeypatch):
    from Application import statistics_index as legacy_index
    from Application.statistics_cache import save_result

    root = tmp_path / "index"
    root.mkdir()
    monkeypatch.setattr(legacy_index, "DEFAULT_INDEX_DIR", root)
    monkeypatch.setattr(ui, "CACHE_PATH", tmp_path / "schema10.json")
    monkeypatch.setattr(ui, "load_filter", AnalysisFilter)
    save_result(ui.CACHE_PATH, legacy_ui.sample_result.__wrapped__())

    window = legacy_ui.QWidget()
    widget = ui.DatasetStatisticsTab(window)
    try:
        assert widget._request()[:2] == ("unified", "INDEXED")
        assert widget.displayed_result is None
        assert widget.process is None
        assert widget.findChild(QComboBox, "statisticsMode") is None
        assert widget.findChild(QComboBox, "statisticsSource") is None
    finally:
        window.deleteLater()
        app.processEvents()
