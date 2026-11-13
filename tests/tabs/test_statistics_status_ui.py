"""DTO 11 presentation and subprocess integration through real Qt widgets."""

from datetime import datetime, timezone
from copy import deepcopy
import json

import pytest
from PyQt6.QtCore import QProcess, QThreadPool, Qt
from PyQt6.QtWidgets import QComboBox, QLabel, QTableWidget, QTableView

from Application import statistics_storage_safety as safety
from Application import statistics_unified_index as index
from Application.statistics_index import protect_directory
from Application.analysis_filter import AnalysisFilter
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


def test_real_reference_qprocess_calculates_and_opens_cache(tab, source, monkeypatch):
    _, widget = tab
    work, cache = _real_process(source, monkeypatch)
    _use_unified(widget, monkeypatch)
    assert widget.displayed_result is None and widget.cache_button.isEnabled()
    widget.start()
    assert widget.process is not None and not widget.refresh.isEnabled()
    _wait(widget)
    result = widget.displayed_result
    assert result is not None, widget.status.text()
    assert result.source["reader"] == "REFERENCE"
    assert result.orders["purchase_orders"]["known"] == 4
    path = cache / "unified-v11-latest.json"
    saved = path.read_bytes()
    assert not list(work.iterdir())
    widget.cache_button.click()
    _wait(widget)
    assert widget.displayed_result == result and path.read_bytes() == saved
    widget.category_code.setText("cat-B")
    assert "Показан отчёт с прежним отбором" in widget.description.text()
    widget.cache_button.click()
    _wait(widget)
    assert "Совместимого сохранённого отчёта нет" in widget.status.text()
    assert widget.displayed_result == result and path.read_bytes() == saved


def test_qt_widgets_cover_dto_status_money_rfm_cohorts_and_navigation(tab, source, monkeypatch):
    _, widget = tab
    _use_unified(widget, monkeypatch)
    dto = integration._report(source, selected_end=None, calculated_at=datetime(2026, 11, 12, tzinfo=timezone.utc))
    widget.render(dto)
    catalog = _table(widget, "orders.catalog")
    assert catalog.rowCount() == 43
    codes = {catalog.item(i, 1).text() for i in range(43)}
    assert {"F", "sold", "CP", "delivering", "Return", "Mindbox-Tech-Cancellation"} <= codes
    tech = next(i for i in range(43) if catalog.item(i, 1).text() == "Mindbox-Tech-Cancellation")
    assert catalog.item(tech, 2).text() == "—"
    quality = _table(widget, "orders.status_quality")
    assert {quality.item(i, 0).text() for i in range(quality.rowCount())} >= {"Неизвестный код", "Невалидный код"}
    main = _table(widget, "orders.main")
    assert main.item(_row(main, "Покупочные заказы"), 1).text() == "4"
    assert main.item(_row(main, "Заказы с CP"), 1).data(Qt.ItemDataRole.UserRole).known == 1
    assert _table(widget, "orders.cancellations").item(0, 1).data(Qt.ItemDataRole.UserRole).known == 1
    assert _table(widget, "orders.return_states").item(0, 1).data(Qt.ItemDataRole.UserRole).known == 1
    rub, kzt = (_table(widget, "orders.money." + code) for code in ("RUB", "KZT"))
    rub_recognized = rub.item(_row(rub, "Признанная стоимость"), 1)
    assert "Недоступно" in rub_recognized.text() and "1 035,00" in rub_recognized.text()
    assert "частичного выполнения" in rub_recognized.toolTip()
    assert kzt.item(_row(kzt, "Признанная стоимость"), 1).text() == "0,00"
    assert _table(widget, "actions.intersections").rowCount() == 4
    assert _table(widget, "products.items").rowCount() == 2
    assert _table(widget, "customers.population").item(0, 1).text() == "3"
    assert _table(widget, "customers.purchases").item(_row(_table(widget, "customers.purchases"), "Повторные покупатели"), 1).text() == "2"
    segments = _table(widget, "rfm.LAST_12M.RUB.segments")
    assert segments.item(_row(segments, "Оценка недоступна"), 1).text() == "2"
    assert _table(widget, "rfm.LAST_48M.KZT.thresholds").rowCount() == 3
    assert _table(widget, "cohorts.repeat").rowCount() == 15
    assert _table(widget, "cohorts.retention").rowCount() == 36
    assert _table(widget, "cohorts.channels").rowCount() >= 2
    assert _table(widget, "cohorts.transitions").rowCount() == len(dto.cohorts["categories"]["transitions"])
    technical = _table(widget, "technical.metadata")
    assert technical.item(_row(technical, "Временная основа"), 1).text() == "ORDER_CREATED_PURCHASE_PROXY_V1"
    assert technical.item(_row(technical, "DTO"), 1).text() == "11"
    for page in range(widget.sections.count()):
        widget.sections.setCurrentIndex(page)
    for group in range(widget.customer_sections.count()):
        widget.customer_sections.setCurrentIndex(group)
    assert widget.displayed_result is dto
    assert widget.export_button.isEnabled()


def test_missing_index_does_not_rebuild_or_fall_back(tab, source, monkeypatch):
    _, widget = tab
    _real_process(source, monkeypatch)
    sentinel = source[-1] / "current.json"
    sentinel.write_text("legacy", encoding="utf-8")
    _use_unified(widget, monkeypatch, "INDEXED")
    widget.start()
    _wait(widget)
    assert "INDEXED 4 отсутствует" in widget.status.text()
    assert widget.displayed_result is None and sentinel.read_text(encoding="utf-8") == "legacy"
    assert not (source[-1] / index.POINTER_NAME).exists()


def test_actual_indexed_qprocess_filter_and_stale_snapshot(tab, source, monkeypatch):
    _, widget = tab
    root, catalog, _, _, target = source
    protect_directory(target)
    with monkeypatch.context() as patch:
        patch.setattr(index, "protect_directory", protect_directory)
        patch.setattr(index, "validate_storage_root", safety.validate_storage_root)
        index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    pointer = (target / index.POINTER_NAME).read_bytes()
    _real_process(source, monkeypatch)
    _use_unified(widget, monkeypatch, "INDEXED")
    widget.category_code.setText("cat-B")
    widget.start()
    _wait(widget)
    dto = widget.displayed_result
    assert dto is not None, widget.status.text()
    assert dto.orders["purchase_orders"]["known"] == 1
    assert dto.contexts["order_product_slice"]["product_filter_key"]
    assert dto.contexts["rfm_48m"]["product_filter_key"] is None
    assert dto.rfm["base_buyers"] == dto.cohorts["quality"]["base_buyers"] == 2
    assert "категория cat-B" in widget.description.text()
    rows = integration._write(source)
    integration.rfm_tests._write_orders(source, rows[:-1])
    widget.start()
    _wait(widget)
    assert "индекс устарел" in widget.status.text()
    assert widget.displayed_result is dto
    assert (target / index.POINTER_NAME).read_bytes() == pointer


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
    assert "отменён" in widget.status.text()
    assert widget.displayed_result is dto
    if close_tab:
        legacy_ui.wait_until(lambda: not widget.isVisible())
    assert QThreadPool.globalInstance().waitForDone(1000)


def test_large_unified_table_keeps_all_rows_without_allocating_items(tab):
    _, widget = tab
    layouts = widget._pages()
    block = ui.report.ReportBlock("table", headers=("Значение",), rows=tuple((i,) for i in range(300)),
                                  table_id="unified.large", options={"context_id": "buyer_period"})
    ui._render_report_blocks(layouts[0], [block])
    table = next(t for t in widget.findChildren(QTableView) if t.property("statisticsTableId") == "unified.large")
    assert not isinstance(table, QTableWidget)
    assert table.model().rowCount() == 300
    assert table.model().index(299, 0).data() == "299"


def test_every_unified_chart_has_a_full_matching_table(tab, source, monkeypatch):
    _, widget = tab
    _use_unified(widget, monkeypatch)
    dto = _report_for_ui(source)
    widget.render(dto)
    assert any(isinstance(label, QLabel) and "датам оформления" in label.text() for label in widget.findChildren(QLabel))
    action = next(block.data for block in widget.report.pages[0].groups[0].blocks if block.kind == "chart")
    table = _table(widget, "actions.totals")
    assert tuple(table.item(row, 1).data(Qt.ItemDataRole.UserRole).value for row in range(table.rowCount())) == action.series[0][1]
    assert widget.findChild(ui.RfmSegmentsWidget).choices
    assert widget.findChild(ui.CohortRetentionHeatmap).model().rowCount() == len(dto.cohorts["months"])


def test_reply_for_another_product_filter_is_discarded(tab, source, monkeypatch, tmp_path):
    _, widget = tab
    previous = _report_for_ui(source)
    _use_unified(widget, monkeypatch)
    widget.render(previous)
    widget.category_code.setText("cat-B")
    filtered = integration._report(source, selected_end=None, selected_items=["000001"])
    _scripted_process(monkeypatch, tmp_path, filtered.envelope(), category="cat-A")
    widget.start()
    _wait(widget)
    assert ui.UNIFIED_ERRORS["INVALID_UNIFIED_DTO"] in widget.status.text()
    assert widget.displayed_result is previous


def test_compatible_cache_is_requested_once_on_show(app, tmp_path, monkeypatch):
    from Application import statistics_index as legacy_index
    from Application.statistics_unified_cache import LATEST_RESULT_NAME

    target = tmp_path / "index"
    target.mkdir()
    (target / index.POINTER_NAME).touch()
    cache = tmp_path / "filtered_data" / "statistics_unified_cache"
    cache.mkdir(parents=True)
    (cache / LATEST_RESULT_NAME).touch()
    monkeypatch.setattr(legacy_index, "DEFAULT_INDEX_DIR", target)
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
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
        app.processEvents()
        assert calls == [{"cache_only": True}]
        widget.hide()
        widget.show()
        app.processEvents()
        assert len(calls) == 1
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
        assert widget.status.text() == ui.UNIFIED_ERRORS["INDEX_MISSING"]
        assert widget.findChild(QComboBox, "statisticsMode") is None
        assert widget.findChild(QComboBox, "statisticsSource") is None
    finally:
        window.deleteLater()
        app.processEvents()
