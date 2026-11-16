"""Pinned Unified detail format 2 through the production Qt worker."""

import csv
from io import TextIOWrapper
import json
from zipfile import ZipFile

import pytest
from PyQt6.QtCore import QProcess

from Application.analysis_filter import AnalysisFilter
from Application.statistics_detail_process import DetailExportProcess
from Application import statistics_detail_process as process
from Application import statistics_storage_safety as safety
from Application import statistics_unified_index as index
from Application.statistics_index import protect_directory
from Application.statistics_unified_detail import filter_snapshot
from Application.tabs import dataset_statistics_tab as ui
from tests import test_statistics_unified_integration as integration
from tests.tabs import test_dataset_statistics_tab as qt


@pytest.fixture
def generation(tmp_path, monkeypatch):
    source = integration.source.__wrapped__(tmp_path, monkeypatch)
    integration._write(source)
    root, catalog, _, _, target = source
    protect_directory(target)
    with monkeypatch.context() as patch:
        patch.setattr(index, "protect_directory", protect_directory)
        patch.setattr(index, "validate_storage_root", safety.validate_storage_root)
        index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    report = integration._report(source, reader="INDEXED", index_dir=target, selected_end=None)

    class WorkerProcess(QProcess):
        def start(self, program, args):
            super().start(program, [*args, "--raw-root", str(root), "--catalog", str(catalog), "--index-dir", str(target)])

    monkeypatch.setattr(process, "QProcess", WorkerProcess)
    return source, report


def _run(report, destination, request):
    app = qt.QApplication.instance() or qt.QApplication([])
    worker = DetailExportProcess()
    messages = []
    worker.completed.connect(messages.append)
    worker.start({"destination": destination}, report_result=report, detail=request)
    try:
        qt.wait_until(lambda: bool(messages), timeout=70000)
        return messages[0], worker.diagnostic
    finally:
        if not messages:
            worker.shutdown()
        worker.deleteLater()
        app.processEvents()


def _rows(archive, name):
    with archive.open(name) as stream, TextIOWrapper(stream, encoding="utf-8-sig") as decoded:
        return list(csv.DictReader(decoded, delimiter=";"))


def test_qprocess_detail_v2_has_matching_links_quality_and_history(generation, tmp_path):
    _, report = generation
    destination = tmp_path / "detail.zip"
    message, diagnostic = _run(report, destination, filter_snapshot(AnalysisFilter(), "", ""))
    assert message == {"event": "finished", "files": 6}, diagnostic
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format_version"] == 2
        assert manifest["source_fingerprint"] == report.metadata["fingerprint"]
        assert manifest["source_binding"] == report.metadata["source_binding"]
        orders = _rows(archive, "orders.csv")
        lines = _rows(archive, "order_lines.csv")
        clients = _rows(archive, "clients.csv")
    assert len(orders) == 8
    assert sum(row["observed_purchase"] == "1" for row in orders) == 3
    assert set(row["order_key"] for row in lines) <= {row["order_key"] for row in orders}
    assert {row["client_key"] for row in orders if row["client_key"]} <= {row["client_key"] for row in clients}
    cp = next(row for row in lines if row["had_cp"] == "1")
    assert cp["RUB_recognized_purchase_value"] == "500"
    assert cp["RUB_partial_nominal_value"] == "500"
    assert any(row["KZT_recognized_purchase_value"] == "0" and row["KZT_valued_positions"] == "1" for row in orders)
    assert any(row["LAST_12M_RUB_segment"] for row in clients)
    assert all(row["order_key"].startswith("'O") for row in orders)


def test_mismatched_detail_filter_keeps_previous_zip(generation, tmp_path):
    _, report = generation
    destination = tmp_path / "detail.zip"
    destination.write_bytes(b"previous")
    message, _ = _run(report, destination, filter_snapshot(AnalysisFilter(), "000002", ""))
    assert message == {"event": "error", "code": "REPORT_CONTEXT_MISMATCH"}
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".statistics-report-*"))


@pytest.mark.parametrize("include_details", [True, False])
def test_existing_tab_exports_filter_of_displayed_report(generation, tmp_path, monkeypatch, include_details):
    source, _ = generation
    selection = AnalysisFilter(nomenclature_types=("Type B",))
    report = integration._report(source, reader="INDEXED", index_dir=source[-1], selected_end=None,
                                 product_filter=lambda _, item: selection.matches(item))
    destination = tmp_path / "from-tab.zip"
    app = qt.QApplication.instance() or qt.QApplication([])
    monkeypatch.setattr(ui, "CACHE_PATH", tmp_path / "legacy.json")
    monkeypatch.setattr(ui, "load_filter", AnalysisFilter)
    from Application import statistics_index as legacy_index

    monkeypatch.setattr(legacy_index, "DEFAULT_INDEX_DIR", source[-1])
    monkeypatch.setattr(ui, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *_: (str(destination), "ZIP"))
    answer = ui.QMessageBox.StandardButton.Yes if include_details else ui.QMessageBox.StandardButton.No
    monkeypatch.setattr(ui.QMessageBox, "question", lambda *_: answer)

    windows = qt.statistics_window(app, ui.DatasetStatisticsTab)
    try:
        _, tab = next(windows)
        qt.wait_until(lambda: not tab._busy())
        tab.set_analysis_filter(selection)
        tab.render(report)
        tab.set_analysis_filter(AnalysisFilter(nomenclature_types=("Type A",)))
        assert tab.export_button.isEnabled()
        tab.export_button.click()
        qt.wait_until(lambda: tab.detail_process is None, timeout=70000)
        assert destination.is_file(), tab.status.text()
    finally:
        next(windows, None)
    with ZipFile(destination) as archive:
        name = next(name for name in archive.namelist() if name.endswith(".html"))
        header = archive.read(name).decode().split("<header>", 1)[1].split("</header>", 1)[0]
        assert "Отбор по виду номенклатуры" in header and "Type B" in header and "Type A" not in header
        if include_details:
            rows = _rows(archive, "order_lines.csv")
            assert len(rows) == 1
            assert rows[0]["item"] == "'000002"
        else:
            assert len(archive.namelist()) == 2


def test_replaced_index_generation_cannot_export_previous_report(generation, tmp_path, monkeypatch):
    source, report = generation
    root, catalog, _, _, target = source
    with monkeypatch.context() as patch:
        patch.setattr(index, "protect_directory", protect_directory)
        patch.setattr(index, "validate_storage_root", safety.validate_storage_root)
        index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    destination = tmp_path / "stale.zip"
    message, _ = _run(report, destination, filter_snapshot(AnalysisFilter(), "", ""))
    assert message == {"event": "error", "code": "SOURCE_CHANGED"}
    assert not destination.exists()
    assert not list(tmp_path.glob(".statistics-report-*"))
