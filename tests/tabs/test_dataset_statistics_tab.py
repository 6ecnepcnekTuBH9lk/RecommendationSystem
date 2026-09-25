"""Exercise real subprocess success, failure, cancellation and deferred close."""

from dataclasses import asdict
from copy import deepcopy
from datetime import datetime
import json
import os
import re
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QProcess, QTimer, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QScrollArea, QTableWidget, QWidget, QVBoxLayout

from Application.mindbox.selection import DEFAULT_SELECTION
from Application.tabs import dataset_statistics_tab as ui
from Application.theme.apply_theme import apply_app_theme
from Application import statistics_cache as cache
from Application.settings.set_status import set_ready_status, schedule_status_reset


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    yield application
    application.setStyleSheet("")


@pytest.fixture
def sample_result():
    from Application.dataset_statistics import Coverage, DatasetStatistics

    stamp = "2026-09-24T09:47:28+00:00"
    interval = (("2025-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),)
    return asdict(DatasetStatistics(
        calculated_at=stamp,
        coverage=tuple(Coverage(name, interval if name != "Customers" else (), ("API",), stamp)
                       for name in ("Actions", "Orders", "CustomerMerges", "Customers")),
        actions=3, orders=1, order_lines=1, customers=2, action_customers=2, order_customers=1,
        interaction_users=2, actions_with_product=2, actions_without_product=1,
        view_interactions=2, favorite_interactions=0, purchase_interactions=1, purchase_quantity="4",
        mean_interactions=1.5, median_interactions=1.5, unique_source_products=1, unique_resolved_items=1,
        resolved_interactions=3, unresolved_interactions=0, resolution_rate=100.,
        action_types=(("ProsmotrProduktaVApiMethod", 3),), line_statuses=(("CP", 1, True),),
        namespaces=(("offline1C", 3, 3, 0, 100.), ("unsupported", 0, 0, 0, 0.)),
        top_products=(("000001", "Рубашка", 2, 0, 1, 3),),
        diagnostics=tuple((key, 0) for key in cache.DIAGNOSTIC_KEYS), warnings=(),
    ))


@pytest.fixture
def window(app):
    window = QWidget()
    window.status_label = QLabel(window)
    window.status_icon = QLabel(window)
    window._status_reset_timer = QTimer(window)
    window._status_reset_timer.setSingleShot(True)
    window._status_reset_timer.timeout.connect(lambda: set_ready_status(window))
    set_ready_status(window)
    layout = QVBoxLayout(window)
    tab = ui.DatasetStatisticsTab(window)
    layout.addWidget(tab)
    window.resize(1000, 700)
    window.show()
    yield window, tab
    tab._shutdown()
    window.close()
    window.deleteLater()
    app.processEvents()


def wait_until(predicate, timeout=10000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(20)
    assert predicate()


def canonical_process(monkeypatch, tmp_path):
    (tmp_path / "canonical").mkdir()
    (tmp_path / "canonical/catalog.json").write_text(json.dumps({
        "schema_version": 2, "revision": "test", "actions": {}, "orders": {}, "customer_merges": None,
        "manual_interactions": None, "selection": asdict(DEFAULT_SELECTION)}))
    catalog = tmp_path / "nomenclature.csv"
    catalog.write_text("КодНоменклатуры|Номенклатура\n000001|Test\n", encoding="utf-8")

    class Process(QProcess):
        def start(self, program, arguments):
            super().start(program, arguments + ["--raw-root", str(tmp_path), "--catalog", str(catalog)])

    monkeypatch.setattr(ui, "QProcess", Process)
    return catalog


def test_success_theme_switch_and_error_preserves_previous_result(window, monkeypatch, tmp_path):
    _, tab = window
    catalog = canonical_process(monkeypatch, tmp_path)
    tab.start()
    process = tab.process
    tab.start()
    assert tab.process is process
    wait_until(lambda: tab.process is None)
    assert tab.card_labels[0].text() == "0"
    assert tab.card_labels[3].text() == "—"
    assert tab.sections.count() == 5
    saved = tab.cache_path.read_bytes()
    assert cache.load_result(tab.cache_path)["actions"] == 0
    from Application.dataset_statistics import calculate_statistics

    result = asdict(calculate_statistics(raw_root=tmp_path, catalog_path=catalog))
    result["resolution_rate"] = 99.99834283990839
    tab.render(result)
    table = tab.sections.widget(4).findChildren(QTableWidget)[1]
    assert table.item(table.rowCount() - 1, 1).text() == "99.9983"
    for dark in (False, True):
        apply_app_theme(QApplication.instance(), dark)
        QApplication.processEvents()
        assert tab.card_labels[0].font().pointSize() == 22
        assert tab.card_labels[0].text() == "0"
    catalog.unlink()
    tab.start()
    wait_until(lambda: tab.process is None)
    assert "Catalog" in tab.status.text()
    assert "не обновлены" in tab.status.text()
    assert tab.card_labels[0].text() == "0"
    assert tab.refresh.isEnabled() and not tab.cancel_button.isEnabled()
    assert tab.cache_path.read_bytes() == saved
    wait_until(lambda: tab.window.status_label.text() == "Готов к работе")


@pytest.mark.parametrize("close", [False, True])
def test_cancel_and_close_reap_child_without_blocking_event_loop(window, monkeypatch, close, sample_result):
    window, tab = window
    cache.save_result(tab.cache_path, sample_result)
    tab.render(sample_result)
    saved = tab.cache_path.read_bytes()

    class SlowProcess(QProcess):
        def start(self, program, arguments):
            super().start(program, ["-c", "import time; time.sleep(30)"])

    monkeypatch.setattr(ui, "QProcess", SlowProcess)
    ticks = []
    timer = QTimer(tab)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    tab.start()
    wait_until(lambda: tab.process.state() == QProcess.ProcessState.Running)
    QTest.qWait(50)
    if close:
        window.close()
    else:
        tab.cancel()
    wait_until(lambda: tab.process is None)
    wait_until(lambda: not window.isVisible() if close else tab.refresh.isEnabled())
    assert ticks
    assert "отменён" in tab.status.text()
    assert not tab.kill_timer.isActive()
    assert tab.cache_path.read_bytes() == saved
    assert tab.card_labels[0].text() == "3"
    wait_until(lambda: window.status_label.text() == "Готов к работе")


def test_failed_start_resets_controls(window, monkeypatch):
    _, tab = window
    monkeypatch.setattr(ui.sys, "executable", "missing-statistics-python-executable")
    tab.start()
    wait_until(lambda: tab.process is None)
    assert "Не удалось запустить" in tab.status.text()
    assert tab.refresh.isEnabled()
    assert not tab.cancel_button.isEnabled()
    wait_until(lambda: tab.window.status_label.text() == "Готов к работе")


def test_initial_page_has_no_internal_heading_or_cache(window):
    _, tab = window
    assert tab.layout().itemAt(0).widget() is tab.description
    assert tab.description.text() == "Исходные данные Mindbox. Отбор не установлен."
    assert tab.status.text() == "В фоновом режиме будут рассчитаны данные из Mindbox."
    assert not [label for label in tab.findChildren(QLabel) if label.property("class") == "sectionHeader"]
    assert [label.text() for label in tab.card_labels] == ["—"] * 4
    assert not tab.cache_path.exists()
    assert card_titles(tab)[:4] == ["Количество взаимодействий", "Количество заказов",
                                   "Количество позиций в заказах", "Количество клиентов"]
    controls = tab.layout().itemAt(2).layout()
    assert controls.count() == 4
    assert [controls.itemAt(i).widget() for i in range(4)] == [
        tab.refresh, tab.cancel_button, tab.status, tab.progress_container]
    assert controls.stretch(3) == 1


def card_titles(page):
    return [card.findChildren(QLabel)[0].text() for card in page.findChildren(QFrame)
            if card.property("class") == "statisticsCard"]


def assert_section_names(tab):
    assert [tab.sections.tabText(i) for i in range(tab.sections.count())] == [
        "Действия", "Заказы", "Товары", "Клиенты", "Техническая информация"]


def test_five_pages_technical_content_scroll_and_no_tooltips(window, sample_result):
    _, tab = window
    assert_section_names(tab)
    for _ in range(2):
        tab.render(sample_result)
        assert_section_names(tab)
        technical = tab.sections.widget(4)
        assert isinstance(technical, QScrollArea)
        assert technical.widgetResizable()
        layout = technical.widget().layout()
        assert layout.spacing() == ui.BLOCK_SPACING
        assert layout.count() == 4
        coverage, summary, explanation, diagnostics = [layout.itemAt(i).widget() for i in range(4)]
        assert technical.findChildren(QTableWidget) == [coverage, summary, diagnostics]
        assert [coverage.horizontalHeaderItem(i).text() for i in range(3)] == [
            "Источник", "Период / состояние", "Источник данных"]
        assert [coverage.item(i, 0).text() for i in range(4)] == [
            "Действия", "Заказы", "Объединения клиентов", "Клиенты"]
        assert summary.rowCount() == 13
        assert summary.item(0, 1).text() == "3"
        assert summary.item(12, 1).text() == "100.0000"
        assert explanation.text() == (
            "Если при чтении данных обнаруживается некорректная запись, расчет завершается "
            "с ошибкой — такие записи не пропускаются. События без указанного товара учитываются отдельно.")
        assert diagnostics.horizontalHeaderItem(0).text() == "Диагностика"
        assert diagnostics.rowCount() == len(sample_result["diagnostics"]) + 5
        assert [diagnostics.item(i, 0).text() for i in range(len(sample_result["diagnostics"]))] == [
            ui.DIAGNOSTIC_LABELS[key] for key, _ in sample_result["diagnostics"]]
        for index, counts in enumerate((2, 1, 2, 1)):
            assert len(tab.sections.widget(index).findChildren(QTableWidget)) == counts
        customers = tab.sections.widget(3).findChildren(QTableWidget)[0]
        assert customers.rowCount() == 6
        assert customers.item(0, 1).text() == "2"
        for index in range(5):
            for table in tab.sections.widget(index).findChildren(QTableWidget):
                for row in range(table.rowCount()):
                    for column in range(table.columnCount()):
                        assert table.item(row, column).toolTip() == ""
    tab.sections.setCurrentIndex(4)
    QApplication.processEvents()
    scrollbar = technical.verticalScrollBar()
    assert scrollbar.maximum() > 0
    blocks = [coverage, summary, explanation, diagnostics]
    assert all(a.geometry().bottom() < b.geometry().top() for a, b in zip(blocks, blocks[1:]))
    scrollbar.setValue(scrollbar.maximum())
    QApplication.processEvents()
    # The last table's bottom is reachable, independent of theme/font pixel sizes.
    bottom = diagnostics.mapTo(technical.viewport(), diagnostics.rect().bottomLeft())
    assert technical.viewport().rect().contains(bottom)


def test_russian_presentation_dates_spacing_and_original_values(window, sample_result):
    _, tab = window
    original = deepcopy(sample_result)
    sample_result["warnings"] = tuple(ui.WARNING_LABELS)
    tab.render(sample_result)
    expected = datetime.fromisoformat(sample_result["calculated_at"]).astimezone().strftime("%d.%m.%Y %H:%M:%S")
    assert tab.description.text() == "Исходные данные Mindbox. Отбор не установлен. Дата и время расчета: " + expected
    assert re.fullmatch(r"Исходные данные Mindbox\. Отбор не установлен\. Дата и время расчета: "
                        r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}", tab.description.text())
    assert tab.status.text().splitlines()[0] == 'Статистика рассчитана за период: 01.01.2025 - 01.01.2026.'
    coverage = tab.sections.widget(4).findChildren(QTableWidget)[0]
    assert coverage.item(0, 1).text() == "01.01.2025 — 01.01.2026"
    assert coverage.item(3, 1).text().startswith("Снимок данных от ")
    assert card_titles(tab.sections.widget(0)) == ["Действия", "С товаром", "Без товара", "Клиенты"]
    assert card_titles(tab.sections.widget(1)) == ["Исходные заказы", "Исходные позиции заказов", "Клиенты", "Позиции покупок"]
    assert card_titles(tab.sections.widget(2)) == ["Уникальные исходные идентификаторы", "Уникальные товары каталога",
                                                "Распознанные взаимодействия", "Нераспознанные взаимодействия"]
    for index in range(5):
        assert tab.sections.widget(index).widget().layout().spacing() == ui.BLOCK_SPACING
    labels = [label.text() for label in tab.findChildren(QLabel)]
    cells = []
    for table in tab.findChildren(QTableWidget):
        labels.extend(table.horizontalHeaderItem(i).text() for i in range(table.columnCount()))
        cells.extend(table.item(r, c).text() for r in range(table.rowCount()) for c in range(table.columnCount()))
    for value in ("ProsmotrProduktaVApiMethod", "offline1C", "CP"):
        assert value in cells  # Technical source values are deliberately preserved.
    technical_values = {"ProsmotrProduktaVApiMethod", "offline1C", "CP"}
    presentation = "\n".join(labels + [value for value in cells if value not in technical_values])
    for phrase in ("Actions", "Orders", "Customers", "Canonical", "canonical", "Snapshot", "UTC", "item interactions",
                   "Raw events", "System name", "VIEW", "FAVORITE", "PURCHASE", "Resolution", "Namespace",
                   "Unique", "Resolved", "Unresolved", "Active users", "product", "unsupported", "Продано единиц"):
        assert phrase not in presentation
    assert sample_result["coverage"] == original["coverage"]
    assert sample_result["purchase_quantity"] == "4"
    assert sample_result["resolution_rate"] == 100.


def test_saved_result_loads_automatically_without_process(app, sample_result, monkeypatch):
    cache.save_result(ui.CACHE_PATH, sample_result)
    monkeypatch.setattr(ui.DatasetStatisticsTab, "start", lambda *args: pytest.fail("Must not recalculate"))
    window = QWidget()
    tab = ui.DatasetStatisticsTab(window)
    try:
        assert tab.process is None
        assert tab.card_labels[0].text() == "3"
        expected = datetime.fromisoformat(sample_result["calculated_at"]).astimezone().strftime("%d.%m.%Y %H:%M:%S")
        assert tab.description.text() == "Исходные данные Mindbox. Отбор не установлен. Дата и время расчета: " + expected
        assert tab.status.text() == 'Статистика рассчитана за период: 01.01.2025 - 01.01.2026.'
        assert_section_names(tab)
        assert len(tab.sections.widget(4).findChildren(QTableWidget)) == 3
    finally:
        window.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("content", [b'{"schema_version":', b'{}', b'[]', b'\xff\xfe',
                                     b'{"schema_version":1,"result":{}}'])
def test_corrupt_cache_starts_empty_without_overwriting_file(app, content, caplog):
    ui.CACHE_PATH.write_bytes(content)
    window = QWidget()
    tab = ui.DatasetStatisticsTab(window)
    try:
        assert tab.description.text() == "Исходные данные Mindbox. Отбор не установлен."
        assert [label.text() for label in tab.card_labels] == ["—"] * 4
        assert ui.CACHE_PATH.read_bytes() == content
        assert "Не удалось загрузить" in caplog.text
        assert "Traceback" not in tab.status.text()
    finally:
        window.deleteLater()
        app.processEvents()


def result_process(monkeypatch, result, *, suffix="", exit_code=0):
    message = json.dumps({"event": "result", "value": result}) + "\n"

    class Process(QProcess):
        def start(self, program, arguments):
            script = f"import sys; sys.stdout.write({(message + suffix)!r}); sys.stdout.flush(); sys.exit({exit_code})"
            super().start(program, ["-c", script])

    monkeypatch.setattr(ui, "QProcess", Process)


def test_new_success_atomically_replaces_cache_before_render(window, sample_result, monkeypatch):
    _, tab = window
    cache.save_result(tab.cache_path, sample_result)
    previous = tab.cache_path.read_bytes()
    candidate = deepcopy(sample_result)
    candidate["calculated_at"] = "2026-09-25T09:47:28+00:00"
    candidate["customers"] = 5
    result_process(monkeypatch, candidate)
    render = tab.render

    def checked_render(result):
        saved = cache.load_result(tab.cache_path)
        assert saved["calculated_at"] == candidate["calculated_at"]
        assert saved["customers"] == 5
        render(result)

    monkeypatch.setattr(tab, "render", checked_render)
    tab.start()
    wait_until(lambda: tab.process is None)
    assert tab.card_labels[3].text() == "5"
    assert tab.cache_path.read_bytes() != previous
    assert not list(tab.cache_path.parent.glob(".dataset-statistics-*.tmp"))
    assert [p.name for p in tab.cache_path.parent.glob("*statistics*.json")] == [tab.cache_path.name]
    assert tab.status.text() == 'Статистика рассчитана за период: 01.01.2025 - 01.01.2026.'
    wait_until(lambda: tab.window.status_label.text() == "Готов к работе")


@pytest.mark.parametrize("failure", ["incomplete", "bad_type", "bad_date", "bad_rows", "nan", "trailing_json",
                                    "json_error", "deep_json", "prior_incomplete", "nonzero_exit", "crash", "replace", "fsync"])
def test_failed_new_result_preserves_cache_and_display(window, sample_result, monkeypatch, failure):
    _, tab = window
    cache.save_result(tab.cache_path, sample_result)
    tab.render(sample_result)
    previous = tab.cache_path.read_bytes()
    candidate = deepcopy(sample_result)
    candidate["customers"] = 9
    if failure == "incomplete":
        del candidate["top_products"]
    elif failure == "bad_type":
        candidate["actions"] = True
    elif failure == "bad_date":
        candidate["calculated_at"] = "not a date"
    elif failure == "bad_rows":
        candidate["namespaces"] = [["offline1C"]]
    elif failure == "nan":
        candidate["resolution_rate"] = float("nan")
    elif failure in ("replace", "fsync"):
        def fail(*args):
            raise OSError("Нет доступа к файлу")
        monkeypatch.setattr(cache.os, failure, fail)
    result_process(monkeypatch, candidate,
                   suffix={"trailing_json": "{", "json_error": "bad JSON\n",
                           "deep_json": "[" * 1200 + "0" + "]" * 1200 + "\n"}.get(failure, ""),
                   exit_code=1 if failure == "nonzero_exit" else 0)
    if failure == "prior_incomplete":
        result_process(monkeypatch, None, suffix=json.dumps({"event": "result", "value": candidate}) + "\n")
    if failure == "crash":
        class CrashProcess(QProcess):
            def start(self, program, arguments):
                super().start(program, ["-c", "import time; time.sleep(30)"])
        monkeypatch.setattr(ui, "QProcess", CrashProcess)
    tab.start()
    if failure == "crash":
        wait_until(lambda: tab.process.state() == QProcess.ProcessState.Running)
        tab.process.kill()
    wait_until(lambda: tab.process is None)
    assert tab.cache_path.read_bytes() == previous
    assert tab.card_labels[3].text() == "2"
    assert "не обновлены" in tab.status.text()
    assert tab.refresh.isEnabled() and not tab.cancel_button.isEnabled()
    assert not list(tab.cache_path.parent.glob(".dataset-statistics-*.tmp"))
    wait_until(lambda: tab.window.status_label.text() == "Готов к работе")


def test_information_style_and_existing_icons_in_both_themes(window):
    _, tab = window
    for dark in (False, True):
        apply_app_theme(QApplication.instance(), dark)
        QApplication.processEvents()
        for label in (tab.description, tab.status):
            assert label.property("class") == "statisticsInfo"
            alignment = Qt.AlignmentFlag.AlignHCenter if label is tab.description else Qt.AlignmentFlag.AlignLeft
            assert label.alignment() & alignment
            assert label.font().italic()
            assert label.font().pointSize() == 10
        for button, filename in ((tab.refresh, "statistic.png"), (tab.cancel_button, "failure.png")):
            expected = QIcon(str(ui.ICONS_DIR / filename))
            assert not button.icon().isNull()
            assert button.icon().pixmap(button.iconSize()).toImage() == expected.pixmap(button.iconSize()).toImage()


def test_progress_and_shared_status_resist_stale_reset(window, monkeypatch):
    window, tab = window
    schedule_status_reset(window, 5)
    window._status_reset_timer.start(20)

    class ProgressProcess(QProcess):
        def start(self, program, arguments):
            message = json.dumps({"event": "progress", "value": "actions: 10 000"})
            super().start(program, ["-u", "-c", f"import time; print({message!r}, flush=True); time.sleep(30)"])

    monkeypatch.setattr(ui, "QProcess", ProgressProcess)
    tab.start()
    assert window.status_label.text() == "Расчет статистики..."
    assert not window._status_reset_timer.isActive()
    wait_until(lambda: tab.status.text() == "Расчет → Действия: 10 000")
    QTest.qWait(50)
    assert tab.process is not None
    assert window.status_label.text() == "Расчет статистики..."
    # Even a later reset requested elsewhere cannot leave this process "ready".
    schedule_status_reset(window, 0)
    QTest.qWait(30)
    assert window.status_label.text() == "Расчет статистики..."
    tab.cancel()
    wait_until(lambda: tab.process is None)
    wait_until(lambda: window.status_label.text() == "Готов к работе")


@pytest.mark.parametrize("actions,orders,expected", [
    ([("2024-01-01", "2024-12-01")], [("2024-06-01", "2025-01-01")],
     'Статистика рассчитана за период: 01.06.2024 - 01.12.2024.'),
    ([("2024-01-01", "2024-02-01"), ("2024-03-01", "2024-04-01")], [("2024-01-01", "2024-04-01")],
     'Статистика рассчитана за периоды: 01.01.2024 - 01.02.2024; 01.03.2024 - 01.04.2024.'),
    ([("2024-01-01", "2024-02-01")], [("2024-02-01", "2024-03-01")], "Статистика рассчитана."),
    ([], [("2024-06-01", "2024-07-01")], 'Статистика рассчитана за период: 01.06.2024 - 01.07.2024.'),
    ([], [], "Статистика рассчитана."),
])
def test_period_uses_interaction_coverage_not_customer_snapshots(sample_result, actions, orders, expected):
    for source, intervals in zip(sample_result["coverage"][:2], (actions, orders)):
        source["intervals"] = [(a + "T00:00:00+00:00", b + "T00:00:00+00:00") for a, b in intervals]
    # Customers and merges still cover 2025: they must not enter the calculation.
    sample_result["coverage"][3]["intervals"] = sample_result["coverage"][2]["intervals"]
    assert ui._calculation_message(sample_result) == expected
