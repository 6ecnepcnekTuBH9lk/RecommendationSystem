import json
from dataclasses import replace

import pytest
from PyQt6.QtCore import QDate, QPoint, QProcess, QThreadPool, QTimer, Qt
from PyQt6.QtWidgets import QFrame, QLabel, QTableWidget, QWidget, QComboBox, QStyleOptionViewItem

from Application.analysis_filter import AnalysisFilter, AnalysisOptions
from Application.analysis_filter_settings import load_filter, save_filter
from Application.tabs import analysis_filter_tab as filters
from Application.tabs import dataset_statistics_tab as statistics
from Application import statistics_cache as cache
from Application.theme.apply_theme import apply_app_theme
from Application.settings.set_status import set_ready_status
import test_dataset_statistics_tab as baseline
import test_dataset_statistics as canonical_fixture
from Application.dataset_statistics import load_analysis_options as real_load_options
from test_dataset_statistics_tab import wait_until, canonical_process

app, sample_result, window = baseline.app, baseline.sample_result, baseline.window
pytestmark = pytest.mark.usefixtures("window_settings")
dataset = canonical_fixture.dataset
real_stores = filters.mapping.load_stores
real_cities = filters.mapping.load_cities


OPTIONS = AnalysisOptions("2025-01-01", "2025-12-31", ("Брюки", "Рубашки", None), ("Зима", "Лето", None))


@pytest.fixture
def filter_tab(app, monkeypatch):
    monkeypatch.setattr(filters, "load_analysis_options", lambda: OPTIONS)
    monkeypatch.setattr(filters.mapping, "load_stores", lambda: (("A", "Магазин A"), ("B", "Магазин B")))
    monkeypatch.setattr(filters.mapping, "load_cities", lambda: ("Казань", "Москва"))
    parent = QWidget()
    parent.status_label, parent.status_icon = QLabel(parent), QLabel(parent)
    parent._status_reset_timer = QTimer(parent)
    parent._status_reset_timer.setSingleShot(True)
    parent._status_reset_timer.timeout.connect(lambda: set_ready_status(parent))
    tab = filters.AnalysisFilterTab(parent)
    yield tab
    QThreadPool.globalInstance().waitForDone(1000)
    app.processEvents()
    QThreadPool.globalInstance().waitForDone(1000)
    app.processEvents()
    parent.deleteLater()
    app.processEvents()


def test_readiness_no_data_then_background_refresh(filter_tab, monkeypatch):
    tab = filter_tab
    assert not tab.start_date.isEnabled() and not tab.apply_button.isEnabled()
    assert "загрузить" in tab.message.text()
    wait_until(lambda: tab._mapping_editable)
    assert tab.controls.isEnabled()
    assert tab.types.items.count() == 3
    assert tab.collections.items.item(2).text() == "Не указано"
    assert tab.types.text() == "Все значения"
    new_options = replace(OPTIONS, collections=("Новая", "Лето", None))
    monkeypatch.setattr(filters, "load_analysis_options", lambda: new_options)
    tab.refresh_options()
    wait_until(lambda: tab.options == new_options)
    assert tab.collections.items.item(0).text() == "Новая"


def test_apply_reset_draft_and_signal_to_statistics(filter_tab, app, monkeypatch):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    parent = tab.parent()
    parent.analysis_filter_tab = tab
    stats = statistics.DatasetStatisticsTab(parent)
    monkeypatch.setattr(stats, "start", lambda: pytest.fail("Applying must not calculate"))
    tab.types.set_checked(False)
    assert not tab.apply_button.isEnabled()
    tab.apply()
    assert not tab.settings_path.exists()
    tab.types.items.item(1).setCheckState(Qt.CheckState.Checked)
    assert tab.types.text() == "Рубашки"
    assert stats.configured_filter == AnalysisFilter()
    assert not tab.settings_path.exists()
    tab.apply()
    assert load_filter(tab.settings_path) == stats.configured_filter == AnalysisFilter(nomenclature_types=("Рубашки",))
    assert stats.process is None and "вид номенклатуры: Рубашки" in stats.description.text()
    assert tab.window.status_label.text() == "Отбор сохранен"
    assert tab.message.isHidden()
    assert tab.window._status_reset_timer.isActive()
    tab.reset()
    assert tab.window.status_label.text() == "Отбор сброшен"
    assert stats.configured_filter == load_filter(tab.settings_path) == AnalysisFilter()
    assert stats.process is None and tab.types.text() == "Все значения"


def test_validation_dates_and_save_failure(filter_tab, monkeypatch):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    tab.start_date.setDate(QDate(2025, 10, 1))
    tab.end_date.setDate(QDate(2025, 3, 1))
    assert not tab.apply_button.isEnabled()
    assert tab.window.status_label.text() == "Дата начала не может быть позже даты окончания"
    assert tab.message.isHidden()
    tab.end_date.setDate(QDate(2025, 11, 1))
    assert tab.apply_button.isEnabled()
    def failure(*args):
        raise OSError
    monkeypatch.setattr(filters, "save_filter", failure)
    tab.apply()
    assert tab.configured_filter == AnalysisFilter()
    assert "Не удалось сохранить" in tab.window.status_label.text()
    assert tab.message.isHidden()


def test_reconciliation_persisted_without_calculation(filter_tab):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    tab.configured_filter = AnalysisFilter("2024-01-01", "2026-01-01", ("gone",), ("Лето", "gone"))
    tab._options_loaded(replace(OPTIONS, nomenclature_types=("Рубашки",)), "")
    assert tab.configured_filter == AnalysisFilter(collections=("Лето",))
    assert load_filter(tab.settings_path) == tab.configured_filter
    assert "скорректирован" in tab.window.status_label.text()
    assert tab.message.isHidden()


@pytest.mark.parametrize("same", [False, True])
def test_cached_result_stale_startup_keeps_values(app, sample_result, monkeypatch, same):
    configured = AnalysisFilter() if same else AnalysisFilter(collections=("Лето",))
    save_filter(configured, filters.SETTINGS_PATH)
    cache.save_result(statistics.CACHE_PATH, sample_result)
    parent = QWidget()
    parent.analysis_filter_tab = filters.AnalysisFilterTab(parent)
    tab = statistics.DatasetStatisticsTab(parent)
    try:
        assert tab.configured_filter == configured
        assert ("требуется пересчитать" in tab.description.text()) is not same
        assert "Исходные данные Mindbox." in tab.description.text()
        assert "Дата и время расчета:" in tab.description.text()
        assert [label.text() for label in tab.card_labels] == ["3", "1", "1", "2"]
        before = tab.sections.widget(1).findChildren(QTableWidget)[0]
        tab.set_analysis_filter(AnalysisFilter(nomenclature_types=("A", "B", "C", "D")))
        assert "4 значений" in tab.description.text()
        assert tab.sections.widget(1).findChildren(QTableWidget)[0] is before
        assert tab.process is None
    finally:
        parent.deleteLater()
        app.processEvents()


def test_worker_receives_snapshot_and_changed_configuration_stays_stale(window, monkeypatch, sample_result, tmp_path):
    _, tab = window
    tab.render(sample_result)
    configured = AnalysisFilter(collections=("Лето",))
    tab.set_analysis_filter(configured)
    canonical_process(monkeypatch, tmp_path)
    tab.start()
    assert tab.running_filter == configured
    assert json.loads(tab.process.arguments()[3]) == json.loads(json.dumps(configured.to_dict()))
    tab.set_analysis_filter(AnalysisFilter(collections=("Зима",)))
    wait_until(lambda: tab.process is None)
    assert AnalysisFilter.from_dict(tab.displayed_result["analysis_filter"]) == configured
    assert "требуется пересчитать" in tab.description.text()
    assert "сезон: Зима" in tab.description.text()


@pytest.mark.parametrize("outcome", ["failure", "cancel"])
def test_failed_or_cancelled_filtered_run_preserves_cache(window, sample_result, monkeypatch, outcome):
    _, tab = window
    cache.save_result(tab.cache_path, sample_result)
    tab.render(sample_result)
    original = tab.cache_path.read_bytes()
    tab.set_analysis_filter(AnalysisFilter(collections=("Лето",)))
    class Process(QProcess):
        def start(self, program, arguments):
            super().start(program, ["-c", "import time; time.sleep(20)" if outcome == "cancel" else "raise SystemExit(1)"])
    monkeypatch.setattr(statistics, "QProcess", Process)
    tab.start()
    if outcome == "cancel":
        tab.cancel()
    wait_until(lambda: tab.process is None)
    assert tab.cache_path.read_bytes() == original
    assert tab.displayed_result == sample_result
    assert "требуется пересчитать" in tab.description.text()


def test_worker_filter_mismatch_preserves_cache(window, sample_result, monkeypatch):
    _, tab = window
    cache.save_result(tab.cache_path, sample_result)
    tab.render(sample_result)
    original = tab.cache_path.read_bytes()
    tab.set_analysis_filter(AnalysisFilter(collections=("Лето",)))
    message = json.dumps({"event": "result", "value": sample_result})
    class Process(QProcess):
        def start(self, program, arguments):
            super().start(program, ["-u", "-c", f"print({message!r})"])
    monkeypatch.setattr(statistics, "QProcess", Process)
    tab.start()
    wait_until(lambda: tab.process is None)
    assert tab.cache_path.read_bytes() == original
    assert "Некорректный ответ" in tab.status.text()
    assert "требуется пересчитать" in tab.description.text()


def test_calculated_period_label_uses_calculated_filter(sample_result):
    sample_result["analysis_filter"] = AnalysisFilter("2025-03-01", "2025-05-31").to_dict()
    assert statistics._calculation_message(sample_result) == "Статистика рассчитана за период 01.03.2025–31.05.2025."


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [700, 1200])
def test_filter_layout_theme_popup(filter_tab, app, monkeypatch, dark, width):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    apply_app_theme(app, dark)
    monkeypatch.setattr(tab, "refresh_options", lambda **kwargs: None)
    tab.setParent(None)
    tab.resize(width, 600)
    tab.show()
    app.processEvents()
    headings = [label.text() for label in tab.findChildren(QLabel)]
    assert "Отбор для анализа" in headings
    assert "Правила формирования рекомендаций" in headings
    assert "В разработке" in headings
    section_headers = [label for label in tab.findChildren(QLabel) if label.property("class") == "sectionHeader"]
    assert len(section_headers) == 3
    assert all(label.height() >= label.heightForWidth(label.width()) for label in section_headers)
    assert all(not label.font().italic() and label.alignment() == Qt.AlignmentFlag.AlignCenter for label in section_headers)
    assert not any(label.property("class") == "statisticsSection" for label in tab.findChildren(QLabel))
    assert {"Дата начала:", "Дата окончания:", "Вид номенклатуры:", "Сезон:"}.issubset(headings)
    separator = tab.findChild(QFrame, "vSeparator")
    layout = tab.layout()
    left, right = layout.itemAt(0).widget(), layout.itemAt(2).widget()
    assert layout.count() == 3
    assert left is tab.controls.parentWidget()
    assert any(label.text() == "В разработке" for label in right.findChildren(QLabel))
    assert layout.stretch(0) == layout.stretch(2) == 1
    assert separator is layout.itemAt(1).widget()
    assert left.geometry().right() < separator.geometry().left()
    assert separator.geometry().right() < right.geometry().left()
    for column in (left, right):
        assert column.isVisible() and column.geometry().isValid()
    assert separator.frameShape() == QFrame.Shape.NoFrame
    assert separator.minimumWidth() == separator.maximumWidth() == 1
    assert separator.height() > tab.controls.height()
    inputs = (tab.start_date, tab.end_date, tab.types, tab.collections)
    assert len({widget.height() for widget in inputs}) == 1
    assert len({widget.maximumWidth() for widget in inputs}) == 1
    assert max(widget.width() for widget in inputs) - min(widget.width() for widget in inputs) <= 2
    assert tab.types.width() < tab.layout().itemAt(0).widget().width()
    buttons = tab.apply_button.parentWidget().layout().itemAt(3).layout()
    assert [buttons.itemAt(i).widget().text().strip() for i in range(2)] == ["Применить", "Сбросить"]
    assert tab.apply_button.width() == tab.reset_button.width()
    if width == 1200:
        assert abs(left.width() - right.width()) <= 1
    # Minimum-size negotiation can give unequal columns in a narrow window.
    assert all(widget.isVisible() and widget.isEnabled() and widget.rect().isValid() for widget in inputs)
    tab.types.populate(tuple(f"Вид {i:02}" for i in range(50)) + (None,), None)
    tab.types.menu.popup(tab.types.mapToGlobal(tab.types.rect().bottomLeft()))
    app.processEvents()
    assert tab.types.items.verticalScrollBar().maximum() > 0
    assert tab.types.items.item(0).checkState() == Qt.CheckState.Checked
    tab.types.menu.hide()
    tab.close()
    tab.deleteLater()


def test_icons_use_existing_resources(app, monkeypatch, tmp_path):
    from PyQt6.QtGui import QIcon
    from pathlib import Path
    paths = []
    def icon(path):
        paths.append(Path(path))
        return QIcon(path)
    monkeypatch.setattr(filters, "QIcon", icon)
    monkeypatch.setattr(filters.AnalysisFilterTab, "refresh_options", lambda self: None)
    monkeypatch.setattr(filters.AnalysisFilterTab, "refresh_mapping", lambda self: None)
    parent = QWidget()
    tab = filters.AnalysisFilterTab(parent)
    assert paths == [filters.ICONS_DIR / "filter.png", filters.ICONS_DIR / "cart.png", filters.ICONS_DIR / "save.png"]
    assert not tab.apply_button.icon().isNull() and not tab.reset_button.icon().isNull()
    assert tab.apply_button.iconSize() == tab.reset_button.iconSize()
    parent.deleteLater()
    app.processEvents()


def test_selection_errors_use_global_status_and_readiness_stays_separate(filter_tab):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    tab.types.set_checked(False)
    assert tab.window.status_label.text() == "Выберите хотя бы один вид номенклатуры"
    assert tab.message.isHidden()
    tab.types.set_checked(True)
    tab.collections.set_checked(False)
    assert tab.window.status_label.text() == "Выберите хотя бы один сезон"
    assert tab.message.isHidden()
    assert not tab.apply_button.isEnabled()
    assert all("Отбор сохранен" not in label.text() for label in tab.findChildren(QLabel))
    tab._options_loaded(None, "Для установки отбора необходимо загрузить исходные данные и справочники.")
    assert not tab.message.isHidden()
    assert not tab.start_date.isEnabled() and not tab.reset_button.isEnabled()


def change_city(tab, row, city):
    index = tab.store_table.model().index(row, 1)
    delegate = tab.city_delegate
    editor = delegate.createEditor(tab.store_table.viewport(), QStyleOptionViewItem(), index)
    assert isinstance(editor, QComboBox)
    delegate.setEditorData(editor, index)
    assert editor.itemText(0) == "Не сопоставлено" and editor.itemData(0) is None
    editor.setCurrentIndex(editor.findData(city))
    delegate.setModelData(editor, tab.store_table.model(), index)
    editor.deleteLater()


def mapping_ready(tab):
    wait_until(lambda: tab._mapping_editable)
    tab._mapping_loaded(filters.mapping.MappingSources((("A", "Магазин A"), ("B", "Магазин B")), ("Казань", "Москва")))


def test_mapping_save_reload_independent_of_filter(filter_tab, app, monkeypatch):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    mapping_ready(tab)
    stats = statistics.DatasetStatisticsTab(tab.parent())
    calls = []
    tab.filter_changed.connect(calls.append)
    before = stats.description.text(), stats.configured_filter, tab.configured_filter
    tab.saved_mapping["historical"] = "Казань"
    assert [tab.store_table.horizontalHeaderItem(i).text() for i in range(2)] == ["Магазин / канал", "Город"]
    assert not tab.save_mapping_button.isEnabled()
    for row, city in enumerate(("Москва", "Казань")):
        assert not tab.store_table.item(row, 0).flags() & Qt.ItemFlag.ItemIsEditable
        assert tab.store_table.item(row, 1).data(Qt.ItemDataRole.UserRole) is None
        assert tab.store_table.item(row, 1).text() == "Не сопоставлено"
        change_city(tab, row, city)
    assert tab.save_mapping_button.isEnabled()
    tab.save_mapping_button.click()
    assert tab.window.status_label.text() == "Распределение сохранено"
    assert tab.window._status_reset_timer.isActive()
    assert not tab.save_mapping_button.isEnabled()
    assert not calls and before == (stats.description.text(), stats.configured_filter, tab.configured_filter)
    assert stats.process is None
    assert not stats.cache_path.exists()
    assert filters.mapping.load_mapping(tab.mapping_path) == {"A": "Москва", "B": "Казань", "historical": "Казань"}
    monkeypatch.setattr(filters.AnalysisFilterTab, "refresh_options", lambda self: None)
    monkeypatch.setattr(filters.AnalysisFilterTab, "refresh_mapping", lambda self: None)
    reloaded = filters.AnalysisFilterTab(tab.parent())
    reloaded.cities = tab.cities
    reloaded._mapping_loaded(tab.mapping_sources)
    reloaded._options_loaded(OPTIONS, "")
    assert reloaded._current_mapping() == {"A": "Москва", "B": "Казань"}
    assert reloaded.store_table.rowCount() == 2
    assert not reloaded.save_mapping_button.isEnabled()


def test_mapping_save_failure_preserves_dirty_state(filter_tab, monkeypatch):
    tab = filter_tab
    mapping_ready(tab)
    change_city(tab, 0, "Казань")
    def fail(*args):
        raise OSError
    monkeypatch.setattr(filters.mapping, "save_mapping", fail)
    tab.save_mapping_button.click()
    assert tab.save_mapping_button.isEnabled()
    assert tab.window.status_label.text() == "Не удалось сохранить распределение"
    assert not tab.mapping_path.exists()



@pytest.mark.parametrize("outcome", ["success", "failure", "cancel", "failed_start"])
def test_running_reason_and_restored_validation(filter_tab, monkeypatch, tmp_path, outcome):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    tab.window.analysis_filter_tab = tab
    stats = statistics.DatasetStatisticsTab(tab.window)
    change_city(tab, 0, "Казань")
    class Process(QProcess):
        def start(self, program, arguments):
            if outcome == "failed_start":
                super().start(str(tmp_path / "missing-executable"), [])
            else:
                super().start(program, ["-c", "import time; time.sleep(20)" if outcome == "cancel" else "raise SystemExit(1)"])
    if outcome == "success":
        canonical_process(monkeypatch, tmp_path)
    else:
        monkeypatch.setattr(statistics, "QProcess", Process)
    transitions = []
    stats.calculation_running_changed.connect(
        lambda running: transitions.append((running, tab.message.text(), tab.controls.isEnabled(), tab.store_table.isEnabled())))
    stats.start()
    assert transitions[0] == (True, "Во время расчета статистики установить отбор невозможно.", False, False)
    assert tab.message.alignment() == Qt.AlignmentFlag.AlignCenter and tab.message.font().italic()
    assert not tab.controls.isEnabled() and not tab.apply_button.isEnabled() and not tab.reset_button.isEnabled()
    assert not tab.store_table.isEnabled() and not tab.save_mapping_button.isEnabled()
    tab.reset()
    tab.apply()
    tab.save_store_mapping()
    assert not tab.settings_path.exists()
    assert not tab.mapping_path.exists()
    if outcome == "cancel":
        stats.cancel()
    wait_until(lambda: stats.process is None)
    assert transitions[-1][0] is False
    wait_until(lambda: tab._mapping_editable)
    assert tab.message.isHidden() and tab.controls.isEnabled()
    assert tab.apply_button.isEnabled()
    assert tab.store_table.isEnabled() and tab.save_mapping_button.isEnabled()


@pytest.mark.parametrize("dark", [False, True])
def test_two_columns_spacing_mapping_scroll(filter_tab, app, monkeypatch, dark):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    mapping_ready(tab)
    sources = filters.mapping.MappingSources(tuple((str(i), f"Магазин {i}") for i in range(50)), ("Москва",))
    monkeypatch.setattr(filters.mapping, "load_stores", lambda: sources.stores)
    tab._mapping_loaded(sources)
    apply_app_theme(app, dark)
    monkeypatch.setattr(tab, "refresh_options", lambda **kwargs: None)
    tab.setParent(None)
    tab.resize(1600, 850)
    tab.show()
    app.processEvents()
    assert tab.date_group.parentWidget() is tab.product_group.parentWidget() is tab.controls
    assert tab.date_group.geometry().right() < tab.product_group.geometry().left()
    start = tab.start_date.mapTo(tab.controls, QPoint(0, 0))
    end = tab.end_date.mapTo(tab.controls, QPoint(0, 0))
    types = tab.types.mapTo(tab.controls, QPoint(0, 0))
    collections = tab.collections.mapTo(tab.controls, QPoint(0, 0))
    assert start.x() < types.x()
    # Allow only small rounding differences in Qt font/style metrics.
    assert abs(start.y() - types.y()) <= 2
    assert abs(end.y() - collections.y()) <= 2
    assert start.y() < end.y() and types.y() < collections.y()
    left = tab.controls.parentWidget().layout()
    assert left.itemAt(2).widget() is tab.controls
    assert left.itemAt(3).layout().itemAt(0).widget() is tab.apply_button
    assert left.spacing() == statistics.BLOCK_SPACING
    assert tab.apply_button.y() - (tab.controls.y() + tab.controls.height()) <= statistics.BLOCK_SPACING
    assert tab.store_table.verticalScrollBar().maximum() > 0
    assert tab.store_table.height() <= tab.store_table.maximumHeight()
    assert not tab.store_table.horizontalHeader().isHidden()
    assert "Распределение магазинов по городам" in [label.text() for label in tab.findChildren(QLabel)]
    tab.close()
    tab.deleteLater()


def assert_work_disabled(tab):
    widgets = (tab.start_date, tab.end_date, tab.types, tab.collections, tab.apply_button,
               tab.reset_button, tab.store_table, tab.save_mapping_button)
    assert all(not widget.isEnabled() for widget in widgets)
    assert all(tab.store_table.cellWidget(row, 1) is None for row in range(tab.store_table.rowCount()))
    assert not tab.message.isHidden()
    assert tab.message.alignment() == Qt.AlignmentFlag.AlignCenter and tab.message.font().italic()


@pytest.mark.parametrize("missing", [None, "actions", "orders", "customer_merges", "customers",
                                     "nomenclature.csv", "site_categories.csv", "city_coordinates.csv"])
def test_complete_prerequisite_gate_uses_real_sources(filter_tab, dataset, monkeypatch, missing):
    tab = filter_tab
    root, catalog, data = dataset
    cities = root / "city_coordinates.csv"
    cities.write_text("Город|Широта|Долгота\nМосква|55|37\n", encoding="utf-8-sig")
    if missing in ("actions", "orders", "customer_merges"):
        entry = data[missing] if missing == "customer_merges" else next(iter(data[missing].values()))
        # Keep the manifest intact: readiness must check the physical source too.
        (root / entry["directory"] / f"{missing}_part_001.json").unlink()
    elif missing == "customers":
        (root / "canonical/customers.sqlite").unlink()
    elif missing:
        (root / missing).unlink()
    monkeypatch.setattr(filters, "load_analysis_options", lambda: real_load_options(raw_root=root, catalog_path=catalog))
    monkeypatch.setattr(filters.mapping, "load_stores", lambda: real_stores(root))
    monkeypatch.setattr(filters.mapping, "load_cities", lambda: real_cities(cities))
    tab.refresh_options(force=True)
    wait_until(lambda: tab._loader is None)
    if missing is None:
        assert tab.all_required_sources_ready
        assert tab.controls.isEnabled()
        wait_until(lambda: tab.store_table.isEnabled())
        assert tab.apply_button.isEnabled() and tab.reset_button.isEnabled()
        wait_until(lambda: tab._mapping_editable)
        change_city(tab, 0, "Москва")
        assert tab.save_mapping_button.isEnabled()
        assert tab.message.isHidden()
    else:
        assert not tab.all_required_sources_ready
        assert_work_disabled(tab)
        assert tab.message.text() == "Для установки отбора необходимо загрузить исходные данные и справочники."
        writes = []
        monkeypatch.setattr(filters, "save_filter", lambda *args: writes.append("filter"))
        monkeypatch.setattr(filters.mapping, "save_mapping", lambda *args: writes.append("mapping"))
        tab.apply()
        tab.reset()
        tab.save_store_mapping()
        assert not tab._save(AnalysisFilter())
        assert not writes
        assert tab.findChild(QFrame, "vSeparator") is not None
        assert "В разработке" in [label.text() for label in tab.findChildren(QLabel)]


def test_running_has_priority_and_refresh_detects_lost_reference(filter_tab, monkeypatch):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    change_city(tab, 0, "Казань")
    assert tab.save_mapping_button.isEnabled()
    tab.set_statistics_running(True)
    assert_work_disabled(tab)
    assert tab.message.text() == "Во время расчета статистики установить отбор невозможно."
    monkeypatch.setattr(filters.mapping, "load_cities", lambda: None)
    tab.refresh_options(force=True)
    wait_until(lambda: tab._loader is None)
    assert_work_disabled(tab)
    assert tab.message.text() == "Во время расчета статистики установить отбор невозможно."
    tab.set_statistics_running(False)
    assert_work_disabled(tab)
    wait_until(lambda: tab._loader is None)
    assert_work_disabled(tab)
    assert tab.message.text() == "Для установки отбора необходимо загрузить исходные данные и справочники."


def test_refresh_does_not_accept_snapshot_started_before_calculation_finished(filter_tab, monkeypatch):
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    old = tab.options, tab.cities
    tab.set_statistics_running(True)
    # A pending background result predates completion and must be superseded.
    tab._loader = object()
    monkeypatch.setattr(filters.mapping, "load_cities", lambda: None)
    tab.set_statistics_running(False)
    assert tab._refresh_again
    tab._prerequisites_loaded(old, "")
    assert_work_disabled(tab)
    wait_until(lambda: tab._loader is None)
    assert_work_disabled(tab)


def test_quick_gate_does_not_wait_for_store_migration_and_coalesces(filter_tab, monkeypatch):
    from threading import Event
    tab = filter_tab
    entered, release = Event(), Event()
    calls = []
    def slow_stores():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        return (("A", "Магазин A"),)
    monkeypatch.setattr(filters.mapping, "load_stores", slow_stores)
    try:
        wait_until(lambda: entered.is_set())
        assert tab.all_required_sources_ready and tab.controls.isEnabled()
        assert tab.apply_button.isEnabled() and tab.reset_button.isEnabled()
        assert tab.message.isHidden()
        assert not tab.store_table.isEnabled() and not tab.save_mapping_button.isEnabled()
        assert tab.mapping_readiness.text() == "Формирование каталога магазинов..."
        for _ in range(3):
            tab.refresh_mapping()
            tab.refresh_options()
        assert calls == [1]
        tab.apply()
        assert tab.settings_path.exists()
    finally:
        release.set()
    wait_until(lambda: tab._mapping_editable)
    assert tab.store_table.rowCount() == 1 and tab.store_table.isEnabled()


def test_store_catalog_failure_keeps_quick_gate_usable(filter_tab, monkeypatch):
    tab = filter_tab
    def failure():
        raise OSError
    monkeypatch.setattr(filters.mapping, "load_stores", failure)
    wait_until(lambda: bool(tab._store_error))
    assert tab.controls.isEnabled() and tab.apply_button.isEnabled()
    assert tab.message.isHidden() and not tab.store_table.isEnabled()
    monkeypatch.setattr(filters.mapping, "load_stores", lambda: (("A", "A"),))
    tab.refresh_mapping()
    wait_until(lambda: tab._mapping_editable)
    assert tab.store_table.isEnabled()


def test_delegate_hundred_stores_no_persistent_editors_and_one_click(filter_tab, app, monkeypatch):
    from PyQt6.QtCore import QCoreApplication, QEvent
    from PyQt6.QtTest import QTest
    tab = filter_tab
    wait_until(lambda: tab._mapping_editable)
    monkeypatch.setattr(tab, "refresh_options", lambda **kwargs: None)
    tab._mapping_loaded(filters.mapping.MappingSources(tuple((str(i), f"Магазин {i}") for i in range(100)), ("Казань", "Москва")))
    tab.setParent(None)
    tab.resize(1280, 850)
    tab.show()
    app.processEvents()
    table = tab.store_table
    assert all(table.cellWidget(row, 1) is None for row in range(100))
    assert not table.findChildren(QComboBox)
    assert not table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable
    assert table.item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    try:
        for dark in (True, False, True):
            apply_app_theme(app, dark)
            app.processEvents()
            assert not table.findChildren(QComboBox)
            QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualItemRect(table.item(0, 0)).center())
            assert not table.findChildren(QComboBox)
            QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=table.visualItemRect(table.item(0, 1)).center())
            editors = table.findChildren(QComboBox)
            assert len(editors) == 1
            editor = editors[0]
            apply_app_theme(app, not dark)
            editor.setCurrentIndex(editor.findData("Москва"))
            editor.activated.emit(editor.currentIndex())
            app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            assert table.item(0, 1).data(Qt.ItemDataRole.UserRole) == "Москва"
            assert table.item(0, 1).text() == "Москва"
            assert not table.findChildren(QComboBox)
        assert tab.save_mapping_button.isEnabled()
        tab.save_mapping_button.click()
        assert filters.mapping.load_mapping(tab.mapping_path)["0"] == "Москва"
    finally:
        tab.close()
        tab.deleteLater()
