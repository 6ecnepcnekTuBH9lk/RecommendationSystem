"""Small offline regression checks for the central theme and runtime switch."""

import os
import socket
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QTableWidgetItem, QLabel, QPushButton, QSizePolicy, QGridLayout, QHBoxLayout

from Application.theme.apply_theme import apply_app_theme


@pytest.fixture
def app():
    application = QApplication.instance() or QApplication([])
    previous_font = application.font()
    yield application
    application.setStyleSheet("")
    application.setFont(previous_font)


def test_material_light_dark_light(app, caplog):
    apply_app_theme(app, False)
    light = app.styleSheet()
    assert light and os.environ["QTMATERIAL_THEME"] == "light_purple.xml"
    apply_app_theme(app, True)
    assert app.styleSheet() != light
    assert os.environ["QTMATERIAL_THEME"] == "dark_purple.xml"
    apply_app_theme(app, False)
    assert app.styleSheet() == light
    assert "must be imported after" not in caplog.text


def test_main_window_switch_preserves_tabs_and_table_widgets(app, monkeypatch):
    from main import MainWindow
    from Application.photo.photo_processing import _set_photo_cell
    from Application.tabs import data_processing_tab as csv_ui

    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No network in theme tests"))
    window = MainWindow()
    try:
        window.apply_theme(False)
        assert window.minimumWidth() < window.maximumWidth()
        assert window.minimumHeight() < window.maximumHeight()
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
            "Получение данных", "Обработка датасета", "Обучение модели", "Выгрузка результатов"]
        assert window.purchases_table.columnCount() == 6
        assert window.recs_table.columnCount() == 7
        assert window.mb_progress.isTextVisible()
        assert window.mb_progress.text() == "0%"
        assert window.btn_load.font().weight() >= QFont.Weight.DemiBold
        csv_fields = window.btn_load.parentWidget().layout().itemAt(1).layout()
        assert isinstance(csv_fields, QGridLayout)
        assert csv_fields.itemAtPosition(0, 0).widget() is window.combo_box_types
        assert csv_fields.itemAtPosition(1, 0).widget().text() == "Полная замена справочника"
        assert csv_fields.itemAtPosition(0, 1).widget() is window.btn_load
        assert csv_fields.itemAtPosition(1, 1).widget() is window.btn_load
        assert isinstance(window.status_files_layout, QHBoxLayout)
        assert window.status_files_layout.itemAt(0).widget() is window.prefix
        acquisition, processing = window.tabs.widget(0), window.tabs.widget(1)
        headings = [label.text() for label in acquisition.findChildren(QLabel)
                    if label.property("class") == "sectionHeader"]
        assert set(headings) == {"Загрузка через API Mindbox", "Загрузка справочников", "Ручная загрузка Mindbox",
                                 "Состояние операции", "Журнал операции"}
        assert len(headings) == 5
        for widget in (window.heading_load_data, window.combo_box_types,
                       window.btn_load, window.status_files_container, window.prefix):
            assert acquisition.isAncestorOf(widget)
            assert not processing.isAncestorOf(widget)
        assert [b for b in window.findChildren(QPushButton) if b.text().strip() == "Загрузить файл"] == [window.btn_load]
        assert window.mb_log.parentWidget() is window.mb_progress.parentWidget()
        assert window.mb_log.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
        assert window.mb_log.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
        assert {label.text() for label in processing.findChildren(QLabel)
                if label.property("class") == "sectionHeader"} == {
                    "Настройки и установка отбора", "Статистика и анализ"}
        dialog = Mock(return_value=("", ""))
        handler = Mock(wraps=csv_ui.load_csv_file)
        monkeypatch.setattr(csv_ui.QFileDialog, "getOpenFileName", dialog)
        monkeypatch.setattr(csv_ui, "load_csv_file", handler)
        window.btn_load.click()
        handler.assert_called_once_with(window)
        dialog.assert_called_once()
        table = window.recs_table
        table.setRowCount(1)
        table.setItem(0, 1, QTableWidgetItem("synthetic SKU"))
        _set_photo_cell(window, table, 0, "synthetic", 0, photo_url=None)
        photo = table.cellWidget(0, 0)
        resize_modes = [table.horizontalHeader().sectionResizeMode(i) for i in range(table.columnCount())]
        table.selectRow(0)
        for dark in (True, False):
            QTest.mouseClick(window.theme_switch, Qt.MouseButton.LeftButton)
            assert window._current_is_dark is dark
            assert os.environ["QTMATERIAL_THEME"] == ("dark_purple.xml" if dark else "light_purple.xml")
            assert table.cellWidget(0, 0) is photo
            assert table.item(0, 1).text() == "synthetic SKU"
            assert table.item(0, 1).isSelected()
            assert resize_modes == [table.horizontalHeader().sectionResizeMode(i) for i in range(table.columnCount())]
        assert all(label.property("class") == "sectionHeader" for label in window.mb_headings)
    finally:
        window.close()
        window.deleteLater()
