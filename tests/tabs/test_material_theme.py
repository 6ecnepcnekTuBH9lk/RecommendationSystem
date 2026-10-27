"""Small offline regression checks for the central theme and runtime switch."""

import os
import socket
from unittest.mock import Mock
from pathlib import Path
import importlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt, QDir, QPoint, QPointF, QEvent
from PyQt6.QtGui import QFont, QWheelEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (QApplication, QLabel, QPushButton, QSizePolicy, QGridLayout,
                            QHBoxLayout, QWidget, QVBoxLayout, QTableWidget, QStyle, QStyleOptionSlider)

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

    assert light
    assert Path(os.environ["QTMATERIAL_THEME"]).name == "light_custom.xml"

    apply_app_theme(app, True)

    assert app.styleSheet() != light
    assert Path(os.environ["QTMATERIAL_THEME"]).name == "dark_custom.xml"

    apply_app_theme(app, False)

    assert app.styleSheet() == light
    assert Path(os.environ["QTMATERIAL_THEME"]).name == "light_custom.xml"
    assert "must be imported after" not in caplog.text


def scrollbar_rect(bar):
    option = QStyleOptionSlider()
    option.initFrom(bar)
    option.orientation = bar.orientation()
    option.minimum, option.maximum = bar.minimum(), bar.maximum()
    option.sliderPosition, option.sliderValue = bar.sliderPosition(), bar.value()
    option.singleStep, option.pageStep = bar.singleStep(), bar.pageStep()
    if bar.orientation() == Qt.Orientation.Horizontal:
        option.state |= QStyle.StateFlag.State_Horizontal
    return bar.style().subControlRect(QStyle.ComplexControl.CC_ScrollBar, option,
                                     QStyle.SubControl.SC_ScrollBarSlider, bar)


@pytest.mark.parametrize('dark', [False, True])
@pytest.mark.parametrize('name', ['plain', 'datasetStatisticsTab', 'analysisFilterTab'])
@pytest.mark.parametrize('rows', [1, 400])
def test_shared_scrollbars_geometry_wheel_drag_and_hover_pressed(app, dark, name, rows):
    apply_app_theme(app, dark)
    root = QWidget()
    root.setObjectName(name)
    layout = QVBoxLayout(root)
    table = QTableWidget(rows, 8)
    layout.addWidget(table)
    for col in range(8):
        table.setColumnWidth(col, 170)
    root.resize(600, 320)
    root.show()
    app.processEvents()
    try:
        for width in (600, 760):
            root.resize(width, 320)
            app.processEvents()
            for bar in (table.verticalScrollBar(), table.horizontalScrollBar()):
                if not bar.isVisible():
                    continue
                vertical = bar.orientation() == Qt.Orientation.Vertical
                assert (bar.width() if vertical else bar.height()) == 9
                rect = scrollbar_rect(bar)
                assert (rect.height() if vertical else rect.width()) >= 36
                assert table.rect().contains(bar.mapTo(table, rect.center()))
                assert not table.viewport().geometry().intersects(bar.geometry().translated(bar.parentWidget().pos()))
        for bar in (table.verticalScrollBar(), table.horizontalScrollBar()):
            if not bar.isVisible():
                continue
            bar.setValue(bar.minimum())
            point = scrollbar_rect(bar).center()
            delta = QPoint(0, -120) if bar.orientation() == Qt.Orientation.Vertical else QPoint(-120, 0)
            wheel = QWheelEvent(QPointF(point), QPointF(bar.mapToGlobal(point)), QPoint(), delta,
                                Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
            app.sendEvent(bar, wheel)
            assert bar.value() > bar.minimum()
            bar.setValue(bar.minimum())
            # Move outside first: repeated offscreen cases can reuse the same
            # global cursor position and otherwise produce no new enter event.
            QTest.mouseMove(root, QPoint(2, 2))
            app.processEvents()
            app.sendEvent(bar, QEvent(QEvent.Type.Leave))
            point = scrollbar_rect(bar).center()
            def color():
                pixmap = bar.grab()
                ratio = pixmap.devicePixelRatio()
                return pixmap.toImage().pixelColor(round(point.x() * ratio), round(point.y() * ratio))
            idle = color()
            QTest.mouseMove(bar, point)
            app.processEvents()
            hovered = color()
            assert hovered != idle
            QTest.mousePress(bar, Qt.MouseButton.LeftButton, pos=point)
            assert bar.isSliderDown()
            pressed = color()
            assert pressed != idle and pressed != hovered
            target = point + (QPoint(0, 65) if bar.orientation() == Qt.Orientation.Vertical else QPoint(65, 0))
            QTest.mouseMove(bar, target)
            QTest.mouseRelease(bar, Qt.MouseButton.LeftButton, pos=target)
            assert not bar.isSliderDown() and bar.value() > bar.minimum()
    finally:
        root.close()
        root.deleteLater()
        app.processEvents()


def test_custom_scrollbar_style_is_global_and_has_rounded_handle_states(app):
    from Application.theme.apply_theme import CUSTOM_QSS
    text = CUSTOM_QSS.read_text()
    assert 'width: 9px' in text and 'height: 9px' in text
    assert 'border-radius: 5px' in text and 'min-height: 36px' in text and 'min-width: 36px' in text
    assert 'background: {QTMATERIAL_PRIMARYCOLOR}' in text and 'background: {QTMATERIAL_PRIMARYLIGHTCOLOR}' in text
    assert 'QScrollBar::handle:vertical:hover' in text and 'QScrollBar::handle:horizontal:pressed' in text
    assert 'QWidget#datasetStatisticsTab QScrollBar' not in text and 'QWidget#analysisFilterTab QScrollBar' not in text


@pytest.mark.usefixtures("window_settings")
def test_main_window_switch_preserves_tabs_and_customer_cards(app, monkeypatch):
    from main import MainWindow
    from Application.customer_card_data import CustomerCardResult, ProductContent, ViewState
    from Application.model.customer_identity_bridge import CustomerModelIdentityResult, CustomerModelIdentityStatus
    from Application.model.raw_bpr_inference import RawBprRecommendation
    from Application.tabs import reference_loading_section as csv_ui

    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No network in theme tests"))
    window = MainWindow()
    try:
        window.apply_theme(True)
        assert window.minimumWidth() < window.maximumWidth()
        assert window.minimumHeight() < window.maximumHeight()
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
            "Получение данных", "Статистика и анализ", "Обучение модели", "Выгрузка результатов", "Пользовательские настройки"]
        assert window.results_tab.scroll.widget() is window.results_tab.page
        assert window.results_tab.export_button.isEnabled()  # Batch ZIP export is available before customer lookup.
        assert window.mb_progress.isTextVisible()
        assert window.mb_progress.text() == "Прогресс загрузки"
        assert window.btn_load.font().weight() >= QFont.Weight.DemiBold
        csv_fields = window.btn_load.parentWidget().layout().itemAt(1).layout()
        assert isinstance(csv_fields, QGridLayout)
        assert not hasattr(window, "combo_box_types")
        for row, kind in enumerate(csv_ui.REFERENCE_TYPES):
            assert csv_fields.itemAtPosition(row, 0).widget() is window.reference_fields[kind]
            assert csv_fields.itemAtPosition(row, 1).widget() is window.reference_buttons[kind]
            assert window.reference_fields[kind].isReadOnly()
        layout = window.btn_load.parentWidget().layout()

        load_status_row = layout.itemAt(2).layout()
        assert isinstance(load_status_row, QHBoxLayout)

        assert load_status_row.itemAt(0).widget() is window.btn_load
        assert load_status_row.itemAt(1).widget() is window.status_files_container
        assert isinstance(window.status_files_layout, QHBoxLayout)
        assert window.status_files_layout.itemAt(0).widget() is window.prefix
        acquisition = window.tabs.widget(0)
        headings = [label.text() for label in acquisition.findChildren(QLabel)
                    if label.property("class") == "sectionHeader"]
        assert set(headings) == {"Загрузка через API Mindbox", "Загрузка справочников", "Ручная загрузка из Mindbox"}
        assert len(headings) == 3
        for widget in (window.heading_load_data, *window.reference_controls,
                       window.btn_load, window.status_files_container, window.prefix):
            assert acquisition.isAncestorOf(widget)
        assert [b for b in window.findChildren(QPushButton) if b.text().strip() == "Загрузить справочники"] == [window.btn_load]
        assert acquisition.isAncestorOf(window.mb_log) and acquisition.isAncestorOf(window.mb_progress)
        assert window.mb_log.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
        assert window.mb_log.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
        dialog = Mock(return_value=("", ""))
        handler = Mock(wraps=csv_ui.load_csv_file)
        monkeypatch.setattr(csv_ui.QFileDialog, "getOpenFileName", dialog)
        monkeypatch.setattr(csv_ui, "load_csv_file", handler)
        window.btn_load.click()
        handler.assert_called_once_with(window)
        dialog.assert_not_called()
        identity = CustomerModelIdentityResult("synthetic", "synthetic", CustomerModelIdentityStatus.MATCHED_EXACT, 1, "g")
        product = ProductContent(RawBprRecommendation(1, "synthetic SKU", 1.1234567), "Synthetic item")
        window.results_tab.apply_result(CustomerCardResult(ViewState.READY, "Synthetic ready", identity, "g", (product,)))
        card = window.results_tab.raw_grid.cards[0]
        photo = card.photo
        for dark in (False, True):
            QTest.mouseClick(window.theme_switch, Qt.MouseButton.LeftButton)
            assert window._current_is_dark is dark
            assert Path(os.environ["QTMATERIAL_THEME"]).name == (
                "dark_custom.xml" if dark else "light_custom.xml"
            )
            assert window.results_tab.raw_grid.cards[0] is card
            assert card.photo is photo and card.code.text() == "Код: synthetic SKU"
            assert card.content.recommendation.raw_score == 1.1234567
        assert all(label.property("class") == "sectionHeader" for label in window.mb_headings)
    finally:
        window.close()
        window.deleteLater()


def test_warmed_switch_uses_cache_and_one_global_application(app, monkeypatch):
    theme = importlib.import_module("Application.theme.apply_theme")
    apply_app_theme(app, False)
    expected = {}
    for dark in (False, True):
        apply_app_theme(app, dark)
        expected[dark] = (app.styleSheet(), {key: value for key, value in os.environ.items()
                                           if key.startswith("QTMATERIAL_")}, QDir.searchPaths("icon"))
        assert ("dark" if dark else "light") in QDir.searchPaths("icon")[0]
    def unexpected(*args, **kwargs):
        pytest.fail("A warmed switch must not rebuild, read files, reset style/font or enumerate fonts")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", unexpected)
        patch.setattr(theme, "build_stylesheet", unexpected)
        patch.setattr(theme.QFontDatabase, "families", unexpected)
        patch.setattr(app, "setFont", unexpected)
        patch.setattr(app, "setStyle", unexpected)
        apply = Mock(wraps=app.setStyleSheet)
        patch.setattr(app, "setStyleSheet", apply)
        for dark in (False, True, False):
            apply.reset_mock()
            apply_app_theme(app, dark)
            stylesheet, environment, icons = expected[dark]
            apply.assert_called_once_with(stylesheet)
            assert all(os.environ[key] == value for key, value in environment.items())
            assert QDir.searchPaths("icon") == icons
