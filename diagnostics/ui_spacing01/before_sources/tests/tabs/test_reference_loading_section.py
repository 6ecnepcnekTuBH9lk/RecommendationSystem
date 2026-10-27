"""Reference-only dispatch: GUI never parses CSV or invokes legacy processors."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from Application.paths import ICONS_DIR

from Application.tabs import reference_loading_section as ui
from Application.files.reference_import import REFERENCE_TYPES


@pytest.mark.parametrize("kind", list(REFERENCE_TYPES))
def test_reference_dispatches_to_background_controller(tmp_path, monkeypatch, kind):
    source = tmp_path / "reference.csv"
    source.write_text("synthetic", encoding="utf-8")
    controller = SimpleNamespace(start_references=Mock())
    window = SimpleNamespace(reference_paths={kind: source}, mb_controller=controller)
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    monkeypatch.setattr("pandas.read_csv", lambda *a, **kw: pytest.fail("CSV parsing in GUI"))
    ui.load_csv_file(window)
    controller.start_references.assert_called_once_with()


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("kind", list(REFERENCE_TYPES))
def test_reference_selector_preserves_filename_tooltip_and_cancel(app, tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    window = QWidget()
    section = ui.create_csv_loading_section(window)
    try:
        source = tmp_path / REFERENCE_TYPES[kind][0]
        source.write_text("synthetic", encoding="utf-8")
        monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
        window.reference_buttons[kind].click()
        assert window.reference_paths[kind] == source.resolve()
        assert window.reference_fields[kind].text() == source.name
        assert window.reference_fields[kind].toolTip() == str(source.resolve())
        assert window.reference_fields[kind].isReadOnly()
        monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *args: ("", ""))
        window.reference_buttons[kind].click()
        assert window.reference_paths[kind] == source.resolve()
        assert len(window.reference_buttons) == 3
        assert window.btn_load.text().strip() == "Загрузить справочники"
    finally:
        section.deleteLater()
        window.deleteLater()


def test_reference_status_preserves_disk_and_import_override(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "input_data"
    directory.mkdir()
    window = QWidget()
    section = ui.create_csv_loading_section(window)
    try:
        kinds = list(REFERENCE_TYPES)
        (directory / REFERENCE_TYPES[kinds[0]][0]).write_text("synthetic", encoding="utf-8")
        window.reference_status_overrides = {kinds[0]: False, kinds[1]: True}
        ui.update_file_status(window)
        assert window.prefix.text() == "Статус загрузки:"
        right = window.status_files_layout.itemAt(1).widget().layout()
        for row, filename in enumerate(("failure.png", "success.png", "failure.png")):
            icon = right.itemAt(row).widget().layout().itemAt(1).widget()
            expected = QPixmap(str(ICONS_DIR / filename)).scaled(
                17, 17, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            assert icon.pixmap().toImage() == expected.toImage()
        assert window.status_files_layout.count() == 2
    finally:
        section.deleteLater()
        window.deleteLater()


@pytest.mark.usefixtures("window_settings")
def test_main_window_has_only_current_tabs_and_reference_section(app, monkeypatch):
    from main import MainWindow
    import socket
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No network in smoke"))
    window = MainWindow()
    try:
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert titles == [
            "Получение данных", "Пользовательские настройки", "Статистика и анализ",
            "Обучение модели", "Выгрузка результатов",
        ]
        assert window.status_label.text() == "Готов к работе"
        assert window._status_reset_timer.isSingleShot()
        for attr in ("btn_weather", "btn_apply", "btn_reset", "filter_summary", "heading_filters"):
            assert not hasattr(window, attr)
        for dark in (True, False, True):
            window.apply_theme(dark)
            window.apply_static_widget_styles()
            app.processEvents()
            assert window._current_is_dark is dark
            assert [window.tabs.tabText(i) for i in range(5)] == titles
            assert window.tabs.widget(0).isAncestorOf(window.btn_load)
            assert "Загрузка справочников" in [
                label.text() for label in window.tabs.widget(0).findChildren(QLabel)
            ]
    finally:
        window.close()
        window.deleteLater()
