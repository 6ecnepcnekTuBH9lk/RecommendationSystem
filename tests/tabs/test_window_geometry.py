"""Offline window placement checks with isolated QSettings."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QByteArray, QRect
from PyQt6.QtWidgets import QApplication

import main


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance() or QApplication([])
    return application


@pytest.mark.parametrize("available", [QRect(0, 0, 1920, 1040), QRect(1920, 0, 1024, 700)])
@pytest.mark.parametrize("cursor_screen", [False, True])
def test_startup_and_resize_follow_available_screen(app, monkeypatch, available, cursor_screen):
    screen = SimpleNamespace(availableGeometry=lambda: available)
    monkeypatch.setattr(main, "QGuiApplication", SimpleNamespace(
        screenAt=lambda point: screen if cursor_screen else None, primaryScreen=lambda: screen))
    window = main.MainWindow()
    try:
        app.processEvents()
        assert window.tabs.count() == 4
        assert window.minimumWidth() == min(1280, available.width() - 32)
        assert window.minimumHeight() == min(900, available.height() - 48)
        assert window.width() == max(window.minimumWidth(), min(1920, int((available.width() - 32) * 0.9)))
        assert window.height() == max(window.minimumHeight(), min(1080, int((available.height() - 48) * 0.9)))
        assert available.contains(window.frameGeometry())
        assert window.maximumWidth() > window.width()
        assert window.maximumHeight() > window.height()
        window.resize(window.width() + 100, window.height() + 50)
        assert window.width() > window.minimumWidth()
        window.resize(window.minimumSize())
        assert window.size() == window.minimumSize()
    finally:
        window.close()
        window.deleteLater()


@pytest.mark.parametrize("maximized", [False, True])
def test_previous_session_does_not_affect_startup(app, monkeypatch, window_settings, maximized):
    screen = SimpleNamespace(availableGeometry=lambda: QRect(1920, 0, 2560, 1440))
    monkeypatch.setattr(main, "QGuiApplication", SimpleNamespace(
        screenAt=lambda point: screen, primaryScreen=lambda: screen))
    first = main.MainWindow()
    first.show()
    app.processEvents()
    startup_size = first.size()
    startup_position = first.pos()
    first.move(8, 32)
    first.resize(first.minimumWidth(), first.minimumHeight())
    assert first.size() != startup_size
    if maximized:
        first.showMaximized()
        app.processEvents()
    # Simulate valid settings left by an older application version.
    saved_geometry = first.saveGeometry()
    saved_state = first.saveState()
    window_settings.setValue("window/geometry", saved_geometry)
    window_settings.setValue("window/state", saved_state)
    first.close()
    assert window_settings.value("window/geometry") == saved_geometry
    assert window_settings.value("window/state") == saved_state
    restore_geometry = Mock(side_effect=AssertionError("Must not restore previous geometry"))
    restore_state = Mock(side_effect=AssertionError("Must not restore previous state"))
    monkeypatch.setattr(main.MainWindow, "restoreGeometry", restore_geometry)
    monkeypatch.setattr(main.MainWindow, "restoreState", restore_state)
    second = main.MainWindow()
    try:
        second.show()
        app.processEvents()
        assert not second.isMaximized()
        assert second.size() == startup_size
        assert second.pos() == startup_position
        restore_geometry.assert_not_called()
        restore_state.assert_not_called()
    finally:
        second.close()
        first.deleteLater()
        second.deleteLater()


def test_close_does_not_create_window_settings(app, window_settings):
    window = main.MainWindow()
    try:
        app.processEvents()
        window.close()
        assert not window_settings.contains("window/geometry")
        assert not window_settings.contains("window/state")
    finally:
        window.deleteLater()


def test_invalid_geometry_falls_back_to_startup(app, monkeypatch, window_settings):
    window_settings.setValue("window/geometry", QByteArray(b"invalid geometry"))
    center = Mock()
    monkeypatch.setattr(main.MainWindow, "center_on_cursor_screen", center)
    window = main.MainWindow()
    try:
        app.processEvents()
        center.assert_called_once_with()
        assert window.width() >= window.minimumWidth() > 0
    finally:
        window.close()
        window.deleteLater()
