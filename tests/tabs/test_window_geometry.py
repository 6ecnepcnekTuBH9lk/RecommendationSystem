"""Offline window placement checks with isolated QSettings."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QByteArray, QRect
from PyQt6.QtWidgets import QApplication

import main


@pytest.fixture
def app(monkeypatch):
    application = QApplication.instance() or QApplication([])
    return application


@pytest.mark.parametrize("available", [QRect(0, 0, 1920, 1040), QRect(1920, 0, 1024, 700)])
def test_startup_and_resize_follow_available_screen(app, monkeypatch, available):
    screen = SimpleNamespace(availableGeometry=lambda: available)
    monkeypatch.setattr(main, "QGuiApplication", SimpleNamespace(
        screenAt=lambda point: screen, primaryScreen=lambda: screen))
    window = main.MainWindow()
    try:
        app.processEvents()
        assert window.tabs.count() == 4
        assert window.minimumWidth() == min(1280, available.width() - 32)
        assert window.minimumHeight() == min(900, available.height() - 48)
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
def test_geometry_round_trip_does_not_recenter(app, monkeypatch, window_settings, maximized):
    first = main.MainWindow()
    first.show()
    app.processEvents()
    first.move(8, 32)
    first.resize(first.minimumWidth(), first.minimumHeight())
    expected_size = first.size()
    expected_position = first.pos()
    if maximized:
        first.showMaximized()
        app.processEvents()
    first.close()
    saved = window_settings.value("window/geometry")
    assert isinstance(saved, QByteArray) and not saved.isEmpty()
    assert not window_settings.value("window/state").isEmpty()
    center = Mock()
    monkeypatch.setattr(main.MainWindow, "center_on_cursor_screen", center)
    second = main.MainWindow()
    try:
        second.show()
        app.processEvents()
        assert second.isMaximized() is maximized
        if not maximized:
            assert second.size() == expected_size
            assert second.pos() == expected_position
        center.assert_not_called()
    finally:
        second.close()
        first.deleteLater()
        second.deleteLater()


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
