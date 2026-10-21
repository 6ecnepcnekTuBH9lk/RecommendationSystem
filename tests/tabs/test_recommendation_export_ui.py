"""Qt protocol/lifecycle tests, with no production model or data access."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QObject, QProcess, pyqtSignal, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget

from Application.tabs import recommendation_export_controller as ui


class Process(QObject):
    readyReadStandardOutput = pyqtSignal()
    readyReadStandardError = pyqtSignal()
    finished = pyqtSignal(int, QProcess.ExitStatus)
    errorOccurred = pyqtSignal(QProcess.ProcessError)
    ProcessError, ExitStatus = QProcess.ProcessError, QProcess.ExitStatus

    def __init__(self, parent):
        super().__init__(parent)
        self.payload = b""
        self.writes = []

    def setWorkingDirectory(self, path):
        self.cwd = path

    def start(self, executable, args):
        self.executable, self.args = executable, args

    def write(self, data):
        self.writes.append(data)

    def readAllStandardOutput(self):
        data, self.payload = self.payload, b""
        return data

    def readAllStandardError(self):
        return b""

    def send(self, **payload):
        self.payload += (json.dumps(payload) + "\n").encode()
        self.readyReadStandardOutput.emit()

    def waitForFinished(self, timeout):
        assert timeout == -1
        self.send(type="cancelled")
        self.finished.emit(2, self.ExitStatus.NormalExit)
        return True


@pytest.fixture
def screen(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    host = QMainWindow()
    tab = QWidget(host)
    tab.window = host  # Actual customer tab uses this attribute too.
    tab.export_button = QPushButton("Экспорт", tab)
    tab.cancel_button = QPushButton("Отмена поиска", tab)
    host.setCentralWidget(tab)
    host.show()
    controller = ui.RecommendationExportController(tab)
    messages = []
    monkeypatch.setattr(ui, "QProcess", Process)
    monkeypatch.setattr(
        ui.QInputDialog,
        "getItem",
        lambda parent, title, label, modes, *a: (modes[0], True),
    )
    monkeypatch.setattr(
        ui.QFileDialog,
        "getSaveFileName",
        lambda *a, **kw: (str(tmp_path / "Recommendations.zip"), ""),
    )
    monkeypatch.setattr(ui.QMessageBox, "information", lambda *a: messages.append(a[2]))
    monkeypatch.setattr(ui.QMessageBox, "warning", lambda *a: messages.append(a[2]))
    yield controller, tab, host, messages, tmp_path
    if controller.process is not None:
        controller.process.finished.emit(2, Process.ExitStatus.NormalExit)
    host.close()
    host.deleteLater()
    controller.deleteLater()
    app.processEvents()


def test_default_trial_is_independent_and_reports_counts(screen):
    controller, tab, _, messages, _ = screen
    controller.start()
    process = controller.process
    assert process.args[-2:] == ["--limit", "50"] and "--full" not in process.args
    assert not tab.export_button.isEnabled() and tab.cancel_button.isEnabled()
    process.send(type="progress", phase="customers", processed=12, total=50)
    assert (
        controller.dialog.value() == 12 and "12 / 50" in controller.dialog.labelText()
    )
    ticks = []
    QTimer.singleShot(0, lambda: ticks.append(True))
    QTest.qWait(10)
    assert ticks  # Active child does not block the Qt event loop.
    process.send(
        type="success",
        summary=dict(
            customers=12,
            online_rows=11,
            mindbox_rows=120,
            raw_rows=120,
            hybrid_rows=120,
        ),
    )
    process.finished.emit(0, Process.ExitStatus.NormalExit)
    assert controller.process is None and tab.export_button.isEnabled()
    assert "Mindbox: 120" in messages[0]


def test_full_requires_explicit_mode_and_overwrite_consent(screen, monkeypatch):
    controller, _, _, _, directory = screen
    (directory / "Recommendations.zip").write_bytes(b"old")
    monkeypatch.setattr(
        ui.QInputDialog, "getItem", lambda p, t, l, modes, *a: (modes[1], True)
    )
    monkeypatch.setattr(
        ui.QMessageBox, "question", lambda *a: ui.QMessageBox.StandardButton.No
    )
    controller.start()
    assert controller.process is None
    monkeypatch.setattr(
        ui.QMessageBox, "question", lambda *a: ui.QMessageBox.StandardButton.Yes
    )
    controller.start()
    assert (
        "--full" in controller.process.args and "--limit" not in controller.process.args
    )
    assert "--overwrite" in controller.process.args


def test_dialog_cancel_never_starts_child(screen, monkeypatch):
    controller, tab, _, _, _ = screen
    monkeypatch.setattr(ui.QFileDialog, "getSaveFileName", lambda *a, **kw: ("", ""))
    controller.start()
    assert controller.process is None and tab.export_button.isEnabled()


def test_cancel_is_cooperative_idempotent_and_distinct_from_search(screen):
    controller, tab, _, messages, _ = screen
    controller.start()
    process = controller.process
    controller.cancel()
    controller.cancel()
    assert process.writes == [b"cancel\n"]
    assert tab.cancel_button.isEnabled() and controller.process is process
    process.send(type="cancelled")
    process.finished.emit(2, Process.ExitStatus.NormalExit)
    assert "временные файлы очищены" in messages[0]


def test_window_close_waits_for_cleanup(screen):
    controller, _, host, messages, _ = screen
    controller.start()
    process = controller.process
    host.close()
    assert host.isVisible() and controller.close_pending
    assert process.writes == [b"cancel\n"]
    process.send(type="cancelled")
    process.finished.emit(2, Process.ExitStatus.NormalExit)
    QTest.qWait(10)
    assert not host.isVisible() and not messages


def test_source_error_and_failed_start_reenable_export(screen):
    controller, tab, _, messages, _ = screen
    controller.start()
    controller.process.send(type="error", code="SOURCES_CHANGED")
    controller.process.finished.emit(1, Process.ExitStatus.NormalExit)
    assert tab.export_button.isEnabled() and "Источники изменились" in messages[0]
    controller.start()
    controller.process.errorOccurred.emit(Process.ProcessError.FailedToStart)
    assert controller.process is None and tab.export_button.isEnabled()


def test_shutdown_and_deleted_tab_retain_child_until_cooperative_exit(screen):
    controller, _, _, _, _ = screen
    controller.start()
    process = controller.process
    controller._tab_destroyed()
    assert controller.tab is None and controller.process is process
    controller._shutdown()
    assert process.writes == [b"cancel\n"] and controller.process is None
