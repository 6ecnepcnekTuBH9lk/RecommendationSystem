"""Offline widget/orchestration tests; Mindbox processes are replaced by a signal fake."""

from dataclasses import replace
from datetime import datetime, timezone
import os
import socket
import sys
import time
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QByteArray, QObject, QProcess, QThread, QTimer, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QTabWidget, QWidget

from Application.mindbox import MindboxConfig
from Application.mindbox import daily_training_batch as backend
from Application.mindbox.training_batch import TrainingBatchExport, TrainingBatchWindow
from Application.tabs import data_loading_tab as ui
from Application.settings.set_status import set_ready_status


REAL_PROCESS = QProcess


class FakeProcess(QObject):
    ProcessChannelMode = REAL_PROCESS.ProcessChannelMode
    ProcessError = REAL_PROCESS.ProcessError
    ExitStatus = REAL_PROCESS.ExitStatus
    ProcessState = REAL_PROCESS.ProcessState
    readyReadStandardOutput = pyqtSignal()
    errorOccurred = pyqtSignal(object)
    finished = pyqtSignal(int, object)
    instances = []

    def __init__(self, parent):
        super().__init__(parent)
        self.output = b""
        self.terminated = self.killed = False
        self.running = False
        self.__class__.instances.append(self)

    def setWorkingDirectory(self, path):
        self.cwd = path

    def setProcessChannelMode(self, mode):
        self.channel_mode = mode

    def start(self, program, arguments):
        self.program, self.arguments = program, arguments
        self.running = True

    def state(self):
        return self.ProcessState.Running if self.running else self.ProcessState.NotRunning

    def readAllStandardOutput(self):
        result, self.output = self.output, b""
        return QByteArray(result)

    def feed(self, data):
        self.output += data.encode("utf-8") if isinstance(data, str) else data
        self.readyReadStandardOutput.emit()

    def finish(self, code=0, crashed=False):
        self.running = False
        self.finished.emit(code, self.ExitStatus.CrashExit if crashed else self.ExitStatus.NormalExit)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    return application


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(5)
    assert predicate(), "Qt callback did not complete"


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Tests must not use network or load Mindbox credentials")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(MindboxConfig, "from_env", forbidden)
    monkeypatch.setattr(ui, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(ui, "QProcess", FakeProcess)
    FakeProcess.instances.clear()


@pytest.fixture
def window(app):
    widget = QWidget()
    widget.tabs = QTabWidget(widget)
    widget.tabs.addTab(QWidget(), "Legacy")
    widget.status_label, widget.status_icon = QLabel(), QLabel()
    widget._status_reset_timer = QTimer(widget)
    widget._status_reset_timer.setSingleShot(True)
    widget._status_reset_timer.timeout.connect(lambda: set_ready_status(widget))
    ui.create_data_loading_widgets_tab(widget)
    QApplication.processEvents()
    wait_until(lambda: not widget.mb_controller.tasks)
    yield widget
    controller = widget.mb_controller
    if controller.state == ui.LoadingState.RUNNING:
        controller.cancel()
        process = widget.mb_process
        if isinstance(process, FakeProcess):
            process.finish(1, True)
    wait_until(lambda: not controller.tasks and widget.mb_process is None)
    widget.close()
    widget.deleteLater()


def synthetic_batch(root, statuses=None, complete=True):
    since = datetime(2026, 8, 1, tzinfo=timezone.utc)
    until = datetime(2026, 8, 3, tzinfo=timezone.utc)
    window = TrainingBatchWindow(since, until, since.replace(year=2025))
    identities = [("customer_merges", window.merge_since, until)]
    identities += [(name, day.since, day.until) for day in backend.split_daily_windows(since, until)
                   for name in ("actions", "orders")]
    statuses = statuses or ["READY"] * 5
    components = []
    for index, ((name, start, end), status) in enumerate(zip(identities, statuses)):
        export = None
        if status == "READY":
            relative = f"{name}/20260801_000000_{index:03d}"
            directory = root / relative
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{name}_part_001.json").write_text("{}", encoding="utf-8")
            export = TrainingBatchExport(name, "synthetic", f"Export.{name}", relative, 1)
        components.append(backend.BatchComponent(name, start, end, f"Export.{name}", status, export))
    batch = backend.ChunkedTrainingBatch("a" * 32, since, window, "b" * 64, tuple(components))
    directory = root / "training_batches" / batch.batch_id
    directory.mkdir(parents=True, exist_ok=True)
    state = directory / "state.json"
    backend._atomic_write(state, batch)
    if complete:
        backend._atomic_write(directory / "manifest.json", replace(batch, transport_complete=True))
    return state


def finish_training(window, state):
    process = window.mb_process
    process.feed(f"State: {state}\nManifest: {state.with_name('manifest.json')}\n")
    process.finish()
    wait_until(lambda: not window.mb_controller.tasks)


def test_initial_widgets_and_checkbox_group(window):
    assert window.tabs.tabText(0) == "Загрузка данных"
    assert window.tabs.tabText(1) == "Legacy"
    assert window.mb_start_button.isEnabled()
    assert not window.mb_cancel_button.isEnabled()
    assert not window.mb_load_customers_checkbox.isChecked()
    for changed in window.mb_training_checkboxes:
        changed.setChecked(False)
        assert not any(c.isChecked() for c in window.mb_training_checkboxes)
        changed.setChecked(True)
        assert all(c.isChecked() for c in window.mb_training_checkboxes)
    assert window.mb_interaction_since.displayFormat() == "dd.MM.yyyy"
    assert window.mb_interaction_until.calendarPopup()


def test_no_source_never_starts_process(window):
    window.mb_load_orders_checkbox.setChecked(False)
    assert not window.mb_start_button.isEnabled()
    window.mb_controller.start()
    assert not FakeProcess.instances
    assert "Выберите хотя бы один" in window.status_label.text()


@pytest.mark.parametrize("field", ["since", "merge"])
def test_invalid_dates_never_start_process(window, field):
    if field == "since":
        window.mb_interaction_since.setDate(window.mb_interaction_until.date())
    else:
        window.mb_merge_since.setDate(window.mb_interaction_since.date().addDays(1))
    window.mb_controller.start()
    assert not FakeProcess.instances
    assert window.mb_start_button.isEnabled()
    assert window.mb_controller.state == ui.LoadingState.IDLE


def test_training_command_and_only_loading_controls_locked(window):
    window.mb_controller.start()
    process = window.mb_process
    assert process.program == sys.executable
    assert process.cwd == str(ui.PROJECT_ROOT)
    assert process.channel_mode == QProcess.ProcessChannelMode.MergedChannels
    args = process.arguments
    assert args[:3] == ["-u", "-X", "utf8"]
    assert args[4] == "export-daily"
    assert args[args.index("--since") + 1] == window.mb_interaction_since.date().toString("yyyy-MM-dd")
    assert args[args.index("--merge-since") + 1] == "2025-01-01 00:00"
    assert window.tabs.isEnabled()
    assert not window.mb_start_button.isEnabled()
    assert not window.mb_interaction_since.isEnabled()
    assert not window.mb_load_customers_checkbox.isEnabled()
    assert window.mb_cancel_button.isEnabled()
    window.mb_controller.start()  # Repeated click cannot start a concurrent writer.
    assert len(FakeProcess.instances) == 1


def test_customers_only_requires_explicit_manifest(window, tmp_path):
    synthetic_batch(tmp_path)
    window.mb_controller.refresh_manifests()
    wait_until(lambda: not window.mb_controller.tasks)
    window.mb_load_actions_checkbox.setChecked(False)
    window.mb_load_customers_checkbox.setChecked(True)
    assert window.mb_manifest_combo.count() == 2
    assert window.mb_manifest_combo.currentData() is None
    assert not window.mb_start_button.isEnabled()
    window.mb_controller.start()
    assert not FakeProcess.instances
    assert "обучающий набор" in window.status_label.text()


def test_customers_only_ignores_dates_and_revalidates_manifest(window, tmp_path):
    state = synthetic_batch(tmp_path)
    controller = window.mb_controller
    controller.refresh_manifests()
    wait_until(lambda: not controller.tasks)
    window.mb_load_actions_checkbox.setChecked(False)
    window.mb_load_customers_checkbox.setChecked(True)
    window.mb_manifest_combo.setCurrentIndex(1)
    window.mb_interaction_since.setDate(window.mb_interaction_until.date())
    controller.start()
    assert window.mb_process is None  # Background validation precedes process creation.
    wait_until(lambda: window.mb_process is not None)
    args = window.mb_process.arguments
    assert args[4:] == ["export", "--training-manifest", str(state.with_name("manifest.json"))]
    assert "--since" not in args and "--until" not in args and "--merge-since" not in args
    assert window.mb_current_state_path is None


def test_manifest_disappearing_after_selection_is_rejected(window, tmp_path):
    state = synthetic_batch(tmp_path)
    controller = window.mb_controller
    controller.refresh_manifests()
    wait_until(lambda: not controller.tasks)
    window.mb_load_actions_checkbox.setChecked(False)
    window.mb_load_customers_checkbox.setChecked(True)
    window.mb_manifest_combo.setCurrentIndex(1)
    state.with_name("manifest.json").unlink()
    controller.start()
    wait_until(lambda: controller.state == ui.LoadingState.FAILED)
    assert not FakeProcess.instances


def test_catalog_filters_invalid_and_incomplete_manifests(tmp_path):
    state = synthetic_batch(tmp_path)
    broken = tmp_path / "training_batches" / ("c" * 32) / "manifest.json"
    broken.parent.mkdir()
    broken.write_text("{invalid", encoding="utf-8")
    result = ui._scan_manifests()
    assert [m["path"] for m in result["manifests"]] == [str(state.with_name("manifest.json"))]
    assert result["invalid"] == 1
    state.with_name("manifest.json").write_bytes(state.read_bytes())
    assert ui._scan_manifests()["manifests"] == []


def test_state_ready_pending_failed_counts(window, tmp_path):
    state = synthetic_batch(tmp_path, ["READY", "READY", "FAILED", "PENDING", "PENDING"], False)
    window.mb_controller.start()
    window.mb_process.feed(f"State: {state}\n")
    wait_until(lambda: not window.mb_controller.tasks)
    assert window.mb_merges_status_label.text() == "READY"
    assert window.mb_actions_status_label.text() == "1 / 2 дней"
    assert window.mb_orders_status_label.text() == "0 / 2 дней; FAILED: 1"
    assert "PENDING: 1" in window.mb_orders_status_label.toolTip()
    assert window.mb_progress.value() == 2
    assert window.mb_progress.maximum() == 5


@pytest.mark.parametrize("content", [None, "{incomplete"])
def test_state_missing_or_being_replaced_is_retried(window, tmp_path, content):
    state = tmp_path / "state.json"
    if content is not None:
        state.write_text(content, encoding="utf-8")
    controller = window.mb_controller
    controller.start()
    window.mb_process.feed(f"State: {state}\n")
    wait_until(lambda: not controller.tasks)
    assert controller.state == ui.LoadingState.RUNNING
    assert window.mb_state_timer.isActive()
    assert window.mb_progress.value() == 0


def test_utf8_split_output_html_and_unterminated_last_line(window, tmp_path):
    window.mb_controller.start()
    state = tmp_path / "кириллица" / "state.json"
    output = f"<b>обычный текст</b>\nBatch: id; State: {state}".encode("utf-8")
    for byte in output:
        window.mb_process.feed(bytes([byte]))
    window.mb_process.finish(1)
    assert window.mb_current_state_path == state
    assert "<b>обычный текст</b>" in window.mb_log.toPlainText()
    assert "�" not in window.mb_log.toPlainText()


@pytest.mark.parametrize("kind", ["nonzero", "crash", "failed_to_start"])
def test_process_failure_restores_ui(window, kind):
    controller = window.mb_controller
    controller.start()
    process = window.mb_process
    if kind == "failed_to_start":
        process.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    else:
        process.finish(1, kind == "crash")
    assert controller.state == ui.LoadingState.FAILED
    assert window.mb_start_button.isEnabled()
    assert not window.mb_cancel_button.isEnabled()
    assert window.mb_load_actions_checkbox.isEnabled()
    assert not window.mb_state_timer.isActive()
    process.finished.emit(1, QProcess.ExitStatus.CrashExit)  # Late duplicate signal is harmless.
    assert controller.state == ui.LoadingState.FAILED


def test_success_requires_verified_final_manifest(window, tmp_path):
    state = synthetic_batch(tmp_path)
    window.mb_controller.start()
    finish_training(window, state)
    assert window.mb_controller.state == ui.LoadingState.SUCCESS
    assert window.mb_manifest_label.text() == "Готов"
    assert window.mb_manifest_label.toolTip() == str(state.with_name("manifest.json"))
    assert window.mb_progress.value() == 5
    assert window.mb_start_button.isEnabled()


def test_exit_zero_without_manifest_is_failure(window):
    window.mb_controller.start()
    window.mb_process.finish()
    assert window.mb_controller.state == ui.LoadingState.FAILED


@pytest.mark.parametrize("failed", [True, False])
def test_customers_never_start_after_failed_or_invalid_batch(window, tmp_path, failed):
    state = synthetic_batch(tmp_path, ["PENDING"] * 5, False)
    window.mb_load_customers_checkbox.setChecked(True)
    window.mb_controller.start()
    window.mb_process.feed(f"State: {state}\nManifest: {state.with_name('manifest.json')}\n")
    window.mb_process.finish(1 if failed else 0)
    wait_until(lambda: not window.mb_controller.tasks)
    assert len(FakeProcess.instances) == 1
    assert window.mb_controller.state == ui.LoadingState.FAILED


def test_customers_chained_only_after_validation_and_snapshot_required(window, tmp_path, monkeypatch):
    state = synthetic_batch(tmp_path)
    window.mb_load_customers_checkbox.setChecked(True)
    controller = window.mb_controller
    controller.start()
    training_process = window.mb_process
    training_process.feed(f"State: {state}\nManifest: {state.with_name('manifest.json')}\n")
    assert len(FakeProcess.instances) == 1
    training_process.finish()
    assert len(FakeProcess.instances) == 1
    wait_until(lambda: len(FakeProcess.instances) == 2)
    assert controller.state == ui.LoadingState.RUNNING
    assert window.mb_process.arguments == ui.customers_arguments(state.with_name("manifest.json"))
    snapshot = tmp_path / "customer_profile_snapshots" / ("d" * 32) / "manifest.json"
    checked = []
    monkeypatch.setattr(ui, "_validate_snapshot", lambda path, manifest: checked.append((path, manifest)) or str(path))
    window.mb_process.feed(f"Manifest: {snapshot}\n")
    window.mb_process.finish()
    wait_until(lambda: controller.state == ui.LoadingState.SUCCESS)
    assert checked == [(snapshot, state.with_name("manifest.json"))]
    assert window.mb_customers_status_label.text() == "READY"
    assert window.mb_manifest_label.toolTip() == str(state.with_name("manifest.json"))


def test_cancel_preserves_state_and_parts_then_resume_uses_exact_path(window, tmp_path):
    state = synthetic_batch(tmp_path, ["READY", "PENDING", "PENDING", "PENDING", "PENDING"], False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    controller = window.mb_controller
    window.mb_load_customers_checkbox.setChecked(True)
    controller.start()
    process = window.mb_process
    process.feed(f"State: {state}\n")
    controller.cancel()
    assert process.terminated and not process.killed
    assert controller.kill_timer.isActive()
    assert controller.state == ui.LoadingState.RUNNING
    controller.kill_timer.timeout.emit()
    assert process.killed
    process.finish(1, True)
    wait_until(lambda: not controller.tasks)
    assert controller.state == ui.LoadingState.CANCELLED
    assert not window.mb_resume_block.isHidden()
    assert not controller.kill_timer.isActive()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.json")} == before
    controller.resume()
    assert window.mb_process.arguments == ui.resume_arguments(state)
    assert controller.want_customers is True
    assert len(FakeProcess.instances) == 2
    # A late event from the old process must not finish the resumed job.
    controller._process_finished(process, 1, QProcess.ExitStatus.CrashExit)
    assert controller.state == ui.LoadingState.RUNNING


def test_cancel_during_validation_never_starts_customers(window, tmp_path):
    state = synthetic_batch(tmp_path)
    window.mb_load_customers_checkbox.setChecked(True)
    window.mb_controller.start()
    window.mb_process.feed(f"State: {state}\nManifest: {state.with_name('manifest.json')}\n")
    window.mb_process.finish()
    window.mb_controller.cancel()
    wait_until(lambda: not window.mb_controller.tasks)
    assert window.mb_controller.state == ui.LoadingState.CANCELLED
    assert len(FakeProcess.instances) == 1


def test_late_metadata_from_old_job_cannot_change_current_job(window, tmp_path):
    state = synthetic_batch(tmp_path)
    controller = window.mb_controller
    controller.start()
    controller._on_metadata((999, controller.generation - 1, "training_manifest",
                             ui._load_training_summary(state.with_name("manifest.json"), True), None))
    assert controller.state == ui.LoadingState.RUNNING
    assert window.mb_manifest_label.text() == "—"


def test_close_waits_asynchronously_for_process(window):
    window.show()
    controller = window.mb_controller
    controller.start()
    window.close()
    assert controller.closing and window.mb_process.terminated
    assert window.isVisible()
    window.mb_process.finish(1, True)
    QApplication.processEvents()
    assert not window.isVisible()


def test_all_four_tabs_themes_and_legacy_controls(app):
    from main import MainWindow
    window = MainWindow()
    QApplication.processEvents()
    wait_until(lambda: not window.mb_controller.tasks)
    assert [window.tabs.tabText(i) for i in range(4)] == [
        "Загрузка данных", "Обработка датасета", "Обучение модели", "Выгрузка результатов"]
    assert window.btn_load is not None and window.start_train is not None
    window.mb_controller.start()
    for dark in (False, True, False):
        window.apply_theme(dark)
        window.tabs.setCurrentIndex(2)
        assert window.tabs.currentIndex() == 2
        assert window.theme_switch.isEnabled()
    window.mb_controller.cancel()
    window.mb_process.finish(1, True)
    window.close()
    window.deleteLater()


def test_real_qprocess_with_harmless_local_script_keeps_event_loop_alive(window, tmp_path, monkeypatch):
    state = synthetic_batch(tmp_path)
    monkeypatch.setattr(ui, "QProcess", REAL_PROCESS)
    script = ("import time; "
              f"print({('State: ' + str(state))!r}, flush=True); "
              "time.sleep(0.2); "
              f"print({('Manifest: ' + str(state.with_name('manifest.json')))!r}, flush=True)")
    ticks = []
    timer = QTimer(window)
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    window.mb_controller._begin()
    window.mb_controller._launch("training", ["-u", "-X", "utf8", "-c", script])
    wait_until(lambda: window.mb_controller.state == ui.LoadingState.SUCCESS)
    timer.stop()
    assert len(ticks) >= 5
    assert window.mb_process is None


def test_snapshot_validation_checks_origin(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from Application.mindbox import customer_profile_snapshot as snapshots
    monkeypatch.setattr(snapshots, "load_customer_profile_snapshot", lambda *a, **kw:
                        SimpleNamespace(originating_training_batch_id="wrong"))
    with pytest.raises(ValueError, match="another training batch"):
        ui._validate_snapshot(tmp_path / "snapshot.json", tmp_path / "batch" / "manifest.json")


@pytest.mark.parametrize("exit_code", [0, 1])
def test_customer_failure_preserves_completed_training_manifest(window, tmp_path, exit_code):
    state = synthetic_batch(tmp_path)
    window.mb_load_customers_checkbox.setChecked(True)
    controller = window.mb_controller
    controller.start()
    finish_training(window, state)
    process = window.mb_process
    assert controller.stage == "customers"
    # Even a successful exit with a missing snapshot must not report success.
    process.feed(f"Manifest: {tmp_path / 'missing-snapshot.json'}\n")
    process.finish(exit_code)
    wait_until(lambda: controller.state == ui.LoadingState.FAILED)
    assert window.mb_customers_status_label.text() == "FAILED"
    assert window.mb_manifest_label.text() == "Готов"
    assert window.mb_start_button.isEnabled()
    assert controller.resume_path is None
    assert state.with_name("manifest.json").exists()


def test_metadata_work_is_off_gui_thread_and_state_recovers(window, tmp_path, monkeypatch):
    state = synthetic_batch(tmp_path, ["PENDING"] * 5, False)
    valid_bytes = state.read_bytes()
    state.write_text("{incomplete", encoding="utf-8")
    threads = []
    original = ui._load_training_summary

    def load(*args):
        threads.append(QThread.currentThread() == QApplication.instance().thread())
        return original(*args)

    monkeypatch.setattr(ui, "_load_training_summary", load)
    controller = window.mb_controller
    controller.start()
    window.mb_process.feed(f"State: {state}\n")
    wait_until(lambda: not controller.tasks)
    assert window.mb_merges_status_label.text() == "—"
    state.write_bytes(valid_bytes)
    window.mb_state_timer.timeout.emit()
    wait_until(lambda: not controller.tasks)
    assert window.mb_merges_status_label.text() == "PENDING"
    assert threads == [False, False]


@pytest.mark.parametrize("training,customers,selected,enabled", [
    (True, False, False, True),
    (False, True, True, True),
    (False, True, False, False),
    (False, False, False, False),
    (False, False, True, False),
])
def test_start_button_source_matrix(window, tmp_path, training, customers, selected, enabled):
    synthetic_batch(tmp_path)
    controller = window.mb_controller
    controller.refresh_manifests()
    wait_until(lambda: not controller.tasks)
    window.mb_load_actions_checkbox.setChecked(training)
    window.mb_load_customers_checkbox.setChecked(customers)
    window.mb_manifest_combo.setCurrentIndex(1 if selected else 0)
    assert window.mb_start_button.isEnabled() is enabled


@pytest.mark.parametrize("validation", [False, True], ids=["export", "validation"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["failed", "cancelled"])
def test_resume_only_for_interrupted_transport(window, tmp_path, validation, cancelled):
    state = synthetic_batch(tmp_path, ["PENDING"] * 5, False)
    controller = window.mb_controller
    controller.start()
    window.mb_process.feed(f"State: {state}\n")
    if validation:
        # Exercise a resumed job too: its old resume_path must be cleared after export.
        window.mb_process.finish(1)
        wait_until(lambda: not controller.tasks)
        controller.resume()
        synthetic_batch(tmp_path)
        if not cancelled:
            state.with_name("manifest.json").write_text("{invalid", encoding="utf-8")
        window.mb_process.feed(f"Manifest: {state.with_name('manifest.json')}\n")
        window.mb_process.finish()
        assert controller.stage == "training_validation"
    if cancelled:
        controller.cancel()
    if not validation:
        window.mb_process.finish(1)
    wait_until(lambda: not controller.tasks)
    assert controller.state == (ui.LoadingState.CANCELLED if cancelled else ui.LoadingState.FAILED)
    assert state.exists()
    assert window.mb_resume_block.isHidden() is validation
    assert window.mb_resume_button.isEnabled() is not validation
    assert controller.resume_path == (None if validation else state)
    if validation:
        count = len(FakeProcess.instances)
        controller.resume()
        assert len(FakeProcess.instances) == count
        if not cancelled:
            assert "Завершённый обучающий набор не прошёл проверку" in window.status_label.text()


@pytest.mark.parametrize("resume", [False, True], ids=["new_export", "resume"])
def test_new_job_resets_component_progress_format(window, tmp_path, resume):
    state = synthetic_batch(tmp_path, ["READY", "PENDING", "PENDING", "PENDING", "PENDING"], False)
    controller = window.mb_controller
    controller.start()
    window.mb_process.feed(f"State: {state}\n")
    wait_until(lambda: not controller.tasks)
    assert window.mb_progress.format() == "%v / %m компонентов"
    window.mb_process.finish(1)
    wait_until(lambda: not controller.tasks)
    (controller.resume if resume else controller.start)()
    assert window.mb_progress.format() == "%p%"
    assert (window.mb_progress.minimum(), window.mb_progress.maximum(), window.mb_progress.value()) == (0, 100, 0)


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_global_status_lifecycle_and_scheduled_reset(window, tmp_path, monkeypatch, outcome):
    spies = {}
    for name in ("set_status_processing", "set_status_ok", "set_status_error", "schedule_status_reset"):
        spy = Mock(wraps=getattr(ui, name))
        monkeypatch.setattr(ui, name, spy)
        spies[name] = spy
    state = synthetic_batch(tmp_path)
    controller = window.mb_controller
    controller.start()
    spies["set_status_processing"].assert_called_once_with(window, "Идёт получение данных из Mindbox...")
    assert window.status_label.text() == "Идёт получение данных из Mindbox..."
    if outcome == "success":
        finish_training(window, state)
        status_function, message = "set_status_ok", "Получение данных завершено."
    else:
        if outcome == "cancel":
            controller.cancel()
        window.mb_process.finish(1)
        status_function = "set_status_ok" if outcome == "cancel" else "set_status_error"
        message = "Получение данных отменено" if outcome == "cancel" else "Ошибка получения данных. Подробности — в журнале операции."
    spies[status_function].assert_called_once_with(window, message)
    assert window.status_label.text() == message
    spies["schedule_status_reset"].assert_called_once_with(window, 5)
    assert window._status_reset_timer.isActive()
    assert window._status_reset_timer.interval() == 5000
    window._status_reset_timer.start(1)
    wait_until(lambda: not window._status_reset_timer.isActive())
    assert window.status_label.text() == "Готов к работе"


@pytest.mark.parametrize("resume", [False, True], ids=["new_export", "resume"])
def test_old_status_reset_does_not_clear_new_running_job(window, tmp_path, resume):
    state = synthetic_batch(tmp_path, ["PENDING"] * 5, False)
    controller = window.mb_controller
    controller.start()
    window.mb_process.feed(f"State: {state}\n")
    window.mb_process.finish(1)
    wait_until(lambda: not controller.tasks)
    assert window._status_reset_timer.isActive()
    window._status_reset_timer.start(20)
    (controller.resume if resume else controller.start)()
    assert not window._status_reset_timer.isActive()
    QTest.qWait(60)
    assert controller.state == ui.LoadingState.RUNNING
    assert window.status_label.text() == "Идёт получение данных из Mindbox..."
