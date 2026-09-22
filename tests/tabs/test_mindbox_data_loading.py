"""Offline widget/orchestration tests; Mindbox processes are replaced by a signal fake."""

from dataclasses import replace
from datetime import datetime, timezone
import os
import json
import socket
import sys
import time
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QByteArray, QDate, QObject, QProcess, QThread, QTimer, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QTabWidget, QWidget

from Application.mindbox import MindboxConfig
from Application.mindbox import daily_training_batch as backend
from Application.mindbox.training_batch import TrainingBatchExport, TrainingBatchWindow
from Application.tabs import data_loading_tab as ui
from Application.settings.set_status import set_ready_status


REAL_PROCESS = QProcess


@pytest.mark.parametrize("stage,source,context", [
    ("manual_customers", "customers", "Ошибка импорта клиентов"),
    ("manual_interactions", "actions/orders", "Ошибка импорта действий и заказов"),
    ("customers", "customers", "Ошибка выгрузки клиентов за 09.2026"),
    ("training", "actions", "Ошибка выгрузки действий за 01.09.2026"),
    ("training", "orders", "Ошибка выгрузки заказов за 01.09.2026"),
    ("reference_csv", "reference_csv", "Ошибка импорта справочника"),
])
def test_structured_error_context_and_reason(window, stage, source, context):
    controller = window.mb_controller
    controller._begin()
    controller._launch(stage, [])
    window.mb_process.feed("Error: " + json.dumps({"category": "failed", "source": source,
        "since": "2026-09-01", "error_type": "ValueError", "message": "Не найдена обязательная колонка"}) + "\n")
    window.mb_process.finish(1)
    text = window.mb_log.toPlainText()
    assert context + ": ValueError: Не найдена обязательная колонка" in text
    assert "без дополнительного описания" not in text
    assert "Error: {" not in text and "Подробности —" not in text


@pytest.mark.parametrize("stage", ["manual_customers", "manual_interactions", "training", "customers", "reference_csv"])
@pytest.mark.parametrize("code", [0, 2])
def test_missing_protocol_result_has_specific_reason(window, stage, code):
    controller = window.mb_controller
    controller._begin()
    controller.reference_result = None
    controller._launch(stage, [])
    window.mb_process.finish(code)
    text = window.mb_log.toPlainText()
    assert ("кодом 2" if code else "процесс не передал") in text
    assert controller.error_reported
    assert controller.state == ui.LoadingState.FAILED


@pytest.mark.parametrize("kind", ["manual_preflight", "manual_result", "training_manifest", "snapshot", "persisted"])
def test_metadata_errors_keep_reason(window, kind):
    controller = window.mb_controller
    if kind != "persisted":
        controller._begin()
        controller.stage = "manual_interactions" if kind == "manual_preflight" else "training_validation"
    def fail():
        raise ValueError("Недостаточно сохранённой истории объединений")
    controller._read_metadata(kind, fail)
    wait_until(lambda: not controller.tasks)
    assert "Недостаточно сохранённой истории объединений" in window.mb_log.toPlainText()


@pytest.mark.parametrize("stage", ["training", "customers", "manual_customers", "manual_interactions"])
def test_successful_subprocess_with_failed_validation_reports_reason(window, tmp_path, monkeypatch, stage):
    def fail(*args):
        raise ValueError("Не совпадает число сохранённых частей")
    monkeypatch.setattr(ui, "_load_training_summary", fail)
    monkeypatch.setattr(ui, "_validate_manual_result", fail)
    controller = window.mb_controller
    controller._begin()
    controller._launch(stage, [])
    window.mb_process.feed(f"Manifest: {tmp_path / 'synthetic.json'}\n")
    window.mb_process.finish(0)
    wait_until(lambda: not controller.tasks)
    assert controller.state == ui.LoadingState.FAILED
    assert "Ошибка проверки сохранённого результата: ValueError: Не совпадает число сохранённых частей" in window.mb_log.toPlainText()


def test_polling_error_is_silent_and_next_poll_succeeds(window):
    controller = window.mb_controller
    controller.start()
    before = window.mb_log.toPlainText()
    def fail():
        raise OSError("Temporary state replacement")
    controller._read_metadata("state", fail)
    wait_until(lambda: not controller.tasks)
    assert window.mb_log.toPlainText() == before
    controller._read_metadata("state", lambda: {"sources": {"actions": {"ready": 3}}, "days_total": 7})
    wait_until(lambda: not controller.tasks)
    assert "3 из 7" in window.mb_actions_status_label.text()


def test_apply_reference_failure_reports_reason(window, monkeypatch):
    from Application.tabs import data_processing_tab as csv_ui
    def fail(*args):
        raise ValueError("Отсутствует список категорий")
    monkeypatch.setattr(csv_ui, "apply_reference_result", fail)
    controller = window.mb_controller
    controller._begin()
    controller._launch("reference_csv", [])
    window.mb_process.feed('Reference: {}\n')
    window.mb_process.finish()
    assert "Ошибка применения справочника: ValueError: Отсутствует список категорий" in window.mb_log.toPlainText()


def test_failed_start_cancel_and_error_flag_reset(window):
    controller = window.mb_controller
    controller.start()
    window.mb_process.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    assert "Проверьте доступность интерпретатора" in window.mb_log.toPlainText()
    assert controller.error_reported
    controller.start()
    assert not controller.error_reported
    controller.cancel()
    window.mb_process.feed('Error: {"message":"Ошибка после отмены"}\n')
    window.mb_process.finish(1)
    assert controller.state == ui.LoadingState.CANCELLED
    assert "Ошибка" not in window.mb_log.toPlainText()


@pytest.mark.parametrize("kind,message", [
    ("AttributeError", "'Foo' object has no attribute 'bar'"),
    ("KeyError", "'ids'"),
    ("ValueError", "Missing column 'Город'"),
    ("ValueError", "Invalid value for field 'email'"),
])
def test_structured_errors_preserve_exception_details(window, kind, message):
    controller = window.mb_controller
    controller.start_customers()
    window.mb_process.feed("Error: " + json.dumps({"message": message, "error_type": kind, "source": "customers"}) + "\n")
    window.mb_process.feed("Traceback PRIVATE\nunknown PRIVATE stdout\nState: C:/PRIVATE/state.json\nBatch: PRIVATE\nManifest: C:/PRIVATE/manifest.json\n")
    text = window.mb_log.toPlainText()
    assert kind + ": " + message in text
    assert "[значение скрыто]" not in text
    assert "PRIVATE" not in text and "Traceback" not in text


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


def test_initial_widgets_offer_only_full_download(window):
    assert window.mb_progress.text() == "Прогресс загрузки"
    assert window.mb_progress.isTextVisible()
    assert window.mb_customers_button.isEnabled()
    texts = [label.text() for label in window.tabs.widget(0).findChildren(QLabel)]
    assert not {"Статус:", "Набор:", "Манифест:", "Прогресс:", "Состояние операции"} & set(texts)
    for name in ("actions", "orders", "merges", "customers"):
        field = getattr(window, f"mb_{name}_status_label")
        assert field.isReadOnly() and field.text() == "Файлов не найдено"
    assert not window.mb_cancel_button.isEnabled()
    assert window.mb_interaction_since.displayFormat() == "dd.MM.yyyy"


@pytest.mark.parametrize("job", ["training", "customers"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_persisted_summary_restored_after_each_outcome(window, tmp_path, monkeypatch, job, outcome):
    baseline = {"since": "2026-01-01T00:00:00+00:00", "until": "2026-02-01T00:00:00+00:00",
                "updated": "2026-02-02T12:00:00+00:00"}
    saved = {name: baseline.copy() for name in ("actions", "orders", "customer_merges")}
    saved["customers"] = dict(updated=baseline["updated"], intervals=[[baseline["since"], baseline["until"]]], count=42)
    monkeypatch.setattr(ui, "_persisted_summary", lambda: saved)
    controller = window.mb_controller
    controller.refresh_persisted()
    wait_until(lambda: not controller.tasks)
    unchanged = window.mb_customers_status_label.text() if job == "training" else window.mb_orders_status_label.text()
    (controller.start if job == "training" else controller.start_customers)()
    assert (window.mb_customers_status_label.text() if job == "training" else window.mb_orders_status_label.text()) == unchanged
    process = window.mb_process
    if outcome == "success":
        monkeypatch.setattr(ui, "_load_training_summary", lambda *a: {})
        monkeypatch.setattr(ui, "_validate_manual_result", lambda *a: None)
        process.feed(f"Manifest: {tmp_path / 'synthetic-result'}\n")
        process.finish()
    else:
        if outcome == "cancel":
            controller.cancel()
        process.finish(1)
    wait_until(lambda: not controller.tasks)
    for source, data in saved.items():
        field = "merges" if source == "customer_merges" else source
        assert getattr(window, f"mb_{field}_status_label").text() == ui._summary_text(data)
    assert window.mb_progress.text() == "Прогресс загрузки"


def test_customer_cancel_resume_preserves_independent_job(window, tmp_path):
    controller = window.mb_controller
    controller.start_customers()
    state = tmp_path / "canonical/jobs" / ("a" * 32) / "state.json"
    window.mb_process.feed(f"State: {state}\n")
    process = window.mb_process
    controller.cancel()
    assert process.terminated
    process.finish(1)
    controller.resume()
    assert controller.stage == "customers"
    assert window.mb_process.arguments == ui.resume_arguments(state)


def test_customer_success_keeps_previous_interaction_resume(window, tmp_path, monkeypatch):
    controller = window.mb_controller
    previous = tmp_path / "canonical/jobs" / ("a" * 32) / "state.json"
    controller.resume_path = previous
    controller.resume_stage = "training"
    monkeypatch.setattr(ui, "_validate_manual_result", lambda *a: None)
    controller.start_customers()
    window.mb_process.feed(f"Manifest: {tmp_path / 'canonical/customers.sqlite'}\n")
    window.mb_process.finish()
    wait_until(lambda: not controller.tasks)
    assert controller.resume_path == previous
    assert controller.resume_stage == "training"


def test_journal_excludes_technical_output_and_formats_timeout(window):
    window.mb_controller.start_customers()
    window.mb_process.feed('State: C:/private/state.json\nBatch: secret-id\nTraceback secret\nhttps://signed.invalid/?token=secret\n')
    window.mb_process.feed('Error: {"category":"timeout","source":"customers","since":"2026-03-01"}\n')
    text = window.mb_log.toPlainText()
    assert "Превышено время ожидания выгрузки клиентов за 03.2026: Неизвестная ошибка." in text
    assert not any(value in text for value in ("C:/", "secret", "Traceback", "https://"))


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
    assert not window.mb_merge_since.isEnabled()
    assert window.mb_cancel_button.isEnabled()
    window.mb_controller.start()  # Repeated click cannot start a concurrent writer.
    assert len(FakeProcess.instances) == 1


def test_state_ready_pending_failed_counts(window, tmp_path):
    window.mb_controller.start()
    window.mb_process.feed('Event: {"source":"actions","ready":3,"total":14}\n')
    window.mb_process.feed('Event: {"source":"orders","ready":2,"total":14}\n')
    assert window.mb_actions_status_label.text() == "Загружено дней: 3 из 14"
    assert window.mb_orders_status_label.text() == "Загружено дней: 2 из 14"
    assert window.mb_customers_status_label.text() == "Файлов не найдено"
    assert window.mb_progress.maximum() == 0
    assert window.mb_progress.text() == "Идёт загрузка..."


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
    output = f"<b>обычный текст</b>\nState: {state}".encode("utf-8")
    for byte in output:
        window.mb_process.feed(bytes([byte]))
    window.mb_process.finish(1)
    assert window.mb_current_state_path == state
    assert "<b>обычный текст</b>" not in window.mb_log.toPlainText()
    assert str(state) not in window.mb_log.toPlainText()
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
    assert window.mb_interaction_since.isEnabled()
    assert not window.mb_state_timer.isActive()
    process.finished.emit(1, QProcess.ExitStatus.CrashExit)  # Late duplicate signal is harmless.
    assert controller.state == ui.LoadingState.FAILED


def test_success_requires_verified_final_manifest(window, tmp_path):
    state = synthetic_batch(tmp_path)
    window.mb_controller.start()
    finish_training(window, state)
    assert window.mb_controller.state == ui.LoadingState.SUCCESS
    assert len(FakeProcess.instances) == 1
    assert window.mb_process is None
    assert window.mb_progress.text() == "Прогресс загрузки"
    assert window.mb_start_button.isEnabled()


def test_exit_zero_without_manifest_is_failure(window):
    window.mb_controller.start()
    window.mb_process.finish()
    assert window.mb_controller.state == ui.LoadingState.FAILED


@pytest.mark.parametrize("failed", [True, False])
def test_customers_never_start_after_failed_or_invalid_batch(window, tmp_path, failed):
    state = synthetic_batch(tmp_path, ["PENDING"] * 5, False)
    window.mb_controller.start()
    window.mb_process.feed(f"State: {state}\nManifest: {state.with_name('manifest.json')}\n")
    window.mb_process.finish(1 if failed else 0)
    wait_until(lambda: not window.mb_controller.tasks)
    assert len(FakeProcess.instances) == 1
    assert window.mb_controller.state == ui.LoadingState.FAILED


def test_customers_run_independently_and_snapshot_required(window, tmp_path, monkeypatch):
    saved = "Обновлено: сохранённые взаимодействия"
    for name in ("actions", "orders", "merges"):
        getattr(window, f"mb_{name}_status_label").setText(saved)
    window.mb_controller.start_customers()
    process = window.mb_process
    assert process.arguments == ui.customers_arguments(
        window.mb_customers_since.date().toString("yyyy-MM-dd"), window.mb_customers_until.date().toString("yyyy-MM-dd"))
    assert not window.mb_start_button.isEnabled()
    window.mb_controller.start()
    assert len(FakeProcess.instances) == 1
    for name in ("actions", "orders", "merges"):
        assert getattr(window, f"mb_{name}_status_label").text() == saved
    checked = []
    monkeypatch.setattr(ui, "_validate_manual_result", lambda path, stage: checked.append(path))
    path = tmp_path / "canonical/customers.sqlite"
    process.feed(f"Manifest: {path}\n")
    process.finish()
    wait_until(lambda: not window.mb_controller.tasks)
    assert checked == [path]
    assert window.mb_controller.state == ui.LoadingState.SUCCESS
    assert window.mb_progress.text() == "Прогресс загрузки"


def test_cancel_preserves_state_and_parts_then_resume_uses_exact_path(window, tmp_path):
    state = synthetic_batch(tmp_path, ["READY", "PENDING", "PENDING", "PENDING", "PENDING"], False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    controller = window.mb_controller
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
    assert len(FakeProcess.instances) == 2
    # A late event from the old process must not finish the resumed job.
    controller._process_finished(process, 1, QProcess.ExitStatus.CrashExit)
    assert controller.state == ui.LoadingState.RUNNING
    synthetic_batch(tmp_path)
    finish_training(window, state)
    assert controller.state == ui.LoadingState.SUCCESS
    assert window.mb_process is None


def test_cancel_during_validation_never_starts_customers(window, tmp_path):
    state = synthetic_batch(tmp_path)
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
    assert len(FakeProcess.instances) == 1


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
        "Получение данных", "Обработка датасета", "Обучение модели", "Выгрузка результатов"]
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
    monkeypatch.setattr(ui, "customers_arguments", lambda manifest: ["-u", "-c", script])
    monkeypatch.setattr(ui, "_validate_snapshot", lambda path, manifest: str(path))
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
    original = state.with_name("manifest.json").read_bytes()
    controller = window.mb_controller
    controller.start_customers()
    process = window.mb_process
    process.feed(f"Manifest: {tmp_path / 'missing-snapshot.json'}\n")
    process.finish(exit_code)
    wait_until(lambda: not controller.tasks)
    assert controller.state == ui.LoadingState.FAILED
    assert window.mb_customers_status_label.text() == "Файлов не найдено"
    assert window.mb_progress.text() == "Прогресс загрузки"
    assert state.with_name("manifest.json").read_bytes() == original


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
    controller._poll_state()
    wait_until(lambda: not controller.tasks)
    assert window.mb_merges_status_label.text() == "Выполняется..."
    state.write_bytes(valid_bytes)
    window.mb_state_timer.timeout.emit()
    wait_until(lambda: not controller.tasks)
    assert window.mb_actions_status_label.text() == "Загружено дней: 0 из 2"
    assert threads == [False, False]


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
            assert "Операция завершилась с ошибкой" in window.status_label.text()


@pytest.mark.parametrize("resume", [False, True], ids=["new_export", "resume"])
def test_new_job_resets_component_progress_format(window, tmp_path, resume):
    state = synthetic_batch(tmp_path, ["READY", "PENDING", "PENDING", "PENDING", "PENDING"], False)
    controller = window.mb_controller
    controller.start()
    window.mb_process.feed(f"State: {state}\n")
    wait_until(lambda: not controller.tasks)
    assert window.mb_progress.format() == "Идёт загрузка..."
    window.mb_process.finish(1)
    wait_until(lambda: not controller.tasks)
    (controller.resume if resume else controller.start)()
    assert window.mb_progress.text() == "Идёт загрузка..."
    assert window.mb_progress.format() == "Идёт загрузка..."
    assert (window.mb_progress.minimum(), window.mb_progress.maximum(), window.mb_progress.value()) == (0, 0, 0)


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
        wait_until(lambda: controller.state == ui.LoadingState.SUCCESS)
        status_function, message = "set_status_ok", "Получение данных завершено."
    else:
        if outcome == "cancel":
            controller.cancel()
        window.mb_process.finish(1)
        status_function = "set_status_ok" if outcome == "cancel" else "set_status_error"
        message = "Получение данных отменено" if outcome == "cancel" else "Операция завершилась с ошибкой."
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


def test_selection_cli_and_controls(window):
    fields = window.mb_selection_fields
    fields["view_action_system_names"].setText(" ViewOne ; ViewTwo; ViewOne; ")
    fields["favorite_action_system_names"].setText("FavoriteOne")
    fields["purchase_line_statuses"].setText("CustomStatus")
    fields["action_product_namespaces"].setText("kanzlerKz")
    fields["order_product_namespaces"].setText("offline1C")
    window.mb_controller.start()
    args = window.mb_process.arguments
    for option, expected in (("--view-action", ["ViewOne", "ViewTwo"]),
                             ("--favorite-action", ["FavoriteOne"]),
                             ("--purchase-status", ["CustomStatus"]),
                             ("--action-product-namespace", ["kanzlerKz"]),
                             ("--order-product-namespace", ["offline1C"])):
        assert [args[i + 1] for i, value in enumerate(args) if value == option] == expected
    assert all(not widget.isEnabled() for widget in fields.values())
    window.mb_process.finish(1)
    assert all(widget.isEnabled() for widget in fields.values())


@pytest.mark.parametrize("problem", ["empty", "overlap", "namespace"])
def test_invalid_selection_prevents_process(window, problem):
    fields = window.mb_selection_fields
    if problem == "empty":
        fields["view_action_system_names"].setText(" ; ; ")
    elif problem == "overlap":
        fields["view_action_system_names"].setText("same")
        fields["favorite_action_system_names"].setText("same")
    else:
        fields["order_product_namespaces"].setText("unknown-secret")
    window.mb_controller.start()
    assert FakeProcess.instances == []
    assert window.mb_controller.state == ui.LoadingState.IDLE
    assert window.mb_log.toPlainText()
    assert "unknown-secret" not in window.mb_log.toPlainText()


def test_resume_ignores_edited_selection_and_cancel_unlocks(window, tmp_path):
    state = synthetic_batch(tmp_path, ["READY", "READY", "FAILED", "PENDING", "PENDING"], complete=False)
    window.mb_controller.resume_path = state
    window.mb_selection_fields["view_action_system_names"].setText("")
    window.mb_controller.resume()
    wait_until(lambda: window.mb_process is not None)
    assert window.mb_process.arguments == ui.resume_arguments(state)
    assert all(not w.isEnabled() for w in window.mb_selection_fields.values())
    window.mb_controller.cancel()
    window.mb_process.finish(1, True)
    wait_until(lambda: window.mb_process is None)
    assert all(w.isEnabled() for w in window.mb_selection_fields.values())


@pytest.mark.parametrize("kind", ["interactions", "customers"])
def test_manual_gui_shared_selection_independent_customers_and_validation(window, tmp_path, monkeypatch, kind):
    for name, editor in window.mb_manual_files.items():
        path = tmp_path / (name + ".json")
        path.write_text("{}", encoding="utf-8")
        editor.setText(str(path))
    fields = window.mb_selection_fields
    fields["view_action_system_names"].setText("CustomView")
    if kind == "customers":
        # Customers does not consume interaction dates or selection.
        fields["view_action_system_names"].setText("")
        window.mb_interaction_until.setDate(window.mb_interaction_since.date())
        window.mb_manual_until.setDate(window.mb_manual_since.date())
    checked = Mock(return_value="valid")
    monkeypatch.setattr(ui, "_validate_manual_result", checked)
    monkeypatch.setattr(ui, "_manual_preflight", lambda dates: None)
    window.mb_controller.start_manual(kind)
    wait_until(lambda: window.mb_process is not None)
    process = window.mb_process
    assert "mindbox_manual_import.py" in process.arguments[3]
    assert process.arguments[4] == kind
    assert ("--view-action" in process.arguments) is (kind == "interactions")
    assert ("--since" in process.arguments) is (kind == "interactions")
    if kind == "interactions":
        assert process.arguments[process.arguments.index("--view-action") + 1] == "CustomView"
    assert all(not widget.isEnabled() for widget in window.mb_manual_controls)
    manifest = tmp_path / "manifest.json"
    process.feed(f"Manifest: {manifest}\n")
    process.finish()
    wait_until(lambda: window.mb_controller.state == ui.LoadingState.SUCCESS)
    checked.assert_called_once_with(manifest, "manual_" + kind)
    assert len(FakeProcess.instances) == 1  # No Customers export after manual interactions.
    assert all(widget.isEnabled() for widget in window.mb_manual_controls)


def canonical_manual_files(window, root):
    from Application.mindbox import canonical_storage as store
    staged = root / "merges"
    staged.mkdir()
    (staged / "customer_merges_part_001.json").write_text('{"customerMerges": []}')
    with store.storage_lock(root):
        store.publish(root, "customer_merges", datetime(2025, 1, 1, tzinfo=timezone.utc),
                      datetime(2027, 1, 1, tzinfo=timezone.utc), staged)
    for name in ("actions", "orders"):
        path = root / (name + ".json")
        path.write_text("{}")
        window.mb_manual_files[name].setText(str(path))
    window.mb_manual_since.setDate(QDate(2026, 1, 1))
    window.mb_manual_until.setDate(QDate(2026, 7, 1))
    wait_until(lambda: not window.mb_controller.tasks)


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
@pytest.mark.parametrize("resume_stage", ["training", "customers"])
def test_manual_independent_period_runtime_summary_and_resume(window, tmp_path, monkeypatch, outcome, resume_stage):
    canonical_manual_files(window, tmp_path)
    window.mb_interaction_since.setDate(QDate(2026, 9, 1))
    window.mb_interaction_until.setDate(QDate(2026, 9, 8))
    controller = window.mb_controller
    state = tmp_path / "canonical/jobs/pending/state.json"
    controller.resume_path, controller.resume_stage = state, resume_stage
    controller.refresh_persisted()
    wait_until(lambda: not controller.tasks)
    prior = {field: getattr(window, f"mb_{field}_status_label").text() for field in ("actions", "orders", "merges", "customers")}
    monkeypatch.setattr(ui, "_validate_manual_result", lambda *args: None)
    controller.start_manual("interactions")
    wait_until(lambda: window.mb_process is not None)
    process = window.mb_process
    args = process.arguments
    assert args[args.index("--since") + 1] == "2026-01-01"
    assert args[args.index("--until") + 1] == "2026-07-01"
    assert "--merge-since" not in args
    for field in ("actions", "orders"):
        assert getattr(window, f"mb_{field}_status_label").text() == "Выполняется..."
    for field in ("merges", "customers"):
        assert getattr(window, f"mb_{field}_status_label").text() == prior[field]
    if outcome == "success":
        process.feed(f"Manifest: {tmp_path / 'canonical/training.json'}\n")
    if outcome == "cancel":
        controller.cancel()
    process.finish(0 if outcome == "success" else 1)
    wait_until(lambda: not controller.tasks)
    assert controller.state == {"success": ui.LoadingState.SUCCESS, "failure": ui.LoadingState.FAILED,
                                "cancel": ui.LoadingState.CANCELLED}[outcome]
    assert controller.resume_path == state and controller.resume_stage == resume_stage
    assert all(getattr(window, f"mb_{field}_status_label").text() == text for field, text in prior.items())


@pytest.mark.parametrize("invalid", ["missing_merges", "insufficient_merges", "period", "selection", "file"])
def test_manual_button_rejects_invalid_input_without_starting_process(window, tmp_path, invalid):
    canonical_manual_files(window, tmp_path)
    if invalid == "missing_merges":
        (tmp_path / "canonical/catalog.json").unlink()
    elif invalid == "insufficient_merges":
        window.mb_manual_since.setDate(QDate(2024, 1, 1))
    elif invalid == "period":
        window.mb_manual_until.setDate(window.mb_manual_since.date())
    elif invalid == "selection":
        window.mb_selection_fields["view_action_system_names"].setText("")
    else:
        window.mb_manual_files["orders"].clear()
    window.mb_controller.start_manual("interactions")
    wait_until(lambda: not window.mb_controller.tasks)
    assert not FakeProcess.instances
    assert window.mb_controller.state != ui.LoadingState.RUNNING
    if invalid.endswith("merges"):
        assert "API Mindbox" in window.mb_log.toPlainText()
    assert str(tmp_path) not in window.mb_log.toPlainText()


def test_manual_preflight_cancel_does_not_launch_late_process(window, tmp_path, monkeypatch):
    import threading
    canonical_manual_files(window, tmp_path)
    entered, release = threading.Event(), threading.Event()
    def preflight(dates):
        entered.set()
        release.wait(5)
    monkeypatch.setattr(ui, "_manual_preflight", preflight)
    window.mb_controller.start_manual("interactions")
    wait_until(entered.is_set)
    window.mb_controller.cancel()
    release.set()
    wait_until(lambda: not window.mb_controller.tasks)
    assert window.mb_controller.state == ui.LoadingState.CANCELLED
    assert not FakeProcess.instances


def test_reference_options_and_background_process(window, monkeypatch, tmp_path):
    from Application.tabs import data_processing_tab as csv_ui
    assert [window.combo_box_types.itemText(i) for i in range(window.combo_box_types.count())] == [
        "Номенклатура из 1С", "Категории сайта из 1С", "Координаты городов и погода"]
    assert getattr(window, "combo_box_add_or_not", None) is None
    applied = Mock()
    monkeypatch.setattr(csv_ui, "apply_reference_result", applied)
    window.mb_controller.start_reference(tmp_path / "input.csv", "Номенклатура из 1С")
    process = window.mb_process
    assert "import_reference_csv.py" in process.arguments[3]
    assert not window.btn_load.isEnabled()
    process.feed('Reference: {"kind": "synthetic"}\n')
    process.finish()
    assert window.mb_controller.state == ui.LoadingState.SUCCESS
    applied.assert_called_once_with(window, {"kind": "synthetic"})
    assert window.btn_load.isEnabled()


def test_manual_cancel_no_api_resume(window, tmp_path):
    path = tmp_path / "customers.json"
    path.write_text('{"customers": []}', encoding="utf-8")
    window.mb_manual_files["customers"].setText(str(path))
    window.mb_controller.start_manual("customers")
    window.mb_controller.cancel()
    window.mb_process.finish(1, True)
    assert window.mb_controller.state == ui.LoadingState.CANCELLED
    assert window.mb_controller.resume_path is None
    assert all(widget.isEnabled() for widget in window.mb_manual_controls)



def test_independent_manual_import_preserves_pending_api_resume(window, tmp_path):
    state = synthetic_batch(tmp_path, ["READY", "READY", "FAILED", "PENDING", "PENDING"], complete=False)
    window.mb_controller.resume_path = state
    path = tmp_path / "customers.json"
    path.write_text('{"customers": []}', encoding="utf-8")
    window.mb_manual_files["customers"].setText(str(path))
    window.mb_controller.start_manual("customers")
    window.mb_process.finish(1)
    assert window.mb_controller.resume_path == state
    window.mb_controller.resume()
    assert window.mb_process.arguments == ui.resume_arguments(state)
