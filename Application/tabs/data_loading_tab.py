"""Mindbox loading UI. CLI owns exports; this module only orchestrates jobs."""

import codecs
from datetime import datetime
from enum import Enum
from pathlib import Path
import sys

from PyQt6.QtCore import (QDate, QEvent, QObject, QProcess, QRunnable, QThreadPool,
                         QTimer, Qt, pyqtSignal, pyqtSlot)
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDateEdit, QFrame, QHBoxLayout,
                            QLabel, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget)

from Application.settings.set_status import (set_status_error, set_status_ok,
                                             set_status_processing, schedule_status_reset)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "ВходныеДанные" / "MindboxRaw"
TRAINING_TOOLTIP = ("Действия, заказы и объединения клиентов формируют единый "
                    "обучающий набор и загружаются совместно.")


class LoadingState(Enum):
    IDLE = "Не запущено"
    RUNNING = "Выполняется"
    SUCCESS = "Завершено"
    FAILED = "Ошибка"
    CANCELLED = "Отменено"


def _heading(text, layout):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)
    return label


def apply_data_loading_theme(aboba, is_dark):
    background, border = ("#464646", "#6E6E6E") if is_dark else ("#FAFAFA", "#C8C8C8")
    for heading in aboba.mb_headings:
        heading.setStyleSheet(
            f"QLabel {{ background-color: {background}; border: 1px solid {border};"
            "padding: 7px 65px; border-radius: 10px; margin: 10px 0px; }")
    field = "#5F5F5F" if is_dark else "#F0F0F0"
    foreground = "white" if is_dark else "black"
    for widget in (aboba.mb_interaction_since, aboba.mb_interaction_until, aboba.mb_merge_since):
        widget.setStyleSheet(f"QDateEdit {{ background-color: {field}; color: {foreground};"
                            f"border: 1px solid {border}; padding: 4px 5px; border-radius: 10px; }}")
    chunk = "#7D7D7D" if is_dark else "#D2D2D2"
    aboba.mb_progress.setStyleSheet(
        f"QProgressBar {{ color: {foreground}; background-color: {field}; text-align: center;"
        f"border: 1px solid {border}; border-radius: 10px; }}"
        f"QProgressBar::chunk {{ background-color: {chunk}; border-radius: 10px; }}")


def _date_widget(date):
    widget = QDateEdit(date)
    widget.setDisplayFormat("dd.MM.yyyy")
    widget.setCalendarPopup(True)
    return widget


def training_arguments(since, until, merge_since):
    """CLI interprets these dates as UTC, with an exclusive upper boundary."""
    if since >= until:
        raise ValueError("Дата начала взаимодействий должна быть раньше даты окончания.")
    if merge_since > since:
        raise ValueError("История объединений должна начинаться не позже периода взаимодействий.")
    return ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_daily_batch.py"),
            "export-daily", "--since", since, "--until", until, "--merge-since", merge_since + " 00:00"]


def resume_arguments(state_path):
    return ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_daily_batch.py"),
            "resume", "--state", str(state_path)]


def customers_arguments(manifest):
    return ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_customer_profiles.py"),
            "export", "--training-manifest", str(manifest)]


def _load_training_summary(path, complete=False):
    # Backend performs schema, completeness, identity and raw-part validation.
    # This can touch many directories, so callers must use a metadata worker.
    from Application.mindbox.daily_training_batch import load_chunked_training_batch
    batch = load_chunked_training_batch(path, raw_root=RAW_ROOT, require_complete=complete)
    counts = batch.diagnostics
    return {
        "path": str(path), "batch_id": batch.batch_id,
        "period": f"{batch.window.interaction_since:%d.%m.%Y} — {batch.window.interaction_until:%d.%m.%Y}",
        "merges": batch.components[0].status,
        "sources": {name: {
            "ready": sum(c.status == "READY" for c in batch.components if c.name == name),
            "failed": sum(c.status == "FAILED" for c in batch.components if c.name == name),
        } for name in ("actions", "orders")},
        **counts, "components_total": len(batch.components),
    }


def _scan_manifests():
    from Application.mindbox.raw_reader import RawExportError
    manifests = []
    invalid = 0
    for path in sorted((RAW_ROOT / "training_batches").glob("*/manifest.json")):
        try:
            manifests.append(_load_training_summary(path, complete=True))
        except (OSError, ValueError, RawExportError):
            invalid += 1
    return {"manifests": manifests, "invalid": invalid}


def _validate_snapshot(path, training_manifest):
    from Application.mindbox.customer_profile_snapshot import load_customer_profile_snapshot
    snapshot = load_customer_profile_snapshot(path, raw_root=RAW_ROOT)
    if snapshot.originating_training_batch_id != Path(training_manifest).parent.name:
        raise ValueError("Snapshot references another training batch")
    return str(path)


class _MetadataSignals(QObject):
    finished = pyqtSignal(object)


class _MetadataTask(QRunnable):
    """Bounded offline work; only immutable results cross back to the GUI."""
    def __init__(self, task_id, generation, kind, function, args):
        super().__init__()
        self.signals = _MetadataSignals()
        self.task_id, self.generation, self.kind = task_id, generation, kind
        self.function, self.args = function, args

    def run(self):
        try:
            result, error = self.function(*self.args), None
        except Exception as exc:
            # Never leak record contents, credentials or arbitrary exception text.
            result, error = None, type(exc).__name__
        self.signals.finished.emit((self.task_id, self.generation, self.kind, result, error))


def create_data_loading_widgets_tab(aboba):
    tab = QWidget()
    root = QVBoxLayout(tab)
    root.setContentsMargins(0, 0, 0, 0)
    top = QHBoxLayout()
    top.setSpacing(0)
    left_wrap, right_wrap = QWidget(), QWidget()
    left, right = QVBoxLayout(left_wrap), QVBoxLayout(right_wrap)
    separator = QFrame()
    separator.setObjectName("vSeparator")
    separator.setFixedWidth(1)
    separator.setFrameShape(QFrame.Shape.NoFrame)
    top.addWidget(left_wrap, 1)
    top.addWidget(separator)
    top.addWidget(right_wrap, 1)
    root.addLayout(top, 3)
    aboba.mb_headings = [_heading("Получение данных", left), _heading("Состояние операции", right)]

    left.addWidget(QLabel("Источники данных:"))
    left.addWidget(QLabel("Данные для обучения:"))
    training = []
    for name, text in (("actions", "Действия"), ("orders", "Заказы"), ("merges", "Объединения клиентов")):
        checkbox = QCheckBox(text)
        checkbox.setChecked(True)
        checkbox.setToolTip(TRAINING_TOOLTIP)
        setattr(aboba, f"mb_load_{name}_checkbox", checkbox)
        training.append(checkbox)
        left.addWidget(checkbox)
    aboba.mb_training_checkboxes = training
    left.addWidget(QLabel("Дополнительные данные:"))
    aboba.mb_load_customers_checkbox = QCheckBox("Клиенты")
    left.addWidget(aboba.mb_load_customers_checkbox)
    left.addWidget(QLabel("Период взаимодействий:"))
    period = QHBoxLayout()
    today = QDate.currentDate()
    aboba.mb_interaction_since = _date_widget(today.addDays(-7))
    aboba.mb_interaction_until = _date_widget(today)
    for text, widget in (("С", aboba.mb_interaction_since), ("По", aboba.mb_interaction_until)):
        period.addWidget(QLabel(text))
        period.addWidget(widget, 1)
    left.addLayout(period)
    left.addWidget(QLabel("Границы суток: 00:00 UTC. Дата «по» не включается."))
    left.addWidget(QLabel("История объединений клиентов с:"))
    aboba.mb_merge_since = _date_widget(QDate(2025, 1, 1))
    left.addWidget(aboba.mb_merge_since)

    aboba.mb_manifest_picker = QWidget()
    picker = QVBoxLayout(aboba.mb_manifest_picker)
    picker.setContentsMargins(0, 0, 0, 0)
    picker.addWidget(QLabel("Обучающий набор:"))
    aboba.mb_manifest_combo = QComboBox()
    aboba.mb_manifest_combo.addItem("Выберите завершённый обучающий набор…", None)
    picker.addWidget(aboba.mb_manifest_combo)
    aboba.mb_manifest_hint = QLabel("Проверка доступных наборов…")
    picker.addWidget(aboba.mb_manifest_hint)
    aboba.mb_refresh_button = QPushButton("Обновить список наборов")
    picker.addWidget(aboba.mb_refresh_button)
    left.addWidget(aboba.mb_manifest_picker)

    buttons = QHBoxLayout()
    aboba.mb_start_button = QPushButton("Получить данные")
    aboba.mb_cancel_button = QPushButton("Отменить")
    buttons.addWidget(aboba.mb_start_button)
    buttons.addWidget(aboba.mb_cancel_button)
    left.addLayout(buttons)
    left.addStretch(1)

    for name, text in (("status", "Статус операции"), ("merges_status", "Объединения клиентов"),
                       ("actions_status", "Действия"), ("orders_status", "Заказы"),
                       ("customers_status", "Клиенты"), ("batch", "Batch"), ("manifest", "Manifest")):
        row = QHBoxLayout()
        row.addWidget(QLabel(text + ":"))
        label = QLabel("—")
        label.setWordWrap(True)
        setattr(aboba, f"mb_{name}_label", label)
        row.addWidget(label, 1)
        right.addLayout(row)
    aboba.mb_progress = QProgressBar()
    aboba.mb_progress.setValue(0)
    right.addWidget(aboba.mb_progress)
    aboba.mb_resume_block = QWidget()
    resume = QVBoxLayout(aboba.mb_resume_block)
    resume.addWidget(QLabel("Незавершённая загрузка"))
    aboba.mb_resume_summary = QLabel()
    aboba.mb_resume_summary.setWordWrap(True)
    resume.addWidget(aboba.mb_resume_summary)
    aboba.mb_resume_button = QPushButton("Продолжить")
    resume.addWidget(aboba.mb_resume_button)
    right.addWidget(aboba.mb_resume_block)
    right.addStretch(1)

    log_wrap = QWidget()
    log_layout = QVBoxLayout(log_wrap)
    aboba.mb_headings.append(_heading("Журнал операции", log_layout))
    aboba.mb_log = QTextEdit()
    aboba.mb_log.setReadOnly(True)
    aboba.mb_log.setAcceptRichText(False)
    aboba.mb_log.setPlaceholderText("Логи загрузки будут отображаться здесь...")
    aboba.mb_log.document().setMaximumBlockCount(3000)
    log_layout.addWidget(aboba.mb_log)
    root.addWidget(log_wrap, 2)
    aboba.tabs.insertTab(0, tab, "Загрузка данных")
    apply_data_loading_theme(aboba, getattr(aboba, "_current_is_dark", False))
    aboba.mb_controller = _LoadingController(aboba)


class _LoadingController(QObject):
    def __init__(self, aboba):
        super().__init__(aboba)
        self.ui = aboba
        self.state = LoadingState.IDLE
        self.resume_path = None
        self.want_customers = False
        self.cancel_requested = False
        self.closing = False
        self.generation = 0
        self.stage = None
        self.manifest_path = None
        self.snapshot_path = None
        self.tasks = {}
        self._task_number = 0
        aboba.mb_process = None
        aboba.mb_current_state_path = None
        aboba.mb_state_timer = QTimer(self)
        aboba.mb_state_timer.setInterval(1000)
        aboba.mb_state_timer.timeout.connect(self._poll_state)
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.setInterval(3000)
        self.kill_timer.timeout.connect(self._kill_process)
        for checkbox in aboba.mb_training_checkboxes:
            checkbox.toggled.connect(self._sync_training)
        aboba.mb_load_customers_checkbox.toggled.connect(self._update_controls)
        aboba.mb_start_button.clicked.connect(self.start)
        aboba.mb_cancel_button.clicked.connect(self.cancel)
        aboba.mb_resume_button.clicked.connect(self.resume)
        aboba.mb_refresh_button.clicked.connect(self.refresh_manifests)
        aboba.mb_manifest_combo.currentIndexChanged.connect(self._update_controls)
        aboba.installEventFilter(self)
        self._update_controls()
        QTimer.singleShot(0, self.refresh_manifests)

    def _sync_training(self, checked):
        for checkbox in self.ui.mb_training_checkboxes:
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        self._update_controls()

    def _update_controls(self):
        a = self.ui
        running = self.state == LoadingState.RUNNING
        training = a.mb_load_actions_checkbox.isChecked()
        customers = a.mb_load_customers_checkbox.isChecked()
        for widget in (*a.mb_training_checkboxes, a.mb_load_customers_checkbox):
            widget.setEnabled(not running)
        for widget in (a.mb_interaction_since, a.mb_interaction_until, a.mb_merge_since):
            widget.setEnabled(not running and training)
        a.mb_manifest_picker.setVisible(customers and not training)
        a.mb_manifest_picker.setEnabled(not running)
        a.mb_start_button.setEnabled(not running and
            (training or (customers and a.mb_manifest_combo.currentData() is not None)))
        a.mb_refresh_button.setEnabled(not running and not any(t.kind == "catalog" for t in self.tasks.values()))
        a.mb_cancel_button.setEnabled(running and not self.cancel_requested)
        a.mb_resume_block.setVisible(not running and self.resume_path is not None)
        a.mb_resume_button.setEnabled(not running and self.resume_path is not None)
        a.mb_status_label.setText(self.state.value)

    def _log(self, message, *, timestamp=True):
        if timestamp:
            message = f"[{datetime.now():%H:%M:%S}] {message}"
        # insertPlainText never interprets backend output as HTML.
        cursor = self.ui.mb_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(message + "\n")
        self.ui.mb_log.setTextCursor(cursor)
        self.ui.mb_log.ensureCursorVisible()

    def _error(self, message):
        self._log(message)
        set_status_error(self.ui, message)

    def _begin(self):
        self.generation += 1
        self.cancel_requested = False
        self.state = LoadingState.RUNNING
        reset_timer = getattr(self.ui, "_status_reset_timer", None)
        if reset_timer is not None:
            reset_timer.stop()
        set_status_processing(self.ui, "Идёт получение данных из Mindbox...")
        self.ui.mb_progress.setRange(0, 100)
        self.ui.mb_progress.setValue(0)
        self.ui.mb_progress.setFormat("%p%")
        self.ui.mb_log.clear()
        self._log("Запуск получения данных...")
        self._update_controls()

    def start(self):
        if self.state == LoadingState.RUNNING:
            return
        a = self.ui
        training = a.mb_load_actions_checkbox.isChecked()
        customers = a.mb_load_customers_checkbox.isChecked()
        if not training and not customers:
            self._error("Выберите хотя бы один источник данных.")
            return
        selected_manifest = a.mb_manifest_combo.currentData()
        if not training and selected_manifest is None:
            self._error("Выберите завершённый обучающий набор для загрузки клиентов.")
            return
        try:
            args = training_arguments(*(widget.date().toString("yyyy-MM-dd") for widget in
                (a.mb_interaction_since, a.mb_interaction_until, a.mb_merge_since))) if training else None
        except ValueError as exc:
            self._error(str(exc))
            return
        self.want_customers = customers
        self.resume_path = None
        a.mb_current_state_path = None
        self.manifest_path = self.snapshot_path = None
        for name in ("merges_status", "actions_status", "orders_status", "customers_status", "batch", "manifest"):
            label = getattr(a, f"mb_{name}_label")
            label.setText("—")
            label.setToolTip("")
        self._begin()
        if customers:
            a.mb_customers_status_label.setText("PENDING")
        if training:
            self._log(f"Период: {a.mb_interaction_since.text()} — {a.mb_interaction_until.text()} [00:00 UTC, по не включительно)")
            self._log("Источники: Действия, Заказы, Объединения клиентов" + (", Клиенты" if customers else ""))
            self._launch("training", args)
        else:
            self.stage = "customer_manifest_validation"
            self.manifest_path = Path(selected_manifest)
            self._log(f"Источник: Клиенты. Обучающий набор: {self.manifest_path}")
            self._read_metadata("customer_manifest", _load_training_summary, self.manifest_path, True)

    def _launch(self, stage, arguments):
        self.stage = stage
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.output_buffer = ""
        process = QProcess(self)
        self.ui.mb_process = process
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(lambda: self._read_output(process))
        process.errorOccurred.connect(lambda error: self._process_error(process, error))
        process.finished.connect(lambda code, status: self._process_finished(process, code, status))
        process.start(sys.executable, arguments)
        if stage == "training":
            self.ui.mb_state_timer.start()
        else:
            self.ui.mb_customers_status_label.setText("Выполняется")

    def _read_output(self, process, final=False):
        if process is not self.ui.mb_process:
            return
        self.output_buffer += self.decoder.decode(bytes(process.readAllStandardOutput()), final=final)
        lines = self.output_buffer.split("\n")
        self.output_buffer = lines.pop()
        if final and self.output_buffer:
            lines.append(self.output_buffer)
            self.output_buffer = ""
        for line in lines:
            self._consume_line(line.rstrip("\r"))

    def _consume_line(self, line):
        self._log(line, timestamp=False)
        if self.stage == "training":
            state_text = None
            if line.startswith("State: "):
                state_text = line[len("State: "):]
            elif line.startswith("Batch: ") and "; State: " in line:
                state_text = line.partition("; State: ")[2]
            if state_text:
                self.ui.mb_current_state_path = self._output_path(state_text)
                self.ui.mb_batch_label.setText(self.ui.mb_current_state_path.parent.name[:8] + "…")
                self._poll_state()
            if line.startswith("Manifest: "):
                self.manifest_path = self._output_path(line[len("Manifest: "):])
        elif self.stage == "customers" and line.startswith("Manifest: "):
            self.snapshot_path = self._output_path(line[len("Manifest: "):])

    @staticmethod
    def _output_path(value):
        path = Path(value.strip())
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _process_error(self, process, error):
        if process is not self.ui.mb_process:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self._log("Не удалось запустить Python-процесс.")
            self._process_finished(process, -1, QProcess.ExitStatus.CrashExit)

    def _process_finished(self, process, code, status):
        if process is not self.ui.mb_process:
            return
        self._read_output(process, final=True)
        self.ui.mb_process = None
        self.kill_timer.stop()
        self.ui.mb_state_timer.stop()
        process.deleteLater()
        if self.stage == "training":
            self._poll_state()
        if self.cancel_requested:
            self._finish(LoadingState.CANCELLED)
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            self._log(f"Процесс завершился с ошибкой (код {code}).")
            self._finish(LoadingState.FAILED)
        elif self.stage == "training":
            # Transport has exited successfully; local validation is not resumable.
            self.stage = "training_validation"
            self.resume_path = None
            if (self.manifest_path is None or self.ui.mb_current_state_path is None
                    or self.manifest_path != self.ui.mb_current_state_path.parent / "manifest.json"):
                self._log("CLI не сообщил final manifest текущего batch.")
                self._finish(LoadingState.FAILED)
                return
            self._log("Проверка завершённого обучающего набора…")
            self._read_metadata("training_manifest", _load_training_summary, self.manifest_path, True)
        elif self.snapshot_path is None:
            self._log("CLI не сообщил manifest customer profile snapshot.")
            self._finish(LoadingState.FAILED)
        else:
            self.stage = "snapshot_validation"
            self._read_metadata("snapshot", _validate_snapshot, self.snapshot_path, self.manifest_path)

    def _finish(self, state):
        self.state = state
        self.ui.mb_state_timer.stop()
        self.resume_path = None
        if self.stage == "training" and state in (LoadingState.CANCELLED, LoadingState.FAILED):
            self.resume_path = self.ui.mb_current_state_path
            if self.resume_path is not None:
                self.ui.mb_resume_summary.setText(f"Batch: {self.resume_path.parent.name[:8]}…\nСостояние сохранено.")
        if self.stage in ("customers", "snapshot_validation", "customer_manifest_validation"):
            self.ui.mb_customers_status_label.setText({LoadingState.FAILED: "FAILED",
                LoadingState.CANCELLED: "Отменено", LoadingState.SUCCESS: "READY"}[state])
        if state == LoadingState.CANCELLED:
            self._log("Операция отменена пользователем. Уже загруженные компоненты сохранены.")
            set_status_ok(self.ui, "Получение данных отменено")
        elif state == LoadingState.FAILED:
            self._error("Завершённый обучающий набор не прошёл проверку. Подробности — в журнале операции."
                        if self.stage == "training_validation" else
                        "Ошибка получения данных. Подробности — в журнале операции.")
        else:
            self._log("Получение данных завершено.")
            set_status_ok(self.ui, "Получение данных завершено.")
        schedule_status_reset(self.ui, 5)
        self._update_controls()
        if self.closing:
            QTimer.singleShot(0, self.ui.close)

    def cancel(self):
        if self.state != LoadingState.RUNNING or self.cancel_requested:
            return
        self.cancel_requested = True
        self._log("Остановка операции…")
        self._update_controls()
        process = self.ui.mb_process
        if process is None:
            self._finish(LoadingState.CANCELLED)
        else:
            process.terminate()
            self.kill_timer.start()

    def _kill_process(self):
        process = self.ui.mb_process
        if process is not None and process.state() != QProcess.ProcessState.NotRunning:
            process.kill()

    def resume(self):
        if self.state == LoadingState.RUNNING or self.resume_path is None:
            return
        self.ui.mb_current_state_path = self.resume_path
        self.manifest_path = None
        self._begin()
        self._log(f"Продолжение batch: {self.resume_path.parent.name}")
        self._launch("training", resume_arguments(self.resume_path))

    def _poll_state(self):
        path = self.ui.mb_current_state_path
        if path is not None and not any(task.kind == "state" and task.generation == self.generation
                                        for task in self.tasks.values()):
            self._read_metadata("state", _load_training_summary, path)

    def _read_metadata(self, kind, function, *args):
        self._task_number += 1
        task = _MetadataTask(self._task_number, self.generation, kind, function, args)
        self.tasks[self._task_number] = task
        task.signals.finished.connect(self._on_metadata, Qt.ConnectionType.QueuedConnection)
        QThreadPool.globalInstance().start(task)

    @pyqtSlot(object)
    def _on_metadata(self, message):
        task_id, generation, kind, result, error = message
        self.tasks.pop(task_id, None)
        if kind == "catalog":
            self._show_catalog(result, error)
            return
        if generation != self.generation:
            return
        if kind == "state":
            # Missing/replaced/partially written state is retried on the next tick.
            if error is None and self.stage in ("training", "training_validation") and self.state != LoadingState.SUCCESS:
                self._show_summary(result)
            return
        if self.state != LoadingState.RUNNING or self.cancel_requested:
            return
        if error is not None:
            self._log(f"Не удалось подтвердить завершённый manifest ({error}).")
            self._finish(LoadingState.FAILED)
        elif kind in ("training_manifest", "customer_manifest"):
            self._show_summary(result)
            self.ui.mb_manifest_label.setText("Готов")
            self.ui.mb_manifest_label.setToolTip(str(self.manifest_path))
            self._log(f"Обучающий manifest готов: {self.manifest_path}")
            self.resume_path = None
            if self.want_customers:
                self._log("Запуск загрузки клиентов и создания customer profile snapshot…")
                self._launch("customers", customers_arguments(self.manifest_path))
            else:
                self._finish(LoadingState.SUCCESS)
        elif kind == "snapshot":
            self.ui.mb_customers_status_label.setToolTip(str(self.snapshot_path))
            self._log(f"Customer profile snapshot готов: {self.snapshot_path}")
            self._finish(LoadingState.SUCCESS)

    def refresh_manifests(self):
        if self.state == LoadingState.RUNNING or any(t.kind == "catalog" for t in self.tasks.values()):
            return
        self.ui.mb_manifest_hint.setText("Проверка доступных наборов…")
        self._read_metadata("catalog", _scan_manifests)
        self._update_controls()

    def _show_catalog(self, result, error):
        a = self.ui
        previous = a.mb_manifest_combo.currentData()
        a.mb_manifest_combo.blockSignals(True)
        a.mb_manifest_combo.clear()
        a.mb_manifest_combo.addItem("Выберите завершённый обучающий набор…", None)
        if error is None:
            for summary in result["manifests"]:
                a.mb_manifest_combo.addItem(f"{summary['period']} | {summary['batch_id'][:8]}…", summary["path"])
        if previous is not None:
            index = a.mb_manifest_combo.findData(previous)
            if index >= 0:
                a.mb_manifest_combo.setCurrentIndex(index)
        a.mb_manifest_combo.blockSignals(False)
        if error is not None:
            a.mb_manifest_hint.setText("Не удалось проверить наборы. Обновите список.")
        elif not result["manifests"]:
            a.mb_manifest_hint.setText("Нет валидных завершённых наборов. Сначала загрузите данные для обучения.")
        else:
            a.mb_manifest_hint.setText("Явно выберите обучающий набор." +
                (f" Пропущено невалидных: {result['invalid']}." if result["invalid"] else ""))
        a.mb_manifest_hint.setWordWrap(True)
        self._update_controls()

    def _show_summary(self, summary):
        a = self.ui
        a.mb_batch_label.setText(summary["batch_id"][:8] + "…")
        a.mb_batch_label.setToolTip(summary["batch_id"])
        a.mb_merges_status_label.setText(summary["merges"])
        for name, counts in summary["sources"].items():
            label = getattr(a, f"mb_{name}_status_label")
            label.setText(f"{counts['ready']} / {summary['days_total']} дней"
                          + (f"; FAILED: {counts['failed']}" if counts['failed'] else ""))
            label.setToolTip(f"READY: {counts['ready']}; FAILED: {counts['failed']}; "
                             f"PENDING: {summary['days_total'] - counts['ready'] - counts['failed']}")
        a.mb_progress.setRange(0, summary["components_total"])
        a.mb_progress.setValue(summary["components_ready"])
        a.mb_progress.setFormat("%v / %m компонентов")
        a.mb_resume_summary.setText(f"Период: {summary['period']}\n"
            f"Готово: {summary['days_ready']} / {summary['days_total']} дней\nBatch: {summary['batch_id'][:8]}…")

    def eventFilter(self, watched, event):
        if watched is self.ui and event.type() == QEvent.Type.Close and self.state == LoadingState.RUNNING:
            event.ignore()
            self.closing = True
            self.cancel()
            return True
        return super().eventFilter(watched, event)
