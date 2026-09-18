"""Mindbox loading UI. CLI owns exports; this module only orchestrates jobs."""

import codecs
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
import sys

from PyQt6.QtCore import (
    QDate, QEvent, QObject, QProcess, QRunnable, QSize,
    QThreadPool, QTimer, Qt, pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QDateEdit, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLineEdit, QScrollArea,
                            QLabel, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget, QSizePolicy)

from Application.tabs.data_processing_tab import create_csv_loading_section
from Application.mindbox.selection import DEFAULT_SELECTION, MindboxSelectionConfig, SELECTION_OPTIONS

from Application.settings.set_status import (set_status_error, set_status_ok,
                                             set_status_processing, schedule_status_reset)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "ВходныеДанные" / "MindboxRaw"


class LoadingState(Enum):
    IDLE = "Не запущено"
    RUNNING = "Выполняется"
    SUCCESS = "Завершено"
    FAILED = "Ошибка"
    CANCELLED = "Отменено"


def _heading(text, layout):
    label = QLabel(text)
    label.setProperty("class", "sectionHeader")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)
    return label


def _date_widget(date):
    widget = QDateEdit(date)
    widget.setDisplayFormat("dd.MM.yyyy")
    widget.setCalendarPopup(True)
    return widget


def parse_selection_values(text):
    return tuple(dict.fromkeys(value.strip() for value in text.split(";") if value.strip()))


def training_arguments(since, until, merge_since, selection=DEFAULT_SELECTION):
    """CLI interprets these dates as UTC, with an exclusive upper boundary."""
    if since >= until:
        raise ValueError("Дата начала взаимодействий должна быть раньше даты окончания.")
    if merge_since > since:
        raise ValueError("История объединений должна начинаться не позже периода взаимодействий.")
    args = ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_daily_batch.py"),
            "export-daily", "--since", since, "--until", until, "--merge-since", merge_since + " 00:00"]
    for field, option in SELECTION_OPTIONS.items():
        for value in getattr(selection, field):
            args.extend((option, value))
    return args


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


def _validate_snapshot(path, training_manifest):
    from Application.mindbox.customer_profile_snapshot import load_customer_profile_snapshot
    snapshot = load_customer_profile_snapshot(path, raw_root=RAW_ROOT)
    if snapshot.originating_training_batch_id != Path(training_manifest).parent.name:
        raise ValueError("Snapshot references another training batch")
    return str(path)


def _manual_merges_info(dates):
    from Application.mindbox.manual_import import select_merges_source, merges_description, ManualImportError
    from Application.mindbox.training_batch import TrainingBatchWindow
    from datetime import timezone

    messages = []
    for mode in ("Взаимодействия", "Customers"):
        try:
            window = TrainingBatchWindow(*(datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
                                           for value in dates)) if mode == "Взаимодействия" else None
            messages.append(mode + ": " + merges_description(select_merges_source(RAW_ROOT, window)))
        except (ManualImportError, ValueError):
            messages.append(mode + ": нет подходящей сохранённой истории объединений клиентов.")
    return dates, "\n".join(messages)


def _validate_manual_result(path, stage):
    if stage == "manual_interactions":
        return _load_training_summary(path, True)
    from Application.mindbox.customer_profile_snapshot import load_customer_profile_snapshot
    load_customer_profile_snapshot(path, raw_root=RAW_ROOT)
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


class _SelectionEventsEdit(QTextEdit):
    """Wrap long technical names and keep all lines visible without scrolling."""
    def __init__(self, values):
        super().__init__()
        self.setAcceptRichText(False)
        self.setTabChangesFocus(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setPlainText(";\n".join(values))
        self.document().documentLayout().documentSizeChanged.connect(self._fit_height)

    def _fit_height(self, *_):
        margins = self.contentsMargins()
        self.setFixedHeight(int(self.document().size().height()) + margins.top() + margins.bottom() + 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_height()


def create_data_loading_widgets_tab(aboba):
    tab = QWidget()

    root = QHBoxLayout(tab)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    left_wrap, right_wrap = QWidget(), QWidget()
    left, right = QVBoxLayout(left_wrap), QVBoxLayout(right_wrap)

    left.setSpacing(12)
    right.setSpacing(10)

    separator = QFrame()
    separator.setObjectName("vSeparator")
    separator.setFixedWidth(1)
    separator.setFrameShape(QFrame.Shape.NoFrame)

    left_scroll = QScrollArea()
    left_scroll.setWidgetResizable(True)
    left_scroll.setFrameShape(QFrame.Shape.NoFrame)
    left_scroll.setWidget(left_wrap)
    root.addWidget(left_scroll, 1)
    root.addWidget(separator)
    root.addWidget(right_wrap, 1)

    aboba.mb_headings = [
        _heading("Загрузка через API Mindbox", left),
        _heading("Состояние операции", right),
    ]

    # ---------------- Период взаимодействий ----------------
    period_row = QHBoxLayout()
    period_row.setContentsMargins(0, 0, 0, 0)
    period_row.setSpacing(8)

    period_label = QLabel("Период взаимодействий:")
    period_label.setSizePolicy(
        QSizePolicy.Policy.Maximum,
        QSizePolicy.Policy.Preferred,
    )

    today = QDate.currentDate()

    aboba.mb_interaction_since = _date_widget(today.addDays(-7))
    aboba.mb_interaction_until = _date_widget(today)

    aboba.mb_interaction_since.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Preferred,
    )
    aboba.mb_interaction_until.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Preferred,
    )

    period_row.addWidget(period_label)
    period_row.addWidget(QLabel("С"))
    period_row.addWidget(aboba.mb_interaction_since, 1)
    period_row.addWidget(QLabel("По"))
    period_row.addWidget(aboba.mb_interaction_until, 1)

    left.addLayout(period_row)

    # ---------------- История объединений клиентов ----------------
    merge_row = QHBoxLayout()
    merge_row.setContentsMargins(0, 0, 0, 0)
    merge_row.setSpacing(8)

    merge_label = QLabel("Объединения клиентов:")
    merge_label.setSizePolicy(
        QSizePolicy.Policy.Maximum,
        QSizePolicy.Policy.Preferred,
    )

    aboba.mb_merge_since = _date_widget(QDate(2025, 1, 1))
    aboba.mb_merge_since.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Preferred,
    )

    merge_row.addWidget(merge_label)
    merge_row.addWidget(aboba.mb_merge_since, 1)

    left.addLayout(merge_row)

    selection_grid = QGridLayout()
    selection_grid.setVerticalSpacing(12)
    aboba.mb_selection_fields = {}

    for row, (field, label_text) in enumerate((
            ("purchase_line_statuses", "Статусы покупок:"),
            ("action_product_namespaces", "ID товаров в действиях:"),
            ("order_product_namespaces", "ID товаров в заказах:"),
            ("view_action_system_names", "События просмотров:"),
            ("favorite_action_system_names", "События избранного:"),
    )):
        label = QLabel(label_text)
        label.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Preferred,
        )

        editor = QLineEdit("; ".join(getattr(DEFAULT_SELECTION, field)))
        editor.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        selection_grid.addWidget(label, row, 0)
        selection_grid.addWidget(editor, row, 1)

        aboba.mb_selection_fields[field] = editor

    left.addLayout(selection_grid)

    # ---------------- Информационная подпись ----------------
    aboba.mb_sources_info = QLabel(
        "Из Mindbox будут получены данные о клиентах, их действия, заказы и объединения"
    )
    aboba.mb_sources_info.setWordWrap(True)
    aboba.mb_sources_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
    aboba.mb_sources_info.setProperty("class", "infoLabel")

    left.addWidget(aboba.mb_sources_info)

    # ---------------- Кнопки ----------------
    buttons = QHBoxLayout()

    aboba.mb_start_button = QPushButton(" Получить данные")
    aboba.mb_start_button.setIcon(
        QIcon(str(PROJECT_ROOT / "Картинки" / "ПровестиАнализ.png"))
    )
    aboba.mb_start_button.setIconSize(QSize(17, 17))

    aboba.mb_cancel_button = QPushButton(" Отменить")
    aboba.mb_cancel_button.setIcon(
        QIcon(str(PROJECT_ROOT / "Картинки" / "Неудача.png"))
    )
    aboba.mb_cancel_button.setIconSize(QSize(17, 17))

    buttons.addWidget(aboba.mb_start_button)
    buttons.addWidget(aboba.mb_cancel_button)

    left.addLayout(buttons)

    _heading("Ручная загрузка Mindbox", left)
    manual = QGridLayout()
    aboba.mb_manual_files = {}
    aboba.mb_manual_controls = []
    for row, name in enumerate(("Actions", "Orders", "Customers")):
        editor = QLineEdit()
        editor.setReadOnly(True)
        editor.setPlaceholderText(name + ".json")
        button = QPushButton("Выбрать " + name)
        def choose(_checked=False, target=editor):
            path, _ = QFileDialog.getOpenFileName(aboba, "Выберите Mindbox JSON", "", "JSON (*.json)")
            if path:
                target.setText(path)
        button.clicked.connect(choose)
        manual.addWidget(editor, row, 0)
        manual.addWidget(button, row, 1)
        aboba.mb_manual_files[name.lower()] = editor
        aboba.mb_manual_controls.extend((editor, button))
    left.addLayout(manual)
    hint = QLabel("Actions и Orders: один период из полей выше (UTC). В Mindbox UTC+03 укажите +3 часа: "
                  "00:00 UTC = 03:00. Actions выгружайте широко: VIEW/FAVORITE отбирает приложение. "
                  "Customers — полный snapshot, даты не используются.")
    hint.setWordWrap(True)
    left.addWidget(hint)
    aboba.mb_manual_merges = QLabel("Проверка сохранённых объединений клиентов…")
    aboba.mb_manual_merges.setWordWrap(True)
    left.addWidget(aboba.mb_manual_merges)
    manual_buttons = QHBoxLayout()
    aboba.mb_manual_interactions_button = QPushButton("Импорт Actions + Orders")
    aboba.mb_manual_customers_button = QPushButton("Импорт Customers")
    for button in (aboba.mb_manual_interactions_button, aboba.mb_manual_customers_button):
        manual_buttons.addWidget(button)
        aboba.mb_manual_controls.append(button)
    left.addLayout(manual_buttons)

    # ---------------- CSV ----------------
    left.addWidget(create_csv_loading_section(aboba))
    left.addStretch(1)

    # ---------------- Состояние операции ----------------
    for name, text in (
        ("status", "Статус"),
        ("merges_status", "Объединения клиентов"),
        ("actions_status", "Действия"),
        ("orders_status", "Заказы"),
        ("customers_status", "Клиенты"),
        ("batch", "Набор"),
        ("manifest", "Манифест"),
    ):
        row = QHBoxLayout()

        row.addWidget(QLabel(text + ":"))

        label = QLabel("—")
        label.setWordWrap(True)
        label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        setattr(aboba, f"mb_{name}_label", label)

        row.addWidget(label, 1)
        right.addLayout(row)

    # ---------------- Прогресс ----------------
    aboba.mb_progress_label = QLabel("Прогресс: —")
    right.addWidget(aboba.mb_progress_label)

    aboba.mb_progress = QProgressBar()
    aboba.mb_progress.setTextVisible(True)
    aboba.mb_progress.setFormat("%p%")
    aboba.mb_progress.setValue(0)

    right.addWidget(aboba.mb_progress)

    # ---------------- Resume block ----------------
    aboba.mb_resume_block = QWidget()

    resume = QVBoxLayout(aboba.mb_resume_block)

    resume.addWidget(QLabel("Незавершённая загрузка"))

    aboba.mb_resume_summary = QLabel()
    aboba.mb_resume_summary.setWordWrap(True)

    resume.addWidget(aboba.mb_resume_summary)

    aboba.mb_resume_button = QPushButton("Продолжить")
    resume.addWidget(aboba.mb_resume_button)

    right.addWidget(aboba.mb_resume_block)

    # ---------------- Журнал операции ----------------
    aboba.mb_headings.append(
        _heading("Журнал операции", right)
    )

    aboba.mb_log = QTextEdit()
    aboba.mb_log.setReadOnly(True)
    aboba.mb_log.setAcceptRichText(False)
    aboba.mb_log.setPlaceholderText(
        "Логи загрузки будут отображаться здесь..."
    )
    aboba.mb_log.document().setMaximumBlockCount(3000)

    aboba.mb_log.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding,
    )

    right.addWidget(aboba.mb_log, 1)

    # ---------------- Добавление вкладки ----------------
    aboba.tabs.insertTab(0, tab, "Получение данных")

    aboba.mb_controller = _LoadingController(aboba)


class _LoadingController(QObject):
    def __init__(self, aboba):
        super().__init__(aboba)
        self.ui = aboba
        self.state = LoadingState.IDLE
        self.resume_path = None
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
        aboba.mb_start_button.clicked.connect(self.start)
        aboba.mb_cancel_button.clicked.connect(self.cancel)
        aboba.mb_resume_button.clicked.connect(self.resume)
        aboba.mb_manual_interactions_button.clicked.connect(lambda: self.start_manual("interactions"))
        aboba.mb_manual_customers_button.clicked.connect(lambda: self.start_manual("customers"))
        for widget in (aboba.mb_interaction_since, aboba.mb_interaction_until, aboba.mb_merge_since):
            widget.dateChanged.connect(self.refresh_merges)
        QTimer.singleShot(0, self.refresh_merges)
        aboba.installEventFilter(self)
        self._update_controls()

    def _update_controls(self):
        a = self.ui
        running = self.state == LoadingState.RUNNING
        for widget in (a.mb_interaction_since, a.mb_interaction_until, a.mb_merge_since,
                       *a.mb_selection_fields.values(), *a.mb_manual_controls, a.btn_load, a.combo_box_types):
            widget.setEnabled(not running)
        a.mb_start_button.setEnabled(not running)
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
        self.ui.mb_progress_label.setText("Прогресс: ожидание...")
        self.ui.mb_log.clear()
        self._log("Запуск получения данных...")
        self._update_controls()

    def start(self):
        if self.state == LoadingState.RUNNING:
            return
        a = self.ui
        try:
            selection = MindboxSelectionConfig(**{
                field: parse_selection_values(editor.toPlainText() if isinstance(editor, QTextEdit) else editor.text())
                for field, editor in a.mb_selection_fields.items()
            })
            args = training_arguments(*(widget.date().toString("yyyy-MM-dd") for widget in
                (a.mb_interaction_since, a.mb_interaction_until, a.mb_merge_since)), selection=selection)
        except ValueError as exc:
            self._error(str(exc))
            return
        self.resume_path = None
        a.mb_current_state_path = None
        self.manifest_path = self.snapshot_path = None
        for name in ("merges_status", "actions_status", "orders_status", "customers_status", "batch", "manifest"):
            label = getattr(a, f"mb_{name}_label")
            label.setText("—")

        self._begin()
        a.mb_customers_status_label.setText("Ожидание")
        self._log(f"Период: {a.mb_interaction_since.text()} — {a.mb_interaction_until.text()} [00:00 UTC, по не включительно)")
        self._log("Источники: Действия, Заказы, Объединения клиентов, Клиенты")
        self._launch("training", args)

    def refresh_merges(self):
        dates = tuple(w.date().toString("yyyy-MM-dd") for w in
                      (self.ui.mb_interaction_since, self.ui.mb_interaction_until, self.ui.mb_merge_since))
        self._read_metadata("merge_info", _manual_merges_info, dates)

    def start_reference(self, path, kind):
        if self.state == LoadingState.RUNNING:
            return
        self.reference_result = None
        self.previous_resume_path = self.resume_path
        self._begin()
        self.ui.mb_progress.setRange(0, 0)
        self.ui.mb_progress_label.setText("Импорт справочника…")
        self._launch("reference_csv", ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/import_reference_csv.py"),
                                       "--file", str(path), "--kind", kind])

    def start_manual(self, kind):
        if self.state == LoadingState.RUNNING:
            return
        a = self.ui
        try:
            names = ("actions", "orders") if kind == "interactions" else ("customers",)
            paths = {name: a.mb_manual_files[name].text() for name in names}
            if any(not value or not Path(value).is_file() for value in paths.values()):
                raise ValueError("Выберите необходимые JSON-файлы для ручного импорта.")
            args = ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_manual_import.py"), kind]
            if kind == "interactions":
                selection = MindboxSelectionConfig(**{
                    field: parse_selection_values(editor.toPlainText() if isinstance(editor, QTextEdit) else editor.text())
                    for field, editor in a.mb_selection_fields.items()
                })
                dates = [w.date().toString("yyyy-MM-dd") for w in
                         (a.mb_interaction_since, a.mb_interaction_until, a.mb_merge_since)]
                args.extend(training_arguments(*dates, selection=selection)[5:])
            for name, path in paths.items():
                args.extend(("--" + name, path))
        except ValueError as exc:
            self._error(str(exc))
            return
        self.previous_resume_path = self.resume_path
        self.manifest_path = self.snapshot_path = self.resume_path = None
        a.mb_current_state_path = None
        self._begin()
        a.mb_progress.setRange(0, 0)
        a.mb_progress_label.setText("Ручной импорт: ход копирования и проверки — в журнале")
        self._launch("manual_" + kind, args)

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
        elif stage in ("customers", "manual_customers"):
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
        if self.stage == "reference_csv" and line.startswith("Reference: "):
            try:
                self.reference_result = json.loads(line[len("Reference: "):])
            except ValueError:
                self.reference_result = None
            return
        self._log(line, timestamp=False)
        if self.stage.startswith("manual_"):
            if line.startswith("Manifest: "):
                self.manifest_path = self._output_path(line[len("Manifest: "):])
            elif line.startswith("Объединения клиентов:"):
                self.ui.mb_manual_merges.setText(line)
            return
        if self.stage == "training":
            state_text = None
            if line.startswith("State: "):
                state_text = line[len("State: "):]
            elif line.startswith("Batch: ") and "; State: " in line:
                state_text = line.partition("; State: ")[2]
            if state_text:
                self.ui.mb_current_state_path = self._output_path(state_text)
                self.ui.mb_batch_label.setText(self.ui.mb_current_state_path.parent.name)
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
        elif self.stage == "reference_csv":
            if self.reference_result is None:
                self._finish(LoadingState.FAILED)
            else:
                from Application.tabs.data_processing_tab import apply_reference_result
                try:
                    apply_reference_result(self.ui, self.reference_result)
                except Exception:
                    self._log("Справочник сохранён; не удалось обновить связанные элементы интерфейса.")
                    self._finish(LoadingState.FAILED)
                    return
                self.ui.mb_progress.setRange(0, 1)
                self.ui.mb_progress.setValue(1)
                self.ui.mb_progress_label.setText("Справочник загружен")
                self._finish(LoadingState.SUCCESS)
        elif self.stage.startswith("manual_"):
            if self.manifest_path is None:
                self._finish(LoadingState.FAILED)
            else:
                self._read_metadata("manual_result", _validate_manual_result, self.manifest_path, self.stage)
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
        self.resume_path = (getattr(self, "previous_resume_path", None)
                            if self.stage.startswith("manual_") or self.stage == "reference_csv" else None)
        if self.stage == "training" and state in (LoadingState.CANCELLED, LoadingState.FAILED):
            self.resume_path = self.ui.mb_current_state_path
            if self.resume_path is not None:
                self.ui.mb_resume_summary.setText(f"Набор: {self.resume_path.parent.name}\nСостояние сохранено.")
        if self.stage in ("customers", "snapshot_validation", "manual_customers"):
            self.ui.mb_customers_status_label.setText({LoadingState.FAILED: "Ошибка",
                LoadingState.CANCELLED: "Отменено", LoadingState.SUCCESS: "Готово"}[state])
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
        if generation != self.generation:
            return
        if kind == "merge_info":
            if error is None:
                dates, text = result
                current = tuple(w.date().toString("yyyy-MM-dd") for w in
                                (self.ui.mb_interaction_since, self.ui.mb_interaction_until, self.ui.mb_merge_since))
                if dates == current:
                    self.ui.mb_manual_merges.setText(text)
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
        elif kind == "manual_result":
            if self.stage == "manual_interactions":
                self.ui.mb_actions_status_label.setText("Готово")
                self.ui.mb_orders_status_label.setText("Готово")
            self.ui.mb_manifest_label.setText("Готов")
            self.ui.mb_progress.setRange(0, 1)
            self.ui.mb_progress.setValue(1)
            self.ui.mb_progress_label.setText("Ручной импорт завершён")
            self._finish(LoadingState.SUCCESS)
        elif kind == "training_manifest":
            self._show_summary(result)
            self.ui.mb_manifest_label.setText("Готов")

            self._log(f"Обучающий manifest готов: {self.manifest_path}")
            self.resume_path = None
            self._log("Запуск загрузки клиентов и создания customer profile snapshot…")
            self._launch("customers", customers_arguments(self.manifest_path))
        elif kind == "snapshot":
            # Customers count as one component, completed only after validation.
            progress = self.ui.mb_progress
            progress.setValue(progress.maximum())
            self.ui.mb_progress_label.setText(
                f"Готово: {progress.maximum()} / {progress.maximum()} компонентов")

            self._log(f"Customer profile snapshot готов: {self.snapshot_path}")
            self._finish(LoadingState.SUCCESS)

    def _show_summary(self, summary):
        a = self.ui
        a.mb_batch_label.setText(summary["batch_id"])

        a.mb_merges_status_label.setText({"READY": "Готово", "PENDING": "Ожидание",
            "FAILED": "Ошибка", "RUNNING": "Выполняется"}.get(summary["merges"], "Неизвестно"))
        for name, counts in summary["sources"].items():
            label = getattr(a, f"mb_{name}_status_label")
            label.setText(f"{counts['ready']} / {summary['days_total']} дней"
                          + (f"; Ошибок: {counts['failed']}" if counts['failed'] else ""))
        a.mb_progress.setRange(0, summary["components_total"] + 1)
        a.mb_progress.setValue(summary["components_ready"])
        a.mb_progress.setFormat("%p%")
        a.mb_progress_label.setText(f"Готово: {summary['components_ready']} / {summary['components_total'] + 1} компонентов")
        a.mb_resume_summary.setText(f"Период: {summary['period']}\n"
            f"Готово: {summary['days_ready']} / {summary['days_total']} дней\nНабор: {summary['batch_id']}")

    def eventFilter(self, watched, event):
        if watched is self.ui and event.type() == QEvent.Type.Close and self.state == LoadingState.RUNNING:
            event.ignore()
            self.closing = True
            self.cancel()
            return True
        return super().eventFilter(watched, event)
