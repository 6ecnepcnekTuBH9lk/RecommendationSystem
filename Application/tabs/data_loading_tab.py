"""Mindbox loading UI. CLI owns exports; this module only orchestrates jobs."""

from Application.paths import ICONS_DIR, INPUT_DATA_DIR

import codecs
import json
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import sys

from PyQt6.QtCore import (
    QDate, QEvent, QObject, QProcess, QRunnable, QSize,
    QThreadPool, QTimer, Qt, pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import QIcon, QPainter
from PyQt6.QtWidgets import (QDateEdit, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLineEdit, QScrollArea,
                             QLabel, QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget, QSizePolicy)

from Application.tabs.data_processing_tab import create_csv_loading_section
from Application.loading_errors import error_record, format_error, safe_message
from Application.mindbox.manual_sources import normalize_sources
from Application.mindbox.selection import DEFAULT_SELECTION, MindboxSelectionConfig, SELECTION_OPTIONS

from Application.settings.set_status import (set_status_error, set_status_ok,
                                             set_status_processing, schedule_status_reset)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = INPUT_DATA_DIR / "MindboxRaw"


def set_manual_files(ui, name, sources):
    paths = normalize_sources(sources)
    ui.mb_manual_paths[name] = paths
    editor = ui.mb_manual_files[name]
    editor.setText(paths[0].name if len(paths) == 1 else f"Выбрано файлов: {len(paths)}")
    editor.setToolTip("\n".join(str(path) for path in paths))


class LoadingState(Enum):
    IDLE = "Не запущено"
    RUNNING = "Выполняется"
    SUCCESS = "Завершено"
    FAILED = "Ошибка"
    CANCELLED = "Отменено"


class _BusyProgressBar(QProgressBar):
    def text(self):
        return "Идёт загрузка..." if self.maximum() == 0 else super().text()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.maximum() == 0:
            painter = QPainter(self)
            painter.setPen(self.palette().text().color())
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())


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
    args = ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_canonical.py"),
            "export-daily", "--since", since, "--until", until, "--merge-since", merge_since + " 00:00"]
    for field, option in SELECTION_OPTIONS.items():
        for value in getattr(selection, field):
            args.extend((option, value))
    return args


def resume_arguments(state_path):
    script = "mindbox_canonical.py" if "canonical" in Path(state_path).parts else "mindbox_daily_batch.py"
    return ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts" / script), "resume", "--state", str(state_path)]


def manual_interactions_arguments(since, until, selection=DEFAULT_SELECTION):
    if since >= until:
        raise ValueError("Дата начала ручной выгрузки должна быть раньше даты окончания.")
    args = ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_manual_import.py"),
            "interactions", "--since", since, "--until", until]
    for field, option in SELECTION_OPTIONS.items():
        for value in getattr(selection, field):
            args.extend((option, value))
    return args


def _manual_preflight(dates):
    from Application.mindbox.canonical_storage import catalog, require_manual_merges
    require_manual_merges(catalog(RAW_ROOT), *(datetime.fromisoformat(value).replace(tzinfo=timezone.utc) for value in dates))


def customers_arguments(since, until):
    if since >= until:
        raise ValueError("Дата начала периода клиентов должна быть раньше окончания.")
    return ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_canonical.py"),
            "customers", "--since", since, "--until", until]


def _persisted_summary():
    from Application.mindbox.canonical_storage import summary
    from Application.mindbox.canonical_customers import customer_summary
    result = summary(RAW_ROOT)
    result["customers"] = customer_summary(RAW_ROOT)
    return result


def _source_text(value):
    kinds = frozenset(value.get("source_kinds", [value.get("source_kind")]))
    return {frozenset({"API"}): "API", frozenset({"MANUAL"}): "Вручную",
            frozenset({"API", "MANUAL"}): "Ручная + API"}.get(kinds, "")


def _summary_text(value):
    if value is None:
        return "Файлов не найдено"
    def date(text):
        return datetime.fromisoformat(text).strftime("%d.%m.%Y")
    updated = datetime.fromisoformat(value["updated"]).astimezone().strftime("%d.%m.%Y %H:%M")
    intervals = value.get("intervals")
    period = ("; ".join(f"{date(a)} — {date(b)}" for a, b in intervals) if intervals
              else "не задан, загружены все клиенты") if intervals is not None else f"{date(value['since'])} — {date(value['until'])}"
    count = f" | Клиентов: {value['count']:,}".replace(",", " ") if "count" in value else ""
    source = _source_text(value)
    source = f" | {source}" if source else ""
    return f"Обновлено: {updated}{source} | Период: {period}{count}"


def _load_training_summary(path, complete=False):
    # Backend performs schema, completeness, identity and raw-part validation.
    # This can touch many directories, so callers must use a metadata worker.
    if "canonical" in Path(path).parts and Path(path).name == "state.json":
        from Application.mindbox.canonical_jobs import load_job
        job = load_job(path, RAW_ROOT)
        items = job["components"]
        return {"days_total": sum(c["name"] == "actions" for c in items),
                "sources": {name: {"ready": sum(c["done"] for c in items if c["name"] == name)}
                            for name in ("actions", "orders")}}
    from Application.mindbox.daily_training_batch import load_chunked_training_batch
    batch = load_chunked_training_batch(path, raw_root=RAW_ROOT, require_complete=complete)
    counts = batch.diagnostics
    return {
        "path": str(path), "batch_id": batch.batch_id,
        "period": f"{batch.window.interaction_since:%d.%m.%Y} — {batch.window.interaction_until:%d.%m.%Y}",
        "merges": batch.components[0].status,
        "sources": {name: {
            "ready": sum((c.until - c.since).days if batch.source_kind == "CANONICAL" else 1
                         for c in batch.components if c.name == name and c.status == "READY"),
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
            result, error = None, error_record(exc)
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

    # Корень вкладки:
    # сверху две равные колонки,
    # снизу журнал операции на всю ширину.
    root = QVBoxLayout(tab)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    # ==================================================================
    # ВЕРХНЯЯ ЧАСТЬ: ДВЕ КОЛОНКИ
    # ==================================================================
    top = QHBoxLayout()
    top.setContentsMargins(0, 0, 0, 0)
    top.setSpacing(0)

    left_wrap = QWidget()
    right_wrap = QWidget()

    left = QVBoxLayout(left_wrap)
    right = QVBoxLayout(right_wrap)

    left.setSpacing(10)
    right.setSpacing(10)

    # Вертикальный разделитель между колонками
    separator = QFrame()
    separator.setObjectName("vSeparator")
    separator.setFixedWidth(1)
    separator.setFrameShape(QFrame.Shape.NoFrame)

    # Левая колонка остаётся прокручиваемой как страховка
    left_scroll = QScrollArea()
    left_scroll.setWidgetResizable(True)
    left_scroll.setFrameShape(QFrame.Shape.NoFrame)
    left_scroll.setWidget(left_wrap)

    left_panel = QWidget()
    left_panel_layout = QVBoxLayout(left_panel)
    left_panel_layout.setContentsMargins(0, 0, 0, 0)
    left_panel_layout.addWidget(left_scroll, 1)
    top.addWidget(left_panel, 1)
    top.addWidget(separator)
    top.addWidget(right_wrap, 1)

    # Верхняя часть занимает большую часть высоты
    root.addLayout(top, 2)

    aboba.mb_headings = [
        _heading("Загрузка через API Mindbox", left),
    ]

    # ==================================================================
    # ЛЕВАЯ КОЛОНКА
    # Загрузка через API Mindbox
    # ==================================================================

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

    customers_period = QHBoxLayout()
    aboba.mb_customers_since = _date_widget(today.addMonths(-1))
    aboba.mb_customers_until = _date_widget(today)
    customers_period.addWidget(QLabel("Период клиентов: С"))
    customers_period.addWidget(aboba.mb_customers_since, 1)
    customers_period.addWidget(QLabel("По"))
    customers_period.addWidget(aboba.mb_customers_until, 1)
    left.addLayout(customers_period)

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

    # ---------------- Selection ----------------
    selection_grid = QGridLayout()
    selection_grid.setVerticalSpacing(8)

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

        editor = QLineEdit(
            "; ".join(getattr(DEFAULT_SELECTION, field))
        )
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
        "Действия, заказы и объединения загружаются вместе. "
        "Клиенты обновляются отдельно. Верхняя граница периодов не включается."
    )
    aboba.mb_sources_info.setWordWrap(True)
    aboba.mb_sources_info.setAlignment(
        Qt.AlignmentFlag.AlignCenter
    )
    aboba.mb_sources_info.setProperty(
        "class",
        "infoLabel",
    )

    left.addWidget(aboba.mb_sources_info)

    # ---------------- Кнопки API ----------------
    buttons = QHBoxLayout()

    aboba.mb_start_button = QPushButton(
        " Получить данные"
    )
    aboba.mb_start_button.setIcon(
        QIcon(
            str(
                ICONS_DIR
                / "get_data.png"
            )
        )
    )
    aboba.mb_start_button.setIconSize(
        QSize(17, 17)
    )

    aboba.mb_cancel_button = QPushButton(
        " Отменить"
    )
    aboba.mb_cancel_button.setIcon(
        QIcon(
            str(
                ICONS_DIR
                / "failure.png"
            )
        )
    )
    aboba.mb_cancel_button.setIconSize(
        QSize(17, 17)
    )

    buttons.addWidget(
        aboba.mb_start_button
    )
    aboba.mb_customers_button = QPushButton(" Обновить клиентов")
    aboba.mb_customers_button.setIcon(
        QIcon(
            str(
                ICONS_DIR
                / "refresh.png"
            )
        )
    )
    aboba.mb_customers_button.setIconSize(
        QSize(17, 17)
    )
    buttons.addWidget(aboba.mb_customers_button)
    buttons.addWidget(aboba.mb_cancel_button)

    left.addLayout(buttons)

    # ==================================================================
    # ЛЕВАЯ КОЛОНКА
    # Состояние операции
    # ==================================================================
    state_grid = QGridLayout()
    state_grid.setColumnStretch(1, 1)
    for row, (name, title) in enumerate((
        ("orders", "Заказы"), ("actions", "Действия"),
        ("merges", "Объединения клиентов"), ("customers", "Клиенты"),
    )):
        value = QLineEdit("Файлов не найдено")
        value.setReadOnly(True)
        setattr(aboba, f"mb_{name}_status_label", value)
        state_grid.addWidget(QLabel(title + ":"), row, 0)
        state_grid.addWidget(value, row, 1)
    left.addLayout(state_grid)

    aboba.mb_progress = _BusyProgressBar()
    aboba.mb_progress.setTextVisible(True)
    aboba.mb_progress.setRange(0, 1)
    aboba.mb_progress.setValue(0)
    aboba.mb_progress.setFormat("Прогресс загрузки")
    left.addWidget(aboba.mb_progress)

    # ---------------- Resume block ----------------
    aboba.mb_resume_block = QWidget()

    resume = QVBoxLayout(
        aboba.mb_resume_block
    )
    resume.setContentsMargins(0, 0, 0, 0)
    resume.setSpacing(6)

    resume_title = QLabel("Загрузка не завершена")
    resume_title.setAlignment(
        Qt.AlignmentFlag.AlignCenter
    )

    resume.addWidget(
        resume_title
    )

    aboba.mb_resume_summary = QLabel()
    aboba.mb_resume_summary.setWordWrap(True)
    aboba.mb_resume_summary.setAlignment(
        Qt.AlignmentFlag.AlignCenter
    )

    resume.addWidget(
        aboba.mb_resume_summary
    )

    aboba.mb_resume_button = QPushButton(
        " Продолжить"
    )

    aboba.mb_resume_button.setIcon(
        QIcon(
            str(
                ICONS_DIR
                / "continue.png"
            )
        )
    )
    aboba.mb_resume_button.setIconSize(
        QSize(17, 17)
    )

    resume.addWidget(
        aboba.mb_resume_button
    )

    left.addWidget(
        aboba.mb_resume_block
    )

    left.addStretch(1)

    # ==================================================================
    # ПРАВАЯ КОЛОНКА
    # Ручная загрузка Mindbox
    # ==================================================================
    _heading(
        "Ручная загрузка из Mindbox",
        right,
    )

    manual = QGridLayout()

    aboba.mb_manual_files = {}
    aboba.mb_manual_paths = {name: () for name in ("actions", "orders", "customers")}
    aboba.mb_manual_controls = []

    for row, (name, button_text, icon_name) in enumerate((
            ("Actions", " Выбрать действия", "Actions.png"),
            ("Orders", " Выбрать заказы", "Order.png"),
            ("Customers", " Выбрать клиентов", "Customers.png"),
    )):
        editor = QLineEdit()
        editor.setReadOnly(True)
        editor.setPlaceholderText(
            name + ".json"
        )

        button = QPushButton(
            " " + button_text
        )

        button.setIcon(
            QIcon(
                str(
                    ICONS_DIR
                    / icon_name
                )
            )
        )
        button.setIconSize(
            QSize(17, 17)
        )

        def choose(
                _checked=False,
                source_name=name.lower(),
        ):
            paths, _ = QFileDialog.getOpenFileNames(
                aboba,
                "Выберите Mindbox JSON",
                "",
                "JSON (*.json)",
            )

            if paths:
                try:
                    set_manual_files(aboba, source_name, paths)
                except ValueError as exc:
                    aboba.mb_controller._error(str(exc))

        button.clicked.connect(choose)

        manual.addWidget(
            editor,
            row,
            0,
        )
        manual.addWidget(
            button,
            row,
            1,
        )

        aboba.mb_manual_files[
            name.lower()
        ] = editor

        aboba.mb_manual_controls.extend(
            (editor, button)
        )

    right.addLayout(manual)

    manual_period = QHBoxLayout()
    manual_period.setContentsMargins(0, 0, 0, 0)
    manual_period.setSpacing(8)

    manual_period_label = QLabel("Период выгрузки действий и заказов:")
    manual_period_label.setSizePolicy(
        QSizePolicy.Policy.Maximum,
        QSizePolicy.Policy.Preferred,
    )

    aboba.mb_manual_since = _date_widget(today.addDays(-7))
    aboba.mb_manual_until = _date_widget(today)

    aboba.mb_manual_since.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Preferred,
    )
    aboba.mb_manual_until.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Preferred,
    )

    manual_period.addWidget(manual_period_label)
    manual_period.addWidget(QLabel("С"))
    manual_period.addWidget(aboba.mb_manual_since, 1)
    manual_period.addWidget(QLabel("По"))
    manual_period.addWidget(aboba.mb_manual_until, 1)

    aboba.mb_manual_controls.extend(
        (aboba.mb_manual_since, aboba.mb_manual_until)
    )

    right.addLayout(manual_period)

    manual_hint = QLabel(
        "Укажите тот же период, что при выгрузке Действий и Заказов в Mindbox. "
        "Верхняя граница не включается (UTC). Период не должен выходить за пределы выгрузки объединений клиентов. "
        "Клиенты загружаются без периода."
    )
    manual_hint.setProperty("class", "infoLabel")
    manual_hint.setWordWrap(True)
    manual_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

    right.addWidget(manual_hint)

    # ---------------- Кнопки manual import ----------------
    manual_buttons = QHBoxLayout()

    aboba.mb_manual_interactions_button = QPushButton(
        " Импорт действий и заказов"
    )

    aboba.mb_manual_interactions_button.setIcon(
        QIcon(
            str(
                ICONS_DIR
                / "plus.png"
            )
        )
    )
    aboba.mb_manual_interactions_button.setIconSize(
        QSize(17, 17)
    )

    aboba.mb_manual_customers_button = QPushButton(
        " Импорт клиентов"
    )

    aboba.mb_manual_customers_button.setIcon(
        QIcon(
            str(
                ICONS_DIR
                / "customers2.png"
            )
        )
    )
    aboba.mb_manual_customers_button.setIconSize(
        QSize(17, 17)
    )

    for button in (
            aboba.mb_manual_interactions_button,
            aboba.mb_manual_customers_button,
    ):
        manual_buttons.addWidget(button)
        aboba.mb_manual_controls.append(
            button
        )

    right.addLayout(
        manual_buttons
    )

    # ==================================================================
    # ПРАВАЯ КОЛОНКА
    # Загрузка справочников
    # ==================================================================
    right.addWidget(
        create_csv_loading_section(aboba)
    )

    right.addStretch(1)

    # ==================================================================
    # ГОРИЗОНТАЛЬНЫЙ РАЗДЕЛИТЕЛЬ
    # ==================================================================
    horizontal_separator = QFrame()
    horizontal_separator.setObjectName(
        "hSeparator"
    )
    horizontal_separator.setFixedHeight(1)
    horizontal_separator.setFrameShape(
        QFrame.Shape.NoFrame
    )

    root.addWidget(
        horizontal_separator
    )

    # ==================================================================
    # НИЖНЯЯ ЧАСТЬ
    # Журнал операции на всю ширину
    # ==================================================================
    log_wrap = QWidget()

    log_layout = QVBoxLayout(
        log_wrap
    )
    log_layout.setContentsMargins(8, 8, 8, 0)
    log_layout.setSpacing(0)

    aboba.mb_log = QTextEdit()
    aboba.mb_log.setReadOnly(True)
    aboba.mb_log.setAcceptRichText(False)
    aboba.mb_log.setPlaceholderText(
        "Это журнал операций, логи загрузки будут отображаться здесь..."
    )

    aboba.mb_log.document().setMaximumBlockCount(
        3000
    )

    aboba.mb_log.setSizePolicy(
        QSizePolicy.Policy.Expanding,
        QSizePolicy.Policy.Expanding,
    )

    log_layout.addWidget(
        aboba.mb_log,
        1,
    )

    root.addWidget(
        log_wrap,
        1,
    )

    # ==================================================================
    # Добавление вкладки
    # ==================================================================
    aboba.tabs.insertTab(
        0,
        tab,
        "Получение данных",
    )

    aboba.mb_controller = _LoadingController(
        aboba
    )


class _LoadingController(QObject):
    def __init__(self, aboba):
        super().__init__(aboba)
        self.ui = aboba
        self.state = LoadingState.IDLE
        self.error_reported = False
        self.resume_path = None
        self.cancel_requested = False
        self.closing = False
        self.generation = 0
        self.stage = None
        self.manifest_path = None
        self.snapshot_path = None
        self.resume_stage = "training"
        self.reference_queue = []
        self.reference_failures = []
        self.reference_kind = None
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
        aboba.mb_customers_button.clicked.connect(self.start_customers)
        aboba.mb_cancel_button.clicked.connect(self.cancel)
        aboba.mb_resume_button.clicked.connect(self.resume)
        aboba.mb_manual_interactions_button.clicked.connect(lambda: self.start_manual("interactions"))
        aboba.mb_manual_customers_button.clicked.connect(lambda: self.start_manual("customers"))
        QTimer.singleShot(0, self.refresh_persisted)
        aboba.installEventFilter(self)
        self._update_controls()

    def _update_controls(self):
        a = self.ui
        running = self.state == LoadingState.RUNNING
        for widget in (a.mb_interaction_since, a.mb_interaction_until, a.mb_merge_since,
                       *a.mb_selection_fields.values(), *a.mb_manual_controls, *a.reference_controls, a.btn_load):
            widget.setEnabled(not running)
        a.mb_start_button.setEnabled(not running)
        a.mb_cancel_button.setEnabled(running and not self.cancel_requested)
        a.mb_resume_block.setVisible(not running and self.resume_path is not None)
        a.mb_resume_button.setEnabled(not running and self.resume_path is not None)
        a.mb_customers_button.setEnabled(not running)
        a.mb_customers_since.setEnabled(not running)
        a.mb_customers_until.setEnabled(not running)

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
        self._report_error(safe_message(message))
        set_status_error(self.ui, safe_message(message))

    def _report_error(self, message):
        if self.state == LoadingState.RUNNING and self.stage == "reference_csv" and self.reference_kind:
            message = f"{self.reference_kind}: {message}"
        self.error_reported = True
        self._log(message)

    def _begin(self):
        self.generation += 1
        self.error_reported = False
        self.cancel_requested = False
        self.state = LoadingState.RUNNING
        reset_timer = getattr(self.ui, "_status_reset_timer", None)
        if reset_timer is not None:
            reset_timer.stop()
        set_status_processing(self.ui, "Идёт получение данных из Mindbox...")
        self.ui.mb_progress.setRange(0, 0)
        self.ui.mb_progress.setFormat("Идёт загрузка...")
        self.ui.mb_log.clear()
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
                                        (a.mb_interaction_since, a.mb_interaction_until, a.mb_merge_since)),
                                      selection=selection)
        except ValueError as exc:
            self._error(str(exc))
            return
        self.resume_path = None
        a.mb_current_state_path = None
        self.manifest_path = self.snapshot_path = None
        self._begin()
        for name in ("merges", "actions", "orders"):
            getattr(a, f"mb_{name}_status_label").setText("Выполняется...")
        self._log("Запуск получения данных")
        self._log(f"Период: {a.mb_interaction_since.text()} — {a.mb_interaction_until.text()}")
        self._launch("training", args)

    def start_customers(self):
        if self.state == LoadingState.RUNNING:
            return
        try:
            args = customers_arguments(self.ui.mb_customers_since.date().toString("yyyy-MM-dd"),
                                       self.ui.mb_customers_until.date().toString("yyyy-MM-dd"))
        except ValueError as exc:
            self._error(str(exc))
            return
        self.previous_resume_path = self.resume_path
        self.previous_resume_stage = self.resume_stage
        self.snapshot_path = None
        self.ui.mb_current_state_path = None
        self._begin()
        self._log("Запуск обновления клиентов")
        self._log(f"Период: {self.ui.mb_customers_since.text()} — {self.ui.mb_customers_until.text()}")
        self._launch("customers", args)

    def refresh_persisted(self):
        self._read_metadata("persisted", _persisted_summary)

    def start_references(self):
        from Application.files.reference_import import REFERENCE_TYPES
        selected = [(kind, self.ui.reference_paths[kind]) for kind in REFERENCE_TYPES
                    if self.ui.reference_paths[kind] is not None]
        self._start_reference_queue(selected)

    def start_reference(self, path, kind):
        self._start_reference_queue([(kind, Path(path))])

    def _start_reference_queue(self, selected):
        from Application.files.reference_import import REFERENCE_TYPES
        if self.state == LoadingState.RUNNING:
            return
        if not selected:
            self._error("Выберите хотя бы один файл справочника.")
            return
        if any(kind not in REFERENCE_TYPES for kind, _ in selected):
            self._error("Допустимы только справочники CSV")
            return
        self.reference_queue = list(selected)
        self.reference_failures = []
        self.previous_resume_path = self.resume_path
        self._begin()
        self.ui.mb_progress.setRange(0, 0)
        self._next_reference()

    def _next_reference(self):
        kind, path = self.reference_queue.pop(0)
        self.reference_kind = kind
        self.reference_result = None
        self.error_reported = False
        self._log(f"Загрузка справочника: {kind}")
        self._launch("reference_csv", ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/import_reference_csv.py"),
                                       "--file", str(path), "--kind", kind])

    def _reference_finished(self, code, status):
        from Application.tabs.data_processing_tab import apply_reference_result, update_file_status
        kind = self.reference_kind or "Справочник"
        failed = code != 0 or status != QProcess.ExitStatus.NormalExit or self.error_reported
        if failed:
            if not self.error_reported:
                self._report_error(f"Ошибка импорта справочника: процесс завершился с кодом {code}.")
        elif self.reference_result is None:
            self._report_error("Ошибка импорта справочника: процесс не передал результат импорта.")
            failed = True
        else:
            try:
                apply_reference_result(self.ui, self.reference_result)
                self._log(f"Справочник загружен: {kind}")
            except Exception as exc:
                self._report_error(format_error(error_record(exc), context="Ошибка применения справочника"))
                failed = True
        self.ui.reference_status_overrides[kind] = not failed
        update_file_status(self.ui)
        if failed:
            self.reference_failures.append(kind)
        if self.reference_queue:
            self._next_reference()
        else:
            if self.reference_failures:
                self.error_reported = True
                self._log("Не загружены справочники: " + ", ".join(self.reference_failures))
            self._finish(LoadingState.FAILED if self.reference_failures else LoadingState.SUCCESS)

    def start_manual(self, kind):
        if self.state == LoadingState.RUNNING:
            return
        a = self.ui
        try:
            names = ("actions", "orders") if kind == "interactions" else ("customers",)
            paths = {name: a.mb_manual_paths[name] for name in names}
            if any(not value for value in paths.values()):
                raise ValueError("Выберите необходимые JSON-файлы для ручного импорта.")
            paths = {name: normalize_sources(value) for name, value in paths.items()}
            args = ["-u", "-X", "utf8", str(PROJECT_ROOT / "scripts/mindbox_manual_import.py"), kind]
            if kind == "interactions":
                selection = MindboxSelectionConfig(**{
                    field: parse_selection_values(
                        editor.toPlainText() if isinstance(editor, QTextEdit) else editor.text())
                    for field, editor in a.mb_selection_fields.items()
                })
                dates = [w.date().toString("yyyy-MM-dd") for w in
                         (a.mb_manual_since, a.mb_manual_until)]
                args = manual_interactions_arguments(*dates, selection=selection)
            for name, sources in paths.items():
                for path in sources:
                    args.extend(("--" + name, str(path)))
        except ValueError as exc:
            self._error(str(exc))
            return
        self.previous_resume_path = self.resume_path
        self.manifest_path = self.snapshot_path = self.resume_path = None
        a.mb_current_state_path = None
        self._begin()
        a.mb_progress.setRange(0, 0)
        if kind == "interactions":
            self.stage = "manual_interactions"
            self.manual_arguments = args
            for name in ("actions", "orders"):
                getattr(a, f"mb_{name}_status_label").setText("Выполняется...")
            self._read_metadata("manual_preflight", _manual_preflight, tuple(dates))
        else:
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
            for field in ("actions", "orders", "merges"):
                getattr(self.ui, f"mb_{field}_status_label").setText("Выполняется...")
            self.ui.mb_state_timer.start()
        elif stage in ("customers", "manual_customers"):
            self.ui.mb_customers_status_label.setText("Выполняется")
        elif stage == "manual_interactions":
            for field in ("actions", "orders"):
                getattr(self.ui, f"mb_{field}_status_label").setText("Выполняется...")

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
        # Manual progress is a closed grammar: no paths, record values or arbitrary stdout.
        if self.stage in ("manual_interactions", "manual_customers"):
            match = re.fullmatch(r"(Копирование|Проверка) (actions|orders|customers): файл ([0-9]{1,9}) из ([0-9]{1,9})", line)
            if match:
                part, total = int(match[3]), int(match[4])
                if 1 <= part <= total:
                    self._log(f"{match[1]} {match[2]}: файл {part} из {total}")
                return
            if line in ("Проверка actions ...", "Проверка orders ..."):
                self._log(line)
                return
        if self.stage == "reference_csv" and line.startswith("Reference: "):
            try:
                self.reference_result = json.loads(line[len("Reference: "):])
            except ValueError:
                self.reference_result = None
            return
        if line.startswith("Event: "):
            try:
                event = json.loads(line[7:])
                source, ready, total = event["source"], event["ready"], event["total"]
                labels = {"actions": ("actions", "Действия", "дней"), "orders": ("orders", "Заказы", "дней"),
                          "customer_merges": ("merges", "Объединения клиентов", ""), "customers": ("customers", "Клиенты", "месяцев")}
                field, title, unit = labels[source]
                if type(ready) is not int or type(total) is not int or not 0 <= ready <= total:
                    return
                text = f"Загружено {unit}: {ready} из {total}" if unit else "Получены"
                getattr(self.ui, f"mb_{field}_status_label").setText(text)
                if source == "customers" or source == "customer_merges" or ready % 5 == 0 or ready == total:
                    self._log(f"{title}: {text.lower()}")
            except (ValueError, KeyError, TypeError):
                pass
            return
        if line.startswith("Error: "):
            if self.cancel_requested:
                return
            try:
                error = json.loads(line[7:])
            except (ValueError, TypeError, KeyError):
                error = None
            self._report_error(format_error(error, self.stage))
            return
        if line.startswith("State: "):
            self.ui.mb_current_state_path = self._output_path(line[7:])
            return
        if line.startswith("Manifest: "):
            path = self._output_path(line[10:])
            if self.stage == "customers":
                self.snapshot_path = path
            else:
                self.manifest_path = path
        # Everything else is technical output, never user-facing text.

    @staticmethod
    def _output_path(value):
        path = Path(value.strip())
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _process_error(self, process, error):
        if process is not self.ui.mb_process:
            return
        if error == QProcess.ProcessError.FailedToStart:
            if not self.cancel_requested:
                self._report_error("Не удалось запустить Python-процесс. Проверьте доступность интерпретатора и права запуска.")
            self._process_finished(process, -1, QProcess.ExitStatus.CrashExit)

    def _process_finished(self, process, code, status):
        if process is not self.ui.mb_process:
            return
        self._read_output(process, final=True)
        self.ui.mb_process = None
        self.kill_timer.stop()
        self.ui.mb_state_timer.stop()
        process.deleteLater()
        if self.cancel_requested:
            self._finish(LoadingState.CANCELLED)
        elif self.stage == "reference_csv":
            self._reference_finished(code, status)
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            if not self.error_reported:
                self._report_error(format_error({"message": f"процесс завершился с кодом {code} без дополнительного описания причины."}, self.stage))
            self._finish(LoadingState.FAILED)
        elif self.stage.startswith("manual_"):
            if self.manifest_path is None:
                self._report_error(format_error({"message": "процесс не передал подтверждение сохранённого результата."}, self.stage))
                self._finish(LoadingState.FAILED)
            else:
                self._read_metadata("manual_result", _validate_manual_result, self.manifest_path, self.stage)
        elif self.stage == "training":
            self.stage = "training_validation"
            if self.manifest_path is None:
                self._report_error("Ошибка выгрузки: процесс не передал подтверждение сохранённого результата.")
                self._finish(LoadingState.FAILED)
            else:
                self._read_metadata("training_manifest", _load_training_summary, self.manifest_path, True)
        elif self.snapshot_path is None:
            self._report_error("Ошибка выгрузки клиентов: процесс не передал подтверждение сохранённого результата.")
            self._finish(LoadingState.FAILED)
        else:
            self.stage = "snapshot_validation"
            self._read_metadata("snapshot", _validate_manual_result, self.snapshot_path, "manual_customers")

    def _finish(self, state):
        self.state = state
        if self.stage == "reference_csv":
            self.reference_queue.clear()
        self.ui.mb_state_timer.stop()
        self.resume_path = getattr(self, "previous_resume_path", None) if self.stage.startswith("manual_") or self.stage == "reference_csv" else None
        if self.stage == "snapshot_validation" and state == LoadingState.SUCCESS:
            self.resume_path = getattr(self, "previous_resume_path", None)
            self.resume_stage = getattr(self, "previous_resume_stage", "training")
        if self.stage in ("training", "customers") and state in (LoadingState.CANCELLED, LoadingState.FAILED):
            self.resume_path = self.ui.mb_current_state_path
            self.resume_stage = self.stage
        if self.resume_path is not None:
            self.ui.mb_resume_summary.setText("Состояние сохранено. Можно продолжить незавершённую загрузку.")
        self.ui.mb_progress.setRange(0, 1)
        self.ui.mb_progress.setValue(0)
        self.ui.mb_progress.setFormat("Прогресс загрузки")
        if state == LoadingState.CANCELLED:
            self._log("Операция отменена. Уже сохранённые данные остаются доступны.")
            set_status_ok(self.ui, "Получение данных отменено")
        elif state == LoadingState.FAILED:
            if not self.error_reported:
                self._report_error("Ошибка: процесс завершился без дополнительного описания причины.")
            self._log("Операция завершилась с ошибкой. Ранее сохранённые данные остаются доступны.")
            set_status_error(self.ui, "Операция завершилась с ошибкой.")
        else:
            self._log("Клиенты успешно обновлены" if self.stage in ("customers", "snapshot_validation", "manual_customers") else "Данные успешно обновлены")
            set_status_ok(self.ui, "Получение данных завершено.")
        schedule_status_reset(self.ui, 5)
        self.refresh_persisted()
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
        self.snapshot_path = None
        self._begin()
        self._log("Продолжение незавершённой загрузки")
        self._launch(self.resume_stage, resume_arguments(self.resume_path))

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
        if kind == "persisted":
            if self.state != LoadingState.RUNNING:
                if error is None:
                    for name, value in result.items():
                        field = "merges" if name == "customer_merges" else name
                        getattr(self.ui, f"mb_{field}_status_label").setText(_summary_text(value))
                else:
                    self._report_error(format_error(error, context="Ошибка чтения сохранённых данных"))
                    for field in ("actions", "orders", "merges", "customers"):
                        getattr(self.ui, f"mb_{field}_status_label").setText("Не удалось прочитать сохранённые данные")
            return
        if kind == "state":
            if error is None and self.stage == "training" and self.state == LoadingState.RUNNING:
                self._show_summary(result)
            return
        if self.state != LoadingState.RUNNING or self.cancel_requested:
            return
        if kind == "manual_preflight":
            if error is not None:
                self._report_error(format_error(error, context="Ошибка проверки условий ручного импорта"))
                self._finish(LoadingState.FAILED)
            else:
                self._launch("manual_interactions", self.manual_arguments)
            return
        if error is not None:
            self._report_error(format_error(error, context="Ошибка проверки сохранённого результата"))
            self._finish(LoadingState.FAILED)
        elif kind in ("manual_result", "training_manifest", "snapshot"):
            self._finish(LoadingState.SUCCESS)

    def _show_summary(self, summary):
        for name, counts in summary["sources"].items():
            getattr(self.ui, f"mb_{name}_status_label").setText(f"Загружено дней: {counts['ready']} из {summary['days_total']}")

    def eventFilter(self, watched, event):
        if watched is self.ui and event.type() == QEvent.Type.Close and self.state == LoadingState.RUNNING:
            event.ignore()
            self.closing = True
            self.cancel()
            return True
        return super().eventFilter(watched, event)
