from Application.paths import ICONS_DIR, INPUT_DATA_DIR, USER_SETTINGS_DIR, PROJECT_ROOT
import os
import sys
import json
import time
import tempfile
import uuid
from pathlib import Path
import pandas as pd
import shutil
from functools import partial
from PyQt6.QtCore import Qt, QSize, QProcess, QTimer, QObject, QEvent
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox,
                             QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QFormLayout,
                             QProgressBar, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView)
from Application.settings.settings_and_filter import get_selected_list_values
from Application.settings.set_status import (set_status_processing, schedule_status_reset,
                                             set_status_error, set_status_ok)
from Application.evaluation.experiments.gui_history import ExperimentHistory, now, clean_record, RESEARCH_FIXED
from Application.mindbox.canonical_storage import atomic_json
from .training_workflow import input_values, metadata_readiness, production_config


class _TrainingLifecycle(QObject):
    """Refresh cheap metadata on entry and keep a running worker owned on close."""
    def __init__(self, window, tab):
        super().__init__(window)
        self.window, self.tab = window, tab

    def eventFilter(self, watched, event):
        if watched is self.tab and event.type() == QEvent.Type.Show:
            refresh_training_state(self.window)
        if watched is self.window and event.type() == QEvent.Type.Close and getattr(self.window, '_training_active', False):
            cancel_training(self.window)
            event.ignore()
            return True
        return False


def create_train_model_widgets_tab(aboba):
    tab = QWidget()
    root = QVBoxLayout(tab)
    root.setContentsMargins(0, 0, 0, 0)
    top = QHBoxLayout()
    root.addLayout(top, 3)
    left, right = QVBoxLayout(), QVBoxLayout()
    top.addLayout(left, 4)
    top.addLayout(right, 6)

    def heading(layout, text):
        label = QLabel(text)
        label.setProperty('class', 'sectionHeader')
        layout.addWidget(label)
        return label

    aboba.heading_enter_parameter = heading(left, 'Параметры')
    form = QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    left.addLayout(form)
    aboba.training_mode = QComboBox()
    aboba.training_mode.addItems(['Эксперимент', 'Рабочая модель'])
    form.addRow('Режим обучения', aboba.training_mode)
    for name, label, default in [('w_purchase', 'Вес покупки', 10.), ('w_favorite', 'Вес избранного', 2.),
                                  ('w_view_item', 'Вес просмотра', .5)]:
        widget = QDoubleSpinBox()
        widget.setDecimals(4)
        widget.setRange(0, 1000000)
        widget.setValue(default)
        widget.setObjectName(name)
        setattr(aboba, name, widget)
        form.addRow(label, widget)
    aboba.epochs_input = QSpinBox()
    aboba.epochs_input.setRange(0, 10000)
    aboba.epochs_input.setValue(50)
    aboba.epochs_input.setObjectName('epochs_input')
    form.addRow('Количество эпох', aboba.epochs_input)
    aboba.mode_description = QLabel()
    aboba.mode_description.setWordWrap(True)
    left.addWidget(aboba.mode_description)
    heading(left, 'Данные для обучения')
    aboba.training_readiness = QLabel()
    aboba.training_readiness.setWordWrap(True)
    left.addWidget(aboba.training_readiness)
    aboba.training_reason = QLabel()
    aboba.training_reason.setWordWrap(True)
    left.addWidget(aboba.training_reason)
    left.addStretch()
    aboba.start_train = QPushButton(QIcon(str(ICONS_DIR / 'start_training.png')), 'Начать эксперимент')
    aboba.start_train.setIconSize(QSize(17, 17))
    aboba.start_train.clicked.connect(lambda: start_training_process(aboba))
    aboba.cancel_train = QPushButton('Отменить')
    aboba.cancel_train.setEnabled(False)
    aboba.cancel_train.clicked.connect(lambda: cancel_training(aboba))
    buttons = QHBoxLayout()
    buttons.addWidget(aboba.start_train, 3)
    buttons.addWidget(aboba.cancel_train, 1)
    left.addLayout(buttons)

    aboba.label_69 = heading(right, 'Процесс обучения')
    aboba.training_progress_text = QLabel('Ожидание запуска')
    aboba.training_progress_text.setWordWrap(True)
    aboba.training_progress = QProgressBar()
    aboba.training_progress.setRange(0, 50)
    aboba.training_progress.setValue(0)
    right.addWidget(aboba.training_progress_text)
    right.addWidget(aboba.training_progress)
    aboba.train_log = QTextEdit()
    aboba.train_log.setAcceptRichText(False)
    aboba.train_log.setReadOnly(True)
    aboba.train_log.setPlaceholderText('Логи подготовки, обучения и оценки появятся здесь…')
    aboba.train_log.document().setMaximumBlockCount(5000)
    right.addWidget(aboba.train_log, 1)
    heading(right, 'Результат')
    aboba.training_result = QLabel('Результатов пока нет')
    aboba.training_result.setTextFormat(Qt.TextFormat.PlainText)
    aboba.training_result.setWordWrap(True)
    right.addWidget(aboba.training_result)

    heading(root, 'История экспериментов')
    aboba.history_notice = QLabel()
    aboba.history_notice.setWordWrap(True)
    root.addWidget(aboba.history_notice)
    aboba.experiment_history = QTableWidget(0, 12)
    aboba.experiment_history.setHorizontalHeaderLabels([
        'Дата', 'VIEW', 'FAVORITE', 'PURCHASE', 'Эпохи', 'NDCG@10', 'Recall@10',
        'VIEW NDCG@10', 'PURCHASE NDCG@10', 'Время', 'Device', 'Статус'])
    aboba.experiment_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    aboba.experiment_history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    aboba.experiment_history.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    aboba.experiment_history.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    aboba.experiment_history.horizontalHeader().setStretchLastSection(True)
    aboba.experiment_history.verticalHeader().hide()
    aboba.experiment_history.setMinimumHeight(130)
    root.addWidget(aboba.experiment_history, 2)
    aboba.experiment_history.itemSelectionChanged.connect(lambda: _history_selection(aboba))
    aboba._experiment_store = ExperimentHistory(USER_SETTINGS_DIR / 'research_experiments')
    aboba._training_active = False
    aboba._training_timer = QTimer(tab)
    aboba._training_timer.setInterval(1000)
    aboba._training_timer.timeout.connect(lambda: _update_progress(aboba))
    aboba._training_lifecycle = _TrainingLifecycle(aboba, tab)
    aboba.installEventFilter(aboba._training_lifecycle)
    tab.installEventFilter(aboba._training_lifecycle)
    aboba.training_mode.currentIndexChanged.connect(lambda: refresh_training_state(aboba))
    for widget in (aboba.w_purchase, aboba.w_favorite, aboba.w_view_item, aboba.epochs_input):
        widget.valueChanged.connect(lambda: refresh_training_state(aboba, metadata=False))
        widget.lineEdit().textEdited.connect(lambda: refresh_training_state(aboba, metadata=False))
    _load_history(aboba, recover=True)
    refresh_training_state(aboba)
    aboba.tabs.addTab(tab, 'Обучение модели')
    return tab


def _research(aboba):
    return getattr(aboba, 'training_mode', None) is not None and aboba.training_mode.currentText() == 'Эксперимент'


def _block_key(aboba):
    try:
        values = tuple(input_values(aboba).items())
        path = USER_SETTINGS_DIR / 'train_config.json'
        stat = path.stat() if not _research(aboba) and path.exists() else None
        settings_stamp = (stat.st_mtime_ns, stat.st_size) if stat else None
        return _research(aboba), aboba._training_ready[2], values, settings_stamp
    except (ValueError, OSError):
        return None


def refresh_training_state(aboba, metadata=True):
    research = _research(aboba)
    active = getattr(aboba, '_training_active', False)
    if not active and hasattr(aboba, 'training_result') and getattr(aboba, '_display_mode', research) != research:
        aboba.training_result.setText('Результатов в выбранном режиме пока нет')
    aboba._display_mode = research
    if metadata and not active:
        aboba._training_ready = metadata_readiness(INPUT_DATA_DIR, research)
    ready, reason, revision = getattr(aboba, '_training_ready', (False, 'Данные ещё не проверены', None))
    if hasattr(aboba, 'training_mode'):
        aboba.start_train.setText('Начать эксперимент' if research else 'Обучить рабочую модель')
        aboba.mode_description.setText('Один запуск и оценка на фиксированном validation benchmark. Модель не публикуется.' if research
                                      else 'Обучение на актуальном canonical batch. Новая generation публикуется при успехе.')
    blocker = ''
    if active:
        blocker = 'Выполняется обучение' if not getattr(aboba, '_cancel_requested', False) else 'Отмена: ожидаем завершения процесса'
    elif not ready:
        blocker = reason
    elif research and getattr(aboba, '_benchmark_block', None) is not None and aboba._benchmark_block == revision:
        blocker = 'Temporal validation benchmark недоступен; обновите данные'
    elif getattr(aboba, '_preflight_block', None) is not None and aboba._preflight_block == _block_key(aboba):
        blocker = getattr(aboba, '_preflight_block_reason', 'Проверка данных заблокировала обучение; обновите данные или параметры')
    else:
        try:
            values = input_values(aboba)
            if not research:
                production_config(values, USER_SETTINGS_DIR, INPUT_DATA_DIR)
        except (ValueError, TypeError, AttributeError) as error:
            blocker = str(error) if not isinstance(error, json.JSONDecodeError) else 'Настройки рабочей модели повреждены'
        except OSError:
            blocker = 'Настройки рабочей модели недоступны'
    aboba.start_train.setEnabled(not blocker)
    if hasattr(aboba, 'training_reason'):
        aboba.training_reason.setText(blocker)
        aboba.training_readiness.setText(('● ' if ready else '⚠ ') + reason)
        aboba.cancel_train.setEnabled(active and not getattr(aboba, '_cancel_requested', False) and not getattr(aboba, '_publishing', False))
        for widget in (aboba.training_mode, aboba.w_purchase, aboba.w_favorite, aboba.w_view_item, aboba.epochs_input):
            widget.setEnabled(not active)
    return not blocker


def _confirm_production(aboba, values):
    text = ('Будет выполнено обучение рабочей модели. При успехе новая generation будет опубликована.\n\n'
            f'Покупка: {values["w_purchase"]:g}\nИзбранное: {values["w_favorite"]:g}\nПросмотр: {values["w_view_item"]:g}\n'
            f'Количество эпох: {values["epochs"]}\n\nПродолжить?')
    return QMessageBox.question(aboba, 'Обучение рабочей модели', text,
                               QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                               QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes


def start_training_process(aboba):
    if getattr(aboba, '_training_active', False) or (getattr(aboba, 'train_proc', None) is not None
                                                   and aboba.train_proc.state() != QProcess.ProcessState.NotRunning):
        set_status_error(aboba, 'Обучение уже запущено')
        return
    aboba._training_active = False
    aboba._train_config_path = None
    if not refresh_training_state(aboba):
        set_status_error(aboba, 'Запуск недоступен: проверьте данные и параметры')
        return
    try:
        values = input_values(aboba)
        research = _research(aboba)
        cfg = values if research else production_config(values, USER_SETTINGS_DIR, INPUT_DATA_DIR)
    except (OSError, ValueError, TypeError):
        set_status_error(aboba, 'Не удалось подготовить параметры обучения')
        return
    if not research and not _confirm_production(aboba, values):
        return
    aboba._training_active = True
    aboba._run_is_research = research
    aboba._cancel_requested = False
    aboba._publishing = False
    aboba._train_publication_result = None
    aboba._train_preflight_result = None
    aboba._train_research_result = None
    aboba._train_output_buffer = b''
    aboba._run_started_clock = time.monotonic()
    aboba._progress_event = {}
    aboba.train_log.clear()
    if hasattr(aboba, 'training_result'):
        aboba.training_result.setText('Запуск выполняется…')
        aboba.training_progress.setRange(0, values['epochs'])
        aboba.training_progress.setValue(0)
        aboba._training_timer.start()
    refresh_training_state(aboba, metadata=False)
    set_status_processing(aboba, 'Проверка данных перед обучением...')
    if research:
        aboba._research_record = {'run_id': uuid.uuid4().hex, 'started_at': now(), 'status': 'running',
                                  'hyperparameters': {**RESEARCH_FIXED, **values}, 'epochs_requested': values['epochs'], 'epochs_completed': 0}
    try:
        USER_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json', prefix='train-run-',
                                         dir=USER_SETTINGS_DIR, delete=False) as stream:
            aboba._train_config_path = stream.name
            json.dump(cfg, stream, ensure_ascii=False, allow_nan=False, indent=2)
        aboba._cancel_path = Path(aboba._train_config_path + '.cancel')
        if research:
            _save_history(aboba, aboba._research_record)
    except (OSError, ValueError):
        set_status_error(aboba, 'Не удалось подготовить параметры обучения')
        if research:
            _finish_research(aboba, 1, QProcess.ExitStatus.CrashExit, failed_start=True)
        _release_training(aboba)
        return
    old_process = getattr(aboba, 'train_proc', None)
    if old_process is not None:
        old_process.deleteLater()
    aboba.train_proc = QProcess(aboba)
    aboba.train_proc.setProgram(sys.executable)
    script = 'run_gui_research.py' if research else 'mindbox_production_train.py'
    args = ['-u', '-X', 'utf8', str(PROJECT_ROOT / 'scripts' / script)]
    args += ['--run-id', aboba._research_record['run_id']] if research else ['gui']
    args += ['--config', aboba._train_config_path, '--cancel-file', str(aboba._cancel_path),
             '--raw-root', str(INPUT_DATA_DIR / 'MindboxRaw'), '--catalog', str(INPUT_DATA_DIR / 'nomenclature.csv')]
    if not research:
        args += ['--manifest', str(INPUT_DATA_DIR / 'MindboxRaw/canonical/training.json'), '--device', 'auto']
    aboba.train_proc.setArguments(args)
    aboba.train_proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
    aboba.train_proc.readyReadStandardOutput.connect(partial(_on_train_output, aboba))
    aboba.train_proc.finished.connect(partial(_on_train_finished, aboba))
    aboba.train_proc.errorOccurred.connect(partial(_on_train_error, aboba))
    if research:
        aboba.train_proc.started.connect(lambda: _research_started(aboba))
    aboba.train_proc.setWorkingDirectory(str(PROJECT_ROOT))
    aboba.train_proc.start()


def _research_started(aboba):
    aboba._research_record['pid'] = int(aboba.train_proc.processId())
    _save_history(aboba, aboba._research_record)


def cancel_training(aboba):
    if (not getattr(aboba, '_training_active', False) or getattr(aboba, '_publishing', False)
            or getattr(aboba, '_train_research_result', None) is not None):
        return
    try:
        aboba._cancel_path.write_text('CANCELLED', encoding='ascii')
    except OSError:
        set_status_error(aboba, 'Не удалось отправить отмену; повторите попытку')
        return
    aboba._cancel_requested = True
    aboba.train_proc.write(b'NO\n')
    refresh_training_state(aboba, metadata=False)
    aboba.train_log.append('Отмена запрошена. Ожидаем завершения процесса…')
    process = aboba.train_proc

    def stop():
        # Single-process runners; there is no multiprocessing training descendant.
        # Production cannot publish without the GUI publication handshake.
        if aboba.train_proc is process and getattr(aboba, '_training_active', False) and not aboba._publishing:
            process.kill()
    QTimer.singleShot(2000, stop)


def _duration(seconds):
    seconds = max(0, int(seconds or 0))
    return f'{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}'


def _metric(record, label, name):
    value = record.get('metrics', {}).get(label, {}).get('10', {}).get(name)
    return '—' if value is None else f'{value:.6f}'


def _show_result(aboba, record, research=True):
    if not hasattr(aboba, 'training_result'):
        return
    if research:
        text = (f'{record.get("status", "failed")} · Validation\n'
                f'NDCG@10: {_metric(record, "overall", "ndcg")}    Recall@10: {_metric(record, "overall", "recall")}\n'
                f'VIEW NDCG@10: {_metric(record, "VIEW", "ndcg")}    PURCHASE NDCG@10: {_metric(record, "PURCHASE", "ndcg")}\n'
                f'Эпохи: {record.get("epochs_completed", 0)}    Время: {_duration(record.get("total_seconds"))}    Device: {record.get("device", "—")}')
    else:
        epochs = (record.get('training_metrics') or {}).get('epochs_completed', 0)
        text = (f'Рабочая модель · Эпохи: {epochs}    Время: {_duration(aboba._progress_event.get("total_seconds"))}    '
                f'Device: {aboba._progress_event.get("device", "—")}\n'
                f'Generation: {record.get("published_generation") or "—"}\n'
                f'Публикация: {"успех" if record.get("published") else "не выполнена"} · '
                f'Проверка чтением: {"успех" if record.get("postpublish_validation") else "не выполнена"}')
    aboba.training_result.setText(text)


def _pid_alive(pid):
    if not pid:
        return False
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if handle:
            try:
                code = ctypes.c_ulong()
                return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
            finally:
                kernel.CloseHandle(handle)
        return ctypes.get_last_error() == 5  # Access denied is not evidence of death.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _save_history(aboba, record):
    try:
        aboba._experiment_store.upsert(record)
        _load_history(aboba)
    except (ValueError, OSError):
        aboba.history_notice.setText('Не удалось сохранить историю. Исходный файл сохранён; результат доступен в artifact запуска.')


def _load_history(aboba, recover=False):
    if not hasattr(aboba, '_experiment_store'):
        return
    try:
        records = aboba._experiment_store.recover(_pid_alive) if recover else aboba._experiment_store.read()
        aboba.history_notice.setText('')
    except (ValueError, OSError):
        records = []
        aboba.history_notice.setText('История экспериментов недоступна. Исходный файл сохранён; новые результаты остаются в artifacts.')
    aboba._history_records = records
    table = aboba.experiment_history
    table.setRowCount(len(records))
    for row, record in enumerate(records):
        cfg = record['hyperparameters']
        values = [record.get('started_at', '')[:19].replace('T', ' '), cfg.get('w_view_item', '—'),
                  cfg.get('w_favorite', '—'), cfg.get('w_purchase', '—'),
                  f'{record.get("epochs_completed") or 0}/{record.get("epochs_requested") or "—"}',
                  _metric(record, 'overall', 'ndcg'), _metric(record, 'overall', 'recall'),
                  _metric(record, 'VIEW', 'ndcg'), _metric(record, 'PURCHASE', 'ndcg'),
                  _duration(record.get('total_seconds')), record.get('device', '—'), record['status']]
        for col, value in enumerate(values):
            table.setItem(row, col, QTableWidgetItem(str(value)))


def _history_selection(aboba):
    row = aboba.experiment_history.currentRow()
    if not getattr(aboba, '_training_active', False) and 0 <= row < len(aboba._history_records):
        _show_result(aboba, aboba._history_records[row])


def _update_progress(aboba):
    if not hasattr(aboba, 'training_progress_text'):
        return
    event = aboba._progress_event
    epoch = event.get('epoch', 0)
    epochs = event.get('epochs', aboba.epochs_input.value())
    loss = event.get('loss')
    elapsed = event.get('total_seconds', time.monotonic() - aboba._run_started_clock)
    aboba.training_progress.setValue(epoch)
    phase = {'loading': 'Загрузка данных', 'preparation': 'Подготовка', 'training': 'Обучение',
             'validation': 'Validation', 'publication': 'Публикация', 'finished': 'Завершено',
             'cancelled': 'Отменено', 'failed': 'Ошибка'}.get(event.get('stage'), 'Обучение')
    aboba.training_progress_text.setText(f'{phase} · Эпоха {epoch}/{epochs} · Loss: {"—" if loss is None else f"{loss:.6f}"} · '
                                        f'{_duration(elapsed)} · {event.get("device", "—").upper()}')


def _on_train_output(aboba):
    buffer = getattr(aboba, '_train_output_buffer', b'') + bytes(aboba.train_proc.readAllStandardOutput())
    lines = buffer.split(b'\n')
    aboba._train_output_buffer = lines.pop()
    for line in lines:
        _handle_train_line(aboba, line.decode('utf-8', errors='replace').rstrip('\r'))


def _handle_train_line(aboba, line):
    from scripts.mindbox_production_train import EVENT_PREFIX
    if not line.startswith(EVENT_PREFIX):
        # CLI emits only safe aggregate/plain diagnostics; no exception traceback.
        aboba.train_log.append(line)
        return
    try:
        event = json.loads(line[len(EVENT_PREFIX):])
        stage = event['stage']
    except (ValueError, KeyError, TypeError):
        aboba.train_log.append('Не удалось прочитать диагностику процесса обучения.')
        return
    if stage == 'publication_ready':
        if getattr(aboba, '_cancel_requested', False):
            aboba.train_proc.write(b'NO\n')
        else:
            aboba._publishing = True
            refresh_training_state(aboba, metadata=False)
            aboba.train_proc.write(b'PUBLISH\n')
        return
    if stage == 'research_started':
        record = event['result']
        if record['run_id'] == aboba._research_record['run_id']:
            aboba._research_record.update(record)
            _save_history(aboba, aboba._research_record)
        return
    if stage in {'loading', 'preparation', 'training', 'epoch', 'validation', 'publication', 'timing'}:
        aboba._progress_event.update(event)
        _update_progress(aboba)
        if getattr(aboba, '_run_is_research', False):
            if stage == 'epoch':
                aboba._research_record['epochs_completed'] = event['epoch']
            if stage == 'training':
                aboba._research_record.update({k: v for k, v in event.items() if k in
                                               {'training_users', 'training_items', 'training_pairs', 'hyperparameters'}})
            aboba._research_record['device'] = event.get('device', aboba._research_record.get('device', 'unknown'))
            if stage in {'training', 'epoch'}:
                _save_history(aboba, aboba._research_record)
        aboba.train_log.append(f'Эпоха {event["epoch"]}: loss={event["loss"]:.6f}' if stage == 'epoch' else
                               {'loading': 'Загрузка canonical данных…', 'preparation': 'Подготовка данных…', 'training': 'Обучение…',
                                'validation': 'Оценка validation…', 'publication': 'Безопасная публикация…', 'timing': 'Запуск завершён.'}[stage])
        return
    if stage == 'research_finished':
        aboba._train_research_result = event['result']
        if event['result'].get('error_summary') == 'BENCHMARK_UNAVAILABLE':
            aboba._benchmark_block = aboba._training_ready[2]
        if hasattr(aboba, 'cancel_train'):
            aboba.cancel_train.setEnabled(False)
        _show_result(aboba, event['result'])
        return
    if stage == 'finished':
        aboba._train_publication_result = event
        _show_result(aboba, event, research=False)
        aboba.train_log.append('Результат публикации:\n' + json.dumps(event, ensure_ascii=False, indent=2))
        return
    if stage != 'preflight':
        return
    aboba._train_preflight_result = event
    quality = event.get('quality')
    aboba.train_log.append('Диагностика preflight:\n' + json.dumps(event, ensure_ascii=False, indent=2))
    if event.get('error_code') or not quality or not quality['training_allowed']:
        aboba._preflight_block = _block_key(aboba)
        aboba._preflight_block_reason = 'Проверка данных заблокировала обучение; обновите данные или параметры'
        set_status_error(aboba, 'Проверка данных заблокировала обучение')
    elif quality['level'] == 'WARN':
        process = aboba.train_proc
        reasons = '\n'.join(f'{issue["message"]} ({issue["code"]}): {issue["count"]}' for issue in quality['issues'])
        answer = QMessageBox.question(aboba, 'Предупреждения перед обучением',
                                      f'Preflight: WARN\n{reasons}\n\nПодробная диагностика доступна в журнале. Продолжить обучение?',
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if aboba.train_proc is not process or not getattr(aboba, '_training_active', False):
            return
        process.write(b'YES\n' if answer == QMessageBox.StandardButton.Yes and not aboba._cancel_requested else b'NO\n')
    else:
        set_status_processing(aboba, 'Идёт обучение модели...')


def _release_training(aboba):
    aboba._training_active = False
    if hasattr(aboba, '_training_timer'):
        aboba._training_timer.stop()
    for name in ('_train_config_path', '_cancel_path'):
        path = getattr(aboba, name, None)
        if path:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                aboba.train_log.append('Не удалось удалить временные параметры запуска.')
            setattr(aboba, name, None)
    refresh_training_state(aboba)


def _finish_research(aboba, exit_code, exit_status, failed_start=False):
    result = getattr(aboba, '_train_research_result', None)
    normal = exit_status == QProcess.ExitStatus.NormalExit
    if result and result.get('status') == 'completed' and not (normal and exit_code == 0):
        result = None
    if result is None:
        status = 'cancelled' if getattr(aboba, '_cancel_requested', False) else 'failed'
        result = {**aboba._research_record, 'status': status, 'finished_at': now(),
                  'total_seconds': time.monotonic() - aboba._run_started_clock,
                  'error_summary': 'CANCELLED' if status == 'cancelled' else 'FAILED_TO_START' if failed_start else 'RUN_FAILED'}
        result = clean_record(result)
        try:
            atomic_json(aboba._experiment_store.directory / result['artifact'], result)
        except OSError:
            aboba.history_notice.setText('Не удалось записать artifact запуска; проверьте доступ к каталогу настроек.')
    _save_history(aboba, result)
    _show_result(aboba, result)
    if hasattr(aboba, 'training_progress'):
        aboba._progress_event.update(stage='finished' if result['status'] == 'completed' else result['status'],
                                      total_seconds=result.get('total_seconds', 0))
        _update_progress(aboba)
    if result['status'] == 'completed':
        set_status_ok(aboba, 'Эксперимент завершён')
    elif result['status'] == 'cancelled':
        set_status_ok(aboba, 'Эксперимент отменён')
    else:
        set_status_error(aboba, 'Эксперимент завершился с ошибкой; проверьте данные и журнал')


def _on_train_error(aboba, error):
    if error == QProcess.ProcessError.FailedToStart and getattr(aboba, '_training_active', False):
        if getattr(aboba, '_run_is_research', False):
            _finish_research(aboba, 1, QProcess.ExitStatus.CrashExit, failed_start=True)
        _release_training(aboba)
        set_status_error(aboba, 'Не удалось запустить обучение')
        aboba.train_log.append('Не удалось запустить процесс обучения модели.')


def _on_train_finished(aboba, exit_code, exit_status):
    if not getattr(aboba, '_training_active', False):
        return
    _on_train_output(aboba)
    if getattr(aboba, '_train_output_buffer', b''):
        _handle_train_line(aboba, aboba._train_output_buffer.decode('utf-8', errors='replace'))
        aboba._train_output_buffer = b''
    if getattr(aboba, '_run_is_research', False):
        _finish_research(aboba, exit_code, exit_status)
        _release_training(aboba)
        return
    result = getattr(aboba, '_train_publication_result', None)
    if (exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0 and result and result.get('published')
            and result.get('postpublish_validation') and not result.get('error_code')):
        set_status_ok(aboba, 'Обучение завершено')
        aboba.train_log.append(f'Обучение успешно завершено. Опубликована generation: {result["published_generation"]}')
    elif exit_code == 130 or getattr(aboba, '_cancel_requested', False):
        set_status_ok(aboba, 'Обучение отменено')
        aboba.train_log.append('Обучение отменено до публикации.')
        if not result:
            _show_result(aboba, {}, research=False)
    else:
        code = (result or getattr(aboba, '_train_preflight_result', None) or {}).get('error_code')
        reasons = {'QUALITY_BLOCK': 'Обучение заблокировано quality gate; причины указаны в журнале',
                   'PREPARATION_FAILED': 'Не удалось подготовить canonical данные; проверьте Mindbox storage и каталог',
                   'PREFLIGHT_CHANGED': 'Canonical batch изменился после preflight; запустите проверку заново',
                   'LOCKED': 'Другой процесс уже выполняет production training',
                   'REPORT_FAILED': 'Не удалось записать отчёт; состояние публикации указано в журнале'}
        message = reasons.get(code, f'Обучение завершилось с ошибкой (код {exit_code})')
        set_status_error(aboba, message)
        aboba.train_log.append(message)
        if not result:
            _show_result(aboba, {}, research=False)
    if hasattr(aboba, 'training_progress'):
        success = result and result.get('published') and result.get('postpublish_validation') and not result.get('error_code')
        aboba._progress_event.update(stage='finished' if success else 'cancelled' if
                                      exit_code == 130 or aboba._cancel_requested else 'failed')
        aboba._progress_event.setdefault('total_seconds', time.monotonic() - aboba._run_started_clock)
        _update_progress(aboba)
    _release_training(aboba)
    schedule_status_reset(aboba, 5)


def _get_store_city_map(aboba) -> dict:
    m = getattr(aboba, "_store_city_map", None)
    if isinstance(m, dict) and m:
        return m

    # запасной вариант: из JSON настроек
    path = os.path.join(os.getcwd(), "user_settings", "filter_settings.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return dict(data.get("store_city_map", {}) or {})

    return {}


def _enrich_orders_with_city_and_weather(
        aboba,
        orders_df: pd.DataFrame,
        weather_path: str,
        diagnostics: list[str] | None = None,
) -> pd.DataFrame:
    df = orders_df.copy()

    def _warn(reason: str) -> None:
        if diagnostics is not None:
            diagnostics.append(
                f"Погода недоступна: {reason}. "
                "Обучение продолжено без погодных данных."
            )

    # --- Магазин -> Город ---
    store_city = _get_store_city_map(aboba)
    if "Магазин" in df.columns:
        df["Магазин"] = df["Магазин"].astype(str).str.strip()
        df["Город"] = df["Магазин"].map(lambda s: store_city.get(s, pd.NA))
    else:
        df["Город"] = pd.NA

    # --- Город + Дата -> Погода ---
    for c in ("ПогодныеУсловия", "СредняяТемпература", "КоличествоОсадков"):
        if c not in df.columns:
            df[c] = pd.NA

    if not os.path.isfile(weather_path):
        _warn("файл Погода.csv отсутствует")
        return df

    try:
        w = pd.read_csv(weather_path, sep="|", encoding="utf-8-sig", dtype=str)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        _warn(f"не удалось прочитать файл Погода.csv ({error})")
        return df

    w.columns = [str(c).replace("\ufeff", "").strip() for c in w.columns]

    duplicate_columns = sorted(set(w.columns[w.columns.duplicated()].tolist()))
    if duplicate_columns:
        _warn(
            "после нормализации обнаружены повторяющиеся колонки: "
            + ", ".join(duplicate_columns)
        )
        return df

    required_weather_columns = {
        "Дата",
        "Город",
        "ПогодныеУсловия",
        "СредняяТемпература",
        "КоличествоОсадков",
    }
    missing_weather_columns = sorted(required_weather_columns.difference(w.columns))
    if missing_weather_columns:
        _warn(
            "в файле Погода.csv отсутствуют колонки: "
            + ", ".join(missing_weather_columns)
        )
        return df

    if w.empty:
        _warn("файл Погода.csv пуст")
        return df

    if "Дата" not in df.columns:
        if len(df):
            diagnostics_message = (
                f"Погодные данные: 0/{len(df)} заказов; "
                f"{len(df)} без совпадения. Обучение продолжено без погодных данных."
            )
            if diagnostics is not None:
                diagnostics.append(diagnostics_message)
        return df

    df["Дата"] = pd.to_datetime(df["Дата"], errors="coerce").dt.normalize()
    w = w[list(required_weather_columns)].copy()
    w["Дата"] = pd.to_datetime(w["Дата"], errors="coerce").dt.normalize()
    w["Город"] = w["Город"].astype("string").str.strip()
    w = w[
        w["Дата"].notna()
        & w["Город"].notna()
        & w["Город"].ne("")
    ].copy()

    if w.empty:
        _warn("файл Погода.csv не содержит корректных ключей Дата/Город")
        return df

    weather_columns = (
        "ПогодныеУсловия",
        "СредняяТемпература",
        "КоличествоОсадков",
    )
    has_weather_value = pd.Series(False, index=w.index)
    for column in weather_columns:
        values = w[column].astype("string").str.strip()
        has_weather_value |= values.notna() & values.ne("")
    if not has_weather_value.any():
        _warn("файл Погода.csv не содержит погодных значений")
        return df

    if w.duplicated(subset=["Дата", "Город"], keep=False).any():
        _warn("в файле Погода.csv обнаружены дубли ключа Дата/Город")
        return df

    rename_columns = {
        column: f"__weather_{column}"
        for column in weather_columns
    }
    w = w.rename(columns=rename_columns)
    w["__weather_match"] = True
    df["Город"] = df["Город"].astype("string").str.strip()

    original = df
    try:
        enriched = df.merge(
            w,
            on=["Дата", "Город"],
            how="left",
            sort=False,
            validate="many_to_one",
        )
    except pd.errors.MergeError as error:
        _warn(f"небезопасное объединение по ключу Дата/Город ({error})")
        return original

    if len(enriched) != len(original):
        _warn("объединение изменило количество заказов")
        return original

    matched = int(enriched.pop("__weather_match").eq(True).sum())
    for column in weather_columns:
        incoming = enriched.pop(rename_columns[column])
        enriched[column] = incoming.combine_first(enriched[column])

    unavailable = len(enriched) - matched
    if unavailable and diagnostics is not None:
        diagnostics.append(
            f"Погодные данные: {matched}/{len(enriched)} заказов; "
            f"{unavailable} без совпадения. "
            "Обучение продолжено без погодных данных для этих заказов."
        )

    return enriched


# -------------------------------------------ФОРМИРУЕМ ИТОГОВЫЙ ДАТАСЕТ ДЛЯ ОБУЧЕНИЯ------------------------------------
def _prepare_training_data_dir(aboba) -> str:

    def _ts() -> str:
        return time.strftime("%d-%m-%Y %H:%M:%S")

    def _n(x: int) -> str:
        return f"{int(x):,}".replace(",", ".")

    def _dbg(tag: str, df: pd.DataFrame, extra: str = "") -> None:
        msg = f"[{_ts()}] [FILTER DEBUG] {tag}: { _n(len(df)) } строк"
        if extra:
            msg += f" | {extra}"

    base_dir = os.path.join(os.getcwd(), INPUT_DATA_DIR.name)

    if not _any_order_filters_set(aboba):
        return INPUT_DATA_DIR.name

    out_rel = "filtered_data"
    out_dir = os.path.join(os.getcwd(), out_rel)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    f = _get_current_order_filters(aboba)

    def _parse_date(s: str):
        if not s:
            return None
        d = pd.to_datetime(s, errors="coerce", dayfirst=True)
        return d if pd.notna(d) else None

    d_from = _parse_date(f["date_from"])
    d_to = _parse_date(f["date_to"])

    # делаем date_to "включительно на весь день", если в данных есть время
    if d_to is not None:
        d_to = d_to.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if d_from is not None:
        d_from = d_from.normalize()

    def _apply_date(df: pd.DataFrame, tag: str = "") -> pd.DataFrame:
        if "Дата" not in df.columns:
            if d_from is not None or d_to is not None:
                return df.iloc[0:0].copy()
            return df

        df = df.copy()
        src = df["Дата"].astype("string")

        # Маски форматов
        m_iso = src.str.match(r"^\d{4}-\d{2}-\d{2}")  # 2025-03-04 или 2025-03-04 00:00:00
        m_dot = src.str.contains(r"\.", regex=True)  # 04.03.2025 или 04.03.2025 00:00:00

        dt = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

        # 1) ISO -> dayfirst=False (иначе месяц/день меняются местами)
        if m_iso.any():
            dt.loc[m_iso] = pd.to_datetime(src.loc[m_iso], errors="coerce", dayfirst=False)

        # 2) dd.mm.yyyy -> dayfirst=True
        if m_dot.any():
            dt.loc[m_dot] = pd.to_datetime(src.loc[m_dot], errors="coerce", dayfirst=True)

        # 3) Остальное (если осталось) -> пробуем обычный парсинг без dayfirst
        rest = dt.isna() & src.notna() & (src.str.strip() != "")
        if rest.any():
            dt.loc[rest] = pd.to_datetime(src.loc[rest], errors="coerce", dayfirst=False)

        # применяем фильтр периода
        if d_from is not None or d_to is not None:
            ok = dt.notna()

            if d_from is not None:
                ok &= (dt >= d_from)
            if d_to is not None:
                ok &= (dt <= d_to)

            df = df[ok]

        # дату оставляем как в исходнике (чтобы не портить формат в filtered_data)
        df["Дата"] = src.loc[df.index]
        return df

    def _apply_kind(df: pd.DataFrame) -> pd.DataFrame:
        kinds = f.get("kinds") or []
        if not kinds or "ВидНоменклатуры" not in df.columns:
            return df
        df = df.copy()
        if f.get("kind_mode") == "Не в группе":
            return df[~df["ВидНоменклатуры"].isin(kinds)]
        return df[df["ВидНоменклатуры"].isin(kinds)]

    def _apply_store(df: pd.DataFrame) -> pd.DataFrame:
        stores = f.get("stores") or []
        if not stores or "Магазин" not in df.columns:
            return df
        df = df.copy()
        if f.get("store_mode") == "Не в группе":
            return df[~df["Магазин"].isin(stores)]
        return df[df["Магазин"].isin(stores)]

    # --- Заказы ---
    p_orders = os.path.join(base_dir, "orders.csv")
    if os.path.isfile(p_orders):
        df = pd.read_csv(p_orders, sep="|", dtype=str)
        _dbg("orders loaded", df)

        df = _apply_date(df, tag="orders")
        _dbg("orders after _apply_date", df)

        df = _apply_kind(df)
        _dbg("orders after _apply_kind", df)

        df = _apply_store(df)
        _dbg("orders after _apply_store", df)

        # полезные доп. метрики (чтобы сравнивать со статистикой)
        try:
            qty_sum = pd.to_numeric(df.get("Количество"), errors="coerce").fillna(0).sum()
            uniq_orders = df.get("НомерЗаказа", pd.Series(dtype=str)).nunique()
            _dbg("orders sanity", df, extra=f"sum(Количество)={_n(qty_sum)}; uniq(НомерЗаказа)={_n(uniq_orders)}")
        except Exception:
            pass

        weather_path = os.path.join(base_dir, "weather.csv")
        before_enrich = len(df)
        weather_diagnostics = []
        df = _enrich_orders_with_city_and_weather(
            aboba,
            df,
            weather_path,
            diagnostics=weather_diagnostics,
        )
        if hasattr(aboba, "train_log"):
            for message in weather_diagnostics:
                aboba.train_log.append(message + "\n")
        _dbg("orders after _enrich_orders_with_city_and_weather", df, extra=f"delta={_n(len(df) - before_enrich)}")

        df.to_csv(os.path.join(out_dir, "orders.csv"), sep="|", index=False)

    # --- Просмотры (дата + вид номенклатуры) ---
    p_views = os.path.join(base_dir, "views.csv")
    if os.path.isfile(p_views):
        df = pd.read_csv(p_views, sep="|", dtype=str)
        _dbg("views loaded", df)

        df = _apply_date(df, tag="views")
        _dbg("views after _apply_date", df)

        df = _apply_kind(df)
        _dbg("views after _apply_kind", df)

        df.to_csv(os.path.join(out_dir, "views.csv"), sep="|", index=False)

    # --- Избранное (дата + вид номенклатуры) ---
    p_favs = os.path.join(base_dir, "favorites.csv")
    if os.path.isfile(p_favs):
        df = pd.read_csv(p_favs, sep="|", dtype=str)
        _dbg("favs loaded", df)

        df = _apply_date(df, tag="favs")
        _dbg("favs after _apply_date", df)

        df = _apply_kind(df)
        _dbg("favs after _apply_kind", df)

        # полезное для сверки
        try:
            uniq_users = df.get("MindboxID", pd.Series(dtype=str)).nunique()
            uniq_items = df.get("КодНоменклатуры", pd.Series(dtype=str)).nunique()
            _dbg("favs sanity", df, extra=f"uniq(MindboxID)={_n(uniq_users)}; uniq(КодНоменклатуры)={_n(uniq_items)}")
        except Exception:
            pass

        df.to_csv(os.path.join(out_dir, "favorites.csv"), sep="|", index=False)

    # --- Справочники (копируем как есть, чтобы тренер не сломался) ---
    for fn in ("nomenclature.csv", "site_categories.csv"):
        src = os.path.join(base_dir, fn)
        dst = os.path.join(out_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    return out_rel


# -------------------------------------------ПОЛУЧАЕМ ТЕКУЩИЕ ЗНАЧЕНИЯ ФИЛЬТРОВ-----------------------------------------
def _any_order_filters_set(aboba) -> bool:
    f = _get_current_order_filters(aboba)
    return bool(f["date_from"] or f["date_to"] or f["kinds"] or f["stores"])


# -------------------------------------------ПОЛУЧАЕМ ТЕКУЩИЕ ЗНАЧЕНИЯ ФИЛЬТРОВ-----------------------------------------
def _get_current_order_filters(aboba) -> dict:
    # даты (учитываем inputMask)
    def _masked_date_is_empty(qle) -> bool:
        if qle is None:
            return True
        t = qle.text()
        if t is None:
            return True
        t = t.replace(" ", "").replace(".", "")
        return t == ""

    date_from = ""
    if hasattr(aboba, "filter_date_from") and not _masked_date_is_empty(aboba.filter_date_from):
        date_from = aboba.filter_date_from.text().strip()

    date_to = ""
    if hasattr(aboba, "filter_date_to") and not _masked_date_is_empty(aboba.filter_date_to):
        date_to = aboba.filter_date_to.text().strip()

    kind_mode = aboba.kind_mode.currentText() if hasattr(aboba, "kind_mode") else "В группе"
    store_mode = aboba.store_mode.currentText() if hasattr(aboba, "store_mode") else "В группе"

    kinds = get_selected_list_values(aboba.filter_kind) if hasattr(aboba, "filter_kind") else []
    stores = get_selected_list_values(aboba.filter_store) if hasattr(aboba, "filter_store") else []

    return {
        "date_from": date_from,
        "date_to": date_to,
        "kind_mode": kind_mode,
        "store_mode": store_mode,
        "kinds": kinds,
        "stores": stores,
    }


