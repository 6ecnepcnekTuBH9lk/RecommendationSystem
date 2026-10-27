"""TRAIN-UI-01: actual Qt widgets with aggregate-only synthetic process events."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QDoubleSpinBox, QSpinBox, QLabel, QPushButton

from Application.tabs import train_model_tab as tab
from Application.tabs import training_workflow as workflow
from Application.evaluation.experiments.gui_history import ExperimentHistory


class Process(QProcess):
    instances = []

    def __init__(self, parent):
        super().__init__(parent)
        self.written = []
        self.killed = False
        self.instances.append(self)

    def start(self):
        self.started.emit()

    def processId(self):
        return os.getpid()

    def write(self, data):
        self.written.append(data)

    def kill(self):
        self.killed = True
        self.finished.emit(9, self.ExitStatus.CrashExit)


@pytest.fixture
def ui(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(tab, 'INPUT_DATA_DIR', tmp_path / 'input_data')
    monkeypatch.setattr(tab, 'USER_SETTINGS_DIR', tmp_path / 'settings')
    monkeypatch.setattr(tab, 'metadata_readiness', lambda *args: (True, 'Synthetic metadata ready', 'a' * 32))
    for name in ('set_status_ok', 'set_status_error', 'set_status_processing', 'schedule_status_reset'):
        monkeypatch.setattr(tab, name, lambda *args: None)
    monkeypatch.setattr(tab, 'QProcess', Process)
    Process.instances.clear()
    window = QWidget()
    window.tabs = QTabWidget(window)
    widget = tab.create_train_model_widgets_tab(window)
    yield window, widget
    window._training_active = False
    window._training_timer.stop()
    window.close()
    window.deleteLater()
    app.processEvents()


def event(window, payload):
    tab._handle_train_line(window, 'TRAINING_EVENT ' + json.dumps(payload))


def finish(window, code=0):
    window.train_proc.finished.emit(code, QProcess.ExitStatus.NormalExit)


def test_four_inputs_default_research_and_removed_controls(ui):
    w, widget = ui
    assert len(widget.findChildren(QDoubleSpinBox)) == 3
    assert len(widget.findChildren(QSpinBox)) == 1
    assert w.training_mode.currentText() == 'Эксперимент'
    assert workflow.input_values(w) == {'w_purchase': 10., 'w_favorite': 2., 'w_view_item': .5, 'epochs': 50}
    removed = ('Количество рекомендаций', 'Стандартные настройки', 'Источник обучения:', 'Признаки номенклатуры:',
               'Скорость обучения', 'Размерность векторов')
    texts = [x.text() for x in widget.findChildren(QLabel) + widget.findChildren(QPushButton)]
    assert not any(s in text for s in removed for text in texts)
    assert not hasattr(w, 'embedding_dim_input') and not hasattr(w, 'top_rec')
    assert w.start_train.isEnabled()
    w.training_mode.setCurrentIndex(1)
    assert w.start_train.text() == 'Обучить рабочую модель'
    assert not w.cancel_train.isEnabled()


@pytest.mark.parametrize('ready,reason', [(False, 'Нет canonical данных'), (False, 'Metadata повреждена'),
                                         (False, 'Temporal benchmark недоступен')])
def test_readiness_block_reason(ui, monkeypatch, ready, reason):
    w, _ = ui
    monkeypatch.setattr(tab, 'metadata_readiness', lambda *args: (ready, reason, None))
    tab.refresh_training_state(w)
    assert not w.start_train.isEnabled() and reason in w.training_reason.text()
    tab.start_training_process(w)
    assert not Process.instances


@pytest.mark.parametrize('name,value', [('w_purchase', 0), ('epochs_input', 0)])
def test_invalid_weights_epochs_disabled(ui, name, value):
    w, _ = ui
    getattr(w, name).setValue(value)
    assert not w.start_train.isEnabled()
    assert w.training_reason.text()


def test_incomplete_keyboard_input_disables_start(ui):
    w, _ = ui
    w.w_purchase.lineEdit().clear()
    w.w_purchase.lineEdit().textEdited.emit('')
    assert not w.start_train.isEnabled()


def test_window_close_waits_for_single_worker_cancellation(ui, monkeypatch):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QCloseEvent
    w, _ = ui
    callbacks = []
    monkeypatch.setattr(QTimer, 'singleShot', lambda ms, fn: callbacks.append(fn))
    tab.start_training_process(w)
    close = QCloseEvent()
    assert w._training_lifecycle.eventFilter(w, close)
    assert close.type() == QEvent.Type.Close and not close.isAccepted()
    callbacks[0]()
    assert not w._training_active


@pytest.mark.parametrize('name,value', [('w_purchase', float('nan')), ('w_view_item', float('inf')),
                                       ('w_favorite', -1), ('epochs_input', 1.5)])
def test_nonfinite_and_fractional_input_rejected(name, value):
    w = SimpleNamespace(**{key: SimpleNamespace(value=lambda v=v: v) for key, v in
                           {'w_purchase': 10, 'w_view_item': .5, 'w_favorite': 2, 'epochs_input': 50}.items()})
    setattr(w, name, SimpleNamespace(value=lambda: value))
    with pytest.raises(ValueError):
        workflow.input_values(w)


def test_research_launch_config_lock_progress_and_completed_history(ui, monkeypatch):
    w, _ = ui
    monkeypatch.setattr(tab.QMessageBox, 'question', lambda *args: pytest.fail('Research needs no confirmation'))
    w.w_view_item.setValue(.25)
    w.epochs_input.setValue(3)
    tab.start_training_process(w)
    args = w.train_proc.arguments()
    assert args[3].endswith('run_gui_research.py')
    assert '--manifest' not in args and '--run-id' in args
    assert w.train_proc.program() == __import__('sys').executable
    config = json.loads(Path(w._train_config_path).read_text(encoding='utf-8'))
    assert config == {'w_purchase': 10., 'w_favorite': 2., 'w_view_item': .25, 'epochs': 3}
    assert not w.training_mode.isEnabled() and not w.w_purchase.isEnabled() and not w.epochs_input.isEnabled()
    assert not w.start_train.isEnabled() and w.cancel_train.isEnabled()
    event(w, {'stage': 'epoch', 'epoch': 3, 'epochs': 3, 'loss': .123456, 'device': 'cuda'})
    assert w.training_progress.value() == 3 and '0,123456' in w.training_progress_text.text()
    record = {**w._research_record, 'status': 'completed', 'finished_at': '2026-10-05T12:00:00+00:00',
              'device': 'cuda', 'total_seconds': 123, 'metrics': {'overall': {'10': {'ndcg': .01234567, 'recall': .03}},
              'VIEW': {'10': {'ndcg': .01}}, 'PURCHASE': {'10': {'ndcg': .02}}}}
    event(w, {'stage': 'research_finished', 'result': record})
    finish(w)
    assert w.training_mode.isEnabled() and w.w_purchase.isEnabled() and w.start_train.isEnabled()
    assert not w.cancel_train.isEnabled()
    assert '0,012346' in w.train_log.toPlainText() and '00:02:03' in w.train_log.toPlainText()
    records = ExperimentHistory(w._experiment_store.directory).read()
    assert len(records) == 1 and records[0]['status'] == 'completed'
    log = w.train_log.toPlainText()
    w.experiment_history.selectRow(0)
    assert w.train_log.toPlainText() == log and 'NDCG@10 (покупки)' in log
    assert not list(tab.USER_SETTINGS_DIR.glob('train-run-*'))


@pytest.mark.parametrize('mode', [0, 1])
def test_cancel_kills_single_process_before_publication_and_unlocks(ui, monkeypatch, mode):
    w, _ = ui
    w.training_mode.setCurrentIndex(mode)
    monkeypatch.setattr(tab, '_confirm_production', lambda *args: True)
    callbacks = []
    monkeypatch.setattr(QTimer, 'singleShot', lambda ms, fn: callbacks.append(fn))
    tab.start_training_process(w)
    config = w._train_config_path
    tab.cancel_training(w)
    assert not w.cancel_train.isEnabled() and w._cancel_requested
    assert w._cancel_path.exists()
    if mode:
        event(w, {'stage': 'publication_ready'})
        assert b'PUBLISH\n' not in w.train_proc.written
    callbacks[0]()
    assert w.train_proc.killed and not w._training_active and w.start_train.isEnabled()
    assert not Path(config).exists()
    records = w._experiment_store.read()
    assert len(records) == (1 if mode == 0 else 0)
    if records:
        assert records[0]['status'] == 'cancelled'
        assert json.loads((w._experiment_store.directory / records[0]['artifact']).read_text())['status'] == 'cancelled'


def test_publication_barrier_disables_cancel_before_ack(ui, monkeypatch):
    w, _ = ui
    w.training_mode.setCurrentIndex(1)
    monkeypatch.setattr(tab, '_confirm_production', lambda *args: True)
    tab.start_training_process(w)
    event(w, {'stage': 'publication_ready'})
    assert w._publishing and not w.cancel_train.isEnabled()
    assert w.train_proc.written == [b'PUBLISH\n']
    tab.cancel_training(w)
    assert not w._cancel_requested
    event(w, {'stage': 'finished', 'published': True, 'postpublish_validation': True,
              'published_generation': 'b' * 32, 'training_metrics': {'epochs_completed': 2}})
    finish(w)
    assert not w._experiment_store.read() and 'Поколение:' in w.train_log.toPlainText()
    assert 'NDCG' not in w.train_log.toPlainText()


def test_production_confirmation_and_weights_preserve_hidden_settings(ui, monkeypatch):
    w, _ = ui
    w.training_mode.setCurrentIndex(1)
    path = tab.USER_SETTINGS_DIR / 'train_config.json'
    path.parent.mkdir()
    original = {'early_stop': True, 'use_item_features': True, 'embedding_dim': 64, 'feature_scale': .3}
    path.write_text(json.dumps(original), encoding='utf-8')
    dialogs = []
    def confirm(*args):
        dialogs.append(args)
        return tab.QMessageBox.StandardButton.No if len(dialogs) == 1 else tab.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(tab.QMessageBox, 'question', confirm)
    tab.start_training_process(w)
    assert not Process.instances
    tab.start_training_process(w)
    assert '50' in dialogs[0][2] and 'поколение' in dialogs[0][2] and '10' in dialogs[0][2]
    cfg = json.loads(Path(w._train_config_path).read_text())
    assert cfg == {**original, **workflow.input_values(w), 'data_dir': str(tab.INPUT_DATA_DIR)}
    assert json.loads(path.read_text()) == original
    assert w.train_proc.arguments()[3].endswith('mindbox_production_train.py')
    finish(w, 1)
    assert not w._experiment_store.read()


@pytest.mark.parametrize('failed_start', [True, False])
def test_failed_run_retained(ui, failed_start):
    w, _ = ui
    tab.start_training_process(w)
    if failed_start:
        w.train_proc.errorOccurred.emit(QProcess.ProcessError.FailedToStart)
    else:
        finish(w, 1)
    assert w._experiment_store.read()[0]['status'] == 'failed'
    assert w.w_view_item.isEnabled() and w.start_train.isEnabled()


def test_corrupt_history_kept_and_tab_production_usable(ui, monkeypatch):
    w, _ = ui
    path = w._experiment_store.path
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{broken-history')
    tab._load_history(w, recover=True)
    assert w.history_notice.text() and w.start_train.isEnabled()
    tab.start_training_process(w)
    finish(w, 1)
    assert path.read_bytes() == b'{broken-history'
    assert list((path.parent / 'runs').glob('*/result.json'))
    w.training_mode.setCurrentIndex(1)
    monkeypatch.setattr(tab, '_confirm_production', lambda *args: True)
    tab.start_training_process(w)
    finish(w, 1)
    assert path.read_bytes() == b'{broken-history'


def test_preflight_failure_blocks_until_revision_changes(ui, monkeypatch):
    w, _ = ui
    w.training_mode.setCurrentIndex(1)
    monkeypatch.setattr(tab, '_confirm_production', lambda *args: True)
    tab.start_training_process(w)
    event(w, {'stage': 'preflight', 'error_code': 'QUALITY_BLOCK', 'quality': None})
    finish(w, 1)
    assert not w.start_train.isEnabled() and 'заблокировала' in w.training_reason.text()
    monkeypatch.setattr(tab, 'metadata_readiness', lambda *args: (True, 'Updated data', 'b' * 32))
    tab.refresh_training_state(w)
    assert w.start_train.isEnabled()


def test_benchmark_failure_stays_blocked_when_weights_change(ui, monkeypatch):
    w, _ = ui
    tab.start_training_process(w)
    event(w, {'stage': 'research_finished', 'result': {**w._research_record, 'status': 'failed',
                                                      'error_summary': 'BENCHMARK_UNAVAILABLE'}})
    finish(w, 1)
    assert not w.start_train.isEnabled() and 'валидации недоступен' in w.training_reason.text()
    w.w_view_item.setValue(1.)
    assert not w.start_train.isEnabled()
    monkeypatch.setattr(tab, 'metadata_readiness', lambda *args: (True, 'Updated data', 'b' * 32))
    tab.refresh_training_state(w)
    assert w.start_train.isEnabled()


def test_progress_is_persisted_for_interrupted_recovery(ui):
    w, _ = ui
    tab.start_training_process(w)
    event(w, {'stage': 'epoch', 'epoch': 2, 'epochs': 50, 'loss': .1, 'device': 'cpu'})
    assert w._experiment_store.read()[0]['epochs_completed'] == 2
    finish(w, 1)
    assert w._experiment_store.read()[0]['status'] == 'failed'


def test_opening_tab_never_ingests_canonical_events(ui, monkeypatch):
    w, _ = ui
    from Application.evaluation import audit
    monkeypatch.setattr(audit, 'load_canonical_events', lambda *args, **kwargs: pytest.fail('Ingestion on tab open'))
    tab.refresh_training_state(w)
    tab._load_history(w)
    assert w.start_train.isEnabled()


def test_metadata_missing_corrupt_and_research_unavailable(tmp_path):
    from datetime import datetime, timedelta, timezone
    from Application.mindbox import canonical_storage as storage
    assert not workflow.metadata_readiness(tmp_path, False)[0]
    (tmp_path / 'nomenclature.csv').write_text('synthetic catalog')
    assert not workflow.metadata_readiness(tmp_path, False)[0]
    raw = tmp_path / 'MindboxRaw'
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for name, key in [('customer_merges', 'customerMerges'), ('actions', 'customerActions'), ('orders', 'orders')]:
        directory = raw / 'staged' / name
        directory.mkdir(parents=True)
        (directory / f'{name}_part_001.json').write_text(json.dumps({key: []}))
        storage.publish(raw, name, since, since + timedelta(days=1), directory)
    assert workflow.metadata_readiness(tmp_path, False)[0]
    assert not workflow.metadata_readiness(tmp_path, True)[0]
    (raw / 'canonical/catalog.json').write_text('{corrupt')
    assert 'ошибкой' in workflow.metadata_readiness(tmp_path, False)[1]


def test_ui02_layout_fields_and_centered_presentation(ui):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QFrame, QFormLayout, QVBoxLayout, QHeaderView
    w, widget = ui
    assert len([f for f in widget.findChildren(QFrame) if f.objectName() == 'vSeparator']) == 1
    assert len([f for f in widget.findChildren(QFrame) if f.objectName() == 'hSeparator']) == 1
    headers = [label for label in widget.findChildren(QLabel) if label.property('class') == 'sectionHeader']
    assert [label.text() for label in headers] == ['Параметры', 'Данные для обучения', 'Процесс обучения', 'История экспериментов']
    assert all(label.alignment() == Qt.AlignmentFlag.AlignCenter for label in headers)
    layouts = widget.findChildren(QVBoxLayout)
    for label in headers:
        owner = next(layout for layout in layouts if layout.indexOf(label) >= 0)
        assert owner.itemAt(owner.indexOf(label)).alignment() & Qt.AlignmentFlag.AlignHCenter
    form = widget.findChild(QFormLayout)
    assert [form.itemAt(i, form.ItemRole.LabelRole).widget().text() for i in range(5)] == [
        'Режим обучения:', 'Количество эпох:', 'Вес покупки:', 'Вес избранного:', 'Вес просмотра:']
    assert [form.itemAt(i, form.ItemRole.FieldRole).widget() for i in range(5)] == [
        w.training_mode, w.epochs_input, w.w_purchase, w.w_favorite, w.w_view_item]
    assert [control.text() for control in (w.w_purchase, w.w_favorite, w.w_view_item)] == ['10,00', '2,00', '0,50']
    assert all(control.decimals() == 2 for control in widget.findChildren(QDoubleSpinBox))
    assert not hasattr(w, 'training_result') and not hasattr(w, 'mode_description')
    assert not any('benchmark' in label.text() or label.text() == 'Результат' for label in widget.findChildren(QLabel))
    assert w.training_progress_text.alignment() == Qt.AlignmentFlag.AlignCenter
    history_layout = next(layout for layout in layouts if layout.indexOf(w.experiment_history) >= 0)
    assert history_layout.spacing() <= 4 and w.history_notice.isHidden()
    table = w.experiment_history
    assert [table.horizontalHeaderItem(i).text() for i in range(12)] == [
        'Дата', 'Просмотр', 'Избранное', 'Покупка', 'Эпохи', 'NDCG@10', 'Recall@10',
        'NDCG@10 (просмотры)', 'NDCG@10 (покупки)', 'Время', 'Устройство', 'Статус']
    assert all(table.horizontalHeader().sectionResizeMode(i) == QHeaderView.ResizeMode.Interactive for i in range(12))
    assert not table.horizontalHeader().stretchLastSection()
    assert table.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded


@pytest.mark.parametrize('status,text', [('running', 'Выполняется'), ('completed', 'Завершено'),
                                        ('failed', 'Ошибка'), ('cancelled', 'Отменено'), ('interrupted', 'Прервано')])
def test_ui02_history_v1_precision_localization_and_resize(ui, status, text):
    w, _ = ui
    record = {'run_id': 'f' * 32, 'status': status, 'started_at': '2026-10-05T10:00:00+00:00',
              'hyperparameters': {'w_view_item': .123456789, 'w_purchase': 10.123456789, 'w_favorite': 2.123456789},
              'device': 'cuda', 'metrics': {'overall': {'10': {'ndcg': .0123456789, 'cases': 98765}}}}
    tab._save_history(w, record)
    original = w._experiment_store.path.read_bytes()
    data = json.loads(original)
    assert data['schema_version'] == 1 and data['runs'][0]['status'] == status
    assert data['runs'][0]['hyperparameters'] == record['hyperparameters']
    assert data['runs'][0]['metrics'] == record['metrics']
    w.experiment_history.setColumnWidth(0, 233)
    w.experiment_history.setColumnWidth(11, 177)
    w.train_log.setPlainText('Журнал текущего запуска')
    tab._load_history(w)
    w.experiment_history.selectRow(0)
    assert w._experiment_store.path.read_bytes() == original
    assert w.experiment_history.columnWidth(0) == 233 and w.experiment_history.columnWidth(11) == 177
    assert w.experiment_history.item(0, 11).text() == text and w.experiment_history.item(0, 10).text() == 'CUDA'
    assert w.experiment_history.item(0, 1).text() == '0,12' and w.experiment_history.item(0, 5).text() == '0,012346'
    assert w.train_log.toPlainText() == 'Журнал текущего запуска'


@pytest.mark.parametrize('ready,reason,expected', [(True, 'Ready', 'Данные готовы'),
                         (False, 'Нет каталога товаров', 'Данные для обучения отсутствуют'),
                         (False, 'Проверка metadata завершилась с ошибкой', 'Ошибка проверки данных')])
def test_ui02_metadata_fallback_and_blocking(ui, monkeypatch, ready, reason, expected):
    w, _ = ui
    from Application.evaluation import audit
    from Application.model import mindbox_production_training as production
    monkeypatch.setattr(audit, 'load_canonical_events', lambda *a, **k: pytest.fail('UI must not ingest'))
    monkeypatch.setattr(production, 'preflight_production_training', lambda *a, **k: pytest.fail('UI must not prepare'))
    monkeypatch.setattr(tab, 'metadata_readiness', lambda *a: (ready, reason, 'b' * 32))
    tab.refresh_training_state(w)
    assert w.training_readiness.text() == expected
    assert w.training_data_counts.text() == 'Взаимодействий: —\nПользователей: —\nТоваров: —'
    assert w.start_train.isEnabled() == ready


@pytest.mark.parametrize('mode', [0, 1])
def test_ui02_exact_prepared_counts_retained_without_validation_cases(ui, monkeypatch, mode):
    w, widget = ui
    w.training_mode.setCurrentIndex(mode)
    monkeypatch.setattr(tab, '_confirm_production', lambda *a: True)
    tab.start_training_process(w)
    dataset = {'bpr_events': 12034, 'users': 123, 'items': 456, 'train_pairs': 987, 'eval_events': 98765}
    research_counts = {'training_events': 12034, 'training_users': 123, 'training_items': 456, 'training_pairs': 987}
    if mode:
        event(w, {'stage': 'preflight', 'dataset': dataset,
                  'quality': {'level': 'PASS', 'training_allowed': True, 'metrics': {'eval_events': 98765, 'cases': 98765}, 'issues': []}})
        assert w.training_readiness.text() == 'Данные готовы' and '12 034' in w.training_data_counts.text()
    event(w, {'stage': 'training', 'device': 'cuda', **({'dataset': dataset} if mode else research_counts),
              'validation_cases': 98765})
    expected = 'Взаимодействий: 12 034\nПользователей: 123\nТоваров: 456\nОбучающих пар: 987'
    assert w.training_readiness.text() == 'Данные используются в обучении'
    assert w.training_data_counts.text() == expected
    event(w, {'stage': 'epoch', 'epoch': 50, 'epochs': 50, 'loss': .045248, 'device': 'cuda'})
    event(w, {'stage': 'validation', 'device': 'cuda', 'validation_cases': 98765})
    assert w.training_data_counts.text() == expected
    if mode:
        event(w, {'stage': 'finished', 'published': True, 'postpublish_validation': True,
                  'published_generation': 'b' * 32, 'dataset': dataset, 'training_metrics': {'epochs_completed': 50}})
    else:
        event(w, {'stage': 'research_finished', 'result': {**w._research_record, 'status': 'completed',
                  'device': 'cuda', 'metrics': {'overall': {'10': {'ndcg': .01, 'recall': .02, 'cases': 98765}}}}})
    finish(w)
    assert w.training_readiness.text() == 'Данные использованы'
    assert w.training_data_counts.text() == expected
    tab.refresh_training_state(w)
    assert w.training_data_counts.text() == expected and w.training_readiness.text() == 'Данные использованы'
    all_text = '\n'.join(label.text() for label in widget.findChildren(QLabel)) + w.train_log.toPlainText()
    assert not any(text in all_text for text in ('Validation cases', 'validation_cases', '98765', 'eval_events'))
    assert ('NDCG@10' in w.train_log.toPlainText()) == (mode == 0)
    assert ('Публикация модели: успешно' in w.train_log.toPlainText()) == (mode == 1)
    w.training_mode.setCurrentIndex(1 - mode)
    assert w.training_data_counts.text().count('—') == 3
    w.training_mode.setCurrentIndex(mode)
    assert w.training_data_counts.text() == expected
    monkeypatch.setattr(tab, 'metadata_readiness', lambda *a: (True, 'Changed dataset', 'c' * 32))
    tab.refresh_training_state(w)
    assert w.training_data_counts.text().count('—') == 3


@pytest.mark.parametrize('stage,expected', [('loading', 'Загрузка данных'), ('preparation', 'Подготовка данных'),
                    ('training', 'Обучение'), ('epoch', 'Эпоха 14'), ('validation', 'Оценка модели')])
def test_ui02_log_and_progress_localization(ui, stage, expected):
    w, _ = ui
    tab.start_training_process(w)
    event(w, {'stage': stage, 'epoch': 14, 'epochs': 50, 'loss': .045248, 'device': 'cuda', 'total_seconds': 1292})
    assert expected in w.train_log.toPlainText()
    assert 'Время: 00:21:32 · CUDA' in w.training_progress_text.text()
    if stage == 'epoch':
        assert w.training_progress_text.text() == 'Эпоха 14 из 50 · Ошибка: 0,045248 · Время: 00:21:32 · CUDA'
    assert not any(s in w.train_log.toPlainText() + w.training_progress_text.text() for s in ('Loss', 'Validation', 'canonical'))
    finish(w, 1)


@pytest.mark.parametrize('status,expected', [('completed', 'Эксперимент завершён.'),
                    ('failed', 'Эксперимент завершился с ошибкой.'), ('cancelled', 'Эксперимент отменён.')])
def test_ui02_single_final_summary_and_new_run_log(ui, status, expected):
    w, _ = ui
    w.train_log.setPlainText('Предыдущий журнал')
    tab.start_training_process(w)
    assert w.train_log.toPlainText() == 'Новый эксперимент'
    event(w, {'stage': 'epoch', 'epoch': 50, 'epochs': 50, 'loss': .1, 'device': 'cuda'})
    event(w, {'stage': 'research_finished', 'result': {**w._research_record, 'status': status,
              'total_seconds': 4354, 'device': 'cuda', 'metrics': {'overall': {'10': {'ndcg': .00984, 'recall': .018891}}}}})
    finish(w, 0 if status == 'completed' else 1)
    assert w.train_log.toPlainText().count(expected) == 1
    assert 'Время: 01:12:34\nУстройство: CUDA' in w.train_log.toPlainText()
    assert w.training_progress_text.text() == f'{tab._STATUS_TEXT[status]} · 50 из 50 эпох · Время: 01:12:34 · CUDA'


def test_ui02_new_configuration_drops_previous_counts(ui):
    w, _ = ui
    tab._receive_data_counts(w, {'training_events': 100, 'training_users': 10, 'training_items': 20, 'training_pairs': 50}, 'used')
    assert 'Обучающих пар: 50' in w.training_data_counts.text()
    tab.start_training_process(w)
    assert w.training_readiness.text() == 'Данные готовы' and w.training_data_counts.text().count('—') == 3
    finish(w, 1)


def test_ui02_opening_with_real_metadata_never_reads_payloads(ui, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from Application.mindbox import canonical_storage as storage
    from Application.evaluation import audit
    data = tab.INPUT_DATA_DIR
    data.mkdir()
    (data / 'nomenclature.csv').write_text('synthetic catalog', encoding='utf-8')
    raw = data / 'MindboxRaw'
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for name, key in [('customer_merges', 'customerMerges'), ('actions', 'customerActions'), ('orders', 'orders')]:
        directory = raw / 'staged' / name
        directory.mkdir(parents=True)
        (directory / f'{name}_part_001.json').write_text(json.dumps({key: []}), encoding='utf-8')
        storage.publish(raw, name, since, since + timedelta(days=1), directory)
    original_open = Path.open
    def metadata_only(path, *args, **kwargs):
        if path.is_relative_to(raw / 'canonical/objects') or path == data / 'nomenclature.csv':
            pytest.fail('Opening the tab read raw payload/catalog rows')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', metadata_only)
    monkeypatch.setattr(audit, 'load_canonical_events', lambda *a, **k: pytest.fail('Unexpected ingestion'))
    monkeypatch.setattr(tab, 'metadata_readiness', workflow.metadata_readiness)
    w = QWidget()
    w.tabs = QTabWidget(w)
    try:
        tab.create_train_model_widgets_tab(w)
        assert not w.start_train.isEnabled()  # January data cannot satisfy November research.
        w.training_mode.setCurrentIndex(1)
        assert w.training_readiness.text() == 'Данные готовы' and w.start_train.isEnabled()
        assert w.training_data_counts.text().count('—') == 3
    finally:
        w.close()
        w.deleteLater()
