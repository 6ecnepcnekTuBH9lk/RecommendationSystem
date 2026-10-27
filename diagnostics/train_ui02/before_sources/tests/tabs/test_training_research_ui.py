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
    assert w.training_progress.value() == 3 and '0.123456' in w.training_progress_text.text()
    record = {**w._research_record, 'status': 'completed', 'finished_at': '2026-10-05T12:00:00+00:00',
              'device': 'cuda', 'total_seconds': 123, 'metrics': {'overall': {'10': {'ndcg': .01234567, 'recall': .03}},
              'VIEW': {'10': {'ndcg': .01}}, 'PURCHASE': {'10': {'ndcg': .02}}}}
    event(w, {'stage': 'research_finished', 'result': record})
    finish(w)
    assert w.training_mode.isEnabled() and w.w_purchase.isEnabled() and w.start_train.isEnabled()
    assert not w.cancel_train.isEnabled()
    assert '0.012346' in w.training_result.text() and '00:02:03' in w.training_result.text()
    records = ExperimentHistory(w._experiment_store.directory).read()
    assert len(records) == 1 and records[0]['status'] == 'completed'
    w.experiment_history.selectRow(0)
    assert 'PURCHASE NDCG@10' in w.training_result.text()
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
    assert not w._experiment_store.read() and 'Generation:' in w.training_result.text()
    assert 'NDCG' not in w.training_result.text()


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
    assert '50' in dialogs[0][2] and 'generation' in dialogs[0][2] and '10' in dialogs[0][2]
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
    assert not w.start_train.isEnabled() and 'benchmark недоступен' in w.training_reason.text()
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
