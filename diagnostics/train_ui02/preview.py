"""Render synthetic Qt states; no subprocess, ingestion, trainer or user writes."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout
from Application.tabs import train_model_tab as tab
from Application.theme.apply_theme import apply_app_theme


def emit(window, **payload):
    tab._handle_train_line(window, 'TRAINING_EVENT ' + json.dumps(payload))


app = QApplication([])
apply_app_theme(app, False)
with tempfile.TemporaryDirectory(prefix='train-ui02-preview-') as temporary:
    tab.USER_SETTINGS_DIR = Path(temporary)
    tab.metadata_readiness = lambda *args: (True, 'Данные готовы', 'a' * 32)
    for name in ('set_status_ok', 'set_status_error', 'set_status_processing', 'schedule_status_reset'):
        setattr(tab, name, lambda *args: None)
    window = QWidget()
    window.setWindowTitle('TRAIN-UI-02 — синтетический preview')
    layout = QVBoxLayout(window)
    window.tabs = QTabWidget()
    layout.addWidget(window.tabs)
    tab.create_train_model_widgets_tab(window)
    for index in range(5):
        record = {'run_id': f'{index+1:032x}', 'status': ['completed', 'failed', 'cancelled', 'interrupted', 'completed'][index],
                  'started_at': f'2026-10-05T{index+10}:00:00+00:00', 'epochs_completed': 50 if index in (0, 4) else 14,
                  'epochs_requested': 50, 'device': 'cuda', 'total_seconds': 1292,
                  'hyperparameters': {'w_view_item': .5, 'w_favorite': 2., 'w_purchase': 10., 'epochs': 50},
                  'metrics': {} if index not in (0, 4) else {'overall': {'10': {'ndcg': .00984, 'recall': .018891}},
                  'VIEW': {'10': {'ndcg': .003012}}, 'PURCHASE': {'10': {'ndcg': .022791}}}}
        tab._save_history(window, record)
    window._research_record = record
    window._run_is_research = True
    window._training_active = True
    window._cancel_requested = False
    window._publishing = False
    window._run_started_clock = time.monotonic()
    window._progress_event = {}
    window.train_proc = SimpleNamespace(readAllStandardOutput=lambda: b'')
    window.train_log.setPlainText('Синтетический preview — обучение не запускалось.\nНовый эксперимент')
    emit(window, stage='loading', device='cuda')
    emit(window, stage='preparation', device='cuda')
    emit(window, stage='training', device='cuda', training_events=12034, training_users=123, training_items=456, training_pairs=987)
    emit(window, stage='epoch', epoch=50, epochs=50, loss=.045248, device='cuda')
    emit(window, stage='validation', device='cuda')
    emit(window, stage='research_finished', result=record)
    tab._on_train_finished(window, 0, QProcess.ExitStatus.NormalExit)
    window.resize(1520, 860)
    window.show()
    app.processEvents()
    window.grab().save(str(ROOT / 'diagnostics/train_ui02/research_preview.png'))

    window.training_mode.setCurrentIndex(1)
    window._run_is_research = False
    window._training_active = True
    window._train_output_buffer = b''
    window._run_started_clock = time.monotonic()
    window._progress_event = {}
    window.train_log.setPlainText('Синтетический preview — обучение не запускалось.\nНовое обучение рабочей модели')
    emit(window, stage='loading', device='cuda')
    emit(window, stage='preparation', device='cuda')
    emit(window, stage='training', device='cuda', dataset={'bpr_events': 16480, 'users': 158, 'items': 512, 'train_pairs': 1320})
    emit(window, stage='epoch', epoch=50, epochs=50, loss=.045248, device='cuda')
    emit(window, stage='publication', device='cuda')
    emit(window, stage='timing', device='cuda', total_seconds=1356)
    emit(window, stage='finished', published=True, postpublish_validation=True, published_generation='b' * 32,
         training_metrics={'epochs_completed': 50})
    tab._on_train_finished(window, 0, QProcess.ExitStatus.NormalExit)
    app.processEvents()
    window.grab().save(str(ROOT / 'diagnostics/train_ui02/production_preview.png'))
    window.close()
print('Saved two synthetic previews in diagnostics/train_ui02')
