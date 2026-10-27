"""Render synthetic TRAIN-UI-01 widget states; no training or user settings writes."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout
from Application.tabs import train_model_tab as tab
from Application.theme.apply_theme import apply_app_theme

app = QApplication([])
apply_app_theme(app, False)
with tempfile.TemporaryDirectory(prefix='train-ui-preview-') as temporary:
    tab.USER_SETTINGS_DIR = Path(temporary)
    tab.metadata_readiness = lambda *args: (True, 'Данные готовы · полная проверка перед обучением', 'a' * 32)
    window = QWidget()
    window.setWindowTitle('TRAIN-UI-01 preview')
    layout = QVBoxLayout(window)
    window.tabs = QTabWidget()
    layout.addWidget(window.tabs)
    tab.create_train_model_widgets_tab(window)
    for index in range(6):
        record = {'run_id': f'{index+1:032x}', 'status': ['completed', 'completed', 'cancelled'][index % 3],
                  'started_at': f'2026-10-05T{index+10}:00:00+00:00', 'epochs_completed': 30 if index % 3 != 2 else 3,
                  'epochs_requested': 30, 'device': 'cuda', 'total_seconds': 2733,
                  'hyperparameters': {'w_view_item': [.1, .5, 1.][index % 3], 'w_favorite': 2., 'w_purchase': 10., 'epochs': 30},
                  'metrics': {} if index % 3 == 2 else {'overall': {'10': {'ndcg': .0098 + index*.00005, 'recall': .0189}},
                  'VIEW': {'10': {'ndcg': .00867}}, 'PURCHASE': {'10': {'ndcg': .012345}}}}
        tab._save_history(window, record)
    window.history_notice.setText('Синтетический preview интерфейса; эти результаты не являются экспериментом.')
    tab._show_result(window, window._history_records[1])
    window._progress_event = {'stage': 'finished', 'epoch': 30, 'epochs': 30, 'loss': .121345, 'device': 'cuda', 'total_seconds': 2733}
    window._run_started_clock = time.monotonic()
    window.training_progress.setRange(0, 30)
    tab._update_progress(window)
    window.train_log.setPlainText('Загрузка canonical данных…\nПодготовка validation history…\nОбучение…\n'
                                  'Эпоха 29: loss=0.124563\nЭпоха 30: loss=0.121345\nОценка validation…\nЭксперимент завершён.')
    window.resize(1360, 820)
    window.show()
    app.processEvents()
    window.grab().save(str(ROOT / 'diagnostics/train_ui01/research_preview.png'))
    window.training_mode.setCurrentIndex(1)
    tab._show_result(window, {'published': True, 'postpublish_validation': True, 'published_generation': 'b' * 32,
                            'training_metrics': {'epochs_completed': 30}}, research=False)
    app.processEvents()
    window.grab().save(str(ROOT / 'diagnostics/train_ui01/production_preview.png'))
    window.close()
