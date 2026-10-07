"""Four synthetic Qt previews. Fake QProcess never starts a child or training."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PyQt6.QtCore import QProcess, QTimer
from PyQt6.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel
from Application.tabs import train_model_tab as tab
from Application.theme.apply_theme import apply_app_theme
from Application.settings.set_status import set_ready_status


class FakeProcess(QProcess):
    def start(self):
        self.started.emit()

    def processId(self):
        return 0

    def write(self, data):
        return len(data)


def emit(window, **payload):
    tab._handle_train_line(window, 'TRAINING_EVENT ' + json.dumps(payload))


def screenshot(window, name):
    app.processEvents()
    from PyQt6.QtWidgets import QStyle, QStyleOptionButton
    def rectangle(rect):
        return [rect.x(), rect.y(), rect.width(), rect.height()]
    buttons = {}
    for key, button in (('start', window.start_train), ('cancel', window.cancel_train)):
        option = QStyleOptionButton()
        button.initStyleOption(option)
        buttons[key] = {'geometry': rectangle(button.geometry()),
                        'size_hint': [button.sizeHint().width(), button.sizeHint().height()],
                        'margins': [button.contentsMargins().top(), button.contentsMargins().bottom()],
                        'contents': rectangle(button.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, button)),
                        'enabled': button.isEnabled()}
    charts = {}
    for key, chart in (('loss', window.training_loss_chart), ('metrics', window.training_metric_chart)):
        charts[key] = {'plot': rectangle(chart.chart().plotArea()),
                       'placeholder': rectangle(chart.placeholder.geometry()),
                       'viewport': rectangle(chart.viewport().rect()),
                       'tick_interval': chart.x_axis.tickInterval()}
    (ROOT / 'diagnostics/train_ui04/before' / (name + '.json')).write_text(
        json.dumps({'buttons': buttons, 'charts': charts}, indent=2), encoding='utf-8')
    window.train_log.verticalScrollBar().setValue(window.train_log.verticalScrollBar().maximum())
    window.grab().save(str(ROOT / 'diagnostics/train_ui04/before' / name))


app = QApplication([])
apply_app_theme(app, False)
with tempfile.TemporaryDirectory(prefix='train-ui03-preview-') as temporary:
    tab.USER_SETTINGS_DIR = Path(temporary)
    tab.metadata_readiness = lambda *args: (True, 'Данные готовы', 'a' * 32)
    tab.QProcess = FakeProcess
    tab._confirm_production = lambda *args: True
    window = QWidget()
    window.setWindowTitle('TRAIN-UI-03 — synthetic preview; no training')
    layout = QVBoxLayout(window)
    window.tabs = QTabWidget()
    layout.addWidget(window.tabs)
    window.status_icon, window.status_label = QLabel(), QLabel('Готов к работе')
    status = QHBoxLayout()
    status.addWidget(window.status_icon)
    status.addWidget(window.status_label, 1)
    layout.addLayout(status)
    window._status_reset_timer = QTimer(window)
    window._status_reset_timer.setSingleShot(True)
    window._status_reset_timer.timeout.connect(lambda: set_ready_status(window))
    tab.create_train_model_widgets_tab(window)
    for index, outcome in enumerate(('completed', 'completed', 'failed', 'cancelled')):
        record = {'run_id': f'{index+1:032x}', 'status': outcome, 'started_at': f'2026-10-05T{index+10}:00:00+00:00',
                  'epochs_requested': 50, 'epochs_completed': 50 if outcome == 'completed' else 14,
                  'device': 'cuda', 'total_seconds': 1292, 'hyperparameters': {
                      'w_view_item': .5, 'w_favorite': 2., 'w_purchase': 10., 'epochs': 50},
                  'metrics': {} if outcome != 'completed' else {'overall': {'10': {'ndcg': .00984, 'recall': .018891}},
                      'VIEW': {'10': {'ndcg': .003012}}, 'PURCHASE': {'10': {'ndcg': .022791}}}}
        tab._save_history(window, record)
    window.train_log.setPlainText('Синтетический preview — обучение не запускалось.')
    window.resize(1520, 1020)
    window.show()
    screenshot(window, 'research_idle.png')
    tab.start_training_process(window)
    window.train_log.append('Синтетические данные — обучение не запускалось.')
    emit(window, stage='loading', device='cuda')
    emit(window, stage='preparation', device='cuda')
    emit(window, stage='training', device='cuda', training_events=12034, training_users=123, training_items=456, training_pairs=987)
    checkpoints = {1: (.002, .005), 5: (.004, .008), 10: (.006, .012), 20: (.008, .015), 30: (.009, .017), 40: (.0095, .018)}
    for epoch in range(1, 51):
        emit(window, stage='epoch', epoch=epoch, epochs=50, loss=.7 / (1 + epoch / 15), device='cuda', total_seconds=epoch*26)
        if epoch in checkpoints:
            emit(window, stage='validation', epoch=epoch, device='cuda')
            ndcg, recall = checkpoints[epoch]
            emit(window, stage='validation_checkpoint', epoch=epoch, ndcg=ndcg, recall=recall, device='cuda')
        if epoch == 14:
            screenshot(window, 'research_running.png')
    emit(window, stage='validation', epoch=50, device='cuda')
    emit(window, stage='validation_checkpoint', epoch=50, ndcg=.00984, recall=.018891, device='cuda')
    result = {**window._research_record, 'status': 'completed', 'total_seconds': 1356, 'device': 'cuda',
              'metrics': {'overall': {'10': {'ndcg': .00984, 'recall': .018891}},
                          'VIEW': {'10': {'ndcg': .003012}}, 'PURCHASE': {'10': {'ndcg': .022791}}}}
    emit(window, stage='research_finished', result=result)
    window.train_proc.finished.emit(0, QProcess.ExitStatus.NormalExit)
    screenshot(window, 'research_completed.png')
    window.training_mode.setCurrentIndex(1)
    tab.start_training_process(window)
    window.train_log.append('Синтетические данные — обучение не запускалось.')
    emit(window, stage='loading', device='cuda')
    emit(window, stage='preparation', device='cuda')
    emit(window, stage='training', device='cuda', dataset={'bpr_events': 16480, 'users': 158, 'items': 512, 'train_pairs': 1320})
    for epoch in range(1, 51):
        emit(window, stage='epoch', epoch=epoch, epochs=50, loss=.8 / (1 + epoch / 12), device='cuda', total_seconds=epoch*24)
    emit(window, stage='publication_ready')
    emit(window, stage='publication', device='cuda')
    emit(window, stage='timing', device='cuda', total_seconds=1218)
    emit(window, stage='finished', published=True, postpublish_validation=True, published_generation='b' * 32,
         training_metrics={'epochs_completed': 50})
    window.train_proc.finished.emit(0, QProcess.ExitStatus.NormalExit)
    screenshot(window, 'production.png')
    window.close()
print('Saved four synthetic previews; no child process/training/model writes')
