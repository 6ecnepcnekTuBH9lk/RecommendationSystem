"""Synthetic chart create/theme/resize/deferred-delete probe; no training."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget
from Application.theme.apply_theme import apply_app_theme
from Application.training_charts import TrainingChart

app = QApplication([])
apply_app_theme(app, False)
for iteration in range(20):
    window = QWidget()
    layout = QVBoxLayout(window)
    for title, names in (('Ошибка обучения', ('Ошибка',)), ('Метрики валидации', ('NDCG@10', 'Recall@10'))):
        layout.addWidget(TrainingChart(title, names, 'Значение'))
    window.resize(950, 700)
    window.show()
    for dark in (True, False, True):
        apply_app_theme(app, dark)
        app.processEvents()
        window.resize(900 if dark else 950, 700)
        app.processEvents()
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    print(f'Completed {iteration + 1}/20', flush=True)
print('Chart/theme stress passed; no training/model writes')
