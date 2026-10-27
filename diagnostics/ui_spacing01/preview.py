"""Five real MainWindow tabs with synthetic state, no loading/training/inference."""
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_FONT_DPI'] = '96'
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PyQt6.QtCore import QDate
from PyQt6.QtWidgets import QApplication, QTableWidgetItem
from Application.analysis_filter import AnalysisFilter
from Application.tabs import analysis_filter_tab as analysis, data_loading_tab as data
from Application.tabs import dataset_statistics_tab as statistics, train_model_tab as training
from Application.theme.apply_theme import prepare_app_theme
from main import MainWindow


def synthetic_window(directory):
    """Caller keeps the patches active until the Qt window is closed."""
    stack = ExitStack()
    stack.enter_context(patch.object(analysis, 'SETTINGS_PATH', directory / 'analysis_filter.json'))
    stack.enter_context(patch.object(analysis.mapping, 'SETTINGS_PATH', directory / 'store_city_mapping.json'))
    stack.enter_context(patch.object(analysis.AnalysisFilterTab, 'refresh_options', lambda *a, **k: None))
    stack.enter_context(patch.object(statistics, 'load_filter', AnalysisFilter))
    stack.enter_context(patch.object(statistics, 'CACHE_PATH', directory / 'statistics.json'))
    stack.enter_context(patch.object(statistics, 'load_result', lambda *a: None))
    stack.enter_context(patch.object(training, 'USER_SETTINGS_DIR', directory / 'settings'))
    stack.enter_context(patch.object(training, 'metadata_readiness', lambda *a: (True, 'Synthetic ready', 'a' * 32)))
    stack.enter_context(patch.object(data._LoadingController, 'refresh_persisted', lambda *a: None))
    window = MainWindow()
    return window, stack


def fill(window):
    for control in (window.mb_interaction_since, window.mb_customers_since, window.mb_merge_since, window.mb_manual_since):
        control.setDate(QDate(2025, 1, 1))
    for control in (window.mb_interaction_until, window.mb_customers_until, window.mb_manual_until):
        control.setDate(QDate(2026, 1, 1))
    window.mb_log.setPlainText('Синтетический preview — загрузка данных, обучение и inference не запускались.')
    tab = window.analysis_filter_tab
    tab.controls.setEnabled(True)
    tab.message.hide()
    tab.types.populate(('Рубашка', 'Брюки', 'Пиджак'), None)
    tab.collections.populate(('Весна-лето', 'Осень-зима'), None)
    tab.start_date.setDate(QDate(2025, 1, 1))
    tab.end_date.setDate(QDate(2026, 1, 1))
    tab.reset_button.setEnabled(True)
    tab.store_table.blockSignals(True)
    tab.store_table.setRowCount(3)
    for row, (store, city) in enumerate((('Сайт', 'Москва'), ('Магазин 1', 'Москва'), ('Магазин 2', 'Санкт-Петербург'))):
        tab.store_table.setItem(row, 0, QTableWidgetItem(store))
        tab.store_table.setItem(row, 1, QTableWidgetItem(city))
    tab.store_table.blockSignals(False)
    tab.mapping_readiness.setText('Синтетический каталог: 3 канала')
    tab.mapping_readiness.show()
    spec = importlib.util.spec_from_file_location('spacing_statistics_fixture', ROOT / 'tests/tabs/test_dataset_statistics_tab.py')
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    window.dataset_statistics_tab.render(fixtures.sample_result.__wrapped__())
    for epoch in range(1, 51):
        window.training_loss_chart.add_point('Ошибка', epoch, .7 / (1 + epoch / 15))
    for epoch in (1, 5, 10, 20, 30, 40, 50):
        window.training_metric_chart.add_point('NDCG@10', epoch, .01 * epoch / 50)
        window.training_metric_chart.add_point('Recall@10', epoch, .02 * epoch / 50)
    window.train_log.setPlainText('Синтетический preview — обучение не запускалось.\nГрафики заполнены synthetic scalar points.')
    window.purchases_table.setRowCount(2)
    window.recs_table.setRowCount(2)
    for row in range(2):
        for table, values in ((window.purchases_table, ('—', f'DEMO-{row}', 'Рубашка', 'Осень-зима', 'Просмотр', '01.01.2025')),
                              (window.recs_table, ('—', f'DEMO-{row+2}', 'Брюки', 'Весна-лето', '0,500', '0,050', '10'))):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
    window.le_mb.setText('DEMO-01')
    window.le_fio.setText('Синтетический клиент')


if __name__ == '__main__':
    output = Path(__file__).resolve().parent / sys.argv[1]
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    prepare_app_theme(app)
    with tempfile.TemporaryDirectory(prefix='spacing-preview-') as temporary:
        window, patches = synthetic_window(Path(temporary))
        try:
            fill(window)
            window.setMinimumSize(0, 0)
            window.resize(1920, 1080)
            window.show()
            records = []
            for dark in (False, True):
                window.apply_theme(dark)
                for index, name in enumerate(('01_data_loading', '02_analysis_filter', '03_statistics', '04_training', '05_results')):
                    window.tabs.setCurrentIndex(index)
                    for _ in range(4):
                        app.processEvents()
                    suffix = '_dark' if dark else ''
                    window.grab().save(str(output / (name + suffix + '.png')))
                    records.append({'name': name, 'dark': dark, 'size': [window.width(), window.height()],
                                    'dpi': window.logicalDpiX(), 'start': [window.start_train.y(), window.start_train.height()],
                                    'cancel': [window.cancel_train.y(), window.cancel_train.height()]})
            (output / 'geometry.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        finally:
            window.close()
            window.deleteLater()
            for _ in range(3):
                app.processEvents()
            patches.close()
    print('Saved ten synthetic screenshots; no loading/training/inference')
