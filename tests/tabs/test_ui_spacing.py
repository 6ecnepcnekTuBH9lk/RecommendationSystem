"""Semantic layout invariants in fully initialized, synthetic MainWindow."""
import ast
from pathlib import Path

import pytest
from PyQt6.QtCore import QMargins, QSize
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QFormLayout, QPushButton, QScrollArea

from Application.theme import layout_metrics as metrics
from Application.theme.apply_theme import prepare_app_theme


@pytest.fixture
def spacing_window(window_settings, monkeypatch, tmp_path):
    from Application.tabs import analysis_filter_tab as analysis, data_loading_tab as data
    from Application.tabs import train_model_tab as training
    from main import MainWindow
    app = QApplication.instance() or QApplication([])
    prepare_app_theme(app)
    monkeypatch.setattr(analysis.AnalysisFilterTab, 'refresh_options', lambda *a, **k: None)
    monkeypatch.setattr(data._LoadingController, 'refresh_persisted', lambda *a: None)
    monkeypatch.setattr(training, 'USER_SETTINGS_DIR', tmp_path / 'settings')
    monkeypatch.setattr(training, 'metadata_readiness', lambda *a: (True, 'Synthetic ready', 'a' * 32))
    monkeypatch.setattr(training.QProcess, 'start', lambda *a: pytest.fail('No training/data process in spacing tests'))
    window = MainWindow()
    window.apply_theme(False)
    window.setMinimumSize(0, 0)
    window.resize(1920, 1080)
    window.show()
    for _ in range(4):
        app.processEvents()
    yield window
    window._training_active = False
    window._training_timer.stop()
    window.mb_state_timer.stop()
    window.close()
    window.deleteLater()
    for _ in range(3):
        app.processEvents()


def row_for(widget):
    return next(layout for layout in widget.parentWidget().findChildren(QHBoxLayout)
                if layout.indexOf(widget) >= 0)


def test_shared_spacing_scale():
    assert (metrics.COMPACT_SPACING, metrics.CONTROL_SPACING, metrics.BLOCK_SPACING,
            metrics.SECTION_SPACING, metrics.PAGE_MARGIN) == (4, 8, 12, 18, 8)


@pytest.mark.parametrize('names', [
    ('mb_start_button', 'mb_customers_button', 'mb_cancel_button'),
    ('mb_manual_interactions_button', 'mb_manual_customers_button'),
    ('analysis_filter_tab.apply_button', 'analysis_filter_tab.reset_button'),
    ('dataset_statistics_tab.refresh', 'dataset_statistics_tab.export_button', 'dataset_statistics_tab.cancel_button'),
    ('start_train', 'cancel_train'), ('btn_excel', 'btn_show_history'),
    ('analysis_filter_tab.types.select_all', 'analysis_filter_tab.types.select_none'),
])
def test_action_rows_share_control_spacing(spacing_window, names):
    def resolve(name):
        target = spacing_window
        for part in name.split('.'):
            target = getattr(target, part)
        return target
    controls = [resolve(name) for name in names]
    row = row_for(controls[0])
    assert row.spacing() == metrics.CONTROL_SPACING
    assert all(row.indexOf(control) >= 0 for control in controls)
    assert [row.indexOf(control) for control in controls] == sorted(row.indexOf(control) for control in controls)


@pytest.mark.parametrize('dark', [False, True])
def test_main_window_training_geometry_comes_from_global_qss_in_both_states(spacing_window, dark):
    window = spacing_window
    window.tabs.setCurrentIndex(3)
    window.apply_theme(dark)
    app = QApplication.instance()
    for running in (False, True, False):
        window.start_train.setEnabled(not running)
        window.cancel_train.setEnabled(running)
        for _ in range(4):
            app.processEvents()
        start, cancel = window.start_train, window.cancel_train
        assert start.height() == cancel.height()
        assert start.geometry().top() == cancel.geometry().top()
        assert start.geometry().bottom() == cancel.geometry().bottom()
        assert start.minimumHeight() < start.maximumHeight()
        assert cancel.minimumHeight() < cancel.maximumHeight()
        assert start.styleSheet() == cancel.styleSheet() == ''
        assert start.iconSize() == cancel.iconSize() == QSize(17, 17)
        assert not cancel.icon().isNull()
        assert row_for(start) is row_for(cancel)
    assert not hasattr(window, 'apply_static_widget_styles')
    assert getattr(window, 'train_proc', None) is None


def test_panel_and_section_metrics_and_form_spacing(spacing_window):
    w = spacing_window
    page = QMargins(metrics.PAGE_MARGIN, metrics.PAGE_MARGIN, metrics.PAGE_MARGIN, metrics.PAGE_MARGIN)
    data_top = w.tabs.widget(0).layout().itemAt(0).layout()
    left_scroll = data_top.itemAt(0).widget().findChild(QScrollArea)
    left, right = left_scroll.widget().layout(), data_top.itemAt(2).widget().layout()
    assert left.contentsMargins() == right.contentsMargins() == page
    assert left.spacing() == right.spacing() == metrics.SECTION_SPACING
    assert left.itemAt(0).layout().spacing() == left.itemAt(1).layout().spacing() == metrics.BLOCK_SPACING
    analysis = w.analysis_filter_tab
    left, right = analysis.layout().itemAt(0).widget().layout(), analysis.layout().itemAt(2).widget().layout()
    assert left.contentsMargins() == right.contentsMargins() == page
    assert left.spacing() == right.spacing() == metrics.BLOCK_SPACING
    assert left.itemAt(4).spacerItem().sizeHint().height() == metrics.SECTION_SPACING - metrics.BLOCK_SPACING
    for form in (analysis.date_form, analysis.product_form):
        assert form.horizontalSpacing() == form.verticalSpacing() == metrics.CONTROL_SPACING
        assert form.rowWrapPolicy() in (QFormLayout.RowWrapPolicy.DontWrapRows, QFormLayout.RowWrapPolicy.WrapAllRows)
    train_top = w.tabs.widget(3).layout().itemAt(0).layout()
    left, right = train_top.itemAt(0).widget().layout(), train_top.itemAt(2).widget().layout()
    assert left.contentsMargins() == right.contentsMargins() == page
    assert left.spacing() == metrics.SECTION_SPACING and right.spacing() == metrics.BLOCK_SPACING
    forms = w.tabs.widget(3).findChildren(QFormLayout)
    assert len(forms) == 1 and forms[0].horizontalSpacing() == forms[0].verticalSpacing() == metrics.CONTROL_SPACING
    history = w.tabs.widget(3).layout().itemAt(2).layout()
    assert history.contentsMargins() == page and history.spacing() == metrics.COMPACT_SPACING
    assert w.dataset_statistics_tab.layout().contentsMargins() == page
    assert w.dataset_statistics_tab.layout().spacing() == metrics.BLOCK_SPACING


def test_status_indicator_spacing_and_main_bar(spacing_window):
    w = spacing_window
    status_groups = w.status_files_layout.itemAt(1).widget().layout()
    assert status_groups.spacing() == metrics.CONTROL_SPACING
    for index in range(3):
        assert status_groups.itemAt(index).widget().layout().spacing() == metrics.COMPACT_SPACING
    assert w.prefix.styleSheet() == ''
    assert w.status_label.parentWidget().layout().spacing() == metrics.COMPACT_SPACING
    assert w.centralWidget().layout().spacing() == metrics.CONTROL_SPACING
    assert w.centralWidget().layout().contentsMargins() == QMargins(*(metrics.PAGE_MARGIN,) * 4)


def test_separator_layouts_keep_structural_zeros_and_results_tables_use_layout(spacing_window):
    w = spacing_window
    for index in (0, 3):
        root = w.tabs.widget(index).layout()
        top = root.itemAt(0).layout()
        assert root.contentsMargins() == top.contentsMargins() == QMargins()
        assert root.spacing() == top.spacing() == 0
        assert top.itemAt(1).widget().objectName() == 'vSeparator'
        assert root.itemAt(1).widget().objectName() == 'hSeparator'
    root = w.tabs.widget(4).layout()
    assert root.spacing() == 0 and root.contentsMargins() == QMargins()
    assert root.itemAt(1).widget().objectName() == 'hSeparator'
    bottom = root.itemAt(2).layout()
    assert bottom.spacing() == 0 and bottom.contentsMargins() == QMargins()
    assert bottom.itemAt(1).widget().objectName() == 'vSeparator'
    for table in (w.purchases_table, w.recs_table):
        assert table.styleSheet() == ''
        assert table.parentWidget().layout().spacing() == metrics.BLOCK_SPACING
        assert table.parentWidget().layout().contentsMargins() == QMargins(*(metrics.PAGE_MARGIN,) * 4)


def test_analysis_spacing_has_no_statistics_tab_dependency():
    from Application.tabs import analysis_filter_tab
    tree = ast.parse(Path(analysis_filter_tab.__file__).read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.ImportFrom) and node.module == 'Application.tabs.dataset_statistics_tab'
                   for node in ast.walk(tree))


def test_ordinary_buttons_have_no_local_height_or_padding_constraints(spacing_window):
    for button in spacing_window.findChildren(QPushButton):
        assert button.minimumHeight() < button.maximumHeight()
        assert not any(word in button.styleSheet() for word in ('padding', 'margin', 'height'))


def test_theme_switch_keeps_geometry_of_action_rows_and_headers(spacing_window):
    from PyQt6.QtWidgets import QLabel
    window = spacing_window
    app = QApplication.instance()
    for index in range(5):
        window.tabs.setCurrentIndex(index)
        records = []
        for dark in (False, True, False):
            window.apply_theme(dark)
            for _ in range(4):
                app.processEvents()
            page = window.tabs.widget(index)
            controls = page.findChildren(QPushButton)
            headers = [label for label in page.findChildren(QLabel) if label.property('class') == 'sectionHeader']
            records.append([widget.geometry() for widget in controls + headers])
        assert records[0] == records[1] == records[2]
