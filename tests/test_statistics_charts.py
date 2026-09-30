"""Snapshot-only chart projections, native widgets, ownership and runtime themes."""
from copy import deepcopy
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtCharts import QLineSeries
from PyQt6.QtWidgets import QLabel

from Application import statistics_charts as charts
from tabs import test_dataset_statistics_tab as samples

app = samples.app
window = samples.window
sample_result = samples.sample_result
pytestmark = pytest.mark.usefixtures('window_settings')


@pytest.mark.parametrize('value,expected', [(1000, '1 тыс.'), (12500, '12,5 тыс.'), (1200000, '1,2 млн'), (0, '0')])
def test_compact_and_exact_format(value, expected):
    assert charts.compact_number(value) == expected
    assert charts.exact_number(1205796) == '1 205 796'
    assert charts.exact_number('1234.125') == '1 234,125'
    assert charts.month_label('2025-01') == 'янв. 2025'
    assert charts.month_label('2025-01', full=True) == 'Январь 2025'


def test_snapshot_projections_order_values_top_ten_and_no_mutation(sample_result):
    result = sample_result
    result['action_monthly_dynamics'] = (('2025-01', 100, 99, 7, 6), ('2025-02', 200, 88, 8, 5))
    result['order_monthly_dynamics'] = (('2025-01', 20, '99999', 0, '88888'), ('2025-02', 30, '77777', 4, '66666'))
    result['top_purchased_products'] = tuple((str(i), 'Очень длинное название ' * 20, 20-i, 3, '4.125', 0, 0) for i in range(12))
    original = deepcopy(result)
    assert charts.monthly_data(result, 'views').series == (('Просмотры', (100, 200)),)
    assert charts.monthly_data(result, 'favorites').series == (('Избранное', (7, 8)),)
    assert charts.monthly_data(result, 'orders').series == (('RUB', (20, 30)), ('KZT', (0, 4)))
    for kind, field in [('basket', 'purchase_basket_distribution'), ('age', 'age_distribution')]:
        data = charts.distribution_data(result, kind)
        assert data.labels == tuple(row[0] for row in result[field])
        assert data.series[0][1] == tuple(row[1] for row in result[field])
    products = charts.purchased_products_data(result)
    assert len(products.labels) == 10
    assert products.series[0][1] == tuple(range(20, 10, -1))
    assert original == result
    assert 'Код: 0' in products.tooltips[0][0] and '4,125' in products.tooltips[0][0]


@pytest.mark.parametrize('value', [None, (), (('2025-01', 0, 0, 0, 0),)])
def test_empty_and_all_zero_widgets(app, value):
    data = charts.monthly_data({'action_monthly_dynamics': value}, 'views')
    assert data.empty
    widget = charts.create_chart(data)
    assert not isinstance(widget, charts.StatisticsChart)
    assert any(label.text() == 'Нет данных для отображения.' for label in widget.findChildren(QLabel))
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _populated(result):
    result = deepcopy(result)
    result['action_monthly_dynamics'] = (('2025-01', 2, 1, 1, 1),)
    return result


def test_six_charts_pages_legends_and_repeated_render(window, sample_result, app):
    _, tab = window
    result = _populated(sample_result)
    original = deepcopy(result)
    tab.render(result)
    widgets = tab.findChildren(charts.StatisticsChart)
    assert len(widgets) == 6
    assert [len(tab.sections.widget(i).findChildren(charts.StatisticsChart)) for i in range(5)] == [2, 2, 1, 1, 0]
    for widget in widgets:
        assert widget.chart().legend().isVisible() == (widget.data.title == 'Динамика заказов')
        assert widget.count_axis.min() == 0
        assert len(widget.chart().axes()) == 2
        for series in widget.chart().series():
            if isinstance(series, QLineSeries):
                assert series.pointsVisible() and series.count() == 1
    old_charts = [widget.chart() for widget in widgets]
    for _ in range(3):
        tab.render(result)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        assert len(tab.findChildren(charts.StatisticsChart)) == 6
    assert all(sip.isdeleted(chart) for chart in old_charts)
    assert result == original and tab.displayed_result is result


def test_tooltip_exact_value_and_mouse_exit(app):
    data = charts.monthly_data({'action_monthly_dynamics': (('2025-01', 1205796, 1, 1, 1),)}, 'views')
    widget = charts.create_chart(data)
    widget.resize(500, 320)
    widget.show()
    app.processEvents()
    widget.show_tooltip(0, 0, True)
    assert '1 205 796' in widget.tooltip.text() and 'Январь 2025' in widget.tooltip.text()
    assert widget.tooltip.isVisible()
    app.sendEvent(widget, QEvent(QEvent.Type.Leave))
    assert not widget.tooltip.isVisible()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_main_window_theme_roundtrip_keeps_chart_objects_and_data(app, sample_result):
    from main import MainWindow
    window = MainWindow()
    try:
        tab = window.dataset_statistics_tab
        result = _populated(sample_result)
        tab.render(result)
        widgets = tab.findChildren(charts.StatisticsChart)
        before = [(id(w.chart()), w.data) for w in widgets]
        backgrounds = []
        for dark in (False, True, False):
            window.apply_theme(dark)
            app.processEvents()
            backgrounds.append(widgets[0].chart().backgroundBrush().color().name())
            assert backgrounds[-1] == os.environ['QTMATERIAL_SECONDARYDARKCOLOR']
            assert [(id(w.chart()), w.data) for w in widgets] == before
            assert tab.displayed_result is result
            for width in (700, 1400):
                window.resize(width, 800)
                app.processEvents()
                for page in range(4):
                    tab.sections.setCurrentIndex(page)
                    app.processEvents()
                    for widget in tab.sections.widget(page).findChildren(charts.StatisticsChart):
                        assert widget.minimumWidth() == 0
        assert backgrounds[0] == backgrounds[2] != backgrounds[1]
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_horizontal_projection_keeps_top_item_and_hover_identity(app):
    data = charts.purchased_products_data({'top_purchased_products': (
        ('A', 'Первый', 20, 3, '4', 0, 0), ('B', 'Второй', 10, 2, '2', 0, 0))})
    widget = charts.create_chart(data)
    widget.resize(480, 320)
    widget.show()
    app.processEvents()
    series = widget.chart().series()[0]
    assert widget.category_axis.categories() == ['2. Второй', '1. Первый']
    assert [series.barSets()[0].at(i) for i in range(2)] == [10, 20]
    series.hovered.emit(True, 1, series.barSets()[0])
    assert 'Код: A' in widget.tooltip.text()
    assert data.labels == ('Первый', 'Второй')
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_empty_state_for_all_builders(app):
    for builder in (charts.create_view_dynamics_chart, charts.create_favorite_dynamics_chart,
                    charts.create_order_dynamics_chart, charts.create_basket_chart,
                    charts.create_purchased_products_chart, charts.create_age_chart):
        widget = builder({})
        assert widget.objectName() == 'statisticsChartEmpty'
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize('name', ['Рубашка', 'Пиджак мужской из шерсти, классический', 'Очень длинное название ' * 30, '', None])
def test_product_labels_adaptive_resize_and_full_tooltip(app, name):
    result = {'top_purchased_products': (('A', name, 20, 3, '4', 0, 0),)}
    original = deepcopy(result)
    widget = charts.create_purchased_products_chart(result)
    try:
        widget.show()
        widget.resize(1400, 320)
        app.processEvents()
        full = f'1. {name or ""}'
        from PyQt6.QtGui import QFontMetrics
        metrics = QFontMetrics(widget.category_axis.labelsFont())
        wide = widget._product_labels[0].text()
        if metrics.horizontalAdvance(full) + charts.PRODUCT_LABEL_PADDING <= charts.MAX_PRODUCT_LABEL_WIDTH:
            assert wide == full
        else:
            assert wide != full and wide.endswith('…')
            assert widget.product_label_width == charts.MAX_PRODUCT_LABEL_WIDTH
        widget.resize(360, 320)
        app.processEvents()
        narrow = widget._product_labels[0].text()
        if metrics.horizontalAdvance(full) > widget.product_label_width - charts.PRODUCT_LABEL_PADDING:
            assert narrow != full and narrow.endswith('…')
            assert len(narrow) > 4
        assert widget.chart().plotArea().width() > 100
        widget.resize(1400, 320)
        app.processEvents()
        assert widget._product_labels[0].text() == wide
        series = widget.chart().series()[0]
        series.hovered.emit(True, 0, series.barSets()[0])
        assert str(name) in widget.tooltip.text()
        assert 'Код: A' in widget.tooltip.text()
        assert widget.data.labels == (name,)
        assert result == original
    finally:
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
