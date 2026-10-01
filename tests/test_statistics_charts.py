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
    assert charts.purchase_count_data(result, 'RUB').series == (('RUB', (20, 30)),)
    assert charts.purchase_count_data(result, 'KZT').series == (('KZT', (0, 4)),)
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
    result['action_channel_statistics'] = (('web', 'Сайт', 2, 1, 100., 1, 1, 100.),)
    result['order_monthly_dynamics'] = (('2025-01', 1, '40.25', 1, '50.75'),)
    result['top_favorited_products'] = (('A', 'Избранный', 1, 1, 2, 1),)
    return result


def test_final_charts_pages_legends_and_repeated_render(window, sample_result, app):
    _, tab = window
    result = _populated(sample_result)
    original = deepcopy(result)
    tab.render(result)
    widgets = tab.findChildren(charts.StatisticsChart)
    assert len(widgets) == 17
    assert [len(tab.sections.widget(i).findChildren(charts.StatisticsChart)) for i in range(5)] == [4, 6, 4, 3, 0]
    for widget in widgets:
        assert widget.chart().legend().isVisible() == (widget.data.kind == 'donut')
        if widget.data.kind == 'donut':
            assert not widget.chart().axes()
        else:
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
        assert len(tab.findChildren(charts.StatisticsChart)) == 17
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
        def series_snapshot(widget):
            from PyQt6.QtCharts import QPieSeries
            output = []
            for series in widget.chart().series():
                if isinstance(series, QLineSeries):
                    output.append(tuple((point.x(), point.y()) for point in series.points()))
                elif isinstance(series, QPieSeries):
                    output.append(tuple(piece.value() for piece in series.slices()))
                else:
                    output.append(tuple(tuple(bars.at(i) for i in range(bars.count())) for bars in series.barSets()))
            return tuple(output)
        native_values = [series_snapshot(widget) for widget in widgets]
        assert len(widgets) == 17
        backgrounds = []
        for dark in (False, True, False):
            window.apply_theme(dark)
            app.processEvents()
            backgrounds.append(widgets[0].chart().backgroundBrush().color().name())
            assert backgrounds[-1] == os.environ['QTMATERIAL_SECONDARYDARKCOLOR']
            assert [(id(w.chart()), w.data) for w in widgets] == before
            assert tab.displayed_result is result
            assert [series_snapshot(widget) for widget in widgets] == native_values
            assert all('color:' in widget.tooltip.styleSheet() for widget in widgets)
            from PyQt6.QtGui import QColor
            for widget in widgets:
                if widget.data.kind == 'donut':
                    continue
                expected = QColor(charts.SECONDARY[dark] if widget.data.secondary else os.environ['QTMATERIAL_PRIMARYCOLOR'])
                series = widget.chart().series()[0]
                actual = series.pen().color() if isinstance(series, QLineSeries) else series.barSets()[0].color()
                assert actual == expected
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
                    lambda r: charts.create_order_dynamics_chart(r, 'RUB'),
                    lambda r: charts.create_order_dynamics_chart(r, 'KZT'), charts.create_interaction_frequency_chart, charts.create_basket_chart,
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


def test_new_projections_order_decimal_and_empty(sample_result):
    from decimal import Decimal
    result = _populated(sample_result)
    result['action_channel_statistics'] = tuple((str(i), 'channel.ru' + str(i), i, i, 10., 20-i, i, 5.) for i in range(12))
    result['payment_type_distribution'] = (('Карта', 3, 75.), ('Наличные', 1, 25.))
    result['wear_season_statistics'] = (('СезонНоски: игнорировать', 1000000),)
    original = deepcopy(result)
    assert charts.channel_chart_data(result, 'views').series[0][1] == tuple(range(11, 1, -1))
    assert charts.channel_chart_data(result, 'favorites').series[0][1] == tuple(range(20, 10, -1))
    assert 'channel.ru11' in charts.channel_chart_data(result, 'views').tooltips[0][0]
    assert charts.revenue_chart_data(result, 'RUB').series == (('RUB', (Decimal('40.25'),)),)
    assert charts.revenue_chart_data(result, 'KZT').series == (('KZT', (Decimal('50.75'),)),)
    assert '40,25 RUB' in charts.revenue_chart_data(result, 'RUB').tooltips[0][0]
    assert charts.payment_chart_data(result).labels == ('Карта', 'Наличные')
    assert charts.payment_chart_data(result).series[0][1] == (3, 1)
    for kind, field in [('views', 'top_viewed_products'), ('favorites', 'top_favorited_products')]:
        result[field] = tuple((str(i), 'Полное название ' * 30, 20-i, 2, 1, 1) for i in range(12))
        before = deepcopy(result)
        data = charts.product_ranking_data(result, kind)
        assert result == before
        assert data.labels == tuple(row[1] for row in result[field][:10])
        assert data.series[0][1] == tuple(range(20, 10, -1))
        assert result[field][0][1] in data.tooltips[0][0]
    result['top_viewed_products'] = original['top_viewed_products']
    result['top_favorited_products'] = original['top_favorited_products']
    season = charts.season_chart_data(result)
    assert season.labels == tuple(row[0] for row in result['product_season_statistics'])
    assert season.series[0][1] == tuple(row[4] for row in result['product_season_statistics'])
    assert charts.gender_chart_data(result).series[0][1] == tuple(row[1] for row in result['gender_distribution'])
    assert result == original
    assert charts.compact_number(1250000000) == '1,2 млрд'
    assert charts.compact_number(.25) == '0,25'
    for helper in (lambda r: charts.channel_chart_data(r, 'views'), lambda r: charts.channel_chart_data(r, 'favorites'),
                   lambda r: charts.revenue_chart_data(r, 'RUB'), lambda r: charts.revenue_chart_data(r, 'KZT'),
                   charts.payment_chart_data, lambda r: charts.product_ranking_data(r, 'views'),
                   lambda r: charts.product_ranking_data(r, 'favorites'), charts.season_chart_data, charts.gender_chart_data):
        assert helper({}).empty


def test_long_season_scroll_and_payment_top_ten(app):
    from PyQt6.QtWidgets import QScrollArea
    result = {'product_season_statistics': tuple((f'Коллекция {i}', 1, 20, 5, 10-i, '3') for i in range(7))}
    assert charts.season_chart_data(result).kind == 'horizontal'
    assert charts.season_chart_data({'product_season_statistics': result['product_season_statistics'][:6]}).kind == 'bar'
    result['product_season_statistics'] *= 2
    scroll = charts.create_season_chart(result)
    assert isinstance(scroll, QScrollArea)
    scroll.resize(480, 320); scroll.show(); app.processEvents()
    assert scroll.verticalScrollBar().maximum() > 0
    assert scroll.horizontalScrollBar().maximum() == 0
    assert len(scroll.widget().data.labels) == 14
    scroll.deleteLater()
    result['payment_type_distribution'] = tuple((str(i), i, 1.) for i in range(12))
    assert charts.payment_chart_data(result).series[0][1] == tuple(range(11, 1, -1))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)



def test_new_empty_widgets_and_donut_hover(app, sample_result):
    helpers = (lambda r: charts.channel_chart_data(r, 'views'), lambda r: charts.channel_chart_data(r, 'favorites'),
               lambda r: charts.revenue_chart_data(r, 'RUB'), lambda r: charts.revenue_chart_data(r, 'KZT'),
               charts.payment_chart_data, lambda r: charts.product_ranking_data(r, 'views'),
               lambda r: charts.product_ranking_data(r, 'favorites'), charts.season_chart_data, charts.gender_chart_data)
    for helper in helpers:
        widget = charts.create_chart(helper({}))
        assert widget.objectName() == 'statisticsChartEmpty'
        widget.deleteLater()
    widget = charts.create_gender_chart(sample_result)
    widget.resize(480, 320); widget.show(); app.processEvents()
    series = widget.chart().series()[0]
    assert series.holeSize() == .5
    assert [piece.value() for piece in series.slices()] == [0, 0, 2]
    assert all(not piece.isExploded() and not piece.isLabelVisible() for piece in series.slices())
    series.slices()[-1].hovered.emit(True)
    assert 'Не указан' in widget.tooltip.text() and '100,00 %' in widget.tooltip.text()
    app.sendEvent(widget, QEvent(QEvent.Type.Leave))
    assert not widget.tooltip.isVisible()
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_vertical_season_labels_fit_narrow_and_wide(app):
    result = {'product_season_statistics': tuple(
        (f'Коллекция {i} Осень-Зима 2025 переходящий остаток', 1, 2, 3, 10-i, '4') for i in range(6))}
    widget = charts.create_season_chart(result)
    widget.show()
    try:
        for width in (480, 1400, 480):
            widget.resize(width, 320)
            app.processEvents()
            assert widget.chart().plotArea().height() > 100
            assert all(item.pos().y() + item.boundingRect().height() <= widget.chart().boundingRect().bottom()
                       for item in widget._bar_labels)
            assert widget.data.labels == tuple(row[0] for row in result['product_season_statistics'])
    finally:
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_purchase_split_roles_and_channel_headings(sample_result):
    result = _populated(sample_result)
    result['order_monthly_dynamics'] = (('2025-01', 12000, '40.25', 2, '50.75'), ('2025-02', 8000, '10', 3, '20'))
    result['action_channel_statistics'] *= 12
    before = deepcopy(result)
    for currency, expected in [('RUB', (12000, 8000)), ('KZT', (2, 3))]:
        data = charts.purchase_count_data(result, currency)
        assert data.labels == ('янв. 2025', 'февр. 2025')
        assert data.series == ((currency, expected),)
        assert data.tooltips[0][0] == f'Январь 2025\nПокупок: {charts.exact_number(expected[0])}\nВалюта: {currency}'
        assert data.secondary == (currency == 'KZT')
        assert charts.revenue_chart_data(result, currency).secondary == (currency == 'KZT')
    assert charts.payment_chart_data(result).secondary
    assert charts.season_chart_data(result).secondary
    for kind, title in [('views', 'Каналы по просмотрам'), ('favorites', 'Каналы по избранному')]:
        data = charts.channel_chart_data(result, kind)
        assert data.title == title and len(data.labels) == 10
    assert result == before


def test_interaction_frequency_order_shares_empty_and_no_mutation():
    labels = ('1', '2–5', '6–10', '11–25', '26–50', '51–100', '101+')
    counts = (55860, 83802, 23852, 14428, 4131, 1746, 1023)
    rows = tuple(zip(labels, counts, (30., 45.34, 12., 8., 2., 1., .5)))
    result = {'interaction_activity_distribution': rows}
    before = deepcopy(result)
    data = charts.interaction_frequency_data(result)
    assert data.labels == labels and data.series == (('Клиентов', counts),)
    assert data.kind == 'bar' and not data.secondary and not data.inner_title
    assert data.tooltips[0][1] == 'Количество взаимодействий: 2–5\nКлиентов: 83 802\nДоля: 45,34 %'
    assert result == before
    assert charts.interaction_frequency_data({}).empty
    assert charts.interaction_frequency_data({'interaction_activity_distribution': rows[-1:]}).labels == ('101+',)


def test_independent_count_scales_and_frequency_resize(app):
    result = {'order_monthly_dynamics': (('2025-01', 12000, '40', 2, '50'),)}
    rub = charts.create_order_dynamics_chart(result, 'RUB')
    kzt = charts.create_order_dynamics_chart(result, 'KZT')
    assert rub.count_axis.max() > 12000 and kzt.count_axis.max() < 10
    assert not rub.chart().legend().isVisible() and not kzt.chart().legend().isVisible()
    labels = ('1', '2–5', '6–10', '11–25', '26–50', '51–100', '101+')
    widget = charts.create_interaction_frequency_chart({'interaction_activity_distribution': tuple((label, 10, 1.) for label in labels)})
    widget.show()
    for width in (480, 1400, 480):
        widget.resize(width, 320); app.processEvents()
        assert widget.chart().plotArea().width() > 100
        assert widget.category_axis.categories() == list(labels)
        assert all(item.toPlainText() == label for item, label in zip(widget._bar_labels, labels))
    widget.show_tooltip(0, 6, True)
    assert '101+' in widget.tooltip.text()
    widget.hide(); assert not widget.tooltip.isVisible()
    for chart in (rub, kzt, widget):
        chart.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
