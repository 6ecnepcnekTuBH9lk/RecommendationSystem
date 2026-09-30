"""Native Qt Charts presentation of an immutable, already calculated snapshot."""

from dataclasses import dataclass
from decimal import Decimal
import math
import os

from PyQt6.QtCore import QEvent, QMargins, QPoint, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QGraphicsSimpleTextItem, QGraphicsTextItem, QLabel, QSizePolicy, QVBoxLayout, QWidget
from PyQt6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QCategoryAxis, QChart, QChartView, QHorizontalBarSeries, QLineSeries


MONTHS = ('янв.', 'февр.', 'март', 'апр.', 'май', 'июнь', 'июль', 'авг.', 'сент.', 'окт.', 'нояб.', 'дек.')
MONTHS_FULL = ('Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь')
# Secondary series complements the existing Material accent, with two variants.
SECONDARY = {False: '#227D91', True: '#72C4D1'}
MAX_PRODUCT_LABEL_WIDTH = 420
PRODUCT_LABEL_PADDING = 16


def exact_number(value):
    number = Decimal(str(value))
    text = format(number, 'f')
    whole, _, fraction = text.partition('.')
    grouped = f'{int(whole):,}'.replace(',', ' ')
    fraction = fraction.rstrip('0')
    return grouped + (',' + fraction if fraction else '')


def compact_number(value):
    scale, suffix = (1_000_000, ' млн') if value >= 1_000_000 else (1_000, ' тыс.') if value >= 1_000 else (1, '')
    return exact_number(round(value / scale, 1)) + suffix


def month_label(month, full=False):
    year, number = month.split('-')
    return f'{(MONTHS_FULL if full else MONTHS)[int(number) - 1]} {year}'


@dataclass(frozen=True)
class ChartData:
    title: str
    kind: str
    labels: tuple[str, ...]
    series: tuple[tuple[str, tuple[int, ...]], ...]
    tooltips: tuple[tuple[str, ...], ...]

    @property
    def empty(self):
        return not self.labels or not any(value for _, values in self.series for value in values)


def monthly_data(result, kind):
    fields = {'views': ('Динамика просмотров', 'action_monthly_dynamics', (('Просмотры', 1),)),
              'favorites': ('Динамика избранного', 'action_monthly_dynamics', (('Избранное', 3),)),
              'orders': ('Динамика заказов', 'order_monthly_dynamics', (('RUB', 1), ('KZT', 3)))}
    title, field, columns = fields[kind]
    rows = result.get(field) or ()
    return ChartData(title, 'line', tuple(month_label(row[0]) for row in rows),
                     tuple((name, tuple(row[index] for row in rows)) for name, index in columns),
                     tuple(tuple(f'{month_label(row[0], full=True)}\n{name}: {exact_number(row[index])}'
                                 for row in rows) for name, index in columns))


def distribution_data(result, kind):
    title, field, unit = {'basket': ('Состав корзины', 'purchase_basket_distribution', 'Заказов'),
                         'age': ('Возрастная структура клиентов', 'age_distribution', 'Клиентов')}[kind]
    rows = result.get(field) or ()
    return ChartData(title, 'bar', tuple(row[0] for row in rows), ((unit, tuple(row[1] for row in rows)),),
                     (tuple(f'{label}\n{unit}: {exact_number(count)}\nДоля: {rate:.2f} %'.replace('.', ',')
                            for label, count, rate in rows),))


def purchased_products_data(result):
    rows = (result.get('top_purchased_products') or ())[:10]
    return ChartData('Популярные товары по покупкам — Top-10', 'horizontal', tuple(row[1] for row in rows),
                     (('Покупки', tuple(row[2] for row in rows)),),
                     (tuple(f'Код: {code}\n{name}\nПокупки: {exact_number(count)}\nПокупатели: {exact_number(users)}\n'
                            f'Продано единиц: {exact_number(quantity)}' for code, name, count, users, quantity, *_ in rows),))


def _count_axis(maximum):
    raw_step = max(1, maximum * 1.08 / 4)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(n for n in (1, 2, 5, 10) if n * magnitude >= raw_step) * magnitude
    step = max(1, int(step))
    axis = QCategoryAxis()
    axis.setStartValue(-1)
    axis.setLabelsPosition(QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionOnValue)
    for i in range(5):
        axis.append(compact_number(i * step), i * step)
    axis.setRange(0, 4 * step)
    axis.setTickCount(5)
    return axis


class StatisticsChart(QChartView):
    """Own chart/series/axes via Qt; only retain this small presentation model."""
    def __init__(self, data, parent=None):
        chart = QChart()
        super().__init__(chart, parent)
        self.data = data
        self.setObjectName('statisticsChart')
        self.setMinimumWidth(0)
        self.setFixedHeight(320)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setFrameShape(QChartView.Shape.NoFrame)
        self.setStyleSheet('QGraphicsView#statisticsChart {border: none; border-radius: 0; padding: 0;}')
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setDropShadowEnabled(False)
        chart.setBackgroundRoundness(0)
        chart.setMargins(QMargins(8, 5, 30, 45 if data.kind == 'bar' else 5))
        if data.title in ("Динамика просмотров", "Динамика избранного"):
            chart.setTitle(data.title.upper())
        chart.legend().setVisible(len(data.series) > 1)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.tooltip = QLabel(self.viewport())
        self.tooltip.setTextFormat(Qt.TextFormat.PlainText)
        self.tooltip.setWordWrap(True)
        self.tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.tooltip.hide()
        self._theme_timer = QTimer(self)
        self._theme_timer.setSingleShot(True)
        self._theme_timer.timeout.connect(self.apply_theme)
        maximum = max(value for _, values in data.series for value in values)
        self.count_axis = _count_axis(maximum)
        self.category_axis = QCategoryAxis() if data.kind == 'line' else QBarCategoryAxis()
        horizontal = data.kind == 'horizontal'
        chart.addAxis(self.count_axis, Qt.AlignmentFlag.AlignBottom if horizontal else Qt.AlignmentFlag.AlignLeft)
        chart.addAxis(self.category_axis, Qt.AlignmentFlag.AlignLeft if horizontal else Qt.AlignmentFlag.AlignBottom)
        self.category_axis.setGridLineVisible(False)
        self._bar_labels = []
        self._product_labels = []
        if horizontal:
            self.category_axis.setLabelsVisible(False)
            self._product_labels = [QGraphicsSimpleTextItem(chart) for _ in data.labels]
            chart.plotAreaChanged.connect(self._position_product_labels)
        if data.kind == 'bar':
            # Qt's bar axis truncates every label to the longest-label budget.
            # Owned graphics labels instead wrap within each category's slot.
            self.category_axis.setLabelsVisible(False)
            self._bar_labels = [QGraphicsTextItem(chart) for _ in data.labels]
            chart.plotAreaChanged.connect(self._position_bar_labels)
        if data.kind == 'line':
            self.category_axis.setLabelsPosition(QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionOnValue)
            self.category_axis.setRange(-.3, max(.3, len(data.labels) - .7))
            for number, (name, values) in enumerate(data.series):
                series = QLineSeries()
                series.setName(name)
                series.setPointsVisible(True)
                for index, value in enumerate(values):
                    series.append(index, value)
                chart.addSeries(series)
                series.attachAxis(self.category_axis)
                series.attachAxis(self.count_axis)
                series.hovered.connect(lambda point, state, n=number: self.show_tooltip(n, round(point.x()), state))
        else:
            series = QHorizontalBarSeries() if horizontal else QBarSeries()
            bars = QBarSet(data.series[0][0])
            bars.append(list(reversed(data.series[0][1])) if horizontal else list(data.series[0][1]))
            series.append(bars)
            series.setBarWidth(.65)
            chart.addSeries(series)
            series.attachAxis(self.category_axis)
            series.attachAxis(self.count_axis)
            series.hovered.connect(lambda state, index, bars: self.show_tooltip(
                0, len(data.labels) - 1 - index if horizontal else index, state))
            # Qt draws horizontal bars bottom-up; reverse only the visual projection.
        self._update_categories()
        self.apply_theme()

    def _update_categories(self):
        font = QFont(QApplication.font())
        font.setPointSizeF(9)
        metrics = QFontMetrics(font)
        if self.data.kind == 'line':
            axis = self.category_axis
            for label in axis.categoriesLabels():
                axis.remove(label)
            axis.setStartValue(-.5)
            capacity = max(2, (self.width() - 110) // 95)
            stride = max(1, math.ceil(len(self.data.labels) / capacity))
            indices = list(range(0, len(self.data.labels), stride))
            for index in indices:
                axis.append(self.data.labels[index], index)
        else:
            self.category_axis.clear()
            if self.data.kind == 'horizontal':
                # The rank also keeps repeated product names distinguishable.
                full_labels = [f'{i + 1}. {text or ""}' for i, text in enumerate(self.data.labels)]
                desired = max(metrics.horizontalAdvance(text) for text in full_labels) + PRODUCT_LABEL_PADDING
                available = max(70, int((self.viewport().width() - 80) * .4))
                self.product_label_width = min(desired, MAX_PRODUCT_LABEL_WIDTH, available)
                width = self.product_label_width - PRODUCT_LABEL_PADDING
                labels = [text if metrics.horizontalAdvance(text) <= width else
                          metrics.elidedText(text, Qt.TextElideMode.ElideRight, width) for text in full_labels]
                for item, label in zip(self._product_labels, labels):
                    item.setFont(font)
                    item.setText(label)
                margins = self.chart().margins()
                margins.setLeft(self.product_label_width + 8)
                self.chart().setMargins(margins)
                self._position_product_labels()
                labels.reverse()
            else:
                # Wrapping keeps the business-defined labels readable on narrow pages.
                labels = [label.replace(' лет и старше', ' лет<br>и старше').replace('Возраст не ', 'Возраст<br>не ')
                          for label in self.data.labels]
                if self.data.title == 'Возрастная структура клиентов' and self.width() < 750:
                    # Compact age ranges; full original buckets remain in the tooltip.
                    labels = [label.replace(' лет и старше', '+').replace(' лет', '')
                              if not label.startswith('Возраст не ') else 'Нет<br>данных'
                              for label in self.data.labels]
            self.category_axis.append(labels)
            self.category_axis.setLabelsAngle(0)
            for item, label in zip(self._bar_labels, labels):
                item.document().setDocumentMargin(0)
                item.setFont(font)
                item.setHtml(f'<div align="center">{label}</div>')
            self._position_bar_labels()
        self.category_axis.setLabelsFont(font)
        self.count_axis.setLabelsFont(font)

    def _position_bar_labels(self, *args):
        if not self._bar_labels:
            return
        area = self.chart().plotArea()
        width = area.width() / len(self._bar_labels)
        for index, item in enumerate(self._bar_labels):
            item.setTextWidth(width)
            item.setPos(area.left() + index * width, area.bottom() + 2)

    def _position_product_labels(self, *args):
        if not self._product_labels:
            return
        area = self.chart().plotArea()
        height = area.height() / len(self._product_labels)
        for index, item in enumerate(self._product_labels):
            bounds = item.boundingRect()
            item.setPos(area.left() - PRODUCT_LABEL_PADDING - bounds.width(),
                        area.top() + (index + .5) * height - bounds.height() / 2)

    def apply_theme(self):
        surface = QColor(os.environ.get('QTMATERIAL_SECONDARYCOLOR', '#f5f5f5'))
        background = QColor(os.environ.get('QTMATERIAL_SECONDARYDARKCOLOR', surface.name()))
        foreground = QColor(os.environ.get('QTMATERIAL_SECONDARYTEXTCOLOR', '#555555'))
        accent = QColor(os.environ.get('QTMATERIAL_PRIMARYCOLOR', '#A65CF2'))
        dark = surface.lightness() < 128
        chart = self.chart()
        chart.setBackgroundBrush(background)
        chart.setBackgroundPen(QPen(Qt.PenStyle.NoPen))
        chart.setPlotAreaBackgroundVisible(False)
        chart.setTitleBrush(foreground)
        if self.data.kind == 'horizontal':
            self._update_categories()
        for item in self._product_labels:
            item.setBrush(foreground)
        for item in self._bar_labels:
            item.setDefaultTextColor(foreground)
        font = QFont(QApplication.font())
        font.setPointSizeF(10)
        chart.setTitleFont(font)
        chart.legend().setFont(font)
        chart.legend().setLabelColor(foreground)
        grid = QColor(foreground)
        grid.setAlpha(40)
        for axis in chart.axes():
            axis.setLabelsBrush(foreground)
            axis.setLinePen(QPen(Qt.PenStyle.NoPen))
            axis.setGridLinePen(QPen(grid, 1))
            axis.setMinorGridLineVisible(False)
        for i, series in enumerate(chart.series()):
            color = accent if i == 0 and self.data.title != 'Динамика избранного' else QColor(SECONDARY[dark])
            if isinstance(series, QLineSeries):
                series.setPen(QPen(color, 2))
            else:
                for bars in series.barSets():
                    bars.setColor(color)
                    bars.setBorderColor(color)
        self.tooltip.setStyleSheet(f'QLabel {{background: {background.name()}; color: {foreground.name()}; '
                                  f'border: 1px solid {accent.name()}; border-radius: 5px; padding: 7px;}}')

    def show_tooltip(self, series, index, state):
        if not state or not 0 <= index < len(self.data.labels):
            self.tooltip.hide()
            return
        self.tooltip.setMaximumWidth(max(80, min(350, self.viewport().width() - 16)))
        self.tooltip.setText(self.data.tooltips[series][index])
        self.tooltip.adjustSize()
        from PyQt6.QtGui import QCursor
        point = self.viewport().mapFromGlobal(QCursor.pos()) + QPoint(12, 12)
        point.setX(max(4, min(point.x(), self.viewport().width() - self.tooltip.width() - 4)))
        point.setY(max(4, min(point.y(), self.viewport().height() - self.tooltip.height() - 4)))
        self.tooltip.move(point)
        self.tooltip.show()
        self.tooltip.raise_()

    def leaveEvent(self, event):
        self.tooltip.hide()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self.tooltip.hide()
        super().hideEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'category_axis'):
            self.tooltip.hide()
            self._update_categories()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange, QEvent.Type.FontChange) and hasattr(self, '_theme_timer'):
            self._theme_timer.start(0)


def create_chart(data):
    if not data.empty:
        return StatisticsChart(data)
    widget = QWidget()
    widget.setObjectName('statisticsChartEmpty')
    widget.setMinimumWidth(0)
    widget.setFixedHeight(300)
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    layout = QVBoxLayout(widget)
    title = QLabel(data.title.upper())
    title.setWordWrap(True)
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label = QLabel('Нет данных для отображения.')
    label.setProperty('class', 'infoLabel')
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    font = label.font()
    font.setItalic(True)
    label.setFont(font)
    if data.title in ("Динамика просмотров", "Динамика избранного"):
        layout.addWidget(title)
    layout.addWidget(label, 1)
    return widget


def create_view_dynamics_chart(result):
    return create_chart(monthly_data(result, 'views'))


def create_favorite_dynamics_chart(result):
    return create_chart(monthly_data(result, 'favorites'))


def create_order_dynamics_chart(result):
    return create_chart(monthly_data(result, 'orders'))


def create_basket_chart(result):
    return create_chart(distribution_data(result, 'basket'))


def create_purchased_products_chart(result):
    return create_chart(purchased_products_data(result))


def create_age_chart(result):
    return create_chart(distribution_data(result, 'age'))
