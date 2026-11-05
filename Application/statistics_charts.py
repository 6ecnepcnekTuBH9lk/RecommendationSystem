"""Native Qt Charts presentation of an immutable, already calculated snapshot."""

import math
import os

from PyQt6.QtCore import QEvent, QMargins, QPoint, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QScrollArea, QGraphicsSimpleTextItem, QGraphicsTextItem, QLabel, QSizePolicy, QVBoxLayout, QWidget
from PyQt6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QCategoryAxis, QChart, QChartView, QHorizontalBarSeries, QLineSeries, QPieSeries


from Application.statistics_chart_data import ChartData as ChartData, exact_number as exact_number
from Application.statistics_chart_data import (
    activity_distribution_data as activity_distribution_data,
    distribution_data as distribution_data,
    gender_chart_data as gender_chart_data,
    interaction_frequency_data as interaction_frequency_data,
    month_label as month_label,
    monthly_data as monthly_data,
    payment_chart_data as payment_chart_data,
    product_ranking_data as product_ranking_data,
    purchase_count_data as purchase_count_data,
    purchased_products_data as purchased_products_data,
    revenue_chart_data as revenue_chart_data,
    season_chart_data as season_chart_data
)



MONTHS = ('янв.', 'февр.', 'март', 'апр.', 'май', 'июнь', 'июль', 'авг.', 'сент.', 'окт.', 'нояб.', 'дек.')
MONTHS_FULL = ('Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь')
# Secondary series complements the existing Material accent, with two variants.
SECONDARY = {False: '#227D91', True: '#72C4D1'}
MAX_PRODUCT_LABEL_WIDTH = 420
PRODUCT_LABEL_PADDING = 16
NEUTRAL = {False: '#8A8F98', True: '#9BA3AD'}


def style_chart(chart, secondary=False, section_title=False):
    """Shared Material palette for native statistics and live training charts."""
    surface = QColor(os.environ.get('QTMATERIAL_SECONDARYCOLOR', '#f5f5f5'))
    background = QColor(os.environ.get('QTMATERIAL_SECONDARYDARKCOLOR', surface.name()))
    foreground = QColor(os.environ.get('QTMATERIAL_SECONDARYTEXTCOLOR', '#555555'))
    accent = QColor(os.environ.get('QTMATERIAL_PRIMARYCOLOR', '#A65CF2'))
    dark = surface.lightness() < 128
    chart.setBackgroundBrush(background)
    chart.setBackgroundPen(QPen(Qt.PenStyle.NoPen))
    chart.setPlotAreaBackgroundVisible(False)
    chart.setTitleBrush(foreground)
    font = QFont(QApplication.font())
    font.setPointSizeF(10)
    chart.legend().setFont(font)
    title_font = QFont(font)
    if section_title:
        # Same typography as QLabel.statisticsSection in the shared QSS.
        title_font.setPointSizeF(12)
        title_font.setItalic(True)
    chart.setTitleFont(title_font)
    chart.legend().setLabelColor(foreground)
    grid = QColor(foreground)
    grid.setAlpha(40)
    for axis in chart.axes():
        axis.setLabelsBrush(foreground)
        axis.setTitleBrush(foreground)
        axis.setLinePen(QPen(Qt.PenStyle.NoPen))
        axis.setGridLinePen(QPen(grid, 1))
        axis.setMinorGridLineVisible(False)
    for i, series in enumerate(chart.series()):
        color = accent if i == 0 and not secondary else QColor(SECONDARY[dark])
        if isinstance(series, QLineSeries):
            series.setPen(QPen(color, 2))
        elif isinstance(series, QPieSeries):
            for piece, fill in zip(series.slices(), (accent, QColor(SECONDARY[dark]), QColor(NEUTRAL[dark]))):
                piece.setBrush(fill)
                piece.setPen(QPen(background, 1))
        else:
            for index, bars in enumerate(series.barSets()):
                fill = color if index == 0 else QColor(SECONDARY[dark])
                bars.setColor(fill)
                bars.setBorderColor(fill)
    return background, foreground, accent


def compact_number(value):
    scale, suffix = ((1_000_000_000, ' млрд') if value >= 1_000_000_000 else
                     (1_000_000, ' млн') if value >= 1_000_000 else (1_000, ' тыс.') if value >= 1_000 else (1, ''))
    return exact_number(round(value / scale, 1) if scale > 1 else value) + suffix


























def _count_axis(maximum, money=False):
    raw_step = max(.01 if money else 1, maximum * 1.08 / 4)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = next(n for n in (1, 2, 5, 10) if n * magnitude >= raw_step) * magnitude
    step = round(step, 10) if money else max(1, int(step))
    axis = QCategoryAxis()
    axis.setStartValue(-1)
    axis.setLabelsPosition(QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionOnValue)
    for i in range(5):
        axis.append(compact_number(round(i * step, 10)), i * step)
    axis.setRange(0, 4 * step)
    axis.setTickCount(5)
    return axis


def _percentage_axis():
    axis = QCategoryAxis()
    axis.setStartValue(-1)
    axis.setLabelsPosition(QCategoryAxis.AxisLabelsPosition.AxisLabelsPositionOnValue)
    for value in (0, 25, 50, 75, 100):
        axis.append(str(value), value)
    axis.setRange(0, 100)
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
        chart.setTitle(data.title.upper() if data.inner_title else '')
        chart.legend().setVisible(len(data.series) > 1 or data.kind == 'donut')
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.tooltip = QLabel(self.viewport())
        self.tooltip.setTextFormat(Qt.TextFormat.PlainText)
        self.tooltip.setWordWrap(True)
        self.tooltip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.tooltip.hide()
        self._theme_timer = QTimer(self)
        self._theme_timer.setSingleShot(True)
        self._theme_timer.timeout.connect(self.apply_theme)
        self._product_labels = []
        self._bar_labels = []
        self._value_labels = []
        self._point_labels = []
        self.count_axis = self.category_axis = None
        if data.kind == 'donut':
            series = QPieSeries()
            series.setHoleSize(.5)
            for index, (name, value) in enumerate(zip(data.labels, data.series[0][1])):
                piece = series.append(name, value)
                piece.setLabelVisible(False)
                piece.hovered.connect(lambda state, i=index: self.show_tooltip(0, i, state))
            chart.addSeries(series)
        else:
            maximum = float(max((value for _, values in data.series for value in values if value is not None), default=0))
            self.count_axis = _percentage_axis() if data.percentage else _count_axis(maximum, data.money)
            self.count_axis.setTitleText(data.value_axis_title)
            self.category_axis = QCategoryAxis() if data.kind == 'line' else QBarCategoryAxis()
            horizontal = data.kind == 'horizontal'
            chart.addAxis(self.count_axis, Qt.AlignmentFlag.AlignBottom if horizontal else Qt.AlignmentFlag.AlignLeft)
            chart.addAxis(self.category_axis, Qt.AlignmentFlag.AlignLeft if horizontal else Qt.AlignmentFlag.AlignBottom)
            self.category_axis.setGridLineVisible(False)
            self._bar_labels = []
            self._product_labels = []
            if horizontal:
                self.count_axis.setLabelsVisible(False)
                self._value_labels = [(self.count_axis.endValue(label), QGraphicsSimpleTextItem(label, chart))
                                      for label in self.count_axis.categoriesLabels()]
                margins = chart.margins()
                margins.setBottom(30)
                chart.setMargins(margins)
                chart.plotAreaChanged.connect(self._position_value_labels)
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
                for number, (name, values) in enumerate(data.series):
                    bars = QBarSet(name)
                    bars.setProperty("statisticsSeriesIndex", number)
                    plotted = [float(value) if value is not None else 0. for value in values]
                    bars.append(list(reversed(plotted)) if horizontal else plotted)
                    series.append(bars)
                if data.zero_is_data and not horizontal:
                    # Missing points have no bar; explicit markers distinguish
                    # unavailable observations from a trustworthy zero.
                    for index, value in enumerate(data.series[0][1]):
                        if value is None or value == 0:
                            label = "—" if value is None else ("0 %" if data.percentage else "0")
                            self._point_labels.append((index, QGraphicsSimpleTextItem(label, chart)))
                    chart.plotAreaChanged.connect(self._position_point_labels)
                series.setBarWidth(.65)
                chart.addSeries(series)
                series.attachAxis(self.category_axis)
                series.attachAxis(self.count_axis)
                series.hovered.connect(lambda state, index, bars: self.show_tooltip(
                    int(bars.property("statisticsSeriesIndex")),
                    len(data.labels) - 1 - index if horizontal else index, state))
                # Qt draws horizontal bars bottom-up; reverse only the visual projection.
        self._update_categories()
        self.apply_theme()

    def _update_categories(self):
        if self.data.kind == 'donut':
            return
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
                full_labels = [f'{i + 1}. {text or ""}' if self.data.ranked else (text or '')
                               for i, text in enumerate(self.data.labels)]
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
            self.category_axis.append([str(i) for i in range(len(labels))] if self.data.kind == 'horizontal'
                                      and not self.data.ranked else labels)
            self.category_axis.setLabelsAngle(0)
            for item, label in zip(self._bar_labels, labels):
                item.document().setDocumentMargin(0)
                item.setFont(font)
                item.setHtml(f'<div align="center">{label}</div>')
            self._position_bar_labels()
        self.category_axis.setLabelsFont(font)
        self.count_axis.setLabelsFont(font)
        for _, item in self._value_labels:
            item.setFont(font)
        self._position_value_labels()

    def _position_value_labels(self, *args):
        if not self._value_labels:
            return
        area = self.chart().plotArea()
        label_width = max(item.boundingRect().width() for _, item in self._value_labels) + 12
        stride = max(1, math.ceil(label_width / max(1, area.width() / 4)))
        for index, (value, item) in enumerate(self._value_labels):
            item.setVisible(index % stride == 0)
            item.setPos(area.left() + value / self.count_axis.max() * area.width() - item.boundingRect().width() / 2,
                        area.bottom() + 3)

    def _position_bar_labels(self, *args):
        if not self._bar_labels:
            return
        area = self.chart().plotArea()
        if area.width() <= 0:
            return
        width = area.width() / len(self._bar_labels)
        for index, item in enumerate(self._bar_labels):
            item.setTextWidth(width)
            item.setPos(area.left() + index * width, area.bottom() + 2)
        if self.data.title == 'Спрос по сезону':
            bottom = min(180, max(45, math.ceil(max(item.boundingRect().height() for item in self._bar_labels)) + 8))
            margins = self.chart().margins()
            if margins.bottom() != bottom:
                margins.setBottom(bottom)
                self.chart().setMargins(margins)

    def _position_point_labels(self, *args):
        area = self.chart().plotArea()
        width = area.width() / max(1, len(self.data.labels))
        for index, item in self._point_labels:
            item.setPos(area.left() + (index + .5) * width - item.boundingRect().width() / 2,
                        area.bottom() - item.boundingRect().height() - 3)

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
        chart = self.chart()
        background, foreground, accent = style_chart(chart, self.data.secondary)
        if self.data.kind == 'horizontal':
            self._update_categories()
        for item in self._product_labels:
            item.setBrush(foreground)
        for _, item in self._value_labels:
            item.setBrush(foreground)
        for item in self._bar_labels:
            item.setDefaultTextColor(foreground)
        for _, item in self._point_labels:
            item.setBrush(foreground)
        self._position_point_labels()
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
        chart = StatisticsChart(data)
        if data.kind == 'horizontal' and len(data.labels) > 10:
            chart.setFixedHeight(len(data.labels) * 24 + 60)
            scroll = QScrollArea()
            scroll.setObjectName('statisticsChartScroll')
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setMinimumWidth(0)
            scroll.setFixedHeight(320)
            scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            scroll.setWidget(chart)
            return scroll
        return chart
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
    if data.inner_title:
        layout.addWidget(title)
    layout.addWidget(label, 1)
    return widget


def create_view_dynamics_chart(result):
    return create_chart(monthly_data(result, 'views'))


def create_favorite_dynamics_chart(result):
    return create_chart(monthly_data(result, 'favorites'))


def create_order_dynamics_chart(result, currency):
    return create_chart(purchase_count_data(result, currency))


def create_interaction_frequency_chart(result):
    return create_chart(interaction_frequency_data(result))


def create_basket_chart(result):
    return create_chart(distribution_data(result, 'basket'))


def create_purchased_products_chart(result):
    return create_chart(purchased_products_data(result))


def create_age_chart(result):
    return create_chart(distribution_data(result, 'age'))


def create_activity_distribution_chart(result, kind):
    return create_chart(activity_distribution_data(result, kind))


def create_revenue_chart(result, currency):
    return create_chart(revenue_chart_data(result, currency))


def create_payment_chart(result):
    return create_chart(payment_chart_data(result))


def create_product_ranking_chart(result, kind):
    return create_chart(product_ranking_data(result, kind))


def create_season_chart(result):
    return create_chart(season_chart_data(result))


def create_gender_chart(result):
    return create_chart(gender_chart_data(result))
