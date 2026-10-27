"""Small live Qt line charts: structured scalar events only, no model polling."""
import math

from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import QEvent, QLocale, QMargins, Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout

from Application.statistics_charts import style_chart


class TrainingChart(QChartView):
    def __init__(self, title, names, y_title, parent=None):
        chart = QChart()
        super().__init__(chart, parent)
        self.setObjectName('trainingChart')
        self.setStyleSheet('QGraphicsView#trainingChart {border: none; padding: 0;}')
        self.setFrameShape(QChartView.Shape.NoFrame)
        self.setMinimumWidth(0)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setDropShadowEnabled(False)
        chart.setBackgroundRoundness(0)
        chart.setMargins(QMargins(8, 5, 12, 5))
        chart.setTitle(title)
        chart.setLocale(QLocale(QLocale.Language.Russian, QLocale.Country.Russia))
        chart.setLocalizeNumbers(True)
        chart.legend().setVisible(len(names) > 1)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.x_axis, self.y_axis = QValueAxis(), QValueAxis()
        self.x_axis.setTitleText('Эпоха')
        self.x_axis.setLabelFormat('%.0f')
        self.y_axis.setTitleText(y_title)
        self.y_axis.setLabelFormat('%.6f' if len(names) > 1 else '%.3f')
        chart.addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)
        self.lines = {}
        for name in names:
            series = QLineSeries()
            series.setName(name)
            series.setPointsVisible(True)
            chart.addSeries(series)
            series.attachAxis(self.x_axis)
            series.attachAxis(self.y_axis)
            self.lines[name] = series
        self.placeholder = QLabel('График появится после начала обучения', self.viewport())
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.placeholder.setProperty('class', 'infoLabel')
        overlay = QVBoxLayout(self.viewport())
        overlay.addWidget(self.placeholder, alignment=Qt.AlignmentFlag.AlignCenter)
        self.reset(50)
        self.apply_theme()

    def reset(self, epochs):
        self.epochs = epochs
        self._maximum = 0.
        for series in self.lines.values():
            series.clear()
        self.x_axis.setRange(0, max(1, epochs))
        self.x_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.x_axis.setTickAnchor(0)
        self.x_axis.setTickInterval(max(1, math.ceil(epochs / 5)))
        self.y_axis.setRange(0, 1)
        self.placeholder.show()

    def add_point(self, name, epoch, value):
        if (name not in self.lines or type(epoch) is not int or not 1 <= epoch <= self.epochs
                or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            return
        series = self.lines[name]
        if series.count() and series.at(series.count() - 1).x() >= epoch:
            # Duplicate final/checkpoint events do not create duplicate points.
            if series.at(series.count() - 1).x() == epoch:
                series.replace(series.count() - 1, epoch, value)
            else:
                return
        else:
            series.append(epoch, value)
        self._maximum = max(self._maximum, value)
        self.y_axis.setRange(0, max(1e-6, self._maximum * 1.1))
        self.placeholder.hide()

    def apply_theme(self):
        _, foreground, _ = style_chart(self.chart())
        self.placeholder.setStyleSheet(f'QLabel {{background: transparent; color: {foreground.name()}; border: none;}}')

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange, QEvent.Type.FontChange) and hasattr(self, 'lines'):
            self.apply_theme()
