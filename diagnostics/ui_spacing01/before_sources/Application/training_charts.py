"""Small live Qt line charts: structured scalar events only, no model polling."""
import math

from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import QEvent, QLocale, QMargins, QMetaObject, Qt, pyqtSlot
from PyQt6.QtGui import QFontMetricsF, QPainter
from PyQt6.QtWidgets import QGraphicsTextItem, QLabel, QSizePolicy

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
        chart.setTitle(title.upper())
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
        self._epoch_labels = []
        self._epoch_label_color = self.x_axis.labelsBrush().color()
        chart.plotAreaChanged.connect(self._update_plot_layout, Qt.ConnectionType.QueuedConnection)
        self.reset(50)
        self.apply_theme()

    def reset(self, epochs):
        self.epochs = epochs
        self._maximum = 0.
        for series in self.lines.values():
            series.clear()
        self.x_axis.setRange(1 if epochs > 1 else 0, max(1, epochs))
        self.x_axis.setTickType(QValueAxis.TickType.TicksDynamic)
        self.x_axis.setTickAnchor(1)
        self.y_axis.setRange(0, 1)
        self.placeholder.show()
        QMetaObject.invokeMethod(self, '_update_plot_layout', Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def _update_plot_layout(self):
        # Avoid mutating chart layout during Qt's stylesheet/layout traversal.
        # Hidden charts are initialized when their tab becomes visible.
        if not self.isVisible():
            return
        area = self.chart().plotArea()
        # Center in the actual plotting area, excluding titles, axes and legend.
        rectangle = self.mapFromScene(self.chart().mapToScene(area)).boundingRect()
        self.placeholder.setGeometry(rectangle if not area.isEmpty() else self.viewport().rect())
        label_width = QFontMetricsF(self.x_axis.labelsFont()).horizontalAdvance(str(self.epochs)) + 2
        capacity = max(1, min(100, int(area.width() / label_width)))
        step = max(1, math.ceil((self.epochs - 1) / capacity))
        if self.x_axis.tickInterval() != step:
            self.x_axis.setTickInterval(step)
        # Narrow views may label fewer epochs; minor ticks retain every position
        # for normal budgets, without drawing thousands of ticks for huge runs.
        self.x_axis.setMinorTickCount(step - 1 if self.epochs <= 100 else 0)
        self.x_axis.setMinorGridLineVisible(self.epochs <= 100 and step > 1)
        # Qt elides dense numeric labels before drawing them. Use the same
        # chart-owned text-item approach as StatisticsChart's bar/value labels.
        values = list(range(1, self.epochs + 1, step))
        if values[-1] != self.epochs:
            values[-1] = self.epochs
        while len(self._epoch_labels) < len(values):
            item = QGraphicsTextItem(self.chart())
            item.document().setDocumentMargin(0)
            item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self._epoch_labels.append(item)
        for index, item in enumerate(self._epoch_labels):
            item.setVisible(index < len(values))
            if index >= len(values):
                continue
            epoch = values[index]
            item.setPlainText(str(epoch))
            item.setFont(self.x_axis.labelsFont())
            item.setDefaultTextColor(self._epoch_label_color)
            fraction = (epoch - self.x_axis.min()) / (self.x_axis.max() - self.x_axis.min())
            item.setPos(area.left() + fraction * area.width() - item.boundingRect().width() / 2,
                        area.bottom() + 3)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'placeholder'):
            QMetaObject.invokeMethod(self, '_update_plot_layout', Qt.ConnectionType.QueuedConnection)

    def showEvent(self, event):
        super().showEvent(event)
        QMetaObject.invokeMethod(self, '_update_plot_layout', Qt.ConnectionType.QueuedConnection)

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
        _, foreground, _ = style_chart(self.chart(), section_title=True)
        axis_font = self.x_axis.labelsFont()
        axis_font.setPointSizeF(8)
        self.x_axis.setLabelsFont(axis_font)
        self._epoch_label_color = foreground
        self.x_axis.setLabelsBrush(Qt.GlobalColor.transparent)
        self.x_axis.setMinorGridLinePen(self.x_axis.gridLinePen())
        self.placeholder.setStyleSheet(f'QLabel {{background: transparent; color: {foreground.name()}; border: none;}}')
        QMetaObject.invokeMethod(self, '_update_plot_layout', Qt.ConnectionType.QueuedConnection)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange, QEvent.Type.FontChange) and hasattr(self, 'lines'):
            self.apply_theme()
