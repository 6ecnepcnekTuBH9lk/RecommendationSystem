"""GUI-only analytical chart controls and a read-only retention heatmap."""
import os
from decimal import Decimal, ROUND_HALF_UP

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QEvent, QTimer
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QStyledItemDelegate, QTableView, QVBoxLayout, QWidget,
)

from Application import statistics_charts as charts
from Application.statistics_chart_data import (
    RFM_WINDOWS, cohort_retention_heatmap_data, rfm_segments_chart_data, exact_number,
)
from Application.statistics_rfm_presentation import WINDOW_LABELS
from Application.theme.layout_metrics import BLOCK_SPACING


def analytics_chart(data):
    widget = charts.create_chart(data)
    widget.setProperty("buyerAnalyticsChart", True)
    if isinstance(widget, charts.StatisticsChart) and not data.percentage:
        widget.count_axis.setTitleText("")
        if len(data.series) > 1:
            # Native legend layout reserves space below our category labels.
            legend = widget.chart().legend()
            legend.setContentsMargins(0, 16, 0, 0)
            legend.layout().setContentsMargins(0, 16, 0, 0)
    return widget


class RfmSegmentsWidget(QWidget):
    """Selectors replace only this fixed-height plot; no shared filter/state IO."""
    def __init__(self, result=None, parent=None, *, choices=None):
        super().__init__(parent)
        self.choices = choices
        self.setObjectName("rfmSegmentsWidget")
        self.result = result
        self.setMinimumWidth(0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(BLOCK_SPACING)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Период"))
        self.window_selector = QComboBox()
        self.window_selector.setObjectName("rfmChartWindow")
        for code in RFM_WINDOWS:
            self.window_selector.addItem(WINDOW_LABELS[code], code)
        controls.addWidget(self.window_selector)
        controls.addWidget(QLabel("Валюта"))
        self.currency_selector = QComboBox()
        self.currency_selector.setObjectName("rfmChartCurrency")
        for currency in ("RUB", "KZT"):
            self.currency_selector.addItem(currency, currency)
        controls.addWidget(self.currency_selector)
        controls.addStretch()
        layout.addLayout(controls)
        self.plot = None
        self.window_selector.currentIndexChanged.connect(self.update_plot)
        self.currency_selector.currentIndexChanged.connect(self.update_plot)
        self.update_plot()

    def update_plot(self):
        selection = self.window_selector.currentData(), self.currency_selector.currentData()
        data = (self.choices[selection] if self.choices is not None
                else rfm_segments_chart_data(self.result, *selection))
        widget = analytics_chart(data)
        widget.setFixedHeight(340)  # Stable outer geometry even for an empty currency.
        old = self.plot
        self.plot = widget
        if old is not None:
            self.layout().replaceWidget(old, widget)
            old.hide()
            old.setParent(None)
            old.deleteLater()
        else:
            self.layout().addWidget(widget)


def _percent(value):
    return format(Decimal(str(value)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP), "f").replace(".", ",") + " %"


def _background(cell):
    surface = QColor(os.environ.get("QTMATERIAL_SECONDARYCOLOR", "#f5f5f5"))
    if cell.value is None:
        return surface
    accent = QColor(os.environ.get("QTMATERIAL_PRIMARYCOLOR", "#a65cf2"))
    intensity = .10 + .80 * min(1., max(0., float(cell.value) / 100.))
    return QColor(*(round(surface.getRgb()[i] * (1-intensity) + accent.getRgb()[i] * intensity) for i in range(3)))


class _RetentionModel(QAbstractTableModel):
    def __init__(self, data, parent):
        super().__init__(parent)
        self.snapshot = data

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.snapshot.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 14

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return ("Когорта", "Покупателей", *self.snapshot.columns)[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.snapshot.rows[index.row()]
        column = index.column()
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        if column < 2:
            if role == Qt.ItemDataRole.DisplayRole:
                return row.month if column == 0 else exact_number(row.buyers)
            return None
        cell = row.cells[column-2]
        if role == Qt.ItemDataRole.DisplayRole:
            if cell.value is None:
                return "—"
            return _percent(cell.value) + (" (частично)" if cell.status != "FULL" else "")
        if role == Qt.ItemDataRole.UserRole:
            return cell
        if role == Qt.ItemDataRole.BackgroundRole:
            color = _background(cell)
            return QBrush(color, Qt.BrushStyle.Dense4Pattern) if cell.value is not None and cell.status != "FULL" else QBrush(color)
        if role == Qt.ItemDataRole.ForegroundRole:
            color = _background(cell)
            return QColor("#ffffff" if color.lightness() < 135 else "#202124")
        return None


class _RetentionDelegate(QStyledItemDelegate):
    """Paint heatmap roles explicitly: Material QSS overrides item backgrounds."""
    def paint(self, painter, option, index):
        painter.save()
        brush = index.data(Qt.ItemDataRole.BackgroundRole)
        painter.fillRect(option.rect, QColor(os.environ.get("QTMATERIAL_SECONDARYCOLOR", "#f5f5f5")))
        painter.fillRect(option.rect, brush)
        painter.setPen(index.data(Qt.ItemDataRole.ForegroundRole))
        painter.setFont(option.font)
        painter.drawText(option.rect.adjusted(4, 0, -4, 0),
                         Qt.AlignmentFlag.AlignCenter, index.data())
        painter.restore()


class CohortRetentionHeatmap(QTableView):
    """Dedicated chart view; original numeric QTableWidgets remain untouched.

    A small read-only model keeps long cohort lists bounded in widget count.
    Partial observations are labelled explicitly, not presented as full cohorts.
    """
    def __init__(self, result=None, parent=None, *, data=None):
        super().__init__(parent)
        self.setObjectName("cohortRetentionHeatmap")
        self.setProperty("buyerAnalyticsChart", True)
        self.snapshot = data if data is not None else cohort_retention_heatmap_data(result)
        self.setModel(_RetentionModel(self.snapshot, self))
        delegate = _RetentionDelegate(self)
        for column in range(2, 14):
            self.setItemDelegateForColumn(column, delegate)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setWordWrap(False)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(36)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setDefaultSectionSize(174)
        self.setColumnWidth(0, 110)
        self.setColumnWidth(1, 140)
        self.setFixedHeight(min(420, max(180, len(self.snapshot.rows)*36 + 60)))
        self.setMinimumWidth(0)
        self.empty_label = QLabel("Нет данных для отображения.", self.viewport())
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setVisible(not self.snapshot.rows)
        self._theme_timer = QTimer(self)
        self._theme_timer.setSingleShot(True)
        self._theme_timer.timeout.connect(self._refresh_colors)

    def _refresh_colors(self):
        model = self.model()
        if model.rowCount():
            model.dataChanged.emit(model.index(0, 2), model.index(model.rowCount()-1, 13),
                                   [Qt.ItemDataRole.BackgroundRole, Qt.ItemDataRole.ForegroundRole])
        self.viewport().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "empty_label"):
            self.empty_label.setGeometry(self.viewport().rect())

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange) and hasattr(self, "_theme_timer"):
            self._theme_timer.start(0)
