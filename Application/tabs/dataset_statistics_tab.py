"""Statistics presentation and subprocess lifecycle; no raw-data business rules."""

import json
from copy import deepcopy
import math
import re
import sys
from datetime import date
from decimal import Decimal
from itertools import groupby
from pathlib import Path
from threading import Event

from PyQt6.QtCore import QAbstractTableModel, QEvent, QObject, QProcess, QRect, QRunnable, QSize, QThreadPool, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QPainter, QRegion
from PyQt6.QtWidgets import QApplication as QApplication, QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QProgressBar, QPushButton, QScrollArea, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QSizePolicy, QStyle, QStyleOptionHeader, QFileDialog as QFileDialog, QMessageBox, QTableView, QToolButton

from Application.statistics_presentation import SOURCE_ACTIONS_HEADING as SOURCE_ACTIONS_HEADING, ACTION_QUALITY_HEADING as ACTION_QUALITY_HEADING, source_event_quality_display_rows as source_event_quality_display_rows, CATEGORY_HEADERS as CATEGORY_HEADERS, DIRECT_CATEGORY_NOTE as DIRECT_CATEGORY_NOTE, product_category_rows as product_category_rows, customer_scope_sections as customer_scope_sections, PROFILE_SLICE_LABEL, CUSTOMER_GROUP_ORDER
from Application import statistics_charts as charts
from Application import statistics_report_presentation as report
from Application.statistics_analytics_charts import (
    analytics_chart, CohortRetentionHeatmap, RfmSegmentsWidget, StatisticsChartRow,
)
from Application.statistics_table_explanation_widget import StatisticsTableExplanation
from Application.statistics_table_explanations import table_explanation
from Application.statistics_rfm_presentation import rfm_sections as rfm_sections
from Application.statistics_cohorts_presentation import cohort_sections as cohort_sections, cohort_display as cohort_display
from Application.loading_errors import format_error
from Application.paths import ICONS_DIR as ICONS_DIR, PROJECT_ROOT as PROJECT_ROOT
from Application.analysis_filter import AnalysisFilter
from Application.statistics_unified_dto import UnifiedResult, read_envelope
from Application.statistics_unified_presentation import UNIFIED_ERRORS, unified_presentation
from Application.statistics_unified_detail import filter_snapshot
from Application.analysis_filter_settings import load_filter
from Application.settings.set_status import set_status_processing, schedule_status_reset
from Application.statistics_detail_process import DetailExportProcess
from Application.statistics_detail_job import ERRORS as DETAIL_ERRORS, STAGES as DETAIL_STAGES, validate_report_destination
from Application.statistics_bundle_export import suggested_filename as suggested_report_filename, validate_report_snapshot
from Application.statistics_cache import CACHE_PATH as CACHE_PATH, load_result, save_result, validate_result
from Application.statistics_presentation import WARNING_LABELS as WARNING_LABELS, DIAGNOSTIC_LABELS as DIAGNOSTIC_LABELS, _date, _calculation_message as _calculation_message
from Application.theme.layout_metrics import COMPACT_SPACING, CONTROL_SPACING as CONTROL_SPACING, BLOCK_SPACING, SECTION_SPACING as SECTION_SPACING, PAGE_MARGIN


SOURCE_DESCRIPTION = "Исходные данные Mindbox. Отбор не установлен."
PROCESSING_STATUS = "Расчет статистики..."
PROGRESS_LABELS = {"actions": "Действия", "orders": "Заказы", "customer_merges": "Объединения клиентов"}

CUSTOMER_TAB_LABELS = report.CUSTOMER_TAB_LABELS
_RESULT_LINE = re.compile(br'^\s*\{\s*"event"\s*:\s*"result"\s*,')


class _PresentationSignals(QObject):
    finished = pyqtSignal(object, object, str)


class _UnifiedPresentationTask(QRunnable):
    """Decode the aggregate payload and prepare its read-only report off the GUI thread."""

    def __init__(self, line, request):
        super().__init__()
        self.line, self.request = line, request
        self.cancelled = Event()
        self.signals = _PresentationSignals()

    def check(self):
        if self.cancelled.is_set():
            raise InterruptedError

    def run(self):
        try:
            self.check()
            payload = json.loads(self.line)["value"]
            applied = AnalysisFilter.from_dict(payload["analysis_filter"]), payload["product_code"], payload["category_code"]
            result = read_envelope(payload["report"], check=self.check)
            if applied != self.request[2:] or not _matches_unified_request(result, self.request):
                raise ValueError("Invalid report binding")
            model = unified_presentation(result, applied_filter=filter_snapshot(*applied))
            self.check()
        except InterruptedError:
            self.signals.finished.emit(None, None, "CANCELLED")
        except Exception:
            self.signals.finished.emit(None, None, "INVALID_UNIFIED_DTO")
        else:
            self.signals.finished.emit(result, model, "")


class _SavedStatisticsTask(QRunnable):
    def __init__(self):
        super().__init__()
        self.cancelled = Event()
        self.signals = _PresentationSignals()

    def check(self):
        if self.cancelled.is_set():
            raise InterruptedError

    def run(self):
        from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT
        from Application.product_resolution import DEFAULT_CATALOG_PATH
        from Application.statistics_index import DEFAULT_INDEX_DIR
        from Application.statistics_storage_safety import StatisticsIndexError
        from Application.statistics_unified_service import open_latest_statistics
        try:
            result, request = open_latest_statistics(
                raw_root=DEFAULT_RAW_ROOT, catalog_path=DEFAULT_CATALOG_PATH, index_dir=DEFAULT_INDEX_DIR,
                cache_dir=PROJECT_ROOT / "filtered_data" / "statistics_unified_cache", check=self.check)
            model = unified_presentation(result, applied_filter=filter_snapshot(*request[2:]))
            self.check()
        except InterruptedError:
            self.signals.finished.emit(None, None, "CANCELLED")
        except StatisticsIndexError as error:
            code = "UNIFIED_CACHE_MISS" if str(error) == "UNIFIED_CACHE_MISS" else "SAVED_REPORT_STALE"
            self.signals.finished.emit(None, None, code)
        except (OSError, ValueError, KeyError, TypeError):
            self.signals.finished.emit(None, None, "SAVED_REPORT_STALE")
        else:
            self.signals.finished.emit(result, (model, request), "")


def _matches_unified_request(result, request):
    if not isinstance(result, UnifiedResult) or request[:2] != ("unified", result.source["reader"]):
        return False
    _, _, selection, product, category = request
    period = [part.isoformat() for part in selection.interval] if selection.interval else None
    params = result.metadata["parameters"]
    return (all(params[key] == period for key in ("order_period", "buyer_period", "customer_period"))
            and (params["selected_start"], params["selected_end"]) == (selection.start_date, selection.end_date)
            and bool(params["product_filter_key"]) == bool(selection.product_restricted or product or category))



def _number(value):
    if isinstance(value, report.ReportNumber):
        return report.number_text(value)
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", " ")
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return str(value)


def _label(text):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def _information_label(text):
    label = _label(text)
    label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    label.setProperty("class", "statisticsInfo")
    return label


class _ProgressStatus(QLabel):
    """Draw the current status over the progress bar, including busy mode."""

    def __init__(self, progress, text):
        super().__init__()
        self.progress = progress
        self.setObjectName("statisticsProgressStatus")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setTextFormat(Qt.TextFormat.PlainText)
        progress.valueChanged.connect(self.update)
        self.setText(text)

    def setText(self, text):
        super().setText(text)
        self.setToolTip(text)
        self.progress.setAccessibleDescription(text)

    def paintEvent(self, event):
        progress = self.progress
        text = " ".join(self.text().split())
        determinate = progress.maximum() > progress.minimum() and progress.value() > progress.minimum()
        if determinate:
            percent = round(100 * (progress.value() - progress.minimum()) / (progress.maximum() - progress.minimum()))
            text = f"{percent}% · {text}"
        metrics = self.fontMetrics()
        area = self.rect().adjusted(6, 0, -6, 0)
        text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, area.width())
        foreground = self.palette().color(self.foregroundRole())
        accent_text = QColor("#17121e")
        active = progress.isVisible() and (determinate or progress.maximum() == progress.minimum())
        filled = QRegion()
        if active and foreground.lightness() > 128:
            # The native indeterminate chunk moves independently of valueChanged.
            # Read its painted center line so both modes use the same glyph positions.
            image = progress.grab().toImage()
            scale = image.devicePixelRatio()
            if not image.isNull():
                y = image.height() // 2
                right = min(area.right() + 1, math.ceil(image.width() / scale))
                for bright, pixels in groupby(range(area.left(), right),
                                             key=lambda x: image.pixelColor(int(x * scale), y).lightness() > 128):
                    if bright:
                        first = next(pixels)
                        last = max(pixels, default=first)
                        filled |= QRegion(QRect(first, area.top(), last - first + 1, area.height()))
        elif active:
            foreground = accent_text
        painter = QPainter(self)
        painter.setClipRegion(QRegion(area).subtracted(filled))
        painter.setPen(foreground)
        painter.drawText(area, Qt.AlignmentFlag.AlignCenter, text)
        if not filled.isEmpty():
            painter.setClipRegion(filled)
            painter.setPen(accent_text)
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, text)


def _section_heading(layout, title):
    section = QVBoxLayout()
    section.setObjectName('statisticsSectionLayout')
    section.setContentsMargins(0, 0, 0, 0)
    section.setSpacing(BLOCK_SPACING)
    label = _label(title.upper())
    label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    label.setProperty("class", "statisticsSection")
    section.addWidget(label)
    layout.addLayout(section)
    return section


class _NumericText(str):
    """Keep numeric alignment when an existing display helper returns text."""


def _numeric(value, *, percentage=False):
    if isinstance(value, report.ReportMeasure):
        return _numeric(value.value if value.value is not None else value.known, percentage=percentage)
    if isinstance(value, (int, float, Decimal, _NumericText, report.ReportNumber)) and not isinstance(value, bool):
        return True
    # A known percent field may already be formatted by a saved presentation.
    # Recognize its text without parsing/rounding it or treating identifiers as numbers.
    return percentage and isinstance(value, str) and re.fullmatch(r"[+-]?\d[\d ]*(?:[.,]\d+)?\s*%?", value) is not None


def _cell_text(value, *, display=None, percentage=False):
    shown = display(value) if display else value
    return report.cell_presentation(shown, percentage=percentage, numeric_hint=_numeric(value)).text


class _StatisticsTable(QTableWidget):
    """Fit short tables; retain readable columns and bounded long tables."""
    def __init__(self, *args):
        super().__init__(*args)
        self._base_widths = []
        self._minimum_widths = []
        self._flex_column = 0
        self._last_minimum = 0
        self._adjusting_columns = False
        self._sizing_rows = False
        self._row_timer = QTimer(self)
        self._row_timer.setSingleShot(True)
        self._row_timer.timeout.connect(self.size_rows)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.horizontalHeader().sectionResized.connect(self._column_resized)

    def size_rows(self):
        if self._sizing_rows:
            return
        self._row_timer.stop()
        self._sizing_rows = True
        try:
            minimum = self.fontMetrics().height() + 2 * CONTROL_SPACING
            self.verticalHeader().setMinimumSectionSize(minimum)
            self.verticalHeader().setDefaultSectionSize(minimum)
            self.resizeRowsToContents()
            if self.wordWrap():
                from PyQt6.QtCore import QRect
                metrics = self.fontMetrics()
                for row in range(self.rowCount()):
                    required = self.rowHeight(row)
                    for column in range(self.columnCount()):
                        item = self.item(row, column)
                        if item is None or not item.textAlignment() & Qt.AlignmentFlag.AlignLeft:
                            continue
                        width = max(1, self.columnWidth(column)-2*CONTROL_SPACING-4)
                        height = metrics.boundingRect(QRect(0, 0, width, 100000),
                            Qt.TextFlag.TextWordWrap | Qt.TextFlag.TextExpandTabs, item.text()).height()
                        required = max(required, height+2*CONTROL_SPACING)
                    self.setRowHeight(row, required)
        finally:
            self._sizing_rows = False
        self.fit_height()

    def _column_resized(self, column, old, width):
        if not self._adjusting_columns and column < len(self._base_widths):
            minimum = self._minimum_widths[column]
            if sum(self._minimum_widths) <= self.viewport().width():
                if column == self._flex_column:
                    width = minimum  # The last section always occupies the remainder.
                else:
                    others = sum(self.columnWidth(c) for c in range(self.columnCount())
                                 if c not in (column, self._flex_column))
                    width = min(width, self.viewport().width()-others-self._minimum_widths[-1])
            self._base_widths[column] = max(minimum, width)
            self.fit_columns()
        self._row_timer.start(0)

    def fit_columns(self):
        if not self._base_widths or self._adjusting_columns:
            return
        self._adjusting_columns = True
        try:
            widths = [max(a,b) for a,b in zip(self._base_widths,self._minimum_widths)]
            available = self.viewport().width()
            if sum(self._minimum_widths) <= available:
                # A smaller window may require surrendering prior manual expansion,
                # but never the content/header minima or the final section.
                overflow = max(0,sum(widths)-available)
                for c in reversed(range(len(widths)-1)):
                    reduction = min(overflow,widths[c]-self._minimum_widths[c])
                    widths[c] -= reduction
                    overflow -= reduction
                widths[-1] = available-sum(widths[:-1])
            for column,width in enumerate(widths):
                if self.columnWidth(column) != width:
                    self.setColumnWidth(column,width)
        finally:
            self._adjusting_columns = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_columns()
        self.fit_height()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange) and hasattr(self, "_row_timer"):
            self._row_timer.start(0)

    def wheelEvent(self, event):
        _scroll_statistics_table(self, event, super().wheelEvent)

    def fit_height(self):
        height = self.horizontalHeader().height() + self.verticalHeader().length() + 2 * self.frameWidth()
        scrollbar = self.horizontalScrollBar()
        if scrollbar.maximum() > scrollbar.minimum():
            height += scrollbar.sizeHint().height()
        self.setFixedHeight(min(450, height))


def _scroll_statistics_table(table, event, scroll):
    delta = event.angleDelta()
    horizontal = delta.x() or event.pixelDelta().x() or event.modifiers() & Qt.KeyboardModifier.ShiftModifier
    bar = table.verticalScrollBar()
    vertical = delta.y() or event.pixelDelta().y()
    movable = ((vertical > 0 and bar.value() > bar.minimum())
               or (vertical < 0 and bar.value() < bar.maximum()))
    if horizontal or movable:
        scroll(event)
    else:
        parent = table.parentWidget()
        while parent is not None and not isinstance(parent, QScrollArea):
            parent = parent.parentWidget()
        if parent is not None:
            QApplication.sendEvent(parent.viewport(), event)
        else:
            event.ignore()


class _StatisticsCards(QWidget):
    """Reflow existing cards; every row uses the available width."""
    def __init__(self, cards):
        super().__init__()
        self.setMinimumWidth(0)
        self.cards = cards
        self.columns = 0
        self.units = 0
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(BLOCK_SPACING)
        self.reflow()

    def reflow(self):
        count = len(self.cards)
        preferred = 3 if count in (5, 6) else min(4, max(1, count))
        minimum = self.fontMetrics().horizontalAdvance("0") * 20 + 2 * PAGE_MARGIN
        columns = min(preferred, max(1, (self.width()+BLOCK_SPACING) // (minimum+BLOCK_SPACING)))
        if self.columns == columns:
            return
        self.columns = columns
        grid = self.layout()
        while grid.count():
            grid.takeAt(0)
        for column in range(self.units):
            grid.setColumnStretch(column, 0)
        rows = [self.cards[i:i+columns] for i in range(0, count, columns)]
        self.units = math.lcm(*(len(row) for row in rows)) if rows else 1
        for column in range(self.units):
            grid.setColumnStretch(column, 1)
        for row_index, row in enumerate(rows):
            span = self.units // len(row)
            for column, card in enumerate(row):
                grid.addWidget(card, row_index, column * span, 1, span)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reflow()


def _cards(layout, values):
    cards, labels = [], []
    for title, value in values:
        card = QFrame()
        card.setMinimumWidth(0)
        card.setProperty("class", "statisticsCard")
        box = QVBoxLayout(card)
        box.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        box.setSpacing(COMPACT_SPACING)
        box.addWidget(_label(title))
        number = _label(_cell_text(value, percentage="%" in title or (isinstance(value, str) and value.endswith("%"))))
        if isinstance(value, report.ReportMeasure):
            number.setToolTip(value.tooltip)
        number.setProperty("class", "statisticsNumber")
        box.addWidget(number)
        cards.append(card)
        labels.append(number)
    layout.addWidget(_StatisticsCards(cards))
    return labels


def _table(layout, headers, rows, *, display=None):
    table = _StatisticsTable(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().hide()
    table.setRowCount(len(rows))
    text_columns = set()
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            percentage = report.percentage_cell(headers, values, column)
            item = QTableWidgetItem(_cell_text(value, display=display, percentage=percentage))
            alignment = Qt.AlignmentFlag.AlignRight if _numeric(value, percentage=percentage) else Qt.AlignmentFlag.AlignLeft
            item.setTextAlignment(alignment | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, column, item)
            item.setData(Qt.ItemDataRole.UserRole, value)
            if isinstance(value, report.ReportMeasure):
                item.setToolTip(value.tooltip)
            if not _numeric(value, percentage=percentage) and value is not None:
                text_columns.add(column)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(False)
    table.ensurePolished()
    header.ensurePolished()
    table.resizeColumnsToContents()
    for column, title in enumerate(headers):
        option = QStyleOptionHeader()
        option.initFrom(header)
        option.text = title.upper()
        header_width = header.style().sizeFromContents(QStyle.ContentsType.CT_HeaderSection, option, QSize(), header).width()
        content_width = table.columnWidth(column)
        if column in text_columns:
            content_width = min(440, content_width)
        numeric_width = max((table.fontMetrics().horizontalAdvance(table.item(row, column).text()) + 2 * CONTROL_SPACING + 4
                             for row in range(table.rowCount())
                             if table.item(row, column) is not None
                             and table.item(row, column).textAlignment() & Qt.AlignmentFlag.AlignRight), default=0)
        table.setColumnWidth(column, max(110, header_width, content_width, numeric_width))
    table._base_widths = [table.columnWidth(c) for c in range(len(headers))]
    table._minimum_widths = list(table._base_widths)
    table._flex_column = max(0, len(headers)-1)
    table._last_minimum = table._base_widths[-1] if table._base_widths else 0
    table.horizontalScrollBar().rangeChanged.connect(table.fit_height)
    header.geometriesChanged.connect(table.fit_height)
    table.verticalHeader().sectionResized.connect(table.fit_height)
    table.size_rows()
    layout.addWidget(table)
    return table


def _decimal_display(value, *, money=True):
    number = Decimal(value)
    formatted = format(number, ",.2f" if money else ",f").replace(",", " ")
    return _NumericText(formatted if money or "." not in formatted else formatted.rstrip("0").rstrip("."))


def _buyer_display(value):
    if isinstance(value, Decimal):
        return _decimal_display(str(value))
    if isinstance(value, date):
        return value.strftime('%d.%m.%Y')
    return value


def _analytics_sections(layout, sections, display=_buyer_display, before_tables=None):
    """Render existing typed sections once, without deriving any statistics."""
    for section_index, (title, cards, tables, notes) in enumerate(sections):
        section = _section_heading(layout, title)
        if cards:
            _cards(section, [(label, display(value)) for label, value in cards])
        for table_index, (headers, rows) in enumerate(tables):
            for factory in (before_tables or {}).get((section_index, table_index), ()):
                section.addWidget(factory())
            table = _table(section, headers, rows, display=display)
            for column, header in enumerate(headers):
                table.setColumnWidth(column, max(table.columnWidth(column), table.fontMetrics().horizontalAdvance(header.upper()) + 32))
            table.horizontalHeader().sectionResized.connect(table.resizeRowsToContents)
            table.resizeRowsToContents()
            table.fit_height()
        if notes:
            section.addWidget(_label('\n'.join(f'{label}: {text}' for label, text in notes)))



class _ReportTableModel(QAbstractTableModel):
    def __init__(self, block, parent):
        super().__init__(parent)
        self.block = block

    def rowCount(self, parent=None):
        return len(self.block.rows)

    def columnCount(self, parent=None):
        return len(self.block.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return self.block.headers[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.block.rows[index.row()]
        column = index.column()
        value = row[column]
        if role == Qt.ItemDataRole.DisplayRole:
            return report.cell_presentation(value, percentage=report.percentage_cell(self.block.headers, row, column)).text
        if role == Qt.ItemDataRole.ToolTipRole:
            if isinstance(value, report.ReportMeasure):
                return value.tooltip
            return value if isinstance(value, str) else None
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int((Qt.AlignmentFlag.AlignRight if _numeric(value) else Qt.AlignmentFlag.AlignLeft)
                       | Qt.AlignmentFlag.AlignVCenter)
        return None


class _LargeReportTable(QTableView):
    def wheelEvent(self, event):
        _scroll_statistics_table(self, event, super().wheelEvent)


def _large_report_table(layout, block):
    table = _LargeReportTable()
    table.setObjectName("statisticsLargeTable")
    table.setModel(_ReportTableModel(block, table))
    table.setProperty("statisticsTableId", block.table_id)
    table.setProperty("statisticsContextId", block.options["context_id"])
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().hide()
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    header.setStretchLastSection(True)
    header.setMinimumSectionSize(110)
    for column, title in enumerate(block.headers):
        width = header.fontMetrics().horizontalAdvance(title.upper()) + 2 * CONTROL_SPACING + 12
        table.setColumnWidth(column, min(440, max(width, 140)))
    table.setWordWrap(False)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.verticalHeader().setDefaultSectionSize(table.fontMetrics().height() + 2 * CONTROL_SPACING)
    table.setMinimumWidth(0)
    table.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    table.setFixedHeight(340)
    layout.addWidget(table)


def _render_report_blocks(layout, blocks):
    section = layout
    for block in blocks:
        if block.kind == "heading":
            section = _section_heading(layout, block.title)
        elif block.kind == "cards":
            _cards(section, block.values)
        elif block.kind == "table":
            target = section
            if details := block.options.get("details"):
                toggle = QToolButton()
                toggle.setText(details)
                toggle.setCheckable(True)
                toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
                toggle.setArrowType(Qt.ArrowType.RightArrow)
                toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                section.addWidget(toggle)
                panel = QWidget()
                target = QVBoxLayout(panel)
                target.setContentsMargins(0, 0, 0, 0)
                section.addWidget(panel)
                panel.hide()
                toggle.toggled.connect(panel.setVisible)
                toggle.toggled.connect(lambda opened, button=toggle: button.setArrowType(
                    Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow))
            if block.table_id.startswith("unified.") and len(block.rows) > 250:
                _large_report_table(target, block)
                continue
            table = _table(target, block.headers, block.rows, display=_buyer_display)
            if block.options.get("context_id"):
                table.setProperty("statisticsContextId", block.options["context_id"])
            if block.table_id:
                table.setProperty("statisticsTableId", block.table_id)
            if block.options.get("nowrap"):
                table.setWordWrap(False)
            for column in block.options.get("full_width_columns", ()):
                natural = table.sizeHintForColumn(column)
                table._minimum_widths[column] = max(table._minimum_widths[column], natural)
                table._base_widths[column] = table._minimum_widths[column]
                table.fit_columns()
            table.size_rows()
            if table_explanation(block.table_id) is not None:
                target.addWidget(StatisticsTableExplanation(block.table_id))
        elif block.kind == "text":
            label=_label(block.title)
            if block.options.get("context"):
                label.setProperty("class","statisticsContext")
            section.addWidget(label)
        elif block.kind == "chart":
            section.addWidget(analytics_chart(block.data) if block.analytics else charts.create_chart(block.data))
        elif block.kind == "chart_row":
            section.addWidget(StatisticsChartRow(block.data))
        elif block.kind == "rfm":
            section.addWidget(RfmSegmentsWidget(choices={(window, currency): data for window, currency, data in block.choices}))
        elif block.kind == "heatmap":
            section.addWidget(CohortRetentionHeatmap(data=block.data, compact=block.options.get("compact", False)))
        else:
            raise ValueError("Unknown report block")


def _customer_overview_page(layout, result, sections):
    _render_report_blocks(layout, report._blocks(report._customer_overview_page, result, sections))


def _customer_purchases_page(layout, result, sections):
    _render_report_blocks(layout, report._blocks(report._customer_purchases_page, result, sections))


def _customer_money_page(layout, result, sections):
    _render_report_blocks(layout, report._blocks(report._customer_money_page, result, sections))


def _customer_rfm_page(layout, result):
    _render_report_blocks(layout, report._blocks(report._customer_rfm_page, result))


def _customer_cohorts_page(layout, result):
    _render_report_blocks(layout, report._blocks(report._customer_cohorts_page, result))


def _scroll_page():
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
    layout.setSpacing(SECTION_SPACING)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    scroll.setWidget(content)
    return scroll, layout



def _actions_page(layout, result):
    _render_report_blocks(layout, report._blocks(report._actions_page, result))



def _products_page(layout, result):
    _render_report_blocks(layout, report._blocks(report._products_page, result))


def _orders_page(layout, result):
    _render_report_blocks(layout, report._blocks(report._orders_page, result))


class DatasetStatisticsTab(QWidget):
    calculation_running_changed = pyqtSignal(bool)

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("datasetStatisticsTab")
        self.window = window
        self.cache_path = CACHE_PATH
        self.displayed_result = None
        self.report = None
        self._display_request = None
        self._running_request = None
        self._presentation_task = None
        self._saved_report_requested = False
        self._scan_from = 0
        self._close_target = None
        self._quit_after_unified = False
        self._export_task = None
        self._export_snapshot = None
        self.detail_process = None
        self._detail_dialog_open = False
        self._detail_closing = self._detail_quitting = False
        self._detail_options = None
        filter_tab = getattr(window, "analysis_filter_tab", None)
        self.configured_filter = filter_tab.configured_filter if filter_tab is not None else load_filter()
        if filter_tab is not None:
            filter_tab.filter_changed.connect(self.set_analysis_filter)
            self.calculation_running_changed.connect(filter_tab.set_statistics_running)
        self.process = None
        self.pending = None
        self.buffer = bytearray()
        self.failure = None
        self.cancelled = self.closing = False
        self.kill_timer = QTimer(self)
        self.kill_timer.setSingleShot(True)
        self.kill_timer.timeout.connect(self._kill)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        layout.setSpacing(BLOCK_SPACING)
        self.description = _information_label(SOURCE_DESCRIPTION)
        if self._mode() == "unified":
            self.description.setObjectName("statisticsSummary")
        self.description.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.description)
        self._legacy_card_titles = (
            "Количество взаимодействий", "Количество заказов", "Количество позиций в заказах", PROFILE_SLICE_LABEL)
        self.card_labels = _cards(layout, [(name, None) for name in self._legacy_card_titles])
        self.card_titles = [label.parentWidget().layout().itemAt(0).widget() for label in self.card_labels]
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(CONTROL_SPACING)

        self.refresh = QPushButton(
            QIcon(str(ICONS_DIR / "statistic.png")),
            " Рассчитать статистику",
        )
        self.cancel_button = QPushButton(
            QIcon(str(ICONS_DIR / "failure.png")),
            " Отменить",
        )
        self.export_button = QPushButton(QIcon(str(ICONS_DIR / "export.png")), " Экспорт отчета")
        self.export_button.setIconSize(QSize(17, 17))
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)

        self.refresh.setIconSize(QSize(17, 17))
        self.cancel_button.setIconSize(QSize(17, 17))
        self.cancel_button.setEnabled(False)

        self.refresh.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)

        self.progress_container = QWidget()
        self.progress_container.setMinimumWidth(180)
        self.progress_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        progress_layout = QGridLayout(self.progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(0)

        self.progress = QProgressBar()
        self.progress.setObjectName("statisticsProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.status = _ProgressStatus(self.progress, "В фоновом режиме будут рассчитаны данные из Mindbox.")
        progress_layout.addWidget(self.progress, 0, 0)
        progress_layout.addWidget(self.status, 0, 0)

        controls.addWidget(self.refresh)
        controls.addWidget(self.export_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.progress_container, 1)

        layout.addLayout(controls)
        # Compatibility references point to the existing upper controls.
        self.detail_status=self.status
        self.detail_progress=self.progress
        self.detail_cancel=self.cancel_button
        self._detail_force_available=False
        self.last_export_diagnostic=None
        self.sections = QTabWidget()
        layout.addWidget(self.sections, 1)
        self._pages()
        window.installEventFilter(self)
        QApplication.instance().installEventFilter(self)
        QApplication.instance().aboutToQuit.connect(self._shutdown)
        timer = getattr(window, "_status_reset_timer", None)
        if timer is not None:
            # Reuse the shared timer: a reset from another operation must not
            # leave an active statistics process displaying "ready".
            timer.timeout.connect(self._keep_processing_status)
        self._update_description()
        self._update_export_button()
        if self._mode() == "legacy":
            saved = load_result(self.cache_path)
            if saved is not None:
                self.render(saved)

    def _calculation_source(self):
        return "unified", "INDEXED"

    def showEvent(self, event):
        super().showEvent(event)
        if self._saved_report_requested or self._mode() != "unified":
            return
        self._saved_report_requested = True
        if self._busy() or self.displayed_result is not None:
            return
        self.status.setText("Загрузка сохраненной статистики…")
        self.progress.setRange(0, 0)
        task = _SavedStatisticsTask()
        self._presentation_task = task
        task.signals.finished.connect(self._saved_presented, Qt.ConnectionType.QueuedConnection)
        self._update_export_button()
        QThreadPool.globalInstance().start(task)

    def _saved_presented(self, result, payload, error):
        self._presentation_task = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        if error:
            self.status.setText(UNIFIED_ERRORS[error])
        else:
            model, request = payload
            self._display(result, model, request, update_status=True)
        self._update_export_button()

    def _mode(self):
        return self._calculation_source()[0]

    def _request(self):
        mode, source = self._calculation_source()
        return mode, source, self.configured_filter, "", ""

    def _busy(self):
        return self.process is not None or self._presentation_task is not None

    def _update_export_button(self):
        detail_busy = self.detail_process is not None or self._detail_dialog_open
        idle = not self._busy() and self._export_task is None and not detail_busy
        self.export_button.setEnabled(idle and self.displayed_result is not None)
        self.refresh.setEnabled(idle)

    def _reset_export_progress(self):
        self.progress.setRange(0,100)
        self.progress.setValue(0)
        self.cancel_button.setText(" Отменить")
        self.cancel_button.setEnabled(False)
        self._detail_force_available=False

    def export(self):
        if (self._busy() or self.displayed_result is None or self._export_task is not None
                or self.detail_process is not None or self._detail_dialog_open):
            return
        captured=deepcopy(self.displayed_result)
        try:
            captured=validate_report_snapshot(captured)
        except Exception:
            self.status.setText(DETAIL_ERRORS["REPORT_CONTEXT_MISMATCH"])
            return
        self._detail_dialog_open=True
        self._update_export_button()
        try:
            value,_=QFileDialog.getSaveFileName(self,"Экспорт отчета",
                str(PROJECT_ROOT.parent/suggested_report_filename(captured)),"ZIP-архив (*.zip)")
        finally:
            self._detail_dialog_open=False
            self._update_export_button()
        if not value:
            return
        target=Path(value if value.lower().endswith(".zip") else value+".zip")
        if target.exists() and target!=Path(value):
            answer=QMessageBox.question(self,"Заменить ZIP?",
                "Заменить существующий ZIP только после успешного экспорта?",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
            if answer!=QMessageBox.StandardButton.Yes:
                return
        try:
            target=validate_report_destination(target)
        except Exception as error:
            code=getattr(error,"code","INVALID_EXPORT_PATH")
            self.status.setText(DETAIL_ERRORS.get(code,DETAIL_ERRORS["INVALID_EXPORT_PATH"]))
            return
        detail = applied_filter = None
        if isinstance(captured, UnifiedResult):
            if self._display_request is None:
                self.status.setText(DETAIL_ERRORS["REPORT_CONTEXT_MISMATCH"])
                return
            _, _, selection, product, category = self._display_request
            applied_filter = filter_snapshot(selection, product, category)
        if isinstance(captured, UnifiedResult) and captured.metadata["detail"]["available"]:
            self._detail_dialog_open = True
            self._update_export_button()
            try:
                choice = QMessageBox.question(
                    self, "Строки заказов и покупателей",
                    "Включить детальные таблицы заказов, позиций и покупателей в ZIP?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.No,
                )
            finally:
                self._detail_dialog_open = False
                self._update_export_button()
            if choice == QMessageBox.StandardButton.Cancel:
                return
            if choice == QMessageBox.StandardButton.Yes:
                detail = applied_filter
        self._export_snapshot=captured
        self.status.setToolTip("")
        options = {"detail": detail} if detail else {"applied_filter": applied_filter} if applied_filter else {}
        self._start_details(dict(destination=target), snapshot=captured, **options)

    def export_details(self):
        """Compatibility entry point for the single report operation."""
        self.export()

    def _start_details(self,options,*,snapshot=None,detail=None,applied_filter=None):
        if self.detail_process is not None or self._busy() or self._export_task is not None:
            return
        self._detail_closing=self._detail_quitting=False
        self._detail_force_available=False
        self._detail_options=dict(options)
        self.detail_process=DetailExportProcess(self)
        self.detail_process.progress.connect(self._detail_progress)
        self.detail_process.completed.connect(self._detail_finished)
        self.detail_process.stalled.connect(self._detail_stalled)
        self.status.setText("Подготовка отчета…")
        self.progress.setRange(0,0)
        self.cancel_button.setEnabled(True)
        self._update_export_button()
        self.calculation_running_changed.emit(True)
        self._keep_processing_status()
        captured = snapshot if snapshot is not None else deepcopy(self.displayed_result)
        presentation_options = {"detail": detail} if detail else {"applied_filter": applied_filter} if applied_filter else {}
        self.detail_process.start(options, report_result=captured, **presentation_options)

    def _detail_progress(self,message):
        if self.detail_process is not None and not self.detail_process.cancel_requested:
            stage=DETAIL_STAGES[message["stage"]]
            self.status.setText(stage+(f" · обработано: {message['processed']:,}".replace(","," ") if message["processed"] else ""))

    def cancel_details(self):
        self.cancel_button.setEnabled(False)
        if self._export_task is not None:
            self._export_task.cancel()
            self.status.setText("Отмена проверки источников…")
        elif self.detail_process is not None:
            self.status.setText("Отмена экспорта… Ожидается закрытие и очистка временных файлов.")
            self.detail_process.cancel()

    def _detail_stalled(self):
        if self.detail_process is not None:
            self._detail_force_available=True
            self.cancel_button.setText(" Завершить принудительно…")
            self.cancel_button.setEnabled(True)
            self.status.setText("Worker еще завершает отмену. Можно ждать или завершить принудительно; очистка не гарантируется.")

    def force_details(self):
        if self.detail_process is not None:
            answer=QMessageBox.warning(self,"Принудительное завершение",
                "Временные файлы могут остаться. Готовый ZIP мог быть сохранен непосредственно перед остановкой. Завершить процесс?",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
            if answer==QMessageBox.StandardButton.Yes:
                self.cancel_button.setEnabled(False)
                self.detail_process.force_stop()

    def _detail_finished(self,message):
        worker,self.detail_process=self.detail_process,None
        if worker is None:
            return
        self.last_export_diagnostic=deepcopy(getattr(worker,"diagnostic",{}))
        worker.deleteLater()
        self._reset_export_progress()
        if message["event"]=="finished":
            self.status.setText("Экспорт завершен. Отчет сохранен. "
                +str(self._detail_options["destination"])
                +(" Запрос отмены поступил после завершения публикации." if worker.cancel_requested else ""))
        elif message["event"]=="cancelled":
            self.status.setText("Экспорт отменен. Временное состояние очищено; прежний ZIP сохранен.")
        else:
            code=message["code"]
            text=DETAIL_ERRORS[code]+" Код: "+code
            if "technical_path" in message:
                text+=" Путь для проверки: "+message["technical_path"]
            if code in ("FORCED_STOP","WORKER_CRASHED","WORKER_PROTOCOL_ERROR") or code.startswith("SENSITIVE_"):
                text+=(" Проверьте файлы .statistics-report-* в "+str(Path(self._detail_options["destination"]).parent)
                       +". Автоматическое удаление неизвестных файлов не выполняется.")
            self.status.setText(text)
        if self.last_export_diagnostic:
            self.status.setToolTip(self.status.text() + "\n\n" + json.dumps(
                self.last_export_diagnostic, ensure_ascii=False, indent=2))
        self._export_snapshot=None
        self.calculation_running_changed.emit(False)
        self._update_export_button()
        schedule_status_reset(self.window,0)
        if self._detail_quitting:
            QTimer.singleShot(0,QApplication.instance().quit)
        elif self._detail_closing:
            QTimer.singleShot(0,self.window.close)


    def set_analysis_filter(self, selection):
        self.configured_filter = selection
        self._update_description()

    def _update_description(self):
        request = self._request()
        text = SOURCE_DESCRIPTION.replace("Отбор не установлен", self.configured_filter.summary())
        if request[0] == "unified":
            if isinstance(self.displayed_result, UnifiedResult):
                text = self.report.summary
                if self._display_request != request:
                    text += "\nПоказан отчет с прежним отбором."
            else:
                text = ""
            self.description.setText(text)
            self.description.setVisible(bool(text))
            return
        if self.displayed_result is not None:
            text += " Дата расчета: " + _date(self.displayed_result["calculated_at"])
            if AnalysisFilter.from_dict(self.displayed_result["analysis_filter"]) != self.configured_filter:
                text += ". Отбор изменен → требуется пересчитать статистику."
        self.description.setText(text)

    def _pages(self):
        outer_index = max(0, self.sections.currentIndex())
        previous = getattr(self, "customer_sections", None)
        active_group = (previous.currentWidget().property("statisticsGroup")
                        if previous is not None and previous.currentWidget() is not None else "overview")
        while self.sections.count():
            page = self.sections.widget(0)
            self.sections.removeTab(0)
            page.deleteLater()
        layouts = []
        for index, name in enumerate(report.PAGE_LABELS):
            if index == 3:
                self.customer_sections = QTabWidget()
                self.customer_sections.setObjectName("statisticsCustomerTabs")
                self.customer_sections.setDocumentMode(True)
                self.customer_sections.tabBar().setDrawBase(False)
                self.customer_sections.setStyleSheet(
                    "QTabWidget#statisticsCustomerTabs::pane { border: 0; }"
                    "QTabWidget#statisticsCustomerTabs > QTabBar::tab { height: 30px; padding: 0px 14px; }")
                customer_layouts = {}
                for group, label in zip(CUSTOMER_GROUP_ORDER, CUSTOMER_TAB_LABELS):
                    scroll, layout = _scroll_page()
                    scroll.setProperty("statisticsGroup", group)
                    scroll.setObjectName("statisticsCustomer_" + group)
                    self.customer_sections.addTab(scroll, label)
                    customer_layouts[group] = layout
                self.customer_sections.setCurrentIndex(
                    CUSTOMER_GROUP_ORDER.index(active_group) if active_group in CUSTOMER_GROUP_ORDER else 0)
                self.sections.addTab(self.customer_sections, name)
                layouts.append(customer_layouts)
            else:
                scroll, layout = _scroll_page()
                self.sections.addTab(scroll, name)
                layouts.append(layout)
        self.sections.setCurrentIndex(outer_index)
        return layouts

    def start(self):
        self._start_calculation()

    def _start_calculation(self):
        if self._busy() or self.detail_process is not None or self._export_task is not None or self._detail_dialog_open:
            return
        request = self._request()
        self._running_request = request
        self.buffer.clear()
        self._scan_from = 0
        self.pending = self.failure = None
        self.cancelled = False
        self.refresh.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("Проверка индекса и расчет статистики…")
        process = QProcess(self)
        self.process = process
        self._update_export_button()
        self.calculation_running_changed.emit(True)
        timer = getattr(self.window, "_status_reset_timer", None)
        if timer is not None:
            timer.stop()
        self._keep_processing_status()
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.readyReadStandardOutput.connect(lambda: self._read() if self.process is process else None)
        process.readyReadStandardError.connect(lambda: process.readAllStandardError())
        process.finished.connect(lambda code, state: self._finished(code, state) if self.process is process else None)
        process.errorOccurred.connect(lambda error: self._error(error) if self.process is process else None)
        mode, backend, self.running_filter, _, _ = request
        args = ["-u", str(PROJECT_ROOT / "scripts/dataset_statistics.py"),
                "--analysis-filter", json.dumps(self.running_filter.to_dict(), ensure_ascii=True)]
        if backend != "AUTO":
            args.extend(("--backend", backend))
        if mode == "unified":
            args.extend(("--mode", "unified"))
        process.start(sys.executable, args)

    def _keep_processing_status(self):
        if self._busy():
            set_status_processing(self.window, PROCESSING_STATUS)
        elif self.detail_process is not None or self._export_task is not None:
            set_status_processing(self.window, "Экспорт отчета…")

    def _read(self):
        if self.process is None:
            return
        self.buffer.extend(bytes(self.process.readAllStandardOutput()))
        unified = self._running_request[0] == "unified"
        while True:
            newline = self.buffer.find(b"\n", self._scan_from)
            if newline < 0:
                self._scan_from = len(self.buffer)
                break
            line = bytes(self.buffer[:newline])
            del self.buffer[:newline + 1]
            self._scan_from = 0
            try:
                if unified and _RESULT_LINE.match(line):
                    if self.pending is not None:
                        raise ValueError
                    if not self.cancelled:
                        self.pending = line
                    continue
                message = json.loads(line)
                kind, value = message["event"], message["value"]
                if kind == "progress" and not self.cancelled:
                    if unified:
                        self.status.setText(value.translate(str.maketrans("\u0451\u0401", "еЕ")))
                    else:
                        source, separator, count = value.partition(":")
                        self.status.setText("Расчет → " + PROGRESS_LABELS.get(source, source) + separator + count)
                elif kind == "unified_error" and unified:
                    self.failure = UNIFIED_ERRORS.get(value, UNIFIED_ERRORS["UNIFIED_FAILED"])
                elif kind == "source_changed" and unified:
                    if self.displayed_result is not None:
                        self.displayed_result = self.report = self._display_request = None
                        for title, label, name in zip(self.card_titles, self.card_labels, self._legacy_card_titles):
                            title.setText(name)
                            label.setText("—")
                            label.setToolTip("")
                        self._pages()
                        self._update_description()
                elif kind == "result" and not unified:
                    if self.pending is not None:
                        raise ValueError
                    self.pending = validate_result(value)
                    if AnalysisFilter.from_dict(self.pending["analysis_filter"]) != self.running_filter:
                        raise ValueError
                elif kind == "error":
                    self.failure = format_error(value, context="Ошибка расчета статистики")
                elif kind != "progress":
                    raise ValueError
            except (ValueError, KeyError, TypeError, AttributeError, RecursionError):
                self.failure = "Некорректный ответ процесса статистики."

    def _error(self, error):
        if self.process is not None and error == QProcess.ProcessError.FailedToStart:
            self.failure = "Не удалось запустить процесс статистики: " + self.process.errorString()
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _finished(self, code, exit_status):
        if self.process is None:
            return
        self._read()
        if self.buffer.strip():
            self.failure = "Некорректный ответ процесса статистики."
        process, self.process = self.process, None
        self.kill_timer.stop()
        process.deleteLater()
        request = self._running_request
        ok = not (self.cancelled or code or exit_status != QProcess.ExitStatus.NormalExit
                  or self.failure or self.pending is None)
        if ok and request[0] == "unified":
            task = _UnifiedPresentationTask(self.pending, request)
            self.pending = None
            self._presentation_task = task
            self.status.setText("Подготовка таблиц и графиков…")
            task.signals.finished.connect(self._unified_presented, Qt.ConnectionType.QueuedConnection)
            QThreadPool.globalInstance().start(task)
            return
        if self.cancelled:
            self.status.setText("Расчет отменен. Показанные ранее данные не обновлены.")
        elif not ok:
            self.status.setText((self.failure or "Процесс статистики завершился без результата.")
                                + " Показанные ранее данные не обновлены.")
        else:
            try:
                save_result(self.cache_path, self.pending)
            except (OSError, ValueError, TypeError, RecursionError) as error:
                self.status.setText(f"Не удалось сохранить результат статистики: {error}. "
                                    "Показанные ранее данные не обновлены.")
            else:
                if request == self._request():
                    self.render(self.pending)
                else:
                    self.status.setText("Расчет завершен для ранее выбранного отбора.")
        self._end_calculation()

    def _unified_presented(self, result, model, error):
        self._presentation_task = None
        request = self._running_request
        if self.cancelled or error == "CANCELLED":
            self.status.setText("Расчет отменен. Показанные ранее данные не обновлены.")
        elif error:
            self.status.setText(UNIFIED_ERRORS[error] + " Показанные ранее данные не обновлены.")
        else:
            if request == self._request():
                self._display(result, model, request, update_status=True)
            else:
                self.status.setText("Единая статистика готова для ранее выбранного источника или отбора.")
        self._end_calculation()

    def _end_calculation(self):
        self.pending = self._running_request = None
        self.calculation_running_changed.emit(False)
        self._reset_export_progress()
        self._update_export_button()
        schedule_status_reset(self.window, 0)
        if self._quit_after_unified:
            self._quit_after_unified = False
            QTimer.singleShot(0, QApplication.instance().quit)
        elif self._close_target is not None:
            target, self._close_target = self._close_target, None
            QTimer.singleShot(0, target.close)
        elif self.closing:
            QTimer.singleShot(0, self.window.close)

    def cancel(self):
        if self.detail_process is not None or self._export_task is not None:
            if self._detail_force_available:
                self.force_details()
            else:
                self.cancel_details()
        elif self._busy() and not self.cancelled:
            self.cancelled = True
            self.cancel_button.setEnabled(False)
            self.status.setText("Отмена расчета…")
            if self._presentation_task is not None:
                self._presentation_task.cancelled.set()
            elif self._running_request[0] == "unified":
                self.process.write(b"cancel\n")
                self.kill_timer.start(35000)
            else:
                self.process.terminate()
                self.kill_timer.start(1500)

    def _kill(self):
        if self.process is not None:
            self.process.kill()

    def _shutdown(self):
        if self._presentation_task is not None:
            self._presentation_task.cancelled.set()
        if self._export_task is not None:
            self._export_task.cancel()
        if self.detail_process is not None:
            self.detail_process.shutdown()
        if self.process is not None:
            process = self.process
            process.kill()
            process.waitForFinished(1000)

    def eventFilter(self, watched, event):
        if (self.detail_process is not None and event.type() in (QEvent.Type.Close, QEvent.Type.Quit)
                and watched in (self.window, QApplication.instance())):
            event.ignore()
            self._detail_quitting = self._detail_quitting or event.type() == QEvent.Type.Quit
            self._detail_closing = self._detail_closing or watched is self.window
            self.cancel_details()
            return True
        request = self._running_request
        if (event.type() in (QEvent.Type.Close, QEvent.Type.Quit)
                and request is not None and request[0] == "unified" and self._busy()):
            if watched is QApplication.instance() and event.type() == QEvent.Type.Quit:
                event.ignore()
                self._quit_after_unified = True
                self.cancel()
                return True
            if watched in (self, self.window) and event.type() == QEvent.Type.Close:
                event.ignore()
                self._close_target = watched
                self.cancel()
                return True
        if watched is self.window and event.type() == QEvent.Type.Close and self.process is not None:
            event.ignore()
            self.closing = True
            self.cancel()
            return True
        return super().eventFilter(watched, event)

    def render(self, result):
        if isinstance(result, UnifiedResult):
            request = self._request()
            if not _matches_unified_request(result, request):
                return
            model = unified_presentation(result, applied_filter=filter_snapshot(*request[2:]))
        else:
            if self._mode() != "legacy":
                return
            request = ("legacy", self._calculation_source()[1], AnalysisFilter.from_dict(result["analysis_filter"]), "", "")
            model = report.report_presentation(result)
        self._display(result, model, request, update_status=True)

    def _display(self, result, model, request, *, update_status=False):
        self.report = model
        for label, title, (name, value) in zip(self.card_labels, self.card_titles, model.global_cards):
            title.setText(name)
            label.setText(_cell_text(value) if isinstance(value, report.ReportMeasure) else _number(value))
            label.setToolTip(value.tooltip if isinstance(value, report.ReportMeasure) else "")
        self.displayed_result, self._display_request = result, request
        self._update_export_button()
        self._update_description()
        if update_status or not self._busy():
            self.status.setText(model.status)
        layouts = self._pages()
        for page, layout in zip(self.report.pages, layouts):
            if page.key == "customers":
                for group in page.groups:
                    _render_report_blocks(layout[group.key], group.blocks)
            else:
                _render_report_blocks(layout, page.groups[0].blocks)


def create_dataset_statistics_tab(window):
    window.dataset_statistics_tab = DatasetStatisticsTab(window)
    window.tabs.addTab(window.dataset_statistics_tab, "Статистика и анализ")
