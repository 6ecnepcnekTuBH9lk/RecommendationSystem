"""Flat product table and bounded photo targets using the existing card queue."""

from collections import deque

from PyQt6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QSize, Qt, QTimer
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QApplication, QAbstractItemView, QHeaderView, QLabel, QStyledItemDelegate, QTableView

from Application.customer_history_records import HistoryKind
from Application.photo.photo_processing import _set_photo_label


HEADERS = {
    kind: ("Фото", "Код товара", "Название товара", "Вид номенклатуры", "Коллекция",
           "Дата заказа" if kind is HistoryKind.ORDER else "Дата события")
    for kind in HistoryKind
}
PHOTO_SIZE = QSize(140, 210)
PHOTO_MARGIN = 8
ROW_HEIGHT = PHOTO_SIZE.height() + 2 * PHOTO_MARGIN
INITIAL_WIDTHS = (156, 150, 300, 220, 170, 185)
MIN_AUTO_WIDTHS = (156, 115, 180, 165, 110, 155)


class HistoryTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.kind = HistoryKind.ORDER
        self._rows = []

    def reset(self, kind):
        self.beginResetModel()
        self.kind, self._rows = kind, []
        self.endResetModel()

    def append(self, rows):
        if not rows:
            return
        start = len(self._rows)
        self.beginInsertRows(QModelIndex(), start, start + len(rows) - 1)
        self._rows.extend(rows)
        self.endInsertRows()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEADERS[self.kind])

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        if orientation is Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole and 0 <= section < self.columnCount():
            return HEADERS[self.kind][section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return row.values[index.column()]
        if role == Qt.ItemDataRole.UserRole:
            return row
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        return None


class _PhotoTarget(QLabel):
    """One hidden queue target per visible URL; discarded when it completes."""
    def __init__(self, done, parent):
        super().__init__(parent)
        self.done = done
        self.hide()

    def set_source_pixmap(self, pixmap):
        self.done()

    def setText(self, text):
        super().setText(text)
        if text != "Загрузка фото…":
            self.done()


class _ProductDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.font.setPointSizeF(option.font.pointSizeF() + 2)

    def sizeHint(self, option, index):
        return QSize(PHOTO_SIZE.width() + 2 * PHOTO_MARGIN, ROW_HEIGHT)

    def paint(self, painter, option, index):
        if index.column() != 0:
            super().paint(painter, option, index)
            return
        super().paint(painter, option, index)
        view = self.parent()
        row = index.data(Qt.ItemDataRole.UserRole)
        url, owner = row.photo_url, view.photo_owner
        pixmap = owner._img_cache.get(url) if owner is not None and url else None
        if pixmap is not None:
            rect = option.rect.adjusted(PHOTO_MARGIN, PHOTO_MARGIN, -PHOTO_MARGIN, -PHOTO_MARGIN)
            scaled = pixmap.scaled(rect.size().boundedTo(PHOTO_SIZE), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(rect.x() + (rect.width() - scaled.width()) // 2,
                               rect.y() + (rect.height() - scaled.height()) // 2, scaled)
        else:
            painter.save()
            painter.setPen(option.palette.text().color())
            painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, "—")
            painter.restore()
            if url and owner is not None:
                view.ensure_photo(url)


class HistoryTable(QTableView):
    def __init__(self, photo_owner=None):
        super().__init__()
        self.photo_owner = photo_owner
        self._photo_targets = {}
        self._attempted = set()
        self._photo_token = 0
        self._photo_generation = getattr(photo_owner, "_img_gen", 0)
        self._manual_widths = {}
        self._fitting_columns = False
        self.setMinimumWidth(0)
        self.setAlternatingRowColors(False)
        self.setStyleSheet("QTableView { selection-background-color: transparent; selection-color: palette(text); }"
                           "QTableView::item:selected { background-color: transparent; color: palette(text); }")
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setSortingEnabled(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setItemDelegate(_ProductDelegate(self))
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.rows = HistoryTableModel(self)
        self.setModel(self.rows)
        self.empty_label = QLabel(self.viewport())
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.empty_label.hide()
        self.configure_columns()
        self.horizontalHeader().sectionResized.connect(self._column_resized)
        self.verticalHeader().sectionResized.connect(self._update_height)
        self.rows.rowsInserted.connect(self._update_height)
        self.rows.modelReset.connect(self._update_height)
        self._fit_columns()

    def configure_columns(self):
        header = self.horizontalHeader()
        header.setMinimumSectionSize(64)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        for column, width in enumerate(INITIAL_WIDTHS):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self.setColumnWidth(column, width)
        header.setStretchLastSection(False)

    def _column_resized(self, column, old, width):
        if not self._fitting_columns:
            self._manual_widths[column] = width
            self._fit_columns()

    def _fit_columns(self):
        if self._fitting_columns:
            return
        automatic = [column for column in range(6) if column not in self._manual_widths]
        available = self.viewport().width() - sum(self._manual_widths.values())
        extra = max(0, available - sum(MIN_AUTO_WIDTHS[column] for column in automatic))
        weight = sum(INITIAL_WIDTHS[column] for column in automatic)
        self._fitting_columns = True
        try:
            for column in automatic:
                share = extra * INITIAL_WIDTHS[column] // weight
                self.setColumnWidth(column, MIN_AUTO_WIDTHS[column] + share)
                extra -= share
                weight -= INITIAL_WIDTHS[column]
        finally:
            self._fitting_columns = False
        self._update_height()

    def _update_height(self, *args):
        rows_height = self.verticalHeader().length() if self.rows.rowCount() else self.fontMetrics().height() * 4 + 2 * PHOTO_MARGIN
        header_height = max(self.horizontalHeader().height(), self.horizontalHeader().sizeHint().height())
        scroll_height = self.horizontalScrollBar().sizeHint().height() if self.horizontalHeader().length() > self.viewport().width() else 0
        height = rows_height + header_height + 2 * self.frameWidth() + scroll_height
        if height != self.height() or self.minimumHeight() != height:
            self.setFixedHeight(height)

    def set_empty_text(self, text):
        self.empty_label.setText(text)
        self.empty_label.setGeometry(self.viewport().rect())
        self.empty_label.setVisible(bool(text) and not self.rows.rowCount())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "empty_label"):
            self.empty_label.setGeometry(self.viewport().rect())
            self._fit_columns()

    def viewportEvent(self, event):
        result = super().viewportEvent(event)
        if event.type() is QEvent.Type.Resize and hasattr(self, "empty_label"):
            self.empty_label.setGeometry(self.viewport().rect())
        return result

    def wheelEvent(self, event):
        if event.angleDelta().x() or event.pixelDelta().x() or event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            super().wheelEvent(event)
        else:
            event.ignore()  # Let the surrounding results page handle vertical scrolling.

    def append(self, rows):
        self.rows.append(rows)
        if self.rows.rowCount():
            self.empty_label.hide()

    def reset_history(self, kind):
        self.release_photos()
        self.rows.reset(kind)
        self.set_empty_text("")

    def release_photos(self):
        self._photo_token += 1
        owner = self.photo_owner
        for url, target in self._photo_targets.items():
            if owner is not None and target is not None:
                targets = owner._img_targets.get(url, [])
                remaining = [(widget, row) for widget, row in targets if widget is not target]
                if remaining:
                    owner._img_targets[url] = remaining
                else:
                    owner._img_targets.pop(url, None)
            if owner is not None and not owner._img_targets.get(url):
                owner._img_queue = deque(pending for pending in owner._img_queue if pending != url)
            if target is not None:
                target.deleteLater()
        self._photo_targets.clear()
        self._attempted.clear()
        self._photo_generation = getattr(owner, "_img_gen", 0)

    def ensure_photo(self, url):
        owner = self.photo_owner
        if owner._img_gen != self._photo_generation:
            self.release_photos()
        if url in self._attempted or url in self._photo_targets or len(self._photo_targets) >= 24:
            return
        token, generation = self._photo_token, owner._img_gen
        self._photo_targets[url] = None
        def request():
            if token != self._photo_token or generation != owner._img_gen:
                return
            def done():
                if token == self._photo_token:
                    if owner._img_cache.get(url) is None:
                        self._attempted.add(url)
                    target = self._photo_targets.pop(url, None)
                    if target is not None:
                        target.deleteLater()
                    self.viewport().update()
            target = _PhotoTarget(done, self)
            self._photo_targets[url] = target
            _set_photo_label(owner, target, url, generation)
        QTimer.singleShot(0, request)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            indexes = self.selectionModel().selectedRows()
            lines = []
            for index in indexes:
                lines.append("\t".join(str(index.siblingAtColumn(c).data() or "") for c in range(self.rows.columnCount())))
            QApplication.clipboard().setText("\n".join(lines))
            return
        super().keyPressEvent(event)
