"""Compact, paged history on the existing customer-card worker."""

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QButtonGroup, QGridLayout, QPushButton, QWidget

from Application.customer_card_data import SectionContent, ViewState
from Application.customer_history_presentation import CATEGORY_NAMES, HistoryDisplayPage, present_history
from Application.customer_history_records import HistoryKind
from Application.customer_history_service import HistoryStatus
from Application.paths import ICONS_DIR
from .customer_card_widgets import StateSection
from .customer_card_worker import CustomerHistoryRequest
from .customer_history_table import HistoryTable


CATEGORIES = tuple((CATEGORY_NAMES[kind], kind) for kind in (HistoryKind.ORDER, HistoryKind.VIEW, HistoryKind.FAVORITE))
MAX_EMPTY_PAGES = 3
STATUS_CONTENT = {
    HistoryStatus.READY: (ViewState.READY, ""),
    HistoryStatus.EMPTY: (ViewState.EMPTY, "Нет событий в выбранной категории."),
    HistoryStatus.PARTIAL: (ViewState.READY, "История может быть неполной."),
    HistoryStatus.INDEX_NOT_READY: (ViewState.UNAVAILABLE, "История пока недоступна."),
    HistoryStatus.SOURCE_UNAVAILABLE: (ViewState.UNAVAILABLE, "История пока недоступна."),
    HistoryStatus.SOURCE_INCONSISTENT: (ViewState.ERROR, "Не удалось загрузить историю."),
    HistoryStatus.STALE_CURSOR: (ViewState.STALE, "История обновилась. Повторите поиск клиента."),
    HistoryStatus.INVALID_INPUT: (ViewState.ERROR, "Не удалось загрузить историю. Повторите поиск клиента."),
}


class HistoryCategories(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.buttons = {}
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._columns = 0
        self.kind = HistoryKind.ORDER
        icons = {HistoryKind.ORDER: "order", HistoryKind.VIEW: "actions", HistoryKind.FAVORITE: "favorites"}
        for title, kind in CATEGORIES:
            button = QPushButton(QIcon(str(ICONS_DIR / (icons[kind] + ".png"))), " " + title)
            button.setIconSize(QSize(20, 20))
            button.setCheckable(True)
            button.setAccessibleName(title)
            self.group.addButton(button)
            self.buttons[kind] = button
            button.clicked.connect(lambda checked, value=kind: self.select(value))
        self.buttons[self.kind].setChecked(True)
        self.reflow()

    def select(self, kind):
        if kind == self.kind or kind not in self.buttons:
            return
        self.kind = kind
        self.buttons[kind].setChecked(True)
        self.changed.emit()

    def reflow(self):
        width = sum(button.sizeHint().width() for button in self.buttons.values()) + 2 * self.grid.spacing()
        columns = 3 if self.width() >= width else 1
        if columns == self._columns:
            return
        self._columns = columns
        while self.grid.count():
            self.grid.takeAt(0)
        for n, button in enumerate(self.buttons.values()):
            self.grid.addWidget(button, n // columns, n % columns)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reflow()


class CustomerHistorySection(StateSection):
    dispatch = pyqtSignal(object)

    def __init__(self, *, photo_owner=None):
        super().__init__("История взаимодействий", SectionContent(ViewState.INITIAL, ""), name="customerHistory")
        self.customer_id = None
        self._request_id = 0
        self._active = self._pending = None
        self._closing = False
        self.cursor = self.generation = self.source_version = None
        self._seen, self._consumed = set(), set()
        self._partial = False
        self._empty_pages = 0
        self.category = HistoryCategories()
        self.buttons = self.category.buttons
        self.category.setEnabled(False)
        self.category.changed.connect(self._category_changed)
        self.box.insertWidget(1, self.category)
        self.table = HistoryTable(photo_owner)
        self.table.hide()
        self.box.addWidget(self.table)
        self.more = QPushButton("Показать ещё")
        self.more.clicked.connect(self.load_more)
        self.more.hide()
        self.box.addWidget(self.more, alignment=Qt.AlignmentFlag.AlignHCenter)

    def set_content(self, content):
        super().set_content(content)
        self.message.setVisible(bool(content.message))

    def _reset(self):
        self._request_id += 1
        if self._active is not None:
            self._active.cancelled.set()
        self._pending = None
        self.cursor = self.generation = self.source_version = None
        self._seen.clear()
        self._consumed.clear()
        self._partial = False
        self._empty_pages = 0
        self.setProperty("historyStatus", None)
        self.table.reset_history(self.category.kind)
        self.table.hide()
        self.more.hide()

    def set_customer(self, customer_id):
        if self._closing or customer_id == self.customer_id:
            return
        self.customer_id = customer_id
        self._category_changed()

    def _category_changed(self):
        self._reset()
        self.category.setEnabled(self.customer_id is not None)
        if self.customer_id is None:
            self.set_content(SectionContent(ViewState.INITIAL, ""))
            return
        self._request(None)

    def _request(self, cursor):
        if self._closing:
            return
        request = CustomerHistoryRequest(self._request_id, self.customer_id, self.category.kind, cursor)
        self.set_content(SectionContent(ViewState.LOADING, "Загрузка истории…"))
        self.more.setEnabled(False)
        if self._active is not None:
            self._pending = request
        else:
            self._start(request)

    def _start(self, request):
        self._active = request
        self.dispatch.emit(request)

    def load_more(self):
        if self.customer_id is not None and self.cursor is not None and self._active is None and self.cursor not in self._consumed:
            self._empty_pages = 0
            self._request(self.cursor)

    def receive(self, request, delivery):
        if request is not self._active:
            return
        self._active = None
        if self._closing:
            return
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._start(pending)
            return
        if request.cancelled.is_set() or request.request_id != self._request_id or delivery is None:
            return
        if not isinstance(delivery, HistoryDisplayPage):
            delivery = present_history(delivery, request.kind)
        page = delivery.page
        if page.status in (HistoryStatus.READY, HistoryStatus.PARTIAL, HistoryStatus.EMPTY):
            if page.canonical_customer_id != self.customer_id or (self.generation is not None and (
                    page.generation != self.generation or page.source_version != self.source_version)):
                self._failed(HistoryStatus.STALE_CURSOR)
                return
            self.generation, self.source_version = page.generation, page.source_version
            self._partial |= page.status is HistoryStatus.PARTIAL
            rows = []
            for event_key, event_rows in delivery.events:
                if event_key not in self._seen:
                    self._seen.add(event_key)
                    rows.extend(event_rows)
            self.table.append(rows)
            self._consumed.add(request.cursor)
            self.cursor = page.next_cursor
            if self.cursor is not None and self.cursor in self._consumed:
                self._failed(HistoryStatus.STALE_CURSOR)
                return
            self._empty_pages = self._empty_pages + 1 if not rows else 0
            if not rows and self.cursor is not None and self._empty_pages < MAX_EMPTY_PAGES:
                self.more.hide()
                self.table.set_empty_text("Поиск товарных событий…" if not self.table.rows.rowCount() else "")
                self.table.show()
                # The worker connection is queued: bounded, serial reads remain
                # cancellable and do not recurse synchronously on the GUI stack.
                self._request(self.cursor)
                return
            has_rows = self.table.rows.rowCount() > 0
            empty_text = "На просмотренных страницах товаров нет." if self.cursor is not None else "Нет событий в выбранной категории."
            self.table.set_empty_text(empty_text if not has_rows else "")
            self.table.show()
            self.more.setText("Показать ещё" if has_rows else "Искать товары дальше")
            self.more.setVisible(self.cursor is not None)
            self.more.setEnabled(True)
            status = HistoryStatus.PARTIAL if self._partial else page.status
            state, message = STATUS_CONTENT[status]
            if status is HistoryStatus.EMPTY:
                message = ""  # The empty state belongs in the table viewport.
            self.set_content(SectionContent(state, message))
            self.setProperty("historyStatus", status.value)
        else:
            self._failed(page.status)

    def _failed(self, status):
        self._reset()
        state, message = STATUS_CONTENT[status]
        self.set_content(SectionContent(state, message))
        self.setProperty("historyStatus", status.value)

    def shutdown(self):
        self._closing = True
        self._reset()
