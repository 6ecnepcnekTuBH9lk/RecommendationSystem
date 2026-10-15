"""Paged history section of the existing customer card; no I/O in widgets."""

from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from Application.customer_card_data import SectionContent, ViewState
from Application.customer_history_records import HistoryKind, OrderSnapshot
from Application.customer_history_service import HistoryStatus
from .customer_card_widgets import StateSection, card_frame, text_label
from .customer_card_worker import CustomerHistoryRequest


CATEGORIES = (("Заказы", HistoryKind.ORDER), ("Просмотры", HistoryKind.VIEW),
              ("Избранное", HistoryKind.FAVORITE), ("Прочее", HistoryKind.UNCLASSIFIED))
STATUS_CONTENT = {
    HistoryStatus.READY: (ViewState.READY, "История загружена в пределах покрытия индекса."),
    HistoryStatus.EMPTY: (ViewState.EMPTY, "В выбранной категории и покрытии индекса событий нет."),
    HistoryStatus.PARTIAL: (ViewState.READY, "Частичная история: покрытие неполное или есть различающиеся версии событий."),
    HistoryStatus.INDEX_NOT_READY: (ViewState.UNAVAILABLE, "Индекс истории отсутствует, устарел или недоступен. Требуется отдельное построение индекса."),
    HistoryStatus.SOURCE_UNAVAILABLE: (ViewState.UNAVAILABLE, "Источники истории недоступны. Повторите загрузку позднее."),
    HistoryStatus.SOURCE_INCONSISTENT: (ViewState.ERROR, "Не удалось прочитать согласованную историю. Повторите загрузку позднее."),
    HistoryStatus.STALE_CURSOR: (ViewState.STALE, "Индекс изменился. Загрузите историю заново с первой страницы."),
    HistoryStatus.INVALID_INPUT: (ViewState.ERROR, "Некорректные параметры истории. Загрузите историю заново."),
}

# Full source values remain selectable in read-only, wrapped detail panels.
FIELD_NAMES = {
    "event": "Событие",
    "event_key": "Ключ события", "source": "Источник", "source_event_id": "ID события в источнике",
    "canonical_customer_id": "Canonical MindboxID", "source_customer_id": "Исходный MindboxID",
    "kind": "Категория", "event_datetime_utc": "Дата события (UTC)", "fingerprint": "Отпечаток версии",
    "provenance": "Происхождение", "origins": "Все исходные записи", "details": "Данные события",
    "quality": "Особенности исходных данных", "source_kind": "Тип источника",
    "component_version": "Версия компонента", "part_number": "Номер части",
    "record_ordinal": "Позиция записи (с нуля)", "observed_at_utc": "Дата наблюдения (UTC)",
    "action_id": "MindboxID действия", "system_name": "Системное имя", "action_name": "Название действия",
    "creation_datetime_utc": "Дата создания (UTC)", "channel": "Канал", "products": "Товары",
    "categories": "Категории товаров", "brand_system_name": "Бренд",
    "product_view_price": "Цена просмотра (валюта не подтверждена)",
    "product_view_is_available": "Доступность товара в источнике", "ordinal": "Позиция (с нуля)",
    "reference": "Идентификаторы товара", "keys": "Все идентификаторы", "product": "Выбранный идентификатор",
    "namespace": "Пространство имён", "value": "Значение", "name": "Название",
    "external_id": "Внешний ID", "mindbox_id": "MindboxID", "logical_order_key": "Ключ заказа",
    "snapshot_key": "Ключ снимка", "order": "Заказ", "lines": "Товарные строки",
    "order_id": "MindboxID заказа", "retail_order_id": "Номер заказа", "order_datetime_utc": "Дата заказа (UTC)",
    "total_price": "Сумма из источника (валюта не подтверждена)", "delivery_cost": "Стоимость доставки",
    "line_key": "Ключ строки снимка", "source_line_key": "Ключ исходной строки", "line_id": "ID строки",
    "source_line_id": "Исходный ID строки", "is_synthetic": "Синтетический ID строки",
    "line_number": "Номер строки", "product_name": "Название товара", "quantity": "Количество",
    "raw_status": "Исходный статус", "price_of_line": "Сумма строки (валюта не подтверждена)",
    "base_price_per_item": "Базовая цена единицы",
}


def detail_text(value, depth=0):
    """Lossless display of typed backend payloads, without HTML interpretation."""
    indent = "  " * depth
    if is_dataclass(value):
        return "\n".join(f"{indent}{FIELD_NAMES.get(f.name, f.name)}:\n{detail_text(getattr(value, f.name), depth + 1)}"
                         for f in fields(value))
    if isinstance(value, tuple):
        return "\n".join(f"{indent}[{n}]\n{detail_text(item, depth + 1)}" for n, item in enumerate(value, 1)) or indent + "Нет"
    if isinstance(value, datetime):
        value = value.isoformat()
    elif isinstance(value, Enum):
        value = value.value
    elif isinstance(value, bytes):
        value = value.hex()
    elif isinstance(value, bool):
        value = "Да" if value else "Нет"
    return indent + ("Не указано" if value is None else str(value))


def date_text(value):
    return value.isoformat(sep=" ") + " (UTC)" if value is not None else "Общая дата не определена"


def product_text(reference):
    return ", ".join(f"{key.namespace}: {key.value}" for key in reference.keys) or "Товар не указан"


def variant_text(variant):
    event, data = variant.event, variant.event.details
    lines = [f"Дата события: {date_text(event.event_datetime_utc)}",
             f"Покупатель: {event.canonical_customer_id} · исходный MindboxID: {event.source_customer_id}"]
    if isinstance(data, OrderSnapshot):
        lines.extend([f"Заказ: {data.order.order_id} · номер: {data.order.retail_order_id or 'Не указан'}",
                      f"Товарных строк: {len(data.lines)}. Количество и статусы приведены из источника."])
        for line in data.lines:
            lines.extend(["", f"Строка {line.line_number}: {line.product_name or 'Название не указано'}",
                          product_text(line.reference), f"Количество: {line.quantity} · исходный статус: {line.raw_status}",
                          f"Сумма строки: {line.price_of_line} (валюта не подтверждена)"])
    else:
        lines.append(f"Действие: {data.action_name or data.system_name} ({data.system_name})")
        lines.extend(f"Товар {p.ordinal + 1}: {product_text(p.reference)}" for p in data.products)
    lines.extend(["", "Полные исходные поля, идентификаторы и происхождение:", detail_text(variant)])
    return "\n".join(lines)


class HistoryEventCard(QWidget):
    """Create heavy detail editors only when an event/variant is expanded."""

    def __init__(self, event):
        super().__init__()
        self.event = event
        self.details = None
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        frame, self.box = card_frame()
        box.addWidget(frame)
        title = "Заказ" if event.source.value == "ORDER" else "Событие"
        self.box.addWidget(text_label(f"{title} {event.source_event_id} · {date_text(event.event_datetime_utc)}"))
        if event.has_conflicts:
            self.box.addWidget(text_label(f"Различающиеся версии: {len(event.variants)}. Показаны все; актуальная версия не определена."))
        if event.owner_conflict:
            self.box.addWidget(text_label("Принадлежность покупателю неоднозначна. Возможные владельцы: " + ", ".join(event.candidate_customer_ids)))
        if event.time_conflict or event.kind_conflict:
            self.box.addWidget(text_label("Дата или категория различаются между версиями. Смотрите значения каждой версии."))
        for n, variant in enumerate(event.variants, 1):
            data = variant.event.details
            if isinstance(data, OrderSnapshot):
                description = f"Снимок {n}: {len(data.lines)} товарных строк. " + "; ".join(
                    f"{line.product_name or product_text(line.reference)} × {line.quantity} · статус: {line.raw_status}"
                    for line in data.lines)
            else:
                description = f"{data.action_name or data.system_name} · " + "; ".join(product_text(p.reference) for p in data.products)
            label = text_label(description)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.box.addWidget(label)
        self.expand = QPushButton("Детали заказа и снимки" if event.source.value == "ORDER" else "Детали события и происхождение")
        self.expand.setCheckable(True)
        self.expand.toggled.connect(self._toggle)
        self.box.addWidget(self.expand)

    def _toggle(self, checked):
        if self.details is None:
            self.details = QWidget()
            box = QVBoxLayout(self.details)
            box.setContentsMargins(0, 0, 0, 0)
            common = text_label(f"Ключ события: {self.event.event_key}\nИсходных записей: {self.event.occurrence_count}")
            common.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box.addWidget(common)
            for number, variant in enumerate(self.event.variants, 1):
                button = QPushButton(f"{'Снимок заказа' if isinstance(variant.event.details, OrderSnapshot) else 'Версия события'} {number} · {date_text(variant.event.event_datetime_utc)}")
                button.setMinimumWidth(0)
                button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                button.setToolTip(button.text())
                button.setCheckable(True)
                box.addWidget(button)
                # Details contain all identifiers, product occurrences, source
                # statuses, precise Decimal values and every origin, unelided.
                holder = QWidget()
                layout = QVBoxLayout(holder)
                layout.setContentsMargins(0, 0, 0, 0)
                holder.hide()
                box.addWidget(holder)
                def show_variant(visible, target=holder, content=variant, target_layout=layout):
                    if visible and target_layout.count() == 0:
                        editor = QPlainTextEdit()
                        editor.setReadOnly(True)
                        editor.setPlainText(variant_text(content))
                        editor.setMinimumWidth(0)
                        editor.setFixedHeight(320)
                        target_layout.addWidget(editor)
                    target.setVisible(visible)
                button.toggled.connect(show_variant)
            self.box.addWidget(self.details)
        self.details.setVisible(checked)


class CustomerHistorySection(StateSection):
    dispatch = pyqtSignal(object)

    def __init__(self):
        super().__init__("История взаимодействий", SectionContent(ViewState.INITIAL, "Выберите покупателя."), name="customerHistory")
        self.customer_id = None
        self._request_id = 0
        self._active = self._pending = None
        self._closing = False
        self.cursor = None
        self.generation = None
        self.source_version = None
        self._seen = set()
        self._consumed = set()
        self._partial = False
        self.cards = []
        self.category = QComboBox()
        self.category.setMinimumWidth(0)
        self.category.setAccessibleName("Категория истории взаимодействий")
        for title, kind in CATEGORIES:
            self.category.addItem(title, kind)
        self.category.setEnabled(False)
        self.category.currentIndexChanged.connect(self._category_changed)
        self.box.addWidget(self.category)
        self.coverage = text_label(style="statisticsInfo")
        self.box.addWidget(self.coverage)
        self.items = QVBoxLayout()
        self.items.setContentsMargins(0, 0, 0, 0)
        self.box.addLayout(self.items)
        self.more = QPushButton("Показать ещё")
        self.more.clicked.connect(self.load_more)
        self.more.hide()
        self.box.addWidget(self.more)
        self.retry = QPushButton("Загрузить заново")
        self.retry.clicked.connect(self._category_changed)
        self.retry.hide()
        self.box.addWidget(self.retry)

    def _reset(self):
        self._request_id += 1
        if self._active is not None:
            self._active.cancelled.set()
        self._pending = None
        self.cursor = self.generation = self.source_version = None
        self._seen.clear()
        self._consumed.clear()
        self._partial = False
        self.setProperty("historyStatus", None)
        self.coverage.clear()
        while self.items.count():
            card = self.items.takeAt(0).widget()
            card.hide()
            card.deleteLater()
        self.cards.clear()
        self.more.hide()
        self.retry.hide()

    def set_customer(self, customer_id):
        if self._closing or customer_id == self.customer_id:
            return
        self.customer_id = customer_id
        self._category_changed()

    def _category_changed(self):
        self._reset()
        self.category.setEnabled(self.customer_id is not None)
        if self.customer_id is None:
            self.set_content(SectionContent(ViewState.INITIAL, "Выберите покупателя."))
            return
        self._request(None)

    def _request(self, cursor):
        if self._closing:
            return
        request = CustomerHistoryRequest(self._request_id, self.customer_id, self.category.currentData(), cursor)
        self.set_content(SectionContent(ViewState.LOADING, "Загрузка истории…"))
        self.more.setEnabled(False)
        self.retry.hide()
        if self._active is not None:
            self._pending = request
        else:
            self._start(request)

    def _start(self, request):
        self._active = request
        self.dispatch.emit(request)

    def load_more(self):
        if self.customer_id is not None and self.cursor is not None and self._active is None and self.cursor not in self._consumed:
            self._request(self.cursor)

    def receive(self, request, page):
        if request is not self._active:
            return
        self._active = None
        if self._closing:
            return
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._start(pending)
            return
        if request.cancelled.is_set() or request.request_id != self._request_id or page is None:
            return
        if page.status in (HistoryStatus.READY, HistoryStatus.PARTIAL, HistoryStatus.EMPTY):
            if page.canonical_customer_id != self.customer_id or (self.generation is not None and (
                    page.generation != self.generation or page.source_version != self.source_version)):
                self._failed(HistoryStatus.STALE_CURSOR)
                return
            self.generation, self.source_version = page.generation, page.source_version
            self._partial |= page.status is HistoryStatus.PARTIAL
            for event in page.events:
                if event.event_key not in self._seen:
                    self._seen.add(event.event_key)
                    card = HistoryEventCard(event)
                    self.cards.append(card)
                    self.items.addWidget(card)
            self._consumed.add(request.cursor)
            self.cursor = page.next_cursor
            self.more.setVisible(self.cursor is not None and self.cursor not in self._consumed)
            self.more.setEnabled(True)
            status = HistoryStatus.PARTIAL if self._partial else page.status
            state, message = STATUS_CONTENT[status]
            self.set_content(SectionContent(state, message + f" Показано событий: {len(self.cards)}."))
            self.setProperty("historyStatus", status.value)
            ranges = "\n".join(f"{c.component} ({c.source_kind}): [{c.since_utc.isoformat()}, {c.until_utc.isoformat()})"
                               for c in page.coverage)
            self.coverage.setText("Покрытие источников (UTC):\n" + ranges +
                                  "\nИстория отражает текущее объединение покупателей. Даты показаны в UTC; оплата и валюта не выводятся из статуса заказа.")
            self.coverage.setToolTip(f"Generation: {page.generation}\nSource version: {page.source_version}")
        else:
            self._failed(page.status)

    def _failed(self, status):
        # A page error must never leave a mixed/stale generation visible.
        self._reset()
        state, message = STATUS_CONTENT[status]
        self.set_content(SectionContent(state, message))
        self.setProperty("historyStatus", status.value)
        self.retry.show()

    def shutdown(self):
        self._closing = True
        self._reset()
