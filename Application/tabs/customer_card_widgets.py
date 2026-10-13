"""Small card components using the existing statistics palette and spacing."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QLayout, QSizePolicy, QVBoxLayout, QWidget

from Application.customer_card_data import DisplayField, SectionContent, ViewState
from Application.theme.layout_metrics import BLOCK_SPACING, COMPACT_SPACING, PAGE_MARGIN


def text_label(text="", *, style=None):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    if style:
        label.setProperty("class", style)
    return label


class AdaptiveCardGrid(QWidget):
    def __init__(self, *, minimum_card_width=220, maximum_columns=4, parent=None):
        super().__init__(parent)
        self.minimum_card_width, self.maximum_columns = minimum_card_width, maximum_columns
        self.cards = []
        self.visible_count = 0
        self.columns = 0
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(BLOCK_SPACING)
        self.grid.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

    def set_cards(self, cards):
        while self.grid.count():
            self.grid.takeAt(0)
        for card in self.cards:
            card.hide()
            card.deleteLater()
        self.cards = list(cards)
        for card in self.cards:
            card.setParent(self)
        self.visible_count = len(self.cards)
        self.reflow()

    def show_prefix(self, count):
        self.visible_count = min(count, len(self.cards))
        self.reflow()

    def reflow(self):
        self.columns = max(1, min(self.maximum_columns,
                                  (self.width() + BLOCK_SPACING) // (self.minimum_card_width + BLOCK_SPACING)))
        while self.grid.count():
            self.grid.takeAt(0)
        for column in range(self.maximum_columns):
            self.grid.setColumnStretch(column, int(column < self.columns))
        for index, card in enumerate(self.cards):
            visible = index < self.visible_count
            card.setVisible(visible)
            if visible:
                self.grid.addWidget(card, index // self.columns, index % self.columns)
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = max(1, min(self.maximum_columns,
                             (self.width() + BLOCK_SPACING) // (self.minimum_card_width + BLOCK_SPACING)))
        if columns != self.columns:
            self.reflow()


class StateSection(QWidget):
    def __init__(self, title, content, *, name):
        super().__init__()
        self.setObjectName(name)
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSpacing(BLOCK_SPACING)
        self.heading = text_label(title.upper(), style="statisticsSection")
        self.heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.message = text_label(style="statisticsInfo")
        self.box.addWidget(self.heading)
        self.box.addWidget(self.message)
        self.set_content(content)

    def set_content(self, content: SectionContent):
        self.content = content
        self.setProperty("state", content.state.value)
        self.message.setText(content.message)


def card_frame():
    card = QFrame()
    card.setProperty("class", "statisticsCard")
    card.setMinimumWidth(0)
    card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    box = QVBoxLayout(card)
    box.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
    box.setSpacing(COMPACT_SPACING)
    return card, box


_FIELD_STATES = {
    ViewState.INITIAL: "Клиент не выбран", ViewState.LOADING: "Загрузка…",
    ViewState.EMPTY: "Нет значения", ViewState.NOT_CONNECTED: "Не подключено",
    ViewState.ERROR: "Ошибка загрузки", ViewState.UNAVAILABLE: "Недоступно",
}


class FieldCard(QFrame):
    def __init__(self, content: DisplayField, *, numeric=False, framed=True):
        super().__init__()
        if framed:
            self.setProperty("class", "statisticsCard")
        else:
            self.setProperty("class", "resultsProfileField")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        box = QVBoxLayout(self)
        box.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        box.setSpacing(COMPACT_SPACING)
        self.title = text_label(content.title)
        self.value = text_label(style="statisticsNumber" if numeric and content.state is ViewState.READY
                                else "resultsPlaceholderValue")
        self.value.setText(content.value if content.value is not None else content.message or _FIELD_STATES.get(content.state, "Недоступно"))
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setProperty("state", content.state.value)
        box.addWidget(self.title)
        box.addWidget(self.value)
        if content.note:
            box.addWidget(text_label(content.note, style="statisticsInfo"))
        self.content = content


class PhotoLabel(QLabel):
    """Rescale the original cached pixmap on resize, preserving its aspect ratio."""
    def __init__(self, text):
        super().__init__(text)
        self._source_pixmap = None

    def set_source_pixmap(self, pixmap):
        self._source_pixmap = pixmap
        self._fit()

    def _fit(self):
        if self._source_pixmap is not None and self.width() > 0 and self.height() > 0:
            self.setPixmap(self._source_pixmap.scaled(
                self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def clear(self):
        self._source_pixmap = None
        super().clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class ProductCard(QFrame):
    def __init__(self, content):
        super().__init__()
        self.content = content
        self.setProperty("class", "statisticsCard")
        self.setMinimumWidth(0)
        box = QVBoxLayout(self)
        box.setContentsMargins(PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN, PAGE_MARGIN)
        box.setSpacing(BLOCK_SPACING)
        rec = content.recommendation
        self.rank = text_label(f"№{rec.rank}", style="resultsCardRank")
        self.photo = PhotoLabel("Загрузка фото…")
        self.photo.setTextFormat(Qt.TextFormat.PlainText)
        self.photo.setMinimumWidth(0)
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo.setFixedHeight(210)
        self.photo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.name = text_label(content.name or "Название недоступно", style="resultsProductName")
        self.name.setToolTip(content.name or "Справочные сведения о товаре недоступны")
        self.code = text_label(f"Код: {rec.model_item_code}")
        self.code.setToolTip(rec.model_item_code)
        self.code.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.score = text_label(f"Исходный BPR-score: {rec.raw_score:.6g}", style="resultsCardRank")
        self.score.setToolTip(f"Полное значение: {rec.raw_score!r}. BPR-score не является вероятностью.")
        for widget in (self.rank, self.photo, self.name, self.code, self.score):
            box.addWidget(widget)
        if content.reference_message:
            box.addWidget(text_label(content.reference_message, style="statisticsInfo"))
        box.addStretch(1)
