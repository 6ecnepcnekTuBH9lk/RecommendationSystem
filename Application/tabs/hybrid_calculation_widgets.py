"""Presentation of immutable worker calculations; no scoring or source I/O."""

import math
from html import escape

from PyQt6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QRectF, QSize, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAbstractTextDocumentLayout, QFont, QKeySequence, QPalette, QTextDocument, QTextOption
from PyQt6.QtWidgets import (
    QApplication, QAbstractItemView, QHeaderView, QSizePolicy, QStyle, QStyledItemDelegate,
    QStyleOptionHeader, QStyleOptionViewItem, QTableView, QToolButton,
)

from Application.customer_card_data import SectionContent, ViewState
from Application.theme.layout_metrics import BLOCK_SPACING, COMPACT_SPACING, PAGE_MARGIN
from .customer_card_widgets import RecommendationSection, card_frame, text_label


def number(value):
    return f"{value:.8g}" if isinstance(value, float) else str(value)


class CalculationTableModel(QAbstractTableModel):
    def __init__(self, headers, parent=None):
        super().__init__(parent)
        self.headers, self.rows = tuple(headers), ()

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = tuple(rows)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.rows[index.row()][index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return number(value)
        if role == Qt.ItemDataRole.ToolTipRole:
            return repr(value) if isinstance(value, float) else str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.headers[section]
        return None


def _document(text, font, width):
    """The same plain-text layout measures and paints wrapped cells/headers."""
    document = QTextDocument()
    document.setDocumentMargin(0)
    document.setDefaultFont(font)
    option = QTextOption(Qt.AlignmentFlag.AlignHCenter)
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    document.setDefaultTextOption(option)
    document.setPlainText(text)
    document.setTextWidth(max(1, width))
    return document


def _paint_document(painter, document, rect, palette):
    painter.save()
    painter.setClipRect(rect)
    painter.translate(rect.x(), rect.y() + max(0, (rect.height() - document.size().height()) / 2))
    context = QAbstractTextDocumentLayout.PaintContext()
    context.palette = palette
    document.documentLayout().draw(painter, context)
    painter.restore()


class _WrappedCells(QStyledItemDelegate):
    def sizeHint(self, option, index):
        self.initStyleOption(option, index)
        width = self.parent().columnWidth(index.column()) - int(self.parent().showGrid()) - 2 * COMPACT_SPACING
        document = _document(option.text, option.font, width)
        return QSize(max(1, width), math.ceil(document.size().height()) + 2 * COMPACT_SPACING)

    def paint(self, painter, option, index):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        document = _document(option.text, option.font, option.rect.width() - 2 * COMPACT_SPACING)
        option.text = ""
        style = option.widget.style() if option.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)
        palette = QPalette(option.palette)
        if option.state & QStyle.StateFlag.State_Selected:
            palette.setColor(QPalette.ColorRole.Text, palette.color(QPalette.ColorRole.HighlightedText))
        _paint_document(painter, document, QRectF(option.rect).adjusted(COMPACT_SPACING, 0, -COMPACT_SPACING, 0), palette)


class _WrappedHeader(QHeaderView):
    def text_font(self):
        font = QFont(self.font())
        font.setWeight(QFont.Weight.DemiBold)
        return font

    def fit_height(self):
        heights = [_document(str(self.model().headerData(column, self.orientation()) or ""), self.text_font(),
                             self.sectionSize(column) - 2 * COMPACT_SPACING).size().height()
                   for column in range(self.count())]
        self.setFixedHeight(math.ceil(max(heights, default=self.fontMetrics().height())) + 2 * PAGE_MARGIN)

    def paintSection(self, painter, rect, column):
        option = QStyleOptionHeader()
        self.initStyleOptionForIndex(option, column)
        option.rect = rect
        document = _document(option.text, self.text_font(), rect.width() - 2 * COMPACT_SPACING)
        option.text = ""
        self.style().drawControl(QStyle.ControlElement.CE_Header, option, painter, self)
        # QSS section colors are painted by the style, not copied into Text in
        # the header option. Use the table's text palette for readable labels.
        palette = QPalette(option.palette)
        palette.setColor(QPalette.ColorRole.Text, self.parentWidget().palette().color(QPalette.ColorRole.Text))
        _paint_document(painter, document, QRectF(rect).adjusted(COMPACT_SPACING, 0, -COMPACT_SPACING, 0), palette)


class CalculationTableView(QTableView):
    compact = False

    def __init__(self):
        super().__init__()
        self.column_weights = ()
        self._fitting = False
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self.fit_contents)

    def use_adaptive_columns(self, weights):
        self.column_weights = tuple(weights)
        self.setHorizontalHeader(_WrappedHeader(Qt.Orientation.Horizontal, self))
        self.horizontalHeader().setMinimumSectionSize(1)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setStretchLastSection(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.setItemDelegate(_WrappedCells(self))
        self.model().modelReset.connect(self.schedule_fit)

    def schedule_fit(self):
        if not self._fitting:
            self._fit_timer.start(0)

    @pyqtSlot()
    def fit_contents(self):
        if self._fitting:
            return
        self._fitting = True
        try:
            if self.column_weights:
                # Like HistoryTable, share viewport space by column weights.
                # Here widths may shrink below initial sizes: scrolling is disabled.
                available, weight = self.viewport().width(), sum(self.column_weights)
                for column, share in enumerate(self.column_weights):
                    width = available * share // weight
                    self.setColumnWidth(column, width)
                    available -= width
                    weight -= share
                self.horizontalHeader().fit_height()
                self.resizeRowsToContents()
                self.updateGeometries()
            if self.compact:
                scroll_height = (self.horizontalScrollBar().sizeHint().height()
                                 if self.horizontalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff else 0)
                height = (max(self.horizontalHeader().height(), self.horizontalHeader().sizeHint().height())
                          + max(self.verticalHeader().length(), self.verticalHeader().defaultSectionSize())
                          + scroll_height + 2 * self.frameWidth())
                if self.height() != height:
                    self.setFixedHeight(height)
        finally:
            self._fitting = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.column_weights:
            self.schedule_fit()

    def viewportEvent(self, event):
        result = super().viewportEvent(event)
        if event.type() is QEvent.Type.Resize and getattr(self, "column_weights", ()):
            # Style/layout changes may resize the viewport without resizing the
            # view itself, including while an earlier fit is in progress.
            self._fit_timer.start(0)
        return result

    def wheelEvent(self, event):
        if self.compact:
            event.ignore()  # The results page owns vertical scrolling.
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            rows = sorted({index.row() for index in self.selectedIndexes()})
            QApplication.clipboard().setText("\n".join(
                "\t".join(repr(value) if isinstance(value, float) else str(value) for value in self.model().rows[row])
                for row in rows))
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event):
        result = super().event(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.FontChange, QEvent.Type.LayoutRequest):
            if hasattr(self, "_fit_timer"):
                self.schedule_fit()
        return result


def table(headers, widths, *, compact=True, adaptive=False):
    view = CalculationTableView()
    view.compact = compact
    model = CalculationTableModel(headers, view)
    view.setModel(model)
    view.setMinimumWidth(0)
    view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.setWordWrap(False)
    view.verticalHeader().hide()
    view.verticalHeader().setDefaultSectionSize(34)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    if compact:
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    for i, width in enumerate(widths):
        view.setColumnWidth(i, width)
    if adaptive:
        view.use_adaptive_columns(widths)
    return view, model


FEATURE_NAMES = {
    "КатегорияНаСайте": "Категория сайта и иерархия", "ВидНоменклатуры": "Вид номенклатуры",
    "СтилеваяГруппа": "Стилевая группа", "ПолНоменклатуры": "Пол номенклатуры", "Марка": "Марка",
    "ГруппаСоставов": "Группа составов", "ВидАссортимента": "Вид ассортимента",
    "НазваниеНаСайте": "Название товара", "СезонНоски": "Сезон носки", "ТипКоллекции": "Тип коллекции",
}


def formula_frame():
    frame, box = card_frame()
    frame.setObjectName("hybridFormula")
    frame.setStyleSheet("QFrame#hybridFormula QLabel { font-size: 11pt; }"
                       "QFrame#hybridFormula QLabel[formulaTitle=\"true\"] { font-size: 12pt; font-weight: 600; }")
    box.setContentsMargins(BLOCK_SPACING, BLOCK_SPACING, BLOCK_SPACING, BLOCK_SPACING)
    groups = (
        ("S = 0.30 × R + 0.30 × B + 0.40 × M",
         "S — базовая оценка соответствия; R — нормализованный ранг BPR; "
         "B — нормализованный BPR-score; M — сходство товаров."),
        ("S_final = (S + 0.05 × I) / 1.05",
         "При заданной основной коллекции деление на 1.05 применяется ко всем кандидатам.\n"
         "I = 1, если актуальный товар относится к основной коллекции, иначе I = 0.\n"
         "Если основная коллекция не выбрана, S_final = S."),
        ("R = 1 − p / max(1, N − 1)",
         "p — позиция исторического товара в полном BPR-ранжировании, начиная с нуля; "
         "N — количество товаров в полном исходном ранжировании BPR-MF.\n"
         "Исключение покупок не меняет исходные позиции и базу нормализации."),
        ("B = (score − score_min) / (score_max − score_min)",
         "score — исходный BPR-score исторического товара; score_min и score_max — "
         "минимальный и максимальный BPR-score полного ранжирования.\nЕсли score_max = score_min, B = 1."),
        ("M = Σ(wᵢ × mᵢ)",
         "wᵢ — вес признака; mᵢ — степень совпадения исторического и актуального товара; M — сумма вкладов десяти признаков.\n"
         "Совпадение может быть частичным; при отсутствии необходимых характеристик вклад равен нулю.\n"
         "Сохраняется обязательная совместимость вида номенклатуры и пола."),
    )
    for position, (formula, description) in enumerate(groups):
        if position:
            box.addSpacing(BLOCK_SPACING)
        for text, title in ((formula, True), (description, False)):
            label = text_label(text)
            if "ᵢ" in text:
                # Segoe UI Variable lacks the Unicode subscript i on some PCs.
                # Rich text is limited to these escaped static formula labels.
                label.setTextFormat(Qt.TextFormat.RichText)
                label.setText(escape(text).replace("ᵢ", "<sub>i</sub>").replace("\n", "<br>"))
            label.setProperty("formulaTitle", title)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard)
            box.addWidget(label)
    box.addSpacing(BLOCK_SPACING)
    note = text_label("Итоговый рейтинг используется для ранжирования товаров и не является вероятностью покупки.")
    note.setAlignment(Qt.AlignmentFlag.AlignCenter)
    box.addWidget(note)
    return frame


class HybridCalculationSection(RecommendationSection):
    def __init__(self):
        super().__init__("Расчет оценок гибридной модели", SectionContent(ViewState.INITIAL, "После поиска появятся расчеты."),
                         name="hybridCalculations")
        self.items = ()
        self.formula = formula_frame()
        self.box.addWidget(self.formula)
        self.table, self.model = table(
            ("№", "Актуальный товар", "Исходный товар", "Ранг BPR", "BPR-score", "R", "B", "M", "Бонус", "Итоговая оценка"),
            (35, 250, 250, 80, 115, 90, 90, 90, 70, 130), adaptive=True)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setDefaultSectionSize(54)
        self.box.addWidget(self.table)
        self.features, self.feature_model = table(
            ("Признак", "Историческое значение", "Актуальное значение", "Вес", "Совпадение", "Вклад"),
            (250, 290, 290, 100, 120, 120), adaptive=True)
        self.box.addWidget(self.features)
        self.other_sources = QToolButton()
        self.other_sources.setText("Другие исходные соответствия")
        self.other_sources.setCheckable(True)
        self.other_sources.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.other_sources.setArrowType(Qt.ArrowType.RightArrow)
        self.box.addWidget(self.other_sources)
        self.alternatives, self.alternative_model = table(
            ("Ранг BPR", "Исходный товар", "BPR-score", "Similarity", "Оценка соответствия"),
            (100, 390, 160, 150, 180), compact=False)
        self.alternatives.setFixedHeight(240)
        self.box.addWidget(self.alternatives)
        self.box.setSpacing(BLOCK_SPACING)
        self.other_sources.toggled.connect(self._toggle_sources)
        self.table.selectionModel().currentRowChanged.connect(self._selected)
        self.clear(ViewState.INITIAL, "После поиска появятся расчеты.")

    def _toggle_sources(self, checked):
        self.alternatives.setVisible(checked)
        self.other_sources.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def clear(self, state, message):
        self.items = ()
        self.model.set_rows(())
        self.feature_model.set_rows(())
        self.alternative_model.set_rows(())
        self.other_sources.setChecked(False)
        self.set_content(SectionContent(state, message))
        for widget in (self.table, self.features, self.other_sources, self.alternatives):
            widget.hide()

    def set_items(self, items):
        self.clear(ViewState.READY, "")
        self.items = tuple(items[:10])
        if not self.items or any(not item.sources or item.sources[0].calculation is None for item in self.items):
            self.clear(ViewState.UNAVAILABLE, "Подробные расчеты недоступны для этого результата.")
            return
        self.model.set_rows((item.final_rank, f"{item.target_item_code}\n{item.target_name or 'Не указано'}",
                             f"{item.sources[0].source_model_item_code}\n{item.sources[0].source_name or 'Не указано'}",
                             item.sources[0].source_raw_rank, item.sources[0].source_raw_score,
                             item.sources[0].calculation.rank_score, item.sources[0].calculation.normalized_bpr_score,
                             item.sources[0].similarity_score, item.sources[0].calculation.bonus, item.final_score)
                            for item in self.items)
        self.table.fit_contents()
        self.table.show()
        self.table.setCurrentIndex(self.model.index(0, 0))

    def _selected(self, current, _previous):
        if not current.isValid() or current.row() >= len(self.items):
            return
        item = self.items[current.row()]
        source = item.sources[0]
        calc = source.calculation
        values = {feature.feature: feature for feature in calc.features}
        self.feature_model.set_rows((title, values[key].historical_value or "Не указано", values[key].target_value or "Не указано",
                                     values[key].weight, values[key].match, values[key].contribution)
                                    for key, title in FEATURE_NAMES.items() if key in values)
        self.features.fit_contents()
        self.alternative_model.set_rows((s.source_raw_rank, f"{s.source_name or 'Не указано'} — {s.source_model_item_code}",
                                         s.source_raw_score, s.similarity_score, s.evidence_score) for s in item.sources[1:])
        self.other_sources.setText(f"Другие исходные соответствия ({len(item.sources) - 1})")
        self.other_sources.setEnabled(len(item.sources) > 1)
        self.other_sources.setChecked(False)
        self.alternatives.hide()
        self.other_sources.setVisible(len(item.sources) > 1)
        self.features.show()
