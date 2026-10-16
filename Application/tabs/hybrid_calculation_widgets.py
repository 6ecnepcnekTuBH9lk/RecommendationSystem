"""Presentation of immutable worker calculations; no scoring or source I/O."""

from PyQt6.QtCore import QAbstractTableModel, QEvent, QModelIndex, Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableView, QToolButton

from Application.customer_card_data import SectionContent, ViewState
from Application.theme.layout_metrics import BLOCK_SPACING
from .customer_card_widgets import StateSection, text_label


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


class CalculationTableView(QTableView):
    compact = False

    @pyqtSlot()
    def fit_contents(self):
        if self.compact:
            height = (max(self.horizontalHeader().height(), self.horizontalHeader().sizeHint().height())
                      + max(self.verticalHeader().length(), self.verticalHeader().defaultSectionSize())
                      + self.horizontalScrollBar().sizeHint().height() + 2 * self.frameWidth())
            if self.height() != height:
                self.setFixedHeight(height)

    def event(self, event):
        result = super().event(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.FontChange, QEvent.Type.LayoutRequest):
            QTimer.singleShot(0, self.fit_contents)
        return result


def table(headers, widths, *, compact=True):
    view = CalculationTableView()
    view.compact = compact
    model = CalculationTableModel(headers, view)
    view.setModel(model)
    view.setMinimumWidth(0)
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
    return view, model


FEATURE_NAMES = {
    "КатегорияНаСайте": "Категория сайта и иерархия", "ВидНоменклатуры": "Вид номенклатуры",
    "СтилеваяГруппа": "Стилевая группа", "ПолНоменклатуры": "Пол номенклатуры", "Марка": "Марка",
    "ГруппаСоставов": "Группа составов", "ВидАссортимента": "Вид ассортимента",
    "НазваниеНаСайте": "Название товара", "СезонНоски": "Сезон носки", "ТипКоллекции": "Тип коллекции",
}


class HybridCalculationSection(StateSection):
    def __init__(self):
        super().__init__("Расчёт оценок гибридной модели", SectionContent(ViewState.INITIAL, "После поиска появятся расчёты."),
                         name="hybridCalculations")
        self.items = ()
        self.formula = text_label("S = 0.30 × R + 0.30 × B + 0.40 × M\n"
                                  "R — нормализованная позиция BPR; B — нормализованный BPR-score; M — сходство товаров.\n"
                                  "При заданной основной коллекции: S_final = (S + 0.05 × I) / 1.05. "
                                  "I = 1 при совпадении коллекции, иначе 0. Без приоритета S_final = S. "
                                  "Оценки не являются вероятностями покупки.")
        self.box.addWidget(self.formula)
        self.table, self.model = table(
            ("№", "Актуальный товар", "Исходный товар", "Ранг BPR", "BPR-score", "R", "B", "M", "Бонус", "Итоговая оценка"),
            (55, 270, 270, 100, 140, 120, 120, 120, 100, 150))
        self.table.setWordWrap(True)
        self.table.verticalHeader().setDefaultSectionSize(54)
        self.box.addWidget(self.table)
        self.summary = text_label("")
        self.bpr_details = text_label("")
        self.box.addWidget(self.summary)
        self.box.addWidget(self.bpr_details)
        self.features, self.feature_model = table(
            ("Признак", "Историческое значение", "Актуальное значение", "Вес", "Совпадение", "Вклад"),
            (250, 290, 290, 100, 120, 120))
        self.box.addWidget(self.features)
        self.final_details = text_label("")
        self.box.addWidget(self.final_details)
        self.merge_note = text_label("Итоговый рейтинг использует максимальную оценку одного источника, а не сумму оценок.")
        self.box.addWidget(self.merge_note)
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
        self.clear(ViewState.INITIAL, "После поиска появятся расчёты.")

    def _toggle_sources(self, checked):
        self.alternatives.setVisible(checked)
        self.other_sources.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def clear(self, state, message):
        self.items = ()
        self.model.set_rows(())
        self.feature_model.set_rows(())
        self.alternative_model.set_rows(())
        for label in (self.summary, self.bpr_details, self.final_details):
            label.setText("")
        self.set_content(SectionContent(state, message))
        self.message.setVisible(bool(message))
        for widget in (self.table, self.summary, self.bpr_details, self.features, self.final_details,
                       self.merge_note, self.other_sources, self.alternatives):
            widget.hide()

    def set_items(self, items):
        self.clear(ViewState.READY, "")
        self.items = tuple(items[:10])
        if not self.items or any(not item.sources or item.sources[0].calculation is None for item in self.items):
            self.clear(ViewState.UNAVAILABLE, "Подробные расчёты недоступны для этого результата.")
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
        self.summary.setText(
            f"Сопоставление товаров\nИсходный BPR-MF: №{source.source_raw_rank}; {source.source_model_item_code}; "
            f"{source.source_name or 'Не указано'}; коллекция: {source.source_collection or 'Не указано'}; "
            f"BPR-score: {source.source_raw_score!r}\n"
            f"Актуальный: №{item.final_rank}; {item.target_item_code}; {item.target_name or 'Не указано'}; "
            f"коллекция: {item.target_collection or 'Не указано'}; остаток: {item.target_stock}")
        self.bpr_details.setText(
            f"Расчёт BPR-компонентов\nПолное ранжирование: {calc.source_count} товаров; позиция источника: {calc.position + 1}.\n"
            f"Минимальный BPR-score: {calc.minimum_bpr_score!r}; максимальный: {calc.maximum_bpr_score!r}.\n"
            f"Исходный BPR-score: {source.source_raw_score!r}.\n"
            f"R = 1 − {calc.position} / max(1, {calc.source_count} − 1) = {calc.rank_score!r}.\n"
            f"B = (score − min) / (max − min), при равных score B = 1: {calc.normalized_bpr_score!r}.\n"
            f"Вклад 0.30 × R: {calc.rank_contribution!r}; вклад 0.30 × B: {calc.bpr_contribution!r}.")
        values = {feature.feature: feature for feature in calc.features}
        self.feature_model.set_rows((title, values[key].historical_value or "Не указано", values[key].target_value or "Не указано",
                                     values[key].weight, values[key].match, values[key].contribution)
                                    for key, title in FEATURE_NAMES.items() if key in values)
        self.features.fit_contents()
        self.final_details.setText(
            f"Итоговый расчёт\nSimilarity M: {source.similarity_score!r}; вклад 0.40 × M: {calc.similarity_contribution!r}.\n"
            f"Вклад ранга: {calc.rank_contribution!r}; вклад BPR-score: {calc.bpr_contribution!r}; "
            f"вклад сходства: {calc.similarity_contribution!r}.\n"
            f"S = {calc.rank_contribution!r} + {calc.bpr_contribution!r} + {calc.similarity_contribution!r} = {calc.before_bonus!r}.\n"
            f"Основная коллекция: {calc.primary_collection or 'Не задана'}; I = {int(calc.bonus > 0)}; "
            f"бонус: {calc.bonus!r}; делитель: {calc.divisor!r}.\nИтоговый рейтинг: {calc.final_score!r}.")
        self.final_details.setText(self.final_details.text() + (
            f"\nS_final = ({calc.before_bonus!r} + {calc.bonus!r}) / {calc.divisor!r} = {calc.final_score!r}."
            if calc.primary_collection is not None else "\nПриоритет отключён: бонус и дополнительное деление не применяются."))
        self.alternative_model.set_rows((s.source_raw_rank, f"{s.source_name or 'Не указано'} — {s.source_model_item_code}",
                                         s.source_raw_score, s.similarity_score, s.evidence_score) for s in item.sources[1:])
        self.other_sources.setText(f"Другие исходные соответствия ({len(item.sources) - 1})")
        self.other_sources.setEnabled(len(item.sources) > 1)
        self.other_sources.setChecked(False)
        self.alternatives.hide()
        for widget in (self.summary, self.bpr_details, self.features, self.final_details, self.merge_note, self.other_sources):
            widget.show()
