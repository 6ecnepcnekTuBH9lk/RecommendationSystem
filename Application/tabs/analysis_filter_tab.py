"""Analysis configuration UI. Applying settings never launches statistics."""

import logging

from PyQt6.QtCore import QDate, QEvent, QObject, QRunnable, QSize, QThreadPool, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QDateEdit, QFormLayout, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QMenu, QPushButton, QSizePolicy, QToolButton, QVBoxLayout, QWidget, QWidgetAction,
                             QComboBox, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)

from Application.analysis_filter import AnalysisFilter
from Application.analysis_filter_settings import SETTINGS_PATH, load_filter, save_filter
from Application.dataset_statistics import load_analysis_options
from Application import store_city_mapping as mapping
from Application.paths import ICONS_DIR
from Application.settings.set_status import set_status_ok, set_status_error, schedule_status_reset
from Application.tabs.data_loading_tab import _heading
from Application.tabs.dataset_statistics_tab import BLOCK_SPACING


class MultiSelect(QToolButton):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analysisMultiSelect")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.menu = QMenu(self)
        self.setMenu(self.menu)
        container = QWidget()
        layout = QVBoxLayout(container)
        buttons = QHBoxLayout()
        self.select_all = QPushButton("Выбрать все")
        self.select_none = QPushButton("Снять все")
        buttons.addWidget(self.select_all)
        buttons.addWidget(self.select_none)
        layout.addLayout(buttons)
        self.items = QListWidget()
        self.items.setMinimumSize(240, 180)
        self.items.setMaximumHeight(280)
        layout.addWidget(self.items)
        action = QWidgetAction(self.menu)
        action.setDefaultWidget(container)
        self.menu.addAction(action)
        self.select_all.clicked.connect(lambda: self.set_checked(True))
        self.select_none.clicked.connect(lambda: self.set_checked(False))
        self.items.itemChanged.connect(self._changed)
        self.setText("Все значения")

    def populate(self, values, selection):
        self.items.blockSignals(True)
        self.items.clear()
        for value in values:
            item = QListWidgetItem(value if value is not None else "Не указано", self.items)
            item.setData(Qt.ItemDataRole.UserRole, value)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if selection is None or value in selection else Qt.CheckState.Unchecked)
        self.items.blockSignals(False)
        self._changed()

    def selected(self):
        return tuple(self.items.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.items.count())
                     if self.items.item(i).checkState() == Qt.CheckState.Checked)

    def set_checked(self, checked):
        self.items.blockSignals(True)
        for i in range(self.items.count()):
            self.items.item(i).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.items.blockSignals(False)
        self._changed()

    def _changed(self):
        selected = self.selected()
        if len(selected) == self.items.count():
            text = "Все значения"
        elif len(selected) == 1:
            text = selected[0] if selected[0] is not None else "Не указано"
        else:
            text = f"{len(selected)} значений выбрано"
        self.setText(text)
        self.changed.emit()


class _Signals(QObject):
    finished = pyqtSignal(object, str)


class _OptionsTask(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = _Signals()

    def run(self):
        try:
            options = load_analysis_options()
        except Exception:
            # No source values, paths or PII in UI diagnostics.
            logging.getLogger(__name__).warning("Отбор недоступен: не удалось прочитать источники или справочники.")
            options = None
        try:
            cities = mapping.load_cities()
        except Exception:
            logging.getLogger(__name__).warning("Отбор недоступен: не удалось прочитать справочник городов.")
            cities = None
        self.signals.finished.emit((options, cities), "")


class _StoreCatalogTask(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = _Signals()

    def run(self):
        try:
            stores = mapping.load_stores()
        except Exception:
            logging.getLogger(__name__).warning("Не удалось сформировать каталог магазинов.")
            self.signals.finished.emit(None, "Не удалось сформировать каталог магазинов.")
        else:
            self.signals.finished.emit(stores, "")


class CityDelegate(QStyledItemDelegate):
    def __init__(self, parent):
        super().__init__(parent)
        self.cities = ()

    def createEditor(self, parent, option, index):
        if index.column() != 1 or not self.parent().isEnabled():
            return None
        editor = QComboBox(parent)
        editor.addItem("Не сопоставлено", None)
        for city in self.cities:
            editor.addItem(city, city)
        editor.activated.connect(lambda: self._commit(editor))
        return editor

    def _commit(self, editor):
        self.commitData.emit(editor)
        self.closeEditor.emit(editor)

    def setEditorData(self, editor, index):
        editor.setCurrentIndex(max(0, editor.findData(index.data(Qt.ItemDataRole.UserRole))))

    def setModelData(self, editor, model, index):
        if not self.parent().isEnabled():
            return
        city = editor.currentData()
        model.setData(index, city, Qt.ItemDataRole.UserRole)
        model.setData(index, city or "Не сопоставлено", Qt.ItemDataRole.DisplayRole)



class AnalysisFilterTab(QWidget):
    filter_changed = pyqtSignal(object)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._populating = False
        self._validation_error = ""
        self.settings_path = SETTINGS_PATH
        self.configured_filter = load_filter(self.settings_path)
        self.options = None
        self._loader = None
        self._store_loader = None
        self.cities = None
        self._store_error = ""
        self._running = False
        self._refresh_again = False
        self._options_changed = False
        self.mapping_path = mapping.SETTINGS_PATH
        self.saved_mapping = mapping.load_mapping(self.mapping_path)
        self.mapping_sources = None
        self._mapping_baseline = {}
        self.setObjectName("analysisFilterTab")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        left, right = QWidget(), QWidget()
        separator = QFrame()
        separator.setObjectName("vSeparator")
        separator.setFixedWidth(1)
        separator.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(left, 1)
        layout.addWidget(separator)
        layout.addWidget(right, 1)
        left_layout, right_layout = QVBoxLayout(left), QVBoxLayout(right)
        left_layout.setSpacing(BLOCK_SPACING)
        right_layout.setSpacing(10)
        self._headings = []
        for target, title in ((left_layout, "Отбор для анализа"),
                              (right_layout, "Правила формирования рекомендаций")):
            heading_row = QHBoxLayout()
            heading_row.addStretch()
            heading = _heading(title, heading_row)
            heading.setWordWrap(True)
            heading.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            heading_row.addStretch()
            target.addLayout(heading_row)
            self._headings.append(heading)
        self.message = QLabel("Для установки отбора необходимо загрузить исходные данные и справочники.")
        self.message.setWordWrap(True)
        self.message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.message.font()
        font.setItalic(True)
        self.message.setFont(font)
        left_layout.addWidget(self.message)
        self.controls = QWidget()
        groups = QHBoxLayout(self.controls)
        groups.setContentsMargins(0, 0, 0, 0)
        self.date_group, self.product_group = QWidget(), QWidget()
        self.date_form, self.product_form = QFormLayout(self.date_group), QFormLayout(self.product_group)
        for form in (self.date_form, self.product_form):
            form.setContentsMargins(0, 0, 0, 0)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        groups.addWidget(self.date_group)
        groups.addStretch()
        groups.addWidget(self.product_group)
        self.start_date, self.end_date = QDateEdit(), QDateEdit()
        for widget in (self.start_date, self.end_date):
            widget.setCalendarPopup(True)
            widget.setDisplayFormat("dd.MM.yyyy")
            widget.dateChanged.connect(self._validate)
        self.types, self.collections = MultiSelect(), MultiSelect()
        self.date_form.addRow("Дата начала:", self.start_date)
        self.date_form.addRow("Дата окончания:", self.end_date)
        self.product_form.addRow("Вид номенклатуры:", self.types)
        self.product_form.addRow("Сезон:", self.collections)
        self.start_date.installEventFilter(self)
        self.types.changed.connect(self._validate)
        self.collections.changed.connect(self._validate)
        left_layout.addWidget(self.controls)
        buttons = QHBoxLayout()
        self.apply_button = QPushButton(QIcon(str(ICONS_DIR / "filter.png")), " Применить")
        self.reset_button = QPushButton(QIcon(str(ICONS_DIR / "cart.png")), " Сбросить")
        for button in (self.apply_button, self.reset_button):
            button.setIconSize(QSize(17, 17))
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch()
        left_layout.addLayout(buttons)
        self.reset_button.clicked.connect(self.reset)
        self.apply_button.clicked.connect(self.apply)
        heading_row = QHBoxLayout()
        heading_row.addStretch()
        heading = _heading("Распределение магазинов по городам", heading_row)
        heading.setWordWrap(True)
        heading.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        heading_row.addStretch()
        left_layout.addLayout(heading_row)
        self._headings.append(heading)
        self.store_table = QTableWidget(0, 2)
        self.store_table.setHorizontalHeaderLabels(["Магазин / канал", "Город"])
        self.store_table.verticalHeader().hide()
        self.store_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.store_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.city_delegate = CityDelegate(self.store_table)
        self.store_table.setItemDelegateForColumn(1, self.city_delegate)
        self.store_table.cellClicked.connect(self._edit_city)
        self.store_table.itemChanged.connect(self._mapping_dirty)
        self.store_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.store_table.setMaximumHeight(self.store_table.verticalHeader().defaultSectionSize() * 9)
        left_layout.addWidget(self.store_table)
        # A readiness state, never a transient save/status message.
        self.mapping_readiness = QLabel("Загрузка каналов Orders…")
        self.mapping_readiness.setWordWrap(True)
        left_layout.addWidget(self.mapping_readiness)
        self.save_mapping_button = QPushButton(QIcon(str(ICONS_DIR / "save.png")), " Сохранить распределение")
        self.save_mapping_button.setIconSize(QSize(17, 17))
        self.save_mapping_button.setEnabled(False)
        self.save_mapping_button.clicked.connect(self.save_store_mapping)
        left_layout.addWidget(self.save_mapping_button, alignment=Qt.AlignmentFlag.AlignLeft)
        left_layout.addStretch()
        placeholder = QLabel("В разработке")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(placeholder, 1)
        self._update_readiness()
        self._size_inputs()
        QTimer.singleShot(0, self.refresh_options)

    @pyqtSlot()
    def _size_inputs(self):
        self.start_date.ensurePolished()
        height = self.start_date.sizeHint().height()
        width = self.start_date.fontMetrics().horizontalAdvance("0") * 28
        label_width = max(self.date_form.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().sizeHint().width()
                          for i in range(2)) + max(
            self.product_form.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().sizeHint().width() for i in range(2))
        narrow = self.controls.width() < 2 * width + label_width + 3 * BLOCK_SPACING
        for form in (self.date_form, self.product_form):
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows if narrow else QFormLayout.RowWrapPolicy.DontWrapRows)
        width = max(80, min(width, (self.controls.width() - BLOCK_SPACING) // 2))
        for group, form in ((self.date_group, self.date_form), (self.product_group, self.product_form)):
            labels = max(form.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().sizeHint().width() for i in range(2))
            group.setFixedWidth(width if narrow else width + labels + max(0, form.horizontalSpacing()))
        for widget in (self.start_date, self.end_date, self.types, self.collections):
            widget.setFixedHeight(height)
            widget.setMaximumWidth(width)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button_width = max(button.sizeHint().width() for button in (self.apply_button, self.reset_button))
        for button in (self.apply_button, self.reset_button):
            button.setFixedWidth(button_width)
        for heading in self._headings:
            heading.setWordWrap(False)
            natural = heading.sizeHint().width()
            heading.setWordWrap(True)
            margins = heading.parentWidget().layout().contentsMargins()
            available = heading.parentWidget().width() - margins.left() - margins.right()
            heading.setMinimumWidth(min(natural, max(0, available)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._size_inputs()

    def eventFilter(self, watched, event):
        if watched is self.start_date and event.type() in (QEvent.Type.StyleChange, QEvent.Type.FontChange):
            QTimer.singleShot(0, self._size_inputs)
        return super().eventFilter(watched, event)

    def _show_status(self, text, *, error=False):
        (set_status_error if error else set_status_ok)(self.window, text)
        schedule_status_reset(self.window, 5)

    @property
    def all_required_sources_ready(self):
        """Statistics prerequisites (load_analysis_options) plus valid cities/Orders."""
        return (self._loader is None and self.options is not None
                and self.cities is not None)

    @property
    def _editing_enabled(self):
        return self.all_required_sources_ready and not self._running

    @property
    def _mapping_editable(self):
        return self._editing_enabled and self._store_loader is None and self.mapping_sources is not None and not self._store_error

    def _update_readiness(self):
        if self._editing_enabled and self._options_changed:
            selection = self.options.normalize(self.configured_filter, reconcile=True)
            if selection != self.configured_filter:
                if not self._save(selection):
                    self.options = None
                else:
                    self._show_status("Отбор скорректирован по актуальным данным")
            if self.options is not None:
                self._options_changed = False
                self._populate()
        enabled = self._editing_enabled
        self.controls.setEnabled(enabled)
        self.store_table.setEnabled(self._mapping_editable)
        self.reset_button.setEnabled(enabled)
        self.message.setVisible(not enabled)
        self.message.setText("Во время расчета статистики установить отбор невозможно." if self._running else
                             "Для установки отбора необходимо загрузить исходные данные и справочники.")
        store_message = ("Формирование каталога магазинов..." if self._store_loader is not None
                         else self._store_error or (self.mapping_sources.error if self.mapping_sources else ""))
        self.mapping_readiness.setText(store_message)
        self.mapping_readiness.setVisible(bool(enabled and store_message))
        self._validate()
        self._mapping_dirty()

    def set_statistics_running(self, running):
        self._running = running
        if not running:
            # Never re-enable from the snapshot taken before the calculation.
            self.refresh_options(force=True)
        self._update_readiness()

    def refresh_mapping(self):
        if not self._editing_enabled or self._store_loader is not None:
            return
        self._store_error = ""
        self._store_loader = _StoreCatalogTask()
        self._store_loader.signals.finished.connect(self._stores_loaded)
        self._update_readiness()
        QThreadPool.globalInstance().start(self._store_loader)

    def _stores_loaded(self, stores, error):
        self._store_loader = None
        self._store_error = error
        if stores is not None:
            self._mapping_loaded(mapping.MappingSources(stores, self.cities))
        self._update_readiness()

    def _edit_city(self, row, column):
        if column == 1 and self._mapping_editable:
            self.store_table.editItem(self.store_table.item(row, column))

    def _mapping_loaded(self, sources, error=""):
        if sources == self.mapping_sources:
            self._update_readiness()
            return
        draft = self._current_mapping()
        self.mapping_sources = sources
        current = mapping.reconcile(sources.stores, sources.cities, {**self.saved_mapping, **draft})
        self._mapping_baseline = mapping.reconcile(sources.stores, sources.cities, self.saved_mapping)
        self.city_delegate.cities = sources.cities or ()
        self.store_table.blockSignals(True)
        self.store_table.setRowCount(len(sources.stores))
        for row, (key, name) in enumerate(sources.stores):
            item = QTableWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.store_table.setItem(row, 0, item)
            city = QTableWidgetItem(current[key] or "Не сопоставлено")
            city.setData(Qt.ItemDataRole.UserRole, current[key])
            self.store_table.setItem(row, 1, city)
        self.store_table.blockSignals(False)
        self._update_readiness()

    def _current_mapping(self):
        return {self.store_table.item(row, 0).data(Qt.ItemDataRole.UserRole):
                self.store_table.item(row, 1).data(Qt.ItemDataRole.UserRole) for row in range(self.store_table.rowCount())}

    def _mapping_dirty(self):
        self.save_mapping_button.setEnabled(self._mapping_editable
            and self._current_mapping() != self._mapping_baseline)

    def save_store_mapping(self):
        if not self._mapping_editable or not self.save_mapping_button.isEnabled():
            return
        current = self._current_mapping()
        saved = {**self.saved_mapping, **current}
        try:
            mapping.save_mapping(saved, self.mapping_path)
        except (OSError, ValueError):
            self._show_status("Не удалось сохранить распределение", error=True)
            return
        self.saved_mapping = saved
        self._mapping_baseline = current
        self._mapping_dirty()
        self._show_status("Распределение сохранено")

    def showEvent(self, event):
        super().showEvent(event)
        self._size_inputs()
        self.refresh_options()

    def refresh_options(self, *, force=False):
        # A migration holds the canonical lock. Keep the already verified quick
        # gate rather than queueing another source read behind that migration.
        if self._store_loader is not None and not force:
            return
        if self._loader is not None:
            self._refresh_again |= force
            return
        self._loader = _OptionsTask()
        self._loader.signals.finished.connect(self._prerequisites_loaded)
        self._update_readiness()
        QThreadPool.globalInstance().start(self._loader)

    def _prerequisites_loaded(self, snapshot, error):
        if self._refresh_again:
            self._loader = None
            self._refresh_again = False
            self.refresh_options(force=True)
            return
        options, self.cities = snapshot
        if self.mapping_sources is not None:
            self._mapping_loaded(mapping.MappingSources(self.mapping_sources.stores, self.cities))
        self._options_loaded(options, error)
        self._loader = None
        self._update_readiness()
        self.refresh_mapping()

    def _options_loaded(self, options, error):
        self._options_changed |= options != self.options
        self.options = options
        self._update_readiness()

    def _populate(self):
        self._populating = True
        selection, options = self.configured_filter, self.options
        for widget, value in ((self.start_date, selection.start_date or options.start_date),
                              (self.end_date, selection.end_date or options.end_date)):
            widget.setDateRange(QDate.fromString(options.start_date, "yyyy-MM-dd"),
                                QDate.fromString(options.end_date, "yyyy-MM-dd"))
            widget.setDate(QDate.fromString(value, "yyyy-MM-dd"))
        self.types.populate(options.nomenclature_types, selection.nomenclature_types)
        self.collections.populate(options.collections, selection.collections)
        self._populating = False
        self._validate()

    def _validate(self):
        if not hasattr(self, "apply_button") or self._populating:
            return
        error = ""
        if self.options is not None:
            if self.start_date.date() > self.end_date.date():
                error = "Дата начала не может быть позже даты окончания"
            elif self.types.items.count() and not self.types.selected():
                error = "Выберите хотя бы один вид номенклатуры"
            elif self.collections.items.count() and not self.collections.selected():
                error = "Выберите хотя бы один сезон"
        self.apply_button.setEnabled(self._editing_enabled and not error)
        if self._editing_enabled and error and error != self._validation_error:
            self._show_status(error, error=True)
        self._validation_error = error

    def _save(self, selection):
        if not self._editing_enabled:
            return False
        try:
            save_filter(selection, self.settings_path)
        except (OSError, ValueError):
            self._show_status("Не удалось сохранить отбор. Предыдущие настройки сохранены.", error=True)
            return False
        self.configured_filter = selection
        self.filter_changed.emit(selection)
        return True

    def apply(self):
        if not self._editing_enabled or not self.apply_button.isEnabled():
            return
        selection = AnalysisFilter(self.start_date.date().toString("yyyy-MM-dd"),
                                   self.end_date.date().toString("yyyy-MM-dd"),
                                   self.types.selected() if self.types.items.count() else None,
                                   self.collections.selected() if self.collections.items.count() else None)
        if self._save(self.options.normalize(selection)):
            self._show_status("Отбор сохранен")

    def reset(self):
        if self._editing_enabled and self._save(AnalysisFilter()):
            self._populate()
            self._show_status("Отбор сброшен")


def create_analysis_filter_tab(window):
    window.analysis_filter_tab = AnalysisFilterTab(window)
    window.tabs.addTab(window.analysis_filter_tab, "Пользовательские настройки")
