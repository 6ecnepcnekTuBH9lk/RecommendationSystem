"""Recommendation-only settings, using the existing multi-select and pool pattern."""

from PyQt6.QtCore import QObject, QRunnable, QSize, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget, QSizePolicy)

from Application.paths import ICONS_DIR
from Application.recommendation_catalog import RecommendationCatalog, signature
from Application.recommendation_rules import RULES_PATH, RecommendationRules, load_rules, save_rules
from Application.theme.layout_metrics import CONTROL_SPACING, BLOCK_SPACING


PRIMARY_UNAVAILABLE = ("Основная коллекция больше не доступна среди выбранных сезонов. "
                       "Выберите другую или явно укажите «Не задана».")


class _Signals(QObject):
    finished = pyqtSignal(object)


class _Task(QRunnable):
    def __init__(self, path, rules=None):
        super().__init__()
        self.path, self.rules = path, rules
        self.signals = _Signals()

    def run(self):
        try:
            catalog = RecommendationCatalog().load()
            if self.rules is not None:
                try:
                    self.rules.validate_catalog(catalog)
                except ValueError as exc:
                    self.signals.finished.emit((catalog, self.rules, str(exc), None))
                    return
                if not catalog.current():
                    raise ValueError("Каталог изменился. Обновите доступные значения.")
                save_rules(self.rules, self.path)
                rules, message = self.rules, "Правила рекомендаций сохранены. Для применения повторите поиск клиента."
            else:
                try:
                    rules = load_rules(self.path)
                    message = ""
                    try:
                        rules.validate_catalog(catalog)
                    except ValueError:
                        message = "В настройках есть удалённые значения. Снимите их выбор и сохраните правила."
                except FileNotFoundError:
                    rules, message = None, ""
                except (ValueError, TypeError, KeyError):
                    rules, message = None, "Настройки некорректны. Выберите ограничения и сохраните их заново."
            self.signals.finished.emit((catalog, rules, message, signature(self.path) if self.path.exists() else None))
        except Exception:
            self.signals.finished.emit((None, None, "Не удалось прочитать справочники или сохранить правила.", None))


class RecommendationRulesWidget(QWidget):
    def __init__(self, parent=None, *, settings_path=RULES_PATH):
        super().__init__(parent)
        # Imported here to reuse the established control without a module cycle.
        from Application.tabs.analysis_filter_tab import MultiSelect
        self.settings_path = settings_path
        self._task = None
        self._initialized = False
        self._primary_collection = None
        self._available_collections = frozenset()
        self._populating = False
        self._dirty = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(BLOCK_SPACING)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(CONTROL_SPACING)
        form.setVerticalSpacing(CONTROL_SPACING)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.kinds, self.collections = MultiSelect(), MultiSelect()
        self.primary_collection = QComboBox()
        self.primary_collection.setMinimumWidth(0)
        self.primary_collection.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.primary_collection.addItem("Не задана", None)
        self.minimum_stock = QSpinBox()
        self.minimum_stock.setRange(0, 2**31 - 1)
        self.kinds.populate((), ())
        self.collections.populate((), ())
        form.addRow("Вид номенклатуры:", self.kinds)
        form.addRow("Актуальные сезоны:", self.collections)
        form.addRow("Основная коллекция:", self.primary_collection)
        form.addRow("Допустимый остаток:", self.minimum_stock)
        layout.addLayout(form)
        self.message = QLabel("Загрузка справочников…")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.save_button = QPushButton(
            QIcon(str(ICONS_DIR / "save.png")),
            " Сохранить правила рекомендаций"
        )
        self.refresh_button = QPushButton(
            QIcon(str(ICONS_DIR / "refresh.png")),
            " Обновить значения"
        )

        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(CONTROL_SPACING)

        for button in (self.save_button, self.refresh_button):
            button.setIconSize(QSize(17, 17))
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed
            )

        self.buttons.addWidget(self.save_button, 1)
        self.buttons.addWidget(self.refresh_button, 1)

        layout.addLayout(self.buttons)
        layout.addStretch()
        self.save_button.clicked.connect(self.save)
        self.refresh_button.clicked.connect(self.refresh)
        self.kinds.changed.connect(self._changed)
        self.collections.changed.connect(self._changed)
        self.minimum_stock.valueChanged.connect(self._changed)
        self.primary_collection.currentIndexChanged.connect(self._primary_selected)
        # Activation of the already displayed "Не задана" explicitly clears a
        # removed priority; refreshing the list must never clear it silently.
        self.primary_collection.activated.connect(self._primary_selected)
        QTimer.singleShot(0, self.refresh)

    def _changed(self):
        if self._populating:
            return
        self._dirty = True
        self._sync_primary()

    def _set_message(self, message):
        self.message.setText(message)
        self.message.setVisible(bool(message))

    def _sync_primary(self):
        selected = set(self.collections.selected()) & self._available_collections
        self.primary_collection.blockSignals(True)
        self.primary_collection.clear()
        self.primary_collection.addItem("Не задана", None)
        for value in sorted(selected):
            self.primary_collection.addItem(value, value)
        self.primary_collection.setCurrentIndex(max(0, self.primary_collection.findData(self._primary_collection)))
        self.primary_collection.blockSignals(False)
        invalid = self._primary_collection is not None and self._primary_collection not in selected
        if invalid:
            self._set_message(PRIMARY_UNAVAILABLE)
        elif self.message.text() == PRIMARY_UNAVAILABLE:
            self._set_message("")
        return invalid

    def _primary_selected(self, _index):
        self._primary_collection = self.primary_collection.currentData()
        self._dirty = True
        if self.message.text() == PRIMARY_UNAVAILABLE:
            self._set_message("")

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _start(self, rules=None):
        if self._task is not None:
            return
        self._task = _Task(self.settings_path, rules)
        self._task.signals.finished.connect(self._received)
        self.setEnabled(False)
        QThreadPool.globalInstance().start(self._task)

    @pyqtSlot()
    def refresh(self):
        self._start()

    @pyqtSlot()
    def save(self):
        if self._sync_primary():
            return
        try:
            rules = RecommendationRules(self.kinds.selected(), self.collections.selected(),
                                        self.minimum_stock.value(), self._primary_collection)
        except ValueError as exc:
            self._set_message(str(exc))
            return
        self._start(rules)

    @pyqtSlot(object)
    def _received(self, payload):
        saving = self._task.rules is not None
        self._task = None
        self.setEnabled(True)
        catalog, rules, message, key = payload
        self._set_message(message)
        if catalog is None:
            self.save_button.setEnabled(False)
            return
        self.save_button.setEnabled(True)
        save_failed = saving and key is None
        preserve = self._initialized and self._dirty and (not saving or save_failed)
        kinds = self.kinds.selected() if preserve else (() if rules is None else rules.kinds)
        collections = self.collections.selected() if preserve else (() if rules is None else rules.collections)
        self._populating = True
        self._available_collections = catalog.collections
        self.kinds.populate(sorted(catalog.kinds | set(kinds)), kinds)
        self.collections.populate(sorted(catalog.collections | set(collections)), collections)
        if not preserve:
            self.minimum_stock.setValue(0 if rules is None else rules.minimum_stock)
            self._primary_collection = None if rules is None else rules.primary_collection
        self._populating = False
        self._sync_primary()
        self._initialized = True
        self._dirty = preserve or save_failed
