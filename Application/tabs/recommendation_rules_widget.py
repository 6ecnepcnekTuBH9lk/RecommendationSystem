"""Recommendation-only settings, using the existing multi-select and pool pattern."""

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFormLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from Application.recommendation_catalog import RecommendationCatalog, signature
from Application.recommendation_rules import RULES_PATH, RecommendationRules, load_rules, save_rules


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
                    rules, message = None, "Выберите виды (В группе), актуальные сезоны и сохраните правила."
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
        self._dirty = False
        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.kinds, self.collections = MultiSelect(), MultiSelect()
        self.minimum_stock = QSpinBox()
        self.minimum_stock.setRange(0, 2**31 - 1)
        self.kinds.populate((), ())
        self.collections.populate((), ())
        form.addRow("Вид номенклатуры (В группе):", self.kinds)
        form.addRow("Актуальные сезоны:", self.collections)
        form.addRow("Допустимый остаток:", self.minimum_stock)
        layout.addLayout(form)
        self.message = QLabel("Загрузка справочников…")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.save_button = QPushButton("Сохранить правила рекомендаций")
        self.refresh_button = QPushButton("Обновить значения")
        layout.addWidget(self.save_button)
        layout.addWidget(self.refresh_button)
        layout.addStretch()
        self.save_button.clicked.connect(self.save)
        self.refresh_button.clicked.connect(self.refresh)
        self.kinds.changed.connect(self._changed)
        self.collections.changed.connect(self._changed)
        self.minimum_stock.valueChanged.connect(self._changed)
        QTimer.singleShot(0, self.refresh)

    def _changed(self):
        self._dirty = True

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
        try:
            rules = RecommendationRules(self.kinds.selected(), self.collections.selected(),
                                        self.minimum_stock.value(), self._primary_collection)
        except ValueError as exc:
            self.message.setText(str(exc))
            return
        self._start(rules)

    @pyqtSlot(object)
    def _received(self, payload):
        saving = self._task.rules is not None
        self._task = None
        self.setEnabled(True)
        catalog, rules, message, key = payload
        self.message.setText(message)
        if catalog is None:
            self.save_button.setEnabled(False)
            return
        self.save_button.setEnabled(True)
        save_failed = saving and key is None
        preserve = self._initialized and self._dirty and (not saving or save_failed)
        kinds = self.kinds.selected() if preserve else (() if rules is None else rules.kinds)
        collections = self.collections.selected() if preserve else (() if rules is None else rules.collections)
        self.kinds.populate(sorted(catalog.kinds | set(kinds)), kinds)
        self.collections.populate(sorted(catalog.collections | set(collections)), collections)
        if not preserve:
            self.minimum_stock.setValue(0 if rules is None else rules.minimum_stock)
            self._primary_collection = None if rules is None else rules.primary_collection
        self._initialized = True
        self._dirty = preserve or save_failed
