"""Statistics presentation and subprocess lifecycle; no raw-data business rules."""

import json
import logging
import sys
from decimal import Decimal
from pathlib import Path

from PyQt6.QtCore import QEvent, QProcess, QSize, QTimer, Qt, pyqtSignal, QObject, QRunnable, QThreadPool, QStandardPaths
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QProgressBar, QPushButton, QScrollArea, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QSizePolicy, QStyle, QStyleOptionHeader, QFileDialog,
)

from Application.statistics_presentation import (
    SOURCE_ACTIONS_HEADING as SOURCE_ACTIONS_HEADING, ACTION_QUALITY_HEADING as ACTION_QUALITY_HEADING,
    source_action_rows, source_action_quality_rows,
)
from Application import statistics_charts as charts
from Application.loading_errors import format_error
from Application.paths import ICONS_DIR, PROJECT_ROOT
from Application.analysis_filter import AnalysisFilter
from Application.analysis_filter_settings import load_filter
from Application.settings.set_status import set_status_processing, schedule_status_reset, set_status_ok, set_status_error
from Application.statistics_export import export_statistics, suggested_filename
from Application.statistics_cache import CACHE_PATH, load_result, save_result, validate_result
from Application.statistics_presentation import (SOURCE_LABELS, SOURCE_KIND_LABELS, WARNING_LABELS, DIAGNOSTIC_LABELS,
                                                 _date, _date_time, _calculation_message)


BLOCK_SPACING = 18
SOURCE_DESCRIPTION = "Исходные данные Mindbox. Отбор не установлен."
PROCESSING_STATUS = "Расчет статистики..."
PROGRESS_LABELS = {"actions": "Действия", "orders": "Заказы", "customer_merges": "Объединения клиентов"}


def _number(value):
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


def _section_heading(layout, title):
    label = _label(title.upper())
    label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    label.setProperty("class", "statisticsSection")
    layout.addWidget(label)


class _StatisticsTable(QTableWidget):
    """Fit short tables to their rows, keeping long tables scrollable."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_height()

    def fit_height(self):
        height = self.horizontalHeader().height() + self.verticalHeader().length() + 2 * self.frameWidth()
        scrollbar = self.horizontalScrollBar()
        if scrollbar.maximum() > scrollbar.minimum():
            height += scrollbar.sizeHint().height()
        self.setFixedHeight(min(450, height))


def _cards(layout, values):
    grid = QGridLayout()
    labels = []
    for column, (title, value) in enumerate(values):
        card = QFrame()
        card.setProperty("class", "statisticsCard")
        box = QVBoxLayout(card)
        box.addWidget(_label(title))
        number = _label(_number(value))
        number.setProperty("class", "statisticsNumber")
        box.addWidget(number)
        grid.addWidget(card, column // 4, column % 4)
        grid.setColumnStretch(column % 4, 1)
        labels.append(number)
    layout.addLayout(grid)
    return labels


def _table(layout, headers, rows):
    table = _StatisticsTable(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().hide()
    table.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for column, value in enumerate(values):
            item = QTableWidgetItem(_number(value))
            table.setItem(row, column, item)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.horizontalHeader().setStretchLastSection(True)
    table.resizeColumnsToContents()
    for column in range(len(headers)):
        table.setColumnWidth(column, min(440, max(110, table.columnWidth(column))))
    table.horizontalScrollBar().rangeChanged.connect(table.fit_height)
    table.horizontalHeader().geometriesChanged.connect(table.fit_height)
    table.verticalHeader().sectionResized.connect(table.fit_height)
    table.ensurePolished()
    table.fit_height()
    layout.addWidget(table)
    return table


def _decimal_display(value, *, money=True):
    number = Decimal(value)
    formatted = format(number, ",.2f" if money else ",f").replace(",", " ")
    return formatted if money or "." not in formatted else formatted.rstrip("0").rstrip(".")


def _actions_page(layout, result):
    def table(headers, rows):
        widget = _table(layout, headers, rows)
        widget.ensurePolished()
        header = widget.horizontalHeader()
        header.ensurePolished()
        # The theme capitalizes headers; measure that text with its styled padding.
        for column, title in enumerate(headers):
            option = QStyleOptionHeader()
            option.initFrom(header)
            option.text = title.upper()
            size = header.style().sizeFromContents(QStyle.ContentsType.CT_HeaderSection, option, QSize(), header)
            widget.setColumnWidth(column, max(widget.columnWidth(column), size.width()))

    _cards(layout, [("Просмотры товаров", result["view_interactions"]),
                    ("Добавления в избранное", result["favorite_interactions"]),
                    ("Клиенты с просмотрами", result["view_users"]),
                    ("Клиенты с избранным", result["favorite_users"])])
    _section_heading(layout, "Активность клиентов")
    table(["Тип действия", "Количество взаимодействий", "Количество клиентов", "Среднее на клиента", "Медиана на клиента"], [
        ("Просмотры", result["view_interactions"], result["view_users"], result["mean_views_per_viewer"], result["median_views_per_viewer"]),
        ("Добавления в избранное", result["favorite_interactions"], result["favorite_users"],
         result["mean_favorites_per_user"], result["median_favorites_per_user"])])
    _section_heading(layout, charts.activity_distribution_data(result, 'views').title)
    layout.addWidget(charts.create_activity_distribution_chart(result, 'views'))
    table(["Количество просмотров", "Количество клиентов", "Доля, %"], result["view_user_activity_distribution"])
    _section_heading(layout, charts.activity_distribution_data(result, 'favorites').title)
    layout.addWidget(charts.create_activity_distribution_chart(result, 'favorites'))
    table(["Количество избранного", "Количество клиентов", "Доля, %"], result["favorite_user_activity_distribution"])
    _section_heading(layout, "Каналы")
    table(["Канал", "Просмотры", "Клиенты с просмотрами", "Доля просмотров, %",
                    "Избранное", "Клиенты с избранным", "Доля избранного, %"],
           [row[1:] for row in result["action_channel_statistics"]])
    _section_heading(layout, "Динамика действий")
    layout.addWidget(charts.create_view_dynamics_chart(result))
    layout.addWidget(charts.create_favorite_dynamics_chart(result))
    table(["Месяц", "Просмотры", "Клиенты с просмотрами", "Избранное", "Клиенты с избранным"],
           [(month[5:] + "." + month[:4], *values) for month, *values in result["action_monthly_dynamics"]])
    _section_heading(layout, "Параметры просмотров")
    table(["Доступность", "Просмотров", "Доля, %"], result["view_availability_distribution"])
    table(["Валюта", "Просмотров с ценой", "Средняя цена", "Медианная цена"],
           [(currency, n, _decimal_display(mean), _decimal_display(middle))
            for currency, n, mean, middle in result["view_price_statistics"]])



def _products_page(layout, result):
    def table(headers, rows):
        widget = _table(layout, headers, rows)
        widget.ensurePolished()
        header = widget.horizontalHeader()
        header.ensurePolished()
        for column, title in enumerate(headers):
            option = QStyleOptionHeader()
            option.initFrom(header)
            option.text = title.upper()
            size = header.style().sizeFromContents(QStyle.ContentsType.CT_HeaderSection, option, QSize(), header)
            widget.setColumnWidth(column, max(widget.columnWidth(column), size.width()))

    def group(first, field):
        table([first, "Товаров", "Просмотры", "Добавления в избранное", "Покупки", "Продано единиц"],
              [(*row[:5], _decimal_display(row[5], money=False)) for row in result[field]])

    _cards(layout, [("Товары с взаимодействиями", result["unique_resolved_items"]),
                    ("Товары с просмотрами", result["products_with_views"]),
                    ("Товары в избранном", result["products_with_favorites"]),
                    ("Купленные товары", result["products_with_purchases"])])
    _section_heading(layout, "Популярные товары по просмотрам")
    layout.addWidget(charts.create_product_ranking_chart(result, "views"))
    table(["Код", "Название", "Просмотры", "Клиенты с просмотрами", "Добавления в избранное", "Покупки"],
          result["top_viewed_products"])
    _section_heading(layout, "Популярные товары по избранному")
    layout.addWidget(charts.create_product_ranking_chart(result, "favorites"))
    table(["Код", "Название", "Добавления в избранное", "Клиенты с избранным", "Просмотры", "Покупки"],
          result["top_favorited_products"])
    _section_heading(layout, "Популярные товары по покупкам")
    layout.addWidget(charts.create_purchased_products_chart(result))
    table(["Код", "Название", "Покупки", "Покупатели", "Продано единиц", "Просмотры", "Добавления в избранное"],
          [(*row[:4], _decimal_display(row[4], money=False), *row[5:]) for row in result["top_purchased_products"]])
    _section_heading(layout, "Категории товаров")
    table(["Категория", "Товаров", "Просмотры", "Добавления в избранное", "Покупки", "Продано единиц"],
          [(*row[1:6], _decimal_display(row[6], money=False)) for row in result["product_category_statistics"]])
    for title, field in (("Спрос по полу товара", "product_gender_statistics"), ("Спрос по сезону", "product_season_statistics"),
                         ("Спрос по стилевой группе", "product_style_statistics")):
        _section_heading(layout, title)
        if field == "product_season_statistics":
            layout.addWidget(charts.create_season_chart(result))
        group("Значение", field)
    _section_heading(layout, "Качество сопоставления")
    table(["Показатель", "Значение"], [
        ("Уникальные исходные идентификаторы товаров", result["unique_source_products"]),
        ("Уникальные распознанные товары", result["unique_resolved_items"]),
        ("Распознанные взаимодействия", result["resolved_interactions"]),
        ("Нераспознанные взаимодействия", result["unresolved_interactions"]),
        ("Доля распознанных, %", f"{result['resolution_rate']:.4f}")])
    table(["Система идентификаторов", "Взаимодействия", "Распознано", "Не распознано", "Доля распознанных, %"],
          [("Неподдерживаемая" if name == "unsupported" else name, total, resolved, unresolved, f"{rate:.4f}")
           for name, total, resolved, unresolved, rate in result["namespaces"]])


def _orders_page(layout, result):
    def table(headers, rows):
        widget = _table(layout, headers, rows)
        header = widget.horizontalHeader()
        header.ensurePolished()
        for column, title in enumerate(headers):
            option = QStyleOptionHeader()
            option.initFrom(header)
            option.text = title.upper()
            size = header.style().sizeFromContents(QStyle.ContentsType.CT_HeaderSection, option, QSize(), header)
            widget.setColumnWidth(column, max(widget.columnWidth(column), size.width()))

    _cards(layout, [("Заказы с покупкой", result["purchase_orders"]),
                    ("Невыкупленные заказы", result["orders_without_purchase"]),
                    ("Позиции покупок", result["purchase_interactions"]), ("Покупатели", result["purchase_users"])])
    _section_heading(layout, "Корзина заказа")
    basket_rates = {label: rate for label, _, rate in result["purchase_basket_distribution"]}
    _cards(layout, [("Среднее число позиций на заказ", result["mean_purchase_lines_per_order"]),
                    ("Медиана числа позиций на заказ", result["median_purchase_lines_per_order"]),
                    ("Доля заказов с 1 позицией", f"{basket_rates['1 позиция']:.2f}%"),
                    ("Доля заказов с 3+ позициями", f"{sum(basket_rates[key] for key in ('3–5 позиций', '6–10 позиций', '11+ позиций')):.2f}%")])
    layout.addWidget(charts.create_basket_chart(result))
    table(["Количество позиций", "Заказов", "Доля, %"], result["purchase_basket_distribution"])
    _section_heading(layout, "Финансовые показатели")
    table(["Валюта", "Заказов", "Позиции", "Единиц", "Сумма покупок", "Средняя сумма заказа", "Медианная сумма заказа"],
           [(currency, n, lines, _decimal_display(units, money=False), *(_decimal_display(v) for v in (amount, mean, median)))
            for currency, n, lines, units, amount, mean, median in result["order_financials"]])
    mixed, unknown = result["mixed_currency_purchase_orders"], result["unknown_currency_purchase_orders"]
    if mixed or unknown:
        layout.addWidget(_label(f"Заказы со смешанной валютой: {_number(mixed)}. С неопределённой валютой: {_number(unknown)}."))
    for currency in ('RUB', 'KZT'):
        _section_heading(layout, charts.purchase_count_data(result, currency).title)
        layout.addWidget(charts.create_order_dynamics_chart(result, currency))
    for currency in ('RUB', 'KZT'):
        _section_heading(layout, 'Динамика суммы покупок — ' + currency)
        layout.addWidget(charts.create_revenue_chart(result, currency))
    table(["Месяц", "Заказы RUB", "Сумма RUB", "Заказы KZT", "Сумма KZT"],
           [(month[5:] + "." + month[:4], rub_n, _decimal_display(rub), kzt_n, _decimal_display(kzt))
            for month, rub_n, rub, kzt_n, kzt in result["order_monthly_dynamics"]])
    for currency in ("RUB", "KZT"):
        _section_heading(layout, "Магазины и каналы — " + currency)
        table(["Магазин / канал", "Заказов", "Покупателей", "Позиции", "Единиц", "Сумма покупок", "Средняя сумма заказа"],
               [(name, n, buyers, lines, _decimal_display(units, money=False), _decimal_display(amount), _decimal_display(mean))
                for unit, key, name, n, buyers, lines, units, amount, mean in result["store_statistics"] if unit == currency])
    _section_heading(layout, "Оплата")
    layout.addWidget(charts.create_payment_chart(result))
    table(["Способ оплаты", "Заказов с этим способом", "Доля заказов, %"], result["payment_type_distribution"])
    _section_heading(layout, "Статусы позиций")
    table(["Статус", "Позиции", "Доля, %", "Считается покупкой"],
           [(name, count, 100 * count / result["order_lines"] if result["order_lines"] else 0, "Да" if purchase else "Нет")
            for name, count, purchase in result["line_statuses"]])


class _ExportSignals(QObject):
    finished = pyqtSignal(bool)


class _ExportTask(QRunnable):
    def __init__(self, result, path):
        super().__init__()
        self.result, self.path = result, path
        self.signals = _ExportSignals()

    def run(self):
        try:
            export_statistics(self.result, self.path)
        except Exception:
            logging.getLogger(__name__).warning("Не удалось экспортировать статистику.")
            self.signals.finished.emit(False)
        else:
            self.signals.finished.emit(True)


class DatasetStatisticsTab(QWidget):
    calculation_running_changed = pyqtSignal(bool)

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName("datasetStatisticsTab")
        self.window = window
        self.cache_path = CACHE_PATH
        self.displayed_result = None
        self._export_task = None
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
        layout.setSpacing(BLOCK_SPACING)
        self.description = _information_label(SOURCE_DESCRIPTION)
        layout.addWidget(self.description)
        self.card_labels = _cards(layout, [(name, None) for name in (
            "Количество взаимодействий", "Количество заказов", "Количество позиций в заказах", "Количество клиентов")])
        controls = QHBoxLayout()
        controls.setSpacing(BLOCK_SPACING)

        self.refresh = QPushButton(
            QIcon(str(ICONS_DIR / "statistic.png")),
            " Рассчитать статистику",
        )
        self.cancel_button = QPushButton(
            QIcon(str(ICONS_DIR / "failure.png")),
            " Отменить",
        )
        self.export_button = QPushButton(QIcon(str(ICONS_DIR / "export.png")), " Экспорт статистики")
        self.export_button.setIconSize(QSize(17, 17))
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export)

        self.refresh.setIconSize(QSize(17, 17))
        self.cancel_button.setIconSize(QSize(17, 17))
        self.cancel_button.setEnabled(False)

        self.refresh.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)

        self.status = _information_label(
            "В фоновом режиме будут рассчитаны данные из Mindbox."
        )
        self.status.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.status.setWordWrap(False)
        self.status.setSizePolicy(
            QSizePolicy.Policy.Minimum,
            QSizePolicy.Policy.Preferred,
        )

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
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.progress_text = QLabel("Прогресс расчета")
        self.progress_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.progress_text.font()
        font.setItalic(True)
        self.progress_text.setFont(font)
        self.progress_text.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )

        progress_layout.addWidget(self.progress, 0, 0)
        progress_layout.addWidget(self.progress_text, 0, 0)

        controls.addWidget(self.refresh)
        controls.addWidget(self.export_button)
        controls.addWidget(self.cancel_button)
        controls.addWidget(self.status)
        controls.addWidget(self.progress_container, 1)

        layout.addLayout(controls)
        self.sections = QTabWidget()
        layout.addWidget(self.sections, 1)
        self._pages()
        window.installEventFilter(self)
        QApplication.instance().aboutToQuit.connect(self._shutdown)
        timer = getattr(window, "_status_reset_timer", None)
        if timer is not None:
            # Reuse the shared timer: a reset from another operation must not
            # leave an active statistics process displaying "ready".
            timer.timeout.connect(self._keep_processing_status)
        self._update_description()
        saved = load_result(self.cache_path)
        if saved is not None:
            self.render(saved)

    def _update_export_button(self):
        self.export_button.setEnabled(self.process is None and self.displayed_result is not None
                                      and self._export_task is None)

    def export(self):
        if self.process is not None or self.displayed_result is None or self._export_task is not None:
            return
        directory = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation) or str(Path.home())
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт статистики",
                                            str(Path(directory) / suggested_filename(self.displayed_result)), "Excel (*.xlsx)")
        if not path or self.process is not None or self._export_task is not None:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        # Capture this snapshot; later render() replaces it rather than mutating it.
        self._export_task = _ExportTask(self.displayed_result, path)
        self._export_task.signals.finished.connect(self._export_finished)
        self._update_export_button()
        QThreadPool.globalInstance().start(self._export_task)

    def _export_finished(self, success):
        self._export_task = None
        self._update_export_button()
        if success:
            set_status_ok(self.window, "Статистика экспортирована")
        else:
            set_status_error(self.window, "Не удалось экспортировать статистику")
        schedule_status_reset(self.window, 5)
        self._keep_processing_status()

    def set_analysis_filter(self, selection):
        self.configured_filter = selection
        self._update_description()

    def _update_description(self):
        text = SOURCE_DESCRIPTION.replace("Отбор не установлен", self.configured_filter.summary())
        if self.displayed_result is not None:
            text += " Дата и время расчета: " + _date_time(self.displayed_result["calculated_at"])
            if AnalysisFilter.from_dict(self.displayed_result["analysis_filter"]) != self.configured_filter:
                text += ". Отбор изменен → требуется пересчитать статистику."
        self.description.setText(text)

    def _pages(self):
        while self.sections.count():
            page = self.sections.widget(0)
            self.sections.removeTab(0)
            page.deleteLater()
        layouts = []
        for name in ("Действия", "Заказы", "Товары", "Клиенты", "Техническая информация"):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setSpacing(BLOCK_SPACING)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            scroll.setWidget(content)
            self.sections.addTab(scroll, name)
            layouts.append(layout)
        return layouts

    def start(self):
        if self.process is not None:
            return
        self.buffer.clear()
        self.pending = self.failure = None
        self.cancelled = False
        self.refresh.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress_text.setText("Идет расчет...")
        self.status.setText("Расчет… Показанные ранее данные обновятся только после успешного завершения.")
        process = QProcess(self)
        self.process = process
        self._update_export_button()
        self.calculation_running_changed.emit(True)
        timer = getattr(self.window, "_status_reset_timer", None)
        if timer is not None:
            timer.stop()
        self._keep_processing_status()
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.readyReadStandardOutput.connect(self._read)
        process.readyReadStandardError.connect(lambda: process.readAllStandardError())
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self.running_filter = self.configured_filter
        process.start(sys.executable, ["-u", str(PROJECT_ROOT / "scripts/dataset_statistics.py"),
                                      "--analysis-filter", json.dumps(self.running_filter.to_dict(), ensure_ascii=True)])

    def _keep_processing_status(self):
        if self.process is not None:
            set_status_processing(self.window, PROCESSING_STATUS)

    def _read(self):
        if self.process is None:
            return
        self.buffer.extend(bytes(self.process.readAllStandardOutput()))
        while b"\n" in self.buffer:
            line, _, rest = self.buffer.partition(b"\n")
            self.buffer = bytearray(rest)
            try:
                message = json.loads(line)
                kind, value = message["event"], message["value"]
                if kind == "progress" and not self.cancelled:
                    source, separator, count = value.partition(":")
                    self.status.setText("Расчет → " + PROGRESS_LABELS.get(source, source) + separator + count)
                elif kind == "result":
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
        self.calculation_running_changed.emit(False)
        self.kill_timer.stop()
        self.refresh.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_text.setText("Прогресс расчета")
        process.deleteLater()
        if self.cancelled:
            self.status.setText("Расчет отменён. Показанные ранее данные не обновлены.")
        elif code or exit_status != QProcess.ExitStatus.NormalExit or self.failure or self.pending is None:
            self.status.setText((self.failure or "Процесс статистики завершился без результата.")
                                + " Показанные ранее данные не обновлены.")
        else:
            try:
                save_result(self.cache_path, self.pending)
            except (OSError, ValueError, TypeError, RecursionError) as error:
                self.status.setText(f"Не удалось сохранить результат статистики: {error}. "
                                    "Показанные ранее данные не обновлены.")
            else:
                self.render(self.pending)
        self.pending = None
        self._update_export_button()
        schedule_status_reset(self.window, 0)
        if self.closing:
            QTimer.singleShot(0, self.window.close)

    def cancel(self):
        if self.process is not None:
            self.cancelled = True
            self.cancel_button.setEnabled(False)
            self.status.setText("Отмена расчета…")
            self.process.terminate()
            self.kill_timer.start(1500)

    def _kill(self):
        if self.process is not None:
            self.process.kill()

    def _shutdown(self):
        if self.process is not None:
            process = self.process
            process.kill()
            process.waitForFinished(1000)

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() == QEvent.Type.Close and self.process is not None:
            event.ignore()
            self.closing = True
            self.cancel()
            return True
        return super().eventFilter(watched, event)

    def render(self, result):
        for label, name in zip(self.card_labels, ("total_interactions", "orders", "order_lines", "customers")):
            label.setText(_number(result[name]))
        self.displayed_result = result
        self._update_export_button()
        self._update_description()
        self.status.setText("\n".join([_calculation_message(result),
                            *(WARNING_LABELS.get(warning, warning) for warning in result["warnings"]
                              if warning != "Позиции заказов с нецелым количеством исключены из статистики.")]))
        actions, orders, products, customers, technical = self._pages()
        coverage = []
        for source in result["coverage"]:
            period = "; ".join(f"{_date(a)} — {_date(b)}" for a, b in source["intervals"])
            if not period and source["updated"]:
                period = "Снимок данных от " + _date_time(source["updated"])
            coverage.append((SOURCE_LABELS[source["source"]], period or "—",
                             "/".join(SOURCE_KIND_LABELS.get(kind, kind) for kind in source["source_kinds"]) or "—"))
        _table(technical, ["Источник", "Период / состояние", "Источник данных"], coverage)
        filtered = AnalysisFilter.from_dict(result["analysis_filter"]).active
        base = [("Количество взаимодействий", result["total_interactions"]), ("Заказы", result["orders"]),
                ("Позиции заказов", result["order_lines"]), ("Профили клиентов", result["customers"]),
                ("Просмотры", result["view_interactions"]),
                ("Добавления в избранное", result["favorite_interactions"]),
                ("Покупки", result["purchase_interactions"]),
                ("Клиенты с взаимодействиями", result["interaction_users"]),
                ("Уникальные исходные идентификаторы товаров", result["unique_source_products"]),
                ("Уникальные распознанные товары", result["unique_resolved_items"]),
                ("Распознанные взаимодействия", result["resolved_interactions"]),
                ("Нераспознанные взаимодействия", result["unresolved_interactions"]),
                ("Доля распознанных, %", f"{result['resolution_rate']:.4f}")]
        _table(technical, ["Показатель", "Значение"], base)
        #_section_heading(technical, SOURCE_ACTIONS_HEADING)
        _table(technical, ["Системное название", "Количество", "Доля, %"], source_action_rows(result))
        #_section_heading(technical, ACTION_QUALITY_HEADING)
        _table(technical, ["Классификация событий", "Количество"], source_action_quality_rows(result))
        _actions_page(actions, result)
        _orders_page(orders, result)
        _products_page(products, result)
        _cards(customers, [("Количество клиентов", result["customers"]),
                          ("Активные клиенты", result["interaction_users"]),
                          ("Покупатели", result["purchase_users"]), ("Повторные покупатели (от 2 покупок)", result["repeat_buyers"])])
        _section_heading(customers, "Портрет клиента")
        if result["customers"] is None:
            customers.addWidget(_label("Данные профилей клиентов отсутствуют. Пол и возраст недоступны."))
        _section_heading(customers, "Распределение по полу")
        customers.addWidget(charts.create_gender_chart(result))
        _table(customers, ["Пол", "Количество клиентов", "Доля, %"], result["gender_distribution"])
        _cards(customers, [("Средний возраст", result["mean_age"]), ("Медианный возраст", result["median_age"])])
        _section_heading(customers, "Возрастная структура клиентов")
        customers.addWidget(charts.create_age_chart(result))
        _table(customers, ["Возрастная группа", "Количество клиентов", "Доля, %"], result["age_distribution"])
        _section_heading(customers, "Активность клиентов")
        _table(customers, ["Показатель", "Значение"], [
            ("Клиенты с просмотрами", result["view_users"]),
            ("Клиенты с добавлениями в избранное", result["favorite_users"]),
            ("Клиенты с покупками", result["purchase_users"]),
            ("Клиенты с просмотрами и покупками", result["view_purchase_users"]),
            ("Клиенты с избранным и покупками", result["favorite_purchase_users"]),
            ("Клиенты со всеми типами взаимодействий", result["all_interaction_type_users"]),
            ("Среднее число взаимодействий на активного клиента", result["mean_interactions"]),
            ("Медиана числа взаимодействий на активного клиента", result["median_interactions"]),
            ("Доля активных клиентов, совершивших покупку, %", result["active_buyer_rate"])])
        _section_heading(customers, charts.interaction_frequency_data(result).title)
        customers.addWidget(charts.create_interaction_frequency_chart(result))
        _table(customers, ["Количество взаимодействий", "Клиентов", "Доля, %"], result["interaction_activity_distribution"])
        _section_heading(customers, "Покупательская активность")
        _table(customers, ["Количество заказов", "Покупателей", "Доля покупателей, %"], result["purchase_order_distribution"])
        _table(customers, ["Показатель", "Значение"], [
            ("Среднее число заказов на покупателя", result["mean_orders_per_buyer"]),
            ("Медиана числа заказов на покупателя", result["median_orders_per_buyer"]),
            ("Доля повторных покупателей, %", result["repeat_buyer_rate"])])
        _table(technical, ["Диагностика", "Количество"],
               [(DIAGNOSTIC_LABELS[key], value) for key, value in result["diagnostics"]]
               + [("Заказы выбранного среза" if filtered else "Исходные заказы", result["orders"]),
                  ("Позиции заказов выбранного среза" if filtered else "Исходные позиции заказов", result["order_lines"]),
                  ("Распознанные взаимодействия", result["resolved_interactions"]),
                  ("Нераспознанные взаимодействия", result["unresolved_interactions"]),
                  ("Доля распознанных, %", f"{result['resolution_rate']:.4f}")])


def create_dataset_statistics_tab(window):
    window.dataset_statistics_tab = DatasetStatisticsTab(window)
    window.tabs.addTab(window.dataset_statistics_tab, "Статистика и анализ")
