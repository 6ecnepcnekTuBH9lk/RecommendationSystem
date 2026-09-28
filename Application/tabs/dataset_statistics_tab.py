"""Statistics presentation and subprocess lifecycle; no raw-data business rules."""

import json
import sys
from decimal import Decimal

from PyQt6.QtCore import QEvent, QProcess, QSize, QTimer, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication, QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QProgressBar, QPushButton, QScrollArea, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QSizePolicy, QStyle, QStyleOptionHeader,
)

from Application.loading_errors import format_error
from Application.paths import ICONS_DIR, PROJECT_ROOT
from Application.statistics_period import shared_intervals
from Application.settings.set_status import set_status_processing, schedule_status_reset
from Application.statistics_cache import CACHE_PATH, load_result, parse_timestamp, save_result, validate_result


BLOCK_SPACING = 18
SOURCE_DESCRIPTION = "Исходные данные Mindbox. Отбор не установлен."
PROCESSING_STATUS = "Расчет статистики..."
SOURCE_LABELS = {"Actions": "Действия", "Orders": "Заказы", "Customers": "Клиенты",
                 "CustomerMerges": "Объединения клиентов"}
PROGRESS_LABELS = {"actions": "Действия", "orders": "Заказы", "customer_merges": "Объединения клиентов"}
SOURCE_KIND_LABELS = {"API": "Через API", "MANUAL": "Ручная загрузка", "MIXED": "Смешанный"}
WARNING_LABELS = {
    "CustomerMerges не покрывает весь период: canonical identity может быть неполной.":
        "История объединений клиентов не покрывает весь период: сопоставление клиентов может быть неполным.",
    "Есть конфликтующие снимки заказов: item interactions рассчитаны по первому снимку.":
        "Есть конфликтующие снимки заказов: взаимодействия с товарами рассчитаны по первому снимку.",
    "Часть Actions не содержит customer ID; число клиентов Actions учитывает только известные ID.":
        "Часть событий не содержит идентификатора клиента; число клиентов учитывает только известные идентификаторы.",
    "Canonical Customers отсутствует: число профилей неизвестно.":
        "Данные клиентов отсутствуют: число профилей неизвестно.",
}


def _date(value):
    return parse_timestamp(value).strftime("%d.%m.%Y")


def _date_time(value):
    return parse_timestamp(value).astimezone().strftime("%d.%m.%Y %H:%M:%S")


def _calculation_message(result):
    """Present shared interaction coverage, retaining gaps; never use snapshots."""
    sources = {source["source"]: source["intervals"] for source in result["coverage"]}
    merged = shared_intervals(sources.get("Actions", ()), sources.get("Orders", ()))
    if not merged:
        return "Статистика рассчитана."
    periods = "; ".join(
        f"{start:%d.%m.%Y} - {end:%d.%m.%Y}"
        for start, end in merged
    )
    return ("Статистика рассчитана за период: " if len(merged) == 1 else "Статистика рассчитана за периоды: ") + periods + "."


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
    table = QTableWidget(0, len(headers))
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
    table.setMinimumHeight(min(450, max(130, 36 + 32 * len(rows))))
    layout.addWidget(table)
    return table


def _decimal_display(value, *, money=True):
    number = Decimal(value)
    formatted = format(number, ",.2f" if money else ",f").replace(",", " ")
    return formatted if money or "." not in formatted else formatted.rstrip("0").rstrip(".")


def _actions_page(layout, result):
    def heading(title):
        label = _label(title)
        label.setProperty("class", "sectionHeader")
        layout.addWidget(label)

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
    layout.addWidget(_label("Показатели рассчитаны по товарным просмотрам и добавлениям в избранное. "
                            "Одно исходное событие с несколькими товарами создаёт несколько взаимодействий."))
    heading("Активность клиентов")
    table(["Тип действия", "Взаимодействий", "Клиентов", "Среднее на клиента", "Медиана на клиента"], [
        ("Просмотры", result["view_interactions"], result["view_users"], result["mean_views_per_viewer"], result["median_views_per_viewer"]),
        ("Добавления в избранное", result["favorite_interactions"], result["favorite_users"],
         result["mean_favorites_per_user"], result["median_favorites_per_user"])])
    table(["Количество просмотров", "Клиентов", "Доля, %"], result["view_user_activity_distribution"])
    table(["Количество добавлений", "Клиентов", "Доля, %"], result["favorite_user_activity_distribution"])
    heading("Каналы взаимодействий")
    table(["Канал", "Просмотры", "Клиенты с просмотрами", "Доля просмотров, %",
                    "Добавления в избранное", "Клиенты с избранным", "Доля избранного, %"],
           [row[1:] for row in result["action_channel_statistics"]])
    heading("Динамика действий")
    layout.addWidget(_label("Месяц определяется по дате события в UTC."))
    table(["Месяц", "Просмотры", "Клиенты с просмотрами", "Добавления в избранное", "Клиенты с избранным"],
           [(month[5:] + "." + month[:4], *values) for month, *values in result["action_monthly_dynamics"]])
    heading("Параметры просмотров")
    layout.addWidget(_label("Доступность и цена учитываются один раз на исходное событие просмотра с единственным товаром. "
                            "Отсутствующие цены не входят в среднее и медиану."))
    table(["Доступность", "Просмотров", "Доля, %"], result["view_availability_distribution"])
    table(["Валюта", "Просмотров с ценой", "Средняя цена", "Медианная цена"],
           [(currency, n, _decimal_display(mean), _decimal_display(middle))
            for currency, n, mean, middle in result["view_price_statistics"]])
    heading("Исходные события")
    table(["Системное имя", "Количество", "Доля, %"],
           [(name, count, 100 * count / result["actions"] if result["actions"] else 0)
            for name, count in result["action_types"]])
    diagnostics = dict(result["diagnostics"])
    table(["Классификация событий", "Количество"],
           [(DIAGNOSTIC_LABELS[key], diagnostics[key]) for key in
            ("mapped_view_actions", "mapped_favorite_actions", "unmapped_actions", "mapped_without_product")])


def _products_page(layout, result):
    def heading(title):
        label = _label(title)
        label.setProperty("class", "sectionHeader")
        layout.addWidget(label)

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
    layout.addWidget(_label("Показатели объединены по товарам справочника. Покупки — число позиций покупки, "
                            "продано единиц — их суммарное количество. В рейтингах показано до 20 товаров."))
    heading("Популярные товары")
    layout.addWidget(_label("По просмотрам"))
    table(["Код", "Название", "Просмотры", "Клиенты с просмотрами", "Добавления в избранное", "Покупки"],
          result["top_viewed_products"])
    layout.addWidget(_label("По добавлениям в избранное"))
    table(["Код", "Название", "Добавления в избранное", "Клиенты с избранным", "Просмотры", "Покупки"],
          result["top_favorited_products"])
    layout.addWidget(_label("По покупкам"))
    table(["Код", "Название", "Покупки", "Покупатели", "Продано единиц", "Просмотры", "Добавления в избранное"],
          [(*row[:4], _decimal_display(row[4], money=False), *row[5:]) for row in result["top_purchased_products"]])
    heading("Категории товаров")
    group("Категория", "product_category_statistics")
    heading("Структура спроса")
    for title, field in (("Пол товара", "product_gender_statistics"), ("Сезон", "product_season_statistics"),
                         ("Стилевая группа", "product_style_statistics")):
        layout.addWidget(_label(title))
        group("Значение", field)
    heading("Качество сопоставления")
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
    def heading(title):
        label = _label(title)
        label.setProperty("class", "sectionHeader")
        layout.addWidget(label)

    _cards(layout, [("Заказы с покупкой", result["purchase_orders"]), ("Покупатели", result["purchase_users"]),
                    ("Позиции покупок", result["purchase_interactions"]),
                    ("Продано единиц", _decimal_display(result["purchase_quantity"], money=False))])
    layout.addWidget(_label("Бизнес-показатели рассчитаны по уникальным заказам с позициями покупки после исключения дублей. "
                            "Статусы исходных позиций приведены отдельно внизу."))
    heading("Корзина заказа")
    _cards(layout, [("Среднее число позиций на заказ", result["mean_purchase_lines_per_order"]),
                    ("Медиана числа позиций на заказ", result["median_purchase_lines_per_order"]),
                    ("Среднее число единиц на заказ", _decimal_display(result["mean_purchase_units_per_order"])),
                    ("Медиана числа единиц на заказ", _decimal_display(result["median_purchase_units_per_order"]))])
    _table(layout, ["Количество позиций", "Заказов", "Доля, %"], result["purchase_basket_distribution"])
    heading("Финансовые показатели")
    _table(layout, ["Валюта", "Заказов", "Позиции", "Единиц", "Сумма покупок", "Средняя сумма заказа", "Медианная сумма заказа"],
           [(currency, n, lines, _decimal_display(units, money=False), *(_decimal_display(v) for v in (amount, mean, median)))
            for currency, n, lines, units, amount, mean, median in result["order_financials"]])
    mixed, unknown = result["mixed_currency_purchase_orders"], result["unknown_currency_purchase_orders"]
    if mixed or unknown:
        layout.addWidget(_label(f"Заказы со смешанной валютой: {_number(mixed)}. С неопределённой валютой: {_number(unknown)}. "
                                "Они учтены в корзине и общем числе заказов с покупкой, но исключены из денежных показателей RUB/KZT."))
    heading("Динамика покупок")
    layout.addWidget(_label("Месяц определяется по исходной дате заказа в UTC."))
    _table(layout, ["Месяц", "Заказы RUB", "Сумма RUB", "Заказы KZT", "Сумма KZT"],
           [(month[5:] + "." + month[:4], rub_n, _decimal_display(rub), kzt_n, _decimal_display(kzt))
            for month, rub_n, rub, kzt_n, kzt in result["order_monthly_dynamics"]])
    heading("Магазины и каналы")
    for currency in ("RUB", "KZT"):
        layout.addWidget(_label(currency + " — 20 магазинов / каналов с наибольшей суммой покупок"))
        _table(layout, ["Магазин / канал", "Заказов", "Покупателей", "Позиции", "Единиц", "Сумма покупок", "Средняя сумма заказа"],
               [(name, n, buyers, lines, _decimal_display(units, money=False), _decimal_display(amount), _decimal_display(mean))
                for unit, key, name, n, buyers, lines, units, amount, mean in result["store_statistics"] if unit == currency])
    heading("Оформление, доставка и оплата")
    _table(layout, ["Способ оформления", "Заказов", "Доля, %"], result["ordering_method_distribution"])
    _table(layout, ["Способ доставки", "Заказов", "Доля, %"], result["delivery_type_distribution"])
    layout.addWidget(_label("В одном заказе может быть несколько способов оплаты; сумма долей может превышать 100%."))
    _table(layout, ["Способ оплаты", "Заказов с этим способом", "Доля заказов, %"], result["payment_type_distribution"])
    _table(layout, ["Валюта", "Заказов с указанной стоимостью доставки", "Сумма доставки", "Средняя стоимость доставки", "Медианная стоимость доставки"],
           [(currency, n, *(_decimal_display(v) for v in (amount, mean, median)))
            for currency, n, amount, mean, median in result["delivery_financials"]])
    heading("Статусы позиций")
    layout.addWidget(_label("Эта таблица показывает статусы всех исходных позиций заказов, включая дубли. "
                            "Бизнес-показатели выше рассчитаны по уникальным заказам после исключения дублей."))
    _table(layout, ["Статус", "Позиции", "Доля, %", "Покупки"],
           [(name, count, 100 * count / result["order_lines"] if result["order_lines"] else 0, "Да" if purchase else "Нет")
            for name, count, purchase in result["line_statuses"]])


DIAGNOSTIC_LABELS = {
    "ambiguous_product_view_actions": "Просмотры с неоднозначным набором товаров",
    "unknown_currency_view_price_actions": "Просмотры с ценой без определённой валюты",
    "orders_outside_statistics_period": "Заказы вне периода статистики",
    "fractional_quantity_order_lines": "Позиции с нецелым количеством",
    "mapped_view_actions": "События просмотра",
    "mapped_favorite_actions": "События добавления в избранное",
    "unmapped_actions": "Неклассифицированные события",
    "mapped_without_product": "Классифицированные события без товара",
    "actions_without_customer_id": "События без идентификатора клиента",
    "unknown_candidate": "Товары, отсутствующие в справочнике",
    "unsupported_namespace": "Неподдерживаемая система идентификаторов товаров",
    "invalid_id": "Некорректные идентификаторы товаров",
    "orders_unique": "Уникальные снимки заказов",
    "orders_duplicate_identical": "Одинаковые дубли заказов",
    "orders_duplicate_conflicting": "Конфликтующие дубли заказов",
    "unique_order_lines": "Позиции после исключения дублей заказов",
    "purchase_lines": "Позиции покупок после исключения дублей",
    "filtered_by_status": "Позиции, исключённые по статусу (без дублей)",
}


class DatasetStatisticsTab(QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.cache_path = CACHE_PATH
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
        saved = load_result(self.cache_path)
        if saved is not None:
            self.render(saved)

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
        timer = getattr(self.window, "_status_reset_timer", None)
        if timer is not None:
            timer.stop()
        self._keep_processing_status()
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.readyReadStandardOutput.connect(self._read)
        process.readyReadStandardError.connect(lambda: process.readAllStandardError())
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        process.start(sys.executable, ["-u", str(PROJECT_ROOT / "scripts/dataset_statistics.py")])

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
        for label, name in zip(self.card_labels, ("actions", "orders", "order_lines", "customers")):
            label.setText(_number(result[name]))
        self.description.setText(SOURCE_DESCRIPTION + " Дата и время расчета: " + _date_time(result["calculated_at"]))
        self.status.setText("\n".join([_calculation_message(result),
                            *(WARNING_LABELS.get(warning, warning) for warning in result["warnings"])]))
        actions, orders, products, customers, technical = self._pages()
        coverage = []
        for source in result["coverage"]:
            period = "; ".join(f"{_date(a)} — {_date(b)}" for a, b in source["intervals"])
            if not period and source["updated"]:
                period = "Снимок данных от " + _date_time(source["updated"])
            coverage.append((SOURCE_LABELS[source["source"]], period or "—",
                             "/".join(SOURCE_KIND_LABELS.get(kind, kind) for kind in source["source_kinds"]) or "—"))
        _table(technical, ["Источник", "Период / состояние", "Источник данных"], coverage)
        base = [("Действия", result["actions"]), ("Заказы", result["orders"]),
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
        _actions_page(actions, result)
        _orders_page(orders, result)
        _products_page(products, result)
        _cards(customers, [("Количество клиентов", result["customers"]),
                          ("Активные клиенты", result["interaction_users"]),
                          ("Покупатели", result["purchase_users"]), ("Повторные покупатели", result["repeat_buyers"])])
        customers.addWidget(_label("Активными считаются клиенты, имеющие хотя бы одно взаимодействие с товаром, "
                                   "включая взаимодействия с товарами, которые не удалось сопоставить со справочником. "
                                   "Покупатель имеет хотя бы один уникальный заказ с покупкой, повторный — не менее двух. "
                                   "Активность учитывается по объединённым идентификаторам клиентов, "
                                   "независимо от наличия профиля в текущем снимке клиентов."))
        portrait = _label("Портрет клиента")
        portrait.setProperty("class", "sectionHeader")
        customers.addWidget(portrait)
        customers.addWidget(_label("Пол и возраст рассчитаны по уникальным профилям текущего снимка клиентов. "
                                   "Возраст клиента определяется на дату расчёта статистики."))
        if result["customers"] is None:
            customers.addWidget(_label("Данные профилей клиентов отсутствуют. Пол и возраст недоступны."))
        _table(customers, ["Пол", "Количество клиентов", "Доля, %"], result["gender_distribution"])
        _cards(customers, [("Средний возраст", result["mean_age"]), ("Медианный возраст", result["median_age"])])
        _table(customers, ["Возрастная группа", "Количество клиентов", "Доля, %"], result["age_distribution"])
        activity = _label("Активность клиентов")
        activity.setProperty("class", "sectionHeader")
        customers.addWidget(activity)
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
        _table(customers, ["Количество взаимодействий", "Клиентов", "Доля, %"], result["interaction_activity_distribution"])
        purchases = _label("Покупательская активность")
        purchases.setProperty("class", "sectionHeader")
        customers.addWidget(purchases)
        _table(customers, ["Количество заказов", "Покупателей", "Доля покупателей, %"], result["purchase_order_distribution"])
        _table(customers, ["Показатель", "Значение"], [
            ("Среднее число заказов на покупателя", result["mean_orders_per_buyer"]),
            ("Медиана числа заказов на покупателя", result["median_orders_per_buyer"]),
            ("Доля повторных покупателей, %", result["repeat_buyer_rate"])])
        technical.addWidget(_label("Если при чтении данных обнаруживается некорректная запись, расчет завершается "
                                 "с ошибкой — такие записи не пропускаются. События без указанного товара учитываются отдельно."))
        _table(technical, ["Диагностика", "Количество"],
               [(DIAGNOSTIC_LABELS[key], value) for key, value in result["diagnostics"]]
               + [("Исходные заказы", result["orders"]), ("Исходные позиции заказов", result["order_lines"]),
                  ("Распознанные взаимодействия", result["resolved_interactions"]),
                  ("Нераспознанные взаимодействия", result["unresolved_interactions"]),
                  ("Доля распознанных, %", f"{result['resolution_rate']:.4f}")])


def create_dataset_statistics_tab(window):
    window.dataset_statistics_tab = DatasetStatisticsTab(window)
    window.tabs.addTab(window.dataset_statistics_tab, "Статистика и анализ")
