"""XLSX projection of an already validated snapshot; no Qt, raw reads or recalculation."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import math
import os
from pathlib import Path
import tempfile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from Application.statistics_presentation import (
    SOURCE_ACTIONS_HEADING, ACTION_QUALITY_HEADING, source_action_rows, source_action_quality_rows,
    SOURCE_EVENT_QUALITY_HEADING, SOURCE_EVENT_QUALITY_NOTES, RESOLUTION_NOTES, source_event_quality_rows,
    CATEGORY_HEADERS, DIRECT_CATEGORY_NOTE, product_category_rows,
    CUSTOMER_SCOPE_NOTES, customer_scope_sections, customer_scope_context,
    PROFILE_SLICE_LABEL, ITEM_CLIENTS_LABEL,
)
from Application.analysis_filter import AnalysisFilter
from Application.statistics_cache import parse_timestamp
from Application.statistics_presentation import (SOURCE_LABELS, SOURCE_KIND_LABELS, DIAGNOSTIC_LABELS,
                                                 _date, _date_time, _calculation_message)


SHEET_NAMES = ("Сводка", "Действия", "Заказы", "Товары", "Клиенты", "Техническая информация")
INK = "242424"
HEADER_FILL = "D6E5F2"
SECTION_FILL = "EAF1F7"
BORDER_COLOR = "BDCCDA"
PERCENT_FORMAT = '0.00"%"'  # Snapshot percentages are already on the 0..100 scale.
MAX_ROW_HEIGHT = 96
ROW_HEIGHT = 21


@dataclass(frozen=True)
class _Numeric:
    value: Decimal
    number_format: str


def _decimal(value, *, money=True):
    number = Decimal(value)
    places = len(format(number, "f").partition(".")[2].rstrip("0"))
    quantity_format = "#,##0" + ("." + "0" * places if places else "")
    return _Numeric(number, "#,##0.00" if money else quantity_format)


class _Sheet:
    """Plain metric/data tables; no charts, navigation or presentation tiles."""

    def __init__(self, worksheet):
        self.ws = worksheet
        self.row = 1
        self.max_columns = 2
        self.ws.sheet_view.showGridLines = True
        self.ws.sheet_view.zoomScale = 90
        self.ws.sheet_properties.pageSetUpPr.fitToPage = True
        self.ws.sheet_format.defaultRowHeight = ROW_HEIGHT
        widths = (34, 48, 18, 20, 20, 20, 22)
        if worksheet.title == "Товары":
            widths = (20, 48, 72, 18, 20, 20, 20, 22)
        for column, width in enumerate(widths, 1):
            self.ws.column_dimensions[get_column_letter(column)].width = width
        self.write(["KANZLER AI"], section=True)
        self.write([worksheet.title], section=True)
        self.row += 1


    def _cell(self, row, column, value, *, header=False, fmt=None):
        cell = self.ws.cell(row, column)
        if isinstance(value, _Numeric):
            value, fmt = value.value, value.number_format
        cell.value = value
        if isinstance(value, str):
            cell.data_type = "s"  # Also protects =, +, -, @ labels and leading-zero SKUs.
            fmt = "@"
        cell.number_format = fmt or ("dd.mm.yyyy" if isinstance(value, date) else
                                    "#,##0.00" if isinstance(value, float) else "#,##0")
        cell.font = Font(name="Arial", size=10, bold=header, color=INK)
        cell.alignment = Alignment(vertical="center", wrap_text=True,
                                   horizontal="center" if header else "left" if isinstance(value, str) else "right")
        if header:
            cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
        side = Side(style="thin", color=BORDER_COLOR)
        cell.border = Border(left=side, right=side, top=side, bottom=side)
        return cell

    def write(self, values, *, header=False, section=False, formats=None):
        height = ROW_HEIGHT
        for column, value in enumerate(values, 1):
            fmt = (formats or {}).get(column)
            if not header and column == 2 and isinstance(values[0], str) and values[0].startswith("Доля"):
                fmt = PERCENT_FORMAT
            cell = self._cell(self.row, column, value, header=header, fmt=fmt)
            if section:
                cell.font = Font(name="Arial", size=11, bold=True, color=INK)
                cell.fill = PatternFill("solid", fgColor=SECTION_FILL)
            value = cell.value
            width = self.ws.column_dimensions[get_column_letter(column)].width
            if value is not None:
                height = max(height, 13.5 * sum(max(1, math.ceil(len(line) / (width - 3)))
                                             for line in str(value).split("\n")) + 6)
        self.ws.row_dimensions[self.row].height = min(48 if header else MAX_ROW_HEIGHT, height)
        self.row += 1

    def heading(self, title):
        self.write([title], section=True)
        self.row += 1

    def table(self, headers, rows):
        header = self.row
        self.max_columns = max(self.max_columns, len(headers))
        self.write(headers, header=True)
        formats = {i: PERCENT_FORMAT for i, label in enumerate(headers, 1) if "%" in label}
        for values in rows:
            if tuple(headers) == CATEGORY_HEADERS and len(values[2]) > 32767:
                raise ValueError("Путь категории превышает лимит ячейки XLSX (32767 символов).")
            self.write(values, formats=formats)
        last = self.row - 1
        if last > header and len(headers) > 1:
            # Each independent block has its own filter, never one worksheet-wide AutoFilter.
            name = f"Stats{self.ws.parent.index(self.ws) + 1}Block{header}"
            table = Table(displayName=name, ref=f"A{header}:{get_column_letter(len(headers))}{last}")
            table.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showFirstColumn=False,
                showLastColumn=False, showRowStripes=False, showColumnStripes=False)
            self.ws.add_table(table)
        elif last == header:
            self.write(["Нет данных для таблицы."])
        self.row += 1

    def finish(self):
        self.ws.freeze_panes = None
        self.ws.print_title_rows = "1:2"
        self.ws.print_options.horizontalCentered = False
        wide = self.max_columns > 3
        self.ws.page_setup.orientation = "landscape" if wide else "portrait"
        self.ws.page_setup.paperSize = self.ws.PAPERSIZE_A3 if wide else self.ws.PAPERSIZE_A4
        self.ws.page_setup.fitToWidth = 1
        self.ws.page_setup.fitToHeight = 0
        self.ws.page_margins.left = self.ws.page_margins.right = .25
        self.ws.print_area = f"A1:{get_column_letter(self.max_columns)}{self.row}"
        self.ws.oddFooter.center.text = "Страница &P из &N"


    def cards(self, values):
        self.table(["Показатель", "Значение"], [
            (label, _Numeric(Decimal(str(value)), PERCENT_FORMAT)
             if label.startswith("Доля") and value is not None and not isinstance(value, str) else value)
            for label, value in values])


def _selection_values(values):
    return "Все значения" if values is None else "; ".join("Не указано" if v is None else v for v in values)


def _summary(sheet, result):
    selection = AnalysisFilter.from_dict(result["analysis_filter"])
    metadata = [
        ("Период расчета", _calculation_message(result)),
        ("Дата и время расчета", _date_time(result["calculated_at"])),
        ("Вид номенклатуры", _selection_values(selection.nomenclature_types)),
        ("Сезон", _selection_values(selection.collections)),
    ]
    metadata = [(label, (f"Выбрано значений: {len(selection.nomenclature_types if label == 'Вид номенклатуры' else selection.collections)}. "
                          "Полный список: лист «Техническая информация».")
                 if label in ("Вид номенклатуры", "Сезон") and isinstance(value, str) and len(value) > 500 else value)
                for label, value in metadata]
    sheet.table(["Параметр отчёта", "Значение"], metadata)
    sheet.heading("Основные показатели")
    main = (
        ("Количество взаимодействий", "total_interactions"), ("Количество заказов", "orders"),
        ("Количество позиций в заказах", "order_lines"), (PROFILE_SLICE_LABEL, "customers"),
        (ITEM_CLIENTS_LABEL, "interaction_users"), ("Покупатели", "purchase_users"),
        ("Повторные покупатели", "repeat_buyers"), ("Доля повторных покупателей", "repeat_buyer_rate"))
    sheet.cards([(label, result[key] if result[key] is not None else "Нет данных") for label, key in main])


def suggested_filename(result):
    return "Статистика_" + parse_timestamp(result["calculated_at"]).strftime("%Y-%m-%d_%H-%M-%S") + ".xlsx"


def export_statistics(result, path):
    """Save the displayed snapshot atomically, leaving an old destination intact on failure.

    Decimal values are passed directly to openpyxl. Excel numeric cells retain
    Excel's native precision limit; formatting never converts numbers to text.
    """
    destination = Path(path)
    book = Workbook()
    book.remove(book.active)
    temporary = None
    try:
        sheets = [_Sheet(book.create_sheet(name)) for name in SHEET_NAMES]
        _summary(sheets[0], result)
        _actions_page(sheets[1], result)
        _orders_page(sheets[2], result)
        _products_page(sheets[3], result)
        _clients_technical(sheets[4], sheets[5], result)
        for sheet in sheets:
            sheet.finish()
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".statistics-", suffix=".xlsx", delete=False) as stream:
            temporary = Path(stream.name)
            book.save(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        book.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _actions_page(layout, result):
    table = layout.table

    layout.cards([("Просмотры товаров", result["view_interactions"]),
                    ("Добавления в избранное", result["favorite_interactions"]),
                    ("Клиенты с просмотрами", result["view_users"]),
                    ("Клиенты с избранным", result["favorite_users"])])
    layout.heading(SOURCE_EVENT_QUALITY_HEADING)
    quality = source_event_quality_rows(result)
    layout.cards([(name, "Нет данных" if value is None else value) for name, value in quality[6:7]])
    table(["Показатель", "Значение"],
          [(name, "Нет данных" if value is None else value) for name, value in quality[:6] + quality[7:]])
    table(["Пояснение", "Определение"], SOURCE_EVENT_QUALITY_NOTES)
    layout.heading("Активность клиентов")
    table(["Тип действия", "Количество взаимодействий", "Количество клиентов", "Среднее на клиента", "Медиана на клиента"], [
        ("Просмотры", result["view_interactions"], result["view_users"], result["mean_views_per_viewer"], result["median_views_per_viewer"]),
        ("Добавления в избранное", result["favorite_interactions"], result["favorite_users"],
         result["mean_favorites_per_user"], result["median_favorites_per_user"])])
    table(["Количество просмотров", "Количество клиентов", "Доля, %"], result["view_user_activity_distribution"])
    table(["Количество избранного", "Количество клиентов", "Доля, %"], result["favorite_user_activity_distribution"])
    layout.heading("Каналы взаимодействий")
    table(["Канал", "Просмотры", "Клиенты с просмотрами", "Доля просмотров, %",
                    "Избранное", "Клиенты с избранным", "Доля избранного, %"],
           [row[1:] for row in result["action_channel_statistics"]])
    layout.heading("Динамика действий")
    table(["Месяц", "Просмотры", "Клиенты с просмотрами", "Избранное", "Клиенты с избранным"],
           [(month[5:] + "." + month[:4], *values) for month, *values in result["action_monthly_dynamics"]])
    layout.heading("Параметры просмотров")
    table(["Доступность", "Просмотров", "Доля, %"], result["view_availability_distribution"])
    table(["Валюта", "Просмотров с ценой", "Средняя цена", "Медианная цена"],
           [(currency, n, _decimal(mean), _decimal(middle))
            for currency, n, mean, middle in result["view_price_statistics"]])


def _orders_page(layout, result):
    table = layout.table

    layout.cards([("Заказы с покупкой", result["purchase_orders"]),
                    ("Невыкупленные заказы", result["orders_without_purchase"]),
                    ("Позиции покупок", result["purchase_interactions"]), ("Покупатели", result["purchase_users"])])
    layout.heading("Корзина заказа")
    basket_rates = {label: rate for label, _, rate in result["purchase_basket_distribution"]}
    layout.cards([("Среднее число позиций на заказ", result["mean_purchase_lines_per_order"]),
                    ("Медиана числа позиций на заказ", result["median_purchase_lines_per_order"]),
                    ("Доля заказов с 1 позицией", basket_rates["1 позиция"]),
                    ("Доля заказов с 3+ позициями", sum(basket_rates[key] for key in ("3–5 позиций", "6–10 позиций", "11+ позиций")))])
    table(["Количество позиций", "Заказов", "Доля, %"], result["purchase_basket_distribution"])
    layout.heading("Финансовые показатели")
    table(["Валюта", "Заказов", "Позиции", "Единиц", "Сумма покупок", "Средняя сумма заказа", "Медианная сумма заказа"],
           [(currency, n, lines, _decimal(units, money=False), *(_decimal(v) for v in (amount, mean, median)))
            for currency, n, lines, units, amount, mean, median in result["order_financials"]])
    mixed, unknown = result["mixed_currency_purchase_orders"], result["unknown_currency_purchase_orders"]
    if mixed or unknown:
        layout.cards([("Заказы со смешанной валютой", mixed), ("Заказы с неопределённой валютой", unknown)])
    layout.heading("Динамика покупок")
    table(["Месяц", "Заказы RUB", "Сумма RUB", "Заказы KZT", "Сумма KZT"],
           [(month[5:] + "." + month[:4], rub_n, _decimal(rub), kzt_n, _decimal(kzt))
            for month, rub_n, rub, kzt_n, kzt in result["order_monthly_dynamics"]])
    for currency in ("RUB", "KZT"):
        layout.heading("Магазины и каналы — " + currency)
        table(["Магазин / канал", "Заказов", "Покупателей", "Позиции", "Единиц", "Сумма покупок", "Средняя сумма заказа"],
               [(name, n, buyers, lines, _decimal(units, money=False), _decimal(amount), _decimal(mean))
                for unit, key, name, n, buyers, lines, units, amount, mean in result["store_statistics"] if unit == currency])
    layout.heading("Оплата")
    table(["Способ оплаты", "Заказов с этим способом", "Доля заказов, %"], result["payment_type_distribution"])
    layout.heading("Статусы позиций")
    table(["Статус", "Позиции", "Доля, %", "Считается покупкой"],
           [(name, count, 100 * count / result["order_lines"] if result["order_lines"] else 0, "Да" if purchase else "Нет")
            for name, count, purchase in result["line_statuses"]])


def _products_page(layout, result):
    table = layout.table

    def group(first, field):
        table([first, "Товаров", "Просмотры", "Добавления в избранное", "Покупки", "Продано единиц"],
              [(*row[:5], _decimal(row[5], money=False)) for row in result[field]])

    layout.cards([("Товары с взаимодействиями", result["unique_resolved_items"]),
                    ("Товары с просмотрами", result["products_with_views"]),
                    ("Товары в избранном", result["products_with_favorites"]),
                    ("Купленные товары", result["products_with_purchases"])])
    layout.heading("Популярные товары по просмотрам")
    table(["Код", "Название", "Просмотры", "Клиенты с просмотрами", "Добавления в избранное", "Покупки"],
          result["top_viewed_products"])
    layout.heading("Популярные товары по избранному")
    table(["Код", "Название", "Добавления в избранное", "Клиенты с избранным", "Просмотры", "Покупки"],
          result["top_favorited_products"])
    layout.heading("Популярные товары по покупкам")
    table(["Код", "Название", "Покупки", "Покупатели", "Продано единиц", "Просмотры", "Добавления в избранное"],
          [(*row[:4], _decimal(row[4], money=False), *row[5:]) for row in result["top_purchased_products"]])
    layout.heading("Категории товаров")
    table(["Пояснение", "Определение"], [("Прямые категории", DIRECT_CATEGORY_NOTE)])
    table(CATEGORY_HEADERS,
          [(*row[:7], _decimal(row[7], money=False)) for row in product_category_rows(result)])
    for title, field in (("Спрос по полу товара", "product_gender_statistics"), ("Спрос по сезону", "product_season_statistics"),
                         ("Спрос по стилевой группе", "product_style_statistics")):
        layout.heading(title)
        group("Значение", field)
    layout.heading("Качество сопоставления")
    table(["Пояснение", "Определение"], RESOLUTION_NOTES)
    table(["Показатель", "Значение"], [
        ("Уникальные исходные идентификаторы товаров", result["unique_source_products"]),
        ("Уникальные распознанные товары", result["unique_resolved_items"]),
        ("Распознанные взаимодействия", result["resolved_interactions"]),
        ("Нераспознанные взаимодействия", result["unresolved_interactions"]),
        ("Доля распознанных, %", result["resolution_rate"])])
    table(["Система идентификаторов", "Взаимодействия", "Распознано", "Не распознано", "Доля распознанных, %"],
          [("Неподдерживаемая" if name == "unsupported" else name, total, resolved, unresolved, rate)
           for name, total, resolved, unresolved, rate in result["namespaces"]])


def _clients_technical(customers, technical, result):
    coverage = []
    for source in result["coverage"]:
        period = "; ".join(f"{_date(a)} — {_date(b)}" for a, b in source["intervals"])
        if not period and source["updated"]:
            period = "Снимок данных от " + _date_time(source["updated"])
        coverage.append((SOURCE_LABELS[source["source"]], period or "—",
                         "/".join(SOURCE_KIND_LABELS.get(kind, kind) for kind in source["source_kinds"]) or "—"))
    technical.table(["Источник", "Период / состояние", "Источник данных"], coverage)
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
            ("Доля распознанных, %", result["resolution_rate"])]
    technical.table(["Показатель", "Значение"], base)
    technical.heading(SOURCE_ACTIONS_HEADING)
    technical.table(["Системное название", "Количество", "Доля, %"], source_action_rows(result))
    technical.heading(ACTION_QUALITY_HEADING)
    technical.table(["Классификация событий", "Количество"], source_action_quality_rows(result))
    customers.cards([(PROFILE_SLICE_LABEL, result["customers"]),
                      (ITEM_CLIENTS_LABEL, result["interaction_users"]),
                      ("Покупатели", result["purchase_users"]), ("Повторные покупатели (от 2 покупок)", result["repeat_buyers"])])
    for title, rows in customer_scope_sections(result):
        customers.heading(title)
        customers.cards(rows)
    customers.table(['Пояснение', 'Определение'], [('Период и полнота', customer_scope_context(result)), *CUSTOMER_SCOPE_NOTES])
    customers.heading("Портрет клиента")
    customers.table(["Пол", "Количество клиентов", "Доля, %"], result["gender_distribution"])
    customers.cards([("Средний возраст", result["mean_age"]), ("Медианный возраст", result["median_age"])])
    customers.table(["Возрастная группа", "Количество клиентов", "Доля, %"], result["age_distribution"])
    customers.heading("Активность клиентов")
    customers.table(["Показатель", "Значение"], [
        ("Клиенты с просмотрами", result["view_users"]),
        ("Клиенты с добавлениями в избранное", result["favorite_users"]),
        ("Клиенты с покупками", result["purchase_users"]),
        ("Клиенты с просмотрами и покупками", result["view_purchase_users"]),
        ("Клиенты с избранным и покупками", result["favorite_purchase_users"]),
        ("Клиенты со всеми типами взаимодействий", result["all_interaction_type_users"]),
        ("Среднее число взаимодействий на активного клиента", result["mean_interactions"]),
        ("Медиана числа взаимодействий на активного клиента", result["median_interactions"]),
        ("Доля активных клиентов, совершивших покупку, %", result["active_buyer_rate"])])
    customers.table(["Количество взаимодействий", "Клиентов", "Доля, %"], result["interaction_activity_distribution"])
    customers.heading("Покупательская активность")
    customers.table(["Количество заказов", "Покупателей", "Доля покупателей, %"], result["purchase_order_distribution"])
    customers.table(["Показатель", "Значение"], [
        ("Среднее число заказов на покупателя", result["mean_orders_per_buyer"]),
        ("Медиана числа заказов на покупателя", result["median_orders_per_buyer"]),
        ("Доля повторных покупателей, %", result["repeat_buyer_rate"])])
    technical.table(["Диагностика", "Количество"],
           [(DIAGNOSTIC_LABELS[key], value) for key, value in result["diagnostics"]]
           + [("Заказы выбранного среза" if filtered else "Исходные заказы", result["orders"]),
              ("Позиции заказов выбранного среза" if filtered else "Исходные позиции заказов", result["order_lines"]),
              ("Распознанные взаимодействия", result["resolved_interactions"]),
              ("Нераспознанные взаимодействия", result["unresolved_interactions"]),
              ("Доля распознанных, %", result["resolution_rate"])])
    _additional_details(technical, result)


def _additional_details(sheet, result):
    """Retain aggregate DTO fields beyond the GUI's compact selection; no raw identities."""
    sheet.heading("Дополнительные агрегаты snapshot")
    sheet.cards([(label, result[key]) for label, key in (
        ("Клиенты исходных Actions", "action_customers"), ("Клиенты исходных Orders", "order_customers"),
        ("Просмотры с параметрами", "view_parameter_actions"),
        ("Распознанные просмотры", "resolved_view_interactions"),
        ("Распознанное избранное", "resolved_favorite_interactions"),
        ("Распознанные позиции покупок", "resolved_purchase_interactions"),
    )] + [(label, _decimal(result[key], money=False)) for label, key in (
        ("Куплено единиц", "purchase_quantity"), ("Куплено распознанных единиц", "resolved_purchase_quantity"),
        ("Среднее число единиц на заказ", "mean_purchase_units_per_order"),
        ("Медиана числа единиц на заказ", "median_purchase_units_per_order"))])
    for title, field in (("Способы оформления", "ordering_method_distribution"),
                         ("Способы доставки", "delivery_type_distribution")):
        sheet.heading(title)
        sheet.table(["Значение", "Заказов", "Доля, %"], result[field])
    sheet.heading("Стоимость доставки по валютам")
    sheet.table(["Валюта", "Заказов с ценой доставки", "Сумма доставки", "Средняя стоимость", "Медианная стоимость"],
                [(currency, count, *(_decimal(value) for value in amounts))
                 for currency, count, *amounts in result["delivery_financials"]])
    sheet.heading("Товары по всем взаимодействиям")
    sheet.table(["Код", "Название", "Просмотры", "Избранное", "Покупки", "Все взаимодействия"], result["top_products"])
    selection = AnalysisFilter.from_dict(result["analysis_filter"])
    sheet.heading("Параметры отбора snapshot")
    sheet.table(["Параметр отбора", "Выбранное значение"], [
        (label, "Не указано" if value is None else value)
        for label, values in (("Вид номенклатуры", selection.nomenclature_types), ("Сезон", selection.collections))
        for value in (("Все значения",) if values is None else values)])
    sheet.cards([
        ("Отбор, по которому рассчитана статистика", selection.summary()),
        ("Дата начала", date.fromisoformat(selection.start_date) if selection.start_date else "Все значения"),
        ("Дата окончания", date.fromisoformat(selection.end_date) if selection.end_date else "Все значения"),
    ])
