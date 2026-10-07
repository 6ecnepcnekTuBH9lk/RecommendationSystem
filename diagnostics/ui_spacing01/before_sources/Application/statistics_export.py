"""XLSX projection of an already validated snapshot; no Qt, raw reads or recalculation."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import math
import os
from pathlib import Path
import tempfile

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from Application.statistics_presentation import (
    SOURCE_ACTIONS_HEADING, ACTION_QUALITY_HEADING, source_action_rows, source_action_quality_rows,
)
from Application.analysis_filter import AnalysisFilter
from Application.statistics_cache import parse_timestamp
from Application.statistics_presentation import (SOURCE_LABELS, SOURCE_KIND_LABELS, WARNING_LABELS, DIAGNOSTIC_LABELS,
                                                 _date, _date_time, _calculation_message)


SHEET_NAMES = ("Сводка", "Действия", "Заказы", "Товары", "Клиенты", "Техническая информация")


@dataclass(frozen=True)
class _Numeric:
    value: Decimal
    number_format: str


def _decimal(value, *, money=True):
    number = Decimal(value)
    places = max(0, -number.normalize().as_tuple().exponent)
    quantity_format = "#,##0" + ("." + "0" * places if places else "")
    return _Numeric(number, "#,##0.00" if money else quantity_format)


class _Sheet:
    """Small section writer; labels are literal text, never Excel formulas."""

    def __init__(self, worksheet):
        self.ws = worksheet
        self.row = 1
        self.heading(worksheet.title)
        self.ws.freeze_panes = "A2"
        self.ws.sheet_view.showGridLines = False

    def write(self, values, *, header=False, title=False):
        height = 22
        for column, value in enumerate(values, 1):
            cell = self.ws.cell(self.row, column)
            fmt = "#,##0.00" if isinstance(value, float) else "#,##0"
            if isinstance(value, _Numeric):
                value, fmt = value.value, value.number_format
            cell.value = value
            if isinstance(value, str):
                cell.data_type = "s"
                fmt = "@"
            elif isinstance(value, date):
                fmt = "dd.mm.yyyy"
            cell.number_format = fmt
            cell.font = Font(name="Calibri", size=12 if title else 11, bold=header or title,
                             color="FFFFFF" if header else "202830")
            cell.alignment = Alignment(vertical="top", wrap_text=True,
                                       horizontal="left" if isinstance(value, str) else "right")
            if header:
                cell.fill = PatternFill("solid", fgColor="3D5266")
            width = 48 if column == 1 else 32 if column == 2 else 23
            self.ws.column_dimensions[get_column_letter(column)].width = width
            if value is not None:
                height = max(height, 16 * sum(max(1, math.ceil(len(line) / (width - 3)))
                                             for line in str(value).split("\n")) + 6)
        self.ws.row_dimensions[self.row].height = height
        self.row += 1

    def heading(self, title):
        self.write([title], title=True)

    def note(self, text):
        self.write([text])
        self.row += 1

    def table(self, headers, rows):
        self.write(headers, header=True)
        for values in rows:
            self.write(values)
        self.row += 1

    def cards(self, values):
        self.table(["Показатель", "Значение"], [
            (label, _Numeric(Decimal(str(value)), '0.00"%"') if label.startswith("Доля") and value is not None else value)
            for label, value in values])


def _selection_values(values):
    return "Все значения" if values is None else "; ".join("Не указано" if v is None else v for v in values)


def _summary(sheet, result):
    selection = AnalysisFilter.from_dict(result["analysis_filter"])
    sheet.cards([
        ("Дата и время расчета", _date_time(result["calculated_at"])),
        ("Период расчета", _calculation_message(result)),
        ("Отбор, по которому рассчитана статистика", selection.summary()),
        ("Дата начала", date.fromisoformat(selection.start_date) if selection.start_date else "Все значения"),
        ("Дата окончания", date.fromisoformat(selection.end_date) if selection.end_date else "Все значения"),
        ("Вид номенклатуры", _selection_values(selection.nomenclature_types)),
        ("Сезон", _selection_values(selection.collections)),
    ])
    sheet.heading("Основные показатели")
    sheet.cards([(label, result[key]) for label, key in (
        ("Количество взаимодействий", "total_interactions"), ("Количество заказов", "orders"),
        ("Количество позиций в заказах", "order_lines"), ("Количество клиентов", "customers"))])
    _warnings(sheet, result)


def _warnings(sheet, result):
    if result["warnings"]:
        sheet.heading("Предупреждения")
        sheet.table(["Предупреждение"], [(WARNING_LABELS.get(w, w),) for w in result["warnings"]])


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
        _warnings(sheets[5], result)
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
    table(["Категория", "Товаров", "Просмотры", "Добавления в избранное", "Покупки", "Продано единиц"],
          [(*row[1:6], _decimal(row[6], money=False)) for row in result["product_category_statistics"]])
    for title, field in (("Спрос по полу товара", "product_gender_statistics"), ("Спрос по сезону", "product_season_statistics"),
                         ("Спрос по стилевой группе", "product_style_statistics")):
        layout.heading(title)
        group("Значение", field)
    layout.heading("Качество сопоставления")
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
    customers.cards([("Количество клиентов", result["customers"]),
                      ("Активные клиенты", result["interaction_users"]),
                      ("Покупатели", result["purchase_users"]), ("Повторные покупатели (от 2 покупок)", result["repeat_buyers"])])
    customers.heading("Портрет клиента")
    if result["customers"] is None:
        customers.note("Данные профилей клиентов отсутствуют. Пол и возраст недоступны.")
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
