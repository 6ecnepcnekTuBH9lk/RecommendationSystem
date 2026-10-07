"""Shared, Qt-free labels and period presentation for statistics UI and export."""

from Application.analysis_filter import AnalysisFilter
from Application.statistics_cache import parse_timestamp
from Application.statistics_period import shared_intervals


SOURCE_LABELS = {"Actions": "Действия", "Orders": "Заказы", "Customers": "Клиенты",
                 "CustomerMerges": "Объединения клиентов"}


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
    selection = AnalysisFilter.from_dict(result["analysis_filter"])
    if selection.start_date is not None:
        return "Статистика рассчитана за " + selection.summary().split(";", 1)[0].removeprefix("Отбор → ") + "."
    sources = {source["source"]: source["intervals"] for source in result["coverage"]}
    merged = shared_intervals(sources.get("Actions", ()), sources.get("Orders", ()))
    if not merged:
        return "Статистика рассчитана."
    periods = "; ".join(
        f"{start:%d.%m.%Y} - {end:%d.%m.%Y}"
        for start, end in merged
    )
    return ("Статистика рассчитана за период: " if len(merged) == 1 else "Статистика рассчитана за периоды: ") + periods + "."


DIAGNOSTIC_LABELS = {
    "ambiguous_product_view_actions": "Просмотры с неоднозначным набором товаров",
    "unknown_currency_view_price_actions": "Просмотры с ценой без определённой валюты",
    "orders_outside_statistics_period": "Заказы вне периода статистики",
    "fractional_quantity_order_lines": "Позиции с нецелым количеством",
    "mapped_view_actions": "Исходные VIEW-события",
    "mapped_favorite_actions": "Исходные FAVORITE-события",
    "unmapped_actions": "Неклассифицированные исходные Actions",
    "mapped_without_product": "Исходные VIEW/FAVORITE без product ID",
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


SOURCE_ACTIONS_HEADING = "Исходные события Actions"
ACTION_QUALITY_HEADING = "Качество событий Actions"
ACTION_SOURCE_KEYS = ("mapped_view_actions", "mapped_favorite_actions", "unmapped_actions", "mapped_without_product")


def source_action_rows(result):
    """Raw event shares use the same date-scoped source population as counts."""
    total = result["source_actions"]
    return [(name, count, 100 * count / total if total else 0.) for name, count in result["action_types"]]


def source_action_quality_rows(result):
    diagnostics = dict(result["diagnostics"])
    return [("Исходные Actions", result["source_actions"]),
            ("Исходные Actions с товаром", result["actions_with_product"]),
            ("Исходные Actions без товара", result["actions_without_product"]),
            *((DIAGNOSTIC_LABELS[key], diagnostics[key]) for key in ACTION_SOURCE_KEYS)]
