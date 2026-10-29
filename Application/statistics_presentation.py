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

SOURCE_EVENT_QUALITY_HEADING = "Качество исходных событий"
SOURCE_EVENT_QUALITY_NOTES = (
    ("Единица счёта", "Одно исходное событие независимо от количества товарных вхождений."),
    ("Знаменатель доли", "Всего исходных VIEW/FAVORITE в выбранном периоде до товарного отбора. "
     "Доля без товара = 100 × события без товара / всего VIEW/FAVORITE."),
    ("Событие с товаром", "Есть товарное вхождение; корректность идентификатора и успешное "
     "сопоставление с каталогом этим не подтверждаются."),
)
RESOLUTION_NOTES = (
    ("Область расчёта", "Доля распознанных рассчитана среди товарных взаимодействий, "
     "включённых в анализ после применения отбора."),
    ("Полнота источника", "Она не характеризует полноту идентификаторов во всех исходных "
     "событиях VIEW/FAVORITE."),
    ("Типы взаимодействий", "В показатель могут входить покупки."),
)

CATEGORY_HEADERS = ("Код категории", "Название категории", "Путь категории", "Товаров",
                    "Просмотры", "Избранное", "Покупки", "Продано единиц")
DIRECT_CATEGORY_NOTE = ("Каждый товар учитывается в одной прямой категории. "
                        "Родительские категории не включают показатели дочерних категорий автоматически.")
LEGACY_CATEGORY_PATH = "Путь недоступен в сохранённом результате"
UNAVAILABLE_CATEGORY_PATH = "Путь недоступен в справочнике"
PROFILE_SLICE_LABEL = "Профили текущего среза"
ITEM_CLIENTS_LABEL = "Клиенты с товарными взаимодействиями"

CUSTOMER_SCOPE_NOTES = (
    ("Источники и время", "Профили Customers — доступный снимок. Actions и Orders — наблюдаемые события периода. "
     "Это не историческая реконструкция всех клиентов компании."),
    ("Множества", "Клиенты разных источников и типов активности пересекаются: их количества нельзя складывать. "
     "Общая база B — объединение канонических ID профилей, Actions и выбранных снимков Orders."),
    ("Снимки заказов", "Используется первый подходящий снимок заказа в периоде. "
     "Одинаковые и конфликтующие дубли не добавляют клиентов и не заменяют владельца заказа."),
    ("Цифровая активность", "Исходные VIEW/FAVORITE учитываются и без товара. Неклассифицированные Actions "
     "не считаются цифровой активностью VIEW/FAVORITE. Товарный отбор не меняет новый охват клиентов."),
    ("Покупки и знаменатель", "Покупатель — клиент с принятой позицией покупки по всем товарам и действующим статусам. "
     "Доли рассчитаны от B. Без наблюдаемой покупки в периоде не означает отсутствие покупок за всю жизнь."),
    ("Прежние показатели", "Количество профилей и клиентов с товарными взаимодействиями в прежних таблицах "
     "сохраняет свой товарный срез; эти показатели не заменяют объединённую базу B."),
)


def customer_scope_sections(result):
    scope = result['customer_scope'] if 'customer_scope' in result else None
    groups = (
        ('Охват клиентской базы', (
            ('Идентифицированные клиенты в доступных источниках', 'identified'),
            ('Уникальные клиенты с профилями', 'profiles'),
            ('Клиенты исходных Actions', 'actions'), ('Клиенты исходных Orders', 'orders'),
            ('Клиенты без доступного профиля', 'without_profile'),
            ('Клиенты только с профилем, без наблюдаемых Actions/Orders за период', 'profile_only'),
            ('Наблюдаемые ID клиентов событий (доступная часть)', 'events'))),
        ('Цифровая активность клиентов', (
            ('Клиенты с исходными VIEW', 'views'), ('Клиенты с исходными FAVORITE', 'favorites'),
            ('Клиенты с VIEW или FAVORITE', 'digital'), ('Клиенты с VIEW и FAVORITE', 'view_favorite'),
            ('Клиенты с VIEW/FAVORITE без товарного идентификатора', 'itemless_digital'))),
        ('Покупатели и клиенты без покупок', (
            ('Покупатели по всем товарам за выбранный период', 'buyers'),
            ('Без наблюдаемых покупок за выбранный период', 'non_buyers'),
            ('Доля покупателей от идентифицированной базы, %', 'buyer_rate'),
            ('Доля клиентов без наблюдаемых покупок, %', 'non_buyer_rate'))),
    )
    return [(title, [(label, 'Нет данных в сохранённом результате' if scope is None else
                     scope[key] if scope[key] is not None else 'Нет данных') for label, key in rows])
            for title, rows in groups]


def customer_scope_context(result):
    scope = result.get('customer_scope')
    if scope is None:
        return 'Новый охват не рассчитан в сохранённом результате. Пересчёт при открытии или экспорте не выполняется.'
    period = AnalysisFilter(**scope['period'])
    text = ('Период событий: ' + period.summary().removeprefix('Отбор → период ') + '.' if period.start_date else
            'Период событий: вся доступная наблюдаемая история; покрытие источников приведено в технической информации.')
    if not all(scope[k] for k in ('profiles_available', 'actions_available', 'orders_available')):
        text += ' Не все источники доступны: общая база B и её доли неизвестны. Наблюдаемые ID событий показывают доступную часть.'
    return text


def product_category_rows(result):
    """Only saved metadata; never consult today's category CSV during display."""
    saved = result["product_category_paths"] if "product_category_paths" in result else None
    paths = dict(saved) if saved is not None else {}
    rows = []
    for code, name, *values in result["product_category_statistics"]:
        if code is None:
            path = "Категория не указана"
        elif saved is None:
            path = LEGACY_CATEGORY_PATH
        else:
            # ASCII separator is readable even with the bundled Material font.
            path = " / ".join(paths[code]) if paths.get(code) else UNAVAILABLE_CATEGORY_PATH
        rows.append((code if code is not None else "Не указан", name, path, *values))
    return rows


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


def source_event_quality_rows(result):
    """Event-level quality from saved counters, independent of item selection.

    Do not substitute all-Action product counts or item-interaction counts.
    Inconsistent counters cannot establish a difference or a percentage; keep
    their original values visible and leave those derived values unavailable.
    """
    diagnostics = dict(result["diagnostics"])
    views = diagnostics["mapped_view_actions"]
    favorites = diagnostics["mapped_favorite_actions"]
    total = views + favorites
    missing = diagnostics["mapped_without_product"]
    unmapped = diagnostics["unmapped_actions"]
    consistent = total + unmapped == result["source_actions"] and 0 <= missing <= total
    return [
        ("Все исходные Actions", result["source_actions"]),
        ("Исходные VIEW", views),
        ("Исходные FAVORITE", favorites),
        ("Всего VIEW/FAVORITE", total),
        ("VIEW/FAVORITE с товаром", total - missing if consistent else None),
        ("VIEW/FAVORITE без товара", missing),
        ("Доля VIEW/FAVORITE без товара, %", 100 * missing / total if consistent and total else None),
        ("Неклассифицированные Actions", unmapped),
    ]
