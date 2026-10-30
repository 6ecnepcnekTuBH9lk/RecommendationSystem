"""Shared, Qt-free labels and period presentation for statistics UI and export."""

from datetime import date
from decimal import Decimal

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


BUYER_LEGACY_TEXT = 'Нет данных в сохранённом результате'
BUYER_NOTES = (
    ('Область наблюдения', 'Все товары; исходный выбранный период и доступное покрытие Orders. '
     'Разрывы покрытия не означают отсутствие покупок. Интервалы связывают наблюдаемые заказы; в разрывах могли быть другие покупки.'),
    ('Политики', 'STRICT: F/sold; EXTENDED: F/sold/CP. Количество положительное и целое. '
     'Денежные суммы CP не доказывают фактически переданное количество и стоимость.'),
    ('Дата покупки', 'firstAction.dateTimeUtc — дата оформления заказа со статусом доступного первого снимка, '
     'а не дата оплаты или доставки. Первая наблюдаемая покупка не обязательно первая за всю жизнь.'),
    ('Денежные знаменатели', 'Средний и медианный чек — по заказам с достоверной суммой. '
     'Известный подытог на покупателя может быть неполным. Полный оборот и квинтили — только по покупателям, '
     'у которых оценены все покупки периода; RUB и KZT раздельно.'),
    ('Повторные покупатели и квинтили', 'Повторный покупатель имеет от двух заказов политики за период, в любых валютах. '
     'Одинаковые полные суммы не разделяются между квинтилями; непустых групп может быть меньше пяти. '
     'Годовые суммы относятся к показанному покрытию и не экстраполируются.'),
    ('Интервалы', 'Календарные UTC-дни между соседними уникальными заказами после сортировки дат. '
     'Среднее взвешено всеми интервалами, а не индивидуальными средними. Два заказа одного дня дают 0 дней.'),
    ('Портрет покупателей', 'Возраст рассчитан по доступному текущему профилю на дату расчёта, '
     'а не на момент совершения покупки. Неопределённые пол и возраст включают покупателей без профиля.'),
    ('Возвраты и денежные операции', 'Возвратные позиции не создают покупки и не вычитаются автоматически. '
     'Сумма принятых товарных позиций не является чистой выручкой или подтверждением денежного поступления. '
     'Счётчик возвратных позиций относится только к адаптированным строкам.'),
)


def buyer_sections(result):
    """Shared typed display projection. No analytical formulas or source reads.

    Each section is (title, KPI pairs, [(headers, rows)], note pairs).
    Decimal/date values remain typed for Excel and HTML; UI formats them only.
    """
    saved = result['buyer_statistics'] if 'buyer_statistics' in result else None
    heading = 'Покупательская аналитика — STRICT / EXTENDED'
    if saved is None:
        return [(heading, [], [(['Показатель', 'Значение'], [('Новые показатели покупателей', BUYER_LEGACY_TEXT)])], [])]
    from Application.statistics_purchase_policy import STRICT_COMPLETED, EXTENDED_COMPLETED
    strict, extended = (saved['policies'][p] for p in (STRICT_COMPLETED, EXTENDED_COMPLETED))
    sections = []
    shown = lambda value: 'Нет данных' if value is None else value
    money = lambda value: 'Нет данных' if value is None else Decimal(value)
    stamp = lambda value: 'Нет данных' if value is None else date.fromisoformat(value)
    if strict is not None:
        sections.append(('Покупатели — подтверждённые продажи (STRICT)', [
            ('Покупатели STRICT', strict['buyers']), ('Заказы STRICT', strict['orders']),
            ('Повторные покупатели STRICT', strict['repeat_buyers']), ('Доля повторных покупателей STRICT, %', shown(strict['repeat_rate'])),
            ('Среднее число заказов STRICT', shown(strict['mean_orders'])), ('Медиана заказов STRICT', shown(strict['median_orders']))],
            [(['Показатель STRICT', 'Значение'], [
                ('Принятые позиции STRICT', strict['positions']), ('Единицы товаров STRICT', money(strict['units'])),
                ('Покупатели с одним заказом STRICT', strict['single_buyers']), ('Среднее единиц на заказ STRICT', money(strict['mean_units'])),
                ('Первая наблюдаемая дата STRICT', stamp(strict['first_date'])), ('Последняя наблюдаемая дата STRICT', stamp(strict['last_date']))])], []))
        intervals = strict['intervals']
        sections.append(('Частота и интервалы покупок — STRICT', [
            ('Средний интервал STRICT, дней', shown(intervals['mean_days'])), ('Медианный интервал STRICT, дней', shown(intervals['median_days'])),
            ('Покупатели с интервалами STRICT', intervals['buyers']), ('Наблюдаемые интервалы STRICT', intervals['count'])], [
            (['Заказов на покупателя STRICT', 'Покупателей', 'Доля покупателей, %', 'Заказов'], strict['frequency']),
            (['Интервал STRICT, дней', 'Интервалов', 'Доля интервалов, %'], intervals['distribution']),
            (['Месяц первой наблюдаемой покупки STRICT', 'Покупателей', 'Доля, %'], strict['first_months']),
            (['Месяц последней наблюдаемой покупки STRICT', 'Покупателей', 'Доля, %'], strict['last_months'])], []))
        for currency in ('RUB', 'KZT'):
            data = strict['money'][currency]
            sections.append((f'Денежные показатели покупателей STRICT — {currency}', [
                (f'Сумма принятых товарных позиций STRICT, {currency} (известная часть)', money(data['amount'])),
                (f'Средний чек STRICT, {currency}', money(data['mean_order'])),
                (f'Медианный чек STRICT, {currency}', money(data['median_order'])),
                (f'Средняя полная сумма на покупателя STRICT, {currency}', money(data['mean_complete_per_buyer'])),
                (f'Медианная полная сумма на покупателя STRICT, {currency}', money(data['median_complete_per_buyer'])),
                (f'Доля повторных покупателей в известной сумме STRICT {currency}, %', shown(data['repeat_amount_rate']))], [
                (['Денежный показатель STRICT '+currency, 'Значение'], [
                    ('Заказы с достоверной суммой', data['known_orders']), ('Покупатели с известными суммами заказов', data['known_buyers']),
                    ('Средний известный подытог на покупателя', money(data['mean_known_per_buyer'])),
                    ('Медианный известный подытог на покупателя', money(data['median_known_per_buyer'])),
                    ('Покупатели с полностью оценёнными покупками периода', data['complete_buyers']),
                    ('Полный оборот включённых покупателей', money(data['complete_amount'])),
                    ('Повторные среди покупателей полного оборота', data['complete_repeat_buyers']),
                    ('Доля повторных среди покупателей полного оборота, %', shown(data['complete_repeat_rate'])),
                    ('Сумма полных оборотов повторных покупателей', money(data['complete_repeat_amount'])),
                    ('Доля повторных в полном обороте, %', shown(data['complete_repeat_amount_rate']))]),
                (['Полный оборот от, '+currency, 'Полный оборот до, '+currency, 'Покупателей', 'Доля покупателей, %',
                  'Сумма, '+currency, 'Повторных покупателей', 'Доля повторных, %', 'Сумма повторных, '+currency,
                  'Доля суммы повторных, %'],
                 [(money(row[0]), money(row[1]), *row[2:4], money(row[4]), *row[5:7], money(row[7]), shown(row[8]))
                  for row in data['quintiles']]),
                (['Год UTC STRICT '+currency, 'Покрытие года', 'Заказов с суммой', 'Покупателей с суммами',
                  'Сумма позиций, '+currency, 'Покупателей полной оценки года', 'Полная сумма года, '+currency,
                  'Средняя полная сумма года, '+currency, 'Медианная полная сумма года, '+currency],
                 [(row['year'], ('Полный календарный год' if row['complete_year'] else 'Неполный год') + ': ' +
                   '; '.join(a + ' — ' + b + ' (конец исключён)' for a, b in row['coverage']),
                   row['money'][currency]['known_orders'], row['money'][currency]['known_buyers'], money(row['money'][currency]['amount']),
                   row['money'][currency]['complete_buyers'], money(row['money'][currency]['complete_amount']),
                   money(row['money'][currency]['mean_complete_per_buyer']), money(row['money'][currency]['median_complete_per_buyer']))
                  for row in strict['annual']])], []))
        demo = strict['demography']
        demo_cards = [(label, 'Нет данных' if demo is None else shown(demo[key])) for label, key in (
            ('Покупатели STRICT с профилем', 'profiled_buyers'), ('Покупатели STRICT без профиля', 'missing_profiles'),
            ('Неопределённый пол покупателей STRICT', 'unknown_gender'), ('Неопределённый возраст покупателей STRICT', 'unknown_age'),
            ('Средний возраст покупателей STRICT', 'mean_age'), ('Медианный возраст покупателей STRICT', 'median_age'))]
        sections.append(('Портрет покупателей STRICT', demo_cards, [
            (['Пол покупателей STRICT', 'Покупателей', 'Доля, %'], demo['gender'] if demo else [('Нет данных', 'Нет данных', 'Нет данных')]),
            (['Возраст покупателей STRICT', 'Покупателей', 'Доля, %'], demo['age'] if demo else [('Нет данных', 'Нет данных', 'Нет данных')])], []))
        comparison = [(label, shown(strict[key]), shown(extended[key])) for label, key in (
            ('Покупатели', 'buyers'), ('Заказы', 'orders'), ('Принятые позиции', 'positions'), ('Повторные покупатели', 'repeat_buyers'),
            ('Доля повторных, %', 'repeat_rate'), ('Среднее заказов на покупателя', 'mean_orders'), ('Медиана заказов', 'median_orders'))]
        comparison.append(('Единицы товаров', money(strict['units']), money(extended['units'])))
        comparison.append(('Среднее единиц на заказ', money(strict['mean_units']), money(extended['mean_units'])))
        for label, key in (('Первая наблюдаемая дата', 'first_date'), ('Последняя наблюдаемая дата', 'last_date')):
            comparison.append((label, stamp(strict[key]), stamp(extended[key])))
        for label, key in (('Покупатели с интервалами', 'buyers'), ('Наблюдаемые интервалы', 'count'),
                           ('Средний интервал, дней', 'mean_days'), ('Медианный интервал, дней', 'median_days')):
            comparison.append((label, shown(strict['intervals'][key]), shown(extended['intervals'][key])))
        for currency in ('RUB','KZT'):
            comparison.append((f'Известная сумма позиций, {currency}', money(strict['money'][currency]['amount']), money(extended['money'][currency]['amount'])))
            comparison.append((f'Заказы с достоверной суммой, {currency}', strict['money'][currency]['known_orders'], extended['money'][currency]['known_orders']))
            for label, key in (('Покупатели с известными суммами заказов', 'known_buyers'),
                               ('Покупатели полной денежной оценки', 'complete_buyers')):
                comparison.append((label + ', ' + currency, strict['money'][currency][key], extended['money'][currency][key]))
            for label, key in (('Средний чек', 'mean_order'), ('Медианный чек', 'median_order'),
                               ('Средний известный подытог на покупателя', 'mean_known_per_buyer'),
                               ('Медианный известный подытог на покупателя', 'median_known_per_buyer'),
                               ('Полный оборот включённых покупателей', 'complete_amount'),
                               ('Средняя полная сумма на покупателя', 'mean_complete_per_buyer'),
                               ('Медианная полная сумма на покупателя', 'median_complete_per_buyer'),
                               ('Известная сумма повторных покупателей', 'repeat_amount')):
                comparison.append((label + ', ' + currency, money(strict['money'][currency][key]), money(extended['money'][currency][key])))
            comparison.append((f'Доля повторных в известной сумме {currency}, %',
                               shown(strict['money'][currency]['repeat_amount_rate']), shown(extended['money'][currency]['repeat_amount_rate'])))
        sections.append(('Сравнение STRICT и EXTENDED', [], [(['Показатель', 'STRICT: F/sold', 'EXTENDED: F/sold/CP'], comparison)], []))
    else:
        message = 'Orders недоступен' if not saved['orders_available'] else 'Нет покрытия Orders выбранного периода'
        sections.append((heading, [], [(['Показатель', 'Значение'], [('Новые показатели покупателей', message)])], []))
    quality = []
    if strict is not None:
        for label, key in (
            ('Заказы без достоверной Monetary', 'unvalued_orders'), ('Неопределённая валюта', 'unknown_currency_orders'),
            ('Смешанная валюта', 'mixed_currency_orders'), ('Некорректная цена', 'invalid_price_orders'),
            ('Покупатели с полностью оценёнными покупками', 'complete_buyers'), ('Покупатели с частичной оценкой', 'partial_buyers'),
            ('Покупатели без достоверной оценки', 'unvalued_buyers')):
            quality.append((label, strict['quality'][key], extended['quality'][key]))
    periods = [(a, b) for a,b in saved['coverage']]
    context = ('Вся доступная история Orders' if saved['period']['start_date'] is None else
               'Выбранный период UTC: '+saved['period']['start_date']+' — '+saved['period']['end_date'])
    notes = [('Период покупателей', context),
             ('Покрытие', 'Покрытие непрерывно и полностью соответствует выбранной области' if saved['coverage_complete'] else
              'Покрытие неполное, содержит разрывы или отсутствует; ненаблюдаемые периоды не равны отсутствию покупок'),
             ('Дата оценки возраста', saved['as_of']), ('Доступность профилей', 'Доступны' if saved['profiles_available'] else 'Нет данных'),
             ('Возвратные адаптированные позиции', shown(saved['return_positions'])), *BUYER_NOTES]
    sections.append(('Качество и ограничения покупательской аналитики', [], [
        (['Начало покрытия Orders, UTC (включительно)', 'Конец покрытия Orders, UTC (исключительно)'], periods),
        (['Качество Monetary', 'STRICT', 'EXTENDED'], quality)], notes))
    return [(title, [(label, shown(value)) for label, value in cards],
             [(headers, [tuple(shown(value) for value in row) for row in rows]) for headers, rows in tables], notes)
            for title, cards, tables, notes in sections]
