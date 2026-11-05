"""Shared, Qt-free labels and period presentation for statistics UI and export."""

from datetime import date, time, timedelta
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


def interval_bounds(start, end):
    """Display a saved half-open interval, without changing its timezone.

    Whole calendar days use an inclusive last date. Partial days retain their
    exact times and explicit exclusive end; never round these into whole days.
    """
    first, last = parse_timestamp(start), parse_timestamp(end)
    if (first.timetz().replace(tzinfo=None) == time()
            and last.timetz().replace(tzinfo=None) == time()
            and first.utcoffset() == last.utcoffset() and last > first):
        return first.strftime("%d.%m.%Y"), (last.date() - timedelta(days=1)).strftime("%d.%m.%Y")
    def stamp(value):
        text = value.strftime("%d.%m.%Y %H:%M")
        if value.second or value.microsecond:
            text += value.strftime(":%S")
        if value.microsecond:
            text += f".{value.microsecond:06d}"
        if first.utcoffset() != last.utcoffset():
            text += " " + value.strftime("%z")
        return text
    return stamp(first), stamp(last) + " (не включая)"


def interval_text(start, end):
    return " — ".join(interval_bounds(start, end))


# Confirmed names supplied by the user in the Mindbox status reference.
# Distinct codes remain distinct display rows; this does not classify purchases.
STATUS_LABELS = {
    "N": "Принят", "DA": "Комплектация заказа", "DD": "Комплектация заказа (некомплект)",
    "DB": "Готов к отправке", "DG": "Заказ отправлен", "DS": "Доставлен в пункт самовывоза",
    "new": "Новый", "no-call": "Недозвон", "availability-confirmed": "Наличие подтверждено",
    "offer-analog": "Предложить замену", "assembling": "Комплектуется", "ready-to-wait": "Готов ждать",
    "client-confirmed": "Согласовано с клиентом", "sh": "Собран в магазине",
    "opr-perem": "Оприходовать перемещения", "soglasovanie": "Согласование",
    "tovar-naiden": "Товар найден", "kv": "К выполнению", "Оплачен": "Оплачен",
    "send-to-assembling": "Передано в комплектацию", "delivering": "Доставляется",
    "assembling-complete": "Укомплектован", "redirect": "Доставка перенесена",
    "prepayed": "Предоплата поступила", "gotov-otpravka": "Готов к отправке (crm)",
    "CP": "Выполнен частично", "F": "Выполнен", "sold": "Продан",
    "send-to-delivery": "Передан в доставку", "ready-for-self-pickup": "Готов к самовывозу",
    "arrived-in-pickup-point": "Прибыл в ПВЗ", "de": "Доставка по графику",
    "Return": "Возврат", "kurerskii-vozvrat": "Курьерский возврат", "klientskii-vozvrat": "Клиентский возврат",
    "CD": "Отменен", "RJ": "Отказ", "out-of-stock": "Нет в наличии", "otmenen": "Отменён",
    "no-product": "Нет в наличии (crm)", "already-buyed": "Купил в другом месте",
    "delyvery-did-not-suit": "Не устроила доставка",
    # Only the meaning/group is confirmed in statistics_purchase_contract.md.
    "Mindbox-Tech-Cancellation": "Техническая отмена",
}


def status_label(code):
    name = STATUS_LABELS.get(code)
    return f"{name} ({code})" if name is not None and name != code else code


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
        interval_text(start.isoformat(), end.isoformat())
        for start, end in merged
    )
    return ("Статистика рассчитана за период: " if len(merged) == 1 else "Статистика рассчитана за периоды: ") + periods + "."


DIAGNOSTIC_LABELS = {
    "ambiguous_product_view_actions": "Просмотры с неоднозначным набором товаров",
    "unknown_currency_view_price_actions": "Просмотры с ценой без определённой валюты",
    "orders_outside_statistics_period": "Заказы вне периода статистики",
    "fractional_quantity_order_lines": "Позиции с нецелым количеством",
    "mapped_view_actions": "Исходные просмотры",
    "mapped_favorite_actions": "Исходные добавления в избранное",
    "unmapped_actions": "Неклассифицированные исходные события",
    "mapped_without_product": "Исходные просмотры / избранное без идентификатора товара",
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


SOURCE_ACTIONS_HEADING = "Исходные события"
ACTION_QUALITY_HEADING = "Качество событий"
ACTION_SOURCE_KEYS = ("mapped_view_actions", "mapped_favorite_actions", "unmapped_actions", "mapped_without_product")

SOURCE_EVENT_QUALITY_HEADING = "Качество исходных событий"
SOURCE_EVENT_QUALITY_NOTES = (
    ("Единица счёта", "Одно исходное событие независимо от количества товарных вхождений."),
    ("Знаменатель доли", "Всего исходных просмотров и добавлений в избранное в выбранном периоде до товарного отбора. "
     "Доля без товара = 100 × события без товара / всего просмотров и добавлений в избранное."),
    ("Событие с товаром", "Есть товарное вхождение; корректность идентификатора и успешное "
     "сопоставление с каталогом этим не подтверждаются."),
)
RESOLUTION_NOTES = (
    ("Область расчёта", "Доля распознанных рассчитана среди товарных взаимодействий, "
     "включённых в анализ после применения отбора."),
    ("Полнота источника", "Она не характеризует полноту идентификаторов во всех исходных "
     "событиях просмотров и добавлений в избранное."),
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
    ("Источники и время", "Профили клиентов — доступный снимок. Действия и заказы — наблюдаемые события периода. "
     "Это не историческая реконструкция всех клиентов компании."),
    ("Множества", "Клиенты разных источников и типов активности пересекаются: их количества нельзя складывать. "
     "Общая база B — объединение канонических ID профилей, действий и выбранных снимков заказов."),
    ("Снимки заказов", "Используется первый подходящий снимок заказа в периоде. "
     "Одинаковые и конфликтующие дубли не добавляют клиентов и не заменяют владельца заказа."),
    ("Цифровая активность", "Исходные просмотры и добавления в избранное учитываются и без товара. Неклассифицированные события "
     "не считаются цифровой активностью просмотров / избранного. Товарный отбор не меняет новый охват клиентов."),
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
            ('Клиенты с исходными действиями', 'actions'), ('Клиенты с исходными заказами', 'orders'),
            ('Клиенты без доступного профиля', 'without_profile'),
            ('Клиенты только с профилем, без наблюдаемых действий или заказов за период', 'profile_only'),
            ('Идентификаторы клиентов событий (доступная часть)', 'events'))),
        ('Цифровая активность клиентов', (
            ('Клиенты с исходными просмотрами', 'views'), ('Клиенты с исходными добавлениями в избранное', 'favorites'),
            ('Клиенты с просмотрами или избранным', 'digital'), ('Клиенты с просмотрами и избранным', 'view_favorite'),
            ('Клиенты с просмотрами / избранным без товарного идентификатора', 'itemless_digital'))),
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
    return [("Исходные действия", result["source_actions"]),
            ("Исходные действия с товаром", result["actions_with_product"]),
            ("Исходные действия без товара", result["actions_without_product"]),
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
        ("Все исходные события", result["source_actions"]),
        ("Исходные просмотры", views),
        ("Исходные добавления в избранное", favorites),
        ("Всего просмотров / избранного", total),
        ("Просмотры / избранное с товаром", total - missing if consistent else None),
        ("Просмотры / избранное без товара", missing),
        ("Доля просмотров / избранного без товара, %", 100 * missing / total if consistent and total else None),
        ("Неклассифицированные события", unmapped),
    ]



def source_event_quality_display_rows(result):
    """Compact GUI values; preserve all source counts and the saved denominator."""
    return [(label, "Нет данных" if value is None else
             f"{value:.2f}".replace(".", ",") + " %" if position == 6 else value)
            for position, (label, value) in enumerate(source_event_quality_rows(result))]


BUYER_LEGACY_TEXT = 'Нет данных в сохранённом результате'
BUYER_NOTES = (
    ('Область наблюдения', 'Все товары; исходный выбранный период и доступное покрытие заказов. '
     'Разрывы покрытия не означают отсутствие покупок. Интервалы связывают наблюдаемые заказы; в разрывах могли быть другие покупки.'),
    ('Политики', 'Подтверждённые покупки: F/sold; Расширенный учёт покупок: F/sold/CP. Количество положительное и целое. '
     'Денежные суммы CP не доказывают фактически переданное количество и стоимость.'),
    ('Дата покупки', 'firstAction.dateTimeUtc — дата оформления заказа со статусом доступного первого снимка, '
     'а не дата оплаты или доставки. Первая наблюдаемая покупка не обязательно первая за всю жизнь.'),
    ('Денежные знаменатели', 'Средний и медианный чек — по заказам с достоверной суммой. '
     'Известный подытог на покупателя может быть неполным. Полный оборот и квинтили — только по покупателям, '
     'у которых оценены все покупки периода; RUB и KZT раздельно.'),
    ('Повторные покупатели и квинтили', 'Повторный покупатель имеет от двух заказов политики за период, в любых валютах. '
     'Одинаковые полные суммы не разделяются между квинтилями; непустых групп может быть меньше пяти. '
     'Годовые суммы относятся к показанному покрытию и не экстраполируются.'),
    ('Интервалы', 'Календарные дни между соседними уникальными заказами после сортировки дат. '
     'Среднее взвешено всеми интервалами, а не индивидуальными средними. Два заказа одного дня дают 0 дней.'),
    ('Портрет покупателей', 'Возраст рассчитан по доступному текущему профилю на дату расчёта, '
     'а не на момент совершения покупки. Неопределённые пол и возраст включают покупателей без профиля.'),
    ('Возвраты и денежные операции', 'Возвратные позиции не создают покупки и не вычитаются автоматически. '
     'Сумма принятых товарных позиций не является чистой выручкой или подтверждением денежного поступления. '
     'Счётчик возвратных позиций относится только к адаптированным строкам.'),
)


CUSTOMER_GROUP_ORDER = ("overview", "purchases", "money", "rfm", "cohorts")


def buyer_sections(result, *, compact=False):
    """Keep the existing ordered section contract for XLSX/HTML callers."""
    return [section for _, section in grouped_buyer_sections(result, compact=compact)]


def grouped_buyer_sections(result, *, compact=False):
    """Return (stable group ID, section) pairs in the original section order.

    Each section is (title, KPI pairs, [(headers, rows)], note pairs).
    Decimal/date values remain typed. No analytical formulas or source reads.
    Group IDs are assigned here; consumers need not inspect visible headings.
    """
    from Application.statistics_copurchases_presentation import copurchase_sections
    saved = result['buyer_statistics'] if 'buyer_statistics' in result else None
    heading = 'Покупательская аналитика'
    if saved is None:
        return [('purchases', (heading, [], [(['Показатель', 'Значение'], [('Новые показатели покупателей', BUYER_LEGACY_TEXT)])], []))] + _buyer_demographic_sections(None, None) + [
            ('purchases',section) for section in copurchase_sections(None,legacy=True)]
    from Application.statistics_purchase_policy import STRICT_COMPLETED, EXTENDED_COMPLETED
    strict, extended = (saved['policies'][p] for p in (STRICT_COMPLETED, EXTENDED_COMPLETED))
    sections = []

    def add(group, section):
        # Tag at construction; never infer routing from a translated heading.
        sections.append((group, section))

    shown = lambda value: 'Нет данных' if value is None else value
    money = lambda value: 'Нет данных' if value is None else Decimal(value)
    stamp = lambda value: 'Нет данных' if value is None else date.fromisoformat(value)
    if strict is not None:
        add('purchases', ('Покупатели — подтверждённые покупки', [
            ('Покупатели (подтверждённые покупки)', strict['buyers']), ('Заказы (подтверждённые покупки)', strict['orders']),
            ('Повторные покупатели (подтверждённые покупки)', strict['repeat_buyers']), ('Доля повторных покупателей, % (подтверждённые покупки)', shown(strict['repeat_rate'])),
            ('Среднее число заказов (подтверждённые покупки)', shown(strict['mean_orders'])), ('Медиана заказов (подтверждённые покупки)', shown(strict['median_orders']))],
            [(['Показатель', 'Значение'], [
                ('Принятые позиции', strict['positions']), ('Единицы товаров', money(strict['units'])),
                ('Покупатели с одним заказом', strict['single_buyers']), ('Среднее единиц на заказ', money(strict['mean_units'])),
                ('Первая наблюдаемая дата', stamp(strict['first_date'])), ('Последняя наблюдаемая дата', stamp(strict['last_date']))])], []))
        intervals = strict['intervals']
        add('purchases', ('Частота и интервалы подтверждённых покупок', [
            ('Средний интервал, дней', shown(intervals['mean_days'])), ('Медианный интервал, дней', shown(intervals['median_days'])),
            ('Покупатели с интервалами', intervals['buyers']), ('Наблюдаемые интервалы', intervals['count'])], [
            (['Заказов на покупателя', 'Покупателей', 'Доля покупателей, %', 'Заказов'], strict['frequency']),
            (['Интервал, дней', 'Интервалов', 'Доля интервалов, %'], intervals['distribution']),
            (['Месяц первой наблюдаемой покупки', 'Покупателей', 'Доля, %'], strict['first_months']),
            (['Месяц последней наблюдаемой покупки', 'Покупателей', 'Доля, %'], strict['last_months'])], []))
        for currency in ('RUB', 'KZT'):
            data = strict['money'][currency]
            add('money', (f'Денежные показатели подтверждённых покупок — {currency}', [
                (f'Сумма принятых товарных позиций, {currency} (известная часть)', money(data['amount'])),
                (f'Средний чек, {currency}', money(data['mean_order'])),
                (f'Медианный чек, {currency}', money(data['median_order'])),
                (f'Средняя полная сумма на покупателя, {currency}', money(data['mean_complete_per_buyer'])),
                (f'Медианная полная сумма на покупателя, {currency}', money(data['median_complete_per_buyer'])),
                (f'Доля повторных покупателей в известной сумме {currency}, %', shown(data['repeat_amount_rate']))], [
                (['Денежный показатель '+currency, 'Значение'], [
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
                (['Год', 'Покрытие года', 'Заказов с суммой', 'Покупателей с суммами',
                  'Сумма позиций, '+currency, 'Покупателей полной оценки года', 'Полная сумма года, '+currency,
                  'Средняя полная сумма года, '+currency, 'Медианная полная сумма года, '+currency],
                 [(row['year'], ('Полный год' if row['complete_year'] else 'Неполный год') + ': ' +
                   '; '.join(interval_text(a, b) for a, b in row['coverage']),
                   row['money'][currency]['known_orders'], row['money'][currency]['known_buyers'], money(row['money'][currency]['amount']),
                   row['money'][currency]['complete_buyers'], money(row['money'][currency]['complete_amount']),
                   money(row['money'][currency]['mean_complete_per_buyer']), money(row['money'][currency]['median_complete_per_buyer']))
                  for row in strict['annual']])], []))
        demo = strict['demography']
        demo_cards = [(label, 'Нет данных' if demo is None else shown(demo[key])) for label, key in (
            ('Покупатели с профилем', 'profiled_buyers'), ('Покупатели без профиля', 'missing_profiles'),
            ('Неопределённый пол покупателей', 'unknown_gender'), ('Неопределённый возраст покупателей', 'unknown_age'),
            ('Средний возраст покупателей', 'mean_age'), ('Медианный возраст покупателей', 'median_age'))]
        add('overview', ('Портрет покупателей — подтверждённые покупки', demo_cards, [
            (['Пол покупателей', 'Покупателей', 'Доля, %'], demo['gender'] if demo else [('Нет данных', 'Нет данных', 'Нет данных')]),
            (['Возраст покупателей', 'Покупателей', 'Доля, %'], demo['age'] if demo else [('Нет данных', 'Нет данных', 'Нет данных')])], []))
        sections.extend(_buyer_demographic_sections(saved, strict))
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
        add('purchases', ('Сравнение политик учёта покупок', [], [(['Показатель', 'Подтверждённые покупки', 'Расширенный учёт покупок'], comparison)], []))
    else:
        message = 'Заказы недоступны' if not saved['orders_available'] else 'Нет покрытия заказов выбранного периода'
        add('purchases', (heading, [], [(['Показатель', 'Значение'], [('Новые показатели покупателей', message)])], []))
    if strict is None:
        sections.extend(_buyer_demographic_sections(saved, strict))
    sections.extend(('purchases',section) for section in copurchase_sections(strict))
    if not compact:
        quality = []
        if strict is not None:
            for label, key in (
                ('Заказы без достоверной денежной оценки', 'unvalued_orders'), ('Неопределённая валюта', 'unknown_currency_orders'),
                ('Смешанная валюта', 'mixed_currency_orders'), ('Некорректная цена', 'invalid_price_orders'),
                ('Покупатели с полностью оценёнными покупками', 'complete_buyers'), ('Покупатели с частичной оценкой', 'partial_buyers'),
                ('Покупатели без достоверной оценки', 'unvalued_buyers')):
                quality.append((label, strict['quality'][key], extended['quality'][key]))
        periods = [interval_bounds(a, b) for a, b in saved['coverage']]
        context = ('Вся доступная история заказов' if saved['period']['start_date'] is None else
                   'Выбранный период: '+saved['period']['start_date']+' — '+saved['period']['end_date'])
        notes = [('Период покупателей', context),
                 ('Покрытие', 'Покрытие непрерывно и полностью соответствует выбранной области' if saved['coverage_complete'] else
                  'Покрытие неполное, содержит разрывы или отсутствует; ненаблюдаемые периоды не равны отсутствию покупок'),
                 ('Дата оценки возраста', saved['as_of']), ('Доступность профилей', 'Доступны' if saved['profiles_available'] else 'Нет данных'),
                 ('Возвратные адаптированные позиции', shown(saved['return_positions'])), *BUYER_NOTES]
        add('purchases', ('Качество и ограничения покупательской аналитики', [], [
            (['Начало покрытия заказов', 'Конец покрытия заказов'], periods),
            (['Качество денежной оценки', 'Подтверждённые покупки', 'Расширенный учёт покупок'], quality)], notes))
    return [(group, (title, [(label, shown(value)) for label, value in cards],
             [(headers, [tuple(shown(value) for value in row) for row in rows]) for headers, rows in tables], notes))
            for group, (title, cards, tables, notes) in sections]


BUSINESS_LEGACY_TEXT = "Нет данных в сохраненном результате. Пересчитайте статистику для получения новых показателей"


def _buyer_demographic_sections(saved, strict):
    """Project saved group aggregates only; currency-free unknown orders stay explicit."""
    if saved is None or strict is not None and "demographic_analytics" not in strict:
        analytics = None
        unavailable = BUSINESS_LEGACY_TEXT
    elif strict is None:
        analytics = None
        unavailable = "Заказы недоступны в выбранной области наблюдения"
    elif strict["demographic_analytics"] is None:
        analytics = None
        unavailable = "Клиентские профили недоступны. Демографическая аналитика не рассчитана"
    else:
        analytics = strict["demographic_analytics"]
        unavailable = None
    shown = lambda value: "Нет данных" if value is None else value
    money = lambda value: "Нет данных" if value is None else Decimal(value)
    def table(headers, rows):
        return [(headers, rows)] if analytics is not None else [
            (["Состояние", "Значение"], [("Демографическая аналитика", unavailable)])]
    sections = []
    for dimension, label, title in (
            ("gender", "Пол", "Покупательская активность по полу"),
            ("age", "Возрастная группа", "Покупательская активность по возрасту")):
        rows = [] if analytics is None else [(g["label"], g["buyers"], shown(g["buyer_rate"]), g["orders"],
                shown(g["mean_orders"]), g["single_buyers"], g["repeat_buyers"], shown(g["repeat_rate"]))
                for g in analytics[dimension]]
        notes = [("Возрастная группа", "Определяется по доступному профилю на дату расчета, "
                  "а не по возрасту клиента в момент исторического заказа.")] if dimension == "age" else []
        sections.append(("overview", (title, [], table(
            [label, "Покупателей", "Доля покупателей, %", "Заказов", "Среднее заказов",
             "С одним заказом", "Повторных", "Доля повторных, %"], rows), notes)))
    sections.append(("money", ("Денежные показатели покупателей по демографии", [], [], [
        ("Суммы", "Известная сумма принятых товарных позиций по priceOfLine; доставка не включена. "
         "Это не подтвержденная оплата и не чистая выручка."),
        ("Полнота", "Заказы без достоверной оценки показаны по всей группе, во всех валютах. "
         "Они не распределяются в RUB/KZT и не входят в средний и медианный чек. "
         "Показатели относятся к наблюдаемому покрытию выбранного периода.")])))
    for dimension, label in (("gender", "полу"), ("age", "возрасту")):
        for currency in ("RUB", "KZT"):
            rows = []
            if analytics is not None:
                for g in analytics[dimension]:
                    m = g["money"][currency]
                    rows.append((g["label"], m["known_buyers"], m["known_orders"], g["unvalued_orders"],
                                 money(m["amount"]), money(m["mean_order"]), money(m["median_order"]), shown(m["amount_rate"])))
            headers = ["Группа", "Покупателей с известной суммой", "Заказов с суммой",
                       "Заказы без достоверной денежной оценки (все валюты)",
                       "Известная сумма принятых товарных позиций, "+currency,
                       "Средний чек, "+currency, "Медианный чек, "+currency, "Доля известной суммы, %"]
            sections.append(("money", (f"По {label} — {currency}", [], table(headers, rows), [])))
    return sections
