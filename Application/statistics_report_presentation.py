"""One Qt-free presentation of the approved statistics interface.

Only saved aggregates enter this model. The three renderers share block order,
typed cells, stable groups, and every chart projection; no source IO occurs here.
"""
from dataclasses import dataclass, field as dataclass_field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from Application import statistics_chart_data as chart_data
from Application.statistics_chart_data import exact_number as _number
from Application.statistics_chart_data import (
    buyer_frequency_chart_data, purchase_policy_comparison_chart_data,
    buyer_turnover_chart_data, cohort_repeat_chart_data,
    rfm_segments_chart_data, cohort_retention_heatmap_data, RFM_WINDOWS,
)
from Application.statistics_presentation import (
    SOURCE_EVENT_QUALITY_HEADING, source_event_quality_rows,
    CATEGORY_HEADERS, DIRECT_CATEGORY_NOTE, product_category_rows, customer_scope_sections,
    PROFILE_SLICE_LABEL, ITEM_CLIENTS_LABEL, CUSTOMER_GROUP_ORDER, grouped_buyer_sections,
    interval_text, status_label, SOURCE_LABELS,
    WARNING_LABELS, _date, _calculation_message,
)
from Application.statistics_rfm_presentation import rfm_report_sections
from Application.statistics_cohorts_presentation import cohort_report_sections, cohort_display, REPEAT_TITLE, RETENTION_TITLE
from Application.analysis_filter import AnalysisFilter

PAGE_LABELS = ("Действия", "Заказы", "Товары", "Клиенты")
CUSTOMER_TAB_LABELS = ("Обзор", "Покупки", "Денежные показатели", "RFM-анализ", "Когорты")

@dataclass(frozen=True)
class ReportNumber:
    value: Decimal
    money: bool = True
    half_up: bool = False
    places: int | None = None

@dataclass(frozen=True)
class ReportMeasure:
    value: object
    known: object
    quality: str
    reasons: tuple[str, ...] = ()
    tooltip: str = ""


def number_text(value):
    if value.places is not None:
        return format(value.value, f",.{value.places}f").replace(",", " ").replace(".", ",")
    number=value.value
    if value.half_up:
        number=number.quantize(Decimal(".01"),rounding=ROUND_HALF_UP)
    text=format(number,",.2f" if value.money else ",f").replace(","," ")
    return text if value.money or "." not in text else text.rstrip("0").rstrip(".")

@dataclass
class ReportBlock:
    kind: str
    title: str = ""
    headers: tuple = ()
    rows: tuple = ()
    values: tuple = ()
    data: object = None
    choices: tuple = ()
    analytics: bool = False
    options: dict = dataclass_field(default_factory=dict)
    table_id: str = ""

class _Blocks(list):
    def addWidget(self, block):
        self.append(block)

@dataclass(frozen=True)
class ReportGroup:
    key: str
    title: str
    blocks: tuple

@dataclass(frozen=True)
class ReportPage:
    key: str
    title: str
    groups: tuple

@dataclass(frozen=True)
class StatisticsReport:
    header: tuple
    global_cards: tuple
    status: str
    pages: tuple
    metadata: dict = dataclass_field(default_factory=dict)
    summary: str = ""

def _section_heading(layout, title):
    layout.append(ReportBlock("heading",title=title))
    return layout

def _cards(layout, values):
    layout.append(ReportBlock("cards",values=tuple(values)))

def _label(text):
    return ReportBlock("text",title=text)

def _decimal_display(value, *, money=True):
    return ReportNumber(Decimal(value),money)

def _buyer_display(value):
    return ReportNumber(value) if isinstance(value,Decimal) else value

def _present_value(value, display):
    if isinstance(value,date):
        return value
    if display is cohort_display and isinstance(value,(float,Decimal)):
        return ReportNumber(Decimal(str(value)),half_up=True)
    return display(value) if display else value

def _table(layout, headers, rows, *, display=None, table_id=""):
    block=ReportBlock("table",table_id=table_id,headers=tuple(headers),rows=tuple(
        tuple(_present_value(v,display) for v in row) for row in rows))
    layout.append(block)
    return block

def _chart(data):
    return ReportBlock("chart",data=data)

def analytics_chart(data):
    return ReportBlock("chart",data=data,analytics=True)

def RfmSegmentsWidget(result):
    return ReportBlock("rfm",choices=tuple((window,currency,rfm_segments_chart_data(result,window,currency))
        for window in RFM_WINDOWS for currency in ("RUB","KZT")))

def CohortRetentionHeatmap(result):
    return ReportBlock("heatmap",data=cohort_retention_heatmap_data(result))

def _analytics_sections(layout, sections, display=_buyer_display, before_tables=None, context=None):
    for section_index,(title,cards,tables,notes) in enumerate(sections):
        section=_section_heading(layout,title)
        if section_index == 0 and context:
            _context(section,context)
        if cards:
            _cards(section,[(label,_present_value(value,display)) for label,value in cards])
        for table_index,(headers,rows) in enumerate(tables):
            for factory in (before_tables or {}).get((section_index,table_index),()):
                section.addWidget(factory())
            _table(section,headers,rows,display=display)
        if notes:
            section.addWidget(_label("\n".join(f"{label}: {text}" for label,text in notes)))


def _context(layout, text):
    layout.append(ReportBlock("text",title=text,options={"context":True}))


def _period_context(period):
    if period is None:
        return "Период не сохранён в результате"
    if period.get("start_date") is None:
        return "Период: доступное покрытие источников"
    return "Период: " + " — ".join(date.fromisoformat(period[k]).strftime("%d.%m.%Y")
                                  for k in ("start_date","end_date"))


def _buyer_context(result, *, comparison=False):
    saved=result.get("buyer_statistics")
    policy=("Подтверждённые покупки F/sold и расширенный учёт F/sold/CP"
            if comparison else "Подтверждённые покупки F/sold")
    text="База: " + policy + " • Все товары • " + _period_context(saved["period"] if saved else None)
    if saved is not None and not saved["coverage_complete"]:
        text+=" • Покрытие заказов неполное"
    return text


def _item_context(result):
    selection=AnalysisFilter.from_dict(result["analysis_filter"])
    products="Все товары" if not selection.product_restricted else "Отобранные товары"
    return ("База: товарный срез • " + products + " • " + _period_context(result["analysis_filter"])
            + " • Правило покупок товарной статистики сохранённого расчёта")


def _overview_buyer_sections(result, sections):
    """Drop a narrow display only when its saved values occur exactly in the richer table."""
    saved=result.get("buyer_statistics")
    strict=saved["policies"]["STRICT_COMPLETED"] if saved else None
    demo=strict.get("demography") if strict else None
    analytics=strict.get("demographic_analytics") if strict else None
    redundant=set()
    if demo is not None and analytics is not None:
        for dimension,header in (("gender","Пол покупателей"),("age","Возраст покупателей")):
            expected=[(g["label"],g["buyers"],g["buyer_rate"]) for g in analytics[dimension]]
            if [tuple(row) for row in demo[dimension]]==expected:
                redundant.add(header)
    return [(title,cards,[(h,rows) for h,rows in tables if h[0] not in redundant],notes)
            for title,cards,tables,notes in sections]


def _customer_overview_page(layout, result, sections):
    for index,(title,rows) in enumerate(customer_scope_sections(result)):
        section=_section_heading(layout,title)
        if index==0:
            scope=result.get("customer_scope")
            context=("База: клиентская база • Идентифицированные клиенты доступных источников • "
                     +_period_context(scope["period"] if scope else None)+" • Все товары, включая события без товара")
            if scope is not None and not all(scope[k] for k in ("profiles_available","actions_available","orders_available")):
                context+=" • Источники доступны не полностью"
            _context(section,context)
        _table(section,['Показатель','Значение'],rows)
    section=_section_heading(layout,'Демография клиентов')
    _context(section,"База: профили клиентов соответствующего среза • Не только подтверждённые покупатели • "
             "Возраст на дату оценки сохранённого расчёта")
    if result['customers'] is None:
        section.addWidget(_label('Данные профилей клиентов отсутствуют. Пол и возраст недоступны.'))
    section=_section_heading(layout,'Возрастная структура клиентов')
    _cards(section,[('Средний возраст',result['mean_age']),('Медианный возраст',result['median_age'])])
    section.addWidget(_chart(chart_data.distribution_data(result,'age')))
    _table(section,['Возрастная группа','Количество клиентов','Доля, %'],result['age_distribution'])
    section=_section_heading(layout,'Распределение по полу')
    section.addWidget(_chart(chart_data.gender_chart_data(result)))
    _table(section,['Пол','Количество клиентов','Доля, %'],result['gender_distribution'])
    section=_section_heading(layout,'Активность клиентов товарного среза')
    _context(section,_item_context(result))
    # customers is already in the shared header; purchase_users remains in the activity table.
    _cards(section,[(ITEM_CLIENTS_LABEL,result['interaction_users']),
                    ('Повторные покупатели (от 2 покупок)',result['repeat_buyers'])])
    _table(section,['Показатель','Значение'],[
        ('Клиенты с просмотрами',result['view_users']),('Клиенты с добавлениями в избранное',result['favorite_users']),
        ('Клиенты с покупками',result['purchase_users']),('Клиенты с просмотрами и покупками',result['view_purchase_users']),
        ('Клиенты с избранным и покупками',result['favorite_purchase_users']),
        ('Клиенты со всеми типами взаимодействий',result['all_interaction_type_users']),
        ('Среднее число взаимодействий на активного клиента',result['mean_interactions']),
        ('Медиана числа взаимодействий на активного клиента',result['median_interactions']),
        ('Доля активных клиентов, совершивших покупку, %',result['active_buyer_rate'])])
    section=_section_heading(layout,chart_data.interaction_frequency_data(result).title)
    section.addWidget(_chart(chart_data.interaction_frequency_data(result)))
    _table(section,['Количество взаимодействий','Клиентов','Доля, %'],result['interaction_activity_distribution'])
    _analytics_sections(layout,_overview_buyer_sections(result,sections),context=_buyer_context(result))


def _customer_purchases_page(layout, result, sections):
    from Application.statistics_copurchases_presentation import copurchase_sections
    saved=result.get("buyer_statistics")
    strict=saved["policies"]["STRICT_COMPLETED"] if saved else None
    comparison=None
    if strict is not None:
        # These two section schemas are the existing grouped_buyer_sections contract.
        main,frequency=sections[:2]
        _analytics_sections(layout,[main],context=_buyer_context(result))
        _,interval_cards,tables,notes=frequency
        _analytics_sections(layout,[('Частота подтверждённых покупок',[],tables[:1],[])],
            before_tables={(0,0):(lambda:analytics_chart(buyer_frequency_chart_data(result)),)})
        _analytics_sections(layout,[('Интервалы между подтверждёнными покупками',interval_cards,tables[1:2],notes)])
        _analytics_sections(layout,[('Первые и последние наблюдаемые покупки периода',[],tables[2:],[
            ('Наблюдение','Первая и последняя покупка внутри показанного периода; это не первая историческая покупка из когорт.')])])
        comparison=next(section for section in sections if any(
            tuple(headers)==('Показатель','Подтверждённые покупки','Расширенный учёт покупок')
            for headers,_ in section[2]))
    else:
        _analytics_sections(layout,sections[:1],context=_buyer_context(result))
    _analytics_sections(layout,copurchase_sections(strict,legacy=saved is None,combined=True))
    section=_section_heading(layout,'Покупательская активность отобранных товаров')
    _context(section,_item_context(result))
    _table(section,['Количество заказов','Покупателей','Доля покупателей, %'],result['purchase_order_distribution'])
    _table(section,['Показатель','Значение'],[
        ('Среднее число заказов на покупателя',result['mean_orders_per_buyer']),
        ('Медиана числа заказов на покупателя',result['median_orders_per_buyer']),
        ('Доля повторных покупателей, %',result['repeat_buyer_rate'])])
    if comparison is not None:
        _,cards,tables,notes=comparison
        _analytics_sections(layout,[('Как изменяются показатели при расширенном учёте покупок',cards,tables,notes)],
            before_tables={(0,0):(lambda:analytics_chart(purchase_policy_comparison_chart_data(result)),)},
            context=_buyer_context(result,comparison=True))

def _money_demographic_tables(result, sections):
    """Remove only the shared unvalued-order column sourced from the same DTO group."""
    saved=result.get("buyer_statistics")
    strict=saved["policies"]["STRICT_COMPLETED"] if saved else None
    analytics=strict.get("demographic_analytics") if strict else None
    demographic=[(headers,list(rows)) for _,_,tables,_ in sections[-4:] for headers,rows in tables]
    common=[]
    if analytics is not None:
        for dimension,label,positions in (("gender","Пол",(0,1)),("age","Возраст",(2,3))):
            expected=[(g["label"],g["unvalued_orders"]) for g in analytics[dimension]]
            if all(len(demographic[i][0])==8 and [(row[0],row[3]) for row in demographic[i][1]]==expected
                   for i in positions):
                common.extend((label,name,count) for name,count in expected)
                for i in positions:
                    headers,rows=demographic[i]
                    demographic[i]=(headers[:3]+headers[4:],[row[:3]+row[4:] for row in rows])
    return demographic,common


def _customer_money_page(layout, result, sections):
    # The provider's contract is two currency sections (when available),
    # one shared demographic note and four demographic tables. Do not route by translated titles.
    available=(len(sections)==7 and all(len(section[2])==3 for section in sections[:2]))
    standard=len(sections) in (5,7) and all(len(section[2])==1 for section in sections[-4:])
    if not standard:
        if sections:
            _analytics_sections(layout,sections)
        else:
            layout.addWidget(_label('Нет денежных показателей в сохранённом результате.'))
        return
    section=_section_heading(layout,'Денежные показатели покупателей')
    _context(section,_buyer_context(result)+" • RUB и KZT отдельно, без суммирования")
    section.addWidget(_label(
        'Известная сумма покупок — сумма достоверно оценённых принятых позиций. '
        'Полная денежная оценка — только покупатели с достоверной оценкой всех наблюдаемых покупок периода '
        'по действующим правилам; известный подытог не заменяет полную оценку.'))
    # Preserve the existing demographic money/coverage limitations once, for both currencies.
    note=sections[2] if available else sections[0]
    if note[3]:
        section.addWidget(_label("\n".join(f"{label}: {text}" for label,text in note[3])))
    demographic,common=_money_demographic_tables(result,sections)
    if common:
        section.addWidget(_label('Общие счётчики заказов без денежной оценки приведены один раз после валютных групп. '
                                'Они относятся ко всем валютам и не складываются между группировками пола и возраста.'))
    for index,currency in enumerate(("RUB","KZT")):
        section=_section_heading(layout,'Денежные показатели — '+currency)
        if available:
            _,cards,tables,notes=sections[index]
            _cards(section,[(label,_present_value(value,_buyer_display)) for label,value in cards])
            _analytics_sections(layout,[('Полнота денежной оценки — '+currency,[],tables[:1],[])])
            _analytics_sections(layout,[('Распределение полной денежной ценности — '+currency,[],tables[1:2],[])],
                before_tables={(0,0):(lambda unit=currency:analytics_chart(buyer_turnover_chart_data(result,unit)),)})
            _analytics_sections(layout,[('Годовая денежная аналитика — '+currency,[],tables[2:],notes)])
        else:
            section.addWidget(_label('Денежные итоги недоступны в сохранённом результате.'))
        for dimension,title,offset in (("gender","По полу",0),("age","По возрасту",2)):
            _analytics_sections(layout,[(title+' — '+currency,[],[demographic[offset+index]],[])])
    if common:
        section=_section_heading(layout,'Общие показатели денежной оценки — все валюты')
        _table(section,['Группировка','Группа','Заказы без достоверной денежной оценки (все валюты)'],common)

def _customer_rfm_page(layout, result):
    for title, cards, tables, notes in rfm_report_sections(result):
        context = ' '.join(text for label, text in notes if label == 'Контекст')
        notes = [(label, text) for label, text in notes if label != 'Контекст']
        chart = title == 'RFM-сегменты покупателей' or (result.get('rfm_statistics') is None and title == 'RFM-анализ покупателей')
        _analytics_sections(layout, [(title, cards, tables, notes)], context=context,
            before_tables={(0, 0): (lambda: RfmSegmentsWidget(result),)} if chart else None)

def _customer_cohorts_page(layout, result):
    for title, cards, tables, notes in cohort_report_sections(result):
        section = _section_heading(layout, title)
        context = ' '.join(text for label, text in notes if label == 'Контекст')
        if context:
            _context(section, context)
        if cards:
            _cards(section, [(label, _present_value(value, cohort_display)) for label, value in cards])
        if title == REPEAT_TITLE:
            section.addWidget(analytics_chart(cohort_repeat_chart_data(result)))
        elif title == RETENTION_TITLE:
            section.addWidget(CohortRetentionHeatmap(result))
        for headers, rows in tables:
            _table(section, headers, rows, display=cohort_display)
        notes = [(label, text) for label, text in notes if label != 'Контекст']
        if notes:
            section.addWidget(_label('\n'.join(f'{label}: {text}' for label, text in notes)))

def _actions_page(layout, result):
    section = layout

    def table(headers, rows):
        return _table(section, headers, rows)
    _cards(section, [('Просмотры товаров', result['view_interactions']), ('Добавления в избранное', result['favorite_interactions']), ('Клиенты с просмотрами', result['view_users']), ('Клиенты с избранным', result['favorite_users'])])
    section = _section_heading(layout, 'Активность клиентов')
    table(['Тип действия', 'Количество взаимодействий', 'Количество клиентов', 'Среднее на клиента', 'Медиана на клиента'], [('Просмотры', result['view_interactions'], result['view_users'], result['mean_views_per_viewer'], result['median_views_per_viewer']), ('Добавления в избранное', result['favorite_interactions'], result['favorite_users'], result['mean_favorites_per_user'], result['median_favorites_per_user'])])
    section = _section_heading(layout, chart_data.activity_distribution_data(result, 'views').title)
    section.addWidget(_chart(chart_data.activity_distribution_data(result, 'views')))
    table(['Количество просмотров', 'Количество клиентов', 'Доля, %'], result['view_user_activity_distribution'])
    section = _section_heading(layout, chart_data.activity_distribution_data(result, 'favorites').title)
    section.addWidget(_chart(chart_data.activity_distribution_data(result, 'favorites')))
    table(['Количество избранного', 'Количество клиентов', 'Доля, %'], result['favorite_user_activity_distribution'])
    section = _section_heading(layout, 'Каналы')
    table(['Канал', 'Просмотры', 'Клиенты с просмотрами', 'Доля просмотров, %', 'Избранное', 'Клиенты с избранным', 'Доля избранного, %'], [row[1:] for row in result['action_channel_statistics']])
    section = _section_heading(layout, 'Динамика действий')
    section.addWidget(_chart(chart_data.monthly_data(result, 'views')))
    section.addWidget(_chart(chart_data.monthly_data(result, 'favorites')))
    table(['Месяц', 'Просмотры', 'Клиенты с просмотрами', 'Избранное', 'Клиенты с избранным'], [(month[5:] + '.' + month[:4], *values) for month, *values in result['action_monthly_dynamics']])
    section = _section_heading(layout, 'Параметры просмотров')
    table(['Доступность', 'Просмотров', 'Доля, %'], result['view_availability_distribution'])
    table(['Валюта', 'Просмотров с ценой', 'Средняя цена', 'Медианная цена'], [(currency, n, _decimal_display(mean), _decimal_display(middle)) for currency, n, mean, middle in result['view_price_statistics']])
    section = _section_heading(layout, SOURCE_EVENT_QUALITY_HEADING)
    table(['Показатель', 'Значение'], tuple((label, 'Нет данных' if value is None else value) for label, value in source_event_quality_rows(result)))

def _products_page(layout, result):
    section = layout

    def table(headers, rows, *, table_id=""):
        return _table(section, headers, rows, table_id=table_id)

    def group(first, field):
        table([first, 'Товаров', 'Просмотры', 'Добавления в избранное', 'Покупки', 'Продано единиц'], [(*row[:5], _decimal_display(row[5], money=False)) for row in result[field]])
    _cards(section, [('Товары с взаимодействиями', result['unique_resolved_items']), ('Товары с просмотрами', result['products_with_views']), ('Товары в избранном', result['products_with_favorites']), ('Купленные товары', result['products_with_purchases'])])
    section = _section_heading(layout, 'Популярные товары по просмотрам')
    section.addWidget(_chart(chart_data.product_ranking_data(result, 'views')))
    table(['Код', 'Название', 'Просмотры', 'Клиенты с просмотрами', 'Добавления в избранное', 'Покупки'], result['top_viewed_products'])
    section = _section_heading(layout, 'Популярные товары по избранному')
    section.addWidget(_chart(chart_data.product_ranking_data(result, 'favorites')))
    table(['Код', 'Название', 'Добавления в избранное', 'Клиенты с избранным', 'Просмотры', 'Покупки'], result['top_favorited_products'])
    section = _section_heading(layout, 'Популярные товары по покупкам')
    section.addWidget(_chart(chart_data.purchased_products_data(result)))
    table(['Код', 'Название', 'Покупки', 'Покупатели', 'Продано единиц', 'Просмотры', 'Добавления в избранное'], [(*row[:4], _decimal_display(row[4], money=False), *row[5:]) for row in result['top_purchased_products']], table_id='products.purchased')
    section = _section_heading(layout, 'Категории товаров')
    section.addWidget(_chart(chart_data.product_category_chart_data(result)))
    section.addWidget(_label(DIRECT_CATEGORY_NOTE))
    categories = table(CATEGORY_HEADERS, [(*row[:7], _decimal_display(row[7], money=False)) for row in product_category_rows(result)], table_id='products.categories')
    categories.options['nowrap'] = True
    categories.options['full_width_columns'] = (2,)
    for title, field in (('Спрос по полу товара', 'product_gender_statistics'), ('Спрос по сезону', 'product_season_statistics'), ('Спрос по стилевой группе', 'product_style_statistics')):
        section = _section_heading(layout, title)
        if field == 'product_season_statistics':
            section.addWidget(_chart(chart_data.season_chart_data(result)))
        group('Значение', field)
    section = _section_heading(layout, 'Качество сопоставления')
    table(['Показатель', 'Значение'], [('Уникальные исходные идентификаторы товаров', result['unique_source_products']), ('Уникальные распознанные товары', result['unique_resolved_items']), ('Распознанные взаимодействия', result['resolved_interactions']), ('Нераспознанные взаимодействия', result['unresolved_interactions']), ('Доля распознанных, %', f"{result['resolution_rate']:.4f}")])
    table(['Система идентификаторов', 'Взаимодействия', 'Распознано', 'Не распознано', 'Доля распознанных, %'], [('Неподдерживаемая' if name == 'unsupported' else name, total, resolved, unresolved, f'{rate:.4f}') for name, total, resolved, unresolved, rate in result['namespaces']])

def _orders_page(layout, result):
    section = layout

    def table(headers, rows, *, table_id=""):
        return _table(section, headers, rows, table_id=table_id)
    _cards(section, [('Заказы с покупкой', result['purchase_orders']), ('Невыкупленные заказы', result['orders_without_purchase']), ('Позиции покупок', result['purchase_interactions']), ('Покупатели', result['purchase_users'])])
    section = _section_heading(layout, 'Корзина заказа')
    basket_rates = {label: rate for label, _, rate in result['purchase_basket_distribution']}
    _cards(section, [('Среднее число позиций на заказ', result['mean_purchase_lines_per_order']), ('Медиана числа позиций на заказ', result['median_purchase_lines_per_order']), ('Доля заказов с 1 позицией', f"{basket_rates['1 позиция']:.2f}%"), ('Доля заказов с 3+ позициями', f"{sum((basket_rates[key] for key in ('3–5 позиций', '6–10 позиций', '11+ позиций'))):.2f}%")])
    section.addWidget(_chart(chart_data.distribution_data(result, 'basket')))
    table(['Количество позиций', 'Заказов', 'Доля, %'], result['purchase_basket_distribution'], table_id='orders.basket')
    section = _section_heading(layout, 'Финансовые показатели')
    table(['Валюта', 'Заказов', 'Позиции', 'Единиц', 'Сумма покупок', 'Средняя сумма заказа', 'Медианная сумма заказа'], [(currency, n, lines, _decimal_display(units, money=False), *(_decimal_display(v) for v in (amount, mean, median))) for currency, n, lines, units, amount, mean, median in result['order_financials']], table_id='orders.financials')
    mixed, unknown = (result['mixed_currency_purchase_orders'], result['unknown_currency_purchase_orders'])
    if mixed or unknown:
        section.addWidget(_label(f'Заказы со смешанной валютой: {_number(mixed)}. С неопределённой валютой: {_number(unknown)}.'))
    for currency in ('RUB', 'KZT'):
        section = _section_heading(layout, chart_data.purchase_count_data(result, currency).title)
        section.addWidget(_chart(chart_data.purchase_count_data(result, currency)))
    for currency in ('RUB', 'KZT'):
        section = _section_heading(layout, 'Динамика суммы покупок — ' + currency)
        section.addWidget(_chart(chart_data.revenue_chart_data(result, currency)))
    table(['Месяц', 'Заказы RUB', 'Сумма RUB', 'Заказы KZT', 'Сумма KZT'], [(month[5:] + '.' + month[:4], rub_n, _decimal_display(rub), kzt_n, _decimal_display(kzt)) for month, rub_n, rub, kzt_n, kzt in result['order_monthly_dynamics']])
    for currency in ('RUB', 'KZT'):
        section = _section_heading(layout, 'Магазины и каналы — ' + currency)
        section.addWidget(_chart(chart_data.store_purchase_chart_data(result, currency)))
        table(['Магазин / канал', 'Заказов', 'Покупателей', 'Позиции', 'Единиц', 'Сумма покупок', 'Средняя сумма заказа'], [(name, n, buyers, lines, _decimal_display(units, money=False), _decimal_display(amount), _decimal_display(mean)) for unit, key, name, n, buyers, lines, units, amount, mean in result['store_statistics'] if unit == currency], table_id='orders.stores.' + currency)
    section = _section_heading(layout, 'Оплата')
    section.addWidget(_chart(chart_data.payment_chart_data(result)))
    table(['Способ оплаты', 'Заказов с этим способом', 'Доля заказов, %'], result['payment_type_distribution'], table_id='orders.payments')
    section = _section_heading(layout, 'Статусы позиций')
    table(['Статус', 'Позиции', 'Доля, %', 'Считается покупкой'], [(status_label(name), count, 100 * count / result['order_lines'] if result['order_lines'] else 0, 'Да' if purchase else 'Нет') for name, count, purchase in result['line_statuses']], table_id='orders.line_statuses')

def _blocks(provider,result,*args):
    blocks=_Blocks()
    provider(blocks,result,*args)
    return tuple(blocks)

def header_rows(result, *, snapshot_diagnostics=None):
    selection=AnalysisFilter.from_dict(result["analysis_filter"])
    rows=[("Период расчёта",_calculation_message(result)),("Дата расчёта",_date(result["calculated_at"])),
          ("Источник","Mindbox"),
          ("Начало периода",date.fromisoformat(selection.start_date) if selection.start_date else "Не ограничено"),
          ("Конец периода",date.fromisoformat(selection.end_date) if selection.end_date else "Не ограничено")]
    for label,values in (("Вид номенклатуры",selection.nomenclature_types),("Сезон",selection.collections)):
        rows.extend((label,"Все значения" if values is None else "Не указано" if v is None else v)
                    for v in ((None,) if values is None else values))
    cutoff=next((result[field]["as_of_date"] for field in ("rfm_statistics","cohort_statistics") if result.get(field)),None)
    if cutoff:
        rows.append(("Дата отсечения покупок и RFM",date.fromisoformat(cutoff)))
    buyer=result.get("buyer_statistics")
    if buyer:
        rows.append(("Дата оценки портрета покупателей в агрегатах",date.fromisoformat(buyer["as_of"])))
    rows.append(("Товарный отбор",
                 "Отбор применяется к товарной части статистики. Покупательская аналитика — по всем товарам."))
    for source in result["coverage"]:
        periods="; ".join(interval_text(a,b) for a,b in source["intervals"])
        if not periods and source["updated"]:
            periods="Снимок данных от "+_date(source["updated"])
        rows.append(("Покрытие: "+SOURCE_LABELS[source["source"]],periods or "—"))
    if snapshot_diagnostics is not None:
        from Application.statistics_detail_job import validate_snapshot_diagnostics
        saved=validate_snapshot_diagnostics(snapshot_diagnostics)
        rows.extend((
            ("Выбор снимков заказов","Первое подходящее вхождение в порядке canonical-источников, частей и строк"),
            ("Область диагностики снимков","История Orders до даты отсечения, все товары"),
            ("Снимков обработано",saved["raw"]),("Снимков принято",saved["accepted"]),
            ("Идентичных повторов снимков",saved["identical"]),
            ("Конфликтующих повторов снимков",saved["conflicting"])))
    return tuple(rows)

def report_presentation(result, *, snapshot_diagnostics=None, applied_filter=None):
    from Application.statistics_unified_dto import UnifiedResult

    if isinstance(result, UnifiedResult):
        from Application.statistics_unified_presentation import unified_presentation

        return unified_presentation(result, applied_filter=applied_filter)
    buyer_groups={key:[] for key in CUSTOMER_GROUP_ORDER}
    for group,section in grouped_buyer_sections(result,compact=True):
        buyer_groups[group].append(section)
    customer_providers=(
        _customer_overview_page,_customer_purchases_page,_customer_money_page,
        _customer_rfm_page,_customer_cohorts_page)
    customer_groups=tuple(ReportGroup(key,title,_blocks(provider,result,
        *([buyer_groups[key]] if key in ("overview","purchases","money") else [])))
        for key,title,provider in zip(CUSTOMER_GROUP_ORDER,CUSTOMER_TAB_LABELS,customer_providers))
    pages=[]
    for key,title,provider in (("actions",PAGE_LABELS[0],_actions_page),("orders",PAGE_LABELS[1],_orders_page),
                               ("products",PAGE_LABELS[2],_products_page)):
        pages.append(ReportPage(key,title,(ReportGroup(key,title,_blocks(provider,result)),)))
    pages.append(ReportPage("customers",PAGE_LABELS[3],customer_groups))
    status="\n".join([_calculation_message(result),
        *(WARNING_LABELS.get(warning,warning) for warning in result["warnings"] if warning not in {
        "Позиции заказов с нецелым количеством исключены из статистики.",
        "Есть конфликтующие снимки заказов: item interactions рассчитаны по первому снимку."})])
    cards=tuple((label,result[key]) for label,key in (
        ("Количество взаимодействий","total_interactions"),("Количество заказов","orders"),
        ("Количество позиций в заказах","order_lines"),(PROFILE_SLICE_LABEL,"customers")))
    return StatisticsReport(header_rows(result,snapshot_diagnostics=snapshot_diagnostics),cards,status,tuple(pages))


@dataclass(frozen=True)
class ReportCell:
    value: object
    text: str
    kind: str
    number_format: str = "@"
    tooltip: str = ""
    known: object = None
    quality: str = ""
    reasons: tuple[str, ...] = ()

    @property
    def displayed_value(self):
        return self.value if self.value is not None else self.known

def cell_presentation(value, *, percentage=False, numeric_hint=False):
    """One display/type boundary; percent values retain the saved 0..100 scale."""
    import re
    from datetime import date
    if isinstance(value, ReportMeasure):
        cell = cell_presentation(value.value, percentage=percentage, numeric_hint=numeric_hint)
        known = value.known.value if isinstance(value.known, ReportNumber) else value.known
        if value.value is not None:
            text = cell.text.replace(".", ",") if isinstance(value.value, ReportNumber) else cell.text
        elif known is not None:
            part = cell_presentation(value.known, percentage=percentage)
            return ReportCell(None, part.text.replace(".", ","), part.kind, part.number_format,
                              value.tooltip, known, value.quality, value.reasons)
        else:
            text = "—"
        return ReportCell(cell.value, text, cell.kind, cell.number_format, value.tooltip, known,
                          value.quality, value.reasons)
    raw = value.value if isinstance(value, ReportNumber) else value
    if isinstance(value, ReportNumber):
        text = number_text(value)
        places = value.places
        if places is None:
            places = 2 if value.money else len(format(raw,"f").partition(".")[2].rstrip("0"))
        fmt = "#,##0" + ("."+"0"*places if places else "")
    elif raw is None:
        return ReportCell(None,"—","empty")
    elif isinstance(raw,date):
        return ReportCell(raw,raw.strftime("%d.%m.%Y"),"date","dd.mm.yyyy")
    elif isinstance(raw,str):
        if (percentage or numeric_hint) and re.fullmatch(r"[+-]?\d[\d ]*(?:[.,]\d+)?\s*%?",raw):
            text=raw.removesuffix("%").rstrip()
            raw=Decimal(text.replace(" ","").replace(",","."))
            places=len(text.replace(",",".").partition(".")[2])
            fmt="#,##0"+("."+"0"*places if places else "")
        else:
            return ReportCell(raw,raw,"text")
    elif isinstance(raw,(int,float,Decimal)) and not isinstance(raw,bool):
        text=(f"{raw:,}".replace(","," ") if isinstance(raw,int)
              else f"{raw:,.2f}".replace(","," "))
        fmt="#,##0" if isinstance(raw,int) else "#,##0.00"
    else:
        return ReportCell(str(raw),str(raw),"text")
    if percentage:
        text=text.replace(".",",")+" %"
        fmt+='" %"'
    return ReportCell(raw,text,"number",fmt)

def percentage_cell(headers, row, column):
    metric_table = headers[0].startswith(("Показатель", "Денежный показатель")) or headers[0] == "Диагностика"
    # Quantile percentages locate boundaries; values retain R/F/M units.
    quantile = headers[column] in ("Порог 20%", "Порог 40%", "Порог 60%", "Порог 80%")
    return not quantile and "%" in headers[column] or column > 0 and metric_table and "%" in str(row[0])

def table_cells(block):
    return tuple(tuple(cell_presentation(value, percentage=percentage_cell(block.headers,row,column))
                       for column,value in enumerate(row)) for row in block.rows)

def kpi_cells(block):
    return tuple((label,cell_presentation(value,percentage="%" in label
        or isinstance(value,str) and value.endswith("%"))) for label,value in block.values)
