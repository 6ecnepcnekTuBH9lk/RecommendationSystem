"""Typed RFM snapshot projection shared by GUI, XLSX and offline HTML."""

from collections import Counter
from datetime import date
from decimal import Decimal

from Application.statistics_rfm_core import POLICIES
from Application.statistics_presentation import interval_bounds


RFM_LEGACY_TEXT = 'Нет данных в сохранённом результате'
WINDOW_LABELS = {'LAST_12M':'12 месяцев', 'LAST_24M':'24 месяца', 'LAST_48M':'48 месяцев',
                 'PREVIOUS_12M':'Предыдущие 12 месяцев'}
SEGMENT_LABELS = {
    'CHAMPIONS': ('Лидеры', 'Относительно недавние, частые и денежно ценные; минимум два заказа.'),
    'LOYAL': ('Лояльные', 'Высокая относительная частота и достаточно недавняя покупка; минимум два заказа.'),
    'NEW_OR_RECENT': ('Недавняя наблюдаемая покупка', 'Одна покупка окна и высокий относительный R; не новый клиент компании.'),
    'AT_RISK': ('Риск снижения активности', 'Давняя последняя покупка при высокой частоте наблюдаемого окна.'),
    'HIBERNATING': ('Давние редкие покупки', 'Низкий относительный R и низкая наблюдаемая частота.'),
    'OTHER': ('Прочие', 'Оценены, остальные правила не выполнены.'),
    'UNSCORABLE': ('Оценка недоступна', 'Неполные деньги, покрытие или недостоверная последняя покупка.'),
}
RFM_NOTES = (
    ('Область', 'Все товары. RFM использует 48 месяцев до отсечения независимо от начала пользовательского периода. '
     'База коротких окон сохраняет наблюдавшихся покупателей за 48 месяцев.'),
    ('Политики', 'Подтверждённые покупки: F/sold — основная; Расширенный учёт покупок: F/sold/CP — сравнение. '
     'Политики и валютные базы не суммируются; покупатель может быть в RUB и KZT.'),
    ('Неполные данные', 'Нули относятся к наблюдаемой истории. При разрывах они не доказывают отсутствие покупок; '
     'окончательные баллы недоступны; недоступная оценка не заменяется нулевым сегментом.'),
    ('R / F / M', 'R — календарные дни с последней наблюдаемой покупки в 48 месяцах; '
     'F — заказы окна; M — сумма принятых товарных позиций. Валютные R/F используют только свою валюту.'),
    ('Деньги', 'Известный подытог может быть неполным. Полная денежная оценка доступна лишь при полном покрытии окна '
     'и оценённых всех заказах политики. Возвраты автоматически не вычитаются; это не чистая выручка.'),
    ('Баллы', 'Пороги 20/40/60/80% — метод ближайшего ранга. Одинаковые значения не разделяются; некоторые баллы могут отсутствовать. '
     'R обращён относительно F/M; неизвестная оценка не равна баллу 1.'),
    ('Ограничение истории', 'Дата оформления и статус доступного снимка не доказывают дату оплаты или передачи товара. '
     'Более ранняя история может существовать. Недавность относительная, новизна за всю историю не установлена.'),
)


def rfm_sections(result, *, compact=False):
    saved = result['rfm_statistics'] if 'rfm_statistics' in result else None
    if saved is None:
        message = (RFM_LEGACY_TEXT if 'rfm_statistics' not in result else
                   'Нет достоверной даты покрытия заказов' if not any(s['intervals'] for s in result['coverage'] if s['source']=='Orders')
                   else RFM_LEGACY_TEXT)
        return [('RFM-анализ покупателей', [], [(['Показатель RFM','Значение'], [('RFM',message)])], [])]
    show = lambda v: 'Нет данных' if v is None else v
    money = lambda v: 'Нет данных' if v is None else Decimal(v)
    primary = saved['policies'][POLICIES[0]]
    windows = tuple(WINDOW_LABELS)[:3]
    coverage_rows = []
    statuses = {'FULL':'Полное','PARTIAL':'Частичное','NONE':'Нет покрытия'}
    for name, meta in saved['windows'].items():
        for a,b in meta['available'] or [('Нет данных','Нет данных')]:
            coverage_rows.append((WINDOW_LABELS[name], *interval_bounds(meta['start'], meta['end']),
                                  statuses[meta['status']],
                                  *(interval_bounds(a, b) if a != 'Нет данных' else (a, b))))
    sections = [('RFM-анализ покупателей', [('Дата отсечения RFM',date.fromisoformat(saved['as_of_date'])),
                                          ('Наблюдаемые покупатели за 48 месяцев (подтверждённые покупки)',primary['base_buyers'])],
        [(['Окно RFM','Начало','Конец','Покрытие','Доступно с','Доступно до'],coverage_rows)],
        list(RFM_NOTES))]
    if compact:
        title, cards, _, _ = sections[0]
        sections[0] = (title, cards, [], [])
    summary, monetary, observations, segments, score_rows, threshold_rows = [], [], [], [], [], []
    for name in windows:
        w = primary['windows'][name]
        b = w['behavior']
        full = saved['windows'][name]['status'] == 'FULL'
        summary.append((WINDOW_LABELS[name],b['buyers'],b['active_buyers'],b['without_observed_orders'],
                        show(b['confirmed_inactive_buyers']),b['observed_orders'],
                        b['buyers']-w['quality']['incomplete_money_buyers'] if full else 'Нет данных',
                        w['quality']['incomplete_money_buyers']))
        for currency, c in w['currencies'].items():
            unavailable = next(n for s,n,_ in c['segments'] if s=='UNSCORABLE')
            monetary.append((WINDOW_LABELS[name],currency,c['buyers'],c['valued_orders'],money(c['known_amount']),
                             c['monetary']['buyers'],c['unavailable_monetary_buyers'],c['buyers']-unavailable,unavailable))
            observations.append((WINDOW_LABELS[name],currency,show(c['recency']['mean']),show(c['recency']['median']),
                                 show(c['frequency']['mean']),show(c['frequency']['median']),
                                 money(c['monetary']['mean']),money(c['monetary']['median']),c['reliable_recency_buyers']))
            for segment, count, share in c['segments']:
                label, explanation = SEGMENT_LABELS[segment]
                segments.append((WINDOW_LABELS[name],currency,label,count,show(share),explanation))
            for i, component in enumerate(('R','F','M')):
                scores = Counter()
                for row in c['score_counts']:
                    scores[row[i]] += row[3]
                score_rows.append((WINDOW_LABELS[name],currency,component,*(scores[k] for k in range(1,6)),scores[None]))
                threshold = c['thresholds'][component]
                values = [money(v) if component=='M' else v for v in threshold['values']]
                threshold_rows.append((WINDOW_LABELS[name],currency,component,threshold['buyers'],*(values or ['Нет данных']*4)))
    sections.append(('RFM — основные показатели: подтверждённые покупки', [], [
        (['Окно','База покупателей 48 месяцев','С покупками окна','Без наблюдаемых покупок окна','Подтверждённое отсутствие покупок',
          'Наблюдаемые заказы','Полная денежная оценка базы','Покупатели с неполными деньгами'],summary),
        (['Окно','Валюта','Валютная база 48 месяцев','Заказы с достоверной суммой','Известная сумма позиций',
          'Полная денежная оценка: покупателей','Денежная оценка недоступна: покупателей','Сегмент определён','Оценка недоступна'],monetary),
        (['Окно','Валюта','Средний R, дней','Медианный R, дней','Средний F','Медианный F','Средняя полная M',
          'Медианная полная M','Достоверный R: покупателей'],observations)], []))
    sections.append(('RFM — сегменты: подтверждённые покупки', [], [
        (['Окно','Валюта','Сегмент','Покупателей','Доля валютной базы, %','Определение'],segments)], []))
    sections.append(('RFM — баллы и пороги: подтверждённые покупки', [], [
        (['Окно','Валюта','Компонент','Балл 1','Балл 2','Балл 3','Балл 4','Балл 5','Оценка недоступна'],score_rows),
        (['Окно','Валюта','Компонент','Основание порогов: покупателей','Порог 20%','Порог 40%','Порог 60%','Порог 80%'],threshold_rows)], []))
    c = primary['comparison']
    group_labels = {'BOTH':'Покупали в обоих интервалах','LAST_ONLY':'Покупали только в последних 12 месяцах',
                    'PREVIOUS_ONLY':'Покупали только в предыдущих 12 месяцах','NEITHER':'Не покупали в обоих; наблюдались в 48 месяцах'}
    sections.append(('RFM — последние и предыдущие 12 месяцев: подтверждённые покупки', [], [
        (['Наблюдаемая группа активности','Покупателей'],[(group_labels[k],n) for k,n in c['observed_groups'].items()]),
        (['Показатель сравнения','Значение'],[
            ('Покрытие обоих интервалов полное','Да' if c['coverage_reliable'] else 'Нет — сравнение неполное'),
            ('Наблюдаемые заказы последних 12 месяцев',c['last_orders']),('Наблюдаемые заказы предыдущих 12 месяцев',c['previous_orders']),
            ('Изменение числа заказов',c['order_change']),('Изменение числа заказов, %',show(c['order_change_percent'])),
            ('Недостоверные денежные сравнения: покупателей',c['unreliable_monetary_buyers'])]),
        (['Валюта','Последние 12 месяцев: известная сумма','Предыдущие 12 месяцев: известная сумма','Изменение известной суммы',
          'Изменение известной суммы, %','Недостоверные денежные сравнения'],
         [(unit,money(m['last_known_amount']),money(m['previous_known_amount']),money(m['known_amount_change']),
           money(m['known_amount_change_percent']),m['unreliable_buyers']) for unit,m in c['money'].items()])], []))
    comparison = []
    for name in windows:
        for p in POLICIES:
            w = saved['policies'][p]['windows'][name]
            currencies = w['currencies']
            scored = [sum(n for s,n,_ in currencies[c]['segments'] if s!='UNSCORABLE') for c in ('RUB','KZT')]
            comparison.append((WINDOW_LABELS[name], 'Подтверждённые покупки' if p==POLICIES[0] else 'Расширенный учёт покупок',w['behavior']['buyers'],
                               w['behavior']['observed_orders'],*scored,money(currencies['RUB']['known_amount']),
                               money(currencies['KZT']['known_amount'])))
    sections.append(('RFM — сравнение политик учёта покупок', [], [
        (['Окно','Политика','База покупателей 48 месяцев','Наблюдаемые заказы','Сегмент определён RUB','Сегмент определён KZT',
          'Известная сумма RUB','Известная сумма KZT'],comparison)], []))
    return sections
