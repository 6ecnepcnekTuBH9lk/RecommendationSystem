"""Saved cohort tables shared by GUI, XLSX and offline HTML; no calculations."""

from datetime import date
from decimal import Decimal, ROUND_HALF_UP, localcontext

from Application.statistics_cohorts_core import HORIZONS, POLICIES
from Application.statistics_cohorts_channels import channel_key
from Application.statistics_cohorts_categories import UNKNOWN_CATEGORY, category_key
from Application.statistics_presentation import interval_bounds


COHORT_LEGACY_TEXT = 'Нет данных в сохранённом результате'
COHORT_TITLE = 'Когортный анализ покупателей'
MATRIX_HEADERS = ('Когорта', 'M0: покупателей', *(f'M{i}, %' for i in range(1,13)))
HORIZON_HEADERS = ('Горизонт, дней', 'Покупатели с полным наблюдением', 'Повторные покупки', 'Повторная доля, %', 'Цензурированы',
                   'Неполное покрытие', 'Пересечение групп', 'Наблюдаемые повторы', 'Повторы вне базы')
COHORT_NOTES = (
    ('Определение', 'Когорта определяется по первой наблюдаемой покупке в доступной истории, '
     'а не обязательно по первой покупке клиента за всё время.'),
    ('Политики',  'Подтверждённые покупки: F + sold — основная; расширенный учёт покупок: F + sold + CP — сравнение. '
     'Дата первой покупки и принадлежность к когорте могут различаться. Политики не суммируются.'),
    ('Область', 'Вся доступная история заказов до отсечения; все товары, без пересечения с действиями. '
     'Выбранный период ограничивает отображаемые месяцы когорт, а не поиск более ранней покупки.'),
    ('Доли', 'Каждый горизонт имеет собственную базу полностью наблюдаемых покупателей. '
     'При отсутствии покупателей с полным наблюдением доля недоступна. Наблюдаемые повторы вне базы не входят в основную долю.'),
    ('Неполнота', 'Цензурирование — горизонт ещё не завершён. Неполное покрытие — пробелы в прошедшей части. '
     'Группы могут пересекаться; пересечение показано отдельно. Они не означают «не вернулись».'),
    ('Матрица', 'M0 — размер когорты; M1–M12 — активность именно в календарном месяце. '
     'Ноль процентов подтверждён только при полном наблюдении. Числитель и знаменатель — в детализации матрицы.'),
    ('Ограничения', 'Пробелы могут скрывать более ранние заказы и фактическую вторую покупку. '
     'Интервал до второй покупки описывает наблюдаемые события, без поправки на цензурирование.'),
)


def cohort_display(value):
    """GUI rounding agrees with HTML/Excel; keep the stored percentage exact."""
    if isinstance(value, (float, Decimal)):
        number = Decimal(str(value))
        with localcontext() as context:
            context.prec = max(len(number.as_tuple().digits), number.adjusted()+1, 1)+2
            rounded = number.quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
        return format(rounded, ',.2f').replace(',', ' ')
    if isinstance(value, date):
        return value.strftime('%d.%m.%Y')
    return 'Нет данных' if value is None else value


def _show(value):
    return 'Нет данных' if value is None else value


def _horizon_row(horizon, count):
    return (horizon, count['eligible'], count['repeated'], _show(count['repeat_rate']), count['censored'],
            count['coverage_incomplete'], count['censored_and_incomplete'], count['observed_repeated'],
            count['observed_repeated_ineligible'])


def _cohort_tables(policy):
    cohort_rows, horizon_rows = [], []
    for month, row in policy['cohorts'].items():
        intervals = row['second_purchase_days']
        cohort_rows.append((month, row['cohort_size'], row['observed_repeat_buyers'], _show(intervals['mean']),
                            _show(intervals['median']), intervals['coverage_incomplete_buyers']))
        for horizon in HORIZONS:
            horizon_rows.append((month, *_horizon_row(horizon,row['horizons'][str(horizon)])))
    return [(['Когорта', 'Покупателей', 'Наблюдаемая вторая покупка', 'Средний интервал, дней',
              'Медианный интервал, дней', 'Интервал с пробелами'], cohort_rows),
            (['Когорта', *HORIZON_HEADERS], horizon_rows)]


def _retention_tables(policy):
    matrix, details = [], []
    for month, row in policy['cohorts'].items():
        cells = row['retention']
        matrix.append((month, cells[0]['buyers'], *(_show(c['retention_rate']) for c in cells[1:])))
        for cell in cells[1:]:
            details.append((month, 'M'+str(cell['month_offset']), cell['month'], cell['active'], cell['eligible'],
                            _show(cell['retention_rate']), cell['censored'], cell['coverage_incomplete'],
                            cell['censored_and_incomplete'], cell['observed_active_ineligible']))
    return [(list(MATRIX_HEADERS),matrix),
            (['Когорта', 'Смещение', 'Календарный месяц', 'Активны в базе', 'Покупатели с полным наблюдением', 'Удержание, %',
              'Цензурированы', 'Неполное покрытие', 'Пересечение групп', 'Активны вне базы'],details)]


MONEY_LEGACY_TEXT = 'Денежные показатели отсутствуют в сохранённом результате. Для их получения пересчитайте статистику.'
MONEY_TITLES = ('Денежные показатели месячных когорт', 'Денежные показатели второй покупки')


def _money_sections(monetary):
    monthly, horizons = [], []
    for currency in ('RUB', 'KZT'):
        if monetary is None:
            placeholder = ([f'Показатели, {currency}', 'Доступность'], [('Денежная аналитика', MONEY_LEGACY_TEXT)])
            monthly.append(placeholder)
            horizons.append(placeholder)
            continue
        rows = []
        for month, row in monetary['cohorts'].items():
            c = row['currencies'][currency]
            rows.append((month, row['cohort_size'], row['orders'], c['known_orders'], Decimal(c['amount']),
                         Decimal(c['mean_order']) if c['mean_order'] is not None else 'Нет данных',
                         Decimal(c['first_amount']), Decimal(c['repeat_amount']),
                         _show(c['repeat_amount_rate']), row['unvalued_orders']))
        monthly.append((['Когорта', 'Покупателей', 'Всего заказов', f'Оценённых заказов, {currency}',
            f'Известная сумма, {currency}', f'Средний чек, {currency}', f'Первые покупки, {currency}',
            f'Повторные покупки, {currency}', 'Доля повторной суммы, %', 'Заказов без оценки (все валюты)'], rows))
        rows = []
        for h in HORIZONS:
            row = monetary['horizons'][str(h)]
            c = row['currencies'][currency]
            rows.append((h, row['eligible'], row['second_buyers'], c['known_orders'], Decimal(c['amount']),
                         Decimal(c['mean_order']) if c['mean_order'] is not None else 'Нет данных',
                         row['unvalued_orders']))
        horizons.append((['Горизонт, дней', 'Покупатели с полным наблюдением', 'Со второй покупкой в базе',
            f'Оценённых вторых заказов, {currency}', f'Известная сумма вторых покупок, {currency}',
            f'Средняя вторая покупка, {currency}', 'Вторых заказов без оценки (все валюты)'], rows))
    common = [('Денежная оценка', 'Только подтверждённые покупки F + sold. Известная сумма принятых товарных позиций '
        'не является подтверждённой чистой выручкой. RUB и KZT не суммируются. '
        'Неоценённые заказы не заменены нулями; средние рассчитаны только по оценённым заказам.'),
        ('Общие счётчики', 'Покупатели, все заказы и заказы без оценки повторяются в таблицах валют; '
         'их нельзя складывать между RUB и KZT. Нулевая достоверная сумма учитывается.')]
    return [(MONEY_TITLES[0], [], monthly, common + [('Область суммы', 'Все заказы покупателей отображаемой когорты до '
        'даты отсечения, включая покупки позднее 365 дней. Повторная сумма включает второй и все последующие заказы.')]),
        (MONEY_TITLES[1], [], horizons, common + [('База горизонта', 'Только покупатели с полностью наблюдаемым горизонтом. '
         'Цензурирование и пробелы исключают покупателя из денежного горизонта. Каждый горизонт имеет свою базу. '
         'Суммируется только второй заказ внутри горизонта, включая покупки в более короткие сроки; третий и последующие исключены.')])]


CHANNEL_TITLE = 'Каналы первой и второй покупки'
CHANNEL_LEGACY_TEXT = 'Нет данных о каналах первой и второй покупки в сохранённом результате. Для их получения пересчитайте статистику.'
CHANNEL_NOTE = ('Канал/магазин определяется по данным первого действия заказа Mindbox. '
                'Первая покупка является первой наблюдаемой, а не гарантированно первой за всю жизнь покупателя.')


def _channel_sections(channels):
    titles = ('Каналы первой покупки','Каналы второй покупки','Переходы между каналами первой и второй покупки')
    if channels is None:
        return [(CHANNEL_TITLE, [], [], [('Методика',CHANNEL_NOTE)])] + [
            (title, [], [(['Показатель','Доступность'], [('Нет данных',CHANNEL_LEGACY_TEXT)])], []) for title in titles]
    catalog = {c['external_id']:c for c in channels['channels']}
    # Display equal names separately; stable IDs, never names, define groups.
    frequencies = {}
    for row in catalog.values():
        frequencies[row['name']] = frequencies.get(row['name'],0)+1
    names = {key:(f"{row['name']} [{key}]" if key is not None and
                 (frequencies[row['name']]>1 or row['name']=='Не указан') else row['name'])
             for key,row in catalog.items()}
    tables = []
    for prefix, count_header, rate_header in (
            ('first','Покупателей','Доля покупателей, %'),('second','Повторных покупателей','Доля повторных, %')):
        rows = sorted(catalog.values(),key=lambda r:(-r[prefix+'_buyers'],channel_key(r['external_id'])))
        tables.append((['Канал/магазин',count_header,rate_header],
                       [(names[r['external_id']],r[prefix+'_buyers'],_show(r[prefix+'_rate'])) for r in rows
                        if r[prefix+'_buyers'] or r['external_id'] is None]))
    tables.append((['Канал первой покупки','Канал второй покупки','Покупателей','Доля от повторных, %',
                    'Доля от повторных первого канала, %'],
        [(names[r['first_external_id']],names[r['second_external_id']],r['buyers'],
          _show(r['repeat_rate']),_show(r['first_channel_repeat_rate'])) for r in channels['transitions']]))
    return [(CHANNEL_TITLE, [
        ('Покупатели со второй наблюдаемой покупкой',channels['repeat_buyers']),
        ('Остались в том же известном канале',channels['same_known_buyers']),
        ('Перешли в другой известный канал',channels['other_known_buyers']),
        ('Переход не определён из-за отсутствия идентификатора',channels['unknown_transition_buyers']),
        ('Доля сохранивших канал среди покупателей с двумя известными идентификаторами, %',_show(channels['same_known_rate']))],
        [], [('Методика',CHANNEL_NOTE),('Доли','Первая покупка: все покупатели отображаемых когорт. '
            'Вторая покупка и переходы: покупатели с наблюдаемой второй покупкой. Доля от повторных первого канала '
            'использует только повторных покупателей, начавших в этом канале. Доля сохранения исключает пары с неизвестным идентификатором.'),
            ('Область','Подтверждённые покупки покупателей отображаемых когорт до даты отсечения. '
            'Учтена наблюдаемая вторая покупка независимо от завершённости горизонта. '
            'При отсутствующем идентификаторе направление показано как «Не указан», без вывода о сохранении или смене канала.'),
            ('Названия','Группы определяются идентификатором канала. Для изменявшихся названий выбрано первое по алфавиту '
             'непустое название среди первых двух заказов рассматриваемых покупателей. Совпадающие названия уточняются идентификатором.')])] + [
        (title, [], [table], []) for title,table in zip(titles,tables)]


CATEGORY_TITLE = 'Категории первой и второй покупки'
CATEGORY_LEGACY_TEXT = 'Нет данных о категориях первой и второй покупки в сохранённом результате. Для их получения пересчитайте статистику.'


def _category_sections(saved):
    titles = ('Категории первой покупки','Категории второй покупки','Переходы категорий первой и второй покупки')
    notes = [('Методика','Только подтверждённые покупки и прямые категории принятых позиций. '
        'Категории определены по снимку каталога при расчёте, без восстановления исторической классификации. '
        'Родительские категории автоматически не добавляются. Путь используется для отображения, а не суммирования.'),
        ('Наблюдаемая повторность','Это наблюдаемая доля, а не вероятность будущей покупки. Молодые когорты подвержены '
         'правому цензурированию и не полностью сопоставимы со старыми. Действующие горизонты повторности не изменены.'),
        ('Несколько категорий','Покупатель учитывается один раз в каждой категории и каждой паре. '
         'Суммы покупателей по категориям и переходам могут превышать число уникальных покупателей, '
         'а сумма долей второй покупки и переходов — 100%. Их нельзя суммировать для получения уникальных покупателей.')]
    if saved is None:
        return [(CATEGORY_TITLE,[],[],notes)] + [
            (title,[],[(['Показатель','Доступность'],[('Нет данных',CATEGORY_LEGACY_TEXT)])],[]) for title in titles]
    catalog = {row['code']:row for row in saved['categories']}
    labels = {code:(' / '.join(row['path']) if row['path'] else row['name']) for code,row in catalog.items()}
    counts = {}
    for label in labels.values():
        counts[label] = counts.get(label,0)+1
    labels = {code:(f'{label} [{code}]' if code is not None and (counts[label]>1 or label==UNKNOWN_CATEGORY) else label)
              for code,label in labels.items()}
    notes.append(('Покрытие',f"Покупателей рассматриваемых когорт: {saved['buyers']}; со второй покупкой: {saved['repeat_buyers']}; "
        f"с неопределёнными категориями первой покупки: {saved['unknown_first_buyers']}; "
        f"с неопределёнными категориями второй покупки: {saved['unknown_second_buyers']}. "
        'Неопределённые категории учитываются также в смешанных заказах с известными категориями.'))
    first = sorted(catalog.values(),key=lambda r:(-r['first_buyers'],category_key(r['code'])))
    second = sorted(catalog.values(),key=lambda r:(-r['second_buyers'],category_key(r['code'])))
    tables = [
        (['Категория','Покупателей','Со второй покупкой','Наблюдаемая доля повторных, %'],
         [(labels[r['code']],r['first_buyers'],r['repeat_buyers'],_show(r['observed_repeat_rate'])) for r in first
          if r['first_buyers'] or r['code'] is None]),
        (['Категория','Покупателей','Доля среди покупателей со второй покупкой, %'],
         [(labels[r['code']],r['second_buyers'],_show(r['second_rate'])) for r in second
          if r['second_buyers'] or r['code'] is None]),
        (['Первая категория','Вторая категория','Покупателей','Доля среди повторных, %'],
         [(labels[r['first_code']],labels[r['second_code']],r['buyers'],_show(r['repeat_rate']))
          for r in saved['transitions']])]
    return [(CATEGORY_TITLE,[],[],notes)] + [(title,[],[table],[]) for title,table in zip(titles,tables)]


def cohort_sections(result, *, include_extended=False, compact=False):
    saved = result['cohort_statistics'] if 'cohort_statistics' in result else None
    if saved is None:
        message = ('Нет достоверной даты отсечения заказов' if 'cohort_statistics' in result else COHORT_LEGACY_TEXT)
        return [(COHORT_TITLE, [], [(['Показатель когорт','Значение'], [('Когорты',message)])], [])] + _money_sections(None) + _channel_sections(None) + _category_sections(None)
    strict = saved['policies'][POLICIES[0]]
    summary = strict['summary']
    coverage = saved['effective_coverage']
    bounds = saved['cohort_bounds']
    sections = [(COHORT_TITLE, [
        ('Дата отсечения когорт',date.fromisoformat(saved['as_of_date'])),
        ('Наблюдаемые когорты',summary['cohort_count']),
        ('Покупатели отображаемых когорт',summary['cohort_size']),
        ('С наблюдаемой второй покупкой',summary['observed_repeat_buyers'])], [
        (['Параметр когорт','Значение'],[
            ('Начало доступной истории заказов',interval_bounds(*coverage[0])[0] if coverage else 'Нет данных'),
            ('Конец наблюдения заказов',interval_bounds(*coverage[-1])[1] if coverage else 'Нет данных'),
            ('Отображаемые когорты с',bounds['start'] or 'Все наблюдаемые'),
            ('Отображаемые когорты по',bounds['end'] or 'Все наблюдаемые'),
            ('Покупатели вне отображаемых когорт',strict['hidden_buyers'])]),
        (['Покрытие заказов с','По'], [interval_bounds(a, b) for a, b in coverage])], list(COHORT_NOTES))]
    if compact:
        title, cards, _, _ = sections[0]
        sections[0] = (title, cards, [], [])
    sections.append(('Когорты — повторные подтверждённые покупки', [], [
        (list(HORIZON_HEADERS),[_horizon_row(h,summary['horizons'][str(h)]) for h in HORIZONS])], []))
    sections.append(('Когорты — месяцы первой наблюдаемой подтверждённой покупки', [], _cohort_tables(strict), []))
    sections.extend(_money_sections(strict.get('monetary')))
    sections.append(('Когорты — матрица удержания M0–M12: подтверждённые покупки', [], _retention_tables(strict), []))
    comparison = []
    for policy in POLICIES:
        label = 'Подтверждённые покупки' if policy == POLICIES[0] else 'Расширенный учёт покупок'
        p = saved['policies'][policy]['summary']
        for h in HORIZONS:
            comparison.append((label,p['cohort_size'],*_horizon_row(h,p['horizons'][str(h)])))
    sections.append(('Когорты — сравнение политик учёта покупок', [], [
        (['Политика','Покупателей в когортах',*HORIZON_HEADERS],comparison)], []))
    if include_extended:
        extended = saved['policies'][POLICIES[1]]
        sections.append(('Когорты — месяцы первой наблюдаемой покупки — расширенный учёт', [], _cohort_tables(extended), []))
        sections.append(('Когорты — матрица удержания M0–M12: расширенный учёт покупок', [], _retention_tables(extended), []))
    sections.extend(_channel_sections(strict.get('purchase_channels')))
    sections.extend(_category_sections(strict.get('purchase_categories')))
    return sections
