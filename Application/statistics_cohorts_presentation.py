"""Saved cohort tables shared by GUI, XLSX and offline HTML; no calculations."""

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from Application.statistics_cohorts_core import HORIZONS, POLICIES


COHORT_LEGACY_TEXT = 'Нет данных в сохранённом результате'
COHORT_TITLE = 'Когортный анализ покупателей'
MATRIX_HEADERS = ('Когорта', 'M0: покупателей', *(f'M{i}, %' for i in range(1,13)))
HORIZON_HEADERS = ('Горизонт, дней', 'Eligible', 'Repeated', 'Повторная доля, %', 'Цензурированы',
                   'Неполное покрытие', 'Пересечение групп', 'Наблюдаемые повторы', 'Повторы вне базы')
COHORT_NOTES = (
    ('Определение', 'Когорта определяется по первой наблюдаемой покупке в доступной истории, '
     'а не обязательно по первой покупке клиента за всё время.'),
    ('Политики', 'STRICT: F + sold — основная; EXTENDED: F + sold + CP — сравнение. '
     'Дата первой покупки и принадлежность к когорте могут различаться. Политики не суммируются.'),
    ('Область', 'Вся доступная история Orders до отсечения; все товары, без пересечения с Actions. '
     'Выбранный период ограничивает отображаемые месяцы когорт, а не поиск более ранней покупки.'),
    ('Доли', 'Каждый горизонт имеет собственную базу полностью наблюдаемых покупателей. '
     'При Eligible = 0 доля недоступна. Наблюдаемые повторы вне базы не входят в основную долю.'),
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
        rounded = Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
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
            (['Когорта', 'Смещение', 'Календарный месяц', 'Активны в базе', 'Eligible', 'Удержание, %',
              'Цензурированы', 'Неполное покрытие', 'Пересечение групп', 'Активны вне базы'],details)]


def cohort_sections(result, *, include_extended=False):
    saved = result['cohort_statistics'] if 'cohort_statistics' in result else None
    if saved is None:
        message = ('Нет достоверной даты отсечения Orders' if 'cohort_statistics' in result else COHORT_LEGACY_TEXT)
        return [(COHORT_TITLE, [], [(['Показатель когорт','Значение'], [('Когорты',message)])], [])]
    strict = saved['policies'][POLICIES[0]]
    summary = strict['summary']
    coverage = saved['effective_coverage']
    bounds = saved['cohort_bounds']
    sections = [(COHORT_TITLE, [
        ('Дата отсечения когорт, UTC',date.fromisoformat(saved['as_of_date'])),
        ('Наблюдаемые когорты STRICT',summary['cohort_count']),
        ('Покупатели отображаемых когорт STRICT',summary['cohort_size']),
        ('С наблюдаемой второй покупкой STRICT',summary['observed_repeat_buyers'])], [
        (['Параметр когорт','Значение'],[
            ('Начало доступной истории Orders, UTC',coverage[0][0] if coverage else 'Нет данных'),
            ('Конец наблюдения Orders, UTC исключительно',coverage[-1][1] if coverage else 'Нет данных'),
            ('Отображаемые когорты с',bounds['start'] or 'Все наблюдаемые'),
            ('Отображаемые когорты по',bounds['end'] or 'Все наблюдаемые'),
            ('Покупатели STRICT вне отображаемых когорт',strict['hidden_buyers'])]),
        (['Покрытие Orders с UTC включительно','По UTC исключительно'],coverage)], list(COHORT_NOTES))]
    sections.append(('Когорты — повторные покупки STRICT', [], [
        (list(HORIZON_HEADERS),[_horizon_row(h,summary['horizons'][str(h)]) for h in HORIZONS])], []))
    sections.append(('Когорты — месяцы первой наблюдаемой покупки STRICT', [], _cohort_tables(strict), []))
    sections.append(('Когорты — матрица удержания M0–M12 STRICT', [], _retention_tables(strict), []))
    comparison = []
    for policy in POLICIES:
        label = 'STRICT' if policy == POLICIES[0] else 'EXTENDED'
        p = saved['policies'][policy]['summary']
        for h in HORIZONS:
            comparison.append((label,p['cohort_size'],*_horizon_row(h,p['horizons'][str(h)])))
    sections.append(('Когорты — сравнение STRICT и EXTENDED', [], [
        (['Политика','Покупателей в когортах',*HORIZON_HEADERS],comparison)], []))
    if include_extended:
        extended = saved['policies'][POLICIES[1]]
        sections.append(('Когорты — месяцы первой наблюдаемой покупки EXTENDED', [], _cohort_tables(extended), []))
        sections.append(('Когорты — матрица удержания M0–M12 EXTENDED', [], _retention_tables(extended), []))
    return sections
