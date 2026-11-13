"""Four open cohort groups preserve saved bases, tables, graphics and compatibility."""
from collections import Counter
from copy import deepcopy
from datetime import date
from decimal import Decimal

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel, QTableWidget

from Application import statistics_cache as cache
from Application.statistics_analytics_charts import CohortRetentionHeatmap
from Application.statistics_charts import StatisticsChart
from Application.statistics_cohorts_core import CohortAggregates, HORIZONS, POLICIES
from Application.statistics_cohorts_presentation import (
    cohort_sections, cohort_report_sections, COHORT_GROUP_TITLES, REPEAT_TITLE,
    RETENTION_TITLE, HORIZON_HEADERS, MATRIX_HEADERS, CHANNEL_TITLE, CATEGORY_TITLE,
    MONEY_LEGACY_TEXT, CHANNEL_LEGACY_TEXT, CATEGORY_LEGACY_TEXT, COHORT_LEGACY_TEXT,
)
from Application.theme.apply_theme import apply_app_theme
from tests.test_statistics_cohorts_money import order, SPACE
from tests.tabs.test_dataset_statistics_tab import app, window, rich_customer_result

__all__ = ['app', 'window', 'rich_customer_result']
pytestmark = pytest.mark.usefixtures('window_settings')


@pytest.fixture
def cohort_result(rich_customer_result):
    r = deepcopy(rich_customer_result)
    saved = r['cohort_statistics']
    core = CohortAggregates(date.fromisoformat(saved['as_of_date']), saved['source_coverage'],
        identity_space=SPACE, cohort_start=saved['cohort_bounds']['start'], cohort_end=saved['cohort_bounds']['end'],
        category_names={'A': 'Рубашка', 'AA': 'Рубашка', 'B': 'Брюки'},
        category_paths={'A': ['Одежда', 'Рубашка'], 'AA': ['Одежда', 'Рубашка'], 'B': ['Одежда', 'Брюки']})
    records = [
        ('a', 1, '2026-01-01', '10.125', 'offline1C', 'A', ['A', 'B']),
        ('a', 2, '2026-01-11', '20.25', 'kanzlerKz', 'B', ['AA', None]),
        ('a', 3, '2026-06-01', '30', 'offline1C', 'A', ['A']),
        ('b', 1, '2026-01-02', '0', 'offline1C', 'A', [None]),
        ('b', 2, '2026-04-03', None, 'offline1C', None, ['B']),
        ('c', 1, '2026-08-25', '1234567.898', 'kanzlerKz', None, ['B']),
        ('d', 1, '2026-01-10', None, 'offline1C', 'A', ['A']),
        ('d', 2, '2026-01-11', '5', 'offline1C', 'A', ['B']),
        ('e', 1, '2026-02-01', '8', 'offline1C', 'B', ['AA']),
    ]
    for buyer, id, day, amount, unit, channel, categories in records:
        core.add(order(day, buyer=buyer, id=id, amount=amount, namespace=unit),
                 channel_external_id=channel, channel_name='Одинаковое имя' if channel else None,
                 category_codes=categories)
    r['cohort_statistics'] = core.result()
    cache.validate_result(r)
    return r


def tables(sections):
    return [(tuple(h), tuple(tuple(row) for row in rows)) for _, _, blocks, _ in sections for h, rows in blocks]


def test_four_groups_order_and_every_old_table_exactly_preserved(cohort_result):
    r = cohort_result
    original = deepcopy(r)
    old, new = cohort_sections(r, compact=True), cohort_report_sections(r)
    titles = [s[0] for s in new]
    assert [t for t in titles if t in COHORT_GROUP_TITLES] == list(COHORT_GROUP_TITLES)
    assert titles.index(REPEAT_TITLE) < titles.index('Повторные покупки по когортам')
    assert titles.index('Повторность по когортам и срокам наблюдения') < titles.index('Сравнение правил учета повторных покупок')
    assert titles.index('Сравнение правил учета повторных покупок') < titles.index(RETENTION_TITLE)
    assert titles.index(RETENTION_TITLE) < titles.index('Денежная ценность месячных когорт')
    assert titles.index('Денежная ценность второй покупки') < titles.index(CHANNEL_TITLE) < titles.index(CATEGORY_TITLE)
    assert titles[-1] == 'Переходы между категориями'
    assert Counter(tables(new)) == Counter(tables(old)) and len(tables(new)) == 16
    assert r == original


def test_nine_original_kpis_have_eight_cards_and_an_exact_context(cohort_result):
    old, new = cohort_sections(cohort_result, compact=True), cohort_report_sections(cohort_result)
    old_cards = [c for s in old for c in s[1]]
    cards = [c for s in new for c in s[1]]
    channel = next(s for s in old if s[0] == CHANNEL_TITLE)
    duplicate = channel[1][0]
    assert len(old_cards) == 9 and len(cards) == 8
    assert Counter(cards) == Counter(old_cards) - Counter([duplicate])
    n = cohort_result['cohort_statistics']['policies'][POLICIES[0]]['summary']['observed_repeat_buyers']
    assert duplicate[1] == n
    assert any(f'База переходов: {n} покупателей' in text for s in new for _, text in s[3])
    assert new[0][1] == old[0][1]


@pytest.mark.parametrize('field', ['buyers', 'repeat_buyers'])
def test_same_number_is_not_enough_without_the_same_transition_base(cohort_result, field):
    r = deepcopy(cohort_result)
    r['cohort_statistics']['policies'][POLICIES[0]]['purchase_channels'][field] += 1
    new = cohort_report_sections(r)
    channel = next(s for s in new if s[0] == CHANNEL_TITLE)
    assert len(channel[1]) == 5
    assert not any(text.startswith('База переходов:') for _, text in channel[3])


def test_horizon_denominators_censoring_and_outside_base_are_saved_values(cohort_result):
    s = cohort_result['cohort_statistics']['policies'][POLICIES[0]]['summary']
    new = cohort_report_sections(cohort_result)
    rows = next(rows for h, rows in tables(new) if h == HORIZON_HEADERS)
    assert [row[0] for row in rows] == list(HORIZONS)
    for row in rows:
        c = s['horizons'][str(row[0])]
        assert row[1:3] == (c['eligible'], c['repeated'])
        assert row[3] == (c['repeat_rate'] if c['repeat_rate'] is not None else 'Нет данных')
        assert row[4:] == (c['censored'], c['coverage_incomplete'], c['censored_and_incomplete'],
                           c['observed_repeated'], c['observed_repeated_ineligible'])
    assert any(row[1] < s['cohort_size'] for row in rows)
    assert rows[-1][1] == 0 and rows[-1][3] == 'Нет данных' and rows[-1][7] > 0


def test_retention_m0_to_m12_and_observability_are_not_horizon_repeat(cohort_result):
    new = cohort_report_sections(cohort_result)
    matrix = next(rows for h, rows in tables(new) if h == MATRIX_HEADERS)
    details = next(rows for h, rows in tables(new) if len(h) == 10 and h[1] == 'Смещение')
    saved = cohort_result['cohort_statistics']['policies'][POLICIES[0]]['cohorts']
    assert len(details) == len(saved)*12
    for row in matrix:
        assert row[1] == saved[row[0]]['cohort_size'] and len(row) == 14
        assert row[-1] == 'Нет данных'
    text = ' '.join(text for s in new for _, text in s[3])
    assert '90 дней' in text and 'Удержание M3' in text and 'не взаимозаменяемы' in text
    assert 'Они не означают «не вернулись»' in text


def test_money_keeps_currencies_first_repeat_second_and_quality_separate(cohort_result):
    new = cohort_report_sections(cohort_result)
    monthly = next(s for s in new if s[0] == 'Денежная ценность месячных когорт')
    second = next(s for s in new if s[0] == 'Денежная ценность второй покупки')
    assert len(monthly[2]) == len(second[2]) == 2
    for group in (monthly, second):
        assert 'RUB' in ' '.join(group[2][0][0]) and 'KZT' in ' '.join(group[2][1][0])
        assert group[2][0][0][-1].endswith('(все валюты)')
        assert [row[-1] for row in group[2][0][1]] == [row[-1] for row in group[2][1][1]]
    assert all(type(row[c]) is Decimal or row[c] == 'Нет данных'
               for _, rows in monthly[2] for row in rows for c in (4, 5, 6, 7))
    assert any(row[4] == Decimal(0) and row[5] == 'Нет данных' for _, rows in second[2] for row in rows)
    text = ' '.join(text for s in new for _, text in s[3])
    assert text.count('Известная сумма принятых товарных позиций') == 1
    assert 'RUB и KZT не суммируются' in text and 'третий и последующие исключены' in text


def test_unknown_channels_categories_multilabel_and_both_transition_shares_survive(cohort_result):
    new = cohort_report_sections(cohort_result)
    channel = next(s for s in new if s[0] == 'Переходы между каналами')[2][0]
    assert len(channel[0]) == 5 and len(channel[1]) == 3
    assert any('Не указан' in row[:2] for row in channel[1])
    category = next(s for s in new if s[0] == 'Переходы между категориями')[2][0]
    saved = cohort_result['cohort_statistics']['policies'][POLICIES[0]]['purchase_categories']
    assert len(category[1]) == len(saved['transitions'])
    assert sum(row[2] for row in category[1]) > saved['repeat_buyers']
    text = str(new)
    assert 'Категория не определена' in text and '[A]' in text and '[AA]' in text
    assert 'идентификаторов каналов не доказывает' in text and 'внутри одного заказа' in text


@pytest.mark.parametrize('missing', ['monetary', 'purchase_channels', 'purchase_categories', 'all_nested', 'cohorts', 'null'])
def test_old_dto_keeps_the_existing_missing_states_and_tables(cohort_result, missing):
    r = deepcopy(cohort_result)
    if missing == 'cohorts':
        r.pop('cohort_statistics')
    elif missing == 'null':
        r['cohort_statistics'] = None
    else:
        s = r['cohort_statistics']['policies'][POLICIES[0]]
        for key in (('monetary', 'purchase_channels', 'purchase_categories') if missing == 'all_nested' else (missing,)):
            s.pop(key)
    before = deepcopy(r)
    old, new = cohort_sections(r, compact=True), cohort_report_sections(r)
    assert Counter(tables(old)) == Counter(tables(new))
    assert [s[0] for s in new if s[0] in COHORT_GROUP_TITLES] == list(COHORT_GROUP_TITLES)
    expected = {'monetary': MONEY_LEGACY_TEXT, 'purchase_channels': CHANNEL_LEGACY_TEXT,
                'purchase_categories': CATEGORY_LEGACY_TEXT, 'cohorts': COHORT_LEGACY_TEXT}
    if missing in expected:
        assert expected[missing] in str(new)
    if missing in ('all_nested', 'cohorts', 'null'):
        assert all(text in str(new) for text in (MONEY_LEGACY_TEXT, CHANNEL_LEGACY_TEXT, CATEGORY_LEGACY_TEXT))
    assert r == before


@pytest.mark.parametrize('dark', [False, True])
@pytest.mark.parametrize('width', [1000, 1600])
def test_graphics_open_tables_both_themes_and_retention_horizontal_scroll(window, cohort_result, dark, width):
    host, tab = window
    apply_app_theme(QApplication.instance(), dark)
    host.resize(width, 900)
    tab.render(cohort_result)
    tab.sections.setCurrentIndex(3)
    tab.customer_sections.setCurrentIndex(4)
    host.show()
    QTest.qWait(40)
    page = tab.customer_sections.widget(4)
    blocks = tab.report.pages[3].groups[4].blocks
    assert [b.title for b in blocks if b.kind == 'heading'] == [s[0] for s in cohort_report_sections(cohort_result)]
    assert not any(b.kind in ('accordion', 'details', 'explanation') for b in blocks)
    assert len(page.findChildren(StatisticsChart)) == len(page.findChildren(CohortRetentionHeatmap)) == 1
    for index, block in enumerate(blocks):
        if block.kind == 'table' and block.headers in (HORIZON_HEADERS, MATRIX_HEADERS):
            assert blocks[index-1].kind == ('chart' if block.headers == HORIZON_HEADERS else 'heatmap')
    widgets = page.findChildren(QTableWidget)
    assert len(widgets) == 16 and all(w.isVisible() for w in widgets)
    matrix = next(t for t in widgets if t.columnCount() == 14)
    matrix.setMaximumWidth(900)
    QTest.qWait(10)
    scroll = matrix.horizontalScrollBar()
    assert scroll.maximum() > 0
    scroll.setValue(scroll.maximum())
    rect = matrix.visualItemRect(matrix.item(0, 13))
    assert rect.right() <= matrix.viewport().width() and rect.right() > 0
    assert matrix.horizontalHeaderItem(13).text() == 'M12, %'
    assert all(t.rowHeight(r) >= t.sizeHintForRow(r) for t in widgets for r in range(t.rowCount()))
    assert all(label.height() >= label.heightForWidth(label.width())
               for label in page.findChildren(QLabel) if label.wordWrap())
