"""RFM structure changes only saved presentation, preserving every unique value."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QComboBox, QLabel, QTableWidget

from Application import statistics_cache as cache
from Application.statistics_analytics_charts import RfmSegmentsWidget
from Application.statistics_rfm_core import RfmAggregates, POLICIES, WINDOWS
from Application.statistics_rfm_presentation import (
    rfm_sections, rfm_report_sections, SEGMENT_LABELS, RFM_BUSINESS_ACTIONS,
)
from Application.theme.apply_theme import apply_app_theme
from tests.test_statistics_rfm_business import segment_population
from tests.test_statistics_rfm_core import AS_OF, SPACE
from tests.tabs.test_dataset_statistics_tab import app, window, rich_customer_result

__all__ = ['app', 'window', 'rich_customer_result']
pytestmark = pytest.mark.usefixtures('window_settings')


@pytest.fixture
def rfm_result(rich_customer_result):
    result = deepcopy(rich_customer_result)
    coverage = [(cache.parse_timestamp(a), cache.parse_timestamp(b))
                for a, b in next(s['intervals'] for s in result['coverage'] if s['source'] == 'Orders')]
    core = RfmAggregates(AS_OF, coverage, identity_space=SPACE)
    for namespace in ('offline1C', 'kanzlerKz'):
        for fact in segment_population(namespace):
            core.add(fact)
    result['rfm_statistics'] = core.result()
    return result


def test_order_and_all_unique_tables_unchanged(rfm_result):
    before = deepcopy(rfm_result)
    old = rfm_sections(rfm_result, compact=True)
    new = rfm_report_sections(rfm_result)
    assert [s[0] for s in new] == [
        'RFM-анализ покупателей', 'Активность покупателей и качество наблюдения',
        'Покупательская активность', 'Денежные данные и качество оценки', 'Показатели R, F и M',
        'RFM-сегменты покупателей', 'Определения сегментов и возможные бизнес-действия',
        'Баллы и пороги RFM', 'Изменение активности за два последовательных 12-месячных периода',
        'RFM — сравнение политик учёта покупок']
    assert new[0][1] == old[0][1]
    assert [s[2][0] for s in new[2:5]] == old[1][2]
    assert new[5][2] == old[3][2]
    assert new[7][2] == old[5][2]  # Exact Decimal thresholds, scores and None states.
    assert new[8][2] == old[6][2]  # Use ready comparisons, never subtract nested windows.
    assert new[9][2] == old[7][2]
    old_rows = old[2][2][0][1]
    expanded = new[5][2][0][1]
    assert [tuple(r[:5]) for r in expanded] == [tuple(r[:5]) for r in old_rows]
    assert len(expanded) == 42
    assert Counter((r[0], r[1]) for r in expanded) == {
        (window, unit): 7 for window in ('12 месяцев', '24 месяца', '48 месяцев') for unit in ('RUB', 'KZT')}
    assert rfm_result == before


def test_reference_keeps_all_definitions_and_actions(rfm_result):
    table = rfm_report_sections(rfm_result)[6][2][0]
    assert table[0] == ['Сегмент', 'Определение', 'Возможное бизнес-действие']
    assert table[1] == [(label, definition, RFM_BUSINESS_ACTIONS[code])
                        for code, (label, definition) in SEGMENT_LABELS.items()]
    assert len(table[1]) == 7
    assert 'не новый клиент компании' in table[1][2][1]
    assert 'Неполные деньги' in table[1][-1][1]


def test_unscorable_known_amount_is_preserved_without_full_m(rfm_result):
    rows = rfm_report_sections(rfm_result)[5][2][0][1]
    unknown = [r for r in rows if r[2] == SEGMENT_LABELS['UNSCORABLE'][0] and r[0] == '12 месяцев']
    assert len(unknown) == 2
    for row in unknown:
        assert row[3] > 0 and row[7] > Decimal(0)
        assert row[10] == 0 and row[11] == 'Нет данных'


def test_legacy_keeps_basic_segments_without_empty_business_table(rfm_result):
    r = deepcopy(rfm_result)
    for name in WINDOWS[:3]:
        for unit in ('RUB', 'KZT'):
            r['rfm_statistics']['policies'][POLICIES[0]]['windows'][name]['currencies'][unit].pop('segment_business')
    before = deepcopy(r)
    old = rfm_sections(r, compact=True)
    new = rfm_report_sections(r)
    assert new[5][2] == old[2][2]
    assert len(new[5][2]) == 1 and len(new[5][2][0][0]) == 6
    assert not any(t[0] == ['Показатель', 'Доступность'] for s in new for t in s[2])
    assert new[6][2][0][1][-1][0] == SEGMENT_LABELS['UNSCORABLE'][0]
    assert r == before


@pytest.mark.parametrize('field', ['buyers', 'buyer_rate', 'segment', 'partial'])
def test_inconsistent_business_cannot_silently_replace_basic_segments(rfm_result, field):
    r = deepcopy(rfm_result)
    c = r['rfm_statistics']['policies'][POLICIES[0]]['windows']['LAST_12M']['currencies']['RUB']
    if field == 'partial':
        c.pop('segment_business')
    elif field == 'segment':
        c['segment_business'][0]['segment'] = 'OTHER'
    else:
        c['segment_business'][0][field] += 7
    with pytest.raises(cache.StatisticsCacheError):
        rfm_report_sections(r)


def test_tolerated_but_inexact_share_retains_both_projections(rfm_result):
    r = deepcopy(rfm_result)
    c = r['rfm_statistics']['policies'][POLICIES[0]]['windows']['LAST_12M']['currencies']['RUB']
    c['segment_business'][0]['buyer_rate'] += 1e-12
    cache.validate_result(r)
    tables = rfm_report_sections(r)[5][2]
    assert len(tables) == 2
    assert tables[0] == rfm_sections(r, compact=True)[2][2][0]
    assert tables[1] == rfm_sections(r, compact=True)[3][2][0]


@pytest.mark.parametrize('dark', [False, True])
@pytest.mark.parametrize('width', [1000, 1600])
def test_rfm_visible_chart_selectors_and_wide_table_scroll(window, rfm_result, dark, width):
    host, tab = window
    apply_app_theme(QApplication.instance(), dark)
    host.resize(width, 850)
    tab.render(rfm_result)
    tab.sections.setCurrentIndex(3)
    tab.customer_sections.setCurrentIndex(3)
    host.show()
    QTest.qWait(40)
    page = tab.customer_sections.widget(3)
    model = tab.report.pages[3].groups[3]
    headings = [b.title for b in model.blocks if b.kind == 'heading']
    assert headings == [s[0] for s in rfm_report_sections(rfm_result)]
    assert not any(b.kind in ('accordion', 'details', 'explanation') for b in model.blocks)
    tables = page.findChildren(QTableWidget)
    assert len(tables) == 11 and all(t.isVisible() for t in tables)
    widgets = page.findChildren(RfmSegmentsWidget)
    assert len(widgets) == 1
    chart = widgets[0]
    period = chart.findChild(QComboBox, 'rfmChartWindow')
    currency = chart.findChild(QComboBox, 'rfmChartCurrency')
    assert period.count() == 3 and currency.count() == 2
    original = [tuple(t.item(r, c).text() for c in range(t.columnCount()))
                for t in tables for r in range(t.rowCount())]
    for p in range(3):
        for c in range(2):
            period.setCurrentIndex(p)
            currency.setCurrentIndex(c)
            QTest.qWait(5)
    assert original == [tuple(t.item(r, c).text() for c in range(t.columnCount()))
                        for t in tables for r in range(t.rowCount())]
    portrait = next(t for t in tables if t.columnCount() == 12)
    scroll = portrait.horizontalScrollBar()
    assert scroll.maximum() > 0
    scroll.setValue(scroll.maximum())
    QTest.qWait(10)
    rect = portrait.visualItemRect(portrait.item(0, 11))
    assert rect.right() <= portrait.viewport().width() and rect.right() > 0
    assert portrait.horizontalHeaderItem(11).text() == 'Средняя полная M'
    assert all(t.rowHeight(r) >= t.sizeHintForRow(r) for t in tables for r in range(t.rowCount()))
    assert all(label.height() >= label.heightForWidth(label.width())
               for label in page.findChildren(QLabel) if label.wordWrap())


def test_zero_known_amount_does_not_become_missing_full_m(rfm_result):
    r = deepcopy(rfm_result)
    coverage = [(cache.parse_timestamp(a), cache.parse_timestamp(b))
                for a, b in next(s['intervals'] for s in r['coverage'] if s['source'] == 'Orders')]
    r['rfm_statistics'] = RfmAggregates(AS_OF, coverage, identity_space=SPACE).result()
    rows = rfm_report_sections(r)[5][2][0][1]
    assert len(rows) == 42
    assert all(row[7] == Decimal(0) and row[8] == row[9] == row[11] == 'Нет данных' for row in rows)
