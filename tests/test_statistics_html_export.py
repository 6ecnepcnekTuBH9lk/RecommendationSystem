"""Offline aggregate-only report, typed parity and hostile labels."""
from copy import deepcopy
from decimal import Decimal, localcontext
from html.parser import HTMLParser
import json
import subprocess
import sys

import pytest

from Application import statistics_html_export as html
from Application import statistics_export as xlsx
from Application.analysis_filter import AnalysisFilter
from statistics_export_samples import complete_snapshot, empty_snapshot, stress_snapshot
from tabs.test_dataset_statistics_tab import sample_result

__all__ = ['sample_result']


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.texts = []
        self.chart_json = ''
        self.in_chart = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if tag == 'script' and attrs.get('id') == 'chart-data':
            self.in_chart = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_chart = False

    def handle_data(self, data):
        self.texts.append(data)
        if self.in_chart:
            self.chart_json += data


def test_six_sections_all_kpi_filters_snapshot_and_no_external_resources(sample_result):
    result = complete_snapshot(sample_result)
    result['analysis_filter'] = AnalysisFilter('2025-01-01', '2025-01-31', ('Рубашки',), ('Зима',)).to_dict()
    before = deepcopy(result)
    doc = Document(html.build_statistics_html(result))
    assert result == before
    assert len([a for t, a in doc.tags if t == 'section']) == 6
    assert len([a for t, a in doc.tags if a.get('class') == 'kpi']) == 8
    assert len([a for t, a in doc.tags if a.get('id') == 'plotly-library']) == 1
    assert all(not any(key in attrs for key in ('src', 'href')) for _, attrs in doc.tags)
    assert not any(tag in ('iframe', 'img', 'object', 'link') for tag, _ in doc.tags)
    texts = ''.join(doc.texts)
    assert all(text in texts for text in ('Рубашки', 'Зима', '01.01.2025', '31.01.2025', '24.09.2026',
                                         'Доля повторных покупателей'))
    assert 'Не обновляется автоматически.' not in texts and 'Строк:' not in texts
    charts = json.loads(doc.chart_json)
    assert len(charts) == 24
    assert {spec['section'] for spec in charts} == {0, 1, 2, 3, 4}
    assert all(chart['data'] for chart in charts)
    for section in html.presentation(result):
        for block in section.blocks:
            if block['kind'] == 'table':
                assert all(header in texts for header in block['headers'])
                assert all(cell['text'] in texts for row in block['rows'] for cell in row)


def test_presentation_cells_match_xlsx_source_values_and_formats(sample_result):
    sections = html.presentation(complete_snapshot(sample_result))
    assert sum(block['kind'] == 'table' for section in sections for block in section.blocks) == 61
    cell = html._cell(xlsx._Numeric(Decimal('12345678901234567890.123456789'), '#,##0.00'))
    assert cell == {'text': '12 345 678 901 234 567 890,12', 'sort': '12345678901234567890.123456789', 'kind': 'number'}
    assert html._cell('000001')['text'] == '000001'
    assert html._cell(None)['kind'] == 'empty'
    assert html._cell(0)['text'] == '0'
    assert html._cell(1e-10)['sort'] == '0.0000000001'
    assert html._cell(xlsx._Numeric(Decimal('43.48'), xlsx.PERCENT_FORMAT))['text'] == '43,48 %'
    age = next(spec for spec in html.chart_specs(complete_snapshot(sample_result)) if spec['title'] == 'Возрастные группы')
    assert age['horizontal'] and age['data'][0]['orientation'] == 'h'


def test_chart_numbers_currencies_ranking_and_full_stress_tables(sample_result):
    result = stress_snapshot(sample_result)
    result['top_favorited_products'] = (('000001', 'Товар', 7, 2, 99, 1),) + result['top_favorited_products'][1:]
    result['order_monthly_dynamics'] = (('2025-01', 1, '12345678901234567890.12345', 2, '123.456'),)
    charts = html.chart_specs(result)
    rub = next(s for s in charts if s['title'] == 'Сумма покупок - RUB')
    kzt = next(s for s in charts if s['title'] == 'Сумма покупок - KZT')
    assert rub['data'][0]['exact'] == ['12345678901234567890.12345']
    assert '12 345 678 901 234 567 890,12' in rub['data'][0]['hovertext'][0]
    assert kzt['data'][0]['exact'] == ['123.456']
    fav = next(s for s in charts if s['title'] == 'Избранное товаров - Top-10')
    assert fav['data'][0]['exact'][0] == '7'
    assert 'Просмотры: 99' in fav['data'][0]['hovertext'][0]
    views = next(s for s in charts if s['title'] == 'Динамика просмотров')
    assert len(views['data'][0]['y']) == 72
    assert views['data'][0]['exact'] == [str(row[1]) for row in result['action_monthly_dynamics']]
    table_rows = [row for section in html.presentation(result) for block in section.blocks if block['kind'] == 'table'
                  for row in block['rows']]
    assert sum('Длинное название тестового магазина' in row[0]['text'] for row in table_rows) == 240
    assert all(any(row[0]['text'] == name for row in table_rows) for name, *_ in result['product_season_statistics'])
    assert len(result['product_season_statistics']) == 60
    for row in result['top_purchased_products']:
        assert any(cells[0]['text'] == row[0] and cells[1]['text'] == row[1] for cells in table_rows)
    collection = next(s for s in charts if s['title'] == 'Коллекции по покупкам - Top-10')
    assert len(collection['data'][0]['x']) == 10


@pytest.mark.parametrize('missing', [False, True])
def test_empty_and_missing_profiles_never_create_fictitious_series(sample_result, missing):
    result = empty_snapshot(sample_result, missing_profiles=missing)
    assert all(not spec['data'] for spec in html.chart_specs(result))
    text = html.build_statistics_html(result)
    assert 'Нет данных для диаграммы.' in text
    if missing:
        assert 'Данные профилей клиентов отсутствуют.' not in text
        section = html.presentation(result)[4]
        for index, block in enumerate(section.blocks):
            if block['kind'] == 'table' and block['headers'][0] in ('Пол', 'Возрастная группа'):
                fragment = text.split(f'<table id="table-4-{index}">', 1)[1].split('</table>', 1)[0]
                assert fragment.count('Нет данных') == 2 * len(block['rows'])
                assert '>0<' not in fragment
    assert html.presentation(result)[4].blocks[0]['rows'][0][1]['kind'] == ('empty' if missing else 'number')


@pytest.mark.parametrize('attack', ['</script><script>window.PWNED=1</script>', '<img src=x onerror="alert(1)">',
                                  '\u2028\u2029 & < > " \''])
def test_html_and_script_injection_never_create_markup(sample_result, attack):
    result = deepcopy(sample_result)
    result['top_viewed_products'] = (('000001', attack, 2, 1, 0, 1),)
    result['warnings'] = (attack,)
    result['private_customer_id'] = 'SECRET-CUSTOMER'
    result['email'] = 'PRIVATE@example.test'
    result['local_path'] = 'C:/PRIVATE-SOURCE'
    result['token'] = 'SECRET-TOKEN'
    text = html.build_statistics_html(result)
    doc = Document(text)
    assert len([1 for tag, _ in doc.tags if tag == 'script']) == 4
    assert not any(tag == 'img' or any(key.startswith('on') for key in attrs) for tag, attrs in doc.tags)
    assert all(secret not in text for secret in ('SECRET-CUSTOMER', 'PRIVATE@example.test', 'PRIVATE-SOURCE', 'SECRET-TOKEN'))
    chart = next(s for s in json.loads(doc.chart_json) if s['title'] == 'Просмотры товаров - Top-10')
    assert '<script>' not in chart['data'][0]['hovertext'][0]
    assert '000001' in chart['labels'][0]


def test_period_gaps_warnings_and_offline_whitelist(sample_result, monkeypatch):
    import sqlite3
    result = deepcopy(sample_result)
    for source in result['coverage']:
        if source['source'] in ('Actions', 'Orders'):
            source['intervals'] = (('2024-01-01T00:00:00Z', '2024-02-01T00:00:00Z'),
                                   ('2025-01-01T00:00:00Z', '2025-02-01T00:00:00Z'))
    result['warnings'] = ('Canonical Customers отсутствует: число профилей неизвестно.',)
    monkeypatch.setattr(sqlite3, 'connect', lambda *args, **kwargs: pytest.fail('SQLite read'))
    text = html.build_statistics_html(result)
    assert 'Статистика рассчитана за периоды:' not in text
    assert '01.01.2024 - 01.02.2024; 01.01.2025 - 01.02.2025' in text
    assert 'Данные клиентов отсутствуют: число профилей неизвестно.' not in text


def test_html_module_import_is_qt_free():
    code = "import sys; import Application.statistics_html_export; assert not any(k.startswith('PyQt') for k in sys.modules)"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('value,expected', [
    ('43.482704072347516', '43,48'), ('39.918924842737425', '39,92'),
    ('15815693602.6700', '15 815 693 602,67'), ('2.675', '2,68'), ('999.995', '1 000'),
    ('100.0000', '100'), ('-0.004', '0'), ('-0.005', '-0,01'), ('0.00000001', '0'),
    ('123456789012345678901234567890.995', '123 456 789 012 345 678 901 234 567 891'),
])
def test_display_number_rounding_does_not_change_exact_value(value, expected):
    number = Decimal(value)
    with localcontext() as context:
        context.prec = 6
        assert html.exact_number(number) == expected
    assert number == Decimal(value)
    assert html._cell(number)['sort'] == format(number, 'f')
    assert html._cell(number)['text'] == expected


def test_html_typography_period_and_identifiers_are_presentation_only(sample_result):
    result = deepcopy(sample_result)
    code = '00Ё–—001'
    result['top_viewed_products'] = ((code, 'Тёплый Ёж — 2–5', 2, 1, 0, 1),)
    result['analysis_filter'] = AnalysisFilter('2025-01-01', '2025-01-31', ('Лёгкие',), ('Зимняя–Ёлка',)).to_dict()
    result['warnings'] = ('UNIQUE-WARNING-NOT-FOR-EXPORT',)
    before = deepcopy(result)
    text = html.build_statistics_html(result)
    doc = Document(text)
    assert 'Аналитический отчет' in text and 'Дата расчета' in text
    assert 'Легкие' in text and 'Зимняя-Елка' in text and 'Теплый Еж - 2-5' in text
    assert code in text  # Identifiers are explicitly exempt from human text normalization.
    assert '01.01.2025 - 31.01.2025' in text and 'Статистика рассчитана' not in text
    assert 'UNIQUE-WARNING-NOT-FOR-EXPORT' not in text
    assert all(token not in text for token in ('class="note"', 'Предупреждения', 'row-count', 'Строк:', 'class="snapshot"'))
    charts = json.loads(doc.chart_json)
    product = next(spec for spec in charts if spec['title'] == 'Просмотры товаров - Top-10')
    assert code in product['labels'][0] and 'Теплый Еж - 2-5' in product['data'][0]['hovertext'][0]
    assert result == before


def test_split_charts_preserve_all_original_months_and_series(sample_result):
    result = stress_snapshot(sample_result)
    result['action_monthly_dynamics'] = tuple((month, 1000000, 1, 2, 1) for month, *_ in result['action_monthly_dynamics'])
    result['order_monthly_dynamics'] = tuple((month, 100000, rub, 1, kzt) for month, _, rub, _, kzt in result['order_monthly_dynamics'])
    specs = html.chart_specs(result)
    assert [sum(spec['section'] == section for spec in specs) for section in range(6)] == [6, 4, 6, 4, 4, 0]
    assert all(len(spec['data']) <= 1 for spec in specs)
    for section in (0, 2):
        for currency, expected in (('RUB', '100000'), ('KZT', '1')):
            spec = next(spec for spec in specs if spec['section'] == section and spec['title'] == f'Количество покупок - {currency}')
            assert spec['data'][0]['exact'] == [expected] * 72
    for title, expected in (('Динамика просмотров', '1000000'), ('Динамика избранного', '2')):
        spec = next(spec for spec in specs if spec['section'] == 0 and spec['title'] == title)
        assert spec['data'][0]['exact'] == [expected] * 72
    assert all(not any(char in spec['title'] for char in 'ёЁ—–') for spec in specs)


def test_unavailable_period_and_raw_machine_labels(sample_result):
    result = deepcopy(sample_result)
    result['coverage'] = tuple(dict(source, intervals=()) for source in result['coverage'])
    assert html.period_text(result) == 'Не указан'
    section = html._Section()
    section.table(('Код', 'Название'), [('00Ё–', 'Ёлка–товар')])
    assert section.blocks[0]['rows'][0][0]['text'] == '00Ё–'
    assert section.blocks[0]['rows'][0][1]['text'] == 'Елка-товар'
    assert section.blocks[0]['rows'][0][1]['sort'] == 'Ёлка–товар'
