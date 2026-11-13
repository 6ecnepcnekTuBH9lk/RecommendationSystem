"""Currency-complete, visible money presentation; old DTO and immutable saved values."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
from io import BytesIO
from zipfile import ZipFile
import sqlite3

import pytest
from openpyxl import load_workbook
from PyQt6.QtWidgets import QApplication,QTableWidget,QLabel
from Application import statistics_report_presentation as report,statistics_cache as cache,statistics_bundle_export as bundle
from Application import dataset_statistics as statistics
from Application.statistics_presentation import grouped_buyer_sections
from Application.statistics_chart_data import buyer_turnover_chart_data
from Application.statistics_table_explanation_widget import StatisticsTableExplanation
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import app,window,rich_customer_result,sample_result
from tests.test_statistics_unified_presentation import Tables

__all__=['app','window','rich_customer_result','sample_result']
pytestmark=pytest.mark.usefixtures('window_settings')


def money(result):
    return report.report_presentation(result).pages[3].groups[2]


def source_sections(result):
    return [s for g,s in grouped_buyer_sections(result,compact=True) if g=='money']


def expected_headings():
    return ['Денежные показатели покупателей']+[
        text+' — '+unit for unit in ('RUB','KZT') for text in (
            'Денежные показатели','Полнота денежной оценки','Распределение полной денежной ценности',
            'Годовая денежная аналитика','По полу','По возрасту')]+['Общие показатели денежной оценки — все валюты']


def test_money_order_all_twelve_kpis_and_every_unique_table_value(rich_customer_result):
    r=rich_customer_result;original=deepcopy(r);g=money(r);old=source_sections(r)
    assert [b.title for b in g.blocks if b.kind=='heading']==expected_headings()
    assert Counter((k,report._present_value(v,report._buyer_display)) for s in old for k,v in s[1])==Counter(
        (k,v) for b in g.blocks if b.kind=='cards' for k,v in b.values)
    assert sum(len(b.values) for b in g.blocks if b.kind=='cards')==12
    tables=[b for b in g.blocks if b.kind=='table']
    assert len(tables)==11
    for index,unit in enumerate(('RUB','KZT')):
        # Completeness, quintiles and annual data retain their entire original schema and exact values.
        for (h,rows),actual in zip(old[index][2],tables[index*5:index*5+3]):
            assert actual.headers==tuple(h)
            assert actual.rows==tuple(tuple(report._present_value(v,report._buyer_display) for v in row) for row in rows)
        for dim,offset in [('gender',0),('age',2)]:
            h,rows=old[-4+offset+index][2][0]
            actual=tables[index*5+3+(dim=='age')]
            assert actual.headers==tuple(h[:3]+h[4:])
            assert actual.rows==tuple(tuple(report._present_value(v,report._buyer_display) for v in row[:3]+row[4:]) for row in rows)
    assert r==original


def test_shared_unvalued_counter_uses_same_non_currency_dto_field(rich_customer_result):
    r=rich_customer_result;s=r['buyer_statistics']['policies']['STRICT_COMPLETED'];g=money(r)
    common=[b for b in g.blocks if b.kind=='table'][-1]
    expected=[(label,x['label'],x['unvalued_orders']) for dim,label in [('gender','Пол'),('age','Возраст')] for x in s['demographic_analytics'][dim]]
    assert common.rows==tuple(expected)
    for label in ('Пол','Возраст'):
        assert sum(n for dimension,group,n in common.rows if dimension==label)==s['quality']['unvalued_orders']
    assert any('не складываются между группировками' in b.title for b in g.blocks if b.kind=='text')
    assert all('Заказы без достоверной денежной оценки (все валюты)' not in b.headers
               for b in g.blocks if b.kind=='table' and b.headers[0]=='Группа')


def test_no_dedup_just_because_kpis_and_fullness_happen_to_be_equal(rich_customer_result):
    r=deepcopy(rich_customer_result);s=r['buyer_statistics']['policies']['STRICT_COMPLETED']
    for unit in ('RUB','KZT'):
        for field in ('amount','mean_order','median_order','mean_complete_per_buyer','median_complete_per_buyer',
                      'mean_known_per_buyer','median_known_per_buyer','complete_amount','complete_repeat_amount'):
            s['money'][unit][field]='0'
    g=money(r)
    for table in (b for b in g.blocks if b.kind=='table' and b.headers[0].startswith('Денежный показатель')):
        assert len(table.rows)==10
        assert dict(table.rows)['Средний известный подытог на покупателя']==report.ReportNumber(Decimal(0))
        assert dict(table.rows)['Полный оборот включённых покупателей']==report.ReportNumber(Decimal(0))
    assert len([b for b in g.blocks if b.kind=='cards'])==2


def test_known_partial_zero_none_repeat_shares_and_precision_remain_distinct(rich_customer_result):
    r=deepcopy(rich_customer_result);s=r['buyer_statistics']['policies']['STRICT_COMPLETED']
    rub=s['money']['RUB'];kzt=s['money']['KZT']
    rub.update(amount='1234567890123456789.123456789',mean_order='0',median_order=None,
               mean_complete_per_buyer=None,repeat_amount_rate=20.,complete_repeat_amount_rate=80.)
    kzt.update(amount='0',mean_order=None,median_order='0',mean_complete_per_buyer='0',
               repeat_amount_rate=None,complete_repeat_amount_rate=None)
    r['buyer_statistics']['coverage_complete']=False
    before=deepcopy(r);g=money(r)
    cards=dict((k,v) for b in g.blocks if b.kind=='cards' for k,v in b.values)
    assert cards['Сумма принятых товарных позиций, RUB (известная часть)']==report.ReportNumber(Decimal(rub['amount']))
    assert cards['Средний чек, RUB']==report.ReportNumber(Decimal(0)) and cards['Медианный чек, RUB']=='Нет данных'
    assert cards['Средний чек, KZT']=='Нет данных' and cards['Средняя полная сумма на покупателя, KZT']==report.ReportNumber(Decimal(0))
    tables=[b for b in g.blocks if b.kind=='table']
    assert dict(tables[0].rows)['Доля повторных в полном обороте, %']==80.
    assert cards['Доля повторных покупателей в известной сумме RUB, %']==20.
    assert any('Покрытие заказов неполное' in b.title for b in g.blocks if b.options.get('context'))
    assert r==before


def test_quintile_charts_immediately_before_full_tables_and_annual_coverage(rich_customer_result):
    r=rich_customer_result;g=money(r)
    for unit in ('RUB','KZT'):
        i=next(i for i,b in enumerate(g.blocks) if b.kind=='table' and b.headers[0]=='Полный оборот от, '+unit)
        assert g.blocks[i-1].kind=='chart' and g.blocks[i-1].data==buyer_turnover_chart_data(r,unit)
        year=next(b for b in g.blocks if b.kind=='table' and b.headers[0]=='Год' and unit in b.headers[4])
        for row,source in zip(year.rows,r['buyer_statistics']['policies']['STRICT_COMPLETED']['annual']):
            assert ('Полный год' if source['complete_year'] else 'Неполный год') in row[1]
    assert sum(b.kind=='chart' for b in g.blocks)==2


@pytest.mark.parametrize('scenario',['old_demography','no_buyer','empty'])
def test_old_results_do_not_receive_fake_shared_counts(rich_customer_result,sample_result,scenario):
    if scenario=='no_buyer':r=sample_result
    elif scenario=='empty':
        from statistics_export_samples import empty_snapshot
        r=empty_snapshot(rich_customer_result)
    else:
        r=deepcopy(rich_customer_result);r['buyer_statistics']['policies']['STRICT_COMPLETED'].pop('demographic_analytics')
    g=money(r);titles=[b.title for b in g.blocks if b.kind=='heading']
    assert 'Денежные показатели — RUB' in titles and 'Денежные показатели — KZT' in titles
    if scenario!='empty':
        assert 'Общие показатели денежной оценки — все валюты' not in titles
        assert any('Нет данных' in str(b.rows) for b in g.blocks if b.kind=='table')


@pytest.mark.parametrize('dark',[False,True])
@pytest.mark.parametrize('scenario',['full','legacy','old_demography'])
def test_saved_both_currencies_visible_and_shared_gui_html_xlsx_no_sources(
        window,rich_customer_result,sample_result,tmp_path,monkeypatch,dark,scenario):
    r=deepcopy(sample_result if scenario=='legacy' else rich_customer_result)
    if scenario=='old_demography':r['buyer_statistics']['policies']['STRICT_COMPLETED'].pop('demographic_analytics')
    path=tmp_path/'snapshot.json';cache.save_result(path,r);r=cache.load_result(path);saved=path.read_bytes();before=deepcopy(r)
    def forbidden(*a,**kw):pytest.fail('Money presentation attempted source IO or recalculation')
    monkeypatch.setattr(statistics,'calculate_statistics',forbidden);monkeypatch.setattr(statistics,'iter_export',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    from Application.tabs import dataset_statistics_tab as ui
    monkeypatch.setattr(ui.DatasetStatisticsTab,'start',forbidden)
    host,tab=window;apply_app_theme(QApplication.instance(),dark);tab.render(r)
    tab.sections.setCurrentIndex(3);tab.customer_sections.setCurrentIndex(2);host.show();QApplication.processEvents()
    page=tab.customer_sections.widget(2);g=tab.report.pages[3].groups[2]
    assert not page.findChildren(StatisticsTableExplanation)
    headers=[x.text() for x in page.findChildren(QLabel) if x.property('class')=='statisticsSection']
    assert 'ДЕНЕЖНЫЕ ПОКАЗАТЕЛИ — RUB' in headers and 'ДЕНЕЖНЫЕ ПОКАЗАТЕЛИ — KZT' in headers
    target=tmp_path/'money.zip';bundle.export_statistics_bundle(r,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==['Статистика.xlsx','Статистика.html']
        document=archive.read('Статистика.html').decode('utf-8')
        book=load_workbook(BytesIO(archive.read('Статистика.xlsx')))
    try:
        parsed=Tables(document);widgets=iter(page.findChildren(QTableWidget));excel=list(book['Клиенты'].values)
        for i,b in enumerate(g.blocks):
            if b.kind=='heading':assert any(row[0]==b.title for row in excel)
            if b.kind!='table':continue
            w=next(widgets);values=[[c.text for c in row] for row in report.table_cells(b)]
            assert w.isVisible() and [[w.item(a,c).text() for c in range(w.columnCount())] for a in range(w.rowCount())]==values
            assert parsed.tables[f'table-3-money-{i}']==(values or [['Нет данных для таблицы.']])
            for row in report.table_cells(b):
                assert any(all(actual==pytest.approx(float(c.value)) if c.kind=='number' else actual==c.value
                               for actual,c in zip(candidate,row)) for candidate in excel)
        assert '<details class="table-explanation"' not in document
        assert not any(b.kind in ('details','fold','accordion') for b in g.blocks)
        assert r==before and path.read_bytes()==saved
    finally:book.close()


def test_common_counter_not_moved_if_projection_cannot_be_proven_equal(rich_customer_result):
    result=rich_customer_result;sections=deepcopy(source_sections(result))
    title,cards,tables,notes=sections[-4]
    headers,rows=tables[0]
    row=list(rows[0]);row[3]+=123
    rows[0]=tuple(row)
    demographic,common=report._money_demographic_tables(result,sections)
    assert len(demographic[0][0])==len(demographic[1][0])==8
    assert len(demographic[2][0])==len(demographic[3][0])==7
    assert all(dimension=='Возраст' for dimension,label,n in common)
    assert demographic[0][1][0][3]==row[3]
