"""Presentation restructuring keeps visible metrics, sources, policies and all category pairs."""
from copy import deepcopy
from decimal import Decimal
from io import BytesIO
import sqlite3
from zipfile import ZipFile

import pytest
from openpyxl import load_workbook
from PyQt6.QtWidgets import QApplication,QTableWidget,QLabel
from PyQt6.QtTest import QTest

from Application import statistics_report_presentation as report,statistics_cache as cache,statistics_bundle_export as bundle
from Application import dataset_statistics as statistics
from Application.statistics_presentation import grouped_buyer_sections,PROFILE_SLICE_LABEL
from Application.statistics_copurchases_presentation import copurchase_sections
from Application.statistics_table_explanation_widget import StatisticsTableExplanation
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import app,window,rich_customer_result,sample_result
from tests.test_statistics_unified_presentation import Tables
from tests.test_statistics_rfm_integration import dataset,calculate,SELECTION
from Application.analysis_filter import AnalysisFilter

__all__=['app','window','rich_customer_result','sample_result','dataset']
pytestmark=pytest.mark.usefixtures('window_settings')


def groups(result):
    return report.report_presentation(result).pages[3].groups


def strict(result):
    return result['buyer_statistics']['policies']['STRICT_COMPLETED']


def tables(group):
    return [b for b in group.blocks if b.kind=='table']


def test_overview_bases_fullness_and_only_proven_narrow_demography_removed(rich_customer_result):
    r=rich_customer_result;before=deepcopy(r);g=groups(r)[0];s=strict(r)
    headers=[b.headers for b in tables(g)]
    assert ('Пол покупателей','Покупателей','Доля, %') not in headers
    assert ('Возраст покупателей','Покупателей','Доля, %') not in headers
    for dim,h in [('gender','Пол'),('age','Возрастная группа')]:
        expected=[(row['label'],row['buyers'],row['buyer_rate']) for row in s['demographic_analytics'][dim]]
        assert [tuple(row) for row in s['demography'][dim]]==expected
        table=next(b for b in tables(g) if b.headers[0]==h and len(b.headers)==8)
        assert [row[:3] for row in table.rows]==expected
        assert sum(row[1] for row in table.rows)==s['buyers']
    labels=[label for b in g.blocks if b.kind=='cards' for label,_ in b.values]
    assert PROFILE_SLICE_LABEL not in labels and 'Покупатели' not in labels
    assert {'Покупатели с профилем','Покупатели без профиля','Неопределённый пол покупателей','Неопределённый возраст покупателей'}<=set(labels)
    context=[b.title for b in g.blocks if b.options.get('context')]
    assert len(context)==4 and all('База:' in text for text in context)
    assert any('клиентская база' in t for t in context) and any('F/sold' in t for t in context)
    assert any('Не только подтверждённые покупатели' in t for t in context)
    headings=[b.title for b in g.blocks if b.kind=='heading']
    assert headings.index('Охват клиентской базы')<headings.index('Демография клиентов')<headings.index('Активность клиентов товарного среза')<headings.index('Портрет покупателей — подтверждённые покупки')
    assert r==before


@pytest.mark.parametrize('scenario',['missing','none','different_gender'])
def test_narrow_demography_is_kept_if_richer_equivalence_is_unproven(rich_customer_result,scenario):
    r=deepcopy(rich_customer_result);s=strict(r)
    if scenario=='missing':s.pop('demographic_analytics')
    elif scenario=='none':s['demographic_analytics']=None
    else:s['demographic_analytics']['gender'][0]['buyer_rate']=12345
    headers=[b.headers for b in tables(groups(r)[0])]
    assert ('Пол покупателей','Покупателей','Доля, %') in headers
    if scenario!='different_gender':assert ('Возраст покупателей','Покупателей','Доля, %') in headers


def test_purchases_order_all_strict_metrics_and_l_remain_visible(rich_customer_result):
    r=rich_customer_result;g=groups(r)[1];s=strict(r)
    headings=[b.title for b in g.blocks if b.kind=='heading']
    assert headings==['Покупатели — подтверждённые покупки','Частота подтверждённых покупок',
        'Интервалы между подтверждёнными покупками','Первые и последние наблюдаемые покупки периода',
        'Совместные покупки категорий','Категории в составе заказов','Совместные покупки категорий',
        'Покупательская активность отобранных товаров','Как изменяются показатели при расширенном учёте покупок']
    cards=dict((k,v) for b in g.blocks if b.kind=='cards' for k,v in b.values)
    assert cards['Покупатели (подтверждённые покупки)']==s['buyers']
    assert cards['Средний интервал, дней']==(s['intervals']['mean_days'] if s['intervals']['mean_days'] is not None else 'Нет данных')
    assert sum(b.kind=='chart' for b in g.blocks)==2
    comparison=next(b for b in tables(g) if b.headers==('Показатель','Подтверждённые покупки','Расширенный учёт покупок'))
    original=[section for group,section in grouped_buyer_sections(r,compact=True) if group=='purchases']
    old=next(section[2][0][1] for section in original if section[2] and section[2][0][0]==list(comparison.headers))
    assert comparison.rows==tuple(tuple(report._present_value(v,report._buyer_display) for v in row) for row in old)
    legacy=next(b for b in tables(g) if b.headers==('Количество заказов','Покупателей','Доля покупателей, %'))
    assert legacy.rows==tuple(tuple(row) for row in r['purchase_order_distribution'])
    assert not any(b.kind in ('details','accordion','fold') for b in g.blocks)


def test_strict_remains_independent_of_product_filter_and_different_from_l(dataset):
    a=calculate(dataset,SELECTION)
    b=calculate(dataset,AnalysisFilter(SELECTION.start_date,SELECTION.end_date,nomenclature_types=('missing',)))
    assert strict(a)==strict(b)
    assert b['purchase_users']==0 and strict(b)['buyers']>0
    for r in (a,b):
        g=groups(r)[1]
        values=dict((label,v) for block in g.blocks if block.kind=='cards' for label,v in block.values)
        assert values['Покупатели (подтверждённые покупки)']==strict(r)['buyers']
        legacy=next(t for t in tables(g) if t.headers[0]=='Количество заказов')
        assert legacy.rows==tuple(tuple(row) for row in r['purchase_order_distribution'])


def pair_snapshot():
    codes=[None]+[str(i) for i in range(14)]
    categories=[dict(code=c,name='Не определена' if c is None else 'Одинаковое',path=None,
                     orders=2,buyers=2,order_rate=10.) for c in codes]
    pairs=[dict(category1=codes[i],category2=codes[i+1],orders=i+1,buyers=i+1,
                order_rate=(i+1)/20*100,given_first_rate=Decimal('12.123456789123456789') if i==0 else i,
                given_second_rate=i+2) for i in range(13)]
    return {'category_copurchases':dict(orders=20,unknown_orders=2,multicategory_orders=13,
        multicategory_rate=65.,multicategory_buyers=13,categories=categories,pairs=pairs)}


def test_pair_merge_all_columns_stable_codes_unknowns_and_more_than_top_ten():
    s=pair_snapshot();before=deepcopy(s)
    old=copurchase_sections(s);new=copurchase_sections(s,combined=True)
    assert len(new)==3 and len(new[-1][2][0][0])==7
    rows=new[-1][2][0][1];a=old[-2][2][0][1];b=old[-1][2][0][1]
    assert len(rows)==13 and rows==[(*first,*second[2:]) for first,second in zip(a,b)]
    assert rows[0][5]==Decimal('12.123456789123456789')
    assert 'неполные данные' in rows[0][0] and rows[1][0]!=rows[1][1]
    assert s==before and new[0][1]==old[0][1] and new[0][3]==old[0][3]


@pytest.mark.parametrize('dark',[False,True])
@pytest.mark.parametrize('scenario',['full','legacy','old_business','no_profiles'])
def test_saved_shared_visible_gui_html_xlsx_and_no_reads(window,rich_customer_result,sample_result,tmp_path,monkeypatch,dark,scenario):
    r=deepcopy(sample_result if scenario=='legacy' else rich_customer_result)
    if scenario=='old_business':strict(r).pop('demographic_analytics');strict(r).pop('category_copurchases')
    if scenario=='no_profiles':
        # Existing no-profile snapshot, not a recalculation.
        from statistics_export_samples import empty_snapshot
        r=empty_snapshot(r)
    path=tmp_path/'saved.json';cache.save_result(path,r);r=cache.load_result(path);raw=path.read_bytes();before=deepcopy(r)
    def forbidden(*a,**kw):pytest.fail('Presentation attempted source access or recalculation')
    monkeypatch.setattr(statistics,'iter_export',forbidden);monkeypatch.setattr(statistics,'calculate_statistics',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    from Application.tabs import dataset_statistics_tab as ui
    monkeypatch.setattr(ui.DatasetStatisticsTab,'start',forbidden)
    host,tab=window;apply_app_theme(QApplication.instance(),dark);tab.render(r)
    tab.sections.setCurrentIndex(3);host.resize(1100,850);host.show();QApplication.processEvents()
    assert not tab.findChildren(StatisticsTableExplanation)
    assert tab.sections.count()==4 and tab.customer_sections.count()==5
    target=tmp_path/'report.zip';bundle.export_statistics_bundle(r,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==['Статистика.xlsx','Статистика.html']
        document=archive.read('Статистика.html').decode('utf-8')
        book=load_workbook(BytesIO(archive.read('Статистика.xlsx')))
    parsed=Tables(document)
    try:
        for group_index in (0,1):
            tab.customer_sections.setCurrentIndex(group_index);QApplication.processEvents()
            page=tab.customer_sections.widget(group_index);g=tab.report.pages[3].groups[group_index]
            widgets=iter(page.findChildren(QTableWidget))
            for i,b in enumerate(g.blocks):
                if b.kind!='table':continue
                w=next(widgets);values=[[c.text for c in row] for row in report.table_cells(b)]
                assert w.isVisible() and [[w.item(a,c).text() for c in range(w.columnCount())] for a in range(w.rowCount())]==values
                assert parsed.tables[f'table-3-{g.key}-{i}']==(values or [['Нет данных для таблицы.']])
            contexts=[l for l in page.findChildren(QLabel) if l.property('class')=='statisticsContext']
            assert contexts and all(l.isVisible() for l in contexts)
        assert '<details class="table-explanation"' not in document
        assert any('База: клиентская база' in str(row) for row in book['Клиенты'].values)
        assert r==before and path.read_bytes()==raw
    finally:book.close()


def test_merged_pair_table_horizontal_scroll_small_window(app):
    # Render the actual combined section using the production shared renderer.
    section=copurchase_sections(pair_snapshot(),combined=True)
    blocks=report._Blocks();report._analytics_sections(blocks,section)
    from PyQt6.QtWidgets import QWidget,QVBoxLayout
    from Application.tabs.dataset_statistics_tab import _render_report_blocks
    widget=QWidget();layout=QVBoxLayout(widget);_render_report_blocks(layout,blocks)
    widget.resize(650,800);widget.show();QTest.qWait(30)
    try:
        table=widget.findChildren(QTableWidget)[-1]
        assert table.columnCount()==7 and table.rowCount()==13 and table.horizontalScrollBar().maximum()>0
        table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum());QTest.qWait(20)
        rect=table.visualItemRect(table.item(0,6))
        assert rect.right()<table.viewport().width() and rect.right()>=0
    finally:widget.close();widget.deleteLater()
