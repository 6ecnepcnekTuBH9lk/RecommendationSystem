"""Saved-only TOP-10 projections and independent, accessible table explanations."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import sqlite3
from zipfile import ZipFile
from io import BytesIO

import pytest
from openpyxl import load_workbook
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QTableWidget,QLabel,QWidget,QVBoxLayout,QApplication

from Application import statistics_chart_data as data, statistics_charts as charts
from Application import statistics_report_presentation as report
from Application import statistics_bundle_export as bundle,statistics_cache as cache,dataset_statistics as statistics
from Application.statistics_table_explanation_widget import StatisticsTableExplanation
from Application.statistics_table_explanations import TABLE_EXPLANATIONS,table_explanation,TableExplanation
from Application.statistics_presentation import DIRECT_CATEGORY_NOTE
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import LegacyStatisticsTab, app,window,sample_result,_visual_items
from tests.test_statistics_unified_presentation import Tables

__all__=['app','window','sample_result']
pytestmark=pytest.mark.usefixtures('window_settings')


def stores():
    return tuple((unit,f'{i:02}',f'Магазин {i}',9999-i,3,5,'7',str(100+i),'12.05')
                 for unit in ('RUB','KZT') for i in range(12))


@pytest.mark.parametrize('unit',['RUB','KZT'])
def test_store_top_ten_exact_money_not_orders_and_other_currency(unit):
    result={'store_statistics':stores()+((unit,'precise','Точный',1,1,1,'1','111.00000000000000001','111'),)}
    original=deepcopy(result);chart=data.store_purchase_chart_data(result,unit)
    assert chart.kind=='horizontal' and chart.money and chart.secondary==(unit=='KZT')
    assert len(chart.labels)==10 and chart.labels[0]=='Точный' and chart.labels[1]=='Магазин 11'
    assert chart.series[0][1][0]==Decimal('111.00000000000000001')
    assert all(type(v) is Decimal for v in chart.series[0][1])
    assert '111,00 '+unit in chart.tooltips[0][0]
    assert 'Валюта: '+unit in chart.tooltips[0][0] and result==original
    tip=chart.tooltips[0][1]
    assert 'Заказов: 9 988' in tip and 'Покупателей: 3' in tip and '12,05 '+unit in tip
    other='KZT' if unit=='RUB' else 'RUB'
    assert not any(other in tip for tip in chart.tooltips[0])


def test_store_ties_use_id_then_name_and_duplicate_labels_distinguish_ids():
    rows=[('RUB',key,name,n,1,1,'1','4.005','1') for key,name,n in [('b','Одинаково',999),('a','Одинаково',1),(None,'Нет ID',100)]]
    expected=None
    for ordered in [rows,list(reversed(rows))]:
        chart=data.store_purchase_chart_data({'store_statistics':ordered},'RUB')
        assert chart.labels==('Нет ID','Одинаково [a]','Одинаково [b]')
        assert chart.series[0][1]==(Decimal('4.005'),)*3
        assert chart==expected if expected else True
        expected=chart
    with pytest.raises(ValueError):data.store_purchase_chart_data({},'USD')


@pytest.mark.parametrize('result',[{}, {'store_statistics':None}, {'store_statistics':()}, {'store_statistics':(('KZT',None,'?',1,1,1,'1','1','1'),)}])
def test_empty_store_currency_and_missing_saved_optional_rows(result,app):
    chart=data.store_purchase_chart_data(result,'RUB')
    assert chart.empty
    widget=charts.create_chart(chart)
    try:assert any(l.text()=='Нет данных для отображения.' for l in widget.findChildren(QLabel))
    finally:widget.deleteLater()


def test_categories_rank_purchases_not_units_views_or_parents_and_saved_paths_only():
    rows=[(f'{i:03}','Одинаковое' if i in (10,11) else f'Категория {i}',1,100-i,30-i,i,'999')
          for i in range(12)]
    rows.append((None,'Не указано',1,0,0,50,'0.5'))
    result={'product_category_statistics':tuple(reversed(rows)),
            'product_category_paths':tuple((f'{i:03}',('Раздел',str(i))) for i in range(12))}
    original=deepcopy(result);chart=data.product_category_chart_data(result)
    assert chart.kind=='horizontal' and not chart.money
    assert chart.series==(('Покупки',(50,11,10,9,8,7,6,5,4,3)),)
    assert chart.labels[1:3]==('Одинаковое [011]','Одинаковое [010]')
    assert 'Не указано' in chart.labels[0] or 'не указана' in chart.labels[0]
    assert 'Код категории: 011' in chart.tooltips[0][1]
    assert 'Путь категории: Раздел / 11' in chart.tooltips[0][1]
    assert 'Просмотры: 89' in chart.tooltips[0][1] and 'Добавления в избранное: 19' in chart.tooltips[0][1]
    assert 'Продано единиц: 0,5' in chart.tooltips[0][0]
    assert len(chart.labels)==10 and result==original
    # No synthetic row for the shared parent is introduced.
    assert not any(label=='Раздел' for label in chart.labels)


def test_category_ties_by_code_legacy_paths_empty_and_zero():
    result={'product_category_statistics':(('b','B',1,99,4,3,'8'),('a','A',1,1,1,3,'1'))}
    chart=data.product_category_chart_data(result)
    assert chart.labels==('A','B')
    assert 'Путь недоступен в сохранённом результате' in chart.tooltips[0][0]
    assert data.product_category_chart_data({}).empty
    assert data.product_category_chart_data({'product_category_statistics':None}).empty
    assert data.product_category_chart_data({'product_category_statistics':(('z','Z',1,0,0,0,'0'),)}).empty


@pytest.mark.parametrize('dark',[False,True])
def test_panels_collapsed_independent_keyboard_geometry_and_live_theme(app,monkeypatch,dark):
    monkeypatch.setitem(TABLE_EXPLANATIONS,'orders.financials',TableExplanation('Пример финансов',(('Сумма','Пояснение'),)))
    monkeypatch.setitem(TABLE_EXPLANATIONS,'products.categories',TableExplanation('Пример категорий',(('Покупки','Пояснение'),)))
    apply_app_theme(app,dark)
    host=QWidget();layout=QVBoxLayout(host)
    first=StatisticsTableExplanation('orders.financials')
    second=StatisticsTableExplanation('products.categories')
    layout.addWidget(first);layout.addWidget(second);layout.addStretch()
    host.resize(1000,650);host.show();QTest.qWait(30)
    try:
        assert not first.toggle.isChecked() and first.content.isHidden() and second.content.isHidden()
        closed=first.height()
        first.toggle.setFocus();QTest.keyClick(first.toggle,Qt.Key.Key_Space);QTest.qWait(25)
        assert first.toggle.isChecked() and not first.content.isHidden() and second.content.isHidden()
        assert first.toggle.text()=='Как читать таблицу'
        assert 'развёрнуто' in first.toggle.accessibleName()
        assert first.height()>closed and first.content.height()>20
        second.toggle.click();QTest.qWait(20)
        assert first.toggle.isChecked() and second.toggle.isChecked()
        first.toggle.click();QTest.qWait(20)
        assert first.height()==closed and not first.toggle.isChecked() and second.toggle.isChecked()
        assert 'свёрнуто' in first.toggle.accessibleName()
        colors=first.styleSheet()
        apply_app_theme(app,not dark);QTest.qWait(30)
        assert first.styleSheet()!=colors and second.toggle.isChecked()
        host.resize(520,800);QTest.qWait(30)
        assert second.content.hasHeightForWidth() and second.content.height()>=second.content.heightForWidth(second.content.width())
        assert second.width()<=host.width() and second.content.width()<=second.width()
        assert second.font().pointSizeF()==host.font().pointSizeF()
    finally:host.close();host.deleteLater()


def test_explanations_ids_are_semantic_safe_and_not_mechanically_assigned():
    assert TABLE_EXPLANATIONS=={}
    assert table_explanation('Клиенты') is None and table_explanation('Финансовые показатели') is None
    example=TableExplanation('Описание',(('Термин','Определение'),),DIRECT_CATEGORY_NOTE)
    assert '<b>' in example.html() and DIRECT_CATEGORY_NOTE in example.html()
    assert '<script>' not in TableExplanation('<script>',(('A','<img>'),)).html()


@pytest.mark.parametrize('dark',[False,True])
def test_saved_gui_html_zip_tables_order_full_rows_and_no_source_access(window,sample_result,tmp_path,monkeypatch,dark):
    before=deepcopy(sample_result)
    saved=tmp_path/'legacy.json';cache.save_result(saved,sample_result)
    result=cache.load_result(saved);payload=saved.read_bytes()
    def forbidden(*a,**kw):pytest.fail('Saved visual components attempted source IO or a calculation')
    monkeypatch.setattr(statistics,'calculate_statistics',forbidden)
    monkeypatch.setattr(statistics,'iter_export',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    from Application.product_statistics import ProductAggregates
    from Application.order_statistics import OrderAggregates
    monkeypatch.setattr(ProductAggregates,'result',forbidden)
    monkeypatch.setattr(OrderAggregates,'result',forbidden)
    original=Path.open
    def guarded(p,*a,**kw):
        if p.suffix=='.csv' or 'canonical' in p.parts or 'input_data' in p.parts:forbidden()
        return original(p,*a,**kw)
    monkeypatch.setattr(Path,'open',guarded)
    apply_app_theme(QApplication.instance(),dark)
    _,tab=window;tab.render(result);QApplication.processEvents()
    panels=tab.findChildren(StatisticsTableExplanation)
    assert panels==[] and TABLE_EXPLANATIONS=={}
    assert not tab.sections.widget(3).findChildren(StatisticsTableExplanation)
    report_tables=[b for page in tab.report.pages for g in page.groups for b in g.blocks if b.kind=='table']
    target=tmp_path/'report.zip';bundle.export_statistics_bundle(result,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==['Статистика.xlsx','Статистика.html']
        document=archive.read('Статистика.html').decode('utf-8')
        book=load_workbook(BytesIO(archive.read('Статистика.xlsx')))
    parsed=Tables(document)
    try:
        assert '<details class="table-explanation"' not in document
        assert document.count(DIRECT_CATEGORY_NOTE)==1
        assert len(parsed.tables)==len(report_tables)
        for page_index,page in enumerate(tab.report.pages):
            for g_index,g in enumerate(page.groups):
                widget=tab.customer_sections.widget(g_index) if page.key=='customers' else tab.sections.widget(page_index)
                widgets=iter(widget.findChildren(QTableWidget))
                for i,b in enumerate(g.blocks):
                    if b.kind!='table':continue
                    table=next(widgets)
                    values=[[c.text for c in row] for row in report.table_cells(b)]
                    assert [[table.item(r,c).text() for c in range(table.columnCount())] for r in range(table.rowCount())]==values
                    assert parsed.tables[f'table-{page_index}-{g.key}-{i}']==(values or [['Нет данных для таблицы.']])
        for page_index,unit in [(1,'RUB'),(1,'KZT'),(2,None)]:
            page=tab.report.pages[page_index]
            table_id='orders.stores.'+unit if unit else 'products.categories'
            blocks=page.groups[0].blocks
            table_index=next(i for i,b in enumerate(blocks) if b.table_id==table_id)
            # No heading/title/index heuristics determine the explanation or chart attachment.
            chart=next(b for b in reversed(blocks[:table_index]) if b.kind=='chart')
            assert chart.data== (data.store_purchase_chart_data(result,unit) if unit else data.product_category_chart_data(result))
            parent=tab.sections.widget(page_index)
            visual=_visual_items(parent.widget().layout())
            table=next(t for t in parent.findChildren(QTableWidget) if t.property('statisticsTableId')==table_id)
            index=next(i for i,item in enumerate(visual) if item.widget() is table)
            preceding=visual[index-1].widget()
            if unit=='KZT':assert isinstance(preceding,QWidget)  # Empty saved currency placeholder.
            elif unit:assert isinstance(preceding,charts.StatisticsChart)
            else:assert isinstance(preceding,QLabel) and preceding.text()==DIRECT_CATEGORY_NOTE
        assert len(book.sheetnames)==4
        assert all('Как читать таблицу' not in str(c.value) for sheet in book for row in sheet for c in row)
        assert result==cache.load_result(saved) and saved.read_bytes()==payload and sample_result==before
    finally:book.close()


def test_full_store_table_is_not_truncated_by_chart(window,sample_result):
    result=deepcopy(sample_result);result['store_statistics']=stores()
    _,tab=window;tab.render(result)
    tables={t.property('statisticsTableId'):t for t in tab.sections.widget(1).findChildren(QTableWidget)}
    for unit in ('RUB','KZT'):
        assert tables['orders.stores.'+unit].rowCount()==12
        chart=next(c for c in tab.sections.widget(1).findChildren(charts.StatisticsChart) if c.data.title=='Магазины и каналы — '+unit)
        assert len(chart.data.labels)==10


def test_money_sort_does_not_round_even_beyond_current_decimal_context():
    low='99999999999999999.000000000001'
    high='99999999999999999.000000000002'
    result={'store_statistics':(('RUB','a','Меньше',1,1,1,'1',low,low),
                                ('RUB','b','Больше',1,1,1,'1',high,high))}
    chart=data.store_purchase_chart_data(result,'RUB')
    assert chart.labels==('Больше','Меньше')
    assert chart.series[0][1]==(Decimal(high),Decimal(low))


def test_old_saved_result_starts_with_help_collapsed_and_no_automatic_calculation(window,sample_result,tmp_path,monkeypatch):
    from Application.tabs import dataset_statistics_tab as ui
    from Application.analysis_filter import AnalysisFilter
    path=tmp_path/'startup.json';cache.save_result(path,sample_result)
    saved=path.read_bytes()
    monkeypatch.setattr(ui,'CACHE_PATH',path)
    monkeypatch.setattr(ui,'load_filter',AnalysisFilter)
    monkeypatch.setattr(ui.DatasetStatisticsTab,'start',lambda self:pytest.fail('Automatic calculation'))
    tab=LegacyStatisticsTab(window[0])
    try:
        assert tab.displayed_result is not None and path.read_bytes()==saved
        assert tab.findChildren(StatisticsTableExplanation)==[]
        assert all(p.content.isHidden() for p in tab.findChildren(StatisticsTableExplanation))
    finally:tab._shutdown()


def test_zero_money_is_a_real_saved_value(app):
    chart=data.store_purchase_chart_data({'store_statistics':(('RUB','z','Нулевой',1,1,1,'1','0','0'),)},'RUB')
    assert not chart.empty and chart.series==(('RUB',(Decimal(0),)),)
    widget=charts.create_chart(chart)
    try:assert isinstance(widget,charts.StatisticsChart)
    finally:widget.deleteLater()
