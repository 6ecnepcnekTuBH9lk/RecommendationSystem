"""Saved co-purchase KPI/tables are identical in GUI/XLSX/HTML, including legacy/empty."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sqlite3
from zipfile import ZipFile

import pytest
from openpyxl import load_workbook
from PyQt6.QtWidgets import QApplication,QLabel,QTableWidget

from Application import dataset_statistics as statistics,statistics_cache as cache,statistics_bundle_export as bundle
from Application import statistics_report_presentation as report
from Application.statistics_copurchases import CategoryCoPurchases
from Application.statistics_copurchases_presentation import COPURCHASE_TITLE,COPURCHASE_LEGACY_TEXT
from Application.statistics_presentation import grouped_buyer_sections
from Application.statistics_charts import StatisticsChart
from Application.theme.apply_theme import apply_app_theme
from Application.analysis_filter import AnalysisFilter
from tests.test_statistics_cohorts_categories import category_dataset,dataset,calculate
from tests.tabs.test_dataset_statistics_tab import app,window,sample_result,card_titles
from tests.test_statistics_unified_presentation import Tables

__all__=['app','window','sample_result','dataset','category_dataset']
pytestmark=pytest.mark.usefixtures('window_settings')


@pytest.mark.parametrize('scenario',['ready','legacy','empty'])
@pytest.mark.parametrize('dark',[False,True])
def test_copurchase_kpis_three_tables_conditional_rates_and_missing_states_match(
        category_dataset,window,tmp_path,monkeypatch,scenario,dark):
    result=calculate(category_dataset,AnalysisFilter('2026-02-01','2026-03-31') if scenario=='empty' else
                     AnalysisFilter('2026-01-01','2026-08-31'))
    if scenario=='legacy':result['buyer_statistics']['policies']['STRICT_COMPLETED'].pop('category_copurchases')
    path=tmp_path/'saved.json';cache.save_result(path,result);result=cache.load_result(path)
    before=deepcopy(result);original=path.read_bytes()
    def forbidden(*a,**kw):pytest.fail('Saved co-purchase report attempted source read or recalculation')
    monkeypatch.setattr(statistics,'calculate_statistics',forbidden)
    monkeypatch.setattr(statistics,'iter_export',forbidden)
    monkeypatch.setattr(CategoryCoPurchases,'add',forbidden)
    monkeypatch.setattr(CategoryCoPurchases,'result',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    root=category_dataset[0];original_open=Path.open
    def guarded_open(p,*args,**kwargs):
        if p.resolve().is_relative_to(root/'canonical') or p.resolve() in {category_dataset[1],root/'site_categories.csv'}:
            forbidden()
        return original_open(p,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded_open)
    _,tab=window;apply_app_theme(QApplication.instance(),dark)
    tab.render(result);tab.sections.setCurrentIndex(3);tab.customer_sections.setCurrentIndex(1)
    QApplication.processEvents()
    sections=[s for group,s in grouped_buyer_sections(result,compact=True) if group=='purchases'][-4:]
    assert sections[0][0]==COPURCHASE_TITLE
    expected=[(h,rows) for _,_,tables,_ in sections for h,rows in tables]
    assert len(expected)==3
    page=tab.customer_sections.widget(1);labels=[label.text() for label in page.findChildren(QLabel)]
    assert all(title.upper() in labels for title,_,_,_ in sections)
    # The original frequency/policy charts remain; no co-purchase graph appears.
    assert len(page.findChildren(StatisticsChart))==(0 if scenario=='empty' else 2)
    widgets=page.findChildren(QTableWidget)[-3:]
    group=tab.report.pages[3].groups[1]
    blocks=[(i,b) for i,b in enumerate(group.blocks) if b.kind=='table'][-3:]
    target=tmp_path/'report.zip';bundle.export_statistics_bundle(result,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==['Статистика.xlsx','Статистика.html'] and archive.testzip() is None
        document=archive.read('Статистика.html').decode('utf-8')
        book=load_workbook(BytesIO(archive.read('Статистика.xlsx')))
    try:
        parsed=Tables(document);excel=list(book['Клиенты'].values)
        for widget,(headers,rows),(i,block) in zip(widgets,expected,blocks):
            assert [widget.horizontalHeaderItem(c).text() for c in range(widget.columnCount())]==headers
            values=[[c.text for c in row] for row in report.table_cells(block)]
            assert [[widget.item(r,c).text() for c in range(widget.columnCount())] for r in range(widget.rowCount())]==values
            assert parsed.tables[f'table-3-purchases-{i}']==(values or [['Нет данных для таблицы.']])
            for row in rows:
                assert any(all(a==pytest.approx(v) if isinstance(v,float) else a==v for a,v in zip(candidate,row)) for candidate in excel)
        if scenario=='ready':
            cards=dict(sections[0][1])
            assert len(cards)==4 and cards['Заказы с двумя и более категориями']==5
            assert cards['Покупатели с многокатегорийными заказами']==3
            assert cards['Доля многокатегорийных заказов, %']==pytest.approx(500/11)
            assert card_titles(page).count('Заказы (подтверждённые покупки)')==1
            for label,value in cards.items():
                assert label in card_titles(page)
                assert any(row[0]==label and (row[1]==pytest.approx(value) if isinstance(value,float) else row[1]==value) for row in excel)
            assert 'неполные данные' in document
            assert 'Одежда / Рубашка [S]' in document and 'Одежда / Рубашка [TWIN]' in document
        else:
            assert sections[0][1]==[]  # Missing/empty data are not substituted by zero KPI values.
            assert all(table.item(0,0).text()=='Нет данных' for table in widgets)
            if scenario=='legacy':assert COPURCHASE_LEGACY_TEXT in str(expected)
        assert book.sheetnames==list(report.PAGE_LABELS)
        assert tab.sections.count()==tab.customer_sections.count()==5
        assert result==before and path.read_bytes()==original
    finally:
        book.close()


def test_old_result_without_buyer_analytics_does_not_receive_shifted_chart(window,sample_result):
    _,tab=window;result=deepcopy(sample_result)
    result.pop('buyer_statistics',None);tab.render(result)
    page=tab.customer_sections.widget(1)
    assert not page.findChildren(StatisticsChart)
    assert any(COPURCHASE_LEGACY_TEXT in table.item(r,c).text()
               for table in page.findChildren(QTableWidget) for r in range(table.rowCount()) for c in range(table.columnCount()))
