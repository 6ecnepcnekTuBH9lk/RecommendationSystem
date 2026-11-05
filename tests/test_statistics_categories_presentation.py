"""Saved direct-category tables share GUI/XLSX/HTML values, labels and order."""
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
from Application.product_resolution import ProductResolver
from Application.statistics_cohorts_core import CohortAggregates
from Application.statistics_cohorts_presentation import cohort_sections,CATEGORY_TITLE,CATEGORY_LEGACY_TEXT,CHANNEL_TITLE
from Application.theme.apply_theme import apply_app_theme
from tests.test_statistics_cohorts_categories import category_dataset,dataset,calculate
from tests.tabs.test_dataset_statistics_tab import app,window
from tests.test_statistics_unified_presentation import Tables

__all__=['app','window','dataset','category_dataset']
pytestmark=pytest.mark.usefixtures('window_settings')


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('dark',[False,True])
def test_saved_category_tables_coverage_and_multilabel_notes_agree_without_source_access(
        category_dataset,window,tmp_path,monkeypatch,legacy,dark):
    result=calculate(category_dataset)
    if legacy:result['cohort_statistics']['policies']['STRICT_COMPLETED'].pop('purchase_categories')
    path=tmp_path/'saved.json';cache.save_result(path,result)
    result=cache.load_result(path);before=deepcopy(result);original=path.read_bytes()
    def forbidden(*a,**kw):pytest.fail('Category presentation attempted source access or recalculation')
    monkeypatch.setattr(statistics,'calculate_statistics',forbidden)
    monkeypatch.setattr(statistics,'iter_export',forbidden)
    monkeypatch.setattr(CohortAggregates,'add',forbidden)
    monkeypatch.setattr(CohortAggregates,'result',forbidden)
    monkeypatch.setattr(ProductResolver,'resolve',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    original_open=Path.open
    root=category_dataset[0]
    def guarded_open(p,*args,**kwargs):
        if p.resolve().is_relative_to(root/'canonical') or p.resolve() in {category_dataset[1],root/'site_categories.csv'}:
            forbidden()
        return original_open(p,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded_open)
    _,tab=window;apply_app_theme(QApplication.instance(),dark)
    tab.render(result);tab.sections.setCurrentIndex(3);tab.customer_sections.setCurrentIndex(4)
    QApplication.processEvents()
    page=tab.customer_sections.widget(4)
    all_sections=cohort_sections(result,compact=True);sections=all_sections[-4:]
    assert sections[0][0]==CATEGORY_TITLE
    assert next(i for i,s in enumerate(all_sections) if s[0]==CHANNEL_TITLE)+4==len(all_sections)-4
    assert all(not cards for _,cards,_,_ in sections)  # No duplicated global KPI tiles.
    labels=[label.text() for label in page.findChildren(QLabel)]
    assert all(title.upper() in labels for title,_,_,_ in sections)
    assert any('правому цензурированию' in text and 'вероятность' in text for text in labels)
    expected=[(h,rows) for _,_,tables,_ in sections for h,rows in tables]
    actual=page.findChildren(QTableWidget)[-3:]
    group=tab.report.pages[3].groups[4]
    blocks=[(i,b) for i,b in enumerate(group.blocks) if b.kind=='table'][-3:]
    target=tmp_path/'statistics.zip';bundle.export_statistics_bundle(result,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==['Статистика.xlsx','Статистика.html'] and archive.testzip() is None
        document=archive.read('Статистика.html').decode('utf-8')
        book=load_workbook(BytesIO(archive.read('Статистика.xlsx')))
    try:
        parsed=Tables(document);excel=list(book['Клиенты'].values)
        for widget,(headers,rows),(i,block) in zip(actual,expected,blocks):
            assert [widget.horizontalHeaderItem(c).text() for c in range(widget.columnCount())]==headers
            values=[[c.text for c in row] for row in report.table_cells(block)]
            assert [[widget.item(r,c).text() for c in range(widget.columnCount())] for r in range(widget.rowCount())]==values
            assert parsed.tables[f'table-3-cohorts-{i}']==(values or [['Нет данных для таблицы.']])
            for row in rows:
                assert any(all(a==pytest.approx(v) if isinstance(v,float) else a==v for a,v in zip(candidate,row)) for candidate in excel)
        if legacy:
            assert CATEGORY_LEGACY_TEXT in str(expected)
        else:
            assert 'Одежда / Рубашка [S]' in document and 'Одежда / Рубашка [TWIN]' in document
            assert any('Покупателей рассматриваемых когорт: 5; со второй покупкой: 4;' in text
                       and 'первой покупки: 1;' in text and 'второй покупки: 1.' in text for text in labels)
            assert any('могут превышать' in text and '100%' in text for text in labels)
            assert any(widget.item(r,0).text()=='Категория не определена' for widget in actual[:2] for r in range(widget.rowCount()))
        assert result==before and path.read_bytes()==original
        assert book.sheetnames==list(report.PAGE_LABELS)
        assert tab.sections.count()==tab.customer_sections.count()==5
    finally:
        book.close()
