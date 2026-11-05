"""Three channel tables/KPIs from one saved snapshot in GUI, XLSX and HTML."""
from copy import deepcopy
from io import BytesIO
import sqlite3
from zipfile import ZipFile

import pytest
from openpyxl import load_workbook
from PyQt6.QtWidgets import QApplication,QLabel,QTableWidget

from Application import statistics_cache as cache,statistics_bundle_export as bundle
from Application import statistics_report_presentation as report,dataset_statistics as statistics
from Application.statistics_cohorts_core import CohortAggregates
from Application.statistics_cohorts_presentation import cohort_sections,CHANNEL_NOTE,CHANNEL_LEGACY_TEXT,CHANNEL_TITLE
from Application.theme.apply_theme import apply_app_theme
from tests.test_statistics_cohorts_channels import channel_dataset,dataset,calculate
from tests.tabs.test_dataset_statistics_tab import app,window
from tests.test_statistics_unified_presentation import Tables

__all__=['app','window','dataset','channel_dataset']
pytestmark=pytest.mark.usefixtures('window_settings')


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('dark',[False,True])
def test_saved_channels_kpis_and_three_tables_agree_without_sources_or_calculation(
        channel_dataset,window,tmp_path,monkeypatch,legacy,dark):
    result=calculate(channel_dataset)
    strict=result['cohort_statistics']['policies']['STRICT_COMPLETED']
    # Equal display names must remain distinguishable by stable channel IDs.
    for row in strict['purchase_channels']['channels']:
        if row['external_id'] in ('A','B'):row['name']='Same'
    if legacy:strict.pop('purchase_channels')
    path=tmp_path/'saved.json';cache.save_result(path,result)
    result=cache.load_result(path);before=deepcopy(result);original=path.read_bytes()
    def forbidden(*a,**kw):pytest.fail('Channel presentation attempted recalculation or source access')
    monkeypatch.setattr(statistics,'calculate_statistics',forbidden)
    monkeypatch.setattr(statistics,'iter_export',forbidden)
    monkeypatch.setattr(CohortAggregates,'add',forbidden)
    monkeypatch.setattr(CohortAggregates,'result',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    _,tab=window
    apply_app_theme(QApplication.instance(),dark)
    tab.render(result);tab.sections.setCurrentIndex(3);tab.customer_sections.setCurrentIndex(4)
    QApplication.processEvents()
    page=tab.customer_sections.widget(4)
    all_sections=cohort_sections(result,compact=True)
    start=next(i for i,s in enumerate(all_sections) if s[0]==CHANNEL_TITLE)
    sections=all_sections[start:start+4]
    labels=[label.text() for label in page.findChildren(QLabel)]
    assert all(title.upper() in labels for title,_,_,_ in sections)
    assert any(CHANNEL_NOTE in text for text in labels)
    expected=[(h,rows) for _,_,tables,_ in sections for h,rows in tables]
    actual=page.findChildren(QTableWidget)[-6:-3]
    assert len(expected)==len(actual)==3
    model_group=tab.report.pages[3].groups[4]
    table_blocks=[(i,b) for i,b in enumerate(model_group.blocks) if b.kind=='table'][-6:-3]
    target=tmp_path/'statistics.zip';bundle.export_statistics_bundle(result,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==['Статистика.xlsx','Статистика.html'] and archive.testzip() is None
        document=archive.read('Статистика.html').decode('utf-8')
        book=load_workbook(BytesIO(archive.read('Статистика.xlsx')))
    parsed=Tables(document)
    try:
        excel=list(book['Клиенты'].values)
        for widget,(headers,rows),(i,block) in zip(actual,expected,table_blocks):
            assert [widget.horizontalHeaderItem(c).text() for c in range(widget.columnCount())]==headers
            cells=report.table_cells(block)
            values=[[c.text for c in row] for row in cells]
            assert [[widget.item(r,c).text() for c in range(widget.columnCount())] for r in range(widget.rowCount())]==values
            assert parsed.tables[f'table-3-cohorts-{i}']==(values or [['Нет данных для таблицы.']])
            for row in rows:
                assert any(all(a==pytest.approx(v) if isinstance(v,float) else a==v for a,v in zip(candidate,row))
                           for candidate in excel)
        for label,value in sections[0][1]:
            assert label in labels
            assert any(row[0]==label and (row[1]==pytest.approx(value) if isinstance(value,float) else row[1]==value)
                       for row in excel)
        if legacy:
            assert CHANNEL_LEGACY_TEXT in str(expected)
            assert 'Нет данных' in actual[0].item(0,0).text()
        else:
            assert dict(sections[0][1])['Доля сохранивших канал среди покупателей с двумя известными идентификаторами, %']==50
            assert 'Same [A]' in document and 'Same [B]' in document
            assert any(widget.item(r,0).text()=='Не указан' for widget in actual[:2] for r in range(widget.rowCount()))
        assert book.sheetnames==list(report.PAGE_LABELS)
        assert result==before and path.read_bytes()==original
        assert tab.sections.count()==tab.customer_sections.count()==5
    finally:
        book.close()
