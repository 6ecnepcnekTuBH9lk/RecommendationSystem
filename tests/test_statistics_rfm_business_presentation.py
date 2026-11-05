"""Shared saved RFM business tables: exact GUI/HTML cells and typed XLSX in ZIP2."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from decimal import Decimal
import sqlite3
from zipfile import ZipFile

import pytest
from openpyxl import load_workbook
from PyQt6.QtWidgets import QApplication,QLabel,QTableWidget

from Application import dataset_statistics as statistics,statistics_cache as cache,statistics_bundle_export as bundle
from Application import statistics_report_presentation as report
from Application.statistics_rfm_core import RfmAggregates,WINDOWS
from Application.statistics_rfm_presentation import rfm_sections,RFM_BUSINESS_TITLE,RFM_BUSINESS_LEGACY_TEXT,SEGMENT_LABELS
from Application.statistics_charts import StatisticsChart
from Application.theme.apply_theme import apply_app_theme
from tests.test_statistics_rfm_business import segment_population
from tests.test_statistics_rfm_core import AS_OF,SPACE,currency
from tests.test_statistics_rfm_integration import dataset,calculate,SELECTION
from tests.tabs.test_dataset_statistics_tab import app,window
from tests.test_statistics_unified_presentation import Tables

__all__=['app','window','dataset']
pytestmark=pytest.mark.usefixtures('window_settings')


@pytest.mark.parametrize('scenario',['ready','partial','legacy'])
@pytest.mark.parametrize('dark',[False,True])
def test_saved_business_segments_three_windows_two_currencies_and_actions_match_zip(
        dataset,window,tmp_path,monkeypatch,scenario,dark):
    result=calculate(dataset)
    order_coverage=next(s for s in result['coverage'] if s['source']=='Orders')
    if scenario=='partial':
        order_coverage['intervals']=[('2025-09-02T00:00:00+00:00','2026-09-01T00:00:00+00:00')]
        result['buyer_statistics']=result['cohort_statistics']=None
    coverage=[(cache.parse_timestamp(a),cache.parse_timestamp(b)) for a,b in order_coverage['intervals']]
    core=RfmAggregates(AS_OF,coverage,identity_space=SPACE)
    for namespace in ('offline1C','kanzlerKz'):
        for f in segment_population(namespace):core.add(f)
    result['rfm_statistics']=core.result()
    if scenario=='legacy':
        for name in WINDOWS[:3]:
            for unit in ('RUB','KZT'):currency(result['rfm_statistics'],name,unit).pop('segment_business')
    path=tmp_path/'saved.json';cache.save_result(path,result);result=cache.load_result(path)
    before=deepcopy(result);original=path.read_bytes()
    def forbidden(*a,**kw):pytest.fail('Saved RFM business presentation attempted source access or recalculation')
    monkeypatch.setattr(statistics,'calculate_statistics',forbidden)
    monkeypatch.setattr(statistics,'iter_export',forbidden)
    monkeypatch.setattr(RfmAggregates,'add',forbidden)
    monkeypatch.setattr(RfmAggregates,'result',forbidden)
    monkeypatch.setattr(sqlite3,'connect',forbidden)
    original_open=Path.open
    def guarded_open(p,*args,**kwargs):
        if p.resolve().is_relative_to(dataset[0]/'canonical') or p.resolve() in {dataset[1],dataset[0]/'site_categories.csv'}:
            forbidden()
        return original_open(p,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded_open)
    _,tab=window;apply_app_theme(QApplication.instance(),dark)
    tab.render(result);tab.sections.setCurrentIndex(3);tab.customer_sections.setCurrentIndex(3)
    QApplication.processEvents()
    sections=rfm_sections(result,compact=True)
    assert sections[2][0]=='RFM — сегменты: подтверждённые покупки'
    assert sections[3][0]==RFM_BUSINESS_TITLE and sections[5][0]=='RFM — баллы и пороги: подтверждённые покупки'
    page=tab.customer_sections.widget(3)
    labels=[label.text() for label in page.findChildren(QLabel)]
    assert all(s[0].upper() in labels for s in sections[3:5])
    assert len(page.findChildren(StatisticsChart))==1
    assert tab.sections.count()==tab.customer_sections.count()==5
    headers,rows=sections[3][2][0]
    widgets=[table for table in page.findChildren(QTableWidget)
             if table.horizontalHeaderItem(0).text()==headers[0] and table.columnCount()==len(headers)
             and table.horizontalHeaderItem(1).text()==headers[1]]
    # Match the business table by its unique width (or its legacy message).
    widget=next(w for w in widgets if w.columnCount()==12 or w.item(0,1).text()==RFM_BUSINESS_LEGACY_TEXT)
    blocks=list(enumerate(tab.report.pages[3].groups[3].blocks))
    i,block=next((i,b) for i,b in blocks if b.kind=='table' and list(b.headers)==headers)
    values=[[cell.text for cell in row] for row in report.table_cells(block)]
    assert [[widget.item(r,c).text() for c in range(widget.columnCount())] for r in range(widget.rowCount())]==values
    target=tmp_path/'rfm-business.zip';bundle.export_statistics_bundle(result,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==['Статистика.xlsx','Статистика.html'] and archive.testzip() is None
        document=archive.read('Статистика.html').decode('utf-8')
        book=load_workbook(BytesIO(archive.read('Статистика.xlsx')))
    try:
        parsed=Tables(document);excel=list(book['Клиенты'].values)
        assert parsed.tables[f'table-3-rfm-{i}']==values
        for row in rows:
            assert any(all(Decimal(str(a))==pytest.approx(v) if isinstance(v,Decimal) else
                           a==pytest.approx(v) if isinstance(v,float) else a==v for a,v in zip(candidate,row))
                       for candidate in excel)
        actions=sections[4][2][0][1]
        assert len(actions)==7 and [r[0] for r in actions]==[v[0] for v in SEGMENT_LABELS.values()]
        for row in actions:assert any(tuple(candidate[:2])==tuple(row) for candidate in excel)
        assert 'автоматически назначенными' in document
        assert 'не доказывает новизну' in document and 'не означают гарантированную потерю' in document
        assert 'data-rfm-window=' in document and 'data-rfm-currency=' in document
        if scenario=='legacy':assert RFM_BUSINESS_LEGACY_TEXT in document
        else:
            assert len(rows)==42 and len(headers)==12
            assert all(type(row[c]) is Decimal or row[c]=='Нет данных' for row in rows for c in (7,8,9,11))
            assert 'Окно' in headers and 'Валюта' in headers
        if scenario=='partial':
            assert 'частичное' in document and 'не даёт достоверного поведенческого статуса' in document
            assert all(row[3]==0 or row[2]==SEGMENT_LABELS['UNSCORABLE'][0] for row in rows)
        assert result==before and path.read_bytes()==original
    finally:book.close()


def test_old_business_snapshot_opens_on_startup_without_automatic_recalculation(dataset,window,tmp_path,monkeypatch):
    from Application.tabs import dataset_statistics_tab as ui
    result=calculate(dataset)
    for name in WINDOWS[:3]:
        for unit in ('RUB','KZT'):currency(result['rfm_statistics'],name,unit).pop('segment_business')
    path=tmp_path/'legacy-startup.json';cache.save_result(path,result)
    original=path.read_bytes()
    monkeypatch.setattr(ui,'CACHE_PATH',path)
    monkeypatch.setattr(ui,'load_filter',lambda:SELECTION)
    monkeypatch.setattr(ui.DatasetStatisticsTab,'start',lambda self:pytest.fail('Automatic RFM recalculation'))
    tab=ui.DatasetStatisticsTab(window[0])
    try:
        assert tab.displayed_result is not None and path.read_bytes()==original
        assert RFM_BUSINESS_LEGACY_TEXT in str(rfm_sections(tab.displayed_result,compact=True))
    finally:tab._shutdown()
