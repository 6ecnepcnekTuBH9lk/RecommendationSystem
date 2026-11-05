"""Presentation-only acceptance: quantile units, calendar dates and three renderers."""
from copy import deepcopy
from datetime import date
from decimal import Decimal
from io import BytesIO
from zipfile import ZipFile

import pytest
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries
from PyQt6.QtWidgets import QLabel,QTableWidget

from Application import statistics_cache as cache,statistics_report_presentation as report
from Application.statistics_presentation import interval_bounds,interval_text
from Application.theme.apply_theme import apply_app_theme
from tests.tabs.test_dataset_statistics_tab import app,window,rich_customer_result
from tests.test_statistics_unified_presentation import Tables

__all__=["app","window","rich_customer_result"]
pytestmark=pytest.mark.usefixtures("window_settings")
QUANTILES=("Порог 20%","Порог 40%","Порог 60%","Порог 80%")
HEADERS=("Окно","Валюта","Компонент","Основание порогов: покупателей",*QUANTILES)


@pytest.mark.parametrize("currency",["RUB","KZT"])
@pytest.mark.parametrize("component,values",[
    ("R",(183,430,743,1096)),
    ("F",(0,1,2,5)),
    ("M",tuple(Decimal(v) for v in ("12784.00","29500.00","40000.125","59123.67"))),
])
def test_quantile_positions_are_not_the_unit_of_thresholds(currency,component,values):
    source=tuple(report.ReportNumber(v) if component=="M" else v for v in values)
    row=("48 месяцев",currency,component,12,*source)
    block=report.ReportBlock("table",headers=HEADERS,rows=(row,))
    before=deepcopy(block)
    cells=report.table_cells(block)[0]
    assert block==before and block.headers[4:]==QUANTILES
    assert cells[1].text==currency
    for cell,value in zip(cells[4:],values):
        assert cell.value==value and cell.kind=="number"
        assert "%" not in cell.text and "%" not in cell.number_format
        assert cell.number_format==("#,##0.00" if component=="M" else "#,##0")
    assert [c.text for c in cells[4:]]==(
        ["183","430","743","1 096"] if component=="R" else
        ["0","1","2","5"] if component=="F" else
        ["12 784.00","29 500.00","40 000.12","59 123.67"])


def test_missing_thresholds_stay_missing_and_real_shares_stay_percentages():
    row=("12 месяцев","KZT","M",0,None,None,None,None)
    cells=report.table_cells(report.ReportBlock("table",headers=HEADERS,rows=(row,)))[0]
    assert all(c.value is None and c.text=="—" and c.kind=="empty" for c in cells[4:])
    for headers,row,column in (
        (("Сегмент","Покупателей","Доля валютной базы, %"),("Лидеры",2,50.),2),
        (("Показатель сравнения","Значение"),("Изменение числа заказов, %",12.5),1),
        (("Когорта","Удержание, %"),("2026-01",0.),1),
    ):
        c=report.table_cells(report.ReportBlock("table",headers=headers,rows=(row,)))[0][column]
        assert c.value==row[column] and c.text.endswith(" %") and '" %"' in c.number_format


@pytest.mark.parametrize("start,end,expected",[
    ("2026-08-31T00:00:00Z","2026-09-01T00:00:00Z",("31.08.2026","31.08.2026")),
    ("2026-08-31T00:00:00+14:00","2026-09-01T00:00:00+14:00",("31.08.2026","31.08.2026")),
    ("2026-08-31T00:00:00Z","2026-08-31T12:30:00Z",("31.08.2026 00:00","31.08.2026 12:30 (не включая)")),
    ("2026-08-31T12:30:00Z","2026-09-01T00:00:00Z",("31.08.2026 12:30","01.09.2026 00:00 (не включая)")),
    ("2026-08-31T12:30:05Z","2026-08-31T12:30:06.123456Z",("31.08.2026 12:30:05","31.08.2026 12:30:06.123456 (не включая)")),
    ("2026-08-31T12:30:00+03:00","2026-08-31T12:30:00Z",("31.08.2026 12:30 +0300","31.08.2026 12:30 +0000 (не включая)")),
])
def test_date_presentation_keeps_actual_half_open_bounds(start,end,expected):
    assert interval_bounds(start,end)==expected
    assert interval_text(start,end)==" — ".join(expected)
    assert (cache.parse_timestamp(start).isoformat(),cache.parse_timestamp(end).isoformat())==(
        start.replace("Z","+00:00"),end.replace("Z","+00:00"))


@pytest.mark.parametrize("dark",[False,True])
def test_saved_snapshot_gui_xlsx_html_agree_without_sources_or_recalculation(
        window,rich_customer_result,tmp_path,monkeypatch,dark):
    from Application import dataset_statistics,statistics_slices,statistics_bundle_export
    from Application.statistics_rfm_core import RfmAggregates
    import sqlite3
    from PyQt6.QtWidgets import QApplication

    result=deepcopy(rich_customer_result)
    result["calculated_at"]="2026-08-31T23:45:12Z"
    result["analysis_filter"]["nomenclature_types"]=["Тестовый вид",None]
    result["analysis_filter"]["collections"]=["Тестовый сезон"]
    for source in result["coverage"]:source["updated"]="2026-08-31T23:45:12Z"
    saved=tmp_path/"snapshot.json";cache.save_result(saved,result)
    result=cache.load_result(saved);assert result is not None
    before=deepcopy(result);saved_bytes=saved.read_bytes()
    def forbidden(*args,**kwargs):pytest.fail("Presentation must only consume the saved snapshot")
    monkeypatch.setattr(dataset_statistics,"calculate_statistics",forbidden)
    monkeypatch.setattr(dataset_statistics,"iter_export",forbidden)
    monkeypatch.setattr(statistics_slices,"calculate_cached_statistics",forbidden)
    monkeypatch.setattr(RfmAggregates,"result",forbidden)
    monkeypatch.setattr(sqlite3,"connect",forbidden)
    _,tab=window
    apply_app_theme(QApplication.instance(),dark);tab.render(result)
    tab.sections.setCurrentIndex(3);tab.customer_sections.setCurrentIndex(3)
    QApplication.processEvents()
    target=tmp_path/"report.zip"
    statistics_bundle_export.export_statistics_bundle(result,target)
    with ZipFile(target) as archive:
        assert archive.namelist()==["Статистика.xlsx","Статистика.html"] and archive.testzip() is None
        book=load_workbook(BytesIO(archive.read("Статистика.xlsx")))
        document=archive.read("Статистика.html").decode("utf-8")
    assert book.sheetnames==list(report.PAGE_LABELS)
    assert tab.sections.count()==tab.customer_sections.count()==5
    assert all(f'id="section-{i}"' in document and f'id="customer-section-{i}"' in document for i in range(5))
    assert "data-rfm-window=" in document and "data-rfm-currency=" in document
    assert "heatmap" in document
    assert dict(tab.report.header)["Дата расчёта"]=="31.08.2026"
    assert "Дата расчета: 31.08.2026" in tab.description.text()
    labels="\n".join(str(v) for h,v in tab.report.header)
    for value in ("Тестовый вид","Тестовый сезон","Не указано"):
        assert value in labels and value in document
    assert all("UTC" not in str(h)+str(v) and "00:00:00" not in str(v) for h,v in tab.report.header)
    assert "UTC" not in document.split("<script")[0]
    assert "Дата и время" not in document.split("<script")[0]+tab.description.text()
    assert any(h=="Дата отсечения покупок и RFM" and v==date(2026,8,31) for h,v in tab.report.header)
    assert any(v=="Снимок данных от 31.08.2026" for h,v in tab.report.header)
    parsed=Tables(document);table_count=0
    for page_index,page in enumerate(tab.report.pages):
        for group_index,group in enumerate(page.groups):
            widget=tab.customer_sections.widget(group_index) if page.key=="customers" else tab.sections.widget(page_index)
            gui_tables=iter(widget.findChildren(QTableWidget))
            for block_index,block in enumerate(group.blocks):
                if block.kind!="table":continue
                table_count+=1;gui=next(gui_tables);cells=report.table_cells(block)
                actual=[[gui.item(r,c).text() for c in range(gui.columnCount())] for r in range(gui.rowCount())]
                assert actual==[[c.text for c in row] for row in cells]
                html_rows=parsed.tables[f"table-{page_index}-{group.key}-{block_index}"]
                assert html_rows==actual if actual else html_rows==[["Нет данных для таблицы."]]
                if tuple(block.headers[4:])==QUANTILES:
                    for raw,row in zip(block.rows,cells):
                        assert raw[1] in ("RUB","KZT") and raw[2] in ("R","F","M")
                        assert all("%" not in c.text and "%" not in c.number_format for c in row[4:])
            assert next(gui_tables,None) is None
    assert table_count==68
    # Workbook tables retain values/types/units, independent of rendered text.
    for sheet in book:
        for table in sheet.tables.values():
            left,top,right,bottom=range_boundaries(table.ref)
            headers=[sheet.cell(top,c).value for c in range(left,right+1)]
            if tuple(headers[4:])==QUANTILES:
                for row in sheet.iter_rows(min_row=top+1,max_row=bottom,min_col=left+4,max_col=right):
                    assert all(c.data_type in ("n","s") and "%" not in c.number_format for c in row)
            assert "UTC" not in " ".join(str(h) for h in headers)
    book.close()
    controls=tab.customer_sections.widget(3).findChildren(QLabel)
    assert controls and result==before and saved.read_bytes()==saved_bytes
