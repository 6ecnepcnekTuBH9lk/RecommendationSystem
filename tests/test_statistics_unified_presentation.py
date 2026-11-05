"""Cross-renderer equality of the approved shared presentation, not text smoke checks."""
from copy import deepcopy
from datetime import date, datetime
from html.parser import HTMLParser
import subprocess
import sys

import pytest
from openpyxl import load_workbook
from openpyxl.utils.cell import range_boundaries
from PyQt6.QtWidgets import QTableWidget

from Application import statistics_report_presentation as model
from Application import statistics_export as xlsx
from Application import statistics_html_export as html
from tests.tabs.test_dataset_statistics_tab import app, window, rich_customer_result
from statistics_export_samples import empty_snapshot

__all__=["app","window","rich_customer_result"]
pytestmark=pytest.mark.usefixtures("window_settings")


class Tables(HTMLParser):
    def __init__(self,document):
        super().__init__()
        self.tables={};self.active=None;self.cell=None;self.row=None
        self.feed(document)
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=="table":
            self.active=attrs.get("id")
            if self.active:self.tables[self.active]=[]
        elif tag=="tr" and self.active:
            self.row=[]
        elif tag=="td" and self.active:
            self.cell=""
    def handle_data(self,text):
        if self.cell is not None:self.cell+=text
    def handle_endtag(self,tag):
        if tag=="td" and self.active:
            self.row.append(self.cell);self.cell=None
        elif tag=="tr" and self.active and self.row:
            self.tables[self.active].append(self.row)
        elif tag=="table":
            self.active=None


def test_report_model_is_qt_free_and_uses_stable_groups():
    program="import sys;from Application.statistics_report_presentation import report_presentation;assert 'PyQt6.QtWidgets' not in sys.modules"
    result=subprocess.run([sys.executable,"-B","-c",program],capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors="replace")


@pytest.mark.parametrize("scenario",["rich","legacy","empty"])
def test_shared_gui_html_xlsx_all_tables_kpi_order_values_and_header(
        window,rich_customer_result,scenario,tmp_path,monkeypatch):
    import sqlite3
    from Application import dataset_statistics,statistics_slices
    result=deepcopy(rich_customer_result)
    if scenario=="legacy":
        result.update(buyer_statistics=None,rfm_statistics=None,cohort_statistics=None)
    elif scenario=="empty":
        result=empty_snapshot(result)
    before=deepcopy(result)
    def forbidden(*args,**kwargs):
        pytest.fail("Aggregate renderers must never read sources, INDEXED or recalculate")
    monkeypatch.setattr(sqlite3,"connect",forbidden)
    monkeypatch.setattr(dataset_statistics,"calculate_statistics",forbidden)
    monkeypatch.setattr(statistics_slices,"calculate_cached_statistics",forbidden)
    _,tab=window
    tab.render(result)
    report=tab.report
    assert [p.title for p in report.pages]==list(model.PAGE_LABELS)
    assert [g.key for g in report.pages[3].groups]==["overview","purchases","money","rfm","cohorts"]
    path=tmp_path/"statistics.xlsx"
    xlsx.export_statistics(result,path)
    document=html.build_statistics_html(result)
    parsed=Tables(document)
    book=load_workbook(path)
    expected_html=set()
    for page_index,page in enumerate(report.pages):
        gui=tab.sections.widget(page_index)
        for group_index,group in enumerate(page.groups):
            target=tab.customer_sections.widget(group_index) if page.key=="customers" else gui
            tables=target.findChildren(QTableWidget)
            blocks=[(i,b) for i,b in enumerate(group.blocks) if b.kind=="table"]
            assert len(tables)==len(blocks)
            for table,(block_index,block) in zip(tables,blocks):
                expected=[[c.text for c in row] for row in model.table_cells(block)]
                assert [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]==list(block.headers)
                actual=[[table.item(r,c).text() for c in range(table.columnCount())] for r in range(table.rowCount())]
                assert actual==expected
                identifier=f"table-{page_index}-{group.key}-{block_index}"
                expected_html.add(identifier)
                assert parsed.tables[identifier]==expected if expected else parsed.tables[identifier]==[["Нет данных для таблицы."]]
        sheet=book[page.title]
        expected_tables=[(tuple(("Параметр отчёта","Значение")),tuple(
            (model.cell_presentation(a),model.cell_presentation(b)) for a,b in report.header)),
            (tuple(("Показатель","Значение")),tuple(
             (model.cell_presentation(label),cell) for label,cell in model.kpi_cells(model.ReportBlock("cards",values=report.global_cards))))]
        for group in page.groups:
            for block in group.blocks:
                if block.kind=="cards":
                    expected_tables.append((("Показатель","Значение"),tuple(
                        (model.cell_presentation(label),cell) for label,cell in model.kpi_cells(block))))
                elif block.kind=="table":
                    expected_tables.append((block.headers,model.table_cells(block)))
        expected_tables=[t for t in expected_tables if len(t[0])>1 and t[1]]
        actual_tables=list(sheet.tables.values())
        assert len(actual_tables)==len(expected_tables)
        for table,(headers,rows) in zip(actual_tables,expected_tables):
            left,top,right,bottom=range_boundaries(table.ref)
            assert tuple(sheet.cell(top,c).value for c in range(left,right+1))==tuple(headers)
            assert bottom-top==len(rows)
            for r,values in enumerate(rows,top+1):
                for c,expected in enumerate(values,left):
                    cell=sheet.cell(r,c)
                    assert cell.data_type!="f"
                    if expected.kind=="number":
                        assert cell.data_type=="n"
                        assert cell.value==pytest.approx(float(expected.value))
                        assert cell.number_format==expected.number_format
                    elif expected.kind=="date":
                        assert isinstance(cell.value,(date,datetime))
                        assert cell.value.date()==expected.value if isinstance(cell.value,datetime) else cell.value==expected.value
                    else:
                        assert cell.value==expected.text
        # Compact GUI sections are the entire report body, not supplemented by legacy method tables.
        values={str(c.value) for row in sheet for c in row if c.value is not None}
        assert "Пояснение RFM" not in values and "Параметр когортного анализа" not in values
    assert book.sheetnames==list(model.PAGE_LABELS)
    book.close()
    assert set(parsed.tables)==expected_html
    assert result==before


def test_html_has_all_shared_charts_rfm_variants_zero_none_partial_and_no_dto(rich_customer_result):
    result=deepcopy(rich_customer_result)
    result["UNSAFE_FULL_DTO"]="SYNTH_DO_NOT_SERIALIZE"
    report=model.report_presentation(result)
    specs=html.chart_specs(result)
    expected_charts=[b for p in report.pages for g in p.groups for b in g.blocks if b.kind in ("chart","rfm")]
    assert len(specs)==len(expected_charts)
    for spec,block in zip(specs,expected_charts):
        data=block.data if block.kind=="chart" else next(data for w,c,data in block.choices if (w,c)==("LAST_12M","RUB"))
        assert spec["title"]==data.title and spec["labels"]==list(data.display_labels)
        if not data.empty:
            assert [t["exact"] for t in spec["data"]]==[
                [None if v is None else str(v) for v in values] for _,values in data.series]
        if block.kind=="rfm":
            assert set(spec["variants"])=={w+"|"+c for w,c,_ in block.choices}
    document=html.build_statistics_html(result)
    assert "SYNTH_DO_NOT_SERIALIZE" not in document
    assert document.count('role="tablist"')==2
    assert all('id="section-'+str(i)+'"' in document for i in range(5))
    assert all('id="customer-section-'+str(i)+'"' in document for i in range(5))
    assert "connect-src 'none'" in document and '<script src=' not in document
    assert "repeating-linear-gradient" in html.heatmap_style(type("Cell",(),dict(value=50.,status="PARTIAL"))())
    assert html.heatmap_style(type("Cell",(),dict(value=0.,status="FULL"))())!=html.heatmap_style(type("Cell",(),dict(value=None,status="UNAVAILABLE"))())
    rates=next(s for s in specs if s["percentage"])
    assert any(None in t["exact"] for t in rates["data"])

