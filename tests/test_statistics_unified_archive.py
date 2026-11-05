"""Synthetic unified ZIP: safe schemas, exact links, splitting and atomicity."""
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from io import BytesIO
from zipfile import ZipFile

from openpyxl import load_workbook
import pytest

from Application import statistics_detail_backend as backend
from Application import statistics_detail_xlsx as xlsx
from Application.analysis_filter import AnalysisFilter
from Application.dataset_statistics import calculate_statistics
from Application.statistics_detail_contract import OrderMode
from Application.statistics_detail_storage import DetailExportError
from tests.test_statistics_detail_backend import job, AS_OF, PERIOD, SYNTH_EMAIL, SYNTH_PHONE, clean, raw_order, snapshot

__all__=["job"]


def aggregate(job):
    return asdict(calculate_statistics(raw_root=job.root,catalog_path=job.data[1],
        analysis_filter=AnalysisFilter(start_date=PERIOD.start_date.isoformat(),end_date=AS_OF.isoformat()),
        calculated_at=datetime(2026,9,1,tzinfo=timezone.utc)))


def run(job,result,**kwargs):
    return backend.export_detail_statistics(raw_root=job.root,destination=job.target,
        temporary_root=job.private,as_of_date=AS_OF,event_period=PERIOD,report_result=result,**kwargs)


def members(job):
    tables={}
    with ZipFile(job.target) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert name.endswith((".xlsx",".html"))
            if name.startswith("Данные покупателей/"):
                book=load_workbook(BytesIO(archive.read(name)),read_only=True)
                sheet=book.active
                rows=list(sheet.values)
                tables[name]=(rows[0],rows[1:])
                book.close()
    return tables


def test_unified_archive_five_sheets_privacy_types_links_and_source_immutability(job):
    result=aggregate(job)
    before=snapshot(job)
    exported=run(job,result)
    with ZipFile(job.target) as archive:
        assert len(archive.namelist())==5
        assert archive.namelist()[:2]==["Статистика.xlsx","Статистика.html"]
        book=load_workbook(BytesIO(archive.read("Статистика.xlsx")),read_only=True)
        assert book.sheetnames==["Действия","Заказы","Товары","Клиенты","Техническая информация"]
        book.close()
        html=archive.read("Статистика.html").decode()
        assert "data-rfm-window=" in html and "customer-section-4" in html
        assert result["calculated_at"][:10] in html or "01.09.2026" in html
        for name in archive.namelist():
            if name.endswith(".xlsx"):
                with ZipFile(BytesIO(archive.read(name))) as inner:
                    text=" ".join(inner.read(p).decode(errors="ignore") for p in inner.namelist())
                    assert SYNTH_EMAIL not in text and SYNTH_PHONE not in text
                    assert "SYNTH_FIRST" not in text and "SYNTH_LAST" not in text
    detail=members(job)
    schemas={name.split("/")[-1].split("_")[0]:value for name,value in detail.items()}
    ch,clients=schemas["Клиенты"];oh,orders=schemas["Заказы"];lh,lines=schemas["Позиции заказов"]
    assert not {"email","mobile_phone","last_name","first_name","source_customer_id","birth_date"}.intersection(ch)
    c=dict(zip(ch,clients[0]));o=dict(zip(oh,orders[0]));line=dict(zip(lh,lines[0]))
    assert c["customer_id"]==o["customer_id"]==line["customer_id"]
    assert c["customer_id"].startswith("C") and len(c["customer_id"])==65
    assert o["order_id"]==line["order_id"] and o["order_id"].startswith("O")
    assert line["product_id"]=="000001-size" and isinstance(line["product_id"],str)
    assert isinstance(line["price_of_line"],(int,float))
    assert exported.manifest["rows"]==dict(clients=1,orders=1,order_lines=1)
    assert snapshot(job)==before
    clean(job)


def test_split_boundaries_repeat_headers_all_rows_links_unicode_and_no_formulas(job):
    orders=[raw_order(f"00000000000000000000000000{i}",f"2026-01-{i+1:02d}",statuses=("F","CP","Return"))
            for i in range(7)]
    for order in orders:
        order["firstAction"]["channel"]["name"]="=Канал Юникод"
    job.configure(orders)
    result=aggregate(job)
    exported=run(job,result,max_rows_per_file=3)
    detail=members(job)
    all_orders=[];all_lines=[];clients=[]
    for name,(header,rows) in detail.items():
        assert len(rows)<=2 and rows
        if "Клиенты_" in name:
            assert tuple(header)==xlsx.SAFE_CLIENT_COLUMNS;clients.extend(dict(zip(header,r)) for r in rows)
        elif "Позиции заказов_" in name:
            assert tuple(header)==xlsx.SAFE_LINE_COLUMNS;all_lines.extend(dict(zip(header,r)) for r in rows)
        else:
            assert tuple(header)==xlsx.SAFE_ORDER_COLUMNS;all_orders.extend(dict(zip(header,r)) for r in rows)
    assert len(all_orders)==7 and len(all_lines)==21 and len(clients)==1
    assert len({row["order_id"] for row in all_orders})==7
    assert len({row["line_id"] for row in all_lines})==21
    assert {row["customer_id"] for row in all_orders+all_lines}=={clients[0]["customer_id"]}
    assert {row["order_id"] for row in all_lines}=={row["order_id"] for row in all_orders}
    assert all(row["channel_name"]=="=Канал Юникод" for row in all_orders)
    with ZipFile(job.target) as archive:
        for name in archive.namelist():
            if name.startswith("Данные покупателей/") and name.endswith(".xlsx"):
                book=load_workbook(BytesIO(archive.read(name)),read_only=True)
                assert not any(c.data_type=="f" for row in book.active for c in row)
                book.close()
    assert exported.manifest["rows"]==dict(clients=1,orders=7,order_lines=21)
    clean(job)


def test_context_mismatch_fails_before_publication(job):
    result=aggregate(job)
    job.target.write_bytes(b"OLD ZIP")
    with pytest.raises(DetailExportError,match="REPORT_CONTEXT_MISMATCH"):
        run(job,result,order_mode=OrderMode.PURCHASE_EITHER)
    assert job.target.read_bytes()==b"OLD ZIP"
    clean(job)


def test_saved_counts_changed_source_fails_closed(job):
    result=aggregate(job)
    job.configure([raw_order("new-order","2026-01-01"),raw_order("new-order-2","2026-01-02")])
    job.target.write_bytes(b"OLD ZIP")
    with pytest.raises(DetailExportError,match="REPORT_DETAIL_COUNTS_MISMATCH"):
        run(job,result)
    assert job.target.read_bytes()==b"OLD ZIP"
    clean(job)


def test_cancel_during_xlsx_closes_xml_cleans_owned_files_and_preserves_old_zip(job,monkeypatch):
    result=aggregate(job);stopped=False
    original=xlsx.SafeProjection.row
    def row(self,kind,encoded):
        nonlocal stopped
        values=original(self,kind,encoded)
        if kind=="clients":
            stopped=True
        return values
    monkeypatch.setattr(xlsx.SafeProjection,"row",row)
    job.target.write_bytes(b"OLD ZIP")
    with pytest.raises(InterruptedError):
        run(job,result,cancelled=lambda:stopped)
    assert job.target.read_bytes()==b"OLD ZIP"
    clean(job)


def test_money_precision_and_cell_size_fail_explicitly():
    from openpyxl import Workbook
    book=Workbook(write_only=True);sheet=book.create_sheet()
    with pytest.raises(DetailExportError,match="XLSX_MONEY_PRECISION"):
        xlsx.excel_cell(sheet,Decimal("1234567890123456.78"),"price_of_line")
    with pytest.raises(DetailExportError,match="XLSX_CELL_TOO_LONG"):
        xlsx.excel_cell(sheet,"X"*32768,"product_name")
    with pytest.raises(DetailExportError,match="XLSX_MONEY_PRECISION"):
        xlsx.excel_cell(sheet,Decimal("10000000000000000000000000000.01"),"price_of_line")
    precise=Decimal("1.123456789012345678901234567")
    cell=xlsx.excel_cell(sheet,precise,"mean_interval_days")
    assert cell.value==str(precise) and cell.data_type=="s"
    for value in ("0000000000000000000123","=1+1","+test","@test","Юникод"):
        cell=xlsx.excel_cell(sheet,value,"product_id")
        assert cell.value==value and cell.data_type=="s"
    book.close()



def test_entire_archive_uses_owned_xml_and_changes_pseudonyms_between_exports(job,monkeypatch):
    import openpyxl.worksheet._writer as xml
    result=aggregate(job)
    def forbidden(*args,**kwargs):
        pytest.fail("Unified report must never create system temporary worksheet XML")
    monkeypatch.setattr(xml,"create_temporary_file",forbidden)
    run(job,result)
    first=members(job)
    run(job,result)
    second=members(job)
    name="Данные покупателей/Клиенты_01.xlsx"
    assert first[name][1][0][0]!=second[name][1][0][0]
    assert first[name][1][0][1:]==second[name][1][0][1:]
    clean(job)
