"""First eligible canonical snapshot, bounded counters and unchanged report/PII contracts."""
from copy import deepcopy
from io import BytesIO
import json
from zipfile import ZipFile
import pytest
from openpyxl import load_workbook
from Application.mindbox.order_dedup import OrderSnapshots,OrderSnapshotConflict,semantic_fingerprint
from Application.statistics_facts import IndexedSnapshots
from Application.statistics_detail_job import validate_snapshot_diagnostics,decode_snapshot_diagnostic,SNAPSHOT_DIAGNOSTIC_FORMAT
from Application.statistics_detail_storage import DetailExportError
from tests.test_statistics_unified_archive import job,aggregate,run,members
from tests.test_statistics_detail_backend import raw_order,clean,snapshot
__all__=["job"]

def versions(kind):
    first=raw_order("SYNTH_PRIVATE_ORDER","2026-01-01","old",("F",))
    first["customer"]["email"]="SYNTH_CONTACT@example.invalid"
    changed=deepcopy(first)
    if kind=="returned_first":
        first["lines"][0]["status"]["ids"]["externalId"]="Return"
        return [first,changed],0,1
    if kind=="same_scope_date":
        changed["firstAction"]["dateTimeUtc"]="2026-02-01T00:00:00Z"
        return [first,changed],0,1
    if kind=="single":return [first],0,0
    if kind=="identical":return [first,deepcopy(first)],1,0
    if kind=="status":
        changed["lines"][0]["status"]["ids"]["externalId"]="Return"
        return [first,changed],0,1
    if kind=="positions":
        changed["lines"].append(deepcopy(changed["lines"][0]))
        changed["lines"][-1]["quantity"]=2
        return [first,changed],0,1
    if kind=="metadata":
        changed["customer"]["email"]="CHANGED_CONTACT@example.invalid"
        return [first,changed],0,1
    changed["lines"][0]["status"]["ids"]["externalId"]="CP"
    last=deepcopy(changed);last["lines"][0]["status"]["ids"]["externalId"]="Return"
    return [first,changed,deepcopy(changed),deepcopy(first),last],1,3

def configure_parts(job,records):
    job.configure(records)
    entry=job.data[2]["manual_interactions"]["orders"]
    directory=job.root/entry["directory"]
    (directory/"orders_part_001.json").write_text(json.dumps(dict(orders=records[:1])),encoding="utf-8")
    (directory/"orders_part_002.json").write_text(json.dumps(dict(orders=records[1:])),encoding="utf-8")
    entry["parts"]=2
    (job.root/"canonical/catalog.json").write_text(json.dumps(job.data[2]),encoding="utf-8")

@pytest.mark.parametrize("kind",["single","identical","status","positions","metadata","sequential","parts","returned_first","same_scope_date"])
def test_reference_selection_matches_export_without_duplicate_rows_and_with_safe_metadata(job,kind):
    records,identical,conflicting=versions("sequential" if kind=="parts" else kind)
    if kind=="parts":configure_parts(job,records)
    else:job.configure(records)
    result=aggregate(job);saved=deepcopy(result);sources=snapshot(job)
    observed=[]
    exported=run(job,result,diagnostic=observed.append)
    expected=dict(rule="FIRST_ELIGIBLE_CANONICAL",scope="ORDERS_HISTORY_BEFORE_CUTOFF",
                  raw=len(records),accepted=1,identical=identical,conflicting=conflicting)
    assert observed==[expected] and exported.manifest["snapshot_diagnostics"]==expected
    tables=members(job)
    ch,clients=tables["Данные покупателей/Клиенты_01.xlsx"]
    oh,orders=tables["Данные покупателей/Заказы_01.xlsx"]
    lh,lines=tables["Данные покупателей/Позиции заказов_01.xlsx"]
    assert len(clients)==len(orders)==len(lines)==1
    c=dict(zip(ch,clients[0]));o=dict(zip(oh,orders[0]));line=dict(zip(lh,lines[0]))
    assert c["customer_id"]==o["customer_id"]==line["customer_id"]
    assert o["order_id"]==line["order_id"] and line["line_status"]==records[0]["lines"][0]["status"]["ids"]["externalId"]
    strict=result["buyer_statistics"]["policies"]["STRICT_COMPLETED"]
    assert strict["orders"]==strict["buyers"]==strict["positions"]==(0 if kind=="returned_first" else 1)
    assert result["orders"]==1
    assert exported.manifest["rows"]==dict(clients=1,orders=1,order_lines=1)
    with ZipFile(job.target) as archive:
        assert len(archive.namelist())==5 and all(n.endswith((".xlsx",".html")) for n in archive.namelist())
        book=load_workbook(BytesIO(archive.read("Статистика.xlsx")),read_only=True)
        assert len(book.sheetnames)==5
        for sheet in book:
            rows=list(sheet.values)
            assert any(r[:2]==("Конфликтующих повторов снимков",conflicting) for r in rows)
            assert any(r[:2]==("Идентичных повторов снимков",identical) for r in rows)
        book.close()
        html=archive.read("Статистика.html").decode()
        assert "Конфликтующих повторов снимков" in html and "Первое подходящее вхождение" in html
    assert all(v not in json.dumps(observed) for v in ("SYNTH_PRIVATE_ORDER","SYNTH_CONTACT","CHANGED_CONTACT"))
    assert result==saved and snapshot(job)==sources
    clean(job)

@pytest.mark.parametrize("dates",[("2025-12-31","2026-01-01"),("2022-08-31","2022-09-01")])
def test_changed_timestamp_crossing_period_or_rfm_boundary_fails_closed(job,dates):
    records=[raw_order("SYNTH_SCOPE_ORDER",d,"old",("F",)) for d in dates]
    job.configure(records);result=aggregate(job);job.target.write_bytes(b"PREVIOUS ZIP")
    with pytest.raises(DetailExportError,match="ORDER_SNAPSHOT_SCOPE_MISMATCH") as error:
        run(job,result)
    assert error.value.snapshot_diagnostics["conflicting"]==1
    assert "SYNTH_SCOPE_ORDER" not in str(error.value)
    assert job.target.read_bytes()==b"PREVIOUS ZIP"
    clean(job)

def test_future_first_raw_is_not_first_eligible_snapshot(job):
    records=[raw_order("SYNTH_FUTURE","2026-09-01","old",("Return",)),
             raw_order("SYNTH_FUTURE","2026-01-01","old",("F",))]
    job.configure(records)
    exported=run(job,aggregate(job))
    assert exported.manifest["snapshot_diagnostics"]==dict(rule="FIRST_ELIGIBLE_CANONICAL",
        scope="ORDERS_HISTORY_BEFORE_CUTOFF",raw=1,accepted=1,identical=0,conflicting=0)
    clean(job)

def test_global_strict_selector_is_unchanged_and_indexed_source_selects_same_first():
    a,b=versions("status")[0]
    strict=OrderSnapshots();assert strict.accept(a)
    with pytest.raises(OrderSnapshotConflict):strict.accept(b)
    reference=OrderSnapshots();indexed=IndexedSnapshots()
    for record,accepted in ((a,True),(b,False),(a,False)):
        assert reference.accept(record,diagnose=True)==accepted
        projected=dict(ids=dict(mindboxId="synthetic-sid"),_fingerprint=semantic_fingerprint(record))
        assert indexed.accept(projected,diagnose=True)==accepted
    assert (reference.identical,reference.conflicting)==(indexed.identical,indexed.conflicting)==(1,1)

def test_diagnostic_accepts_only_counters_and_constant_rule():
    safe=dict(rule="FIRST_ELIGIBLE_CANONICAL",scope="ORDERS_HISTORY_BEFORE_CUTOFF",
              raw=3,accepted=1,identical=1,conflicting=1)
    assert validate_snapshot_diagnostics(safe)==safe
    assert decode_snapshot_diagnostic(json.dumps(dict(format=SNAPSHOT_DIAGNOSTIC_FORMAT,order_snapshots=safe)))==safe
    for bad in (dict(safe,email="PRIVATE@example.invalid"),dict(safe,raw=4),dict(safe,conflicting=True)):
        with pytest.raises(ValueError):validate_snapshot_diagnostics(bad)



from PyQt6.QtTest import QTest
from Application.statistics_detail_process import DetailExportProcess
from Application.statistics_detail_storage import protect_directory as real_protect_directory
from tests.tabs.test_dataset_statistics_tab import app,wait_until
import os
__all__ += ["app"]

@pytest.mark.skipif(os.name!="nt",reason="Windows production QProcess")
@pytest.mark.parametrize("changed_scope",[False,True])
def test_production_qprocess_uses_saved_aggregates_without_reselecting_snapshots(job,app,changed_scope):
    records,identical,conflicting=versions("sequential")
    if changed_scope:
        records=[raw_order("SYNTH_SCOPE_ORDER",d,"old",("F",)) for d in ("2025-12-31","2026-01-01")]
    configure_parts(job,records)
    result=aggregate(job)
    real_protect_directory(job.private)
    if changed_scope:job.target.write_bytes(b"PREVIOUS ZIP")
    worker=DetailExportProcess();terminal=[]
    worker.completed.connect(terminal.append)
    # The active report no longer selects raw snapshots: already calculated result is authoritative.
    for part in (job.root/"canonical").rglob("orders_part_*.json"):
        part.write_text('{"orders":[]}',encoding="utf-8")
    worker.start(dict(destination=job.target),report_result=result)
    try:
        wait_until(lambda:bool(terminal),timeout=70000)
    finally:
        if not terminal:worker.shutdown()
    assert "order_snapshots" not in worker.diagnostic
    assert "SYNTH_" not in json.dumps(worker.diagnostic)
    assert worker.diagnostic["exit_status"]=="NormalExit" and not worker.diagnostic["crash_exit"]
    assert terminal==[dict(event="finished",files=2)]
    with ZipFile(job.target) as archive:
        assert archive.testzip() is None and archive.namelist()==["Статистика.xlsx","Статистика.html"]
        book=load_workbook(BytesIO(archive.read("Статистика.xlsx")),read_only=True)
        assert len(book.sheetnames)==5
        assert not any(row[0]=="Конфликтующих повторов снимков" for row in book["Клиенты"].values)
        book.close()
    clean(job)
    worker.deleteLater();QTest.qWait(10)

