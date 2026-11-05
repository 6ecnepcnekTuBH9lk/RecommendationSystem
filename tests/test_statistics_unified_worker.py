"""Bounded report stdin, privacy on errors, owned XML and streaming safety."""
from io import StringIO
import json
from threading import Event
from zipfile import ZipFile

import pytest

from Application import statistics_detail_xlsx as writer
from Application.statistics_detail_storage import DetailExportError, DetailWorkspace, DetailControl
from scripts import statistics_detail_export as cli
from tests.test_statistics_unified_archive import job, aggregate, run, clean

__all__=["job"]


class ReportChannel:
    def __init__(self,payload):
        self.payload=payload;self.first=True;self.release=Event()
    def readline(self,limit):
        if self.first:
            self.first=False
            return self.payload[:limit]
        self.release.wait(10)
        return b""


def args(job):
    return ["--destination",str(job.target)]


def test_worker_uses_captured_snapshot_and_emits_only_safe_protocol(job):
    result=aggregate(job)
    channel=ReportChannel(json.dumps(dict(command="report",snapshot=result)).encode()+b"\n")
    output=StringIO()
    try:
        assert cli.main(args(job),control=channel,output=output)==0
    finally:
        channel.release.set()
    messages=[json.loads(line) for line in output.getvalue().splitlines()]
    assert messages[-1]==dict(event="finished",files=2)
    assert all(m["event"] in ("progress","finished") for m in messages)
    assert "source_customer_id" not in output.getvalue() and "SYNTH_FIRST" not in output.getvalue()
    with ZipFile(job.target) as archive:
        assert archive.testzip() is None
        assert all(name.endswith((".xlsx",".html")) for name in archive.namelist())
    clean(job)


@pytest.mark.parametrize("kind",["missing","extra","invalid","duplicate","oversized","schema"])
def test_worker_rejects_invalid_report_before_source_processing_and_keeps_old_target(job,monkeypatch,kind):
    result=aggregate(job)
    payload=json.dumps(dict(command="report",snapshot=result)).encode()+b"\n"
    if kind=="missing":payload=b'{"command":"cancel"}\n'
    if kind=="extra":payload=b'{"command":"report","snapshot":{},"email":"PRIVATE@example.invalid"}\n'
    if kind=="invalid":payload=b'PRIVATE@example.invalid\n'
    if kind=="duplicate":payload=b'{"command":"report","command":"report","snapshot":{}}\n'
    if kind=="oversized":monkeypatch.setattr(cli,"MAX_REPORT_INPUT",100)
    if kind=="schema":
        result["analysis_filter"]["unknown"]="PRIVATE@example.invalid"
        payload=json.dumps(dict(command="report",snapshot=result)).encode()+b"\n"
    def forbidden(*a,**k):pytest.fail("Invalid snapshot must not start report export")
    monkeypatch.setattr(cli,"export_statistics_bundle",forbidden)
    job.target.write_bytes(b"OLD ZIP")
    channel=ReportChannel(payload);output=StringIO()
    try:
        assert cli.main(args(job),control=channel,output=output)==1
    finally:
        channel.release.set()
    terminal=json.loads(output.getvalue().splitlines()[-1])
    assert terminal==dict(event="error",code="REPORT_CONTEXT_MISMATCH")
    assert "PRIVATE" not in output.getvalue()
    assert job.target.read_bytes()==b"OLD ZIP"
    clean(job)


def test_foreign_replacement_of_owned_worksheet_is_never_deleted_or_published(job,monkeypatch):
    result=aggregate(job);job.target.write_bytes(b"OLD ZIP")
    original=writer._OwnedWorksheetWriter.cleanup
    foreign=[]
    def replace(self):
        if not foreign:
            original_path=self.path
            original_path.rename(original_path.with_suffix(".owned-copy"))
            original_path.write_bytes(b"FOREIGN FILE DO NOT DELETE")
            foreign.append(original_path)
        return original(self)
    monkeypatch.setattr(writer._OwnedWorksheetWriter,"cleanup",replace)
    with pytest.raises(DetailExportError,match="SENSITIVE_STATE_CLEANUP_FAILED"):
        run(job,result)
    assert foreign and foreign[0].read_bytes()==b"FOREIGN FILE DO NOT DELETE"
    assert job.target.read_bytes()==b"OLD ZIP"
    assert not list(job.target.parent.glob(".statistics-detail-*.partial"))


def test_writer_never_uses_system_temp_and_keeps_memory_bounded(job,monkeypatch):
    import tracemalloc
    from decimal import Decimal
    from Application.statistics_detail_contract import csv_cell, ORDER_LINE_COLUMNS
    import openpyxl.worksheet._writer as xml
    def forbidden(*args,**kwargs):
        pytest.fail("All worksheet XML must use the admitted private workspace")
    monkeypatch.setattr(xml,"create_temporary_file",forbidden)
    control=DetailControl(job.private,job.target,progress=None,cancelled=None,reserve_bytes=0,max_temporary_bytes=64*1024**2)
    workspace=DetailWorkspace(job.private,job.target,control)
    workspace.prepare()
    def rows():
        for i in range(2000):
            data=dict(order_id=str(i),customer_id="raw-client",source_customer_id="raw-source",line_ordinal=0,
                      line_id=str(i),line_number=i,product_id="000001",product_namespace="offline1C",
                      product_name="Рубашка",quantity=Decimal("1"),price_of_line=Decimal("12.34"),
                      line_status="F",status_group="DELIVERED",accepted_strict=True,accepted_extended=True,
                      currency="RUB",price_status="KNOWN")
            yield tuple(csv_cell(data.get(name)) for name in ORDER_LINE_COLUMNS)
    try:
        tracemalloc.start()
        with workspace.archive_stream() as stream:
            with ZipFile(stream,"w") as archive:
                count,names=writer.write_detail_members(archive,"order_lines",rows(),workspace,writer.SafeProjection(),max_rows=1001)
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        assert count==2000 and len(names)==2 and peak<16*1024**2
    finally:
        if tracemalloc.is_tracing():tracemalloc.stop()
        workspace.cleanup_state();workspace.cleanup_partial()
    clean(job)

