"""Two aggregate report members only; snapshot-only IO, atomicity, timing and IPC."""
from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile
import pytest
from openpyxl import load_workbook
from Application import statistics_bundle_export as bundle
from Application.statistics_detail_storage import DetailExportError
from tests.tabs.test_dataset_statistics_tab import sample_result,rich_customer_result,app,window
__all__=["sample_result","rich_customer_result","app","window"]

def assert_archive(path):
    with ZipFile(path) as archive:
        assert archive.namelist()==["Статистика.xlsx","Статистика.html"]
        assert archive.testzip() is None
        book=load_workbook(BytesIO(archive.read("Статистика.xlsx")),read_only=True)
        assert book.sheetnames==["Действия","Заказы","Товары","Клиенты","Техническая информация"]
        rows=list(book["Клиенты"].values)
        assert all(any(r[0]==title for r in rows) for title in ("Обзор","Покупки","Денежные показатели","RFM-анализ","Когорты"))
        book.close()
        document=archive.read("Статистика.html").decode()
        assert all('id="section-'+str(i)+'"' in document for i in range(5))
        assert all('id="customer-section-'+str(i)+'"' in document for i in range(5))
        assert 'data-rfm-window=' in document and 'data-rfm-currency=' in document
        assert "connect-src 'none'" in document and '<script src=' not in document
        return document

def clean(directory):
    assert not list(directory.glob(".statistics-report-*"))

def test_two_files_preserve_snapshot_and_never_call_sources_or_detail_backend(sample_result,tmp_path,monkeypatch):
    import sqlite3
    from Application import statistics_detail_backend as detail,statistics_detail_xlsx as individual
    from Application.mindbox import raw_reader,canonical_storage
    from Application import dataset_statistics,statistics_query
    source=tmp_path/"changed-source";source.mkdir();(source/"orders.json").write_text("CHANGED AFTER CALCULATION")
    def forbidden(*a,**k):pytest.fail("Report must use the captured aggregates only")
    for module,names in ((detail,("export_detail_statistics","_ingest","_prepare_customers")),
                         (individual,("write_detail_members","SafeProjection")),
                         (raw_reader,("iter_export",)),(canonical_storage,("catalog",)),
                         (dataset_statistics,("calculate_statistics",))):
        for name in names:monkeypatch.setattr(module,name,forbidden)
    monkeypatch.setattr(sqlite3,"connect",forbidden)
    monkeypatch.setattr(statistics_query,"query_statistics",forbidden,raising=False)
    before=deepcopy(sample_result);stages=[]
    target=tmp_path/"report.zip"
    result=bundle.export_statistics_bundle(sample_result,target,progress=lambda m:stages.append(m["stage"]))
    assert result.destination==target and result.files==bundle.FILES
    assert set(result.timings)=={"presentation","excel","html","zip","total"}
    assert all(v>=0 for v in result.timings.values())
    assert stages==["report_preparation","statistics_xlsx","statistics_html","archive"]
    document=assert_archive(target)
    assert "Покрытие:" in document and "Дата расчёта" in document
    assert sample_result==before
    clean(tmp_path)

@pytest.mark.parametrize("phase",["report_preparation","statistics_xlsx","statistics_html","archive"])
def test_cancel_each_phase_keeps_existing_zip_and_cleans_only_own_files(sample_result,tmp_path,phase):
    target=tmp_path/"report.zip";target.write_bytes(b"OLD ZIP")
    other=tmp_path/"unrelated.txt";other.write_text("PRESERVE")
    stopped=False
    def stage(m):
        nonlocal stopped
        stopped=m["stage"]==phase
    with pytest.raises(InterruptedError):
        bundle.export_statistics_bundle(sample_result,target,progress=stage,cancelled=lambda:stopped)
    assert target.read_bytes()==b"OLD ZIP" and other.read_text()=="PRESERVE"
    clean(tmp_path)

@pytest.mark.parametrize("phase",["xlsx","html","zip","crc","fsync","replace"])
def test_write_errors_preserve_old_zip_and_no_unknown_cleanup(sample_result,tmp_path,monkeypatch,phase):
    target=tmp_path/"report.zip";target.write_bytes(b"OLD ZIP")
    other=tmp_path/"unrelated.txt";other.write_text("PRESERVE")
    def fail(*a,**k):raise PermissionError("SYNTH_PRIVATE@example.invalid")
    if phase=="xlsx":monkeypatch.setattr(bundle,"write_statistics_xlsx",fail)
    elif phase=="html":monkeypatch.setattr(bundle,"build_statistics_html",fail)
    elif phase=="crc":monkeypatch.setattr(bundle,"_verify_archive",fail)
    elif phase=="fsync":monkeypatch.setattr(bundle.os,"fsync",fail)
    elif phase=="replace":monkeypatch.setattr(bundle.os,"replace",fail)
    else:
        original=ZipFile.open
        def member(self,name,mode="r",*a,**k):
            if mode=="w" and name=="Статистика.html":fail()
            return original(self,name,mode,*a,**k)
        monkeypatch.setattr(ZipFile,"open",member)
    with pytest.raises(DetailExportError) as error:bundle.export_statistics_bundle(sample_result,target)
    assert "SYNTH_PRIVATE" not in str(error.value)
    assert target.read_bytes()==b"OLD ZIP" and other.read_text()=="PRESERVE"
    clean(tmp_path)

def test_foreign_staging_file_prevents_publication_and_is_not_deleted(sample_result,tmp_path,monkeypatch):
    target=tmp_path/"report.zip";target.write_bytes(b"OLD ZIP")
    foreign=[]
    original=bundle._verify_archive
    def verify(path,check):
        original(path,check)
        folder=next(tmp_path.glob(".statistics-report-*"))
        if not folder.is_dir():folder=next(p for p in tmp_path.glob(".statistics-report-*") if p.is_dir())
        p=folder/"foreign.txt";p.write_text("DO NOT DELETE");foreign.append(p)
    monkeypatch.setattr(bundle,"_verify_archive",verify)
    with pytest.raises(DetailExportError,match="SENSITIVE_STATE_CLEANUP_FAILED"):
        bundle.export_statistics_bundle(sample_result,target)
    assert foreign[0].read_text()=="DO NOT DELETE" and target.read_bytes()==b"OLD ZIP"

def test_same_prepared_model_and_no_system_temp_xml(sample_result,tmp_path,monkeypatch):
    import openpyxl.worksheet._writer as xml
    calls=[]
    original=bundle.report_presentation
    def prepared(result):calls.append(result);return original(result)
    monkeypatch.setattr(bundle,"report_presentation",prepared)
    monkeypatch.setattr(xml,"create_temporary_file",lambda *a,**k:pytest.fail("No XML tempfile required"))
    bundle.export_statistics_bundle(sample_result,tmp_path/"report.zip")
    assert len(calls)==1
    assert_archive(tmp_path/"report.zip")



@pytest.mark.parametrize("phase",["excel","html","zip"])
def test_cancellation_during_actual_writing_is_atomic(sample_result,tmp_path,monkeypatch,phase):
    from Application import statistics_export as excel,statistics_html_export as html
    target=tmp_path/"report.zip";target.write_bytes(b"OLD ZIP")
    stopped=False;calls=0
    if phase=="excel":
        original=excel._Sheet.write
        def row(self,*args,**kwargs):
            nonlocal calls,stopped
            calls+=1
            if calls==30:stopped=True
            return original(self,*args,**kwargs)
        monkeypatch.setattr(excel._Sheet,"write",row)
    elif phase=="html":
        original=html._table_html
        def table(*args,**kwargs):
            nonlocal calls,stopped
            calls+=1
            result=original(*args,**kwargs)
            if calls==2:stopped=True
            return result
        monkeypatch.setattr(html,"_table_html",table)
    else:
        original=ZipFile.open
        def member(self,name,mode="r",*args,**kwargs):
            nonlocal stopped
            result=original(self,name,mode,*args,**kwargs)
            if mode=="w" and name==bundle.FILES[1]:stopped=True
            return result
        monkeypatch.setattr(ZipFile,"open",member)
    with pytest.raises(InterruptedError):
        bundle.export_statistics_bundle(sample_result,target,cancelled=lambda:stopped)
    assert stopped and target.read_bytes()==b"OLD ZIP"
    clean(tmp_path)


@pytest.mark.parametrize("kind",["folder","file","partial"])
def test_substituted_owned_artifacts_are_not_deleted_or_published(sample_result,tmp_path,monkeypatch,kind):
    target=tmp_path/"report.zip";target.write_bytes(b"OLD ZIP")
    original=bundle._verify_archive;foreign=[]
    def replace(path,check):
        original(path,check)
        folder=next(p for p in tmp_path.glob(".statistics-report-*") if p.is_dir())
        replaced=folder if kind=="folder" else folder/bundle.FILES[0] if kind=="file" else path
        replaced.rename(replaced.with_name(replaced.name+".owned-original"))
        if kind=="folder":
            replaced.mkdir();replacement=replaced/"foreign.txt"
        else:replacement=replaced
        replacement.write_bytes(b"FOREIGN FILE")
        foreign.append(replacement)
    monkeypatch.setattr(bundle,"_verify_archive",replace)
    with pytest.raises(DetailExportError):
        bundle.export_statistics_bundle(sample_result,target)
    assert foreign[0].read_bytes()==b"FOREIGN FILE" and target.read_bytes()==b"OLD ZIP"


@pytest.mark.parametrize("kind",["disk","temporary"])
def test_resource_guards_preserve_old_zip(sample_result,tmp_path,monkeypatch,kind):
    from types import SimpleNamespace
    target=tmp_path/"report.zip";target.write_bytes(b"OLD ZIP")
    if kind=="disk":
        monkeypatch.setattr(bundle.shutil,"disk_usage",lambda p:SimpleNamespace(free=0))
    with pytest.raises(DetailExportError,match="INSUFFICIENT_SPACE" if kind=="disk" else "TEMPORARY_BUDGET_EXCEEDED"):
        bundle.export_statistics_bundle(sample_result,target,reserve_bytes=1,max_temporary_bytes=1024**2)
    assert target.read_bytes()==b"OLD ZIP"
    clean(tmp_path)


def test_timing_protocol_and_technical_path_are_bounded_and_private(tmp_path):
    import json
    from Application.statistics_detail_job import decode_report_timings,allowed_report_cleanup_path
    safe=dict(format="statistics-report-timings-1",
              timings=dict(presentation=0,excel=.3,html=.4,zip=.1,total=.8))
    assert decode_report_timings(json.dumps(safe))==safe["timings"]
    for change in (dict(email="PRIVATE"),dict(excel=-1),dict(total=float("nan")),dict(zip=True)):
        bad=dict(safe,timings=dict(safe["timings"],**change))
        with pytest.raises(ValueError):decode_report_timings(json.dumps(bad))
    target=tmp_path/"report.zip"
    assert allowed_report_cleanup_path(tmp_path/".statistics-report-abc12345",target)
    assert allowed_report_cleanup_path(tmp_path/".statistics-report-abc12345.partial",target)
    assert not allowed_report_cleanup_path(tmp_path/"foreign.txt",target)
    assert not allowed_report_cleanup_path(tmp_path.parent/".statistics-report-abc12345",target)

