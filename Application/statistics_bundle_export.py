"""Atomic two-file report from a captured aggregate snapshot; no source/data IO."""
from copy import deepcopy
from dataclasses import dataclass
import errno
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from zipfile import ZipFile,ZIP_DEFLATED
from Application.statistics_cache import validate_result
from Application.statistics_export import write_statistics_xlsx,suggested_filename as xlsx_filename
from Application.statistics_html_export import build_statistics_html
from Application.statistics_report_presentation import report_presentation
from Application.statistics_detail_job import validate_report_destination
from Application.statistics_detail_storage import DetailExportError,admit_destination
from Application.statistics_storage_safety import path_identity,reject_reparse,check_owned_staging

FILES=("Статистика.xlsx","Статистика.html")

@dataclass(frozen=True)
class ReportExportResult:
    destination: Path
    timings: dict
    files: tuple=FILES

def suggested_filename(result):
    return Path(xlsx_filename(result)).with_suffix(".zip").name

def _version(path):
    if not path.exists():return None
    s=path.stat()
    return path_identity(path),s.st_size,s.st_mtime_ns

class _ReportWorkspace:
    """Owned artifacts beside the destination; no SQLite or private detail root."""
    def __init__(self,destination):
        self.destination=destination
        self.parent=destination.parent
        self.parent_id=path_identity(self.parent)
        self.before=_version(destination)
        self.folder=self.folder_id=self.partial=self.partial_id=None
        self.owned={}
    def prepare(self):
        self.folder=Path(tempfile.mkdtemp(prefix=".statistics-report-",dir=self.parent))
        self.folder_id=path_identity(self.folder)
    def check_folder(self):
        check_owned_staging(self.parent,self.folder,self.parent_id,self.folder_id)
    def create(self,name):
        self.check_folder()
        path=self.folder/name
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|getattr(os,"O_BINARY",0),0o600)
        self.owned[name]=path_identity(path)
        return os.fdopen(fd,"wb")
    def create_archive(self):
        fd,name=tempfile.mkstemp(prefix=".statistics-report-",suffix=".partial",dir=self.parent)
        self.partial=Path(name);self.partial_id=path_identity(self.partial)
        return os.fdopen(fd,"wb")
    def check_file(self,path):
        self.check_folder()
        reject_reparse(path)
        if self.owned.get(path.name)!=path_identity(path):
            raise DetailExportError("SENSITIVE_STATE_CLEANUP_FAILED",technical_path=self.folder)
    def cleanup_state(self):
        if self.folder is None:return
        try:
            self.check_folder()
            children=list(self.folder.iterdir())
            if {p.name for p in children}!=set(self.owned):raise ValueError
            for path in children:self.check_file(path)
            for path in children:path.unlink()
            self.folder.rmdir()
        except Exception:
            raise DetailExportError("SENSITIVE_STATE_CLEANUP_FAILED",technical_path=self.folder) from None
        self.folder=None;self.owned.clear()
    def check_partial(self):
        reject_reparse(self.partial)
        if path_identity(self.parent)!=self.parent_id or path_identity(self.partial)!=self.partial_id:
            raise DetailExportError("PUBLICATION_PATH_CHANGED")
    def cleanup_partial(self):
        if self.partial is None:return
        try:
            self.check_partial();self.partial.unlink()
        except Exception:
            raise DetailExportError("SENSITIVE_ARCHIVE_CLEANUP_FAILED",technical_path=self.partial) from None
        self.partial=None
    def publish(self):
        self.check_partial();reject_reparse(self.destination)
        if _version(self.destination)!=self.before:raise DetailExportError("DESTINATION_CHANGED")
        os.replace(self.partial,self.destination)
        self.partial=None

class _ReportControl:
    def __init__(self,workspace,progress,cancelled,reserve,budget):
        if type(reserve) is not int or reserve<0 or type(budget) is not int or budget<1024**2:
            raise DetailExportError("INVALID_EXPORT_OPTIONS")
        self.workspace=workspace;self.progress=progress;self.cancelled=cancelled
        self.reserve=reserve;self.budget=budget;self.ticks=0
    def check(self):
        if self.cancelled and self.cancelled():raise InterruptedError("Report export cancelled")
        self.ticks+=1
        if self.ticks%256==0:self.space()
    def space(self):
        w=self.workspace
        if shutil.disk_usage(w.parent).free<self.reserve:
            raise DetailExportError("INSUFFICIENT_SPACE")
        size=sum((w.folder/name).stat().st_size for name in w.owned) if w.folder else 0
        if w.partial is not None:size+=w.partial.stat().st_size
        if size>self.budget:raise DetailExportError("TEMPORARY_BUDGET_EXCEEDED")
    def stage(self,name):
        self.check();self.space()
        if self.progress:self.progress(dict(stage=name,processed=0))
        self.check()

def _verify_archive(path,check):
    with ZipFile(path) as archive:
        if tuple(archive.namelist())!=FILES:raise DetailExportError("ARCHIVE_MEMBERS_MISMATCH")
        for name in FILES:
            with archive.open(name) as member:
                while member.read(64*1024):check()

def export_statistics_bundle(result,path,*,progress=None,cancelled=None,
                             reserve_bytes=128*1024**2,max_temporary_bytes=8*1024**3):
    """Render one saved presentation, package exactly two members and publish last."""
    started=perf_counter();workspace=None;timings={}
    try:
        destination=validate_report_destination(path)
        workspace=_ReportWorkspace(destination)
        control=_ReportControl(workspace,progress,cancelled,reserve_bytes,max_temporary_bytes)
        control.stage("report_preparation")
        stamp=perf_counter()
        captured=validate_result(deepcopy(result))
        model=report_presentation(captured)
        timings["presentation"]=perf_counter()-stamp
        admit_destination(destination.parent)
        control.check();workspace.prepare()
        control.stage("statistics_xlsx");stamp=perf_counter()
        with workspace.create(FILES[0]) as stream:
            write_statistics_xlsx(captured,stream,check=control.check,presentation=model)
            stream.flush();os.fsync(stream.fileno())
        timings["excel"]=perf_counter()-stamp
        control.stage("statistics_html");stamp=perf_counter()
        html=build_statistics_html(captured,presentation=model,check=control.check).encode("utf-8")
        with workspace.create(FILES[1]) as stream:
            for offset in range(0,len(html),64*1024):
                control.check();stream.write(html[offset:offset+64*1024])
            stream.flush();os.fsync(stream.fileno())
        del html
        timings["html"]=perf_counter()-stamp
        control.stage("archive");stamp=perf_counter()
        with workspace.create_archive() as stream:
            with ZipFile(stream,"w",ZIP_DEFLATED,allowZip64=True) as archive:
                for name in FILES:
                    source=workspace.folder/name;workspace.check_file(source)
                    with source.open("rb") as original,archive.open(name,"w",force_zip64=True) as member:
                        while chunk:=original.read(64*1024):
                            control.check();member.write(chunk)
            stream.flush();os.fsync(stream.fileno())
        workspace.check_partial()
        _verify_archive(workspace.partial,control.check)
        control.space()
        workspace.cleanup_state()
        control.check()
        workspace.publish()  # No cancellation/error-producing callbacks after commit.
        timings["zip"]=perf_counter()-stamp
        timings["total"]=perf_counter()-started
        return ReportExportResult(destination,timings)
    except (DetailExportError,InterruptedError):
        raise
    except OSError as error:
        code="ACCESS_DENIED" if isinstance(error,PermissionError) else "INSUFFICIENT_SPACE" if error.errno==errno.ENOSPC else "IO_ERROR"
        raise DetailExportError(code) from None
    finally:
        if workspace is not None:
            try:workspace.cleanup_state()
            finally:workspace.cleanup_partial()
