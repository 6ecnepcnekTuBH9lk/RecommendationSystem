"""Atomic report from a captured aggregate snapshot and optional detail rows."""
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
from Application.statistics_unified_dto import RESULT_SCHEMA, UnifiedResult, read_envelope
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
    def create_archive(self, *, private=False):
        fd,name=tempfile.mkstemp(prefix=".statistics-report-",suffix=".partial",dir=self.parent)
        self.partial=Path(name);self.partial_id=path_identity(self.partial)
        if private:
            from Application.statistics_detail_storage import protect_file

            os.close(fd)
            self.check_partial()
            protect_file(self.partial)
            self.check_partial()
            fd = os.open(self.partial, os.O_WRONLY | getattr(os, "O_BINARY", 0))
            if not os.path.samestat(os.fstat(fd), self.partial.stat()):
                os.close(fd)
                raise DetailExportError("PUBLICATION_PATH_CHANGED")
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
    def consume(self, _size):
        self.check()

    def stage(self,name,processed=0):
        self.check();self.space()
        if self.progress:self.progress(dict(stage=name,processed=processed))
        self.check()

def _verify_archive(path,check,files=FILES):
    with ZipFile(path) as archive:
        if tuple(archive.namelist())!=files:raise DetailExportError("ARCHIVE_MEMBERS_MISMATCH")
        for name in files:
            with archive.open(name) as member:
                while member.read(64*1024):check()

def validate_report_snapshot(result):
    if isinstance(result, UnifiedResult):
        return read_envelope(result.envelope())
    if isinstance(result, dict) and result.get("schema_version") == RESULT_SCHEMA:
        return read_envelope(result)
    return validate_result(result)


def export_statistics_bundle(result,path,*,progress=None,cancelled=None,
                             reserve_bytes=128*1024**2,max_temporary_bytes=8*1024**3,detail=None,detail_source=None):
    """Publish one captured report and, when requested, pinned detail rows."""
    started=perf_counter();workspace=None;timings={}
    try:
        destination=validate_report_destination(path)
        if detail is not None and os.name == "nt" and destination.drive.upper() != "D:":
            raise DetailExportError("TEMPORARY_ROOT_MUST_BE_ON_D")
        workspace=_ReportWorkspace(destination)
        control=_ReportControl(workspace,progress,cancelled,reserve_bytes,max_temporary_bytes)
        control.stage("report_preparation")
        stamp=perf_counter()
        captured=validate_report_snapshot(result if isinstance(result, UnifiedResult) else deepcopy(result))
        model=report_presentation(captured)
        timings["presentation"]=perf_counter()-stamp
        admit_destination(destination.parent)
        control.check();workspace.prepare()
        if detail is not None:
            from Application.statistics_index import protect_directory
            from Application.statistics_storage_safety import validate_storage_root

            if not isinstance(captured, UnifiedResult) or detail_source is None:
                raise DetailExportError("REPORT_CONTEXT_MISMATCH")
            workspace.check_folder()
            protect_directory(workspace.folder)
            workspace.check_folder()
            validate_storage_root(workspace.folder)
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
        files = FILES
        if detail is not None:
            from Application.statistics_unified_detail import FILES as DETAIL_FILES
            from Application.statistics_unified_detail import write_detail

            files += DETAIL_FILES
        with workspace.create_archive(private=detail is not None) as stream:
            with ZipFile(stream,"w",ZIP_DEFLATED,allowZip64=True) as archive:
                for name in FILES:
                    source=workspace.folder/name;workspace.check_file(source)
                    with source.open("rb") as original,archive.open(name,"w",force_zip64=True) as member:
                        while chunk:=original.read(64*1024):
                            control.check();member.write(chunk)
                if detail is not None:
                    write_detail(archive, workspace, control, captured, detail, **detail_source)
            stream.flush();os.fsync(stream.fileno())
        workspace.check_partial()
        if detail is None:
            _verify_archive(workspace.partial, control.check)
        else:
            _verify_archive(workspace.partial, control.check, files)
        control.space()
        workspace.cleanup_state()
        control.check()
        workspace.publish()  # No cancellation/error-producing callbacks after commit.
        timings["zip"]=perf_counter()-stamp
        timings["total"]=perf_counter()-started
        return ReportExportResult(destination,timings,files)
    except (DetailExportError,InterruptedError):
        raise
    except OSError as error:
        code="ACCESS_DENIED" if isinstance(error,PermissionError) else "INSUFFICIENT_SPACE" if error.errno==errno.ENOSPC else "IO_ERROR"
        raise DetailExportError(code) from None
    finally:
        if workspace is not None:
            try:workspace.cleanup_state()
            finally:workspace.cleanup_partial()
