"""Owned temporary state and publication files; no canonical or repository writes."""

import base64
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import uuid

from Application.statistics_storage_safety import (
    reject_reparse, validate_storage_root, path_identity, check_owned_staging,
    local_volume, inspect_acls, check_acl_facts,
)


class DetailExportError(ValueError):
    def __init__(self, code='DETAIL_EXPORT_FAILED', *, technical_path=None):
        self.code = code
        self.technical_path = str(technical_path) if technical_path is not None else None
        super().__init__(code if technical_path is None else code+': '+self.technical_path)


def outside_repository(path):
    return not any((p/'.git').exists() for p in (path,*path.parents))


def admit_destination(parent):
    reject_reparse(parent)
    local_volume(parent)
    if os.name=='nt':
        for fact in inspect_acls((parent,*parent.parents)):
            check_acl_facts(fact,private=False)
    else:
        for p in (parent,*parent.parents):
            info=p.stat()
            if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
                raise DetailExportError('UNSAFE_DESTINATION_NAMESPACE')


def _protect_owned(path, *, directory):
    """Restrict only a newly created owned object; bound the ACL subprocess."""
    if os.name!='nt':
        path.chmod(0o700 if directory else 0o600)
        return
    literal="'"+str(path).replace("'","''")+"'"
    rights='(OI)(CI)F' if directory else 'F'
    script=(f'$p={literal}; $sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; '
            f'& icacls.exe $p /inheritance:r /grant:r "*$($sid):{rights}" '
            f"'*S-1-5-18:{rights}' '*S-1-5-32-544:{rights}' | Out-Null; exit $LASTEXITCODE")
    r=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',
                      base64.b64encode(script.encode('utf-16le')).decode()],capture_output=True,timeout=30,
                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if r.returncode:
        raise DetailExportError('PRIVATE_OBJECT_ACL_FAILED')
    check_acl_facts(inspect_acls((path,))[0],private=True,root=directory)


def protect_directory(path):
    _protect_owned(path,directory=True)


def protect_file(path):
    _protect_owned(path,directory=False)


class DetailControl:
    def __init__(self, root, destination, *, progress, cancelled, reserve_bytes, max_temporary_bytes):
        if (type(reserve_bytes) is not int or reserve_bytes<0 or type(max_temporary_bytes) is not int
                or max_temporary_bytes<16*1024**2 or progress is not None and not callable(progress)
                or cancelled is not None and not callable(cancelled)):
            raise DetailExportError('INVALID_EXPORT_OPTIONS')
        self.root,self.destination = root,destination
        self.progress,self.cancelled = progress,cancelled
        self.reserve,self.budget = reserve_bytes,max_temporary_bytes
        self.folder=self.partial=None
        self.stopped=False
        self.ticks=self.bytes_since_check=0

    def space(self, extra=0):
        size=sum(p.stat().st_size for p in self.folder.iterdir() if p.is_file()) if self.folder is not None else 0
        if self.partial is not None:
            size += self.partial.stat().st_size
        if size+extra>self.budget:
            raise DetailExportError('TEMPORARY_BUDGET_EXCEEDED')
        for parent in (self.root,self.destination.parent):
            if shutil.disk_usage(parent).free < self.reserve+extra+8*1024**2:
                raise DetailExportError('INSUFFICIENT_SPACE')

    def check(self):
        try:
            self.stopped = self.stopped or bool(self.cancelled and self.cancelled())
        except Exception:
            raise DetailExportError('CANCELLATION_CALLBACK_FAILED') from None
        if self.stopped:
            raise InterruptedError('Detail export cancelled')
        self.ticks += 1
        if self.ticks%256==0:
            self.space()

    def consume(self, count):
        self.check()
        self.bytes_since_check += count
        if self.bytes_since_check >= 1024**2:
            self.space(extra=count*2+8*1024**2)
            self.bytes_since_check=0

    def stage(self, name, processed=0):
        self.check()
        self.space()
        if self.progress is not None:
            try:
                self.progress(dict(stage=name,processed=processed))
            except Exception:
                raise DetailExportError('PROGRESS_CALLBACK_FAILED') from None
        self.check()

    def sql_progress(self):
        try:
            self.check()
            return 0
        except Exception:
            return 1


class DetailWorkspace:
    def __init__(self, root, destination, control):
        self.root,self.destination,self.control = root,destination,control
        self.folder=self.partial=self.db=None
        self.root_id=self.folder_id=self.file_id=None
        self.parent_id=None
        self._fd=None

    def prepare(self):
        reject_reparse(self.root)
        if os.name=='nt' and self.root.drive.upper()!='D:':
            raise DetailExportError('TEMPORARY_ROOT_MUST_BE_ON_D')
        if not self.root.is_dir() or not outside_repository(self.root):
            raise DetailExportError('PRIVATE_TEMPORARY_ROOT_REQUIRED')
        self.root_id=validate_storage_root(self.root)
        admit_destination(self.destination.parent)
        self.parent_id=path_identity(self.destination.parent)
        self.control.space()
        self.folder=Path(tempfile.mkdtemp(prefix='detail-',dir=self.root))
        self.folder_id=path_identity(self.folder)
        protect_directory(self.folder)
        validate_storage_root(self.root,children=(self.folder,))
        self.control.folder=self.folder
        self.db=sqlite3.connect(self.folder/'state.sqlite')
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA cache_size=-8192')
        self.db.execute('PRAGMA temp_store=MEMORY')  # Queries use incremental disk B-trees, not huge sort buffers.
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA max_page_count='+str(self.control.budget//4096))
        self.db.set_progress_handler(self.control.sql_progress,10000)
        self.partial=self.destination.parent/('.statistics-detail-'+uuid.uuid4().hex+'.partial')
        self._fd=os.open(self.partial,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,'O_BINARY',0),0o600)
        self.file_id=path_identity(self.partial)
        protect_file(self.partial)
        self.control.partial=self.partial
        return self.db

    def archive_stream(self):
        fd,self._fd=self._fd,None
        return os.fdopen(fd,'wb',buffering=0)

    def validate_partial(self):
        reject_reparse(self.partial)
        if path_identity(self.destination.parent)!=self.parent_id or path_identity(self.partial)!=self.file_id:
            raise DetailExportError('PUBLICATION_PATH_CHANGED')

    def cleanup_state(self):
        if self.db is not None:
            try:
                self.db.set_progress_handler(None,0)
                self.db.close()
                self.db=None
            except Exception:
                raise DetailExportError('SENSITIVE_STATE_CLOSE_FAILED',technical_path=self.folder) from None
        if self.folder is not None:
            try:
                check_owned_staging(self.root,self.folder,self.root_id,self.folder_id)
                allowed={'state.sqlite','state.sqlite-journal','state.sqlite-wal','state.sqlite-shm'}
                if any(p.name not in allowed for p in self.folder.iterdir()):
                    raise ValueError('Unexpected job file')
                for p in self.folder.iterdir():
                    p.unlink()
                self.folder.rmdir()
            except Exception:
                raise DetailExportError('SENSITIVE_STATE_CLEANUP_FAILED',technical_path=self.folder) from None
            self.folder=None
            self.control.folder=None

    def cleanup_partial(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd=None
        if self.partial is not None and self.file_id is None:
            self.partial=None
            self.control.partial=None
        if self.partial is not None:
            try:
                self.validate_partial()
                self.partial.unlink()
            except Exception:
                raise DetailExportError('SENSITIVE_ARCHIVE_CLEANUP_FAILED',technical_path=self.partial) from None
            self.partial=None
            self.control.partial=None
