"""Aggregate report child: same QProcess/stdin control, never canonical or INDEXED."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
import os
import json
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application.statistics_detail_job import (
    ERRORS,CLEANUP_CODES,MAX_MESSAGE,MAX_REPORT_INPUT,REPORT_TIMING_FORMAT,
    encode_message,decode_report_timings,allowed_report_cleanup_path,exception_diagnostic,
)
from Application.statistics_detail_storage import DetailExportError
from Application.statistics_bundle_export import export_statistics_bundle



class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise DetailExportError('INVALID_OPTIONS')


class _Discard(io.TextIOBase):
    def write(self, text):
        return len(text)


class Reporter:
    def __init__(self, output, cancelled):
        self.output, self.cancelled = output, cancelled
        self.diagnostics = sys.stderr
        self.lock = Lock()
        self.last = 0.
        self.current = dict(event='progress', stage='report_preparation', processed=0)
        self.closed = False

    def emit(self, message):
        with self.lock:
            if self.closed:
                return
            try:
                self.output.write(encode_message(message).decode('ascii'))
                self.output.flush()
            except (BrokenPipeError, OSError):
                self.cancelled.set()
                self.closed = True

    def progress(self, value):
        message = dict(event='progress', **value)
        encode_message(message)
        now = monotonic()
        changed = self.current['stage'] != message['stage']
        self.current = message
        if changed or now-self.last >= .25:
            self.last = now
            self.emit(message)

    def timings(self,value):
        data=json.dumps(dict(format=REPORT_TIMING_FORMAT,timings=value),ensure_ascii=True,separators=(",",":"))+"\n"
        decode_report_timings(data)
        with self.lock:
            try:
                self.diagnostics.write(data)
                self.diagnostics.flush()
            except (OSError,AttributeError,ValueError):
                pass

    def terminal(self, message):
        with self.lock:
            if not self.closed:
                try:
                    self.output.write(encode_message(message).decode('ascii'))
                    self.output.flush()
                except (BrokenPipeError, OSError):
                    self.cancelled.set()
                self.closed = True


class _ControlPipe:
    """Bounded private stdin without BufferedReader's finalization lock.

    A daemon may wait for parent cancellation until process exit. No Python
    buffered IO lock or Windows CRT descriptor lock is held during that wait.
    One reader serves report and cancellation, preserving prefetched bytes.
    """
    def __init__(self, fd):
        self.fd = fd
        self.pending = bytearray()
        self.read_bytes = lambda size: os.read(fd,size)
        if os.name=="nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes
            handle=msvcrt.get_osfhandle(fd)
            kernel=ctypes.WinDLL("kernel32",use_last_error=True)
            kernel.ReadFile.argtypes=(wintypes.HANDLE,wintypes.LPVOID,wintypes.DWORD,
                                      ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID)
            kernel.ReadFile.restype=wintypes.BOOL
            def read_pipe(size):
                buffer=ctypes.create_string_buffer(size)
                count=wintypes.DWORD()
                if not kernel.ReadFile(handle,buffer,size,ctypes.byref(count),None):
                    code=ctypes.get_last_error()
                    if code in (109,232): return b""  # Parent closed its pipe.
                    raise ctypes.WinError(code)
                return buffer.raw[:count.value]
            # A pending CRT _read holds stdin's descriptor lock on Windows;
            # native extension startup may need that same lock (e.g. NumPy).
            self.read_bytes=read_pipe

    def readline(self, limit):
        while True:
            newline = self.pending.find(b"\n")
            if newline >= 0 or len(self.pending) >= limit:
                end = min(newline+1 if newline >= 0 else limit, limit)
                data = bytes(self.pending[:end])
                del self.pending[:end]
                return data
            chunk = self.read_bytes(min(8192, limit-len(self.pending)))
            if not chunk:
                data = bytes(self.pending)
                self.pending.clear()
                return data
            self.pending.extend(chunk)


def _read_control(stream, cancelled, invalid):
    try:
        line = stream.readline(MAX_MESSAGE+1)
        if line == b'{"command":"cancel"}\n':
            cancelled.set()
        elif not line:  # Parent closed its private pipe or disappeared.
            cancelled.set()
        else:
            invalid.set()
            cancelled.set()
    except Exception:
        invalid.set()
        cancelled.set()


def _prepare_root(root, cancelled=lambda: False):
    """Create only missing owned directories; never alter an existing root ACL."""
    from Application import statistics_detail_storage as storage
    if cancelled():
        raise InterruptedError
    if root.exists():
        storage.validate_storage_root(root)
        return
    missing = []
    parent = root
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    storage.admit_destination(parent)
    for path in reversed(missing):
        if cancelled():
            raise InterruptedError
        path.mkdir()  # exclusive, no parents=True or repair of foreign directories
        storage.protect_directory(path)
        storage.validate_storage_root(path)


def _safe_error(error, temporary, destination):
    code = getattr(error, 'code', '')
    if code not in ERRORS:
        text = str(error)
        code = text if text in ERRORS else 'DETAIL_EXPORT_FAILED'
    message = dict(event='error', code=code)
    path = getattr(error, 'technical_path', None)
    if code in CLEANUP_CODES and path:
        candidate = Path(path)
        if len(str(candidate)) <= 1024 and allowed_report_cleanup_path(candidate,destination):
            candidate_message = dict(message, technical_path=str(candidate))
            try:
                encode_message(candidate_message)
            except ValueError:
                pass  # Keep the safe code when an escaped technical path exceeds IPC limits.
            else:
                message = candidate_message
    return message


def _unique_object(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError("Duplicate report key")
        result[key]=value
    return result


def main(argv=None, *, control=None, output=None):
    output=output or sys.stdout
    cancelled,invalid,heartbeat_stop=Event(),Event(),Event()
    reporter=Reporter(output,cancelled)
    destination=Path()
    parser=_Parser(add_help=False,allow_abbrev=False)
    parser.add_argument("--destination",required=True)
    parser.add_argument("--reserve-bytes",type=int,default=128*1024**2)
    parser.add_argument("--max-temporary-bytes",type=int,default=8*1024**3)
    def heartbeat():
        while not heartbeat_stop.wait(2):
            reporter.emit(reporter.current)
    try:
        args=parser.parse_args(argv)
        destination=Path(args.destination)
        channel=control if control is not None else _ControlPipe(sys.stdin.fileno())
        from Application.statistics_cache import validate_result
        line=channel.readline(MAX_REPORT_INPUT+1)
        try:
            payload=json.loads(line,object_pairs_hook=_unique_object)
            if len(line)>MAX_REPORT_INPUT or set(payload)!={"command","snapshot"} or payload["command"]!="report":
                raise ValueError
            captured=validate_result(payload["snapshot"])
        except Exception:
            raise DetailExportError("REPORT_CONTEXT_MISMATCH") from None
        Thread(target=_read_control,args=(channel,cancelled,invalid),daemon=True).start()
        Thread(target=heartbeat,daemon=True).start()
        with redirect_stdout(_Discard()),redirect_stderr(_Discard()):
            result=export_statistics_bundle(captured,destination,progress=reporter.progress,
                cancelled=cancelled.is_set,reserve_bytes=args.reserve_bytes,max_temporary_bytes=args.max_temporary_bytes)
        reporter.timings(result.timings)
        reporter.terminal(dict(event="finished",files=len(result.files)))
        return 0
    except InterruptedError:
        reporter.terminal(dict(event="error",code="CONTROL_PROTOCOL_ERROR") if invalid.is_set() else dict(event="cancelled"))
        return 1 if invalid.is_set() else 2
    except Exception as error:
        diagnostic=exception_diagnostic(error,reporter.current["stage"])
        try:
            sys.stderr.write(json.dumps(diagnostic,ensure_ascii=True,separators=(",",":"))+"\n")
            sys.stderr.flush()
        except (OSError,AttributeError,ValueError):
            pass
        reporter.terminal(_safe_error(error,None,destination))
        return 1
    finally:
        heartbeat_stop.set()


if __name__=="__main__":
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
