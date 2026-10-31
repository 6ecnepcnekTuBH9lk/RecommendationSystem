"""Detail export child: private stdin control and bounded safe JSON Lines."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import date
import io
import logging
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Application.statistics_detail_job import (
    ERRORS, CLEANUP_CODES, MAX_MESSAGE, encode_message, allowed_cleanup_path, inspect_sources, validate_paths, validate_period,
)
from Application.statistics_detail_contract import DetailPeriod, OrderMode
from Application.statistics_detail_storage import DetailExportError
from Application import statistics_detail_storage as storage
from Application.statistics_detail_backend import export_detail_statistics


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise DetailExportError('INVALID_OPTIONS')


class _Discard(io.TextIOBase):
    def write(self, text):
        return len(text)


class Reporter:
    def __init__(self, output, cancelled):
        self.output, self.cancelled = output, cancelled
        self.lock = Lock()
        self.last = 0.
        self.current = dict(event='progress', stage='source_validation', processed=0)
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

    def terminal(self, message):
        with self.lock:
            if not self.closed:
                try:
                    self.output.write(encode_message(message).decode('ascii'))
                    self.output.flush()
                except (BrokenPipeError, OSError):
                    self.cancelled.set()
                self.closed = True


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
        if len(str(candidate)) <= 1024 and allowed_cleanup_path(candidate, temporary, destination):
            candidate_message = dict(message, technical_path=str(candidate))
            try:
                encode_message(candidate_message)
            except ValueError:
                pass  # Keep the safe code when an escaped technical path exceeds IPC limits.
            else:
                message = candidate_message
    return message


def main(argv=None, *, control=None, output=None):
    output = output or sys.stdout
    cancelled, invalid, heartbeat_stop = Event(), Event(), Event()
    reporter = Reporter(output, cancelled)
    temporary = destination = Path()
    configured = False
    parser = _Parser(add_help=False, allow_abbrev=False)
    for name in ('raw-root', 'destination', 'temporary-root', 'as-of'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--order-mode', default=OrderMode.ALL_SELECTED.value)
    parser.add_argument('--reserve-bytes', type=int, default=128*1024**2)
    parser.add_argument('--max-temporary-bytes', type=int, default=8*1024**3)
    def heartbeat():
        while not heartbeat_stop.wait(2):
            reporter.emit(reporter.current)
    try:
        args = parser.parse_args(argv)
        as_of = date.fromisoformat(args.as_of)
        period = DetailPeriod(date.fromisoformat(args.start) if args.start else None,
                              date.fromisoformat(args.end) if args.end else None)
        mode = OrderMode(args.order_mode)
        validate_period(as_of, period, mode)
        root, destination, temporary = validate_paths(args.raw_root, args.destination, args.temporary_root)
        configured = True
        channel = control if control is not None else sys.stdin.buffer
        Thread(target=_read_control, args=(channel, cancelled, invalid), daemon=True).start()
        Thread(target=heartbeat, daemon=True).start()
        with redirect_stdout(_Discard()), redirect_stderr(_Discard()):
            inspect_sources(root)
            if cancelled.is_set():
                raise InterruptedError
            _prepare_root(temporary, cancelled.is_set)
            result = export_detail_statistics(raw_root=root, destination=destination, temporary_root=temporary,
                as_of_date=as_of, event_period=period, order_mode=mode, progress=reporter.progress,
                cancelled=cancelled.is_set, reserve_bytes=args.reserve_bytes, max_temporary_bytes=args.max_temporary_bytes)
        reporter.terminal(dict(event='finished', rows=result.manifest['rows']))
        return 0
    except InterruptedError:
        reporter.terminal(dict(event='error', code='CONTROL_PROTOCOL_ERROR') if invalid.is_set() else dict(event='cancelled'))
        return 1 if invalid.is_set() else 2
    except Exception as error:
        if not configured and not isinstance(error, DetailExportError):
            error = DetailExportError('INVALID_OPTIONS')
        reporter.terminal(_safe_error(error, temporary, destination))
        return 1
    finally:
        heartbeat_stop.set()


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    raise SystemExit(main())
