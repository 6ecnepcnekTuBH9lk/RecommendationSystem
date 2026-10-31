"""One detail QProcess; bounded IPC, private stdin cancellation and reaping."""
from datetime import date
from pathlib import Path
import sys

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

from Application.paths import PROJECT_ROOT
from Application.statistics_detail_job import MAX_MESSAGE, allowed_cleanup_path, decode_message


class DetailExportProcess(QObject):
    progress = pyqtSignal(dict)
    completed = pyqtSignal(dict)
    stalled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.buffer = bytearray()
        self.terminal = None
        self.failure = None
        self.cancel_requested = False
        self.options = None
        self.discarded = 0
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.stalled)

    def start(self, options):
        if self.process is not None:
            return
        self.options = dict(options)
        self.discarded = 0
        self.buffer.clear()
        self.terminal = self.failure = None
        self.cancel_requested = False
        process = self.process = QProcess(self)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.started.connect(self._send_cancel)
        process.readyReadStandardOutput.connect(self._read)
        process.readyReadStandardError.connect(self._drain_stderr)
        process.errorOccurred.connect(self._error)
        process.finished.connect(self._finished)
        args = ['-u', '-B', str(PROJECT_ROOT/'scripts/statistics_detail_export.py')]
        for key, value in options.items():
            if value is not None:
                args += ['--'+key.replace('_', '-'), value.isoformat() if isinstance(value, date) else str(value)]
        process.start(sys.executable, args)

    def cancel(self):
        if self.process is not None and not self.cancel_requested:
            self.cancel_requested = True
            self._send_cancel()
            self.timer.start(30000)

    def _send_cancel(self):
        if self.process is not None and self.cancel_requested and self.process.state() == QProcess.ProcessState.Running:
            self.process.write(b'{"command":"cancel"}\n')

    def force_stop(self):
        if self.process is not None:
            self.failure = 'FORCED_STOP'
            self.buffer.clear()
            self.process.kill()  # explicit emergency action, never normal cancellation

    def _protocol_error(self):
        self.failure = 'WORKER_PROTOCOL_ERROR'
        self.buffer.clear()
        self.cancel()

    def _read(self):
        process = self.process
        if process is None:
            return
        process.setReadChannel(QProcess.ProcessChannel.StandardOutput)
        budget = 64*1024
        while process.bytesAvailable() and budget > 0:
            chunk = bytes(process.read(min(8192, process.bytesAvailable())))
            budget -= len(chunk)
            if self.failure:
                self._discard(len(chunk))
                continue
            for part in chunk.splitlines(keepends=True):
                self.buffer.extend(part)
                if len(self.buffer) > MAX_MESSAGE:
                    self._protocol_error()
                    break
                if self.buffer.endswith(b'\n'):
                    line = bytes(self.buffer)
                    self.buffer.clear()
                    try:
                        message = decode_message(line)
                        if self.terminal is not None:
                            raise ValueError
                        if 'technical_path' in message:
                            p = Path(message['technical_path'])
                            temporary = Path(self.options['temporary_root'])
                            destination = Path(self.options['destination'])
                            if not allowed_cleanup_path(p, temporary, destination):
                                raise ValueError
                        if message['event'] == 'progress':
                            self.progress.emit(message)
                        else:
                            self.terminal = message
                    except (ValueError, KeyError, TypeError, UnicodeError, RecursionError):
                        self._protocol_error()
                        break

        if process.bytesAvailable():
            QTimer.singleShot(0, self._read)

    def _discard(self, count):
        self.discarded += count
        if self.discarded > 1024*1024 and self.process is not None:
            # A flooding child cannot be trusted to cooperate. Emergency only.
            self.failure = self.failure or 'WORKER_PROTOCOL_ERROR'
            self.process.kill()

    def _drain_stderr(self):
        process = self.process
        if process is not None:
            process.setReadChannel(QProcess.ProcessChannel.StandardError)
            budget = 64*1024
            while process.bytesAvailable() and budget > 0:
                data = process.read(min(8192, process.bytesAvailable()))
                budget -= len(data)
                self._discard(len(data))
            remaining = process.bytesAvailable()
            process.setReadChannel(QProcess.ProcessChannel.StandardOutput)
            if remaining:
                QTimer.singleShot(0, self._drain_stderr)

    def _error(self, error):
        if self.process is not None and error == QProcess.ProcessError.FailedToStart:
            self.failure = 'FAILED_TO_START'
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _finished(self, code, status):
        if self.process is None:
            return
        self._read()
        self._drain_stderr()
        if self.buffer:
            self.failure = 'WORKER_PROTOCOL_ERROR'
        process, self.process = self.process, None
        self.timer.stop()
        terminal = self.terminal
        expected = {'finished': 0, 'cancelled': 2, 'error': 1}
        if self.failure:
            terminal = dict(event='error', code=self.failure)
        elif status != QProcess.ExitStatus.NormalExit or terminal is None or code != expected[terminal['event']]:
            terminal = dict(event='error', code='WORKER_CRASHED')
        process.close()
        process.deleteLater()
        self.completed.emit(terminal)

    def shutdown(self):
        """Last-resort app teardown only; normal close is deferred by the tab."""
        if self.process is not None:
            self.cancel()
            if self.process.waitForFinished(3000):
                return
            self.force_stop()
            if self.process is not None:
                self.process.waitForFinished(1000)
