"""Independent, cooperative export process; no model or source I/O in Qt."""

import json
from pathlib import Path
import sys

from PyQt6.QtCore import QObject, QEvent, QProcess, QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
    QWidget,
)


class RecommendationExportController(QObject):
    def __init__(self, tab):
        super().__init__(QApplication.instance())
        self.tab = tab
        self.process = None
        self.dialog = None
        self.window = None
        self.close_pending = False
        self.cancelling = False
        self.buffer = b""
        self.outcome = None
        tab.destroyed.connect(self._tab_destroyed)
        QApplication.instance().aboutToQuit.connect(self._shutdown)

    def start(self):
        if self.process is not None or self.tab is None:
            return
        modes = (
            "Пробный экспорт — первые 50 клиентов BPR-MF",
            "Полный экспорт — клиенты BPR-MF",
        )
        choice, accepted = QInputDialog.getItem(
            self.tab,
            "Экспорт рекомендаций",
            "Аудитория BPR-MF. Полный экспорт может занять несколько часов.",
            modes,
            0,
            False,
        )
        if not accepted:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self.tab,
            "Сохранить Recommendations.zip",
            "Recommendations.zip",
            "ZIP-архив (*.zip)",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not filename:
            return
        destination = Path(filename)
        if destination.name != "Recommendations.zip":
            QMessageBox.warning(
                self.tab, "Экспорт", "Имя архива должно быть Recommendations.zip."
            )
            return
        overwrite = destination.exists()
        if (
            overwrite
            and QMessageBox.question(
                self.tab,
                "Заменить архив?",
                "Заменить существующий Recommendations.zip?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.destination = destination
        self.close_pending = self.cancelling = False
        self.outcome = None
        self.buffer = b""
        self.window = QWidget.window(self.tab)
        self.window.installEventFilter(self)
        self.tab.export_button.setEnabled(False)
        self.dialog = QProgressDialog(
            "Подготовка источников…", "Отменить экспорт", 0, 0, self.window
        )
        self.dialog.setWindowTitle("Экспорт рекомендаций")
        self.dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)
        self.dialog.canceled.connect(self.cancel)
        self.dialog.show()
        process = self.process = QProcess(self)
        process.readyReadStandardOutput.connect(self._read)
        process.readyReadStandardError.connect(
            lambda: process.readAllStandardError()
        )  # Never echo source error payloads.
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        project = Path(__file__).resolve().parents[2]
        process.setWorkingDirectory(str(project))
        args = [
            "-B",
            "-X",
            "utf8",
            str(project / "scripts" / "export_recommendations.py"),
            "--destination",
            str(destination),
            "--audience",
            "bpr",
        ]
        args += ["--full"] if choice == modes[1] else ["--limit", "50"]
        if overwrite:
            args.append("--overwrite")
        process.start(sys.executable, args)

    def cancel(self):
        if self.process is None or self.cancelling:
            return
        self.cancelling = True
        self.process.write(b"cancel\n")
        if self.dialog is not None:
            self.dialog.setLabelText(
                "Отмена экспорта: ожидание очистки временных файлов…"
            )
            self.dialog.setCancelButton(None)
            self.dialog.show()

    def _read(self):
        if self.process is None:
            return
        self.buffer += bytes(self.process.readAllStandardOutput())
        if len(self.buffer) > 1_048_576:
            self.buffer = b""
            self.outcome = {"type": "error", "code": "INVALID_PROTOCOL"}
            self.cancel()
            return
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                payload = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(payload, dict):
                continue
            kind = payload.get("type")
            if kind in ("success", "error", "cancelled"):
                self.outcome = payload
            elif kind == "progress" and self.dialog is not None and not self.cancelling:
                phase = {
                    "prepare": "Подготовка источников",
                    "customers": "Расчёт рекомендаций",
                    "xlsx": "Запись Excel",
                    "verify": "Проверка архива",
                }.get(payload.get("phase"), "Экспорт")
                processed, total = payload.get("processed", 0), payload.get("total", 0)
                if (
                    type(processed) is int
                    and type(total) is int
                    and 0 <= processed <= total <= 2**31 - 1
                ):
                    self.dialog.setRange(0, total)
                    self.dialog.setValue(processed)
                    self.dialog.setLabelText(
                        f"{phase}: {processed:,} / {total:,} покупателей"
                    )

    def _error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.outcome = {"type": "error", "code": "PROCESS_START_FAILED"}
            self._finished(-1, QProcess.ExitStatus.CrashExit)

    def _finished(self, code, status):
        if self.process is None:
            return
        self._read()
        process, self.process = self.process, None
        process.deleteLater()
        if self.dialog is not None:
            self.dialog.close()
            self.dialog.deleteLater()
            self.dialog = None
        if self.window is not None:
            self.window.removeEventFilter(self)
        if self.tab is not None:
            self.tab.export_button.setEnabled(True)
        else:
            self.deleteLater()
            return
        result = self.outcome or {"type": "error"}
        if self.close_pending and self.window is not None:
            QTimer.singleShot(0, self.window.close)
        elif self.tab is not None:
            if (
                code == 0
                and status == QProcess.ExitStatus.NormalExit
                and result.get("type") == "success"
            ):
                summary = result.get("summary", {})
                counts = [
                    summary.get(k)
                    for k in (
                        "customers",
                        "online_rows",
                        "mindbox_rows",
                        "raw_rows",
                        "hybrid_rows",
                    )
                ]
                if all(type(n) is int and n >= 0 for n in counts):
                    QMessageBox.information(
                        self.tab,
                        "Экспорт завершён",
                        f"Сохранён {self.destination}\nПокупателей: {counts[0]}\n"
                        f"Online Store: {counts[1]} строк\nMindbox: {counts[2]} строк\n"
                        f"Excel: {counts[0]} покупателей, {counts[3]} BPR и {counts[4]} hybrid строк.",
                    )
                    return
            if code == 2 and result.get("type") == "cancelled":
                QMessageBox.information(
                    self.tab,
                    "Экспорт отменён",
                    "Построение архива отменено, временные файлы очищены.",
                )
            else:
                messages = {
                    "SOURCES_CHANGED": "Источники изменились. Архив не опубликован.",
                    "PROPENSITY_SOURCES_CHANGED": "Артефакты прогноза изменились. Архив не опубликован.",
                    "INSUFFICIENT_DISK": "Недостаточно свободного места.",
                    "RESOURCE_OR_STORAGE_GUARD": "Сработала защита памяти или дискового пространства.",
                    "DESTINATION_EXISTS": "Файл назначения уже существует.",
                    "PROCESS_START_FAILED": "Не удалось запустить процесс экспорта.",
                }
                QMessageBox.warning(
                    self.tab,
                    "Ошибка экспорта",
                    messages.get(
                        result.get("code"),
                        "Экспорт завершился ошибкой. Готовый архив не подтверждён; повторный запуск не выполнялся.",
                    ),
                )

    def eventFilter(self, watched, event):
        if (
            watched is self.window
            and event.type() == QEvent.Type.Close
            and self.process is not None
        ):
            self.close_pending = True
            self.cancel()
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _tab_destroyed(self):
        self.tab = None
        self.dialog = None
        # Tab deletion is also cooperative; the application owns this controller
        # and process until finished, preventing QProcess destruction while running.
        self.cancel()
        self.window = None
        if self.process is None:
            self.deleteLater()

    def _shutdown(self):
        if self.process is not None:
            self.cancel()
            # QApplication.quit() can bypass the window close event. Retain the
            # process until its cooperative cleanup completes; never terminate it.
            self.process.waitForFinished(-1)


def start_recommendation_export(tab):
    if not hasattr(tab, "_export_controller"):
        tab._export_controller = RecommendationExportController(tab)
    tab._export_controller.start()
