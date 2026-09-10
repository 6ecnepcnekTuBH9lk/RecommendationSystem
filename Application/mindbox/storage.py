"""Хранение raw-байтов; здесь нет HTTP и преобразования бизнес-данных."""

import gzip
import logging
import os
import shutil
import tempfile
import zlib
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from .exceptions import MindboxDownloadError


DEFAULT_RAW_ROOT = Path(__file__).resolve().parents[2] / "ВходныеДанные" / "MindboxRaw"
EXPORT_NAMES = ("actions", "orders", "customers", "customer_merges")
COPY_BUFFER_SIZE = 1024 * 1024
logger = logging.getLogger(__name__)


class RawExportStorage:
    def __init__(self, root: str | Path = DEFAULT_RAW_ROOT) -> None:
        self.root = Path(root)

    def save_export(
        self,
        export_name: str,
        urls: Sequence[str],
        download: Callable[[str, Path], None],
    ) -> list[Path]:
        """download пишет транспортные байты в заданный временный путь.

        Повторные попытки download должны начинать запись файла с нуля.
        Публикуем каталог только после получения и распаковки всех частей.
        """
        if export_name not in EXPORT_NAMES:
            raise MindboxDownloadError("Неизвестное имя raw-экспорта")
        if not urls or isinstance(urls, (str, bytes)):
            raise MindboxDownloadError("Экспорт не содержит списка URL частей")
        staging = None
        try:
            parent = self.root / export_name
            parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
            names = []
            for index, url in enumerate(urls, start=1):
                name = f"{export_name}_part_{index:03d}.json"
                wire = staging / (name + ".download")
                temporary = staging / (name + ".tmp")
                download(url, wire)
                self._unpack(wire, temporary)
                os.replace(temporary, staging / name)
                wire.unlink()
                names.append(name)
            published = self._publish(staging, parent, timestamp)
            return [published / name for name in names]
        except (OSError, EOFError, zlib.error) as exc:
            # Не включаем содержимое поврежденного gzip или сетевые URL.
            raise MindboxDownloadError(
                f"Не удалось сохранить raw-экспорт ({type(exc).__name__}, "
                f"errno={getattr(exc, 'errno', None)}); выгрузка не опубликована"
            ) from None
        finally:
            if staging is not None and staging.exists():
                # Только собственный каталог, созданный mkdtemp, включая Ctrl+C.
                try:
                    shutil.rmtree(staging)
                except OSError:
                    logger.warning("Не удалось удалить незавершённый .staging-каталог Mindbox")

    @staticmethod
    def _unpack(wire: Path, temporary: Path) -> None:
        with wire.open("rb") as source, temporary.open("xb") as target:
            if source.peek(2)[:2] == b"\x1f\x8b":
                with gzip.GzipFile(fileobj=source, mode="rb") as decoded:
                    shutil.copyfileobj(decoded, target, COPY_BUFFER_SIZE)
            else:
                shutil.copyfileobj(source, target, COPY_BUFFER_SIZE)
            if target.tell() == 0:
                raise MindboxDownloadError("Получена пустая часть экспорта; выгрузка не опубликована")
            target.flush()
            os.fsync(target.fileno())

    @staticmethod
    def _publish(staging: Path, parent: Path, timestamp: str) -> Path:
        # mkdir — межпроцессное резервирование имени. Повтор в ту же секунду
        # получает суффикс, прежние выгрузки никогда не заменяются.
        index = 0
        while True:
            name = timestamp if index == 0 else f"{timestamp}_{index:03d}"
            target = parent / name
            reservation = parent / ("." + name + ".reserve")
            index += 1
            try:
                reservation.mkdir()
            except FileExistsError:
                continue
            try:
                if target.exists():
                    continue
                os.rename(staging, target)
                return target
            finally:
                try:
                    reservation.rmdir()
                except OSError:
                    # Ошибка уборки не меняет результат уже завершённой публикации
                    # и не должна скрывать исходную ошибку rename при неудаче.
                    logger.warning("Не удалось удалить служебный .reserve-каталог Mindbox")
