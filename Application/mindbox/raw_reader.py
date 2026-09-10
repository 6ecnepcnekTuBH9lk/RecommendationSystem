"""Общее чтение опубликованных raw exports для profiler и adapters. Только локальный I/O."""

import json
import re
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = PROJECT_ROOT / "ВходныеДанные" / "MindboxRaw"
EXPORT_ROOTS = {
    "actions": "customerActions", "orders": "orders", "customers": "customers",
    "customer_merges": "customerMerges",
}
TIMESTAMP_PATTERN = re.compile(r"(\d{8}_\d{6})(?:_(\d{3,}))?")


class RawExportError(Exception):
    """Ошибка raw export без значений данных в диагностике."""


def part_files(directory: Path, export_name: str) -> list[Path]:
    pattern = re.compile(re.escape(export_name) + r"_part_(\d{3,})\.json")
    parts = []
    try:
        for path in directory.iterdir():
            if path.name.endswith((".tmp", ".download")):
                raise RawExportError("Каталог содержит незавершённые части экспорта")
            match = pattern.fullmatch(path.name)
            if match:
                if path.is_symlink() or not path.is_file():
                    raise RawExportError("Часть экспорта должна быть обычным файлом")
                parts.append((int(match[1]), path))
            elif path.suffix == ".json":
                raise RawExportError("Неожиданное имя JSON-файла в каталоге экспорта")
    except OSError:
        raise RawExportError("Не удалось прочитать каталог экспорта") from None
    parts.sort()
    if not parts:
        raise RawExportError("В каталоге нет part-файлов выбранного экспорта")
    if [number for number, _ in parts] != list(range(1, len(parts) + 1)):
        raise RawExportError("Нарушена последовательность номеров part-файлов")
    return [path for _, path in parts]


def select_export_directory(
    export_name: str, *, raw_root: str | Path = DEFAULT_RAW_ROOT,
    input_dir: str | Path | None = None,
) -> Path:
    """Явный каталог либо последний timestamp с числовым суффиксом коллизии."""
    if export_name not in EXPORT_ROOTS:
        raise RawExportError("Неизвестный тип экспорта")
    if input_dir is not None:
        directory = Path(input_dir)
        part_files(directory, export_name)
        return directory
    candidates = []
    try:
        for directory in (Path(raw_root) / export_name).iterdir():
            match = TIMESTAMP_PATTERN.fullmatch(directory.name)
            if not match or directory.is_symlink() or not directory.is_dir():
                continue
            try:
                stamp = datetime.strptime(match[1], "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            candidates.append((stamp, int(match[2] or 0), directory))
    except OSError:
        raise RawExportError("Не удалось найти каталог опубликованных экспортов") from None
    if not candidates:
        raise RawExportError("Нет опубликованных timestamp-каталогов выбранного экспорта")
    directory = max(candidates)[2]
    # Повреждённый последний экспорт не подменяем незаметно старым.
    part_files(directory, export_name)
    return directory


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RawExportError("JSON содержит повторяющиеся ключи объекта")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise RawExportError("JSON содержит недопустимую числовую константу")


def load_export_part(
    path: Path, export_name: str, part_number: int, *, parse_float: type = Decimal,
) -> list[dict[str, Any]]:
    if export_name not in EXPORT_ROOTS:
        raise RawExportError("Неизвестный тип экспорта")
    root = EXPORT_ROOTS[export_name]
    try:
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream, object_pairs_hook=_unique_object,
                             parse_constant=_reject_constant, parse_float=parse_float)
        if not isinstance(data, dict) or root not in data or not isinstance(data[root], list):
            raise RawExportError(f"Часть {part_number}: ожидается объект с массивом {root}")
        if set(data) != {root}:
            raise RawExportError(f"Часть {part_number}: неожиданные дополнительные поля корневого объекта")
        if any(not isinstance(item, dict) for item in data[root]):
            raise RawExportError(f"Часть {part_number}: элементы корневого массива должны быть объектами")
        return data[root]
    except json.JSONDecodeError as exc:
        raise RawExportError(
            f"Часть {part_number}: некорректный JSON, строка {exc.lineno}, столбец {exc.colno}"
        ) from None
    except (OSError, UnicodeError):
        raise RawExportError(f"Часть {part_number}: не удалось прочитать UTF-8 JSON") from None
    except (ValueError, InvalidOperation):
        raise RawExportError(f"Часть {part_number}: некорректное JSON число") from None
    except RecursionError:
        raise RawExportError(f"Часть {part_number}: превышена допустимая глубина JSON") from None


def iter_export(
    export_name: str, *, input_dir: str | Path | None = None,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
) -> Iterator[dict[str, Any]]:
    """Yield объектов по порядку частей; дробные JSON numbers читаются точно как Decimal."""
    directory = select_export_directory(export_name, raw_root=raw_root, input_dir=input_dir)
    for number, path in enumerate(part_files(directory, export_name), start=1):
        yield from load_export_part(path, export_name, number)
