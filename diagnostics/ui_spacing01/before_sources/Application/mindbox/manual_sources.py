"""Shared normalization of manually selected export parts."""

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
import re


class ManualImportError(ValueError):
    """Validation categories without source paths or record values."""


def natural_path_key(path):
    # Tagged segments avoid comparing str with int; full path breaks filename ties.
    parts = tuple((1, int(part)) if part.isdecimal() else (0, part.casefold())
                  for part in re.split(r"([0-9]+)", path.name))
    return parts, path.name, path.as_posix()


def normalize_sources(sources):
    """Accept one path or a sequence, validate files, return a sorted immutable tuple."""
    if isinstance(sources, (str, PathLike)):
        sources = (sources,)
    if not isinstance(sources, Sequence) or not sources:
        raise ManualImportError("Выберите хотя бы один JSON-файл для ручного импорта.")
    paths = []
    seen = set()
    for source in sources:
        if not isinstance(source, (str, PathLike)) or not str(source).strip():
            raise ManualImportError("Некорректный путь выбранного JSON-файла.")
        try:
            path = Path(source).resolve()
            if not path.is_file():
                raise ManualImportError("Выбранный JSON-файл отсутствует или не является обычным файлом.")
        except (OSError, RuntimeError):
            raise ManualImportError("Не удалось проверить выбранный JSON-файл.") from None
        if path in seen:
            raise ManualImportError("Один и тот же JSON-файл выбран несколько раз.")
        seen.add(path)
        paths.append(path)
    return tuple(sorted(paths, key=natural_path_key))
