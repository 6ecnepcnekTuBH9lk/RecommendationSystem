"""Reference-only CSV snapshot processing; callable without a QApplication."""

import os
import logging
from pathlib import Path
import tempfile

import chardet
import pandas as pd

from .files_processing import process_nomenclature_file, process_categories_file, process_coordinates_file


REFERENCE_TYPES = {
    "Номенклатура из 1С": ("Номенклатура.csv", "|", process_nomenclature_file),
    "Категории сайта из 1С": ("КатегорииСайта.csv", "|", process_categories_file),
    "Координаты городов и погода": ("КоординатыГородов.csv", ",", process_coordinates_file),
}


def import_reference(source, kind, *, output_dir):
    if kind not in REFERENCE_TYPES:
        raise ValueError("Unsupported reference type")
    filename, separator, processor = REFERENCE_TYPES[kind]
    with Path(source).open("rb") as stream:
        encoding = chardet.detect(stream.read(200000)).get("encoding") or "utf-8"
    frame = processor(None, pd.read_csv(source, sep=separator, encoding=encoding))
    result = {"kind": kind}

    def values(column):
        return sorted({str(value).strip() for value in frame[column].dropna()
                       if str(value).strip() and str(value).strip().lower() not in ("nan", "none", "null", "<na>", "-")})

    if kind == "Номенклатура из 1С":
        result.update(kinds=values("ВидНоменклатуры"), seasons=values("Коллекция"))
    elif kind == "Координаты городов и погода":
        result["cities"] = values("Город")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".reference-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
            frame.to_csv(stream, index=False, sep="|")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / filename)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            logging.getLogger(__name__).warning("Не удалось удалить временный CSV; итоговый файл не повреждён.")
    return result
