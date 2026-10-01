"""Versioned analysis settings, separate from the last successful statistics."""

import json
import logging
import os
from pathlib import Path
import tempfile

from Application.analysis_filter import AnalysisFilter
from Application.paths import USER_SETTINGS_DIR

SETTINGS_PATH = USER_SETTINGS_DIR / "analysis_filter.json"
SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повторяющееся поле настроек.")
        result[key] = value
    return result


def load_filter(path=SETTINGS_PATH):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream, object_pairs_hook=_unique)
        if (not isinstance(data, dict) or set(data) != {"schema_version", "analysis_filter"}
                or type(data["schema_version"]) is not int or data["schema_version"] != SCHEMA_VERSION):
            raise ValueError
        return AnalysisFilter.from_dict(data["analysis_filter"])
    except FileNotFoundError:
        return AnalysisFilter()
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        logger.warning("Не удалось загрузить отбор для анализа: используются все данные; файл не изменён.")
        return AnalysisFilter()


def save_filter(selection, path=SETTINGS_PATH):
    payload = json.dumps({"schema_version": SCHEMA_VERSION, "analysis_filter": selection.to_dict()},
                         ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".analysis-filter-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            logger.warning("Не удалось удалить временный файл отбора.")
