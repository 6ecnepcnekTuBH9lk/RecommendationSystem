"""Validated last-successful statistics snapshot; no Qt or canonical writes."""

from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
import logging
import math
import os
from pathlib import Path
import tempfile

from Application.paths import USER_SETTINGS_DIR


CACHE_PATH = USER_SETTINGS_DIR / "dataset_statistics.json"
logger = logging.getLogger(__name__)

COUNT_FIELDS = (
    "actions", "orders", "order_lines", "action_customers", "order_customers", "interaction_users",
    "actions_with_product", "actions_without_product", "view_interactions", "favorite_interactions",
    "purchase_interactions", "unique_source_products", "unique_resolved_items",
    "resolved_interactions", "unresolved_interactions",
)
DIAGNOSTIC_KEYS = (
    "mapped_view_actions", "mapped_favorite_actions", "unmapped_actions", "mapped_without_product",
    "actions_without_customer_id", "unknown_candidate", "unsupported_namespace", "invalid_id",
    "orders_unique", "orders_duplicate_identical", "orders_duplicate_conflicting", "unique_order_lines",
    "purchase_lines", "filtered_by_status",
)


class StatisticsCacheError(ValueError):
    """Safe validation error for both subprocess messages and saved snapshots."""


def parse_timestamp(value):
    if not isinstance(value, str):
        raise StatisticsCacheError("Некорректная дата результата статистики.")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError
        # Also verify conversion used by the presentation layer (range/offset).
        result.astimezone()
        return result
    except (ValueError, OverflowError, OSError):
        raise StatisticsCacheError("Некорректная дата результата статистики.") from None


def _count(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _rows(value, checks):
    return isinstance(value, (list, tuple)) and all(
        isinstance(row, (list, tuple)) and len(row) == len(checks)
        and all(check(cell) for check, cell in zip(checks, row)) for row in value)


def validate_result(result):
    """Validate the complete display contract without recalculating statistics."""
    text = lambda value: isinstance(value, str)
    if not isinstance(result, dict):
        raise StatisticsCacheError("Неполный или некорректный результат статистики.")
    try:
        valid = all(_count(result[key]) for key in COUNT_FIELDS)
        valid &= result["customers"] is None or _count(result["customers"])
        valid &= all(_number(result[key]) for key in ("mean_interactions", "median_interactions", "resolution_rate"))
        valid &= result["resolution_rate"] <= 100
        valid &= isinstance(result["purchase_quantity"], str) and Decimal(result["purchase_quantity"]).is_finite()
        for key, checks in (
            ("action_types", (text, _count)),
            ("line_statuses", (text, _count, lambda value: type(value) is bool)),
            ("namespaces", (text, _count, _count, _count, lambda value: _number(value) and value <= 100)),
            ("top_products", (text, text, _count, _count, _count, _count)),
            ("diagnostics", (text, _count)),
        ):
            valid &= _rows(result[key], checks)
        if not valid:
            raise ValueError
        keys = [key for key, _ in result["diagnostics"]]
        if len(keys) != len(set(keys)) or set(keys) != set(DIAGNOSTIC_KEYS):
            raise ValueError
        if not isinstance(result["warnings"], (list, tuple)) or not all(text(v) for v in result["warnings"]):
            raise ValueError
        parse_timestamp(result["calculated_at"])
        if not isinstance(result["coverage"], (list, tuple)) or len(result["coverage"]) != 4:
            raise ValueError
        sources = set()
        for source in result["coverage"]:
            if not isinstance(source, dict) or not text(source["source"]):
                raise ValueError
            sources.add(source["source"])
            if (not _rows(source["intervals"], (text, text))
                    or not isinstance(source["source_kinds"], (list, tuple))
                    or not all(text(v) for v in source["source_kinds"])):
                raise ValueError
            for start, end in source["intervals"]:
                if parse_timestamp(start) >= parse_timestamp(end):
                    raise ValueError
            if source["updated"] is not None:
                parse_timestamp(source["updated"])
        if sources != {"Actions", "Orders", "CustomerMerges", "Customers"}:
            raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError, InvalidOperation):
        raise StatisticsCacheError("Неполный или некорректный результат статистики.") from None
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StatisticsCacheError("Повторяющиеся поля в сохранённом результате.")
        result[key] = value
    return result


def load_result(path):
    """Missing cache is normal; damaged cache is logged and left untouched."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            saved = json.load(stream, object_pairs_hook=_unique_object)
        if (not isinstance(saved, dict) or type(saved.get("schema_version")) is not int
                or saved["schema_version"] != 1):
            raise StatisticsCacheError("Неподдерживаемый формат сохранённой статистики.")
        return validate_result(saved.get("result"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError, RecursionError):
        logger.warning("Не удалось загрузить сохранённую статистику: файл недоступен или повреждён.")
        return None


def save_result(path, result):
    """Replace one JSON only after validation, serialization, flush and fsync."""
    validate_result(result)
    payload = json.dumps({"schema_version": 1, "result": result}, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".dataset-statistics-", suffix=".tmp", dir=path.parent)
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
            logger.warning("Не удалось удалить временный файл статистики; сохранённый результат не повреждён.")
