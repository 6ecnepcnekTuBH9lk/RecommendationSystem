"""Validated last-successful statistics snapshot; no Qt or canonical writes."""

from datetime import datetime
from decimal import Decimal, DecimalException, InvalidOperation
import json
import logging
import math
import os
from pathlib import Path
import tempfile

from Application.paths import USER_SETTINGS_DIR
from Application.order_statistics import BASKET_LABELS, CURRENCIES
from Application.action_statistics import AVAILABILITY_LABELS, FAVORITE_BUCKETS, VIEW_BUCKETS


CACHE_PATH = USER_SETTINGS_DIR / "dataset_statistics.json"
SCHEMA_VERSION = 6
logger = logging.getLogger(__name__)

COUNT_FIELDS = (
    "actions", "orders", "order_lines", "action_customers", "order_customers", "interaction_users",
    "actions_with_product", "actions_without_product", "view_interactions", "favorite_interactions",
    "purchase_interactions", "unique_source_products", "unique_resolved_items",
    "resolved_interactions", "unresolved_interactions",
    "view_users", "favorite_users", "purchase_users", "view_purchase_users", "favorite_purchase_users",
    "all_interaction_type_users", "repeat_buyers",
    "purchase_orders", "mixed_currency_purchase_orders", "unknown_currency_purchase_orders",
    "view_parameter_actions",
    "products_with_views", "products_with_favorites", "products_with_purchases",
    "resolved_view_interactions", "resolved_favorite_interactions", "resolved_purchase_interactions",
)
DIAGNOSTIC_KEYS = (
    "ambiguous_product_view_actions", "unknown_currency_view_price_actions",
    "orders_outside_statistics_period", "fractional_quantity_order_lines",
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


def _decimal_string(value):
    if not isinstance(value, str):
        return False
    try:
        parsed = Decimal(value)
        return parsed.is_finite() and parsed >= 0
    except InvalidOperation:
        return False


def _validate_orders(result):
    text = lambda value: isinstance(value, str) and bool(value.strip())
    currency = lambda value: value in CURRENCIES
    money = _decimal_string
    total = result["purchase_orders"]
    for key in ("mean_purchase_lines_per_order", "median_purchase_lines_per_order"):
        if not _number(result[key]):
            raise ValueError
    for key in ("mean_purchase_units_per_order", "median_purchase_units_per_order"):
        if not money(result[key]):
            raise ValueError
    for key in ("purchase_basket_distribution", "ordering_method_distribution", "delivery_type_distribution", "payment_type_distribution"):
        rows = result[key]
        if not _rows(rows, (text, _count, _number)) or len({r[0] for r in rows}) != len(rows):
            raise ValueError
        if key == "purchase_basket_distribution" and tuple(r[0] for r in rows) != BASKET_LABELS:
            raise ValueError
        if key != "payment_type_distribution" and sum(r[1] for r in rows) != total:
            raise ValueError
        if key == "payment_type_distribution" and sum(r[1] for r in rows) < total:
            raise ValueError
        if any(n > total or not math.isclose(rate, 100 * n / total if total else 0., abs_tol=1e-9) for _, n, rate in rows):
            raise ValueError
    financials = result["order_financials"]
    delivery = result["delivery_financials"]
    if not _rows(financials, (currency, _count, _count, money, money, money, money)):
        raise ValueError
    if not _rows(delivery, (currency, _count, money, money, money)):
        raise ValueError
    if tuple(r[0] for r in financials) != CURRENCIES or tuple(r[0] for r in delivery) != CURRENCIES:
        raise ValueError
    if sum(r[1] for r in financials) + result["mixed_currency_purchase_orders"] + result["unknown_currency_purchase_orders"] != total:
        raise ValueError
    for row, shipping in zip(financials, delivery):
        _, n, lines, units, amount, mean, middle = row
        if lines < n or (not n and any(Decimal(v) for v in (units, amount, mean, middle))):
            raise ValueError
        if Decimal(mean) != (Decimal(amount) / n if n else 0):
            raise ValueError
        if shipping[1] > n or Decimal(shipping[3]) != (Decimal(shipping[2]) / shipping[1] if shipping[1] else 0):
            raise ValueError
        if not shipping[1] and any(Decimal(v) for v in shipping[2:]):
            raise ValueError
    monthly = result["order_monthly_dynamics"]
    if not _rows(monthly, (text, _count, money, _count, money)):
        raise ValueError
    months = [r[0] for r in monthly]
    if months != sorted(set(months)):
        raise ValueError
    for month in months:
        parsed = datetime.strptime(month, "%Y-%m")
        if parsed.strftime("%Y-%m") != month:
            raise ValueError
    for i, row in enumerate(financials):
        if sum(r[1 + 2 * i] for r in monthly) != row[1]:
            raise ValueError
        if sum((Decimal(r[2 + 2 * i]) for r in monthly), Decimal(0)) != Decimal(row[4]):
            raise ValueError
    stores = result["store_statistics"]
    if not _rows(stores, (currency, lambda v: v is None or text(v), text, _count, _count, _count, money, money, money)):
        raise ValueError
    if len({(r[0], r[1]) for r in stores}) != len(stores):
        raise ValueError
    for unit in CURRENCIES:
        rows = [r for r in stores if r[0] == unit]
        if len(rows) > 20 or rows != sorted(rows, key=lambda r: (-Decimal(r[7]), -r[3], r[2], r[1] is not None, r[1] or "")):
            raise ValueError
        for row in rows:
            if not 0 < row[4] <= row[3] <= total or row[5] < row[3] or Decimal(row[8]) != Decimal(row[7]) / row[3]:
                raise ValueError


def _validate_actions(result):
    text = lambda value: isinstance(value, str) and bool(value.strip())
    rate = lambda value: _number(value) and value <= 100
    views, favorites = result["view_interactions"], result["favorite_interactions"]
    for prefix, suffix, total, users in (("views", "viewer", views, result["view_users"]),
                                         ("favorites", "user", favorites, result["favorite_users"])):
        mean, middle = (result[f"{stat}_{prefix}_per_{suffix}"] for stat in ("mean", "median"))
        if not _number(mean) or not _number(middle) or not 0 <= users <= total or bool(users) != bool(total):
            raise ValueError
        if not math.isclose(mean, total / users if users else 0., abs_tol=1e-9):
            raise ValueError
        if (not users and middle != 0) or (users and not 1 <= middle <= total):
            raise ValueError
    for field, labels, total in (
        ("view_user_activity_distribution", VIEW_BUCKETS, result["view_users"]),
        ("favorite_user_activity_distribution", FAVORITE_BUCKETS, result["favorite_users"]),
        ("view_availability_distribution", AVAILABILITY_LABELS, result["view_parameter_actions"]),
    ):
        rows = result[field]
        if not _rows(rows, (text, _count, rate)) or tuple(row[0] for row in rows) != labels:
            raise ValueError
        if sum(row[1] for row in rows) != total or any(
                not math.isclose(r, 100 * n / total if total else 0., abs_tol=1e-9) for _, n, r in rows):
            raise ValueError
    channels = result["action_channel_statistics"]
    if not _rows(channels, (lambda v: v is None or text(v), text, _count, _count, rate, _count, _count, rate)):
        raise ValueError
    if len({r[0] for r in channels}) != len(channels) or list(channels) != sorted(
            channels, key=lambda r: (-r[2], -r[5], r[1], r[0] is not None, r[0] or "")):
        raise ValueError
    for index, total, users in ((2, views, result["view_users"]), (5, favorites, result["favorite_users"])):
        if sum(r[index] for r in channels) != total:
            raise ValueError
        for row in channels:
            n, unique, percent = row[index:index + 3]
            if unique > min(n, users) or bool(unique) != bool(n) or not math.isclose(
                    percent, 100 * n / total if total else 0., abs_tol=1e-9):
                raise ValueError
    monthly = result["action_monthly_dynamics"]
    if not _rows(monthly, (text, _count, _count, _count, _count)):
        raise ValueError
    months = [r[0] for r in monthly]
    if months != sorted(set(months)):
        raise ValueError
    for month in months:
        parsed = datetime.strptime(month, "%Y-%m")
        if month != f"{parsed.year:04d}-{parsed.month:02d}":
            raise ValueError
    for index, total, users in ((1, views, result["view_users"]), (3, favorites, result["favorite_users"])):
        if sum(r[index] for r in monthly) != total:
            raise ValueError
        if any(r[index + 1] > min(r[index], users) or bool(r[index + 1]) != bool(r[index]) for r in monthly):
            raise ValueError
    prices = result["view_price_statistics"]
    if not _rows(prices, (text, _count, _decimal_string, _decimal_string)) or tuple(r[0] for r in prices) != CURRENCIES:
        raise ValueError
    if any(not n and (Decimal(mean) != 0 or Decimal(middle) != 0) for _, n, mean, middle in prices):
        raise ValueError
    parameters = result["view_parameter_actions"]
    diagnostics = dict(result["diagnostics"])
    if sum(r[1] for r in prices) + diagnostics["unknown_currency_view_price_actions"] > parameters:
        raise ValueError
    if parameters + 2 * diagnostics["ambiguous_product_view_actions"] > views:
        raise ValueError


def _integral_quantity(value):
    return _decimal_string(value) and Decimal(value) == Decimal(value).to_integral_value()


def _validate_products(result):
    text = lambda value: isinstance(value, str)
    label = lambda value: text(value) and bool(value.strip())
    items = result["unique_resolved_items"]
    totals = tuple(result[f"resolved_{kind}_interactions"] for kind in ("view", "favorite", "purchase"))
    if sum(totals) != result["resolved_interactions"]:
        raise ValueError
    for key, total, global_key in zip(("products_with_views", "products_with_favorites", "products_with_purchases"),
                                      totals, ("view_interactions", "favorite_interactions", "purchase_interactions")):
        if not 0 <= result[key] <= min(items, total) or total > result[global_key] or bool(result[key]) != bool(total):
            raise ValueError
    quantity = result["resolved_purchase_quantity"]
    if not _integral_quantity(quantity) or Decimal(quantity) > Decimal(result["purchase_quantity"]):
        raise ValueError
    if not totals[2] and Decimal(quantity):
        raise ValueError
    for field, count_field, purchased in (
        ("top_viewed_products", "products_with_views", False),
        ("top_favorited_products", "products_with_favorites", False),
        ("top_purchased_products", "products_with_purchases", True),
    ):
        rows = result[field]
        checks = (label, text, _count, _count, _integral_quantity, _count, _count) if purchased else (label, text, _count, _count, _count, _count)
        if not _rows(rows, checks) or len(rows) != min(20, result[count_field]) or len({r[0] for r in rows}) != len(rows):
            raise ValueError
        key = (lambda r: (-r[2], -Decimal(r[4]), -r[3], r[0])) if purchased else (lambda r: (-r[2], -r[3], -r[5], r[0]))
        if list(rows) != sorted(rows, key=key) or any(not 0 < r[3] <= r[2] for r in rows):
            raise ValueError
        positions = (5, 6, 2) if purchased else (2, 4, 5) if field == "top_viewed_products" else (4, 2, 5)
        if any(sum(r[index] for r in rows) > total for index, total in zip(positions, totals)):
            raise ValueError
        if purchased and sum((Decimal(r[4]) for r in rows), Decimal(0)) > Decimal(quantity):
            raise ValueError
    for field in ("product_category_statistics", "product_gender_statistics", "product_season_statistics", "product_style_statistics"):
        rows = result[field]
        if not _rows(rows, (label, _count, _count, _count, _count, _integral_quantity)):
            raise ValueError
        if len({r[0] for r in rows}) != len(rows) or any(not r[1] or r[1] > sum(r[2:5]) for r in rows):
            raise ValueError
        if list(rows) != sorted(rows, key=lambda r: (-r[4], -r[2], -r[3], r[0])):
            raise ValueError
        if any(sum(r[index] for r in rows) != total for index, total in enumerate((items, *totals), 1)):
            raise ValueError
        if sum((Decimal(r[5]) for r in rows), Decimal(0)) != Decimal(quantity):
            raise ValueError


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
        valid &= all(result[key] is None or _number(result[key]) for key in ("mean_age", "median_age"))
        valid &= all(_number(result[key]) for key in ("mean_orders_per_buyer", "median_orders_per_buyer"))
        valid &= all(_number(result[key]) and result[key] <= 100 for key in ("repeat_buyer_rate", "active_buyer_rate"))
        for key, length, total in (
            ("gender_distribution", 3, result["customers"] or 0),
            ("age_distribution", 8, result["customers"] or 0),
            ("interaction_activity_distribution", 7, result["interaction_users"]),
            ("purchase_order_distribution", 5, result["purchase_users"]),
        ):
            rows = result[key]
            if not _rows(rows, (text, _count, lambda value: _number(value) and value <= 100)):
                raise ValueError
            valid &= len(rows) == length and len({row[0] for row in rows}) == length
            valid &= sum(row[1] for row in rows) == total
            valid &= all(math.isclose(rate, 100 * count / total if total else 0.0, abs_tol=1e-9)
                         for _, count, rate in rows)
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
        _validate_orders(result)
        _validate_actions(result)
        _validate_products(result)
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
    except (KeyError, TypeError, ValueError, OverflowError, DecimalException):
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
                or saved["schema_version"] != SCHEMA_VERSION):
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
    payload = json.dumps({"schema_version": SCHEMA_VERSION, "result": result}, ensure_ascii=False, allow_nan=False)
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
