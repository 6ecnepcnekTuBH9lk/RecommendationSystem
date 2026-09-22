"""Safe error records shared by data-loading subprocesses and Qt workers."""

from datetime import datetime
import json
from urllib.parse import quote


SOURCES = {"actions", "orders", "actions/orders", "customer_merges", "customers", "reference_csv", "data"}


def safe_message(value):
    """Normalize structured messages without hiding exception details."""
    return "" if value is None else str(value).strip()


def error_record(exc, *, source="data", category="failed", since=None, secrets=()):
    kind = type(exc).__name__
    message = str(exc).strip()
    for secret in secrets:
        if isinstance(secret, str) and secret:
            for variant in (secret, quote(secret, safe=""), json.dumps(secret)[1:-1]):
                message = message.replace(variant, "[секрет скрыт]")
    return {"category": category, "source": source if source in SOURCES else "data",
            "error_type": kind, "message": safe_message(message), "since": since}


def emit_error(exc, **context):
    print("Error: " + json.dumps(error_record(exc, **context), ensure_ascii=True), flush=True)


def format_error(record, stage=None, *, context=None):
    """Validate even structured subprocess output; unknown stdout stays invisible."""
    if not isinstance(record, dict):
        return "Ошибка: некорректное описание причины от процесса."
    source = record.get("source")
    if (source is None or source == "data") and stage in ("customers", "snapshot_validation"):
        source = "customers"
    title = {"actions": "действий", "orders": "заказов", "actions/orders": "действий и заказов",
             "customer_merges": "объединений клиентов", "customers": "клиентов"}.get(source if isinstance(source, str) else "", "данных")
    if context is None:
        if stage == "reference_csv":
            context = "Ошибка импорта справочника"
        elif stage in ("manual_customers", "manual_interactions"):
            context = "Ошибка импорта " + ("клиентов" if stage == "manual_customers" else "действий и заказов")
        else:
            context = ("Превышено время ожидания выгрузки " if record.get("category") == "timeout"
                       else "Ошибка выгрузки ") + title
            try:
                period = datetime.fromisoformat(record["since"]).strftime("%m.%Y" if source == "customers" else "%d.%m.%Y")
                context += " за " + period
            except (KeyError, TypeError, ValueError):
                pass
    kind = safe_message(record.get("error_type"))
    message = safe_message(record.get("message"))
    detail = f"{kind}: {message}" if kind and message else message or kind or "Неизвестная ошибка."
    return context + ": " + detail
