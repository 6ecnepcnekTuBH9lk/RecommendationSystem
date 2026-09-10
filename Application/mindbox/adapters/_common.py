"""Общие проверки типов. В исключения попадают только статические пути полей."""

import re
from collections.abc import Mapping
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from ..records import ProductKey


class AdapterError(Exception):
    """Нарушение ожидаемого контракта raw объекта."""


def get(raw: Mapping[str, Any], path: str, *, required: bool = False) -> Any:
    current: Any = raw
    for part in path.split("."):
        if current is None:
            break
        if not isinstance(current, Mapping):
            raise AdapterError(f"{path}: ожидается вложенный object")
        current = current.get(part)
    if required and current is None:
        raise AdapterError(f"{path}: отсутствует обязательное поле")
    return current


def text(raw: Mapping[str, Any], path: str, *, required: bool = False) -> str | None:
    value = get(raw, path, required=required)
    if value is None:
        return None
    if not isinstance(value, str) or (required and not value.strip()):
        raise AdapterError(f"{path}: ожидается строка")
    return value


def identifier(raw: Mapping[str, Any], path: str, *, required: bool = True) -> str | None:
    value = get(raw, path, required=required)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise AdapterError(f"{path}: identifier должен быть string или integer")
    result = str(value)
    if not result.strip() or result != result.strip():
        raise AdapterError(f"{path}: пустой identifier или краевые пробелы")
    return result  # Без усечения, числовых преобразований строк и удаления ведущих нулей.


def timestamp(raw: Mapping[str, Any], path: str, *, required: bool = True) -> datetime | None:
    value = text(raw, path, required=required)
    if value is None:
        return None
    pattern = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?"
    if not re.fullmatch(pattern, value):
        raise AdapterError(f"{path}: некорректный timestamp")
    if re.search(r"[+-]\d{2}:\d{2}$", value):
        if int(value[-5:-3]) > 23 or int(value[-2:]) > 59:
            # fromisoformat сам нормализует +00:99; повреждённые даты не исправляем.
            raise AdapterError(f"{path}: некорректное смещение UTC")
    try:
        # Python 3.10 fromisoformat принимает только 3/6 дробных знаков.
        # Дополняем корректную дробь нулями до микросекунд, без округления/потери точности.
        normalized = re.sub(r"\.(\d{1,6})", lambda match: "." + match[1].ljust(6, "0"), value)
        result = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        # Поля *Utc без суффикса уже заданы в UTC по контракту источника.
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise AdapterError(f"{path}: некорректная дата или время") from None


def birth_date(raw: Mapping[str, Any]) -> date | None:
    value = text(raw, "birthDate")
    if value is None:
        return None
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        return date.fromisoformat(value)
    except ValueError:
        raise AdapterError("birthDate: некорректная дата") from None


def number(raw: Mapping[str, Any], path: str, *, required: bool = True) -> Decimal | None:
    value = get(raw, path, required=required)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise AdapterError(f"{path}: ожидается JSON number")
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        raise AdapterError(f"{path}: некорректное число") from None
    if not result.is_finite():
        raise AdapterError(f"{path}: число должно быть конечным")
    return result


def integer(raw: Mapping[str, Any], path: str) -> int:
    value = get(raw, path, required=True)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdapterError(f"{path}: ожидается integer")
    return value


def boolean(raw: Mapping[str, Any], path: str) -> bool | None:
    value = get(raw, path)
    if value is not None and not isinstance(value, bool):
        raise AdapterError(f"{path}: ожидается boolean")
    return value


def objects(raw: Mapping[str, Any], path: str, *, required: bool = False) -> list[Mapping[str, Any]]:
    value = get(raw, path, required=required)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise AdapterError(f"{path}: ожидается массив объектов")
    return value


def mapping(raw: Mapping[str, Any], path: str) -> Mapping[str, Any] | None:
    value = get(raw, path)
    if value is not None and not isinstance(value, Mapping):
        raise AdapterError(f"{path}: ожидается object")
    return value


def product_key(raw: Mapping[str, Any], namespaces: tuple[str, ...]) -> ProductKey:
    ids = get(raw, "ids", required=True)
    if not isinstance(ids, Mapping):
        raise AdapterError("product.ids: ожидается object")
    supported = [name for name in namespaces if name in ids]
    if len(supported) != 1:
        raise AdapterError("product.ids: отсутствует поддержанный namespace или неоднозначность нескольких namespaces")
    name = supported[0]
    return ProductKey(name, identifier(raw, "ids." + name))
