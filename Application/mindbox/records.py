"""Неизменяемые записи источника, без CSV/UI и бизнес-классификации."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any


def freeze_json(value: Any) -> Any:
    """Создаёт независимый глубоко immutable снимок динамических JSON-структур."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(child) for child in value)
    return value


@dataclass(frozen=True, repr=False)
class ProductKey:
    namespace: str
    value: str


@dataclass(frozen=True, repr=False)
class ActionRecord:
    action_id: str
    action_system_name: str
    event_datetime_utc: datetime
    creation_datetime_utc: datetime
    source_customer_id: str
    customer_id: str
    action_name: str | None = None
    brand_system_name: str | None = None
    channel_system_name: str | None = None
    channel_external_id: str | None = None
    channel_mindbox_id: str | None = None
    channel_name: str | None = None
    products: tuple[ProductKey, ...] = ()
    categories: tuple[ProductKey, ...] = ()
    product_view_price: Decimal | None = None
    product_view_is_available: bool | None = None


@dataclass(frozen=True, repr=False)
class OrderLineRecord:
    order_id: str
    order_datetime_utc: datetime
    channel_external_id: str
    channel_name: str
    source_customer_id: str
    customer_id: str
    line_id: str
    line_number: int
    product: ProductKey
    product_name: str
    quantity: Decimal
    base_price_per_item: Decimal
    price_of_line: Decimal
    line_status: str
    retail_order_id: str | None = None
    order_total_price: Decimal | None = None
    delivery_cost: Decimal | None = None


@dataclass(frozen=True, repr=False)
class CustomerRecord:
    source_customer_id: str
    customer_id: str
    change_datetime_utc: datetime | None = None
    birth_date: date | None = None
    sex: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    middle_name: str | None = None
    email: str | None = None
    mobile_phone: str | None = None
    iana_time_zone: str | None = None
    time_zone_source: str | None = None
    custom_fields: Mapping[str, Any] = field(default_factory=dict)
    last_activated_card: Mapping[str, Any] | None = None
    segmentations: tuple[Mapping[str, Any], ...] = ()
    balances: tuple[Mapping[str, Any], ...] = ()
    discount_cards: tuple[Mapping[str, Any], ...] = ()
    subscriptions: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in ("custom_fields", "last_activated_card", "segmentations", "balances", "discount_cards", "subscriptions"):
            object.__setattr__(self, name, freeze_json(getattr(self, name)))


@dataclass(frozen=True, repr=False)
class CustomerMergeRecord:
    merge_id: str
    event_datetime_utc: datetime
    resulting_customer_id: str
    merged_customer_ids: tuple[str, ...]
    method: str | None = None
