"""Actions -> ActionRecord; без классификации и копирования customer profile."""

from collections.abc import Mapping
from typing import Any

from ..identity import CustomerIdResolver
from ..selection import DEFAULT_ACTION_NAMESPACES
from ..records import ActionRecord
from ._common import boolean, identifier, number, objects, product_key, text, timestamp


def adapt_action_system_name(raw: Mapping[str, Any]) -> str:
    """Read only the required technical name, without identity/product parsing."""
    return text(raw, "actionTemplate.ids.systemName", required=True)


def adapt_action(raw: Mapping[str, Any], resolver: CustomerIdResolver, *,
                 product_namespaces: tuple[str, ...] = DEFAULT_ACTION_NAMESPACES) -> ActionRecord:
    source = identifier(raw, "customer.ids.mindboxId")
    return ActionRecord(
        action_id=identifier(raw, "ids.mindboxId"),
        action_system_name=adapt_action_system_name(raw),
        action_name=text(raw, "actionTemplate.name"),
        event_datetime_utc=timestamp(raw, "dateTimeUtc"),
        creation_datetime_utc=timestamp(raw, "creationDateTimeUtc"),
        source_customer_id=source, customer_id=resolver.resolve(source),
        brand_system_name=text(raw, "brand.ids.systemName"),
        channel_system_name=text(raw, "channel.ids.systemName"),
        channel_external_id=identifier(raw, "channel.ids.externalId", required=False),
        channel_mindbox_id=identifier(raw, "channel.ids.mindboxId", required=False),
        channel_name=text(raw, "channel.name"),
        products=tuple(product_key(item, product_namespaces) for item in objects(raw, "products")),
        categories=tuple(product_key(item, ("offline1C",)) for item in objects(raw, "productCategories")),
        product_view_price=number(raw, "productView.price", required=False),
        product_view_is_available=boolean(raw, "productView.isAvailable"),
    )
