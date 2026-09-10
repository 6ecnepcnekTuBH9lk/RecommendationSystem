"""Только OrdersAPI создаёт позиции покупок. Одно raw order -> tuple позиций."""

from collections.abc import Mapping
from typing import Any

from ..identity import CustomerIdResolver
from ..records import OrderLineRecord
from ._common import get, identifier, integer, number, objects, product_key, text, timestamp


def adapt_order(raw: Mapping[str, Any], resolver: CustomerIdResolver) -> tuple[OrderLineRecord, ...]:
    source = identifier(raw, "customer.ids.mindboxId")
    common = dict(
        order_id=identifier(raw, "ids.mindboxId"),
        retail_order_id=identifier(raw, "ids.retailOrderId", required=False),
        order_datetime_utc=timestamp(raw, "firstAction.dateTimeUtc"),
        channel_external_id=identifier(raw, "firstAction.channel.ids.externalId"),
        channel_name=text(raw, "firstAction.channel.name", required=True),
        source_customer_id=source, customer_id=resolver.resolve(source),
        order_total_price=number(raw, "totalPrice", required=False),
        delivery_cost=number(raw, "deliveryCost", required=False),
    )
    # Проверяем весь заказ перед выдачей: ошибка последней позиции не выдаёт половину заказа.
    return tuple(OrderLineRecord(
        **common, line_id=identifier(line, "id"), line_number=integer(line, "number"),
        product=product_key(get(line, "product", required=True), ("offline1C", "kanzlerKz")),
        product_name=text(line, "product.name", required=True), quantity=number(line, "quantity"),
        base_price_per_item=number(line, "basePricePerItem"), price_of_line=number(line, "priceOfLine"),
        line_status=identifier(line, "status.ids.externalId"),
    ) for line in objects(raw, "lines", required=True))
