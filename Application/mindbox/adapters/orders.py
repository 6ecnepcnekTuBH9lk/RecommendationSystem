"""Только OrdersAPI создаёт позиции покупок. Одно raw order -> tuple позиций."""

from collections.abc import Mapping
from typing import Any

from ..identity import CustomerIdResolver
from ..selection import DEFAULT_ORDER_NAMESPACES
from ..records import OrderLineRecord
from ._common import AdapterError, get, identifier, integer, number, objects, product_key, text, timestamp


def _product_name(line: Mapping[str, Any]) -> str | None:
    value = text(line, "product.name", required=False)
    if value is not None and not value.strip():
        raise AdapterError("product.name: ожидается строка")
    return value


def adapt_order(
    raw: Mapping[str, Any],
    resolver: CustomerIdResolver,
    *,
    product_namespaces: tuple[str, ...] = DEFAULT_ORDER_NAMESPACES,
) -> tuple[OrderLineRecord, ...]:
    source = identifier(raw, "customer.ids.mindboxId")
    order_id = identifier(raw, "ids.mindboxId")

    common = dict(
        order_id=order_id,
        retail_order_id=identifier(
            raw,
            "ids.retailOrderId",
            required=False,
        ),
        order_datetime_utc=timestamp(
            raw,
            "firstAction.dateTimeUtc",
        ),
        channel_external_id=identifier(
            raw,
            "firstAction.channel.ids.externalId",
        ),
        channel_name=text(
            raw,
            "firstAction.channel.name",
            required=True,
        ),
        source_customer_id=source,
        customer_id=resolver.resolve(source),
        order_total_price=number(
            raw,
            "totalPrice",
            required=False,
        ),
        delivery_cost=number(
            raw,
            "deliveryCost",
            required=False,
        ),
    )

    result = []

    for index, line in enumerate(
        objects(raw, "lines", required=True)
    ):
        line_number = integer(line, "number")
        line_id = identifier(
            line,
            "id",
            required=False,
        )

        if line_id is None:
            line_id = f"{order_id}:missing-line:{index}"

        result.append(
            OrderLineRecord(
                **common,
                line_id=line_id,
                line_number=line_number,
                product=product_key(
                    get(line, "product", required=True),
                    product_namespaces,
                ),
                product_name=_product_name(line),
                quantity=number(
                    line,
                    "quantity",
                ),
                base_price_per_item=number(
                    line,
                    "basePricePerItem",
                ),
                price_of_line=number(
                    line,
                    "priceOfLine",
                ),
                line_status=identifier(
                    line,
                    "status.ids.externalId",
                ),
            )
        )

    return tuple(result)
