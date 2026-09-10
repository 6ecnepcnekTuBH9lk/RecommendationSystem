"""CustomerMerges -> все source IDs и resulting ID, без поиска профиля target."""

from collections.abc import Mapping
from typing import Any

from ..records import CustomerMergeRecord
from ._common import AdapterError, identifier, objects, text, timestamp


def adapt_customer_merge(raw: Mapping[str, Any]) -> CustomerMergeRecord:
    merged = objects(raw, "mergedCustomers", required=True)
    if not merged:
        raise AdapterError("mergedCustomers: массив источников объединения не должен быть пустым")
    return CustomerMergeRecord(
        merge_id=identifier(raw, "id"), event_datetime_utc=timestamp(raw, "dateTimeUtc"),
        resulting_customer_id=identifier(raw, "resultingCustomer.ids.mindboxId"),
        merged_customer_ids=tuple(identifier(customer, "ids.mindboxId") for customer in merged),
        method=text(raw, "method"),
    )
