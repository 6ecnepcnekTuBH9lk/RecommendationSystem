"""CustomersAPI — единственный источник профиля клиента."""

from collections.abc import Mapping
from typing import Any

from ..identity import CustomerIdResolver
from ..records import CustomerRecord
from ._common import birth_date, identifier, mapping, objects, text, timestamp


def adapt_customer(raw: Mapping[str, Any], resolver: CustomerIdResolver) -> CustomerRecord:
    source = identifier(raw, "ids.mindboxId")
    return CustomerRecord(
        source_customer_id=source, customer_id=resolver.resolve(source),
        change_datetime_utc=timestamp(raw, "changeDateTimeUtc", required=False), birth_date=birth_date(raw),
        sex=text(raw, "sex"), first_name=text(raw, "firstName"), last_name=text(raw, "lastName"),
        middle_name=text(raw, "middleName"), email=text(raw, "email"),
        mobile_phone=identifier(raw, "mobilePhone", required=False),
        iana_time_zone=text(raw, "ianaTimeZone"), time_zone_source=text(raw, "timeZoneSource"),
        custom_fields=mapping(raw, "customFields") or {}, last_activated_card=mapping(raw, "lastActivatedCard"),
        segmentations=tuple(objects(raw, "segmentations")), balances=tuple(objects(raw, "balances")),
        discount_cards=tuple(objects(raw, "discountCards")), subscriptions=tuple(objects(raw, "subscriptions")),
    )
