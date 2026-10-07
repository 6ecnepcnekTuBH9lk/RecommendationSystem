"""Minimal contact projection; full Customers adapter remains unchanged."""

from types import SimpleNamespace

from Application.customer_profiles import contact_candidate_from_record
from ._common import identifier, mapping, objects, text, timestamp


def adapt_customer_contact_candidate(raw, resolver, model_users=None):
    source = identifier(raw, "ids.mindboxId")
    canonical = resolver.resolve(source)
    if model_users is not None and canonical not in model_users:
        # Identity only, for exact global canonical-count diagnostics.
        return SimpleNamespace(customer_id=canonical)
    return contact_candidate_from_record(SimpleNamespace(
        source_customer_id=source, customer_id=canonical,
        change_datetime_utc=timestamp(raw, "changeDateTimeUtc", required=False),
        email=text(raw, "email"), mobile_phone=identifier(raw, "mobilePhone", required=False),
        last_activated_card=mapping(raw, "lastActivatedCard"), discount_cards=objects(raw, "discountCards"),
    ))
