"""Transient contacts aligned to model users; never serialize into model artifacts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json


class CustomerContactError(ValueError):
    """Safe contact-contract error, without raw values."""


def user_mapping_digest(user_ids):
    if (not isinstance(user_ids, (tuple, list)) or any(not isinstance(u, str) or not u for u in user_ids)
            or len(set(user_ids)) != len(user_ids)):
        raise CustomerContactError("Invalid model user mapping")
    return hashlib.sha256(json.dumps(list(user_ids), ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, repr=False)
class CustomerContact:
    email: str | None = None
    mobile_phone: str | None = None
    discount_card: str | None = None

    def __post_init__(self):
        if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in
               (self.email, self.mobile_phone, self.discount_card)):
            raise CustomerContactError("Contact fields must be nonempty strings or absent")


@dataclass(frozen=True)
class ContactDiagnostics:
    customer_records: int
    canonical_profiles: int
    model_users: int
    matched_profiles: int
    missing_profiles: int
    with_email: int
    with_phone: int
    with_card: int
    with_last_activated_card_number: int
    single_card_fallback: int
    ambiguous_discount_cards: int
    cards_without_number: int
    invalid_card_numbers: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in vars(self).values()):
            raise CustomerContactError("Invalid contact diagnostic counts")
        if self.matched_profiles + self.missing_profiles != self.model_users:
            raise CustomerContactError("Inconsistent contact coverage")


@dataclass(frozen=True)
class CustomerContactIndex:
    contacts: tuple[CustomerContact, ...] = field(repr=False)
    mapping_digest: str = field(repr=False)
    diagnostics: ContactDiagnostics

    def __post_init__(self):
        contacts = tuple(self.contacts)
        if (not isinstance(self.diagnostics, ContactDiagnostics)
                or any(not isinstance(c, CustomerContact) for c in contacts) or len(contacts) != self.diagnostics.model_users):
            raise CustomerContactError("Invalid contact index dimensions")
        if (not isinstance(self.mapping_digest, str) or len(self.mapping_digest) != 64
                or any(c not in "0123456789abcdef" for c in self.mapping_digest)):
            raise CustomerContactError("Invalid contact mapping fingerprint")
        object.__setattr__(self, "contacts", contacts)

    def validate_alignment(self, user_ids):
        if self.mapping_digest != user_mapping_digest(user_ids):
            raise CustomerContactError("Contact index does not match model user mapping")


def _card_number(card):
    """Return number plus safe characterization: missing/invalid/valid."""
    if card is None:
        return None, "missing"
    if not isinstance(card, Mapping):
        return None, "invalid"
    ids = card.get("ids")
    if ids is None:
        return None, "missing"
    if not isinstance(ids, Mapping):
        return None, "invalid"
    number = ids.get("number")
    if number is None or (isinstance(number, str) and not number.strip()):
        return None, "missing"
    if isinstance(number, str):
        return number.strip(), "valid"
    if type(number) is int and number >= 0:
        return str(number), "valid"
    return None, "invalid"


def build_customer_contact_index(records, model_user_ids=None):
    """Choose a whole profile: canonical source, latest timestamp, last export row.

    Selection is streaming, retaining one winning record per relevant canonical user.
    Without model mappings, inspect all canonical profiles in sorted-ID order.
    """
    if model_user_ids is not None:
        user_mapping_digest(model_user_ids)
        wanted = set(model_user_ids)
    else:
        wanted = None
    selected = {}
    canonical_ids = set()
    record_count = 0
    minimum = datetime.min.replace(tzinfo=timezone.utc)
    for order, record in enumerate(records):
        record_count += 1
        canonical_ids.add(record.customer_id)
        if wanted is not None and record.customer_id not in wanted:
            continue
        stamp = record.change_datetime_utc
        if stamp is not None and (stamp.tzinfo is None or stamp.utcoffset() is None):
            raise CustomerContactError("Profile timestamps must be timezone aware")
        priority = (record.source_customer_id == record.customer_id, stamp or minimum, order)
        previous = selected.get(record.customer_id)
        if previous is None or priority > previous[0]:
            selected[record.customer_id] = (priority, record)
    user_ids = list(model_user_ids) if model_user_ids is not None else sorted(selected)
    contacts = []
    activated = fallback = ambiguous = missing_numbers = invalid_numbers = 0
    for user in user_ids:
        if user not in selected:
            contacts.append(CustomerContact())
            continue
        record = selected[user][1]
        number, state = _card_number(record.last_activated_card)
        cards = [_card_number(card) for card in record.discount_cards]
        characterized = cards + ([(number, state)] if record.last_activated_card is not None else [])
        missing_numbers += sum(status == "missing" for _, status in characterized)
        invalid_numbers += sum(status == "invalid" for _, status in characterized)
        if number is not None:
            activated += 1
        else:
            numbers = {value for value, _ in cards if value is not None}
            if len(numbers) == 1:
                number = next(iter(numbers))
                fallback += 1
            elif len(numbers) > 1:
                ambiguous += 1
        contacts.append(CustomerContact(record.email, record.mobile_phone, number))
    diagnostics = ContactDiagnostics(record_count, len(canonical_ids), len(user_ids), len(selected),
        len(user_ids) - len(selected), sum(c.email is not None for c in contacts),
        sum(c.mobile_phone is not None for c in contacts), sum(c.discount_card is not None for c in contacts),
        activated, fallback, ambiguous, missing_numbers, invalid_numbers)
    return CustomerContactIndex(tuple(contacts), user_mapping_digest(user_ids), diagnostics)
