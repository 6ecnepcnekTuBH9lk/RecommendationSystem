"""Pure analytical status/quantity rules; not the production PURCHASE policy.

Callers own period selection, first-snapshot deduplication, canonical identity
and currency/Decimal aggregation. No product resolution or monetary deduction
is implied by accepting a line. See docs/statistics_purchase_contract.md.
"""

from decimal import Decimal
from enum import Enum
from types import MappingProxyType


STRICT_COMPLETED = "STRICT_COMPLETED"
EXTENDED_COMPLETED = "EXTENDED_COMPLETED"

STRICT_COMPLETED_STATUSES = frozenset({"F", "sold"})
EXTENDED_COMPLETED_STATUSES = STRICT_COMPLETED_STATUSES | {"CP"}
PAID_OR_IN_PROGRESS_STATUSES = frozenset({
    "delivering", "Оплачен", "prepayed", "send-to-assembling",
    "assembling-complete", "redirect", "gotov-otpravka",
})
RETURNED_STATUSES = frozenset({"Return", "klientskii-vozvrat", "kurerskii-vozvrat"})
CANCELLED_STATUSES = frozenset({
    "CD", "RJ", "otmenen", "no-product", "out-of-stock",
    "already-buyed", "delyvery-did-not-suit", "Mindbox-Tech-Cancellation",
})
ORDER_IN_PROGRESS_STATUSES = frozenset({
    "N", "DA", "DD", "DB", "DG", "DS", "new", "no-call",
    "availability-confirmed", "offer-analog", "assembling", "ready-to-wait",
    "client-confirmed", "sh", "opr-perem", "soglasovanie", "tovar-naiden", "kv",
})
DELIVERY_IN_PROGRESS_STATUSES = frozenset({
    "ready-for-self-pickup", "send-to-delivery", "arrived-in-pickup-point", "de",
})

PURCHASE_POLICY_STATUSES = MappingProxyType({
    STRICT_COMPLETED: STRICT_COMPLETED_STATUSES,
    EXTENDED_COMPLETED: EXTENDED_COMPLETED_STATUSES,
})


class StatusGroup(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    PAID_OR_IN_PROGRESS = "PAID_OR_IN_PROGRESS"
    RETURNED = "RETURNED"
    CANCELLED = "CANCELLED"
    ORDER_IN_PROGRESS = "ORDER_IN_PROGRESS"
    DELIVERY_IN_PROGRESS = "DELIVERY_IN_PROGRESS"
    UNKNOWN = "UNKNOWN"


_STATUS_GROUPS = MappingProxyType({
    status: group
    for statuses, group in (
        (STRICT_COMPLETED_STATUSES, StatusGroup.COMPLETED),
        (frozenset({"CP"}), StatusGroup.PARTIALLY_COMPLETED),
        (PAID_OR_IN_PROGRESS_STATUSES, StatusGroup.PAID_OR_IN_PROGRESS),
        (RETURNED_STATUSES, StatusGroup.RETURNED),
        (CANCELLED_STATUSES, StatusGroup.CANCELLED),
        (ORDER_IN_PROGRESS_STATUSES, StatusGroup.ORDER_IN_PROGRESS),
        (DELIVERY_IN_PROGRESS_STATUSES, StatusGroup.DELIVERY_IN_PROGRESS),
    )
    for status in statuses
})


def classify_status(status: str | None) -> StatusGroup:
    """Exact, case-sensitive code classification; no trimming or inference."""
    return _STATUS_GROUPS.get(status, StatusGroup.UNKNOWN) if isinstance(status, str) else StatusGroup.UNKNOWN


def _positive_integer_quantity(quantity):
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float, Decimal)):
        return False
    if isinstance(quantity, int):
        return quantity > 0
    value = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))
    return value.is_finite() and value > 0 and value == value.to_integral_value()


def eligible_purchase_policies(status: str | None, quantity: Decimal | int | float | None) -> frozenset[str]:
    """Evaluate both policies once for one line; the result is not an order count.

    Numeric strings, bools, non-finite, nonpositive and fractional quantities
    are rejected. Price (including zero), product and its resolution are not
    eligibility inputs. A qualifying F/sold line belongs once to each policy.
    """
    if not isinstance(status, str) or not _positive_integer_quantity(quantity):
        return frozenset()
    return frozenset(policy for policy, statuses in PURCHASE_POLICY_STATUSES.items() if status in statuses)


def accepts_purchase_line(status: str | None, quantity: Decimal | int | float | None, *, policy: str) -> bool:
    """Check an explicitly selected analytical policy; reject unknown policy IDs."""
    if not isinstance(policy, str) or policy not in PURCHASE_POLICY_STATUSES:
        raise ValueError("Unknown analytical purchase policy")
    return policy in eligible_purchase_policies(status, quantity)
