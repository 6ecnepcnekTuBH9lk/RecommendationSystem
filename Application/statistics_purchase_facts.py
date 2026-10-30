"""One selected, adapted order -> two analytical policies. No I/O or history.

The caller owns source validation, period, identity resolution and first-snapshot
selection. Pass all adapted lines BEFORE product filtering. IDs and their space
are private calculation inputs, never public statistics DTO/export fields.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from Application.mindbox.adapters._common import AdapterError
from Application.order_statistics import CURRENCY_BY_NAMESPACE, _currency, _decimal
from Application.statistics_purchase_policy import (
    STRICT_COMPLETED, EXTENDED_COMPLETED, StatusGroup, classify_status, eligible_purchase_policies,
)


class MonetaryIssue(str, Enum):
    UNKNOWN_CURRENCY = "UNKNOWN_CURRENCY"
    MIXED_CURRENCY = "MIXED_CURRENCY"
    INVALID_PRICE = "INVALID_PRICE"


@dataclass(frozen=True, slots=True)
class PurchasePolicyResult:
    line_count: int
    units: Decimal
    currency: str | None
    amount: Decimal | None
    monetary_issues: frozenset[MonetaryIssue]

    @property
    def has_purchase(self):
        return self.line_count > 0

    @property
    def monetary_available(self):
        return self.has_purchase and not self.monetary_issues


@dataclass(frozen=True, slots=True, repr=False)
class PurchaseOrderFact:
    identity_space: str
    customer_id: str | int
    order_id: str | int
    order_datetime_utc: datetime
    strict: PurchasePolicyResult
    extended: PurchasePolicyResult
    status_groups: tuple[tuple[StatusGroup, int], ...]

    @property
    def return_line_count(self):
        return next((count for group, count in self.status_groups if group is StatusGroup.RETURNED), 0)


@dataclass
class _Totals:
    line_count: int = 0
    units: Decimal = Decimal(0)
    amount: Decimal = Decimal(0)
    namespaces: set = field(default_factory=set)
    invalid_price: bool = False

    def add(self, quantity, price, namespace):
        self.line_count += 1
        self.units += quantity
        # Bound memory to the two supported namespaces plus one unknown bucket.
        self.namespaces.add(namespace)
        if price is None:
            self.invalid_price = True
        else:
            self.amount += price

    def result(self):
        currency = _currency(self.namespaces) if self.line_count else None
        issues = set()
        if currency == "unknown":
            issues.add(MonetaryIssue.UNKNOWN_CURRENCY)
        elif currency == "mixed":
            issues.add(MonetaryIssue.MIXED_CURRENCY)
        if self.invalid_price:
            issues.add(MonetaryIssue.INVALID_PRICE)
        return PurchasePolicyResult(self.line_count, self.units, currency,
                                    self.amount if self.line_count and not issues else None, frozenset(issues))


def _valid_identity(value):
    return (isinstance(value, str) and bool(value.strip()) and value == value.strip()
            or type(value) is int and value > 0)


def build_purchase_order_fact(lines, *, identity_space, customer_id, order_id, order_datetime_utc, check=None):
    """Consume an iterable of OrderLineRecord/IndexedInput.order() rows once.

    identity_space must identify the caller's source/generation (e.g.
    REFERENCE:<snapshot> or INDEXED:<generation>). Do not combine different
    spaces in one accumulator. No ID conversion/resolution is performed here.
    Order identity is explicit because IndexedInput.order() lines omit order_id.

    Price validation matches OrderAggregates, but a bad price supplied to this
    isolated operation marks Monetary unavailable without removing the purchase.
    This does not bypass source adapters or their existing fail-closed errors.
    Returns are classified/countable only; no return money/quantity aggregate.
    """
    if not isinstance(identity_space, str) or not identity_space.strip() or identity_space != identity_space.strip():
        raise ValueError("Invalid purchase fact identity space")
    if not _valid_identity(customer_id) or not _valid_identity(order_id):
        raise ValueError("Invalid purchase fact identity")
    if (not isinstance(order_datetime_utc, datetime) or order_datetime_utc.tzinfo is None
            or order_datetime_utc.utcoffset() != timedelta(0)):
        raise ValueError("Purchase fact requires a validated UTC order datetime")
    if check is not None:
        check()
    totals = {STRICT_COMPLETED: _Totals(), EXTENDED_COMPLETED: _Totals()}
    groups = Counter()
    for line in lines:
        if check is not None:
            check()
        groups[classify_status(line.line_status)] += 1
        policies = eligible_purchase_policies(line.line_status, line.quantity)
        if not policies:
            continue
        quantity = line.quantity if isinstance(line.quantity, Decimal) else Decimal(str(line.quantity))
        namespace = getattr(getattr(line, "product", None), "namespace", None)
        namespace = namespace if isinstance(namespace, str) and namespace in CURRENCY_BY_NAMESPACE else None
        try:
            price = _decimal(getattr(line, "price_of_line", None), "priceOfLine")
        except AdapterError:
            price = None
        for policy in policies:
            totals[policy].add(quantity, price, namespace)
    if check is not None:
        check()
    return PurchaseOrderFact(identity_space, customer_id, order_id, order_datetime_utc,
                             totals[STRICT_COMPLETED].result(), totals[EXTENDED_COMPLETED].result(),
                             tuple(sorted(groups.items(), key=lambda item: item[0].value)))
