"""Normalize and reconcile one order for UNIFIED_PURCHASE_V2.

Readers supply all observations of one scoped order. This module has no I/O,
source ordering rules, persistent caches or statistics/ML entry points.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal, localcontext
from enum import Enum
from typing import TypeVar

from Application.mindbox.adapters._common import AdapterError, get, identifier, text, timestamp
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.order_dedup import semantic_fingerprint
from Application.mindbox.records import ProductKey
from Application.mindbox.selection import DEFAULT_ORDER_NAMESPACES
from Application.order_statistics import CURRENCIES, CURRENCY_BY_NAMESPACE
from Application.statistics_unified_policy import (
    EvidenceQuality as Quality,
    Measured,
    PurchasePolicy,
    StatusGroup,
    _decimal_input,
    classify_status_code,
    is_purchase_candidate,
    qualify_quantity,
)

T = TypeVar("T")
SourceRef = tuple[int, int, int]
Interval = tuple[datetime, datetime]
OrderAttributes = tuple[tuple[str, Measured[str | Decimal | tuple[str, ...]]], ...]
Issue = tuple[Quality, str]
_ZERO = Decimal(0)
_QUALITY_ORDER = (Quality.CONFLICT, Quality.INVALID, Quality.INSUFFICIENT_COVERAGE, Quality.UNKNOWN)


class OrderKind(str, Enum):
    COMPLETED_ONLY = "COMPLETED_ONLY"
    PARTIAL_ONLY = "PARTIAL_ONLY"
    CANCELLED_ONLY = "CANCELLED_ONLY"
    RETURNED_ONLY = "RETURNED_ONLY"
    IN_PROGRESS_ONLY = "IN_PROGRESS_ONLY"
    MIXED = "MIXED"
    EMPTY = "EMPTY"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True, repr=False)
class OrderLineState:
    stable_line_key: Measured[str]
    ordinal: int | None
    status_code: Measured[str]
    status_group: StatusGroup
    input_quantity: Decimal | None
    nominal_units: Measured[Decimal]
    fulfilled_units: Measured[Decimal]
    source_price: Measured[Decimal]
    currency: Measured[str]
    product_key: Measured[str]
    product_token: tuple[tuple[str, str], ...] | None


@dataclass(frozen=True, slots=True, repr=False)
class OrderObservation:
    source_ref: SourceRef
    order_key: Measured[str]
    customer_key: Measured[str]
    order_created_at: Measured[datetime]
    raw_fingerprint: bytes | None
    lines_quality: Quality
    lines: tuple[OrderLineState, ...]
    channel: Measured[tuple[str, str | None]]
    order_attributes: OrderAttributes
    raw_line_count: int


@dataclass(frozen=True, slots=True, repr=False)
class PurchaseEvidence:
    observed: Measured[bool]
    evidence_sources: tuple[SourceRef, ...]
    known_positions: int
    positions: Measured[int]
    known_nominal_units: Decimal
    nominal_units: Measured[Decimal]
    fulfilled_units: Measured[Decimal]
    executed_at: Measured[datetime]
    excluded_positions: int = 0


@dataclass(frozen=True, slots=True)
class MonetaryEvidence:
    currency: str
    source_nominal: Measured[Decimal]
    known_purchase_subtotal: Measured[Decimal]
    recognized_purchase: Measured[Decimal]
    partial_nominal: Measured[Decimal]
    cancelled_nominal: Measured[Decimal]
    returned_nominal: Measured[Decimal]
    confirmed_refund: Measured[Decimal]
    net_after_refunds: Measured[Decimal]
    known_nominals: tuple[tuple[str, Decimal], ...]
    valued_purchase_positions: int
    unvalued_purchase_positions: int


@dataclass(frozen=True, slots=True)
class SnapshotCounts:
    raw_orders: int
    raw_lines: int
    exact_duplicates: int
    business_equivalent: int
    conflicting_snapshots: int


@dataclass(frozen=True, slots=True, repr=False)
class ProductOrderEvidence:
    item_id: str
    purchase: PurchaseEvidence
    positions: Measured[int]
    current_groups: tuple[tuple[StatusGroup, int], ...]
    money: tuple[MonetaryEvidence, MonetaryEvidence]
    known_positions: int
    order_currency: Measured[str]


@dataclass(frozen=True, slots=True, repr=False)
class OrderBusinessFact:
    order_key: str
    customer_key: Measured[str]
    order_sort_rank: int
    order_created_at: Measured[datetime]
    current_kind: OrderKind
    state_quality: Quality
    state_groups: frozenset[StatusGroup]
    resolved_lines: tuple[OrderLineState, ...]
    purchase: PurchaseEvidence
    money: tuple[MonetaryEvidence, MonetaryEvidence]
    order_currency: Measured[str]
    channel: Measured[tuple[str, str | None]]
    order_attributes: OrderAttributes
    diagnostics: tuple[str, ...]
    snapshots: SnapshotCounts
    whole_order_kind: OrderKind
    observed_creation_dates: tuple[datetime, ...]
    observation_groups: tuple[tuple[SourceRef, ...], ...]
    policy: PurchasePolicy = PurchasePolicy()
    products: tuple[ProductOrderEvidence, ...] = ()
    historical_statuses: tuple[tuple[Measured[str], int], ...] = ()
    positions: Measured[int] | None = None
    line_history: tuple[tuple[OrderLineState, ...], ...] = ()
    unassigned_product_issues: tuple[Issue, ...] = ()

    @property
    def has_completed(self) -> bool:
        return StatusGroup.COMPLETED in self.state_groups

    @property
    def has_partial(self) -> bool:
        return StatusGroup.PARTIALLY_COMPLETED in self.state_groups

    @property
    def has_cancelled(self) -> bool:
        return StatusGroup.CANCELLED in self.state_groups

    @property
    def has_returned(self) -> bool:
        return StatusGroup.RETURNED in self.state_groups

    @property
    def has_in_progress(self) -> bool:
        return any(group.in_progress for group in self.state_groups)

    @property
    def has_unknown(self) -> bool:
        return self.state_quality is not Quality.KNOWN


@dataclass(frozen=True, slots=True, repr=False)
class AnalyticsContext:
    context_id: str
    policy: PurchasePolicy
    identity_space: str
    calculated_at: datetime
    as_of_date: date | None
    period: Interval | None
    product_filter_key: str | None
    orders_intervals: tuple[Interval, ...]
    actions_intervals: tuple[Interval, ...]
    coverage_basis: str
    created_history_verified: bool
    status_history_verified: bool
    provenance_digest: str


def _known(value: T) -> Measured[T]:
    return Measured(value, Quality.KNOWN)


def _unknown(reason: str) -> Measured:
    return Measured(None, Quality.UNKNOWN, (reason,))


def _invalid(reason: str) -> Measured:
    return Measured(None, Quality.INVALID, (reason,))


def _issues(value: Measured) -> list[Issue]:
    return [] if value.quality is Quality.KNOWN else [(value.quality, reason) for reason in value.reasons]


def _result(value: T, issues: Iterable[Issue]) -> Measured[T]:
    failures = set(issues)
    if not failures:
        return _known(value)
    qualities = {quality for quality, _ in failures}
    quality = next(quality for quality in _QUALITY_ORDER if quality in qualities)
    return Measured(None, quality, tuple(reason for _, reason in failures))


def _read(reader, raw, path, *, missing, invalid):
    try:
        value = reader(raw, path, required=False)
    except AdapterError:
        return _invalid(invalid)
    return _unknown(missing) if value is None else _known(value)


def _price(value: object) -> Measured[Decimal]:
    if value is None:
        return _unknown("PRICE_MISSING")
    amount = _decimal_input(value)
    return _invalid("PRICE_INVALID") if amount is None or amount < 0 else _known(amount)


def _product(raw, order_currencies, resolve_product):
    try:
        ids = get(raw, "product.ids")
    except AdapterError:
        ids = None
    if not isinstance(ids, Mapping):
        return _unknown("PRODUCT_UNKNOWN"), None, _unknown("CURRENCY_UNKNOWN")
    keys = []
    for namespace in DEFAULT_ORDER_NAMESPACES:
        if namespace in ids:
            key = _read(identifier, ids, namespace, missing="PRODUCT_UNKNOWN", invalid="PRODUCT_INVALID")
            keys.append((namespace, key))
    token = tuple((namespace, key.value) for namespace, key in keys if key.value is not None) or None
    currencies = {CURRENCY_BY_NAMESPACE[ns] for ns, _ in keys if ns in CURRENCY_BY_NAMESPACE}
    if len(currencies) > 1:
        currency = _result("", [(Quality.CONFLICT, "CURRENCY_CONFLICT")])
    elif not currencies or not token:
        currency = _unknown("CURRENCY_UNKNOWN")
    else:
        expected = next(iter(currencies))
        explicit = (*order_currencies, raw.get("currency"), raw.get("currencyCode"))
        if any(isinstance(code, str) and code in CURRENCIES and code != expected for code in explicit):
            currency = _result("", [(Quality.CONFLICT, "CURRENCY_CONFLICT")])
        else:
            currency = _known(expected)
    if len(keys) > 1:
        product = _result("", [(Quality.CONFLICT, "PRODUCT_CONFLICT")])
    elif not keys:
        product = _unknown("PRODUCT_UNKNOWN")
    else:
        namespace, key = keys[0]
        if key.value is None:
            product = key
        else:
            item = resolve_product(ProductKey(namespace, key.value))
            if item is not None and (not isinstance(item, str) or not item):
                raise ValueError("Product resolution requires an item key or None")
            product = _unknown("PRODUCT_UNKNOWN") if item is None else _known(item)
    return product, token, currency


def _line(raw, ordinal, order_currencies, resolve_product):
    line_id = _read(identifier, raw, "id", missing="LINE_ID_MISSING", invalid="LINE_ID_INVALID")
    try:
        status_code, group = classify_status_code(get(raw, "status.ids.externalId"))
    except AdapterError:
        status_code, group = _invalid("INVALID_CODE"), StatusGroup.UNKNOWN
    quantity = raw.get("quantity")
    product, token, currency = _product(raw, order_currencies, resolve_product)
    return OrderLineState(
        line_id, ordinal, status_code, group, _decimal_input(quantity), qualify_quantity(quantity),
        _unknown("FULFILLMENT_UNVERIFIED"), _price(raw.get("priceOfLine")), currency, product, token,
    )


def _channel(raw):
    key = _read(identifier, raw, "firstAction.channel.ids.externalId",
                missing="CHANNEL_MISSING", invalid="CHANNEL_INVALID")
    if key.value is None:
        return key
    try:
        name = text(raw, "firstAction.channel.name")
    except AdapterError:
        return _invalid("CHANNEL_INVALID")
    return _known((key.value, name))


def _order_attributes(raw) -> OrderAttributes:
    attributes = [
        ("delivery_cost", _price(raw.get("deliveryCost"))),
        ("delivery_type", _read(text, raw, "customFields.deliveryType",
                                missing="ATTRIBUTE_MISSING", invalid="ATTRIBUTE_INVALID")),
        ("ordering_method", _read(text, raw, "customFields.orderingMethod",
                                  missing="ATTRIBUTE_MISSING", invalid="ATTRIBUTE_INVALID")),
    ]
    payments = raw.get("payments")
    if payments is None:
        types = _unknown("ATTRIBUTE_MISSING")
    elif not isinstance(payments, list):
        types = _invalid("ATTRIBUTE_INVALID")
    else:
        try:
            types = _known(tuple(sorted(text(payment, "type", required=True) for payment in payments)))
        except AdapterError:
            types = _invalid("ATTRIBUTE_INVALID")
    attributes.append(("payment_types", types))
    return tuple(attributes)


def normalize_order_observation(
    raw: Mapping, source_ref: SourceRef, identities: CustomerIdResolver,
    resolve_product: Callable[[ProductKey], str | None],
) -> OrderObservation:
    """Use the caller's identity and product catalog from the pinned source scope."""
    if not isinstance(raw, Mapping):
        raise ValueError("Order observation requires an object")
    if (not isinstance(source_ref, tuple) or len(source_ref) != 3
            or any(type(part) is not int or part < 0 for part in source_ref)):
        raise ValueError("Invalid observation reference")
    order_key = _read(identifier, raw, "ids.mindboxId", missing="ORDER_ID_INVALID", invalid="ORDER_ID_INVALID")
    owner = _read(identifier, raw, "customer.ids.mindboxId", missing="OWNER_MISSING", invalid="OWNER_INVALID")
    if owner.value is not None:
        owner = _known(identities.resolve(owner.value))
    created = _read(timestamp, raw, "firstAction.dateTimeUtc",
                    missing="CREATED_TIME_MISSING", invalid="CREATED_TIME_INVALID")
    try:
        raw_hash = semantic_fingerprint(raw)
    except (TypeError, ValueError):
        raw_hash = None

    raw_lines = raw.get("lines")
    lines = []
    lines_quality = Quality.KNOWN
    if not isinstance(raw_lines, list):
        lines_quality = Quality.INSUFFICIENT_COVERAGE if raw_lines is None else Quality.INVALID
    else:
        order_currencies = raw.get("currency"), raw.get("currencyCode")
        for ordinal, line in enumerate(raw_lines):
            if isinstance(line, Mapping):
                lines.append(_line(line, ordinal, order_currencies, resolve_product))
            else:
                lines_quality = Quality.INSUFFICIENT_COVERAGE

    return OrderObservation(source_ref, order_key, owner, created, raw_hash, lines_quality, tuple(lines),
                            _channel(raw), _order_attributes(raw), len(raw_lines) if isinstance(raw_lines, list) else 0)


def _merge(values: Iterable[Measured[T]], conflict: str) -> Measured[T]:
    readings = tuple(values)
    known = {value.value for value in readings if value.quality is Quality.KNOWN}
    issues = [issue for value in readings for issue in _issues(value)]
    if len(known) > 1:
        issues.append((Quality.CONFLICT, conflict))
    if issues:
        return _result(None, issues)
    return Measured(known.pop(), Quality.KNOWN, tuple(reason for v in readings for reason in v.reasons))


def _sum(values: list[Decimal]) -> Decimal:
    nonzero = [value for value in values if value]
    if not nonzero:
        return _ZERO
    exponent = min(value.as_tuple().exponent for value in nonzero)
    adjusted = max(value.adjusted() for value in nonzero)
    carry = len(str(len(nonzero)))
    with localcontext() as context:
        context.prec = adjusted - exponent + 1 + carry
        context.Emax = max(context.Emax, adjusted + carry)
        context.Emin = min(context.Emin, exponent)
        return _decimal_input(sum(nonzero, _ZERO))


@dataclass(frozen=True, slots=True, repr=False)
class _BusinessSnapshot:
    owner: Measured[str]
    created: Measured[datetime]
    lines_quality: Quality
    lines: tuple[OrderLineState, ...]
    channel: Measured[tuple[str, str | None]]
    attributes: OrderAttributes


def _signature(observation: OrderObservation) -> _BusinessSnapshot:
    # Ordinal matters only if a line has no unambiguous stable key in this snapshot.
    by_key = {}
    for line in observation.lines:
        if line.stable_line_key.value is not None:
            by_key.setdefault(line.stable_line_key.value, set()).add(replace(line, ordinal=None))
    stable = []
    local = []
    seen = set()
    for line in observation.lines:
        key = line.stable_line_key.value
        if key is not None and len(by_key[key]) == 1:
            if key not in seen:
                stable.append(replace(line, ordinal=None))
                seen.add(key)
        else:
            local.append(line)
    ordered = tuple(sorted(stable, key=lambda line: line.stable_line_key.value)) + tuple(local)
    return _BusinessSnapshot(observation.customer_key, observation.order_created_at, observation.lines_quality,
                             ordered, observation.channel, observation.order_attributes)


def _line_groups(versions: tuple[_BusinessSnapshot, ...], check: Callable[[], None]):
    by_version = []
    issues = []
    for version in versions:
        check()
        lines, quality = version.lines, version.lines_quality
        counts = Counter(line.stable_line_key.value for line in lines if line.stable_line_key.value is not None)
        if quality is not Quality.KNOWN:
            issues.append((quality, "LINES_INCOMPLETE"))
        if any(count > 1 for count in counts.values()):
            issues.append((Quality.CONFLICT, "LINE_ID_DUPLICATE"))
        if len(versions) == 1:
            groups = tuple((line,) for line in lines if counts[line.stable_line_key.value] <= 1)
            return groups, tuple(issues)
        if any(line.stable_line_key.value is None for line in lines):
            issues.append((Quality.INSUFFICIENT_COVERAGE, "LINE_ID_MISSING"))
        by_version.append({line.stable_line_key.value: line for line in lines
                           if counts[line.stable_line_key.value] == 1})
    first = by_version[0].keys()
    # A missing identifier can explain unmatched keys without a demonstrated change of basket.
    comparable = not any(reason in ("LINE_ID_MISSING", "LINES_INCOMPLETE") for _, reason in issues)
    if comparable and any(mapping.keys() != first for mapping in by_version[1:]):
        issues.append((Quality.CONFLICT, "LINE_SET_CONFLICT"))
    complete_sizes = {len(version.lines) for version in versions if version.lines_quality is Quality.KNOWN}
    if len(complete_sizes) > 1:
        issues.append((Quality.CONFLICT, "LINE_SET_CONFLICT"))
    common = sorted(set(first).intersection(*(mapping.keys() for mapping in by_version[1:])))
    return tuple(tuple(mapping[key] for mapping in by_version) for key in common), tuple(issues)


def _resolve_line(lines: tuple[OrderLineState, ...]) -> OrderLineState:
    first = lines[0]
    if len(lines) == 1:
        return first
    status = _merge((line.status_code for line in lines), "STATUS_CONFLICT")
    if status.quality is Quality.CONFLICT and all(
            line.status_code.value in ("F", "sold", "CP") for line in lines):
        codes = {line.status_code.value for line in lines}
        status = _known(next(code for code in ("F", "sold", "CP") if code in codes))
    product = _merge((line.product_key for line in lines), "PRODUCT_CONFLICT")
    tokens = {line.product_token for line in lines}
    if product.value is None and len(tokens - {None}) > 1:
        product = _result(None, [*_issues(product), (Quality.CONFLICT, "PRODUCT_CONFLICT")])
    quantities = {line.input_quantity for line in lines}
    return replace(
        first, ordinal=None, status_code=status,
        status_group=classify_status_code(status.value)[1],
        input_quantity=quantities.pop() if len(quantities) == 1 else None,
        nominal_units=_merge((line.nominal_units for line in lines), "QUANTITY_CONFLICT"),
        source_price=_merge((line.source_price for line in lines), "PRICE_CONFLICT"),
        currency=_merge((line.currency for line in lines), "CURRENCY_CONFLICT"),
        product_key=product, product_token=tokens.pop() if len(tokens) == 1 else None,
    )


def _state(lines, completeness):
    groups = frozenset(line.status_group for line in lines)
    issues = [*completeness]
    for line in lines:
        if line.status_group is StatusGroup.UNKNOWN:
            issues.extend(_issues(line.status_code) or [(Quality.UNKNOWN, "UNKNOWN_CODE")])
    quality = _result(False, issues).quality
    if quality is Quality.CONFLICT:
        kind = OrderKind.CONFLICT
    elif quality is not Quality.KNOWN:
        kind = OrderKind.UNAVAILABLE
    elif not groups:
        kind = OrderKind.EMPTY
    else:
        kinds = {_STATE_KINDS[group] for group in groups}
        kind = next(iter(kinds)) if len(kinds) == 1 else OrderKind.MIXED
    return kind, quality, groups


_STATE_KINDS = {
    StatusGroup.COMPLETED: OrderKind.COMPLETED_ONLY,
    StatusGroup.PARTIALLY_COMPLETED: OrderKind.PARTIAL_ONLY,
    StatusGroup.CANCELLED: OrderKind.CANCELLED_ONLY,
    StatusGroup.RETURNED: OrderKind.RETURNED_ONLY,
    **{group: OrderKind.IN_PROGRESS_ONLY for group in StatusGroup if group.in_progress},
}


def _qualifies(line: OrderLineState) -> bool:
    return is_purchase_candidate(line.status_code.value, line.nominal_units.value)


def _conflicting(line):
    return any(reading.quality is Quality.CONFLICT for reading in (
        line.status_code, line.nominal_units, line.source_price, line.currency, line.product_key))


def _eligible(line):
    return _qualifies(line) and not _conflicting(line)


def _purchase(sources, eligible, completeness, evidence_issues, excluded_positions=0):
    units = [line.nominal_units.value for line in eligible]
    count = len(eligible)
    subtotal = _sum(units)
    observed = _known(True) if eligible else _result(False, evidence_issues)
    return PurchaseEvidence(
        observed, tuple(sorted(sources)), count, _result(count, completeness), subtotal,
        _result(subtotal, completeness), _unknown("FULFILLMENT_UNVERIFIED"),
        _unknown("EXECUTION_TIME_UNVERIFIED"), excluded_positions,
    )


def _order_currency(positive, completeness):
    if not positive:
        return _result(None, [*completeness, (Quality.UNKNOWN, "CURRENCY_UNKNOWN")])
    currencies = {unit.value for unit in positive if unit.value is not None}
    issues = [*completeness, *(issue for unit in positive for issue in _issues(unit))]
    if len(currencies) > 1:
        issues.append((Quality.CONFLICT, "MIXED_ORDER_CURRENCY"))
    return _result(next(iter(currencies), None), issues)


_NOMINALS = (
    ("source_nominal", None),
    ("partial_nominal", StatusGroup.PARTIALLY_COMPLETED),
    ("cancelled_nominal", StatusGroup.CANCELLED),
    ("returned_nominal", StatusGroup.RETURNED),
)


def _nominals(currency, agreed_lines, groups, completeness):
    full = []
    known = []
    for name, target in _NOMINALS:
        values = []
        issues = list(completeness)
        for line, group in zip(agreed_lines, groups, strict=True):
            if line.currency.value is not None and line.currency.value != currency:
                continue
            if target is not None and line.status_group is not target:
                if all(version.status_group not in (StatusGroup.UNKNOWN, target) for version in group):
                    continue
                issues.extend(_issues(line.status_code) or [(Quality.UNKNOWN, "UNKNOWN_CODE")])
                continue
            issues.extend(_issues(line.currency))
            issues.extend(_issues(line.source_price))
            if line.currency.value == currency and line.source_price.value is not None:
                values.append(line.source_price.value)
        total = _sum(values)
        full.append(_result(total, issues))
        if values or not issues:
            known.append((name, total))
    return full, tuple(sorted(known))


def evaluate_money(
    agreed_lines: tuple[OrderLineState, ...],
    qualifying_evidence: tuple[tuple[OrderLineState, ...], ...],
    *,
    completeness: tuple[Issue, ...],
    composition_currencies: frozenset[str] = frozenset(CURRENCIES),
) -> tuple[MonetaryEvidence, MonetaryEvidence]:
    """Evaluate each agreed line and its aligned versions, one contribution per line."""
    valuations = []
    for line in agreed_lines:
        if _eligible(line):
            valuations.append((line.currency, line.source_price.value,
                               [*_issues(line.currency), *_issues(line.source_price)]))

    result = []
    ledger = _unknown("LEDGER_UNAVAILABLE")
    for currency in CURRENCIES:
        currency_completeness = tuple(
            issue for issue in completeness
            if currency in composition_currencies or issue[1] in ("LINES_INCOMPLETE", "PRODUCT_SCOPE_UNRESOLVED")
        )
        issues = list(currency_completeness)
        subtotal = []
        valued = unvalued = 0
        for unit, amount, line_issues in valuations:
            if unit.value is not None and unit.value != currency:
                continue
            issues.extend(line_issues)
            if unit.value == currency:
                valued += not line_issues
                unvalued += bool(line_issues)
                if amount is not None:
                    subtotal.append(amount)
        nominal, known_nominals = _nominals(currency, agreed_lines, qualifying_evidence, currency_completeness)
        source, partial, cancelled, returned = nominal
        amount = _sum(subtotal)
        # The V2 amount is the sum of admissible costs. An admitted zero
        # price supplies a monetary observation, unlike a missing price.
        recognized = _known(amount) if subtotal else _result(amount, issues)
        known_subtotal = recognized
        result.append(MonetaryEvidence(currency, source, known_subtotal, recognized, partial, cancelled, returned,
                                       ledger, ledger, known_nominals, valued, unvalued))
    return result[0], result[1]


def _product_evidence(pairs, sources, omitted, completeness, check):
    products = {}
    issues_by_item = {}
    unassigned = set()
    if any(reason == "LINES_INCOMPLETE" for _, reason in completeness):
        unassigned.update(completeness)
    for version in omitted:
        check()
        item = version.product_key.value
        if item is None:
            unassigned.update(completeness)
        else:
            issues_by_item.setdefault(item, set()).update(completeness)
    for line, group in pairs:
        check()
        item = line.product_key.value
        if item is not None:
            products.setdefault(item, []).append((line, group))
        else:
            issue = (Quality.INSUFFICIENT_COVERAGE, "PRODUCT_SCOPE_UNRESOLVED")
            for version in group:
                item = version.product_key.value
                if item is None:
                    unassigned.add(issue)
                else:
                    issues_by_item.setdefault(item, set()).add(issue)
    result = []
    for item in sorted(products.keys() | sources.keys() | issues_by_item.keys()):
        check()
        selected = products.get(item, ())
        lines = tuple(line for line, _ in selected)
        groups = tuple(group for _, group in selected)
        issues = unassigned | issues_by_item.get(item, set())
        eligible = tuple(line for line in lines if _eligible(line))
        evidence_issues = [
            issue for issue in unassigned
            if issue[1] == "LINES_INCOMPLETE" or issue[1] == "PRODUCT_SCOPE_UNRESOLVED"
        ]
        purchase = _purchase(sources.get(item, ()), eligible, issues, evidence_issues)
        result.append(ProductOrderEvidence(
            item, purchase, _result(len(lines), issues),
            tuple(sorted(Counter(line.status_group for line in lines).items())),
            evaluate_money(lines, groups, completeness=tuple(issues)), len(lines),
            _order_currency({line.currency for line in eligible}, issues),
        ))
    return tuple(result), tuple(sorted(unassigned))


def _no_check() -> None:
    pass


def reconcile_order(
    observations: Iterable[OrderObservation], *, order_sort_rank: int,
    product_predicate: Callable[[str], bool] | None = None,
    include_products: bool = False,
    include_line_history: bool = False,
    check: Callable[[], None] = _no_check,
) -> OrderBusinessFact:
    """Reconcile one pre-grouped order. Product predicates operate on resolved item keys."""
    prepared = prepare_order(observations, order_sort_rank=order_sort_rank, check=check)
    return scope_order(prepared, product_predicate=product_predicate, include_products=include_products,
                       include_line_history=include_line_history, check=check)


@dataclass(frozen=True, slots=True, repr=False)
class PreparedOrder:
    order_key: str
    rank: int
    pairs: tuple
    composition: tuple[Issue, ...]
    omitted: tuple[OrderLineState, ...]
    positives: tuple
    owner: Measured
    created: Measured
    channel: Measured
    attrs: OrderAttributes
    diagnostics: tuple[str, ...]
    dates: tuple[datetime, ...]
    provenance: tuple
    snapshots: SnapshotCounts
    whole_kind: OrderKind


def prepare_order(observations, *, order_sort_rank, check=_no_check):
    """Materialize the common reconciliation before evaluating any product predicate."""
    if type(order_sort_rank) is not int or order_sort_rank < 1:
        raise ValueError("Invalid order sort rank")

    versions: dict[_BusinessSnapshot, list[SourceRef]] = {}
    fingerprints = set()
    positives = set()
    diagnostics = set()
    order_key = None
    raw_orders = raw_lines = exact = equivalent = 0

    for observation in observations:
        check()
        key = observation.order_key.value
        if key is None or (order_key is not None and key != order_key):
            raise ValueError("Order group requires one valid order key")
        order_key = key
        raw_orders += 1
        raw_lines += observation.raw_line_count
        signature = _signature(observation)
        fingerprint = (observation.raw_fingerprint, signature)
        if observation.raw_fingerprint is not None and fingerprint in fingerprints:
            exact += 1
        else:
            if observation.raw_fingerprint is not None:
                fingerprints.add(fingerprint)
            if signature in versions:
                equivalent += 1
        versions.setdefault(signature, []).append(observation.source_ref)
        for line in observation.lines:
            check()
            for reading in (line.status_code, line.stable_line_key, line.nominal_units, line.source_price,
                            line.currency, line.product_key):
                diagnostics.update(reading.reasons)

    if order_key is None:
        raise ValueError("Order group is empty")

    signatures = tuple(versions)
    groups, composition = _line_groups(signatures, check)
    pairs = []
    for group in groups:
        check()
        pairs.append((_resolve_line(group), group))
    accepted = {id(version): line for line, group in pairs if _eligible(line) for version in group}
    for signature, refs in versions.items():
        for version in signature.lines:
            line = accepted.get(id(version))
            if line is not None:
                positives.update((ref, line.product_key.value, line.currency) for ref in refs)
    full_lines = tuple(line for line, _ in pairs)
    whole_kind = _state(full_lines, composition)[0]
    omitted = ()
    if composition:
        retained = {id(line) for group in groups for line in group}
        omitted = tuple(line for snapshot in signatures for line in snapshot.lines if id(line) not in retained)
    owner = _merge((signature.owner for signature in signatures), "OWNER_CONFLICT")
    created = _merge((signature.created for signature in signatures), "CREATED_TIME_CONFLICT")
    channel = _merge((signature.channel for signature in signatures), "CHANNEL_CONFLICT")
    attrs = tuple(
        (name, _merge((dict(signature.attributes)[name] for signature in signatures), "ATTRIBUTE_CONFLICT"))
        for name, _ in signatures[0].attributes
    )
    for line in full_lines:
        for reading in (line.status_code, line.nominal_units, line.source_price, line.currency, line.product_key):
            diagnostics.update(reading.reasons)
    dates = tuple(sorted({snapshot.created.value for snapshot in signatures if snapshot.created.value is not None}))
    provenance = tuple(sorted(tuple(sorted(refs)) for refs in versions.values()))
    return PreparedOrder(order_key, order_sort_rank, tuple(pairs), composition, omitted, tuple(positives),
                         owner, created, channel, attrs, tuple(sorted(diagnostics)), dates, provenance,
                         SnapshotCounts(raw_orders, raw_lines, exact, equivalent, len(signatures) - 1), whole_kind)


def scope_order(prepared, *, product_predicate=None, include_products=True, include_line_history=False, check=_no_check):
    full_pairs = pairs = prepared.pairs
    composition, omitted = prepared.composition, prepared.omitted
    diagnostics = set(prepared.diagnostics)
    owner, created, channel, attrs = prepared.owner, prepared.created, prepared.channel, prepared.attrs
    sources, positive, product_sources, matches = set(), set(), {}, {}
    uncertain_membership = False

    def contains(key):
        if product_predicate is None:
            return True
        if key is None:
            return None
        if key not in matches:
            matches[key] = product_predicate(key)
        return matches[key]

    def membership(line):
        return contains(line.product_key.value)

    for source, item, currency in prepared.positives:
        included = contains(item)
        if included is True:
            sources.add(source)
            positive.add(currency)
            if include_products and item is not None:
                product_sources.setdefault(item, set()).add(source)
        elif included is None:
            uncertain_membership = True
    scoped_composition = composition
    if product_predicate is not None and all(membership(line) is False for line in omitted):
        scoped_composition = tuple(issue for issue in composition if issue[1] == "LINES_INCOMPLETE")
    scope_issues = list(scoped_composition)
    purchase_issues = list(scoped_composition)
    evidence_issues = [(q, reason) for q, reason in composition if reason == "LINES_INCOMPLETE"]
    if uncertain_membership:
        evidence_issues.append((Quality.INSUFFICIENT_COVERAGE, "PRODUCT_SCOPE_UNRESOLVED"))
        purchase_issues.append((Quality.INSUFFICIENT_COVERAGE, "PRODUCT_SCOPE_UNRESOLVED"))
    if product_predicate is not None:
        selected = []
        for line, group in pairs:
            check()
            if membership(line) is True:
                selected.append((line, group))
            elif any(membership(version) is not False for version in group):
                issue = (Quality.INSUFFICIENT_COVERAGE, "PRODUCT_SCOPE_UNRESOLVED")
                scope_issues.append(issue)
                if _eligible(line):
                    purchase_issues.append(issue)
        pairs = selected
    resolved_lines = tuple(line for line, _ in pairs)
    scoped_groups = tuple(group for _, group in pairs)
    eligible = tuple(line for line in resolved_lines if _eligible(line))
    current_kind, state_quality, state_groups = _state(resolved_lines, scope_issues)
    excluded = sum(_conflicting(line) for line in resolved_lines) + len({
        line.stable_line_key.value for line in omitted
        if line.stable_line_key.value is not None and membership(line) is not False})
    purchase = _purchase(sources, eligible, purchase_issues, evidence_issues, excluded)
    order_currency = _order_currency(positive, evidence_issues)
    omitted_currencies = {line.currency.value for line in omitted if membership(line) is not False}
    money = evaluate_money(
        resolved_lines, scoped_groups, completeness=tuple(scope_issues),
        composition_currencies=frozenset(CURRENCIES if None in omitted_currencies else omitted_currencies),
    )
    diagnostics.update(reason for _, reason in (*scope_issues, *purchase_issues))
    for reading in (owner, created, channel, order_currency, purchase.observed, purchase.positions, purchase.nominal_units):
        diagnostics.update(reading.reasons)
    for entry in money:
        diagnostics.update(entry.recognized_purchase.reasons)
    diagnostics.update(reason for _, attribute in attrs for reason in attribute.reasons)
    products = ()
    unassigned_product_issues = ()
    historical = ()
    positions = None
    if include_products:
        products, unassigned_product_issues = _product_evidence(full_pairs, product_sources, omitted, composition, check)
        if product_predicate is not None:
            products = tuple(item for item in products if matches.get(item.item_id, False))
        statuses = Counter(line.status_code for line in eligible)
        historical = tuple(sorted(statuses.items(), key=lambda item: (
            item[0].value or "", item[0].quality.value, item[0].reasons)))
        positions = _result(len(resolved_lines), scope_issues)
    check()
    return OrderBusinessFact(
        prepared.order_key, owner, prepared.rank, created, current_kind, state_quality, state_groups,
        resolved_lines, purchase, money, order_currency, channel, attrs, tuple(sorted(diagnostics)),
        prepared.snapshots, prepared.whole_kind, prepared.dates, prepared.provenance,
        products=products, historical_statuses=historical, positions=positions,
        line_history=scoped_groups if include_line_history else (),
        unassigned_product_issues=unassigned_product_issues,
    )
