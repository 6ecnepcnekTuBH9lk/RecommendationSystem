"""Normalize and reconcile one order for UNIFIED_PURCHASE_V1.

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
    counts = Counter(line.stable_line_key.value for line in observation.lines)
    stable = []
    local = []
    for line in observation.lines:
        key = line.stable_line_key.value
        if key is not None and counts[key] == 1:
            stable.append(replace(line, ordinal=None))
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


def _purchase(sources, eligible, completeness, evidence_issues):
    units = []
    quantity_issues = []
    for lines in eligible:
        quantity = _merge((line.nominal_units for line in lines), "QUANTITY_CONFLICT")
        if quantity.value is not None:
            units.append(quantity.value)
        quantity_issues.extend(_issues(quantity))
    count = len(eligible)
    subtotal = _sum(units)
    observed = _known(True) if sources else _result(False, evidence_issues)
    return PurchaseEvidence(
        observed, tuple(sorted(sources)), count, _result(count, completeness), subtotal,
        _result(subtotal, (*completeness, *quantity_issues)), _unknown("FULFILLMENT_UNVERIFIED"),
        _unknown("EXECUTION_TIME_UNVERIFIED"),
    )


def _order_currency(positive, completeness):
    if not positive:
        return _result(None, [*completeness, (Quality.UNKNOWN, "CURRENCY_UNKNOWN")])
    currencies = {unit.value for unit in positive if unit.value is not None}
    issues = [*completeness, *(issue for unit in positive for issue in _issues(unit))]
    if len(currencies) > 1:
        issues.append((Quality.CONFLICT, "MIXED_ORDER_CURRENCY"))
    return _result(next(iter(currencies), None), issues)


def _purchase_line_value(group, eligible):
    currency = _merge((line.currency for line in eligible), "CURRENCY_CONFLICT")
    issues = _issues(currency)
    price = _merge((line.source_price for line in eligible), "PRICE_CONFLICT")
    issues.extend(_issues(price))
    has_cp = any(line.status_group is StatusGroup.PARTIALLY_COMPLETED for line in eligible)
    if has_cp:
        issues.append((Quality.UNKNOWN, "CP_ALLOCATION_UNKNOWN"))
    subtotal = price.value if not has_cp and currency.value is not None else None
    if len(eligible) != len(group) or len({line.status_group for line in eligible}) > 1:
        reason = "STATUS_CONFLICT" if len({line.status_code for line in group}) > 1 else "QUANTITY_CONFLICT"
        issues.append((Quality.CONFLICT, reason))
    issues.extend(_issues(_merge((line.nominal_units for line in eligible), "QUANTITY_CONFLICT")))
    issues.extend(issue for line in group
                  if line.status_group in (StatusGroup.COMPLETED, StatusGroup.PARTIALLY_COMPLETED)
                  for issue in _issues(line.nominal_units))
    return currency, subtotal, issues


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
) -> tuple[MonetaryEvidence, MonetaryEvidence]:
    """Evaluate each agreed line and its aligned versions, one contribution per line."""
    valuations = []
    bad_quantity = []
    for line, group in zip(agreed_lines, qualifying_evidence, strict=True):
        eligible = tuple(version for version in group if _qualifies(version))
        if eligible:
            valuations.append(_purchase_line_value(group, eligible))
        else:
            for version in group:
                if version.status_group in (StatusGroup.COMPLETED, StatusGroup.PARTIALLY_COMPLETED):
                    bad_quantity.extend((line.currency, issue) for issue in _issues(version.nominal_units))
                    if version.nominal_units.value == 0:
                        bad_quantity.append((line.currency, (Quality.INSUFFICIENT_COVERAGE, "ZERO_QUANTITY")))

    result = []
    ledger = _unknown("LEDGER_UNAVAILABLE")
    for currency in CURRENCIES:
        issues = list(completeness)
        if valuations:
            issues.extend(issue for unit, issue in bad_quantity if unit.value in (None, currency))
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
        nominal, known_nominals = _nominals(currency, agreed_lines, qualifying_evidence, completeness)
        source, partial, cancelled, returned = nominal
        amount = _sum(subtotal)
        recognized = _result(amount, issues)
        known_subtotal = _known(amount) if subtotal else recognized
        result.append(MonetaryEvidence(currency, source, known_subtotal, recognized, partial, cancelled, returned,
                                       ledger, ledger, known_nominals, valued, unvalued))
    return result[0], result[1]


def _eligible_groups(groups):
    return tuple(eligible for group in groups
                 if (eligible := tuple(line for line in group if _qualifies(line))))


def _product_evidence(pairs, sources, completeness, check):
    products = {}
    unresolved = set()
    for line, group in pairs:
        check()
        item = line.product_key.value
        if item is not None:
            products.setdefault(item, []).append((line, group))
        else:
            possible = {version.product_key.value for version in group}
            unresolved.update(possible)
    result = []
    for item in sorted(products.keys() | sources.keys()):
        check()
        selected = products.get(item, ())
        lines = tuple(line for line, _ in selected)
        groups = tuple(group for _, group in selected)
        issues = list(completeness)
        if item in unresolved or None in unresolved:
            issues.append((Quality.INSUFFICIENT_COVERAGE, "PRODUCT_SCOPE_UNRESOLVED"))
        eligible = _eligible_groups(groups)
        purchase = _purchase(sources.get(item, ()), eligible, issues, issues)
        result.append(ProductOrderEvidence(
            item, purchase, _result(len(lines), issues),
            tuple(sorted(Counter(line.status_group for line in lines).items())),
            evaluate_money(lines, groups, completeness=tuple(issues)), len(lines),
            _order_currency({version.currency for group in eligible for version in group}, issues),
        ))
    return tuple(result)


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
    if type(order_sort_rank) is not int or order_sort_rank < 1:
        raise ValueError("Invalid order sort rank")

    versions: dict[_BusinessSnapshot, list[SourceRef]] = {}
    fingerprints = set()
    sources = set()
    product_sources = {}
    positive = set()
    uncertain_membership = False
    diagnostics = set()
    order_key = None
    raw_orders = raw_lines = exact = equivalent = 0
    matches = {}

    def membership(line: OrderLineState) -> bool | None:
        if product_predicate is None:
            return True
        key = line.product_key.value
        if key is None:
            return None
        if key not in matches:
            matches[key] = product_predicate(key)
        return matches[key]

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
            if not _qualifies(line):
                continue
            included = membership(line)
            if included is True:
                sources.add(observation.source_ref)
                positive.add(line.currency)
                if include_products and line.product_key.value is not None:
                    product_sources.setdefault(line.product_key.value, set()).add(observation.source_ref)
            elif included is None:
                uncertain_membership = True

    if order_key is None:
        raise ValueError("Order group is empty")

    signatures = tuple(versions)
    groups, composition = _line_groups(signatures, check)
    pairs = []
    for group in groups:
        check()
        pairs.append((_resolve_line(group), group))
    full_lines = tuple(line for line, _ in pairs)
    full_pairs = pairs
    whole_kind = _state(full_lines, composition)[0]
    scope_issues = list(composition)
    purchase_issues = list(composition)
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
                if any(_qualifies(version) and membership(version) is not False for version in group):
                    purchase_issues.append(issue)
        pairs = selected
    resolved_lines = tuple(line for line, _ in pairs)
    scoped_groups = tuple(group for _, group in pairs)
    eligible = _eligible_groups(scoped_groups)
    current_kind, state_quality, state_groups = _state(resolved_lines, scope_issues)
    purchase = _purchase(sources, eligible, purchase_issues, evidence_issues)
    owner = _merge((signature.owner for signature in signatures), "OWNER_CONFLICT")
    created = _merge((signature.created for signature in signatures), "CREATED_TIME_CONFLICT")
    channel = _merge((signature.channel for signature in signatures), "CHANNEL_CONFLICT")
    attrs = tuple(
        (name, _merge((dict(signature.attributes)[name] for signature in signatures), "ATTRIBUTE_CONFLICT"))
        for name, _ in signatures[0].attributes
    )
    order_currency = _order_currency(positive, purchase_issues)
    money = evaluate_money(resolved_lines, scoped_groups, completeness=tuple(scope_issues))
    for line in full_lines:
        for reading in (line.status_code, line.nominal_units, line.source_price, line.currency, line.product_key):
            diagnostics.update(reading.reasons)
    diagnostics.update(reason for _, reason in (*scope_issues, *purchase_issues))
    for reading in (owner, created, channel, order_currency, purchase.observed, purchase.positions, purchase.nominal_units):
        diagnostics.update(reading.reasons)
    for entry in money:
        diagnostics.update(entry.recognized_purchase.reasons)
    diagnostics.update(reason for _, attribute in attrs for reason in attribute.reasons)
    dates = tuple(sorted({snapshot.created.value for snapshot in signatures if snapshot.created.value is not None}))
    provenance = tuple(sorted(tuple(sorted(refs)) for refs in versions.values()))
    products = ()
    historical = ()
    positions = None
    if include_products:
        products = _product_evidence(full_pairs, product_sources, composition, check)
        if product_predicate is not None:
            products = tuple(item for item in products if matches.get(item.item_id, False))
        statuses = Counter(_merge((line.status_code for line in group), "STATUS_CONFLICT") for group in eligible)
        historical = tuple(sorted(statuses.items(), key=lambda item: (
            item[0].value or "", item[0].quality.value, item[0].reasons)))
        positions = _result(len(resolved_lines), scope_issues)
    check()
    return OrderBusinessFact(
        order_key, owner, order_sort_rank, created, current_kind, state_quality, state_groups,
        resolved_lines, purchase, money, order_currency, channel, attrs, tuple(sorted(diagnostics)),
        SnapshotCounts(raw_orders, raw_lines, exact, equivalent, len(signatures) - 1), whole_kind, dates, provenance,
        products=products, historical_statuses=historical, positions=positions,
        line_history=scoped_groups if include_line_history else (),
    )
