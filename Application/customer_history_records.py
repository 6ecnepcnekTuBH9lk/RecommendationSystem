"""Immutable history contracts: source orders are not evidence of payment.

Current canonical merges are a present-day identity graph. Neither they nor
current order snapshots establish historical information availability for ML
as-of. observed_at is provenance supplied by a caller, never inferred from an
event/creation timestamp. Monetary values have no confirmed currency here.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum

from Application.mindbox.adapters._common import identifier
from Application.mindbox.records import ProductKey


HISTORY_SCHEMA_VERSION = 1


class HistoryKind(str, Enum):
    ORDER = "ORDER"
    VIEW = "VIEW"
    FAVORITE = "FAVORITE"
    UNCLASSIFIED = "UNCLASSIFIED"


class HistorySource(str, Enum):
    ACTION = "ACTION"
    ORDER = "ORDER"


class HistoryQuality(str, Enum):
    AS_OF_UNVERIFIED = "AS_OF_UNVERIFIED"
    UNCLASSIFIED_ACTION = "UNCLASSIFIED_ACTION"
    MISSING_CREATION_TIME = "MISSING_CREATION_TIME"
    ORDER_UPDATE_TIME_UNKNOWN = "ORDER_UPDATE_TIME_UNKNOWN"
    MISSING_PRODUCT = "MISSING_PRODUCT"
    UNSUPPORTED_PRODUCT_NAMESPACE = "UNSUPPORTED_PRODUCT_NAMESPACE"
    AMBIGUOUS_PRODUCT_NAMESPACE = "AMBIGUOUS_PRODUCT_NAMESPACE"
    SYNTHETIC_LINE_ID = "SYNTHETIC_LINE_ID"
    DUPLICATE_LINE_ID = "DUPLICATE_LINE_ID"
    FRACTIONAL_QUANTITY = "FRACTIONAL_QUANTITY"
    NEGATIVE_QUANTITY = "NEGATIVE_QUANTITY"
    NEGATIVE_AMOUNT = "NEGATIVE_AMOUNT"
    CURRENCY_UNCONFIRMED = "CURRENCY_UNCONFIRMED"


class HistoryComparison(str, Enum):
    DISTINCT = "DISTINCT"
    EQUIVALENT = "EQUIVALENT"
    CONFLICT = "CONFLICT"


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("History datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _id(value):
    if not isinstance(value, str):
        raise TypeError("History identifier must be a string")
    identifier({"value": value}, "value")


def _ordinal(value):
    if type(value) is not int or value < 0:
        raise ValueError("History ordinal must be a nonnegative integer")


def _decimal(value, *, optional=False):
    if value is None and optional:
        return
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError("History amount/quantity must be a finite Decimal")


def _tuple(values, kind):
    if not isinstance(values, tuple) or any(not isinstance(value, kind) for value in values):
        raise TypeError("History collections must be typed immutable tuples")


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid history key")


def _optional_text(*values):
    if any(value is not None and not isinstance(value, str) for value in values):
        raise TypeError("History labels must be strings or absent")


@dataclass(frozen=True, slots=True, repr=False)
class HistoryProvenance:
    source_kind: str = "UNKNOWN"
    component_version: str | None = None
    part_number: int | None = None
    record_ordinal: int | None = None
    observed_at_utc: datetime | None = None

    def __post_init__(self):
        if self.source_kind not in ("API", "MANUAL", "UNKNOWN"):
            raise ValueError("Invalid history provenance kind")
        if self.component_version is not None:
            _id(self.component_version)
        if self.part_number is not None:
            _ordinal(self.part_number)
            if self.part_number == 0:
                raise ValueError("Part numbers are one-based")
        if self.record_ordinal is not None:
            _ordinal(self.record_ordinal)
        if self.observed_at_utc is not None:
            object.__setattr__(self, "observed_at_utc", _utc(self.observed_at_utc))


@dataclass(frozen=True, slots=True, repr=False)
class HistoryChannel:
    system_name: str | None = None
    external_id: str | None = None
    mindbox_id: str | None = None
    name: str | None = None

    def __post_init__(self):
        for value in (self.external_id, self.mindbox_id):
            if value is not None:
                _id(value)
        if any(value is not None and not isinstance(value, str) for value in (self.system_name, self.name)):
            raise TypeError("History channel labels must be strings")


@dataclass(frozen=True, slots=True, repr=False)
class HistoryProductReference:
    """All source ID namespaces; product is selected only by existing adapter rules."""
    keys: tuple[ProductKey, ...]
    product: ProductKey | None
    quality: tuple[HistoryQuality, ...] = ()

    def __post_init__(self):
        _tuple(self.keys, ProductKey)
        _tuple(self.quality, HistoryQuality)
        for key in self.keys:
            _id(key.namespace)
            _id(key.value)
        if self.product is not None and self.product not in self.keys:
            raise ValueError("Selected history product must be a source reference")


@dataclass(frozen=True, slots=True, repr=False)
class ActionProductOccurrence:
    ordinal: int
    reference: HistoryProductReference

    def __post_init__(self):
        _ordinal(self.ordinal)
        if not isinstance(self.reference, HistoryProductReference):
            raise TypeError("Invalid action product reference")


@dataclass(frozen=True, slots=True, repr=False)
class ActionDetails:
    action_id: str
    system_name: str
    creation_datetime_utc: datetime | None
    channel: HistoryChannel
    products: tuple[ActionProductOccurrence, ...]
    categories: tuple[ActionProductOccurrence, ...] = ()
    action_name: str | None = None
    brand_system_name: str | None = None
    product_view_price: Decimal | None = None
    product_view_is_available: bool | None = None

    def __post_init__(self):
        _id(self.action_id)
        if not isinstance(self.system_name, str) or not self.system_name.strip():
            raise ValueError("Invalid action system name")
        if self.creation_datetime_utc is not None:
            object.__setattr__(self, "creation_datetime_utc", _utc(self.creation_datetime_utc))
        if not isinstance(self.channel, HistoryChannel):
            raise TypeError("Invalid action channel")
        _tuple(self.products, ActionProductOccurrence)
        _tuple(self.categories, ActionProductOccurrence)
        _optional_text(self.action_name, self.brand_system_name)
        _decimal(self.product_view_price, optional=True)
        if self.product_view_is_available is not None and type(self.product_view_is_available) is not bool:
            raise TypeError("Invalid source availability flag")


@dataclass(frozen=True, slots=True, repr=False)
class OrderDetails:
    order_id: str
    retail_order_id: str | None
    order_datetime_utc: datetime
    channel: HistoryChannel
    total_price: Decimal | None
    delivery_cost: Decimal | None

    def __post_init__(self):
        _id(self.order_id)
        if self.retail_order_id is not None:
            _id(self.retail_order_id)
        object.__setattr__(self, "order_datetime_utc", _utc(self.order_datetime_utc))
        if not isinstance(self.channel, HistoryChannel):
            raise TypeError("Invalid order channel")
        _decimal(self.total_price, optional=True)
        _decimal(self.delivery_cost, optional=True)


@dataclass(frozen=True, slots=True, repr=False)
class OrderLineDetails:
    """line_key is always snapshot-local; source_line_key is absent for missing/duplicate IDs."""
    line_key: str
    source_line_key: str | None
    line_id: str
    source_line_id: str | None
    is_synthetic: bool
    ordinal: int
    line_number: int
    reference: HistoryProductReference
    product_name: str | None
    quantity: Decimal
    raw_status: str
    price_of_line: Decimal
    base_price_per_item: Decimal | None
    quality: tuple[HistoryQuality, ...] = ()

    def __post_init__(self):
        _digest(self.line_key)
        if self.source_line_key is not None:
            _digest(self.source_line_key)
        _id(self.line_id)
        _id(self.raw_status)
        if self.source_line_id is not None:
            _id(self.source_line_id)
        if type(self.is_synthetic) is not bool or self.is_synthetic != (self.source_line_id is None):
            raise ValueError("Inconsistent source line identity")
        if self.is_synthetic and self.source_line_key is not None:
            raise ValueError("Synthetic lines cannot have cross-snapshot identities")
        _ordinal(self.ordinal)
        if type(self.line_number) is not int or not isinstance(self.reference, HistoryProductReference):
            raise TypeError("Invalid order line")
        _decimal(self.quantity)
        _decimal(self.price_of_line)
        _decimal(self.base_price_per_item, optional=True)
        _optional_text(self.product_name)
        _tuple(self.quality, HistoryQuality)


@dataclass(frozen=True, slots=True, repr=False)
class OrderSnapshot:
    logical_order_key: str
    snapshot_key: str
    fingerprint: bytes
    order: OrderDetails
    lines: tuple[OrderLineDetails, ...]
    provenance: HistoryProvenance

    def __post_init__(self):
        _digest(self.logical_order_key)
        _digest(self.snapshot_key)
        if not isinstance(self.fingerprint, bytes) or len(self.fingerprint) != 32:
            raise ValueError("Invalid order fingerprint")
        if not isinstance(self.order, OrderDetails) or not isinstance(self.provenance, HistoryProvenance):
            raise TypeError("Invalid order snapshot")
        _tuple(self.lines, OrderLineDetails)


@dataclass(frozen=True, slots=True, repr=False)
class HistoryEvent:
    event_key: str
    source: HistorySource
    source_event_id: str
    canonical_customer_id: str
    source_customer_id: str
    kind: HistoryKind
    event_datetime_utc: datetime
    fingerprint: bytes
    provenance: HistoryProvenance
    details: ActionDetails | OrderSnapshot
    quality: tuple[HistoryQuality, ...] = ()

    def __post_init__(self):
        _digest(self.event_key)
        for value in (self.source_event_id, self.canonical_customer_id, self.source_customer_id):
            _id(value)
        if not isinstance(self.source, HistorySource) or not isinstance(self.kind, HistoryKind):
            raise TypeError("Invalid history source/type")
        object.__setattr__(self, "event_datetime_utc", _utc(self.event_datetime_utc))
        if not isinstance(self.fingerprint, bytes) or len(self.fingerprint) != 32:
            raise ValueError("Invalid history fingerprint")
        if not isinstance(self.provenance, HistoryProvenance):
            raise TypeError("Invalid history provenance")
        _tuple(self.quality, HistoryQuality)
        if self.source is HistorySource.ORDER:
            if (self.kind is not HistoryKind.ORDER or not isinstance(self.details, OrderSnapshot)
                    or self.source_event_id != self.details.order.order_id
                    or self.event_key != self.details.logical_order_key or self.fingerprint != self.details.fingerprint
                    or self.event_datetime_utc != self.details.order.order_datetime_utc
                    or self.provenance != self.details.provenance):
                raise ValueError("Inconsistent order event")
        elif (self.kind is HistoryKind.ORDER or not isinstance(self.details, ActionDetails)
              or self.source_event_id != self.details.action_id):
            raise ValueError("Inconsistent action event")
