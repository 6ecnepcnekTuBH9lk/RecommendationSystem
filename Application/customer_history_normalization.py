"""Single raw record -> history. No I/O, Qt, BPR, index or global deduplication.

Fingerprints describe full source content, not a normalized projection or a
latest-state policy. Provenance, current merge resolution and selection are not
part of source equality; a future index must version them independently.
Unknown/missing/ambiguous product references are retained with quality flags.
Malformed present identifiers/numbers/timestamps remain errors.
"""

from collections import Counter
from collections.abc import Mapping
from decimal import Decimal
import hashlib
import json

from Application.customer_history_records import (
    HISTORY_SCHEMA_VERSION, ActionDetails, ActionProductOccurrence, HistoryChannel, HistoryComparison,
    HistoryEvent, HistoryKind, HistoryProductReference, HistoryProvenance, HistoryQuality as Q,
    HistorySource, OrderDetails, OrderLineDetails, OrderSnapshot,
)
from Application.interactions import InteractionType, classify_action_system_name
from Application.mindbox.adapters.actions import adapt_action
from Application.mindbox.adapters.orders import adapt_order, _product_name
from Application.mindbox.adapters._common import AdapterError, get, identifier, integer, mapping, number, objects, product_key, text, timestamp
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.order_dedup import semantic_fingerprint
from Application.mindbox.records import ProductKey
from Application.mindbox.selection import DEFAULT_SELECTION, MindboxSelectionConfig


class HistoryNormalizationError(ValueError):
    """Static message only; line_index is safe, one-based adapter diagnostics."""
    def __init__(self, message, *, line_index=None):
        super().__init__(message)
        self.line_index = line_index


def _key(*parts):
    encoded = json.dumps((HISTORY_SCHEMA_VERSION, *parts), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def history_event_key(source: HistorySource, source_event_id: str) -> str:
    if not isinstance(source, HistorySource) or not isinstance(source_event_id, str):
        raise HistoryNormalizationError("Invalid history event key input")
    try:
        identifier({"value": source_event_id}, "value")
        return _key("event", source.value, source_event_id)
    except Exception:
        raise HistoryNormalizationError("Invalid history event key input") from None


def _source_json(value):
    # Accept Mapping inputs without storing mutable raw payloads. Retain number
    # types for the existing Decimal-stable semantic fingerprint.
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Invalid source object keys")
        return {key: _source_json(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_source_json(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float, Decimal)):
        return value
    raise ValueError("Invalid source JSON type")


def _context(raw, resolver, selection, provenance):
    if (not isinstance(raw, Mapping) or not isinstance(resolver, CustomerIdResolver)
            or not isinstance(selection, MindboxSelectionConfig)):
        raise TypeError("Invalid history normalization inputs")
    origin = HistoryProvenance() if provenance is None else provenance
    if not isinstance(origin, HistoryProvenance):
        raise TypeError("Invalid history origin")
    return origin, semantic_fingerprint(_source_json(raw))


def _reference(raw, namespaces):
    if raw is None:
        return HistoryProductReference((), None, (Q.MISSING_PRODUCT,))
    ids = mapping(raw, "ids")
    if not ids:
        return HistoryProductReference((), None, (Q.MISSING_PRODUCT,))
    keys = []
    for namespace in sorted(ids):
        if not isinstance(namespace, str) or not namespace or namespace != namespace.strip():
            raise ValueError("Invalid product namespace")
        # Never reinterpret dots in a namespace as a nested identifier path.
        keys.append(ProductKey(namespace, identifier({"value": ids[namespace]}, "value")))
    supported = [key for key in keys if key.namespace in namespaces]
    if len(supported) == 1:
        selected = product_key(raw, namespaces)
        flags = ()
    else:
        selected = None
        flags = (Q.AMBIGUOUS_PRODUCT_NAMESPACE if supported else Q.UNSUPPORTED_PRODUCT_NAMESPACE,)
    return HistoryProductReference(tuple(keys), selected, flags)


def _occurrences(raw, path, namespaces):
    return tuple(ActionProductOccurrence(i, _reference(item, namespaces)) for i, item in enumerate(objects(raw, path)))


def _flags(values):
    return tuple(dict.fromkeys(values))


def normalize_action(raw: Mapping, resolver: CustomerIdResolver, *,
                     selection: MindboxSelectionConfig = DEFAULT_SELECTION,
                     provenance: HistoryProvenance | None = None) -> HistoryEvent:
    """Keep all products/occurrences, including repeated identical products.

    The full adapter is used unchanged for ordinary records. For references it
    cannot represent, a minimal header projection clears only product/category
    arrays. Missing creation time is kept as None; the adapter-only substitute
    validates the header and is never exposed as a source creation timestamp.
    """
    try:
        origin, fingerprint = _context(raw, resolver, selection, provenance)
        products = _occurrences(raw, "products", selection.action_product_namespaces)
        categories = _occurrences(raw, "productCategories", ("offline1C",))
        created = timestamp(raw, "creationDateTimeUtc", required=False)
        projection = raw
        if created is None or any(item.reference.quality for item in (*products, *categories)):
            projection = {**raw, "products": [], "productCategories": []}
            if created is None:
                projection["creationDateTimeUtc"] = get(raw, "dateTimeUtc", required=True)
        action = adapt_action(projection, resolver, product_namespaces=selection.action_product_namespaces)
        classified = classify_action_system_name(action.action_system_name, selection.interaction_rules())
        kind = {InteractionType.VIEW: HistoryKind.VIEW, InteractionType.FAVORITE: HistoryKind.FAVORITE}.get(classified, HistoryKind.UNCLASSIFIED)
        flags = [Q.AS_OF_UNVERIFIED]
        if kind is HistoryKind.UNCLASSIFIED:
            flags.append(Q.UNCLASSIFIED_ACTION)
        elif not products:
            flags.append(Q.MISSING_PRODUCT)
        if created is None:
            flags.append(Q.MISSING_CREATION_TIME)
        for item in (*products, *categories):
            flags.extend(item.reference.quality)
        if action.product_view_price is not None:
            flags.append(Q.CURRENCY_UNCONFIRMED)
            if action.product_view_price < 0:
                flags.append(Q.NEGATIVE_AMOUNT)
        details = ActionDetails(action.action_id, action.action_system_name, created,
                                HistoryChannel(action.channel_system_name, action.channel_external_id,
                                               action.channel_mindbox_id, action.channel_name),
                                products, categories, action.action_name, action.brand_system_name,
                                action.product_view_price, action.product_view_is_available)
        return HistoryEvent(history_event_key(HistorySource.ACTION, action.action_id), HistorySource.ACTION,
                            action.action_id, action.customer_id, action.source_customer_id, kind,
                            action.event_datetime_utc, fingerprint, origin, details, _flags(flags))
    except Exception:
        raise HistoryNormalizationError("Cannot normalize Action history record") from None


def normalize_order(raw: Mapping, resolver: CustomerIdResolver, *,
                    selection: MindboxSelectionConfig = DEFAULT_SELECTION,
                    provenance: HistoryProvenance | None = None) -> HistoryEvent:
    """One logical order + one snapshot + all source lines, never a purchase count.

    Strict adapter handles representable lines/header; unresolved product lines
    use only the minimal separate projection through the same scalar validators.
    No source snapshot is selected or merged with another snapshot here.
    """
    line_index = None
    try:
        origin, fingerprint = _context(raw, resolver, selection, provenance)
        raw_lines = objects(raw, "lines", required=True)
        references, source_ids, representable, adapter_ordinals = [], [], [], []
        for i, line in enumerate(raw_lines):
            line_index = i + 1
            ref = _reference(mapping(line, "product"), selection.order_product_namespaces)
            references.append(ref)
            source_ids.append(identifier(line, "id", required=False))
            if ref.product is not None:
                representable.append(line)
                adapter_ordinals.append(i + 1)
        line_index = None
        try:
            adapted = iter(adapt_order({**raw, "lines": representable}, resolver,
                                       product_namespaces=selection.order_product_namespaces))
        except AdapterError as exc:
            if exc.line_index is not None:
                exc.line_index = adapter_ordinals[exc.line_index - 1]
            raise
        # Preserve common fields also for empty or wholly unresolved orders;
        # adapt_order validates their header but returns no header object.
        source = identifier(raw, "customer.ids.mindboxId")
        order_id = identifier(raw, "ids.mindboxId")
        event_key = history_event_key(HistorySource.ORDER, order_id)
        snapshot_key = _key("snapshot", event_key, fingerprint.hex())
        order = OrderDetails(order_id, identifier(raw, "ids.retailOrderId", required=False),
                             timestamp(raw, "firstAction.dateTimeUtc"),
                             HistoryChannel(external_id=identifier(raw, "firstAction.channel.ids.externalId"),
                                            name=text(raw, "firstAction.channel.name", required=True)),
                             number(raw, "totalPrice", required=False), number(raw, "deliveryCost", required=False))
        counts = Counter(value for value in source_ids if value is not None)
        flags = [Q.AS_OF_UNVERIFIED, Q.ORDER_UPDATE_TIME_UNKNOWN, Q.CURRENCY_UNCONFIRMED]
        if any(value is not None and value < 0 for value in (order.total_price, order.delivery_cost)):
            flags.append(Q.NEGATIVE_AMOUNT)
        lines = []
        for i, (line, ref, original_id) in enumerate(zip(raw_lines, references, source_ids)):
            line_index = i + 1
            record = next(adapted) if ref.product is not None else None
            quantity = record.quantity if record is not None else number(line, "quantity")
            price = record.price_of_line if record is not None else number(line, "priceOfLine")
            base = record.base_price_per_item if record is not None else number(line, "basePricePerItem", required=False)
            status = record.line_status if record is not None else identifier(line, "status.ids.externalId")
            name = record.product_name if record is not None else _product_name(line)
            line_number = record.line_number if record is not None else integer(line, "number")
            line_key = _key("line-occurrence", snapshot_key, i)
            synthetic = original_id is None
            line_flags = list(ref.quality)
            if synthetic:
                line_flags.append(Q.SYNTHETIC_LINE_ID)
            elif counts[original_id] > 1:
                line_flags.append(Q.DUPLICATE_LINE_ID)
            if quantity < 0:
                line_flags.append(Q.NEGATIVE_QUANTITY)
            if quantity != quantity.to_integral_value():
                line_flags.append(Q.FRACTIONAL_QUANTITY)
            if price < 0 or base is not None and base < 0:
                line_flags.append(Q.NEGATIVE_AMOUNT)
            flags.extend(line_flags)
            source_line_key = _key("source-line", event_key, original_id) if original_id is not None and counts[original_id] == 1 else None
            lines.append(OrderLineDetails(line_key, source_line_key, original_id if not synthetic else "synthetic:" + line_key,
                                          original_id, synthetic, i, line_number, ref, name, quantity, status, price, base,
                                          _flags(line_flags)))
        snapshot = OrderSnapshot(event_key, snapshot_key, fingerprint, order, tuple(lines), origin)
        return HistoryEvent(event_key, HistorySource.ORDER, order_id, resolver.resolve(source), source, HistoryKind.ORDER,
                            order.order_datetime_utc, fingerprint, origin, snapshot, _flags(flags))
    except Exception as exc:
        raise HistoryNormalizationError("Cannot normalize Order history record",
                                        line_index=getattr(exc, "line_index", None) or line_index) from None


def compare_history_events(first: HistoryEvent, second: HistoryEvent) -> HistoryComparison:
    """Source equality only, independent of provenance/current merge/selection.

    Changed variants retain the same logical key; they are not independent
    purchases and this comparison supplies no last-write-wins ordering.
    """
    if not isinstance(first, HistoryEvent) or not isinstance(second, HistoryEvent):
        raise HistoryNormalizationError("Expected two history events")
    if first.event_key != second.event_key:
        return HistoryComparison.DISTINCT
    return HistoryComparison.EQUIVALENT if first.fingerprint == second.fingerprint else HistoryComparison.CONFLICT
