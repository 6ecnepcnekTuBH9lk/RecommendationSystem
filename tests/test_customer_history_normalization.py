"""Small synthetic Mindbox records; no dataset, model, database or Qt lifecycle."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import traceback
from types import MappingProxyType

import pytest

from Application.customer_history_normalization import (
    HistoryNormalizationError, compare_history_events, history_event_key, normalize_action, normalize_order,
)
from Application.customer_history_records import (
    ActionDetails, HistoryComparison as C, HistoryKind as K, HistoryProductReference,
    HistoryProvenance, HistoryQuality as Q, HistorySource,
)
from Application.mindbox.adapters import adapt_action, adapt_order
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.order_dedup import semantic_fingerprint
from Application.mindbox.records import CustomerMergeRecord, ProductKey
from Application.mindbox.selection import DEFAULT_FAVORITE_ACTIONS, DEFAULT_VIEW_ACTIONS, MindboxSelectionConfig


STAMP = "2026-01-01T12:00:00Z"
UTC = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
def resolver():
    return CustomerIdResolver((CustomerMergeRecord("merge-1", UTC, "B", ("A",)),
                               CustomerMergeRecord("merge-2", UTC, "C", ("B",))))


@pytest.fixture
def action():
    return {"ids": {"mindboxId": "action-1"}, "customer": {"ids": {"mindboxId": "A"}},
            "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}, "name": "Синтетическое действие"},
            "dateTimeUtc": STAMP, "creationDateTimeUtc": "2026-01-02T12:00:00Z",
            "channel": {"ids": {"systemName": "web", "externalId": "001", "mindboxId": "002"}, "name": "Синтетический канал"},
            "products": [{"ids": {"offline1C": "000123456789"}}],
            "productView": {"price": Decimal("10.500"), "isAvailable": True}}


def line(**fields):
    return {"id": "line-1", "number": 1, "product": {"ids": {"offline1C": "000123456789"}, "name": "Синтетический товар"},
            "quantity": Decimal("2.50"), "priceOfLine": Decimal("15.123456789"),
            "basePricePerItem": Decimal("7.00"), "status": {"ids": {"externalId": "CP"}}, **fields}


@pytest.fixture
def order():
    return {"ids": {"mindboxId": "order-1", "retailOrderId": "000001"},
            "customer": {"ids": {"mindboxId": "A"}},
            "firstAction": {"dateTimeUtc": STAMP, "channel": {"ids": {"externalId": "001"}, "name": "Синтетический канал"}},
            "totalPrice": Decimal("99.999"), "deliveryCost": Decimal("0.25"), "lines": [line()]}


@pytest.mark.parametrize("name,kind", [(n, K.VIEW) for n in DEFAULT_VIEW_ACTIONS] + [(n, K.FAVORITE) for n in DEFAULT_FAVORITE_ACTIONS])
def test_exact_selection_and_existing_action_adapter_parity(action, resolver, name, kind):
    action["actionTemplate"]["ids"]["systemName"] = name
    actual = normalize_action(action, resolver)
    baseline = adapt_action(action, resolver)
    assert actual.kind is kind and actual.source is HistorySource.ACTION
    assert actual.source_customer_id == "A" and actual.canonical_customer_id == "C"
    assert actual.source_event_id == baseline.action_id
    assert actual.event_datetime_utc == baseline.event_datetime_utc == UTC
    assert actual.details.creation_datetime_utc == baseline.creation_datetime_utc != actual.event_datetime_utc
    assert actual.details.products[0].reference.product == baseline.products[0]
    assert actual.details.product_view_price.as_tuple() == Decimal("10.500").as_tuple()
    assert actual.details.channel.external_id == "001" and actual.details.channel.mindbox_id == "002"
    assert Q.AS_OF_UNVERIFIED in actual.quality and Q.CURRENCY_UNCONFIRMED in actual.quality


@pytest.mark.parametrize("name", ["UnknownAction", "prosmotrProdukta", "ProsmotrProdukta ", "UstanovkaSpiskaProduktovV", "RemoveFavorite"])
def test_unmapped_actions_are_not_views_or_favorite_deletions(action, resolver, name):
    action["actionTemplate"]["ids"]["systemName"] = name
    actual = normalize_action(action, resolver)
    assert actual.kind is K.UNCLASSIFIED and Q.UNCLASSIFIED_ACTION in actual.quality
    assert len(actual.details.products) == 1


def test_selection_is_caller_supplied_and_not_reimplemented(action, resolver):
    selection = MindboxSelectionConfig(view_action_system_names=("CustomView",), favorite_action_system_names=("CustomFavorite",))
    assert normalize_action(action, resolver, selection=selection).kind is K.UNCLASSIFIED
    action["actionTemplate"]["ids"]["systemName"] = "CustomFavorite"
    assert normalize_action(action, resolver, selection=selection).kind is K.FAVORITE


def test_action_product_occurrences_and_namespaces_are_not_deduplicated(action, resolver):
    action["products"] = [{"ids": {"offline1C": "000123456789"}}, {"ids": {"kanzlerKz": "000123456789"}},
                          {"ids": {"offline1C": "000123456789"}}]
    actual = normalize_action(action, resolver)
    assert [p.ordinal for p in actual.details.products] == [0, 1, 2]
    assert [p.reference.product.namespace for p in actual.details.products] == ["offline1C", "kanzlerKz", "offline1C"]
    assert all(p.reference.product.value == "000123456789" for p in actual.details.products)


@pytest.mark.parametrize("ids,flag", [({}, Q.MISSING_PRODUCT), ({"website": "000001"}, Q.UNSUPPORTED_PRODUCT_NAMESPACE),
                                     ({"offline1C": "001", "kanzlerKz": "002"}, Q.AMBIGUOUS_PRODUCT_NAMESPACE)])
def test_unresolved_references_are_preserved_without_arbitrary_selection(action, order, resolver, ids, flag):
    action["products"] = [{"ids": ids}]
    order["lines"][0]["product"]["ids"] = ids
    for event in (normalize_action(action, resolver), normalize_order(order, resolver)):
        ref = event.details.products[0].reference if isinstance(event.details, ActionDetails) else event.details.lines[0].reference
        assert ref.product is None and flag in ref.quality and flag in event.quality
        assert {(key.namespace, key.value) for key in ref.keys} == set(ids.items())
    assert normalize_order(order, resolver).details.lines[0].quantity == Decimal("2.50")


def test_supported_reference_still_uses_adapter_selection_when_other_ids_present(action, resolver):
    action["products"][0]["ids"]["website"] = "separate-source-id"
    ref = normalize_action(action, resolver).details.products[0].reference
    assert ref.product == ProductKey("offline1C", "000123456789") and len(ref.keys) == 2


def test_missing_products_categories_and_creation_time_have_explicit_quality(action, resolver):
    action.pop("creationDateTimeUtc")
    action["products"] = []
    action["productCategories"] = [{"ids": {"website": "category-1"}}]
    actual = normalize_action(action, resolver)
    assert actual.details.creation_datetime_utc is None
    assert actual.details.products == () and len(actual.details.categories) == 1
    assert Q.MISSING_CREATION_TIME in actual.quality and Q.MISSING_PRODUCT in actual.quality
    assert Q.UNSUPPORTED_PRODUCT_NAMESPACE in actual.quality


def test_action_equivalence_distinct_ids_and_conflicting_content(action, resolver):
    first = normalize_action(action, resolver)
    repeat = normalize_action(deepcopy(action), resolver, provenance=HistoryProvenance("API", "different-version"))
    assert compare_history_events(first, repeat) is C.EQUIVALENT
    action["ids"]["mindboxId"] = "action-2"
    distinct = normalize_action(action, resolver)
    assert compare_history_events(first, distinct) is C.DISTINCT
    action["ids"]["mindboxId"] = "action-1"
    action["products"][0]["ids"]["offline1C"] = "999999"
    conflict = normalize_action(action, resolver)
    assert conflict.event_key == first.event_key and compare_history_events(first, conflict) is C.CONFLICT


def test_order_snapshot_repeat_conflict_and_logical_key(order, resolver):
    first = normalize_order(order, resolver)
    repeat = normalize_order(deepcopy(order), resolver, provenance=HistoryProvenance("MANUAL", "other-source"))
    assert first.kind is K.ORDER
    assert first.event_key == first.details.logical_order_key
    assert first.fingerprint == first.details.fingerprint == semantic_fingerprint(order)
    assert first.details.snapshot_key == repeat.details.snapshot_key
    assert compare_history_events(first, repeat) is C.EQUIVALENT
    order["lines"][0]["status"]["ids"]["externalId"] = "cancelled-unknown-semantics"
    conflict = normalize_order(order, resolver)
    assert conflict.event_key == first.event_key
    assert conflict.details.snapshot_key != first.details.snapshot_key
    assert compare_history_events(first, conflict) is C.CONFLICT
    assert conflict.details.lines[0].raw_status == "cancelled-unknown-semantics"
    assert not hasattr(conflict.details, "latest_snapshot")


def test_order_numeric_fingerprint_and_object_key_order_match_existing_rules(order, resolver):
    repeat = deepcopy(order)
    repeat["totalPrice"] = Decimal("99.9990")
    repeat["lines"][0]["quantity"] = Decimal("2.5000")
    repeat = dict(reversed(list(repeat.items())))
    assert compare_history_events(normalize_order(order, resolver), normalize_order(repeat, resolver)) is C.EQUIVALENT


def test_missing_line_ids_are_snapshot_local_and_reordering_is_not_cross_snapshot_identity(order, resolver):
    order["lines"] = [line(), line(number=2, product={"ids": {"kanzlerKz": "009999"}})]
    for row in order["lines"]:
        row.pop("id")
    first = normalize_order(order, resolver)
    lines = first.details.lines
    assert all(row.is_synthetic and row.source_line_id is None and row.source_line_key is None for row in lines)
    assert lines[0].line_id != lines[1].line_id
    assert lines[0].line_key != lines[1].line_key
    order["lines"].reverse()
    changed = normalize_order(order, resolver)
    assert changed.details.snapshot_key != first.details.snapshot_key
    assert {row.line_id for row in changed.details.lines}.isdisjoint(row.line_id for row in lines)
    assert Q.SYNTHETIC_LINE_ID in first.quality


def test_explicit_ids_have_stable_order_scoped_identity_and_occurrences_do_not_collide(order, resolver):
    first = normalize_order(order, resolver)
    order["lines"][0]["quantity"] = Decimal("3")
    changed = normalize_order(order, resolver)
    assert first.details.lines[0].source_line_key == changed.details.lines[0].source_line_key
    assert first.details.lines[0].line_key != changed.details.lines[0].line_key
    order["ids"]["mindboxId"] = "order-2"
    other = normalize_order(order, resolver)
    assert other.details.lines[0].source_line_key != changed.details.lines[0].source_line_key
    order["lines"].append(deepcopy(order["lines"][0]))
    duplicates = normalize_order(order, resolver)
    assert all(row.source_line_key is None and Q.DUPLICATE_LINE_ID in row.quality for row in duplicates.details.lines)
    assert len({row.line_key for row in duplicates.details.lines}) == 2


@pytest.mark.parametrize("status", ["CP", "delivering", "F", "unknown-return", "unknown-cancel"])
def test_raw_statuses_quantity_and_money_are_not_business_metrics(order, resolver, status):
    order["lines"][0]["status"]["ids"]["externalId"] = status
    order["lines"][0]["quantity"] = Decimal("-2.500")
    order["lines"][0]["priceOfLine"] = Decimal("-15.123456789")
    event = normalize_order(order, resolver)
    baseline, = adapt_order(order, resolver)
    actual = event.details.lines[0]
    assert actual.raw_status == baseline.line_status == status and event.kind is K.ORDER
    assert actual.quantity.as_tuple() == Decimal("-2.500").as_tuple()
    assert actual.price_of_line == baseline.price_of_line
    assert actual.base_price_per_item == baseline.base_price_per_item
    assert Q.NEGATIVE_QUANTITY in actual.quality and Q.FRACTIONAL_QUANTITY in actual.quality
    assert Q.NEGATIVE_AMOUNT in actual.quality and Q.CURRENCY_UNCONFIRMED in event.quality
    assert event.details.order.total_price == Decimal("99.999") and event.details.order.delivery_cost == Decimal("0.25")
    assert not hasattr(event.details.order, "currency") and not hasattr(event.details.order, "paid")
    assert Q.ORDER_UPDATE_TIME_UNKNOWN in event.quality


def test_empty_order_keeps_validated_header_and_origin(order, resolver):
    order["lines"] = []
    origin = HistoryProvenance("API", "synthetic-version", 1, 0, UTC + timedelta(days=20))
    actual = normalize_order(order, resolver, provenance=origin)
    assert actual.details.lines == () and actual.details.order.order_id == "order-1"
    assert actual.event_datetime_utc == UTC and actual.provenance == actual.details.provenance == origin
    assert actual.provenance.observed_at_utc != actual.event_datetime_utc
    assert actual.canonical_customer_id == "C" and actual.source_customer_id == "A"


@pytest.mark.parametrize("value", ["2026-01-01T15:00:00+03:00", "2026-01-01T12:00:00", "2026-01-01 12:00:00.123456Z"])
def test_utc_dates_and_microseconds_preserved(action, order, resolver, value):
    action["dateTimeUtc"] = value
    order["firstAction"]["dateTimeUtc"] = value
    for actual in (normalize_action(action, resolver), normalize_order(order, resolver)):
        assert actual.event_datetime_utc.utcoffset() == timedelta(0)
        assert actual.event_datetime_utc.hour == 12
        assert actual.event_datetime_utc.microsecond == (123456 if "123456" in value else 0)


@pytest.mark.parametrize("source", ["A", "B", "C", "unknown"])
def test_current_canonical_identity_uses_existing_resolver(action, order, resolver, source):
    for raw, normalize in ((action, normalize_action), (order, normalize_order)):
        raw["customer"]["ids"]["mindboxId"] = source
        actual = normalize(raw, resolver)
        assert actual.source_customer_id == source
        assert actual.canonical_customer_id == ("unknown" if source == "unknown" else "C")


def test_keys_are_deterministic_source_scoped_and_do_not_embed_ids(action, order, resolver):
    action["ids"]["mindboxId"] = order["ids"]["mindboxId"] = "same-sensitive-id"
    a, o = normalize_action(action, resolver), normalize_order(order, resolver)
    assert a.event_key != o.event_key
    assert a.event_key == history_event_key(HistorySource.ACTION, "same-sensitive-id")
    assert o.event_key == normalize_order(order, resolver).event_key
    assert "same-sensitive-id" not in a.event_key + o.event_key + o.details.snapshot_key


def test_immutable_contracts_detached_from_raw_and_safe_repr(action, order, resolver):
    action["customer"]["ids"]["mindboxId"] = "PRIVATE_SOURCE_CUSTOMER"
    order["customer"]["ids"]["mindboxId"] = "PRIVATE_SOURCE_CUSTOMER"
    events = normalize_action(action, resolver), normalize_order(order, resolver)
    action["products"][0]["ids"]["offline1C"] = "MUTATED"
    order["lines"][0]["quantity"] = Decimal(999)
    for event in events:
        with pytest.raises(FrozenInstanceError):
            event.kind = K.VIEW
        assert not hasattr(event, "__dict__")
        assert "PRIVATE_SOURCE_CUSTOMER" not in repr(event)
        assert "000123456789" not in repr(event)
        with pytest.raises(FrozenInstanceError):
            event.provenance.source_kind = "API"
    assert events[0].details.products[0].reference.product.value == "000123456789"
    assert events[1].details.lines[0].quantity == Decimal("2.50")
    assert "order-1" not in repr(events[1].details) + repr(events[1].details.order) + repr(events[1].details.lines[0])
    assert "action-1" not in repr(events[0].details) + repr(events[0].details.products[0].reference)
    with pytest.raises(TypeError):
        HistoryProductReference([ProductKey("offline1C", "id")], None)


@pytest.mark.parametrize("value", [True, False, 1.5, "", " id", "id ", None, [], {}])
def test_bad_present_identifiers_are_not_repaired_or_exposed(action, order, resolver, value, caplog):
    action["customer"]["ids"]["mindboxId"] = value
    order["customer"]["ids"]["mindboxId"] = value
    for raw, normalize in ((action, normalize_action), (order, normalize_order)):
        with pytest.raises(HistoryNormalizationError) as error:
            normalize(raw, resolver)
        assert " id" not in str(error.value) and str(error.value).startswith("Cannot normalize")
    assert not caplog.text


@pytest.mark.parametrize("value", [None, True, 1.5, "", " 001", "001 "])
def test_malformed_existing_product_identifier_remains_an_error_even_for_unmapped(action, resolver, value):
    action["actionTemplate"]["ids"]["systemName"] = "Unknown"
    action["products"] = [{"ids": {"website": value}}]
    with pytest.raises(HistoryNormalizationError):
        normalize_action(action, resolver)


@pytest.mark.parametrize("mutate", ["quantity", "price", "timestamp", "status", "name", "optional"])
def test_bad_values_return_static_safe_error_and_safe_line_index(order, resolver, mutate):
    order["customer"]["email"] = "PRIVATE-person@example.test"
    row = order["lines"][0]
    if mutate == "quantity":
        row["quantity"] = Decimal("NaN")
    elif mutate == "price":
        row["priceOfLine"] = "PRIVATE-person@example.test"
    elif mutate == "timestamp":
        order["firstAction"]["dateTimeUtc"] = "PRIVATE-person@example.test"
    elif mutate == "status":
        row["status"]["ids"]["externalId"] = False
    elif mutate == "name":
        row["product"]["name"] = ""
    else:
        row["basePricePerItem"] = True
    try:
        normalize_order(order, resolver)
        pytest.fail("Expected invalid source")
    except HistoryNormalizationError as error:
        assert "PRIVATE-person" not in str(error) + repr(error) + traceback.format_exc()
        assert error.line_index in (None, 1)


def test_no_raw_mutation_or_provenance_in_fingerprint(action, order, resolver):
    for raw, normalize in ((action, normalize_action), (order, normalize_order)):
        before = deepcopy(raw)
        actual = normalize(MappingProxyType(raw), resolver, provenance=HistoryProvenance("MANUAL", "snapshot"))
        assert raw == before and actual.fingerprint == semantic_fingerprint(raw)


def test_compare_is_source_equality_not_a_latest_or_current_merge_rule(action, resolver):
    first = normalize_action(action, resolver)
    other = normalize_action(action, CustomerIdResolver())
    assert first.canonical_customer_id != other.canonical_customer_id
    assert compare_history_events(first, other) is C.EQUIVALENT
    with pytest.raises(HistoryNormalizationError):
        compare_history_events(first, {})


def test_invalid_provenance_and_direct_contract_values_are_rejected(action, resolver):
    with pytest.raises(ValueError):
        HistoryProvenance(observed_at_utc=datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        HistoryProvenance(part_number=0)
    with pytest.raises(ValueError):
        HistoryProvenance(record_ordinal=-1)
    actual = normalize_action(action, resolver)
    with pytest.raises(ValueError):
        replace(actual, fingerprint=b"bad")
    with pytest.raises(ValueError):
        replace(actual, kind=K.ORDER)


def test_nested_labels_cannot_retain_mutable_inputs(action, order, resolver):
    a = normalize_action(action, resolver)
    o = normalize_order(order, resolver)
    with pytest.raises(TypeError):
        replace(a.details, action_name=[])
    with pytest.raises(TypeError):
        replace(a.details, brand_system_name={})
    with pytest.raises(TypeError):
        replace(o.details.lines[0], product_name=[])
    with pytest.raises(TypeError):
        replace(o.details, lines=list(o.details.lines))
    with pytest.raises(TypeError):
        replace(a.details, products=list(a.details.products))


@pytest.mark.parametrize("unresolved", [False, True])
def test_error_line_index_is_original_ordinal_not_filtered_adapter_ordinal(order, resolver, unresolved):
    order["lines"] = [line(id="first", product={"ids": {"website": "unknown"}}),
                      line(id="second", quantity="invalid",
                           product={"ids": {"website" if unresolved else "offline1C": "valid-id"}})]
    with pytest.raises(HistoryNormalizationError) as error:
        normalize_order(order, resolver)
    assert error.value.line_index == 2


def test_wholly_missing_order_products_keep_lines_and_validate_all_scalar_fields(order, resolver):
    order["lines"][0].pop("product")
    actual = normalize_order(order, resolver)
    assert actual.details.lines[0].reference.product is None
    assert Q.MISSING_PRODUCT in actual.details.lines[0].quality
    assert actual.details.lines[0].quantity == Decimal("2.50")
    order["lines"][0]["status"]["ids"]["externalId"] = " bad-id"
    with pytest.raises(HistoryNormalizationError):
        normalize_order(order, resolver)


def test_provenance_datetime_converts_to_utc_without_claiming_availability(action, resolver):
    observed = datetime(2026, 1, 2, 15, tzinfo=timezone(timedelta(hours=3)))
    origin = HistoryProvenance("API", observed_at_utc=observed)
    event = normalize_action(action, resolver, provenance=origin)
    assert event.provenance.observed_at_utc == datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    assert Q.AS_OF_UNVERIFIED in event.quality


@pytest.mark.parametrize("field", ["id", "number", "quantity", "priceOfLine", "status"])
def test_unresolved_product_projection_does_not_relax_scalar_validation(order, resolver, field):
    row = order["lines"][0]
    row["product"] = {"ids": {"website": "valid-unmapped-id"}}
    row[field] = True
    with pytest.raises(HistoryNormalizationError):
        normalize_order(order, resolver)
