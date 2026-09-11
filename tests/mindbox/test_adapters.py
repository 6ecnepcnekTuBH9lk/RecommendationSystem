import copy
import json
import traceback
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from Application.mindbox.adapters import (
    AdapterError, adapt_action, adapt_customer, adapt_customer_merge, adapt_order,
)
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import EXPORT_ROOTS, RawExportError, iter_export
from Application.mindbox.records import ProductKey
from scripts import mindbox_adapter_smoke as cli


STAMP = "2026-01-01T12:30:45.123456Z"


@pytest.fixture
def action():
    return {"ids": {"mindboxId": 1001}, "actionTemplate": {"ids": {"systemName": "Synthetic.Template"}},
            "dateTimeUtc": STAMP, "creationDateTimeUtc": STAMP, "customer": {"ids": {"mindboxId": 100}}}


@pytest.fixture
def order():
    return {"ids": {"mindboxId": 2001},
            "firstAction": {"dateTimeUtc": STAMP, "channel": {"ids": {"externalId": "synthetic-store"}, "name": "synthetic channel"}},
            "customer": {"ids": {"mindboxId": 100}},
            "lines": [{"id": "synthetic-line", "number": 1, "quantity": 2.5,
                       "basePricePerItem": 10.1, "priceOfLine": 25.25,
                       "product": {"ids": {"offline1C": "000123456789"}, "name": "synthetic product"},
                       "status": {"ids": {"externalId": "synthetic-status"}}}]}


@pytest.fixture
def raw_merge():
    return {"id": 1, "dateTimeUtc": STAMP, "resultingCustomer": {"ids": {"mindboxId": 300}},
            "mergedCustomers": [{"ids": {"mindboxId": 100}}, {"ids": {"mindboxId": 200}}]}


@pytest.fixture
def resolver(raw_merge):
    return CustomerIdResolver([adapt_customer_merge(raw_merge)])


def write_part(directory, name, records, number=1):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_part_{number:03d}.json"
    path.write_text(json.dumps({EXPORT_ROOTS[name]: records}), encoding="utf-8")
    return path


def test_reader_multipart_and_exact_decimal(tmp_path):
    write_part(tmp_path, "customers", [{"ids": {"mindboxId": 1}}])
    second = write_part(tmp_path, "customers", [], 2)
    second.write_text('{"customers": [{"value": 0.1234567890123456789}]}', encoding="utf-8")
    reader = iter_export("customers", input_dir=tmp_path)
    assert next(reader)["ids"]["mindboxId"] == 1
    assert next(reader)["value"] == Decimal("0.1234567890123456789")
    with pytest.raises(StopIteration):
        next(reader)


def test_reader_empty_and_latest(tmp_path):
    write_part(tmp_path / "orders" / "20260101_000000", "orders", [{}])
    write_part(tmp_path / "orders" / "20260102_000000", "orders", [])
    assert list(iter_export("orders", raw_root=tmp_path)) == []


@pytest.mark.parametrize("text", ['{"orders": [', '{"wrong": []}', '{"orders": [null]}', '{"orders": {}}'])
def test_reader_errors(tmp_path, text):
    path = tmp_path / "orders_part_001.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RawExportError):
        list(iter_export("orders", input_dir=tmp_path))


def test_basic_action_without_optional_fields(action, resolver):
    record = adapt_action(action, resolver)
    assert record.action_id == "1001"
    assert record.source_customer_id == "100"
    assert record.customer_id == "300"
    assert record.event_datetime_utc.tzinfo == timezone.utc
    assert record.creation_datetime_utc.microsecond == 123456
    assert record.action_system_name == "Synthetic.Template"
    assert record.products == record.categories == ()
    assert record.channel_name is None
    assert record.product_view_price is None
    assert record.product_view_is_available is None
    with pytest.raises(FrozenInstanceError):
        record.customer_id = "other"


@pytest.mark.parametrize("count", [0, 1, 3])
def test_action_products_categories_and_product_view(action, resolver, count):
    action["products"] = [{"ids": {"offline1C": f"00000-{index}-long"}} for index in range(count)]
    action["productCategories"] = [{"ids": {"offline1C": "category-long"}}]
    action["productView"] = {"price": 12.34, "isAvailable": False}
    record = adapt_action(action, resolver)
    assert len(record.products) == count
    assert record.categories == (ProductKey("offline1C", "category-long"),)
    if count:
        assert record.products[0].value == "00000-0-long"
    assert record.product_view_price == Decimal("12.34")
    assert record.product_view_is_available is False


def test_action_channel_fields_and_no_classification(action, resolver):
    action["actionTemplate"]["ids"]["systemName"] = "ProsmotrProduktaVApiMethod"
    action["actionTemplate"]["name"] = "synthetic name"
    action["brand"] = {"ids": {"systemName": "synthetic brand"}}
    action["channel"] = {"ids": {"systemName": "synthetic channel", "externalId": "001", "mindboxId": 4}, "name": "channel name"}
    record = adapt_action(action, resolver)
    assert record.channel_external_id == "001"
    assert record.channel_mindbox_id == "4"
    assert record.brand_system_name == "synthetic brand"
    assert record.action_system_name == "ProsmotrProduktaVApiMethod"
    assert not hasattr(record, "weight")
    assert not hasattr(record, "action_type")


@pytest.mark.parametrize("value", [None, True, 1.5, "", " 100", [], {}])
def test_action_malformed_required_customer_id(action, resolver, value):
    action["customer"]["ids"]["mindboxId"] = value
    with pytest.raises(AdapterError, match="customer.ids.mindboxId"):
        adapt_action(action, resolver)


@pytest.mark.parametrize("key", ["dateTimeUtc", "creationDateTimeUtc"])
@pytest.mark.parametrize("value", [None, "", "not a date", "2026-02-30T12:00:00Z", "2026-01-01", 123,
                                  "2026-01-01T00:00:00+00:99", "2026-01-01T00:00:00+25:00"])
def test_required_action_dates(action, resolver, key, value):
    action[key] = value
    with pytest.raises(AdapterError, match=key):
        adapt_action(action, resolver)


@pytest.mark.parametrize("stamp", ["2026-01-01 12:30:45", "2026-01-01T12:30:45Z", "2026-01-01T15:30:45+03:00"])
def test_timestamp_utc_semantics(action, resolver, stamp):
    action["dateTimeUtc"] = stamp
    result = adapt_action(action, resolver).event_datetime_utc
    assert result == datetime(2026, 1, 1, 12, 30, 45, tzinfo=timezone.utc)
    assert result.utcoffset() == timedelta(0)


@pytest.mark.parametrize("fraction,microsecond", [("1", 100000), ("12", 120000), ("123", 123000),
                                               ("1234", 123400), ("12345", 123450), ("123456", 123456)])
@pytest.mark.parametrize("suffix", ["", "Z", "+03:00"])
def test_all_valid_fraction_lengths_on_python310(action, resolver, fraction, microsecond, suffix):
    action["dateTimeUtc"] = f"2026-01-01T12:30:45.{fraction}{suffix}"
    record = adapt_action(action, resolver)
    assert record.event_datetime_utc.microsecond == microsecond
    assert record.event_datetime_utc.hour == (9 if suffix == "+03:00" else 12)


@pytest.mark.parametrize("optional,value", [("channel", "malformed"), ("products", {}), ("products", [None]),
                                           ("productView", {"price": "private"}), ("productView", {"isAvailable": 1})])
def test_present_malformed_action_optional_fields_raise(action, resolver, optional, value):
    action[optional] = value
    with pytest.raises(AdapterError):
        adapt_action(action, resolver)


@pytest.mark.parametrize("namespace", ["offline1C", "kanzlerKz"])
def test_order_line_product_namespaces(order, resolver, namespace):
    order["lines"][0]["product"]["ids"] = {namespace: "00000123456789"}
    records = adapt_order(order, resolver)
    assert len(records) == 1
    line = records[0]
    assert line.product == ProductKey(namespace, "00000123456789")
    assert line.source_customer_id == "100"
    assert line.customer_id == "300"
    assert line.quantity == Decimal("2.5")
    assert line.base_price_per_item == Decimal("10.1")
    assert line.retail_order_id is None
    assert line.delivery_cost is None


def test_many_order_lines_optional_totals_and_empty_lines(order, resolver):
    order["lines"].append(copy.deepcopy(order["lines"][0]))
    order["lines"][1]["id"] = "second"
    order["lines"][1]["number"] = 2
    order["ids"]["retailOrderId"] = "0000456"
    order["totalPrice"] = 50.5
    order["deliveryCost"] = 0
    lines = adapt_order(order, resolver)
    assert len(lines) == 2
    assert lines[1].line_id == "second"
    assert lines[0].retail_order_id == "0000456"
    assert lines[0].order_total_price == Decimal("50.5")
    assert lines[0].delivery_cost == Decimal(0)
    order["lines"] = []
    assert adapt_order(order, resolver) == ()


def test_order_without_product_name_creates_purchase(order, resolver):
    from Application.interactions import InteractionBuilder, InteractionType

    del order["lines"][0]["product"]["name"]
    order["lines"][0]["status"]["ids"]["externalId"] = "CP"
    line, = adapt_order(order, resolver)
    assert line.product_name is None
    builder = InteractionBuilder()
    interaction = builder.from_order_line(line)
    assert interaction.interaction_type is InteractionType.PURCHASE
    assert interaction.product is line.product
    assert interaction.quantity == line.quantity == Decimal("2.5")
    assert builder.diagnostics.actions_malformed == 0


@pytest.mark.parametrize("name", ["synthetic product", " Название товара ", None])
def test_optional_order_product_name_preserved(order, resolver, name):
    order["lines"][0]["product"]["name"] = name
    assert adapt_order(order, resolver)[0].product_name == name


@pytest.mark.parametrize("name", [123, False, [], {}, "", "   "])
def test_invalid_present_order_product_name_rejected(order, resolver, name):
    order["lines"][0]["product"]["name"] = name
    with pytest.raises(AdapterError, match="product.name"):
        adapt_order(order, resolver)


@pytest.mark.parametrize("ids", [{}, {"unsupported": "secret"}, {"offline1C": "a", "kanzlerKz": "b"},
                                  {"offline1C": None, "kanzlerKz": "b"}, {"offline1C": ""}])
def test_order_missing_or_ambiguous_product(order, resolver, ids):
    order["lines"][0]["product"]["ids"] = ids
    with pytest.raises(AdapterError):
        adapt_order(order, resolver)


@pytest.mark.parametrize("key,value", [("quantity", None), ("number", True), ("basePricePerItem", "1.0"),
                                      ("priceOfLine", float("nan")), ("id", ""), ("product", "bad")])
def test_malformed_order_line(order, resolver, key, value):
    order["lines"][0][key] = value
    with pytest.raises(AdapterError):
        adapt_order(order, resolver)


def test_actions_orders_do_not_copy_profile_or_action_order(action, order, resolver):
    profile = {"firstName": "PRIVATE", "email": "private@example.test", "mobilePhone": 123,
               "birthDate": "malformed but ignored copy", "customFields": {"value": "private"}}
    action["customer"].update(profile)
    order["customer"].update(profile)
    action["order"] = order
    for record in (adapt_action(action, resolver), adapt_order(order, resolver)[0]):
        names = {field.name for field in fields(record)}
        assert not names.intersection({"first_name", "email", "mobile_phone", "birth_date", "custom_fields", "order"})


def test_minimal_customer(resolver):
    record = adapt_customer({"ids": {"mindboxId": 100}}, resolver)
    assert record.source_customer_id == "100"
    assert record.customer_id == "300"
    assert record.birth_date is None
    assert record.change_datetime_utc is None
    assert dict(record.custom_fields) == {}
    assert record.balances == record.segmentations == record.subscriptions == record.discount_cards == ()
    assert record.last_activated_card is None


def test_full_customer_and_deep_immutable_structures(resolver):
    raw = {"ids": {"mindboxId": 100}, "changeDateTimeUtc": STAMP, "birthDate": "2000-02-29",
           "sex": "synthetic", "firstName": "PRIVATE", "lastName": "PRIVATE_LAST", "middleName": "PRIVATE_MIDDLE",
           "email": "secret@example.test", "mobilePhone": 79999999999, "ianaTimeZone": "Europe/Moscow", "timeZoneSource": "synthetic",
           "customFields": {"unknownFutureField": {"nested": [1, {"other": "PRIVATE_VALUE"}]}},
           "segmentations": [{"segment": {"ids": {"externalId": "PRIVATE_SEGMENT"}}}],
           "balances": [{"available": 12.5}], "subscriptions": [{"isSubscribed": False}],
           "discountCards": [{"ids": {"number": "PRIVATE_CARD"}, "status": "synthetic"}],
           "lastActivatedCard": {"ids": {"number": "PRIVATE_CARD"}, "status": {"ids": {"systemName": "synthetic"}}}}
    before = copy.deepcopy(raw)
    record = adapt_customer(raw, resolver)
    assert raw == before
    assert record.birth_date == date(2000, 2, 29)
    assert record.mobile_phone == "79999999999"
    assert record.custom_fields["unknownFutureField"]["nested"][1]["other"] == "PRIVATE_VALUE"
    assert record.subscriptions[0]["isSubscribed"] is False
    assert record.discount_cards[0]["status"] == "synthetic"
    assert record.last_activated_card["status"]["ids"]["systemName"] == "synthetic"
    raw["customFields"]["unknownFutureField"]["nested"].append(4)
    assert len(record.custom_fields["unknownFutureField"]["nested"]) == 2
    with pytest.raises(TypeError):
        record.segmentations[0]["segment"]["ids"]["externalId"] = "changed"
    with pytest.raises(TypeError):
        record.custom_fields["new"] = 1
    assert "PRIVATE" not in repr(record)


@pytest.mark.parametrize("birth", ["", "2000-02-30", "PRIVATE_INVALID_DATE", "2000-01-01T00:00:00", 1])
def test_malformed_optional_birth_date(resolver, birth):
    with pytest.raises(AdapterError, match="birthDate"):
        adapt_customer({"ids": {"mindboxId": 100}, "birthDate": birth}, resolver)


@pytest.mark.parametrize("phone", ["+79999999999", "000123"])
def test_phone_strings_preserved(resolver, phone):
    assert adapt_customer({"ids": {"mindboxId": 100}, "mobilePhone": phone}, resolver).mobile_phone == phone


@pytest.mark.parametrize("key,value", [("customFields", []), ("subscriptions", ["bad"]),
                                      ("balances", {}), ("lastActivatedCard", []), ("mobilePhone", 12.5)])
def test_malformed_optional_customer_structures(resolver, key, value):
    with pytest.raises(AdapterError):
        adapt_customer({"ids": {"mindboxId": 100}, key: value}, resolver)


def test_merge_adapter_all_sources(raw_merge):
    record = adapt_customer_merge(raw_merge)
    assert record.merged_customer_ids == ("100", "200")
    assert record.resulting_customer_id == "300"
    assert record.event_datetime_utc.tzinfo == timezone.utc


@pytest.mark.parametrize("key,value", [("mergedCustomers", []), ("mergedCustomers", [{}]),
                                      ("mergedCustomers", None), ("resultingCustomer", {"ids": {}})])
def test_merge_adapter_rejects_missing_ids(raw_merge, key, value):
    raw_merge[key] = value
    with pytest.raises(AdapterError):
        adapt_customer_merge(raw_merge)


def test_local_smoke_counts_without_pii_and_no_files_written(tmp_path, action, order, raw_merge, capsys):
    secret = "PRIVATE_SECRET_ID"
    action["customer"]["ids"]["mindboxId"] = secret
    order["customer"]["ids"]["mindboxId"] = secret
    raw_merge["mergedCustomers"] = [{"ids": {"mindboxId": secret}}]
    customer = {"ids": {"mindboxId": secret}, "firstName": "SECRET_FIRST_NAME", "email": "secret@example.test",
                "mobilePhone": "+79999999999", "customFields": {"field": "SECRET_CUSTOM_VALUE"}}
    data = {"actions": [action], "orders": [order], "customers": [customer], "customer_merges": [raw_merge]}
    for name, records in data.items():
        write_part(tmp_path / name / "20260101_000000", name, records)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert cli.main(["all", "--raw-root", str(tmp_path)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert "Customer merges: 1" in output.out
    assert "Merge aliases: 1" in output.out
    assert "Order lines: 1" in output.out
    assert "Canonicalized customer references:\nActions: 1\nOrders: 1\nCustomers: 1" in output.out
    for value in (secret, "SECRET_FIRST_NAME", "secret@example.test", "+79999999999", "SECRET_CUSTOM_VALUE", "000123456789"):
        assert value not in output.out
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*.json")}


def test_smoke_error_is_safe(tmp_path, action, capsys):
    write_part(tmp_path / "customer_merges" / "20260101_000000", "customer_merges", [])
    action["dateTimeUtc"] = "PRIVATE_INVALID_DATE"
    write_part(tmp_path / "actions" / "20260101_000000", "actions", [action])
    assert cli.main(["actions", "--raw-root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "PRIVATE_INVALID_DATE" not in output.err
    assert "dateTimeUtc" in output.err
    assert "#1" in output.err


def test_adapter_exception_traceback_does_not_echo_input(action, resolver):
    action["dateTimeUtc"] = "SECRET_DATE"
    with pytest.raises(AdapterError) as error:
        adapt_action(action, resolver)
    assert "SECRET_DATE" not in "".join(traceback.format_exception(error.value))
