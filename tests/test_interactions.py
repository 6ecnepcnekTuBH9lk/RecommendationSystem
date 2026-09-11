from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from Application.interactions import (
    InteractionBuilder, InteractionBuildError, InteractionConfigError,
    InteractionRules, InteractionSource, InteractionType, classify_action,
)
from Application.mindbox.records import ActionRecord, OrderLineRecord, ProductKey


@pytest.fixture
def action():
    return ActionRecord(
        action_id="SECRET_ACTION_ID", action_system_name="ProsmotrProdukta",
        event_datetime_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        creation_datetime_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        source_customer_id="SECRET_SOURCE_CUSTOMER", customer_id="SECRET_CANONICAL_CUSTOMER",
        products=(ProductKey("offline1C", "0000123456789_SECRET_PRODUCT"),),
    )


@pytest.fixture
def line():
    return OrderLineRecord(
        order_id="SECRET_ORDER_ID", order_datetime_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        channel_external_id="SECRET_CHANNEL", channel_name="SECRET_CHANNEL_NAME",
        source_customer_id="SECRET_SOURCE_CUSTOMER", customer_id="SECRET_CANONICAL_CUSTOMER",
        line_id="SECRET_LINE_ID", line_number=1, product=ProductKey("offline1C", "0000123456789_SECRET_PRODUCT"),
        product_name="SECRET_PRODUCT_NAME", quantity=Decimal("2.50"),
        base_price_per_item=Decimal("100.00"), price_of_line=Decimal("250.00"), line_status="CP",
    )


@pytest.mark.parametrize("name", ["ProsmotrProdukta", "ProsmotrProduktaVApiMethod"])
def test_confirmed_views_and_classification_is_separate(action, name):
    action = replace(action, action_system_name=name)
    assert classify_action(action) is InteractionType.VIEW
    result, = InteractionBuilder().from_action(action)
    assert result.interaction_type is InteractionType.VIEW
    assert result.source is InteractionSource.ACTION
    assert result.source_event_id == action.action_id
    assert result.event_datetime_utc is action.event_datetime_utc
    assert result.quantity is None


@pytest.mark.parametrize("name", ["ProsmotrProduktaExtra", "PrefixProsmotrProdukta", "prosmotrProdukta",
                                  "Prosmotr", "ProsmotrProdukta ", " ProsmotrProdukta", "Synthetic.OrderEvent"])
def test_similar_and_other_actions_are_unmapped(action, name):
    action = replace(action, action_system_name=name)
    builder = InteractionBuilder()
    assert classify_action(action) is None
    assert builder.from_action(action) == ()
    assert builder.diagnostics.actions_unmapped == 1
    assert builder.diagnostics.unmapped_action_system_names == {name: 1}


@pytest.mark.parametrize("count", [1, 3])
def test_one_action_creates_one_interaction_per_product(action, count):
    products = tuple(ProductKey("offline1C", f"000000-{index}-long") for index in range(count))
    builder = InteractionBuilder()
    records = builder.from_action(replace(action, products=products))
    assert len(records) == count
    assert tuple(record.product for record in records) == products
    assert builder.diagnostics.actions_view == 1
    assert builder.diagnostics.view_interactions == count


def test_duplicate_products_and_repeated_actions_preserve_every_occurrence(action):
    action = replace(action, products=(action.products[0], action.products[0]))
    builder = InteractionBuilder()
    records = list(builder.iter_interactions([action, action]))
    assert len(records) == 4
    assert all(record.source_event_id == action.action_id for record in records)
    assert builder.diagnostics.actions_total == 2
    assert builder.diagnostics.actions_view == 2
    assert builder.diagnostics.view_interactions == 4


@pytest.mark.parametrize("name", ["ProsmotrProdukta", "ProsmotrProduktaVApiMethod"])
def test_view_without_products_is_diagnostic_error_even_with_category(action, name):
    action = replace(action, action_system_name=name, products=(), categories=(ProductKey("offline1C", "category"),))
    builder = InteractionBuilder()
    assert classify_action(action) is InteractionType.VIEW
    with pytest.raises(InteractionBuildError, match="VIEW.*product") as error:
        builder.from_action(action)
    assert "SECRET" not in str(error.value)
    assert builder.diagnostics.actions_total == builder.diagnostics.actions_view == 1
    assert builder.diagnostics.actions_malformed == 1
    assert builder.diagnostics.view_interactions == 0
    assert builder.diagnostics.malformed_action_system_names == {name: 1}


def test_malformed_names_aggregate_and_snapshots_are_independent(action):
    from Application.interactions import InteractionDiagnostics

    builder = InteractionBuilder()
    names = ["ProsmotrProdukta", "ProsmotrProdukta", "ProsmotrProduktaVApiMethod",
             "DobavlenieProduktaVSpisokVOperaciiDobavlenie"]
    for name in names:
        with pytest.raises(InteractionBuildError):
            builder.from_action(replace(action, action_system_name=name, products=()))
    snapshot = builder.diagnostics
    assert snapshot.malformed_action_system_names == {
        "ProsmotrProdukta": 2, "ProsmotrProduktaVApiMethod": 1,
        "DobavlenieProduktaVSpisokVOperaciiDobavlenie": 1,
    }
    builder.from_action(action)
    builder.from_action(replace(action, action_system_name="Unmapped", products=()))
    assert builder.diagnostics.malformed_action_system_names == snapshot.malformed_action_system_names
    with pytest.raises(TypeError):
        snapshot.malformed_action_system_names["ProsmotrProdukta"] = 99
    with pytest.raises(InteractionBuildError):
        builder.from_action(replace(action, products=()))
    assert snapshot.malformed_action_system_names["ProsmotrProdukta"] == 2
    assert builder.diagnostics.malformed_action_system_names["ProsmotrProdukta"] == 3
    source = {"ProsmotrProdukta": 1}
    direct = InteractionDiagnostics(malformed_action_system_names=source)
    source.clear()
    assert direct.malformed_action_system_names == {"ProsmotrProdukta": 1}


@pytest.mark.parametrize("name", [
    "DobavlenieProduktaVSpisokVOperaciiDobavlenie",
    "DobavlenieProduktaVSpisokVOperaciiDobavlenieTovara",
])
@pytest.mark.parametrize("count", [1, 3])
def test_confirmed_favorites_create_one_interaction_per_product(action, name, count):
    products = tuple(ProductKey("offline1C", f"000000-{index}-long") for index in range(count))
    action = replace(action, action_system_name=name, products=products)
    builder = InteractionBuilder()
    assert classify_action(action) is InteractionType.FAVORITE
    records = builder.from_action(action)
    assert len(records) == count
    assert tuple(record.product for record in records) == products
    assert all(record.interaction_type is InteractionType.FAVORITE for record in records)
    assert all(record.source is InteractionSource.ACTION for record in records)
    assert all(record.source_event_id == action.action_id for record in records)
    assert all(record.quantity is None for record in records)
    assert builder.diagnostics.actions_favorite == 1
    assert builder.diagnostics.favorite_interactions == count
    assert builder.diagnostics.actions_unmapped == 0


def test_default_mappings_are_confirmed_and_disjoint():
    rules = InteractionRules()
    assert rules.favorite_action_system_names == frozenset({
        "DobavlenieProduktaVSpisokVOperaciiDobavlenie",
        "DobavlenieProduktaVSpisokVOperaciiDobavlenieTovara",
    })
    assert rules.view_action_system_names == frozenset({"ProsmotrProdukta", "ProsmotrProduktaVApiMethod"})
    assert rules.purchase_line_statuses == frozenset({"CP", "delivering", "F"})
    assert rules.view_action_system_names.isdisjoint(rules.favorite_action_system_names)


@pytest.mark.parametrize("name", [
    "DobavlenieProduktaVSpisokVOperaciiDobavlenie",
    "DobavlenieProduktaVSpisokVOperaciiDobavlenieTovara",
])
@pytest.mark.parametrize("variation", ["prefix", "suffix", "lower", "leading_space", "trailing_space"])
def test_similar_favorite_names_are_unmapped(action, name, variation):
    names = {
        "prefix": "Prefix" + name, "suffix": name + "Extra", "lower": name.lower(),
        "leading_space": " " + name, "trailing_space": name + " ",
    }
    action = replace(action, action_system_name=names[variation])
    builder = InteractionBuilder()
    assert classify_action(action) is None
    assert builder.from_action(action) == ()
    assert builder.diagnostics.actions_unmapped == 1


@pytest.mark.parametrize("name", [
    "DobavlenieProduktaVSpisokVOperaciiUstanovka",
    "UdalenieProduktaIzSpiskaVOperaciiUdalenieTovara",
    "UstanovkaSpiskaProduktov", "UstanovkaSpiskaProduktovV",
    "OchistkaSpiskaProduktov", "OchistkaSpiskaProduktovV",
])
@pytest.mark.parametrize("with_products", [False, True])
def test_other_product_list_actions_are_unmapped_by_default(action, name, with_products):
    builder = InteractionBuilder()
    action = replace(action, action_system_name=name, products=action.products if with_products else ())
    assert classify_action(action) is None
    assert builder.from_action(action) == ()
    assert builder.diagnostics.actions_unmapped == 1
    assert builder.diagnostics.view_interactions == 0
    assert builder.diagnostics.favorite_interactions == 0
    assert builder.diagnostics.actions_malformed == 0


def test_custom_favorites_use_explicit_exact_mapping(action):
    rules = InteractionRules(favorite_action_system_names=frozenset({"Synthetic.Favorite"}))
    action = replace(action, action_system_name="Synthetic.Favorite", products=action.products * 3)
    builder = InteractionBuilder(rules)
    assert classify_action(action, rules) is InteractionType.FAVORITE
    records = builder.from_action(action)
    assert len(records) == 3
    assert all(record.interaction_type is InteractionType.FAVORITE for record in records)
    assert all(record.quantity is None for record in records)
    assert builder.from_action(replace(action, action_system_name="Synthetic.FavoriteExtra")) == ()
    assert builder.diagnostics.actions_favorite == 1
    assert builder.diagnostics.favorite_interactions == 3


def test_explicit_favorite_without_products_is_error(action):
    builder = InteractionBuilder(InteractionRules(favorite_action_system_names=frozenset({"Synthetic.Favorite"})))
    with pytest.raises(InteractionBuildError, match="FAVORITE"):
        builder.from_action(replace(action, action_system_name="Synthetic.Favorite", products=()))
    assert builder.diagnostics.actions_malformed == 1


@pytest.mark.parametrize("status", ["CP", "delivering", "F"])
@pytest.mark.parametrize("namespace", ["offline1C", "kanzlerKz"])
def test_purchase_allowed_statuses_and_namespaces(line, status, namespace):
    line = replace(line, line_status=status, product=ProductKey(namespace, "00000123456789"))
    builder = InteractionBuilder()
    result = builder.from_order_line(line)
    assert result.interaction_type is InteractionType.PURCHASE
    assert result.source is InteractionSource.ORDER
    assert result.source_event_id == line.line_id
    assert result.source_event_id != line.order_id
    assert result.product is line.product
    assert result.product.value == "00000123456789"
    assert result.quantity is line.quantity
    assert result.event_datetime_utc is line.order_datetime_utc
    assert builder.diagnostics.order_lines_purchase == builder.diagnostics.purchase_interactions == 1


@pytest.mark.parametrize("status", ["cancelled", "cp", "CP ", " delivering", "F-more", "secret@example.test"])
def test_other_line_statuses_are_filtered_exactly_without_logging_values(line, status):
    builder = InteractionBuilder()
    assert builder.from_order_line(replace(line, line_status=status)) is None
    assert builder.diagnostics.order_lines_total == 1
    assert builder.diagnostics.order_lines_filtered_by_status == 1
    assert builder.diagnostics.purchase_interactions == 0
    assert status not in repr(builder.diagnostics)


@pytest.mark.parametrize("quantity", [Decimal("0"), Decimal("-1.5"), Decimal("3.000")])
def test_quantity_is_preserved_without_business_reinterpretation(line, quantity):
    result = InteractionBuilder().from_order_line(replace(line, quantity=quantity))
    assert result.quantity is quantity


def test_customer_ids_and_product_preserved_without_resolution(action, line, monkeypatch):
    from Application.mindbox.identity import CustomerIdResolver

    def forbidden(*args):
        pytest.fail("Interaction layer must not resolve customer identity again")

    monkeypatch.setattr(CustomerIdResolver, "resolve", forbidden)
    records = list(InteractionBuilder().iter_interactions([action], [line]))
    for record in records:
        assert record.source_customer_id == "SECRET_SOURCE_CUSTOMER"
        assert record.customer_id == "SECRET_CANONICAL_CUSTOMER"
        assert "SECRET" not in repr(record)
        with pytest.raises(FrozenInstanceError):
            record.customer_id = "other"


def test_diagnostics_exact_counts_and_immutable_snapshot(action, line):
    builder = InteractionBuilder(InteractionRules(favorite_action_system_names=frozenset({"Synthetic.Favorite"})))
    initial = builder.diagnostics
    records = list(builder.iter_interactions(
        [replace(action, products=action.products * 2), replace(action, action_system_name="Synthetic.Favorite"),
         replace(action, action_system_name="Synthetic.Unknown"), replace(action, action_system_name="Synthetic.Unknown")],
        [line, replace(line, line_status="F"), replace(line, line_status="filtered")],
    ))
    stats = builder.diagnostics
    assert (stats.actions_total, stats.actions_view, stats.actions_favorite, stats.actions_unmapped) == (4, 1, 1, 2)
    assert (stats.view_interactions, stats.favorite_interactions, stats.purchase_interactions) == (2, 1, 2)
    assert (stats.order_lines_total, stats.order_lines_purchase, stats.order_lines_filtered_by_status) == (3, 2, 1)
    assert stats.total_interactions == len(records) == 5
    assert stats.unmapped_action_system_names == {"Synthetic.Unknown": 2}
    assert initial.actions_total == 0
    assert initial.unmapped_action_system_names == {}
    with pytest.raises(TypeError):
        stats.unmapped_action_system_names["new"] = 10
    with pytest.raises(FrozenInstanceError):
        stats.actions_total = 10


def test_rules_defensively_freeze_inputs_and_reject_conflicts():
    favorites = {"Synthetic.Favorite"}
    rules = InteractionRules(favorite_action_system_names=favorites)
    favorites.add("another")
    assert rules.favorite_action_system_names == frozenset({"Synthetic.Favorite"})
    with pytest.raises(FrozenInstanceError):
        rules.favorite_action_system_names = frozenset()
    with pytest.raises(InteractionConfigError):
        InteractionRules(favorite_action_system_names=frozenset({"ProsmotrProdukta"}))


@pytest.mark.parametrize("values", ["bad", None, {None}, {""}, {"   "}, {123}])
def test_invalid_config(values):
    with pytest.raises(InteractionConfigError):
        InteractionRules(favorite_action_system_names=values)


def test_explicit_purchase_status_configuration(line):
    builder = InteractionBuilder(InteractionRules(purchase_line_statuses=frozenset({"Synthetic.Status"})))
    assert builder.from_order_line(line) is None
    assert builder.from_order_line(replace(line, line_status="Synthetic.Status")) is not None


def test_builder_is_lazy_and_empty_inputs_are_valid(action):
    seen = []

    def actions():
        seen.append(1)
        yield action
        seen.append(2)
        yield action

    builder = InteractionBuilder()
    stream = builder.iter_interactions(actions())
    assert seen == []
    next(stream)
    assert seen == [1]
    assert builder.diagnostics.actions_total == 1
    list(stream)
    assert seen == [1, 2]
    assert builder.diagnostics.actions_total == 2
    assert list(InteractionBuilder().iter_interactions()) == []


def test_cli_outputs_only_counts_for_synthetic_records(action, line, monkeypatch, capsys, tmp_path):
    from Application.mindbox import adapters, raw_reader
    from scripts import mindbox_interaction_smoke as cli
    import requests

    def no_http(*args, **kwargs):
        pytest.fail("Smoke must not access the API")

    monkeypatch.setattr(requests.Session, "request", no_http)
    monkeypatch.setattr(adapters, "adapt_action", lambda record, resolver: record)
    monkeypatch.setattr(adapters, "adapt_order", lambda records, resolver: records)
    # Сами тестовые источники содержат только typed records, не production fixtures.
    data = {"customer_merges": [], "actions": [action, replace(action, action_system_name="secret@example.test")],
            "orders": [(line, replace(line, line_status="+79999999999")), ()]}
    monkeypatch.setattr(raw_reader, "iter_export", lambda name, **kwargs: iter(data[name]))
    assert cli.main(["all", "--raw-root", str(tmp_path)]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    for text in ("Actions total: 2", "View actions: 1", "View interactions: 1", "Favorite interactions: 0",
                 "Unmapped actions: 1", "Orders: 2", "Order lines: 2", "Purchase lines: 1",
                 "Filtered order lines: 1", "TOTAL: 2"):
        assert text in output.out
    for secret in ("SECRET", "secret@example.test", "+79999999999", "0000123456789"):
        assert secret not in output.out
    assert list(tmp_path.iterdir()) == []


def test_cli_malformed_view_prints_safe_error_without_partial_counts(action, monkeypatch, capsys, tmp_path):
    from Application.mindbox import adapters, raw_reader
    from scripts import mindbox_interaction_smoke as cli

    monkeypatch.setattr(adapters, "adapt_action", lambda record, resolver: record)
    data = {"customer_merges": [], "actions": [action, replace(action, products=())]}
    monkeypatch.setattr(raw_reader, "iter_export", lambda name, **kwargs: iter(data[name]))
    assert cli.main(["all", "--raw-root", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "VIEW" in output.err
    assert "#2" in output.err
    assert "SECRET" not in output.err


def test_diagnostic_cli_reports_malformed_counts_continues_orders_and_returns_failure(
    action, line, monkeypatch, capsys, tmp_path,
):
    from Application.mindbox import adapters, raw_reader
    from scripts import mindbox_interaction_smoke as cli

    monkeypatch.setattr(adapters, "adapt_action", lambda record, resolver: record)
    monkeypatch.setattr(adapters, "adapt_order", lambda records, resolver: records)
    data = {"customer_merges": [], "actions": [replace(action, products=()), action], "orders": [(line,)]}
    monkeypatch.setattr(raw_reader, "iter_export", lambda name, **kwargs: iter(data[name]))
    assert cli.main(["all", "--raw-root", str(tmp_path), "--diagnose"]) == 1
    output = capsys.readouterr()
    assert "Actions total: 2" in output.out
    assert "View actions: 2" in output.out
    assert "View interactions: 1" in output.out
    assert "Malformed actions: 1" in output.out
    assert "Purchase lines: 1" in output.out
    assert "TOTAL: 2" in output.out
    assert "malformed actions" in output.err
    assert "SECRET" not in output.out + output.err
    assert list(tmp_path.iterdir()) == []
