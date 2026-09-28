"""Synthetic canonical-only statistics; never use the developer's dataset."""

from copy import deepcopy
from contextlib import closing
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import sqlite3

import pytest

from Application.dataset_statistics import _Interactions, calculate_statistics
from Application.interactions import InteractionRecord, InteractionSource, InteractionType
from Application.mindbox.canonical_storage import storage_lock
from Application.mindbox.raw_reader import EXPORT_ROOTS
from Application.mindbox.records import ProductKey
from Application.mindbox.selection import DEFAULT_SELECTION
from Application.product_resolution import ProductResolver, load_catalog
from scripts.dataset_statistics import main as cli_main


STAMP = "2026-01-01T12:00:00+00:00"
SINCE = "2026-01-01T00:00:00+00:00"
UNTIL = "2026-01-02T00:00:00+00:00"


def action(name, products=(), customer="old", *, stamp=STAMP, channel=None, price=None, available=None, namespace="offline1C"):
    return {"ids": {"mindboxId": "event"}, "customer": {"ids": {"mindboxId": customer}},
            "actionTemplate": {"ids": {"systemName": name}}, "dateTimeUtc": stamp, "creationDateTimeUtc": STAMP,
            "channel": channel, "productView": {"price": price, "isAvailable": available},
            "products": [{"ids": {namespace: product}} for product in products]}


def order(statuses, key="order"):
    return {"ids": {"mindboxId": key}, "customer": {"ids": {"mindboxId": "new"}},
            "firstAction": {"dateTimeUtc": STAMP, "channel": {"ids": {"externalId": "shop"}, "name": "Shop"}},
            "customFields": {"orderingMethod": "online", "deliveryType": "courier"},
            "deliveryCost": 0, "payments": [{"type": "card"}], "totalPrice": 999999,
            "lines": [{"id": str(index), "number": index, "quantity": 2, "basePricePerItem": 10,
                       "priceOfLine": 20, "status": {"ids": {"externalId": status}},
                       "product": {"ids": {"offline1C": "000001-size"}}}
                      for index, status in enumerate(statuses)]}


def put_entry(root, name, records, token, since=SINCE, until=UNTIL, kind="API"):
    directory = root / "canonical/objects" / f"{token:032x}" / name
    directory.mkdir(parents=True)
    (directory / f"{name}_part_001.json").write_text(json.dumps({EXPORT_ROOTS[name]: records}), encoding="utf-8")
    return {"directory": directory.relative_to(root).as_posix(), "since": since, "until": until,
            "updated": STAMP, "source_kind": kind, "export_id": None, "operation": "MANUAL", "parts": 1}


def save(root, data):
    (root / "canonical/catalog.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    views, favorites = DEFAULT_SELECTION.view_action_system_names, DEFAULT_SELECTION.favorite_action_system_names
    actions = [action(views[0], ["000001-a", "000002-a"]), action(views[1], ["000001-b"]),
               action(views[2], ["999999-a"]), action(favorites[0], ["000001-a"]),
               action(favorites[1], ["000002-a"], "other"), action(views[0]), action(views[0].lower())]
    orders = [order([*DEFAULT_SELECTION.purchase_line_statuses, "cancelled"])]
    merge = {"id": 1, "dateTimeUtc": STAMP, "resultingCustomer": {"ids": {"mindboxId": "new"}},
             "mergedCustomers": [{"ids": {"mindboxId": "old"}}]}
    data = {"schema_version": 2, "revision": "synthetic", "selection": asdict(DEFAULT_SELECTION),
            "actions": {SINCE[:10]: put_entry(tmp_path, "actions", actions, 1)},
            "orders": {SINCE[:10]: put_entry(tmp_path, "orders", orders, 2)},
            "customer_merges": put_entry(tmp_path, "customer_merges", [merge], 3), "manual_interactions": None}
    save(tmp_path, data)
    (tmp_path / "site_categories.csv").write_text("КодКатегории|НазваниеКатегории\n", encoding="utf-8-sig")
    catalog = tmp_path / "nomenclature.csv"
    catalog.write_text("КодНоменклатуры|Номенклатура|НазваниеНаСайте\n000001|Рубашка|Рубашка на сайте\n000002|Брюки|\n",
                       encoding="utf-8-sig")
    connection = sqlite3.connect(tmp_path / "canonical/customers.sqlite")
    connection.executescript("CREATE TABLE profiles(id TEXT PRIMARY KEY, changed TEXT, raw TEXT NOT NULL); "
                             "CREATE TABLE metadata(key TEXT, value TEXT);")
    connection.executemany("INSERT INTO profiles VALUES (?, '', ?)",
                           [(key, json.dumps({"ids": {"mindboxId": key}})) for key in ("new", "other", "inactive")])
    connection.execute("INSERT INTO metadata VALUES ('summary', ?)",
                       (json.dumps({"count": 3, "source_kind": "MANUAL", "updated": STAMP, "intervals": []}),))
    connection.commit()
    connection.close()
    return tmp_path, catalog, data


def calculate(dataset, **kwargs):
    root, catalog, _ = dataset
    return calculate_statistics(raw_root=root, catalog_path=catalog, **kwargs)


def test_counts_classification_identity_resolution_and_no_legacy(dataset):
    root, _, _ = dataset
    for filename in ("orders.csv", "views.csv", "favorites.csv"):
        (root / filename).write_text("not a CSV; must never be read")
    result = calculate(dataset)
    assert (result.actions, result.orders, result.order_lines, result.customers) == (7, 1, 4, 3)
    assert (result.view_interactions, result.favorite_interactions, result.purchase_interactions) == (4, 2, 3)
    assert (result.actions_with_product, result.actions_without_product) == (5, 2)
    assert (result.action_customers, result.order_customers, result.interaction_users) == (2, 1, 2)
    assert (result.resolved_interactions, result.unresolved_interactions) == (8, 1)
    assert result.resolution_rate == pytest.approx(800 / 9)
    assert result.mean_interactions == result.median_interactions == 4.5
    assert result.purchase_quantity == "6"
    assert result.unique_source_products == 5
    assert result.unique_resolved_items == 2
    diagnostics = dict(result.diagnostics)
    assert diagnostics["mapped_view_actions"] == 4
    assert diagnostics["mapped_favorite_actions"] == 2
    assert diagnostics["mapped_without_product"] == diagnostics["unmapped_actions"] == 1
    assert diagnostics["unknown_candidate"] == diagnostics["filtered_by_status"] == 1
    assert {status for status, _, purchase in result.line_statuses if purchase} == set(DEFAULT_SELECTION.purchase_line_statuses)
    assert sum(count for _, count, _ in result.line_statuses) == 4
    assert result.top_products[0] == ("000001", "Рубашка на сайте", 2, 1, 3, 6)
    assert result.top_products[1][1] == "Брюки"
    assert result.coverage[-1].source_kinds == ("MANUAL",)
    assert result.warnings == ()
    with pytest.raises(FrozenInstanceError):
        result.actions = 0
    json.dumps(asdict(result))  # DTO has no retained per-user/per-event objects.


def test_persisted_selection_is_used(dataset):
    root, _, data = dataset
    data["selection"] = asdict(replace(DEFAULT_SELECTION, view_action_system_names=("custom",),
                                      favorite_action_system_names=("different",), purchase_line_statuses=("cancelled",)))
    save(root, data)
    result = calculate(dataset)
    assert (result.view_interactions, result.favorite_interactions, result.purchase_interactions) == (0, 0, 1)
    assert dict(result.diagnostics)["unmapped_actions"] == 7
    assert {status for status, _, purchase in result.line_statuses if purchase} == {"cancelled"}


def test_resolution_boundary_uses_existing_resolver_for_all_failures(dataset):
    _, catalog, _ = dataset
    collector = _Interactions(ProductResolver(load_catalog(catalog)))
    for key in (ProductKey("offline1C", "000001-a"), ProductKey("kanzlerKz", "999999"),
                ProductKey("website", "000001"), ProductKey("offline1C", "")):
        collector.add(InteractionRecord("u", "u", key, InteractionType.VIEW,
                                        datetime.now(timezone.utc), InteractionSource.ACTION, "e"))
    assert collector.failures == {"unknown_candidate": 1, "unsupported_namespace": 1, "invalid_id": 1}
    assert collector.resolver.diagnostics.total.resolution_rate_percent == 25
    assert collector.resolver.diagnostics.by_namespace["unsupported"].unresolved == 1


def test_all_coverage_gaps_unpaired_days_and_manual_precedence(dataset):
    root, _, data = dataset
    # Manual interval supersedes the API pair, even with a gap to later data.
    manual = {name: put_entry(root, name, [], 4, kind="MANUAL") for name in ("actions", "orders")}
    data["manual_interactions"] = {"since": SINCE, "until": UNTIL, "updated": STAMP, **manual}
    data["actions"]["2026-01-05"] = put_entry(root, "actions", [action(DEFAULT_SELECTION.view_action_system_names[0])],
                                               5, "2026-01-05T00:00:00+00:00", "2026-01-06T00:00:00+00:00")
    save(root, data)
    result = calculate(dataset)
    assert result.actions == 1 and result.orders == 0
    assert len(result.coverage[0].intervals) == 2
    assert result.coverage[0].source_kinds == ("API", "MANUAL")
    assert "CustomerMerges" in result.warnings[0]


def test_duplicate_snapshots_reuse_domain_dedup(dataset):
    root, _, data = dataset
    original = order(["CP"])
    conflict = deepcopy(original)
    conflict["lines"][0]["quantity"] = 9
    directory = root / data["orders"][SINCE[:10]]["directory"]
    (directory / "orders_part_001.json").write_text(json.dumps({"orders": [original, original, conflict]}))
    result = calculate(dataset)
    assert result.orders == result.order_lines == 3
    assert result.purchase_interactions == 1
    assert Decimal(result.purchase_quantity) == 2
    diagnostics = dict(result.diagnostics)
    assert diagnostics["orders_duplicate_identical"] == diagnostics["orders_duplicate_conflicting"] == 1
    assert "конфликтующие" in result.warnings[0]


def test_read_only_cancel_lock_and_progress(dataset):
    root, _, _ = dataset
    before = {p: p.read_bytes() for p in (root / "canonical").rglob("*") if p.is_file()}
    progress = []
    calculate(dataset, progress=progress.append)
    assert any("actions:" in message for message in progress)
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(InterruptedError):
        calculate(dataset, cancelled=lambda: True)
    with storage_lock(root), pytest.raises(Exception, match="Another writer"):
        calculate(dataset)
    calculate(dataset)  # Failure/cancellation releases the lock.


def test_missing_customers_is_unknown_and_damaged_parts_do_not_fallback(dataset):
    root, _, data = dataset
    (root / "canonical/customers.sqlite").unlink()
    assert calculate(dataset).customers is None
    data["actions"][SINCE[:10]]["parts"] = 2
    save(root, data)
    with pytest.raises(ValueError, match="parts mismatch"):
        calculate(dataset)


def test_empty_sources_and_missing_canonical(dataset, tmp_path):
    root, catalog, data = dataset
    data["actions"] = data["orders"] = {}
    save(root, data)
    result = calculate(dataset)
    assert result.actions == result.orders == result.resolution_rate == result.median_interactions == 0
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="canonical dataset"):
        calculate_statistics(raw_root=missing, catalog_path=catalog)
    assert not missing.exists()


def test_cli_result_and_structured_failure(dataset, capsys):
    root, catalog, _ = dataset
    assert cli_main(["--raw-root", str(root), "--catalog", str(catalog)]) == 0
    messages = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert messages[-1]["event"] == "result"
    assert messages[-1]["value"]["actions"] == 7
    assert cli_main(["--raw-root", str(root / "missing")]) == 1
    assert json.loads(capsys.readouterr().out)["event"] == "error"


def test_unmapped_event_does_not_require_full_action_payload(dataset):
    root, _, data = dataset
    directory = root / data["actions"][SINCE[:10]]["directory"]
    (directory / "actions_part_001.json").write_text(json.dumps({"customerActions": [
        {"actionTemplate": {"ids": {"systemName": "Unmapped.Mobile.Event"}}}]}))
    result = calculate(dataset)
    assert result.actions == result.actions_without_product == 1
    assert result.action_customers == 0
    assert dict(result.diagnostics)["actions_without_customer_id"] == 1
    assert any("customer ID" in warning for warning in result.warnings)


@pytest.mark.parametrize("product_ids", [{"website": "000001"}, {"offline1C": ""}])
def test_invalid_canonical_product_fails_without_partial_result(dataset, product_ids):
    from Application.mindbox.adapters import AdapterError

    root, _, data = dataset
    directory = root / data["actions"][SINCE[:10]]["directory"]
    raw = action(DEFAULT_SELECTION.view_action_system_names[0])
    raw["products"] = [{"ids": product_ids}]
    (directory / "actions_part_001.json").write_text(json.dumps({"customerActions": [raw]}))
    with pytest.raises(AdapterError):
        calculate(dataset)


def test_missing_identity_never_uses_legacy_merges(dataset):
    root, _, data = dataset
    data["customer_merges"] = None
    save(root, data)
    with pytest.raises(ValueError, match="CustomerMerges"):
        calculate(dataset)


def test_catalog_replacement_cannot_mix_ids_and_names(dataset, monkeypatch):
    from Application import dataset_statistics as statistics
    from Application.product_resolution import CatalogError

    original = statistics._catalog_metadata

    def replace_catalog(path):
        names = original(path)
        path.write_text("КодНоменклатуры|Номенклатура\n000099|Replacement\n", encoding="utf-8")
        return names

    monkeypatch.setattr(statistics, "_catalog_metadata", replace_catalog)
    with pytest.raises(CatalogError, match="во время чтения"):
        calculate(dataset)


def write_profiles(dataset, profiles):
    with closing(sqlite3.connect(dataset[0] / "canonical/customers.sqlite")) as connection, connection:
        connection.execute("DELETE FROM profiles")
        connection.executemany("INSERT INTO profiles VALUES (?, '', ?)",
                               [(str(i), json.dumps(raw)) for i, raw in enumerate(profiles)])


@pytest.fixture
def fixed_clock(monkeypatch):
    from Application import dataset_statistics as statistics

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 6, 15, 12, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(statistics, "datetime", Clock)


@pytest.mark.parametrize("age,group", [(0, 0), (17, 0), (18, 1), (25, 1), (26, 2), (35, 2),
                                      (36, 3), (45, 3), (46, 4), (55, 4), (56, 5), (65, 5),
                                      (66, 6), (81, 6), (105, 6)])
def test_age_group_boundaries_and_calculation_date(dataset, fixed_clock, age, group):
    write_profiles(dataset, [{"birthDate": f"{2026 - age:04d}-06-15"}])
    result = calculate(dataset)
    assert datetime.fromisoformat(result.calculated_at).astimezone().date().isoformat() == "2026-06-15"
    assert result.mean_age == result.median_age == age
    assert result.age_distribution[group][1:] == (1, 100.)
    assert sum(row[1] for row in result.age_distribution) == 1


@pytest.mark.parametrize("birthday,expected", [("2000-06-14", 26), ("2000-06-15", 26),
                                              ("2000-06-16", 25), ("2000-02-29", 26)])
def test_full_birthday_not_days_divided_by_365(dataset, fixed_clock, birthday, expected):
    write_profiles(dataset, [{"birthDate": birthday}])
    assert calculate(dataset).mean_age == expected


@pytest.mark.parametrize("birthday", [None, "", "bad", "2000-02-30", "2027-01-01", 10,
                                     "2000-01-01T00:00:00", {}, []])
def test_invalid_birthdays_are_visible_unknowns(dataset, fixed_clock, birthday):
    write_profiles(dataset, [{"birthDate": birthday}])
    result = calculate(dataset)
    assert result.mean_age is result.median_age is None
    assert result.age_distribution[-1] == ("Возраст не определён", 1, 100.)


def test_demographics_unique_profiles_rates_means_and_no_private_dto(dataset, fixed_clock):
    write_profiles(dataset, [
        {"birthDate": "2006-06-15", "sex": "male", "firstName": "PRIVATE", "email": "PRIVATE"},
        {"birthDate": "1996-06-15", "sex": "female"},
        {"birthDate": "1986-06-15", "sex": "unknown"},
        {"birthDate": "2006-06-15", "sex": ""}, {}, {"sex": 12}])
    result = calculate(dataset)
    assert result.customers == 6  # Interaction count has no effect on demographics.
    assert result.mean_age == 27.5 and result.median_age == 25
    assert [row[:2] for row in result.gender_distribution] == [("Мужчины", 1), ("Женщины", 1), ("Не указан", 4)]
    assert [row[2] for row in result.gender_distribution] == pytest.approx([100 / 6, 100 / 6, 400 / 6])
    assert result.age_distribution[-1][1:] == pytest.approx((2, 100 / 3))
    assert sum(row[2] for row in result.age_distribution) == pytest.approx(100)
    serialized = json.dumps(asdict(result))
    for private in ("PRIVATE", "2006-06-15", '"customer_id"', "birth_date", "firstName", "email"):
        assert private not in serialized


def test_activity_intersections_unresolved_and_unique_purchase_orders(dataset):
    root, _, data = dataset
    first = order(["CP", "CP"], "one")
    second = order(["CP"], "two")
    second["customer"]["ids"]["mindboxId"] = "old"  # Same canonical buyer.
    conflict = deepcopy(first)
    conflict["lines"][0]["quantity"] = 99
    other = order(["cancelled"], "three")
    (root / data["orders"][SINCE[:10]]["directory"] / "orders_part_001.json").write_text(
        json.dumps({"orders": [first, first, conflict, second, other]}))
    result = calculate(dataset)
    assert result.interaction_users == 2
    assert (result.view_users, result.favorite_users, result.purchase_users) == (1, 2, 1)
    assert (result.view_purchase_users, result.favorite_purchase_users, result.all_interaction_type_users) == (1, 1, 1)
    assert result.repeat_buyers == 1
    assert result.mean_orders_per_buyer == result.median_orders_per_buyer == 2.
    assert result.repeat_buyer_rate == 100. and result.active_buyer_rate == 50.
    assert result.purchase_order_distribution[1][1:] == (1, 100.)
    assert result.interaction_activity_distribution[0][1:] == (1, 50.)
    assert result.interaction_activity_distribution[2][1:] == (1, 50.)  # Includes unresolved VIEW.
    assert result.mean_interactions == result.median_interactions == 4.5


def test_one_order_many_lines_is_not_repeat_purchase(dataset):
    result = calculate(dataset)
    assert result.purchase_interactions == 3
    assert result.purchase_users == 1 and result.repeat_buyers == 0
    assert result.mean_orders_per_buyer == result.median_orders_per_buyer == 1
    assert result.purchase_order_distribution[0][1:] == (1, 100.)


def test_all_activity_and_order_bucket_edges(dataset):
    root, _, data = dataset
    counts = [1, 2, 5, 6, 10, 11, 25, 26, 50, 51, 100, 101]
    orders = []
    for user, count in enumerate(counts):
        for index in range(count):
            raw = order(["CP"], f"{user}-{index}")
            raw["customer"]["ids"]["mindboxId"] = f"buyer-{user}"
            orders.append(raw)
    (root / data["orders"][SINCE[:10]]["directory"] / "orders_part_001.json").write_text(json.dumps({"orders": orders}))
    data["actions"] = {}
    save(root, data)
    result = calculate(dataset)
    assert [row[1] for row in result.interaction_activity_distribution] == [1, 2, 2, 2, 2, 2, 1]
    assert [row[1] for row in result.purchase_order_distribution] == [1, 1, 1, 2, 7]
    assert sum(row[2] for row in result.purchase_order_distribution) == pytest.approx(100)
    assert result.mean_orders_per_buyer == pytest.approx(sum(counts) / len(counts))
    assert result.median_orders_per_buyer == 18
    assert result.repeat_buyers == 11
    assert result.repeat_buyer_rate == pytest.approx(1100 / 12)
    assert result.active_buyer_rate == 100.
    assert result.view_users == result.favorite_users == result.all_interaction_type_users == 0


@pytest.mark.parametrize("missing", [False, True])
def test_empty_audience_and_missing_profiles(dataset, missing):
    root, _, data = dataset
    write_profiles(dataset, [])
    if missing:
        (root / "canonical/customers.sqlite").unlink()
    data["actions"] = data["orders"] = {}
    save(root, data)
    result = calculate(dataset)
    assert result.customers == (None if missing else 0)
    assert result.mean_age is result.median_age is None
    assert result.purchase_users == result.repeat_buyers == result.interaction_users == 0
    assert result.mean_orders_per_buyer == result.median_orders_per_buyer == 0
    assert result.active_buyer_rate == result.repeat_buyer_rate == 0
    for distribution in (result.gender_distribution, result.age_distribution,
                         result.interaction_activity_distribution, result.purchase_order_distribution):
        assert all(count == rate == 0 for _, count, rate in distribution)


def test_single_pass_read_only_projection_and_customers_cancellation(dataset, monkeypatch):
    from Application import dataset_statistics as statistics

    original = statistics.iter_export
    reads = []

    def observed(name, **kwargs):
        reads.append(name)
        yield from original(name, **kwargs)

    monkeypatch.setattr(statistics, "iter_export", observed)
    calculate(dataset)
    assert reads == ["customer_merges", "actions", "orders"]
    progress = []
    with pytest.raises(InterruptedError):
        calculate(dataset, progress=progress.append, cancelled=lambda: any(s.startswith("Клиенты:") for s in progress))
    calculate(dataset)  # Cancellation also closes the SQLite connection and releases the lock.


def test_new_cache_roundtrip_old_cache_ignored_without_overwrite(dataset, tmp_path):
    from Application import statistics_cache as cache

    result = asdict(calculate(dataset))
    path = tmp_path / "statistics.json"
    cache.save_result(path, result)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 8
    assert cache.load_result(path) == json.loads(json.dumps(result))
    path.write_text(json.dumps({"schema_version": 7, "result": result}), encoding="utf-8")
    before = path.read_bytes()
    assert cache.load_result(path) is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("field,bad", [("mean_age", float("nan")), ("median_age", -1), ("view_users", True),
                                      ("repeat_buyers", -1), ("active_buyer_rate", 101),
                                      ("gender_distribution", []), ("purchase_order_distribution", "bad")])
def test_customer_aggregate_cache_validation(dataset, field, bad):
    from Application import statistics_cache as cache

    result = asdict(calculate(dataset))
    result[field] = bad
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def write_orders(dataset, orders):
    root, _, data = dataset
    from Application.mindbox.canonical_customers import encode
    (root / data["orders"][SINCE[:10]]["directory"] / "orders_part_001.json").write_text(
        encode({"orders": orders}), encoding="utf-8")


def financial_order(key, namespace="offline1C", amount="0.1", quantity="1", store="shop", name="Shop"):
    raw = order(["CP", "cancelled"], key)
    raw["firstAction"]["channel"] = {"ids": {"externalId": store}, "name": name}
    raw["lines"][0].update(priceOfLine=Decimal(amount), quantity=Decimal(quantity))
    raw["lines"][0]["product"]["ids"] = {namespace: "000001-size"}
    raw["lines"][1]["priceOfLine"] = 99999
    raw["deliveryCost"] = None
    return raw


def test_order_financials_decimal_currency_delivery_and_raw_statuses(dataset):
    first = financial_order("private-order-1")
    second = financial_order("private-order-2", amount="0.2", quantity="2")
    third = financial_order("private-order-3", "kanzlerKz", "100.05", "3")
    first["deliveryCost"] = Decimal("0.2")
    third["deliveryCost"] = Decimal("2.5")
    conflict = deepcopy(first)
    conflict["lines"][0]["priceOfLine"] = 1000
    write_orders(dataset, [first, second, third, first, conflict])
    result = calculate(dataset)
    assert result.purchase_orders == 3
    assert result.purchase_users == 1 and result.repeat_buyers == 1
    assert result.purchase_interactions == 3 and Decimal(result.purchase_quantity) == Decimal("6")
    assert result.order_financials == (("RUB", 2, 2, "3", "0.3", "0.15", "0.15"),
                                       ("KZT", 1, 1, "3", "100.05", "100.05", "100.05"))
    assert result.delivery_financials == (("RUB", 1, "0.2", "0.2", "0.2"), ("KZT", 1, "2.5", "2.5", "2.5"))
    assert result.mean_purchase_lines_per_order == result.median_purchase_lines_per_order == 1.
    assert Decimal(result.mean_purchase_units_per_order) == Decimal(result.median_purchase_units_per_order) == Decimal("2")
    assert sum(count for _, count, _ in result.line_statuses) == 10
    assert sum(row[1] for row in result.purchase_basket_distribution) == 3
    from Application.statistics_cache import validate_result
    validate_result(asdict(result))
    serialized = json.dumps(asdict(result))
    assert "private-order" not in serialized and '"new"' not in serialized


def test_mixed_currency_retains_basket_but_excludes_money(dataset):
    raw = order(["CP", "F"])
    raw["lines"][1]["product"]["ids"] = {"kanzlerKz": "000001-size"}
    write_orders(dataset, [raw])
    result = calculate(dataset)
    assert result.purchase_orders == result.mixed_currency_purchase_orders == 1
    assert result.unknown_currency_purchase_orders == 0
    assert result.purchase_interactions == 2 and Decimal(result.purchase_quantity) == 4
    assert all(row[1] == 0 for row in result.order_financials)
    assert not result.order_monthly_dynamics and not result.store_statistics
    assert result.purchase_basket_distribution[1][1:] == (1, 100.)
    assert any("RUB/KZT" in warning for warning in result.warnings)
    from Application.statistics_cache import validate_result
    validate_result(asdict(result))


def test_unknown_currency_future_mapping_does_not_lose_order(dataset, monkeypatch):
    from Application.order_statistics import CURRENCY_BY_NAMESPACE, _currency
    assert _currency(["futureNamespace"]) == "unknown"
    # Simulate a supported namespace without a confirmed currency mapping;
    # canonical selection/adapters keep their current validation contract.
    monkeypatch.delitem(CURRENCY_BY_NAMESPACE, "offline1C")
    result = calculate(dataset)
    assert result.purchase_orders == result.unknown_currency_purchase_orders == 1
    assert result.purchase_interactions == 3
    assert all(row[1] == 0 for row in result.order_financials)
    assert any("RUB/KZT" in warning for warning in result.warnings)


def test_basket_histograms_bucket_edges_and_integral_units(dataset):
    counts = [1, 2, 3, 5, 6, 10, 11]
    raws = [order(["CP"] * n, str(n)) for n in counts]
    for raw in raws:
        for line in raw["lines"]:
            line["quantity"] = Decimal("2.000")
    write_orders(dataset, raws)
    result = calculate(dataset)
    assert result.purchase_orders == 7
    assert [r[1] for r in result.purchase_basket_distribution] == [1, 1, 2, 2, 1]
    assert result.mean_purchase_lines_per_order == pytest.approx(38 / 7)
    assert result.median_purchase_lines_per_order == 5
    assert Decimal(result.mean_purchase_units_per_order) == Decimal(76) / 7
    assert Decimal(result.median_purchase_units_per_order) == Decimal("10")


def test_store_keys_names_buyers_missing_and_currency_separation(dataset):
    raws = [financial_order("a", name="Z"), financial_order("b", name="A"),
            financial_order("c", "kanzlerKz", name="A"),
            financial_order("d", store="other", name="A"), financial_order("e", store=None),
            financial_order("f", store="statistics-missing-channel", name="Real ID")]
    raws[1]["customer"]["ids"]["mindboxId"] = "other"
    raws[2]["customer"]["ids"]["mindboxId"] = "old"
    write_orders(dataset, raws)
    result = calculate(dataset)
    rows = {(row[0], row[1]): row for row in result.store_statistics}
    assert rows["RUB", "shop"][2:6] == ("A", 2, 2, 2)
    assert rows["KZT", "shop"][2:6] == ("A", 1, 1, 1)
    assert rows["RUB", None][2] == "Не указано"
    assert rows["RUB", "statistics-missing-channel"][2] == "Real ID"
    assert len(rows) == 5
    # A two-way tie is resolved lexicographically, irrespective of input order.
    write_orders(dataset, raws[:2][::-1])
    assert calculate(dataset).store_statistics[0][2] == "A"


def test_all_stores_deterministic_sort_and_cache_roundtrip(dataset, tmp_path):
    from Application import statistics_cache as cache
    raws = [financial_order(f"{namespace}-{i}", namespace, amount=str(i), store=str(i), name="Shop")
            for namespace in ("offline1C", "kanzlerKz") for i in range(24)]
    write_orders(dataset, raws[::-1])
    result = asdict(calculate(dataset))
    for currency in ("RUB", "KZT"):
        rows = [r for r in result["store_statistics"] if r[0] == currency]
        assert len(rows) == 24
        assert [r[1] for r in rows] == [str(i) for i in range(23, -1, -1)]
    path = tmp_path / "statistics.json"
    cache.save_result(path, result)
    assert cache.load_result(path) == json.loads(json.dumps(result))
    for change in ("order", "duplicate", "currency", "count", "money"):
        broken = json.loads(json.dumps(result))
        rows = broken["store_statistics"]
        if change == "order":
            rows[0], rows[1] = rows[1], rows[0]
        elif change == "duplicate":
            rows.append(rows[0])
        elif change == "currency":
            rows[0][0] = "EUR"
        elif change == "count":
            rows[0][3] = -1
        else:
            rows[0][7] = "NaN"
        with pytest.raises(cache.StatisticsCacheError):
            cache.validate_result(broken)



def test_month_uses_source_utc_and_optional_distributions(dataset):
    root, _, data = dataset
    for i, source in enumerate(("actions", "orders")):
        for j, (start, end) in enumerate((("2025-12-31", "2026-01-01"), ("2026-01-31", "2026-02-01"),
                                         ("2026-02-01", "2026-02-02"))):
            data[source][start] = put_entry(root, source, [], 100 + 10 * i + j,
                                            start + "T00:00:00+00:00", end + "T00:00:00+00:00")
    save(root, data)
    raws = [financial_order("a"), financial_order("b", "kanzlerKz"), financial_order("c")]
    raws[0]["firstAction"]["dateTimeUtc"] = "2026-02-01T00:30:00+03:00"  # January UTC.
    raws[1]["firstAction"]["dateTimeUtc"] = "2025-12-31T23:59:59Z"
    raws[2]["firstAction"]["dateTimeUtc"] = "2026-02-01T00:00:00Z"
    raws[0]["customFields"] = {}
    raws[1]["customFields"] = {"orderingMethod": "", "deliveryType": None}
    raws[0]["payments"] = [{"type": "card"}, {"type": "card"}, {"type": "cash"}, {"type": " "}]
    raws[1]["payments"] = []
    raws[2]["payments"] = [{"type": "card"}, {"type": "cash"}]
    write_orders(dataset, raws)
    result = calculate(dataset)
    assert [row[0] for row in result.order_monthly_dynamics] == ["2025-12", "2026-01", "2026-02"]
    for rows in (result.ordering_method_distribution, result.delivery_type_distribution):
        assert sum(row[1] for row in rows) == 3
        assert next(row[1] for row in rows if row[0] == "Не указано") == 2
    assert dict((name, count) for name, count, _ in result.payment_type_distribution) == {"card": 2, "cash": 2, "Не указано": 1}
    assert sum(row[2] for row in result.payment_type_distribution) == pytest.approx(500 / 3)
    from Application.statistics_cache import validate_result
    validate_result(asdict(result))


@pytest.mark.parametrize("field", ["priceOfLine", "quantity", "deliveryCost"])
@pytest.mark.parametrize("value", [True, "0.1", "bad", -1, float("nan"), float("inf")])
def test_financial_parser_rejects_malformed_values(field, value):
    from Application.order_statistics import _decimal
    from Application.mindbox.adapters import AdapterError
    with pytest.raises(AdapterError):
        _decimal(value, field)


@pytest.mark.parametrize("field", ["priceOfLine", "quantity", "deliveryCost"])
def test_malformed_purchase_money_fails_without_zero_fallback(dataset, field):
    from Application.mindbox.adapters import AdapterError
    raw = order(["CP"])
    (raw if field == "deliveryCost" else raw["lines"][0])[field] = -1
    write_orders(dataset, [raw])
    with pytest.raises(AdapterError):
        calculate(dataset)


@pytest.mark.parametrize("mutation", ["decimal", "nan", "infinity", "negative", "currency", "month", "month_order", "store",
                                      "basket_count", "basket_rate", "ordering", "payment_rate", "missing_field"])
def test_order_cache_rejects_invalid_contract(dataset, mutation):
    from Application import statistics_cache as cache
    result = json.loads(json.dumps(asdict(calculate(dataset))))
    if mutation in ("decimal", "nan", "infinity", "negative"):
        result["order_financials"][0][4] = {"decimal": "bad", "nan": "NaN", "infinity": "Infinity", "negative": "-1"}[mutation]
    elif mutation == "currency":
        result["order_financials"].append(result["order_financials"][0])
    elif mutation == "month":
        result["order_monthly_dynamics"].append(result["order_monthly_dynamics"][0])
    elif mutation == "month_order":
        result["order_monthly_dynamics"] = [["2026-02", 0, "0", 0, "0"], *result["order_monthly_dynamics"]]
    elif mutation == "store":
        result["store_statistics"].append(result["store_statistics"][0])
    elif mutation == "basket_count":
        result["purchase_basket_distribution"][0][1] += 1
    elif mutation == "basket_rate":
        result["purchase_basket_distribution"][0][2] = 10
    elif mutation == "ordering":
        result["ordering_method_distribution"] = []
    elif mutation == "payment_rate":
        result["payment_type_distribution"][0][2] = 3
    else:
        del result["delivery_financials"]
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_cache_decimal_arithmetic_overflow_is_safe(dataset, tmp_path):
    from Application import statistics_cache as cache
    result = json.loads(json.dumps(asdict(calculate(dataset))))
    result["order_financials"][0][4:7] = ["1e999999", "1e999999", "1e999999"]
    result["order_monthly_dynamics"] = [["2026-01", 1, "9e999999", 0, "0"], ["2026-02", 0, "9e999999", 0, "0"]]
    path = tmp_path / "overflow.json"
    path.write_text(json.dumps({"schema_version": 8, "result": result}), encoding="utf-8")
    assert cache.load_result(path) is None


@pytest.mark.parametrize("stamp,included", [
    ("2025-12-31T23:59:59Z", False), (SINCE, True), (STAMP, True), (UNTIL, False),
    ("2026-01-03T00:00:00Z", False), ("2026-01-01T03:00:00+03:00", True),
    ("2026-01-02T03:00:00+03:00", False)])
def test_order_period_filters_every_aggregate(dataset, stamp, included):
    raw = order(["CP"], "period-order")
    raw["firstAction"]["dateTimeUtc"] = stamp
    write_orders(dataset, [raw] if included else [])
    expected = asdict(calculate(dataset))
    write_orders(dataset, [raw])
    actual = asdict(calculate(dataset))
    for key in expected:
        if key not in ("calculated_at", "diagnostics"):
            assert actual[key] == expected[key], key
    assert dict(actual["diagnostics"])["orders_outside_statistics_period"] == (0 if included else 1)


def test_shared_period_gaps_and_second_interval(dataset):
    root, _, data = dataset
    for source in ("actions", "orders"):
        data[source]["2026-01-04"] = put_entry(root, source, [], 10 if source == "actions" else 11,
                                              "2026-01-04T00:00:00+00:00", "2026-01-05T00:00:00+00:00")
    save(root, data)
    gap, second = order(["CP"], "gap"), order(["CP"], "second")
    gap["firstAction"]["dateTimeUtc"] = "2026-01-03T12:00:00Z"
    second["firstAction"]["dateTimeUtc"] = "2026-01-04T12:00:00Z"
    write_orders(dataset, [gap, second])
    result = calculate(dataset)
    assert result.orders == result.purchase_orders == result.purchase_interactions == 1
    assert dict(result.diagnostics)["orders_outside_statistics_period"] == 1
    assert len(result.coverage[0].intervals) == 2


def test_order_coverage_without_shared_action_coverage_is_excluded(dataset):
    root, _, data = dataset
    data["actions"] = {"2026-01-04": put_entry(root, "actions", [], 25,
                                               "2026-01-04T00:00:00+00:00", "2026-01-05T00:00:00+00:00")}
    save(root, data)
    result = calculate(dataset)
    assert result.orders == result.order_lines == result.purchase_orders == result.purchase_users == 0
    assert dict(result.diagnostics)["orders_outside_statistics_period"] == 1
    assert result.coverage[1].intervals == ((SINCE, UNTIL),)


@pytest.mark.parametrize("value", [1, 1.0, Decimal("2.000"), Decimal("3.0"), Decimal("1.5"), Decimal("0.1")])
def test_quantity_eligibility_is_shared_across_all_metrics(dataset, value):
    raw = order(["CP"])
    raw["lines"][0]["quantity"] = value
    integral = Decimal(str(value)) == Decimal(str(value)).to_integral_value()
    expected_order = deepcopy(raw)
    if not integral:
        expected_order["lines"] = []
    write_orders(dataset, [expected_order])
    expected = asdict(calculate(dataset))
    write_orders(dataset, [raw])
    actual = asdict(calculate(dataset))
    for key in expected:
        if key not in ("calculated_at", "diagnostics", "warnings"):
            assert actual[key] == expected[key], key
    assert actual["orders"] == 1
    assert dict(actual["diagnostics"])["fractional_quantity_order_lines"] == (0 if integral else 1)
    assert any("нецелым" in warning for warning in actual["warnings"]) == (not integral)


def test_fractional_lines_do_not_affect_currency_basket_or_dedup(dataset):
    raw = order(["CP", "CP", "cancelled"])
    raw["lines"][1]["quantity"] = Decimal("1.5")
    raw["lines"][1]["product"]["ids"] = {"kanzlerKz": "000001-size"}
    raw["lines"][2]["quantity"] = Decimal("0.1")
    conflicting = deepcopy(raw)
    conflicting["lines"][1]["quantity"] = Decimal("1.7")
    write_orders(dataset, [raw, raw, conflicting])
    result = calculate(dataset)
    assert result.orders == result.order_lines == 3
    assert result.purchase_orders == result.purchase_interactions == 1
    assert Decimal(result.purchase_quantity) == 2
    assert result.order_financials[0][1:3] == (1, 1)
    assert result.order_financials[1][1] == result.mixed_currency_purchase_orders == 0
    diagnostics = dict(result.diagnostics)
    assert diagnostics["fractional_quantity_order_lines"] == 6
    assert diagnostics["orders_duplicate_identical"] == diagnostics["orders_duplicate_conflicting"] == 1


@pytest.mark.parametrize("value", [None, True, "1.5", "bad"])
def test_malformed_quantity_is_not_an_excluded_fraction(dataset, value):
    from Application.mindbox.adapters import AdapterError
    raw = order(["cancelled"])
    raw["lines"][0]["quantity"] = value
    write_orders(dataset, [raw])
    with pytest.raises(AdapterError):
        calculate(dataset)


@pytest.mark.parametrize("value", [None, "bad", "2026-02-30T00:00:00Z"])
def test_invalid_order_date_fails_before_filter(dataset, value):
    from Application.mindbox.adapters import AdapterError
    raw = order(["CP"])
    raw["firstAction"]["dateTimeUtc"] = value
    write_orders(dataset, [raw])
    with pytest.raises(AdapterError):
        calculate(dataset)


@pytest.mark.parametrize("key", ["orders_outside_statistics_period", "fractional_quantity_order_lines"])
@pytest.mark.parametrize("bad", [-1, True, "1"])
def test_eligibility_diagnostics_strict_validation(dataset, key, bad):
    from Application.statistics_cache import StatisticsCacheError, validate_result
    result = asdict(calculate(dataset))
    result["diagnostics"] = tuple((name, bad if name == key else value) for name, value in result["diagnostics"])
    with pytest.raises(StatisticsCacheError):
        validate_result(result)



def write_actions(dataset, records):
    from Application.mindbox.canonical_customers import encode
    root, _, data = dataset
    (root / data["actions"][SINCE[:10]]["directory"] / "actions_part_001.json").write_text(
        encode({"customerActions": records}), encoding="utf-8")


def test_action_activity_bucket_edges_and_means(dataset):
    from statistics import median
    view, favorite = DEFAULT_SELECTION.view_action_system_names[0], DEFAULT_SELECTION.favorite_action_system_names[0]
    views = (1, 2, 5, 6, 10, 11, 25, 26, 50, 51, 100, 101)
    favorites = (1, 2, 3, 5, 6, 10, 11)
    write_actions(dataset, [action(kind, ["000001-a"], f"private-{kind}-{i}")
                            for kind, counts in ((view, views), (favorite, favorites))
                            for i, n in enumerate(counts) for _ in range(n)])
    result = calculate(dataset)
    assert (result.view_interactions, result.favorite_interactions) == (sum(views), sum(favorites))
    assert (result.view_users, result.favorite_users) == (12, 7)
    assert [n for _, n, _ in result.view_user_activity_distribution] == [1, 2, 2, 2, 2, 2, 1]
    assert [n for _, n, _ in result.favorite_user_activity_distribution] == [1, 1, 2, 2, 1]
    assert result.mean_views_per_viewer == pytest.approx(sum(views) / len(views))
    assert result.mean_favorites_per_user == pytest.approx(sum(favorites) / len(favorites))
    assert result.median_views_per_viewer == median(views)
    assert result.median_favorites_per_user == median(favorites)
    for rows, users in ((result.view_user_activity_distribution, 12), (result.favorite_user_activity_distribution, 7)):
        assert all(rate == pytest.approx(100 * n / users) for _, n, rate in rows)


def test_action_multi_product_unresolved_malformed_and_raw_semantics(dataset):
    result = calculate(dataset)
    assert result.view_interactions == 4  # Includes unresolved product and two-product action.
    assert result.mean_views_per_viewer == result.median_views_per_viewer == 4
    assert result.view_users == 1  # old -> new canonical identity.
    assert result.favorite_users == 2
    assert result.action_channel_statistics == ((None, "Не указано", 4, 1, 100., 2, 2, 100.),)
    assert result.action_monthly_dynamics == (("2026-01", 4, 1, 2, 2),)
    assert result.view_parameter_actions == 2
    assert result.view_availability_distribution[-1] == ("Не указано", 2, 100.)
    assert result.view_price_statistics == (("RUB", 0, "0", "0"), ("KZT", 0, "0", "0"))
    diagnostics = dict(result.diagnostics)
    assert diagnostics["ambiguous_product_view_actions"] == 1
    assert diagnostics["mapped_without_product"] == diagnostics["unknown_candidate"] == 1
    assert diagnostics["unknown_currency_view_price_actions"] == 0
    assert result.actions == 7 and result.actions_with_product == 5
    assert sum(n for _, n in result.action_types) == 7
    payload = json.dumps(asdict(result), ensure_ascii=False)
    assert all(token not in payload for token in ('"old"', '"new"', '"other"', '"event"', '"customer_id"', '"productView"'))


def test_action_channels_names_identity_months_and_no_date_filter(dataset):
    view, favorite = DEFAULT_SELECTION.view_action_system_names[0], DEFAULT_SELECTION.favorite_action_system_names[0]
    web = lambda name: {"ids": {"systemName": "web", "externalId": "ignored"}, "name": name}
    records = [action(view, ["000001-a", "000002-a"], channel=web("Zulu"), stamp="2026-03-01T00:30:00+01:00"),
               action(view, ["000001-a"], "new", channel=web("Alpha")),
               action(favorite, ["000001-a"], channel=web("Alpha")),
               action(view, ["000001-a"], channel={"ids": {"externalId": "external"}}),
               action(view, ["000001-a"], channel={"ids": {"systemName": "tie"}, "name": "B"}),
               action(favorite, ["000001-a"], channel={"ids": {"systemName": "tie"}, "name": "A"}),
               action(view, ["000001-a"]),
               action("technical", ["000001-a"], channel=web("Technical"))]
    write_actions(dataset, records)
    result = calculate(dataset)
    rows = {r[0]: r for r in result.action_channel_statistics}
    assert rows["web"][1:4] == ("Alpha", 3, 1)
    assert rows["web"][5:7] == (1, 1)
    assert rows["tie"][1] == "A"
    assert rows["external"][1] == "external"
    assert rows[None][1] == "Не указано"
    assert result.view_users == 1 and sum(r[3] for r in rows.values()) == 4
    assert result.action_monthly_dynamics == (("2026-01", 4, 1, 2, 1), ("2026-02", 2, 1, 0, 0))
    assert result.action_channel_statistics[0][0] == "web"
    assert sum(r[2] for r in rows.values()) == result.view_interactions == 6
    assert all(r[4] == pytest.approx(100 * r[2] / 6) and r[7] == pytest.approx(100 * r[5] / 2) for r in rows.values())


def test_action_availability_and_decimal_prices(dataset):
    view = DEFAULT_SELECTION.view_action_system_names[0]
    records = [action(view, ["000001-a"], price=Decimal("0.1"), available=True),
               action(view, ["000001-a"], price=Decimal("0.2"), available=False),
               action(view, ["000001-a"], price=None),
               action(view, ["000001"], namespace="kanzlerKz", price=Decimal("120.25"), available=True),
               action(view, ["000001-a", "000002-a"], price=999999, available=False)]
    write_actions(dataset, records)
    result = calculate(dataset)
    assert result.view_interactions == 6 and result.view_parameter_actions == 4
    assert result.view_availability_distribution == (("Доступен", 2, 50.), ("Недоступен", 1, 25.), ("Не указано", 1, 25.))
    assert result.view_price_statistics == (("RUB", 2, "0.15", "0.15"), ("KZT", 1, "120.25", "120.25"))
    assert dict(result.diagnostics)["ambiguous_product_view_actions"] == 1


def test_action_unknown_currency_and_internal_invariant():
    from Application.action_statistics import ActionAggregates
    from Application.interactions import InteractionBuilder
    from Application.mindbox.records import ActionRecord
    record = ActionRecord("private-event", DEFAULT_SELECTION.view_action_system_names[0],
                          datetime.now(timezone.utc), datetime.now(timezone.utc), "private-source", "private-canonical",
                          products=(ProductKey("future", "private-product"),), product_view_price=Decimal("0.123456789123456789"))
    aggregate = ActionAggregates()
    builder = InteractionBuilder()
    aggregate.add(record, builder.from_action(record))
    absent = replace(record, product_view_price=None)
    aggregate.add(absent, builder.from_action(absent))
    assert aggregate.unknown_price_currency == 1
    assert aggregate.result()["view_parameter_actions"] == 2
    assert aggregate.result()["view_price_statistics"] == (("RUB", 0, "0", "0"), ("KZT", 0, "0", "0"))
    with pytest.raises(ValueError):
        aggregate.add(record, ())
    interactions = builder.from_action(record)
    with pytest.raises(ValueError):
        aggregate.add(record, (interactions[0], replace(interactions[0], interaction_type=InteractionType.FAVORITE)))


@pytest.mark.parametrize("mutation", ["mean_nan", "median_inf", "mean_negative", "view_count", "favorite_labels",
                                      "availability", "channel_duplicate", "channel_rate", "channel_users", "channel_count",
                                      "month_duplicate", "month_unsorted", "month_format", "month_total", "month_users",
                                      "price_nan", "price_inf", "price_negative", "price_numeric", "price_duplicate",
                                      "price_empty_nonzero", "diagnostic_missing", "diagnostic_negative"])
def test_action_cache_rejects_malformed_aggregates(dataset, mutation):
    from Application import statistics_cache as cache
    result = json.loads(json.dumps(asdict(calculate(dataset))))
    if mutation == "mean_nan":
        result["mean_views_per_viewer"] = float("nan")
    elif mutation == "median_inf":
        result["median_favorites_per_user"] = float("inf")
    elif mutation == "mean_negative":
        result["mean_favorites_per_user"] = -1
    elif mutation == "view_count":
        result["view_user_activity_distribution"][0][1] += 1
    elif mutation == "favorite_labels":
        result["favorite_user_activity_distribution"][0][0] = "bad"
    elif mutation == "availability":
        result["view_availability_distribution"][2][2] = 50
    elif mutation.startswith("channel_"):
        rows = result["action_channel_statistics"]
        if mutation == "channel_duplicate":
            rows.append(rows[0])
        else:
            rows[0][{"channel_rate": 4, "channel_users": 3, "channel_count": 2}[mutation]] = 999
    elif mutation.startswith("month_"):
        rows = result["action_monthly_dynamics"]
        if mutation == "month_duplicate":
            rows.append(rows[0])
        elif mutation == "month_unsorted":
            rows.insert(0, ["2026-02", 0, 0, 0, 0])
        elif mutation == "month_format":
            rows[0][0] = "2026-13"
        else:
            rows[0][1 if mutation == "month_total" else 2] = 999
    elif mutation.startswith("price_"):
        rows = result["view_price_statistics"]
        if mutation == "price_duplicate":
            rows[1] = rows[0]
        else:
            rows[0][2] = {"price_nan": "NaN", "price_inf": "Infinity", "price_negative": "-1",
                          "price_numeric": 1, "price_empty_nonzero": "1"}[mutation]
    elif mutation == "diagnostic_missing":
        result["diagnostics"].pop(0)
    else:
        result["diagnostics"][0][1] = -1
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)



def test_product_resolution_reuse_and_single_pass(dataset, monkeypatch):
    from Application import dataset_statistics as statistics
    calls, reads = [], []
    original_resolve = ProductResolver.resolve_interaction
    original_read = statistics.iter_export
    def resolve(self, interaction, **kwargs):
        calls.append(interaction)
        return original_resolve(self, interaction, **kwargs)
    def read(name, **kwargs):
        reads.append(name)
        yield from original_read(name, **kwargs)
    monkeypatch.setattr(ProductResolver, "resolve_interaction", resolve)
    monkeypatch.setattr(statistics, "iter_export", read)
    result = calculate(dataset)
    assert reads == ["customer_merges", "actions", "orders"]
    assert len(calls) == result.resolved_interactions + result.unresolved_interactions == 9
    assert (result.resolved_view_interactions, result.resolved_favorite_interactions, result.resolved_purchase_interactions) == (3, 2, 3)
    assert result.top_viewed_products == (("000001", "Рубашка на сайте", 2, 1, 1, 3), ("000002", "Брюки", 1, 1, 1, 0))
    assert result.top_purchased_products == (("000001", "Рубашка на сайте", 3, 1, "6", 2, 1),)
    assert result.unique_resolved_items == result.products_with_views == result.products_with_favorites == 2
    assert result.products_with_purchases == 1
    assert result.product_category_statistics == ((None, "Не указано", 2, 3, 2, 3, "6"),)


def test_product_groups_quantity_and_unresolved_purchase(dataset):
    _, catalog, _ = dataset
    catalog.write_text("КодНоменклатуры|Номенклатура|НазваниеНаСайте|КатегорияНаСайте|ПолНоменклатуры|СезонНоски|СтилеваяГруппа\n"
                       "000001|A| Shirt | Рубашки |Мужской|Всесезон|Деловой\n"
                       "000002|B||Брюки|Мужской| |Casual\n000003||||||\n", encoding="utf-8-sig")
    view, favorite = DEFAULT_SELECTION.view_action_system_names[0], DEFAULT_SELECTION.favorite_action_system_names[0]
    write_actions(dataset, [action(view, ["000001-a", "000001-b"], "old"), action(view, ["000001"], "other", namespace="kanzlerKz"),
                            action(view, ["000002-a"]), action(favorite, ["000001-a"]),
                            action(favorite, ["000002-a", "000002-b"]), action(favorite, ["000002-a"], "other"),
                            action(view, ["999999-unresolved"])])
    first = order(["CP", "CP"], "first")
    first["lines"][0]["quantity"], first["lines"][1]["quantity"] = 2, 3
    second = order(["CP", "CP"], "second")
    for line in second["lines"]:
        line["product"]["ids"] = {"offline1C": "000003-a"}
    third = deepcopy(second)
    third["ids"]["mindboxId"] = "third"
    third["customer"]["ids"]["mindboxId"] = "other"
    third["lines"] = third["lines"][:1]
    unknown = order(["CP"], "unresolved")
    unknown["lines"][0]["product"]["ids"] = {"offline1C": "999999-x"}
    unknown["lines"][0]["quantity"] = 9
    write_orders(dataset, [first, second, third, unknown])
    result = calculate(dataset)
    assert result.top_viewed_products[0] == ("000001", "Shirt", 3, 2, 1, 2)
    assert result.top_favorited_products[0] == ("000002", "B", 3, 2, 1, 0)
    assert result.top_purchased_products[0] == ("000003", "", 3, 2, "6", 0, 0)
    assert result.top_purchased_products[1][2:5] == (2, 1, "5")
    assert result.resolved_purchase_quantity == "11" and result.purchase_quantity == "20"
    assert result.resolved_purchase_interactions == 5 and result.purchase_interactions == 6
    assert result.unresolved_interactions == dict(result.diagnostics)["unknown_candidate"] == 2
    assert result.product_category_statistics == ((None, "Не указано", 1, 0, 0, 3, "6"), ("Рубашки", "Рубашки", 1, 3, 1, 2, "5"), ("Брюки", "Брюки", 1, 1, 3, 0, "0"))
    assert result.product_gender_statistics == (("Не указано", 1, 0, 0, 3, "6"), ("Мужской", 2, 4, 4, 2, "5"))
    assert result.product_season_statistics == (("Не указано", 2, 1, 3, 3, "6"), ("Всесезон", 1, 3, 1, 2, "5"))
    assert result.product_style_statistics == (("Не указано", 1, 0, 0, 3, "6"), ("Деловой", 1, 3, 1, 2, "5"), ("Casual", 1, 1, 3, 0, "0"))
    from Application.statistics_cache import validate_result
    validate_result(asdict(result))
    encoded = json.dumps(asdict(result))
    assert all(value not in encoded for value in ('"old"', '"new"', '"other"', '999999-x', '000001-b'))


def test_product_rank_ties_limit_and_whole_metadata_values():
    from Application.product_statistics import ProductAggregates, ProductMetadata
    from Application.product_resolution import ResolvedInteraction, CatalogError
    metadata = {f"{i:06d}": ProductMetadata(category="A,B;C/D") for i in range(25)}
    aggregate = ProductAggregates(metadata, {})
    def add(code, kind, user, quantity=None):
        record = InteractionRecord("source", user, ProductKey("offline1C", code + "-size"), kind,
                                   datetime.now(timezone.utc), InteractionSource.ACTION, "private-event", quantity)
        aggregate.add(ResolvedInteraction(record, code))
    for code in reversed(metadata):
        for kind in InteractionType:
            add(code, kind, "same-user", Decimal(2) if kind == InteractionType.PURCHASE else None)
    add("000024", InteractionType.VIEW, "another")
    add("000023", InteractionType.FAVORITE, "another")
    add("000022", InteractionType.PURCHASE, "another", Decimal(10))
    result = aggregate.result()
    assert all(len(result[field]) == 20 for field in ("top_viewed_products", "top_favorited_products", "top_purchased_products"))
    assert [r[0] for r in result["top_viewed_products"][:3]] == ["000024", "000022", "000000"]
    assert [r[0] for r in result["top_favorited_products"][:3]] == ["000023", "000022", "000000"]
    assert result["top_purchased_products"][0][0] == "000022"
    assert result["product_category_statistics"] == (("A,B;C/D", "A,B;C/D", 25, 26, 26, 26, "60"),)
    # Equal purchases sort by Decimal quantity, then buyers, then catalog code.
    add("000021", InteractionType.PURCHASE, "same-user", Decimal(8))
    add("000020", InteractionType.PURCHASE, "another", Decimal(8))
    add("000019", InteractionType.PURCHASE, "another", Decimal(7))
    assert [r[0] for r in aggregate.result()["top_purchased_products"][:3]] == ["000022", "000020", "000021"]
    add("000023", InteractionType.VIEW, "same-user")
    add("000024", InteractionType.FAVORITE, "same-user")
    assert aggregate.result()["top_viewed_products"][0][0] == "000024"
    assert aggregate.result()["top_favorited_products"][0][0] == "000023"
    del metadata["000000"]
    with pytest.raises(CatalogError, match="метаданные"):
        add("000000", InteractionType.VIEW, "user")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_catalog_metadata_projection_immutable_and_missing_values(tmp_path, encoding):
    from Application.dataset_statistics import _catalog_snapshot
    path = tmp_path / "catalog.csv"
    path.write_text("КодНоменклатуры|НазваниеНаСайте|Номенклатура|КатегорияНаСайте|Остаток|ТитульнаяФотография\n"
                    "000001|  | Fallback | A,B;C/D |not-a-number|private-photo\n000002|||||\n", encoding=encoding)
    (tmp_path / "site_categories.csv").write_text("КодКатегории|НазваниеКатегории\n", encoding="utf-8-sig")
    catalog, metadata, _ = _catalog_snapshot(path)
    assert catalog.item_ids == metadata.keys()
    assert metadata["000001"].name == "Fallback"
    assert metadata["000001"].category == "A,B;C/D"
    assert metadata["000002"].name == "" and metadata["000002"].season is None
    assert len(asdict(metadata["000001"])) == 5
    with pytest.raises(TypeError):
        metadata["bad"] = metadata["000001"]
    with pytest.raises(FrozenInstanceError):
        metadata["000001"].name = "bad"


@pytest.mark.parametrize("content", ["НазваниеНаСайте|КатегорияНаСайте\nA|B\n", "КодНоменклатуры;Номенклатура\n000001;A\n",
                                     "КодНоменклатуры|Номенклатура\n000001|A|extra\n"])
def test_catalog_metadata_invalid_required_structure(tmp_path, content):
    (tmp_path / "site_categories.csv").write_text("КодКатегории|НазваниеКатегории\n", encoding="utf-8-sig")
    from Application.dataset_statistics import _catalog_snapshot
    from Application.product_resolution import CatalogError
    path = tmp_path / "catalog.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(CatalogError):
        _catalog_snapshot(path)


def test_catalog_missing_metadata_is_error(dataset, monkeypatch):
    from Application import dataset_statistics as statistics
    from Application.product_resolution import CatalogError
    monkeypatch.setattr(statistics, "_catalog_metadata", lambda path: {})
    with pytest.raises(CatalogError, match="Метаданные"):
        calculate(dataset)


@pytest.mark.parametrize("mutation", ["product_count", "type_sum", "quantity_nan", "quantity_inf", "quantity_negative", "quantity_fractional",
                                      "quantity_exceeds", "top_duplicate", "top_order", "top_users", "top_quantity",
                                      "group_duplicate", "group_items", "group_view", "group_favorite", "group_purchase", "group_quantity"])
def test_product_cache_rejects_invalid_aggregates(dataset, mutation):
    from Application import statistics_cache as cache
    result = json.loads(json.dumps(asdict(calculate(dataset))))
    if mutation == "product_count":
        result["products_with_views"] = 999
    elif mutation == "type_sum":
        result["resolved_view_interactions"] += 1
    elif mutation.startswith("quantity_"):
        result["resolved_purchase_quantity"] = {"quantity_nan": "NaN", "quantity_inf": "Infinity", "quantity_negative": "-1",
                                                "quantity_fractional": "1.5", "quantity_exceeds": "999"}[mutation]
    elif mutation.startswith("top_"):
        if mutation == "top_duplicate":
            result["top_viewed_products"][1] = result["top_viewed_products"][0]
        elif mutation == "top_order":
            result["top_viewed_products"].reverse()
        elif mutation == "top_users":
            result["top_viewed_products"][0][3] = 999
        else:
            result["top_purchased_products"][0][4] = "1.1"
    else:
        rows = result["product_category_statistics"]
        if mutation == "group_duplicate":
            rows.append(rows[0])
        else:
            index = {"group_items": 2, "group_view": 3, "group_favorite": 4, "group_purchase": 5, "group_quantity": 6}[mutation]
            rows[0][index] = "999" if index == 6 else 999
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


@pytest.mark.parametrize("purchase_status", ["CP", "F", "delivering"])
def test_orders_without_purchase_uses_eligible_unique_snapshots(dataset, purchase_status):
    from Application.statistics_cache import validate_result
    mixed = order([purchase_status, "Return"], "mixed")
    purchased = order(["CP"], "purchased")
    returned = order(["Return"], "returned")
    conflict = deepcopy(returned)
    conflict["lines"][0]["status"]["ids"]["externalId"] = "CP"
    outside = order(["Return"], "outside")
    outside["firstAction"]["dateTimeUtc"] = "2020-01-01T00:00:00Z"
    write_orders(dataset, [mixed, purchased, returned, mixed, returned, conflict, outside])
    result = calculate(dataset)
    diagnostics = dict(result.diagnostics)
    assert result.purchase_orders == 2
    assert result.orders_without_purchase == 1
    assert result.purchase_orders + result.orders_without_purchase == diagnostics["orders_unique"] == 3
    assert diagnostics["orders_duplicate_identical"] == 2
    assert diagnostics["orders_duplicate_conflicting"] == 1
    assert diagnostics["orders_outside_statistics_period"] == 1
    assert result.orders == 6
    validate_result(asdict(result))


@pytest.mark.parametrize("bad", [-1, True, 1.5, "0", None, 999])
def test_orders_without_purchase_cache_validation(dataset, bad):
    from Application.statistics_cache import validate_result, StatisticsCacheError
    result = asdict(calculate(dataset))
    result["orders_without_purchase"] = bad
    with pytest.raises(StatisticsCacheError):
        validate_result(result)


def test_orders_without_purchase_empty_and_fractional_only(dataset):
    from Application.statistics_cache import validate_result
    raw = order(["CP"], "fractional")
    raw["lines"][0]["quantity"] = Decimal("1.5")
    write_orders(dataset, [raw])
    result = calculate(dataset)
    assert result.purchase_orders == 0 and result.orders_without_purchase == 1
    validate_result(asdict(result))
    write_orders(dataset, [])
    result = calculate(dataset)
    assert result.purchase_orders == result.orders_without_purchase == 0
    validate_result(asdict(result))


def test_category_codes_names_fallbacks_and_no_hierarchy(dataset):
    root, catalog, _ = dataset
    catalog.write_text("КодНоменклатуры|КатегорияНаСайте\n000001| 638 \n000002|\n000003|999\n000004|empty\n000005|0638\n000006|100\n000007|200\n", encoding="utf-8-sig")
    (root / "site_categories.csv").write_text(
        "КодКатегории|НазваниеКатегории|КодРодительскойКатегории\n 638 | Рубашки |parent\nempty| |parent\n"
        "0638|С ведущим нулём|638\n100|Категория|638\n200|Категория|638\n100|Категория|different-parent\n", encoding="utf-8-sig")
    write_actions(dataset, [action(DEFAULT_SELECTION.view_action_system_names[0], [f'{i:06d}-size']) for i in range(1, 8)])
    write_orders(dataset, [])
    result = calculate(dataset)
    rows = {row[0]: row for row in result.product_category_statistics}
    assert {code: row[1] for code, row in rows.items()} == {
        '638': 'Рубашки', None: 'Не указано', '999': '999', 'empty': 'empty',
        '0638': 'С ведущим нулём', '100': 'Категория', '200': 'Категория'}
    assert all(row[2:] == (1, 1, 0, 0, '0') for row in rows.values())
    assert [r[0] for r in result.product_category_statistics] == ['999', 'empty', '100', '200', None, '638', '0638']
    from Application import statistics_cache as cache
    path = root / 'category-cache.json'
    cache.save_result(path, asdict(result))
    assert cache.load_result(path) == json.loads(json.dumps(asdict(result)))
    for mutation in ('duplicate', 'code', 'blank_code', 'blank_name', 'totals', 'order', 'quantity', 'duplicate_none'):
        broken = json.loads(json.dumps(asdict(result)))
        values = broken['product_category_statistics']
        if mutation == 'duplicate':
            values[1][0] = values[0][0]
        elif mutation == 'duplicate_none':
            values[0][0] = None
        elif mutation == 'code':
            values[0][0] = 999
        elif mutation == 'blank_code':
            values[0][0] = ' '
        elif mutation == 'blank_name':
            values[0][1] = ''
        elif mutation == 'totals':
            values[0][3] += 1
        elif mutation == 'quantity':
            values[0][6] = 'NaN'
        else:
            values[2], values[3] = values[3], values[2]
        with pytest.raises(cache.StatisticsCacheError):
            cache.validate_result(broken)


@pytest.mark.parametrize('content', [
    'КодКатегории|НазваниеКатегории\n638|A\n638|B\n',
    'КодКатегории|Другое\n638|A\n',
    'КодКатегории;НазваниеКатегории\n638;A\n',
    'КодКатегории|НазваниеКатегории\n638|A|extra\n',
    'КодКатегории|НазваниеКатегории\n638\n',
    'КодКатегории|НазваниеКатегории|НазваниеКатегории\n638|A|A\n',
])
def test_category_reference_invalid_structure_is_safe(dataset, content):
    from Application.product_resolution import CatalogError
    (dataset[0] / 'site_categories.csv').write_text(content, encoding='utf-8-sig')
    with pytest.raises(CatalogError):
        calculate(dataset)


@pytest.mark.parametrize('content', [None, b'\xff\xff'])
def test_category_reference_read_errors_are_safe(dataset, content):
    from Application.product_resolution import CatalogError
    path = dataset[0] / 'site_categories.csv'
    if content is None:
        path.unlink()
    else:
        path.write_bytes(content)
    with pytest.raises(CatalogError, match='Не удалось прочитать'):
        calculate(dataset)


@pytest.mark.parametrize('changed_file', ['nomenclature.csv', 'site_categories.csv'])
def test_reference_snapshot_covers_both_files(dataset, monkeypatch, changed_file):
    from Application import dataset_statistics as statistics
    from Application.product_resolution import CatalogError
    original = statistics._category_names
    def changed(path):
        result = original(path)
        with (dataset[0] / changed_file).open('a', encoding='utf-8') as stream:
            stream.write('\n')
        return result
    monkeypatch.setattr(statistics, '_category_names', changed)
    with pytest.raises(CatalogError, match='во время чтения'):
        calculate(dataset)
