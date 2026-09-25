"""Synthetic canonical-only statistics; never use the developer's dataset."""

from copy import deepcopy
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


def action(name, products=(), customer="old"):
    return {"ids": {"mindboxId": "event"}, "customer": {"ids": {"mindboxId": customer}},
            "actionTemplate": {"ids": {"systemName": name}}, "dateTimeUtc": STAMP, "creationDateTimeUtc": STAMP,
            "products": [{"ids": {"offline1C": product}} for product in products]}


def order(statuses, key="order"):
    return {"ids": {"mindboxId": key}, "customer": {"ids": {"mindboxId": "new"}},
            "firstAction": {"dateTimeUtc": STAMP, "channel": {"ids": {"externalId": "shop"}, "name": "Shop"}},
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
    catalog = tmp_path / "nomenclature.csv"
    catalog.write_text("КодНоменклатуры|Номенклатура|НазваниеНаСайте\n000001|Рубашка|Рубашка на сайте\n000002|Брюки|\n",
                       encoding="utf-8-sig")
    connection = sqlite3.connect(tmp_path / "canonical/customers.sqlite")
    connection.executescript("CREATE TABLE profiles(id TEXT PRIMARY KEY); CREATE TABLE metadata(key TEXT, value TEXT);")
    connection.executemany("INSERT INTO profiles VALUES (?)", [("new",), ("other",), ("inactive",)])
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

    original = statistics._catalog_names

    def replace_catalog(path):
        names = original(path)
        path.write_text("КодНоменклатуры|Номенклатура\n000099|Replacement\n", encoding="utf-8")
        return names

    monkeypatch.setattr(statistics, "_catalog_names", replace_catalog)
    with pytest.raises(CatalogError, match="обновилась"):
        calculate(dataset)
