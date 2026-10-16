"""Filtered historical presentation over full, unchanged BPR and one history read."""

from contextlib import closing, contextmanager
from dataclasses import replace
import sqlite3

import pytest

from Application import customer_history_index as index
from Application.customer_card_data import ViewState
from Application.hybrid_recommendations import compose, select_displayed_raw
from Application.purchased_products import PurchaseEvidence
from Application.recommendation_rules import save_rules
from tests.test_hybrid_recommendations import (
    BASE, catalog as catalog, raw_result, rules, sqlite_history as sqlite_history,
)
from tests.test_recommendation_explanations import card_service as card_service


def evidence(raw, *, purchased=(), uncertain=(), status="READY"):
    return PurchaseEvidence("READY", historical=PurchaseEvidence(
        status, frozenset(purchased), frozenset(uncertain),
        checked_codes=frozenset(r.model_item_code for r in raw.recommendations)))


@pytest.mark.parametrize("excluded", [(), (0, 2), tuple(range(15)), tuple(range(20))])
def test_full_ranking_backfill_fewer_empty_and_original_values(card_service, excluded):
    service, _ = card_service
    raw, _ = service.inference.get_candidate_sources.return_value
    bought = {raw.recommendations[i].model_item_code for i in excluded}
    service.hybrid.purchases.get.return_value = evidence(raw, purchased=bought)
    result = service.load("canonical")
    expected = tuple(rec for rec in raw.recommendations if rec.model_item_code not in bought)[:10]
    assert tuple(p.recommendation for p in result.raw_products) == expected
    assert all(p.recommendation is raw.recommendations[p.recommendation.rank - 1] for p in result.raw_products)
    assert result.raw_state is (ViewState.READY if expected else ViewState.EMPTY)
    assert result.hybrid.raw_bpr is raw and len(raw.recommendations) == 20
    service.inference.get_candidate_sources.assert_called_once()
    service.inference.get_recommendations.assert_not_called()
    service.hybrid.purchases.get.assert_called_once()
    assert service.hybrid.purchases.get.call_args.args[0] == "canonical"
    assert service.hybrid.purchases.get.call_args.kwargs["historical_codes"] == {r.model_item_code for r in raw.recommendations}


def test_selected_metadata_after_top10_and_hybrid_unchanged(card_service):
    service, catalog = card_service
    raw, _ = service.inference.get_candidate_sources.return_value
    metadata = {r.model_item_code: {**BASE, "Номенклатура": f"Исторический {r.rank}"} for r in raw.recommendations}
    metadata["source-15"] = {**BASE, "НазваниеНаСайте": "Исторический 16", "Коллекция": "Осень-Зима 2024"}
    service.inference.get_candidate_sources.return_value = raw, metadata
    save_rules(rules(catalog), service.hybrid.rules_path)
    checked = evidence(raw, purchased=[r.model_item_code for r in raw.recommendations[:15]], uncertain=("source-16",))
    service.hybrid.purchases.get.return_value = checked
    expected_hybrid = compose(raw, metadata, catalog, rules(catalog), checked)
    result = service.load("canonical")
    assert result.hybrid.items == expected_hybrid
    assert result.raw_products[0].recommendation.rank == 16
    assert result.raw_products[0].name == "Исторический 16"
    assert result.raw_products[0].collection == "Осень-Зима 2024"
    assert [r.rank for r in result.hybrid.displayed_raw] == [16, 18, 19, 20]
    # Purchased source 1 remains the best hybrid evidence, not source 16.
    assert any(item.sources[0].source_raw_rank == 1 for item in result.hybrid.items)


@pytest.mark.parametrize("status", ["INDEX_NOT_READY", "INCOMPLETE_COVERAGE", "SOURCE_UNAVAILABLE", "UNRESOLVED_PRODUCTS"])
def test_unavailable_history_never_displays_unverified_raw(card_service, status):
    service, catalog = card_service
    save_rules(rules(catalog), service.hybrid.rules_path)
    service.hybrid.purchases.get.return_value = PurchaseEvidence(status)
    result = service.load("canonical")
    assert result.raw_state is ViewState.UNAVAILABLE and not result.raw_products
    assert "историю покупок" in result.raw_message
    assert result.profile is not None


def test_historical_unknown_scope_does_not_disable_valid_hybrid(card_service):
    service, catalog = card_service
    save_rules(rules(catalog), service.hybrid.rules_path)
    raw, _ = service.inference.get_candidate_sources.return_value
    service.hybrid.purchases.get.return_value = evidence(raw, status="UNRESOLVED_PRODUCTS")
    result = service.load("canonical")
    assert result.state is ViewState.READY and result.products
    assert result.raw_state is ViewState.UNAVAILABLE and not result.raw_products


def test_incomplete_identity_scope_is_not_treated_as_no_purchases(card_service):
    service, _ = card_service
    service.hybrid.purchases.get.return_value = PurchaseEvidence("READY", historical=PurchaseEvidence("READY"))
    result = service.load("canonical")
    assert result.raw_state is ViewState.UNAVAILABLE and not result.raw_products


@pytest.mark.parametrize("rules_ready", [False, True])
def test_history_freshness_at_delivery_even_without_hybrid_rules(card_service, rules_ready):
    service, catalog = card_service
    if rules_ready:
        save_rules(rules(catalog), service.hybrid.rules_path)
    result = service.load("canonical")
    assert result.raw_state is ViewState.READY
    service.hybrid.purchases.is_current.return_value = False
    result = service.validate_result(result)
    assert result.raw_state is ViewState.STALE and not result.raw_products
    assert result.profile is not None


def test_rules_change_does_not_invalidate_checked_historical_presentation(card_service):
    service, catalog = card_service
    save_rules(rules(catalog), service.hybrid.rules_path)
    result = service.load("canonical")
    service.hybrid.rules_path.unlink()
    stale = service.validate_result(result)
    assert stale.state is ViewState.STALE and not stale.products
    assert stale.raw_products == result.raw_products and stale.raw_state is ViewState.READY


@pytest.mark.parametrize("status,excluded", [("F", True), ("CP", True), ("delivering", True),
                                           ("otmenen", False), ("Return", True), ("CD", True)])
def test_historical_code_absent_from_csv_checked_with_existing_consensus(sqlite_history, catalog, monkeypatch, status, excluded):
    root, service, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='999999-size' WHERE variant_key='variant-64'")
        db.execute("UPDATE order_lines SET raw_status=? WHERE variant_key='variant-64'", (status,))
    baseline = service.get("canonical", catalog, raw_root=root)
    reader, reads = index._read_index, []
    @contextmanager
    def tracked(path):
        with reader(path) as db:
            reads.append(db)
            yield db
    monkeypatch.setattr(index, "_read_index", tracked)
    result = service.get("canonical", catalog, raw_root=root, historical_codes={"999999", "888888"})
    assert len(reads) == 1
    assert (result.status, result.purchased, result.uncertain, result.diagnostics) == (
        baseline.status, baseline.purchased, baseline.uncertain, baseline.diagnostics)
    historical = result.historical
    assert historical.status == "READY" and historical.checked_codes == {"999999", "888888"}
    assert ("999999" in historical.purchased | historical.uncertain) is excluded
    assert historical.generation == result.generation and historical.source_version == result.source_version
    assert historical.diagnostics["rows_by_status"] == ({"F": 65} if status == "F" else {"F": 64, status: 1})
    with pytest.raises(sqlite3.ProgrammingError):
        reads[0].execute("SELECT 1")


def test_scope_resolution_is_separate_short_model_id_does_not_weaken_current_guard(sqlite_history, catalog):
    root, service, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='short' WHERE variant_key='variant-64'")
    result = service.get("canonical", catalog, raw_root=root, historical_codes={"short"})
    assert result.status == "UNRESOLVED_PRODUCTS"  # Current catalog guard is unchanged.
    assert result.historical.status == "READY" and result.historical.purchased == {"short"}


def test_unresolvable_model_alias_is_excluded_even_without_orders(sqlite_history, catalog):
    root, service, _, _ = sqlite_history
    result = service.get("empty-customer", catalog, raw_root=root, historical_codes={"999999-alias", "999999"})
    assert result.historical.status == "READY"
    assert result.historical.uncertain == {"999999-alias"}


@pytest.mark.parametrize("kind", ["VIEW", "FAVORITE"])
def test_actions_are_not_purchases(sqlite_history, catalog, kind):
    root, service, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE events SET source='ACTION',kind=? WHERE event_key='event-64'", (kind,))
    checked = service.get("canonical", catalog, raw_root=root, historical_codes={"000002"})
    assert checked.historical.status == "READY"
    assert not checked.historical.purchased and not checked.historical.uncertain


def test_select_rejects_missing_purchase_evidence():
    assert not select_displayed_raw(raw_result(), None)
    assert not select_displayed_raw(raw_result(), PurchaseEvidence("READY"))


def test_cancellation_after_purchase_read_does_not_expose_raw(card_service):
    service, _ = card_service
    stopped = False
    checked = service.hybrid.purchases.get.return_value
    def get(*args, **kwargs):
        nonlocal stopped
        stopped = True
        return checked
    service.hybrid.purchases.get.side_effect = get
    result = service.load("canonical", lambda: stopped)
    assert result.state is ViewState.CANCELLED and not result.raw_products and not result.products


def test_real_sqlite_card_filters_old_code_beyond_first_fifty_and_reads_once(card_service, sqlite_history, monkeypatch):
    service, catalog = card_service
    root, purchases, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='999999-size' WHERE variant_key='variant-64'")
    raw, _ = service.inference.get_candidate_sources.return_value
    raw = replace(raw, recommendations=tuple(replace(rec, model_item_code=("999999" if n == 0 else f"{700000+n}"))
                                             for n, rec in enumerate(raw.recommendations)))
    metadata = {rec.model_item_code: BASE for rec in raw.recommendations}
    service.inference.get_candidate_sources.return_value = raw, metadata
    service.hybrid.purchases = purchases
    service.raw_root = root
    save_rules(rules(catalog), service.hybrid.rules_path)
    reader, reads = index._read_index, []
    @contextmanager
    def tracked(path):
        with reader(path) as db:
            if any(row[0] == "order_lines" for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")):
                # Distinguish order reads from cheap freshness inspections.
                db.set_trace_callback(lambda sql: reads.append(sql) if "FROM order_lines" in sql else None)
            yield db
    monkeypatch.setattr(index, "_read_index", tracked)
    result = service.load("canonical")
    assert result.state is ViewState.READY and result.raw_state is ViewState.READY
    assert [p.recommendation.rank for p in result.raw_products] == list(range(2, 12))
    assert result.hybrid.purchases.purchased == {"000001"}
    assert result.hybrid.purchases.historical.purchased == {"999999"}
    assert len(reads) == 65  # Each of 65 order snapshots is read once, not once per scope.
    assert result.hybrid.items == compose(raw, metadata, catalog, rules(catalog), result.hybrid.purchases)
    service.inference.get_candidate_sources.assert_called_once()
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE metadata SET value=? WHERE key='generation'", ("c" * 32,))
    stale = service.validate_result(result)
    assert stale.raw_state is ViewState.STALE and stale.state is ViewState.STALE
    assert not stale.raw_products and not stale.products and stale.profile == result.profile


def test_real_conflicting_snapshots_exclude_old_and_current_identity(sqlite_history, catalog):
    root, service, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("INSERT INTO variants SELECT 'changed',event_key,?,canonical_customer_id,source_customer_id,kind,event_us,quality FROM variants WHERE variant_key='variant-64'", (bytes([1]) * 32,))
        db.execute("INSERT INTO order_snapshots SELECT 'changed','changed',retail_order_id,order_us,channel_system_name,channel_external_id,channel_mindbox_id,channel_name,total_price,delivery_cost FROM order_snapshots WHERE variant_key='variant-64'")
        db.execute("INSERT INTO order_lines SELECT 'changed-line','changed',source_line_key,line_id,source_line_id,is_synthetic,ordinal,line_number,product_name,quantity,raw_status,price_of_line,base_price_per_item,quality FROM order_lines WHERE variant_key='variant-64'")
        db.execute("INSERT INTO product_occurrences VALUES ('changed','line',0,'offline1C','999999-old','[]')")
    result = service.get("canonical", catalog, raw_root=root, historical_codes={"999999", "000002"})
    assert result.uncertain == {"000002"}
    assert result.historical.status == "READY" and result.historical.uncertain == {"999999", "000002"}
    assert not result.historical.purchased
    assert result.historical.diagnostics["conflicting_orders"] == 1


def test_source_change_at_delivery_clears_historical_cards(card_service, sqlite_history):
    service, _ = card_service
    root, purchases, _, sources = sqlite_history
    service.raw_root, service.hybrid.purchases = root, purchases
    raw, _ = service.inference.get_candidate_sources.return_value
    # Valid normalized historical codes, independent of today's CSV.
    raw = replace(raw, recommendations=tuple(replace(rec, model_item_code=f"{700000+n}")
                                             for n, rec in enumerate(raw.recommendations)))
    service.inference.get_candidate_sources.return_value = raw, {}
    result = service.load("canonical")
    assert result.raw_state is ViewState.READY
    sources.version = "replaced"
    result = service.validate_result(result)
    assert result.raw_state is ViewState.STALE and not result.raw_products
