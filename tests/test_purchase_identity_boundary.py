"""Synthetic regression for printable size suffixes outside a resolver scope."""

from contextlib import closing
from dataclasses import replace
import logging
import sqlite3
from types import SimpleNamespace

import pytest

from Application import customer_history_index as index
from Application.customer_card_data import ViewState
from Application.purchased_products import PurchaseEvidence, assess_order
from tests.test_hybrid_recommendations import (
    catalog as catalog, line, resolver as resolver, sqlite_history as sqlite_history,
)
from tests.test_recommendation_explanations import card_service as card_service


@pytest.mark.parametrize("namespace", ["offline1C", "kanzlerKz"])
def test_printable_suffix_spaces_do_not_change_a_complete_legacy_identity(resolver, namespace):
    # Membership must not change how the SAME valid identity is normalized.
    known = assess_order([[line(code="000001-size 40 42", namespace=namespace)]], resolver)
    outside = assess_order([[line(code="999999-size 40 42", namespace=namespace)]], resolver)
    assert known[0] == {"000001"} and not known[-1]
    assert outside[0] == {"999999"} and not outside[-1]


def test_sqlite_current_ready_historical_ready_with_disjoint_printable_suffix(sqlite_history, catalog):
    root, purchases, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='000002-size 40 42' WHERE variant_key='variant-64'")
    checked = purchases.get("canonical", catalog, raw_root=root, historical_codes={"000001", "888888"})
    assert checked.status == checked.historical.status == "READY"
    assert checked.purchased == {"000001", "000002"}
    assert checked.historical.purchased == {"000001"}
    assert checked.historical.checked_codes == {"000001", "888888"}
    assert checked.historical.diagnostics["unsafe_unresolved_identifier_lines"] == 0
    assert checked.historical.diagnostics["historical_out_of_scope_lines"] == 1
    assert checked.generation == checked.historical.generation
    assert checked.source_version == checked.historical.source_version
    assert checked.historical.diagnostics["normalized_suffix_space_lines"] == 1


@pytest.mark.parametrize("status,expected_bought,expected_uncertain", [
    ("F", {"000002"}, set()), ("Return", set(), {"000002"}),
    ("otmenen", set(), set()), ("CD", set(), {"000002"}),
])
def test_suffix_does_not_hide_a_checked_purchase_or_uncertainty(sqlite_history, catalog, status, expected_bought, expected_uncertain):
    root, purchases, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='000002-size 40 42' WHERE variant_key='variant-64'")
        db.execute("UPDATE order_lines SET raw_status=? WHERE variant_key='variant-64'", (status,))
    checked = purchases.get("canonical", catalog, raw_root=root, historical_codes={"000002", "888888"})
    assert checked.historical.status == "READY"
    assert checked.historical.purchased == expected_bought
    assert checked.historical.uncertain == expected_uncertain


@pytest.mark.parametrize("namespace,value,reason", [
    (None, None, "missing_reference"), ("website", "999999-size 40", "unsupported_namespace"),
    ("offline1C", "", "invalid_identifier"), ("offline1C", " 999999-size 40", "invalid_identifier"),
    ("offline1C", "999999-size 40 ", "invalid_identifier"), ("offline1C", "short", "incomplete_identifier"),
    ("offline1C", "999 99-size 40", "damaged_identity_prefix"),
    ("offline1C", "999999-size\x00 40", "nonprintable_identifier"),
    ("offline1C", "999999-size\t40", "nonprintable_identifier"),
])
def test_printable_suffix_rule_does_not_release_unbounded_uncertainty(sqlite_history, catalog, namespace, value, reason):
    root, purchases, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_namespace=?,selected_value=? WHERE variant_key='variant-64'", (namespace, value))
    checked = purchases.get("canonical", catalog, raw_root=root, historical_codes={"888888"})
    assert checked.status == checked.historical.status == "UNRESOLVED_PRODUCTS"
    assert checked.historical.diagnostics["unsafe_unresolved_identifier_lines"] == 1
    assert checked.historical.diagnostics["unresolved_identifier_reasons"] == {reason: 1}


def test_suffix_space_does_not_release_a_possible_model_alias(sqlite_history, catalog):
    root, purchases, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='999999-size 40' WHERE variant_key='variant-64'")
    checked = purchases.get("canonical", catalog, raw_root=root, historical_codes={"999999-model-alias", "888888"})
    assert checked.status == "READY" and checked.historical.status == "UNRESOLVED_PRODUCTS"
    assert checked.historical.uncertain == {"999999-model-alias"}
    assert checked.historical.diagnostics["unresolved_identifier_reasons"] == {"ambiguous_prefix": 1}


@pytest.mark.parametrize("second,code,purchased,uncertain", [
    ("Return", "000001-size 42", {"000001"}, set()),
    ("otmenen", "000001-size 42", set(), {"000001"}),
    ("F", "999999-size 42", set(), {"000001", "999999"}),
])
def test_suffix_identities_still_participate_in_snapshot_consensus(resolver, second, code, purchased, uncertain):
    result = assess_order([[line(code="000001-size 40")], [line(second, code=code)]], resolver)
    assert result[0] == purchased and result[1] == uncertain and not result[-1]


def test_source_change_reason_preserves_fail_closed_behavior(sqlite_history, catalog, monkeypatch):
    root, purchases, _, sources = sqlite_history
    changed = SimpleNamespace(version="changed", components=sources.components)
    versions = iter((sources, changed))
    monkeypatch.setattr(index, "_capture_sources", lambda _: next(versions))
    checked = purchases.get("canonical", catalog, raw_root=root, historical_codes={"888888"})
    assert checked.status == "INDEX_NOT_READY" and checked.historical is None
    assert checked.diagnostics["failure_reason"] == "sources_changed"


def test_purchase_logs_are_aggregated_and_messages_explain_the_scope(card_service, caplog):
    service, _ = card_service
    private = "PRIVATE_PHONE_EMAIL_PRODUCT_ORDER_SENTINEL"
    checked = service.hybrid.purchases.get.return_value
    service.hybrid.purchases.get.return_value = replace(checked, historical=replace(
        checked.historical, status="UNRESOLVED_PRODUCTS", diagnostics={
            "rows_by_status": {private: 1}, "unsafe_unresolved_identifier_lines": 1,
            "unresolved_non_cancelled_products": True, "coverage_complete": True,
            "unresolved_identifier_reasons": {private: 1, "unsupported_namespace": 1}, "failure_reason": private}))
    with caplog.at_level(logging.INFO, logger="Application.hybrid_recommendations"):
        result = service.load("canonical")
    assert result.raw_state is ViewState.UNAVAILABLE and not result.raw_products and not result.products
    assert "исторических товаров BPR-MF" in result.raw_message
    assert "часть товарных идентификаторов" in result.raw_message
    assert "scope=historical status=UNRESOLVED_PRODUCTS" in caplog.text
    assert "historical_codes_complete=True" in caplog.text
    assert "'unsupported_namespace': 1" in caplog.text
    assert private not in caplog.text and "canonical" not in caplog.text
    assert "source-0" not in caplog.text and "model-user" not in caplog.text


def test_current_failure_does_not_erase_verified_historical_recommendations(card_service):
    service, _ = card_service
    checked = service.hybrid.purchases.get.return_value
    service.hybrid.purchases.get.return_value = replace(checked, status="UNRESOLVED_PRODUCTS")
    result = service.load("canonical")
    assert result.raw_state is ViewState.READY and len(result.raw_products) == 10
    assert not result.products and result.state is ViewState.UNAVAILABLE
    assert "актуальных товаров" in result.message and "часть товарных идентификаторов" in result.message


def test_changed_sources_are_explained_without_exposing_identifiers(card_service):
    service, _ = card_service
    service.hybrid.purchases.get.return_value = PurchaseEvidence("INDEX_NOT_READY", diagnostics={"failure_reason": "sources_changed"})
    result = service.load("canonical")
    assert result.raw_state is ViewState.UNAVAILABLE and not result.raw_products
    assert "данные истории изменились во время поиска" in result.raw_message
