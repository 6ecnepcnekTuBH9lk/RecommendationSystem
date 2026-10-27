"""Real purchase checks on synthetic SQLite through the existing serial Qt worker."""

from contextlib import closing
from dataclasses import replace
import sqlite3
from threading import get_ident

import pytest

from Application.customer_card_data import ViewState
from Application.customer_contact_search import ContactType
from Application.hybrid_recommendations import HybridRecommendationService
from Application.recommendation_rules import save_rules
from tests.tabs.test_customer_contact_search_ui import app as app, card as card, matched, search
from tests.test_hybrid_recommendations import (
    BASE, catalog as catalog, raw_result, rules, sqlite_history as sqlite_history,
)


@pytest.fixture
def indexed_card(card, sqlite_history, catalog, tmp_path, monkeypatch):
    root, purchases, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='000002-size 40 42' WHERE variant_key='variant-64'")
    original = raw_result(30)
    raw = replace(original, model_user_id="A", recommendations=tuple(
        replace(rec, model_item_code="000001" if n == 0 else f"{800000+n}")
        for n, rec in enumerate(original.recommendations)))
    # FIX-02: the first ten ranks are forbidden; backfill from the full ranking.
    metadata = {rec.model_item_code: {**BASE, "ВидНоменклатуры": "Носки" if rec.rank <= 10 else "Брюки"}
                for rec in raw.recommendations}
    card.service.inference.get_candidate_sources.return_value = (raw, metadata)
    hybrid = HybridRecommendationService(catalog_path=catalog.path, categories_path=catalog.categories_path,
                                         rules_path=tmp_path / "rules.json", purchases=purchases)
    save_rules(rules(catalog), hybrid.rules_path)
    card.service.hybrid = hybrid
    card.service.raw_root = root
    card.service.catalog_path = catalog.path
    card.backend.side_effect = lambda *a, **kw: matched("canonical")
    calls = []
    get = purchases.get
    def tracked(customer, *args, **kwargs):
        calls.append((customer, get_ident()))
        return get(customer, *args, **kwargs)
    monkeypatch.setattr(purchases, "get", tracked)
    return card, path, raw, calls


def test_phone_direct_and_repeat_share_canonical_purchase_evidence(indexed_card):
    card, _, raw, calls = indexed_card
    search(card, None, value="canonical")
    direct = card.tab.result
    assert direct.state is ViewState.READY and direct.raw_state is ViewState.READY
    assert len(card.tab.raw_grid.cards) == 10 and card.tab.product_grid.cards
    assert tuple(p.recommendation for p in direct.raw_products) == raw.recommendations[10:20]
    for kind, value in ((ContactType.PHONE, None), (None, "canonical")):
        search(card, kind, value=value)
        result = card.tab.result
        assert result.raw_products == direct.raw_products and result.products == direct.products
        assert result.hybrid.purchases == direct.hybrid.purchases
        assert result.profile.canonical_customer_id == result.identity.canonical_customer_id == "canonical"
        assert card.tab.history.customer_id == "canonical"
        assert [p.display_rank for p in result.raw_products] == list(range(1, 11))
    assert len(calls) == 3 and all(customer == "canonical" and thread != get_ident() for customer, thread in calls)
    assert "000001" not in {p.recommendation.model_item_code for p in direct.raw_products}
    assert not ({"000001", "000002"} & {p.hybrid_item.target_item_code for p in direct.products})


def test_damaged_identity_blocks_cards_with_specific_reason_and_keeps_profile_history(indexed_card):
    card, path, _, _ = indexed_card
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='999 99-size 40' WHERE variant_key='variant-64'")
    search(card, ContactType.PHONE)
    result = card.tab.result
    assert result.raw_state is ViewState.UNAVAILABLE and result.state is ViewState.UNAVAILABLE
    assert not card.tab.raw_grid.cards and not card.tab.product_grid.cards
    assert not card.tab.calculations.items
    assert "часть товарных идентификаторов" in card.tab.recommendations.message.text()
    assert card.tab.profile.content.state is ViewState.READY
    assert card.tab.history.customer_id == "canonical"
    assert result.hybrid.purchases.historical.status == "UNRESOLVED_PRODUCTS"
