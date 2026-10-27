"""Known historical kinds and request-bound rules, synthetic snapshots only."""

from dataclasses import replace

import pytest

from Application import hybrid_recommendations as backend
from Application.customer_card_data import ViewState
from Application.recommendation_rules import RecommendationRules, save_rules
from tests.test_hybrid_recommendations import (
    BASE, catalog as catalog, checked_purchases, raw_result, rules, write_catalog,
)
from tests.test_recommendation_explanations import card_service as card_service


def test_first_ten_forbidden_sources_backfill_full_raw_top10(card_service):
    service, _ = card_service
    raw = raw_result(30)
    metadata = {rec.model_item_code: {**BASE, "ВидНоменклатуры": "Носки" if rec.rank <= 10 else "Брюки"}
                for rec in raw.recommendations}
    service.inference.get_candidate_sources.return_value = raw, metadata
    service.hybrid.purchases.get.return_value = checked_purchases(raw)
    result = service.load("canonical")
    assert result.raw_state is ViewState.READY
    assert [p.recommendation.rank for p in result.raw_products] == list(range(11, 21))
    assert [p.display_rank for p in result.raw_products] == list(range(1, 11))
    assert all(p.recommendation is raw.recommendations[p.recommendation.rank - 1] for p in result.raw_products)
    assert result.hybrid.raw_bpr is raw
    service.inference.get_candidate_sources.assert_called_once()
    service.inference.get_recommendations.assert_not_called()
    service.hybrid.purchases.get.assert_called_once()


def test_interleaved_kinds_purchases_uncertain_and_missing_metadata(card_service):
    service, _ = card_service
    raw = raw_result(10)
    kinds = ("Носки", "Брюки", "Брюки", "Носки", "Брюки", "Брюки", "Брюки", "Носки", "Брюки", "Брюки")
    metadata = {rec.model_item_code: {**BASE, "ВидНоменклатуры": kind} for rec, kind in zip(raw.recommendations, kinds)}
    del metadata["source-6"]["ВидНоменклатуры"]
    service.inference.get_candidate_sources.return_value = raw, metadata
    checked = checked_purchases(raw)
    service.hybrid.purchases.get.return_value = replace(checked, historical=replace(
        checked.historical, purchased=frozenset(("source-1",)), uncertain=frozenset(("source-4",))))
    result = service.load("canonical")
    assert [p.recommendation.rank for p in result.raw_products] == [3, 6, 9, 10]
    assert [p.display_rank for p in result.raw_products] == [1, 2, 3, 4]
    assert {i.sources[0].source_raw_rank for i in result.hybrid.items} <= {3, 6, 9, 10}


@pytest.mark.parametrize("value", [None, "", "Носки", "брюки", "Брюки ", 123, ["Брюки"]])
def test_unknown_or_nonexact_kind_never_passes_by_assortment_type(card_service, value):
    service, _ = card_service
    raw, metadata = service.inference.get_candidate_sources.return_value
    metadata = {code: {**values, "ВидНоменклатуры": value, "ВидАссортимента": "Брюки"} for code, values in metadata.items()}
    service.inference.get_candidate_sources.return_value = raw, metadata
    result = service.load("canonical")
    assert result.raw_state is ViewState.EMPTY and result.state is ViewState.EMPTY
    assert not result.raw_products and not result.products


def test_missing_snapshot_kind_is_not_recovered_from_current_csv(card_service):
    service, _ = card_service
    catalog = write_catalog(service.catalog_path.parent, [("source-0", {"Коллекция": "NOS"})])
    save_rules(rules(catalog), service.hybrid.rules_path)
    raw, metadata = service.inference.get_candidate_sources.return_value
    metadata = {code: dict(values) for code, values in metadata.items()}
    del metadata["source-0"]["ВидНоменклатуры"]
    service.inference.get_candidate_sources.return_value = raw, metadata
    result = service.load("canonical")
    assert result.raw_products[0].recommendation.rank == 2
    assert "source-0" not in {p.recommendation.model_item_code for p in result.raw_products}


@pytest.mark.parametrize("metadata", [None, {}, []])
def test_unavailable_metadata_cannot_authorize_filtered_raw(card_service, metadata):
    service, _ = card_service
    raw, _ = service.inference.get_candidate_sources.return_value
    service.inference.get_candidate_sources.return_value = raw, metadata
    result = service.load("canonical")
    assert result.raw_state is ViewState.UNAVAILABLE and not result.raw_products
    assert result.hybrid.raw_status == "METADATA_NOT_READY" and "метаданные" in result.raw_message


def test_historical_collection_stock_and_catalog_membership_do_not_gate_raw(card_service):
    service, catalog = card_service
    raw, metadata = service.inference.get_candidate_sources.return_value
    metadata = {code: {**values, "Коллекция": "Весна-Лето 2024", "Остаток": "0"} for code, values in metadata.items()}
    service.inference.get_candidate_sources.return_value = raw, metadata
    save_rules(replace(rules(catalog), minimum_stock=99999), service.hybrid.rules_path)
    result = service.load("canonical")
    assert result.raw_state is ViewState.READY and len(result.raw_products) == 10
    assert result.state is ViewState.EMPTY
    assert all(p.collection == "Весна-Лето 2024" and p.recommendation.model_item_code not in catalog.all_codes
               for p in result.raw_products)
    service.hybrid.catalog.path.unlink()
    available = service.load("canonical")
    assert available.raw_state is ViewState.READY and len(available.raw_products) == 10
    assert available.state is ViewState.UNAVAILABLE


@pytest.mark.parametrize("content", [None, "broken json", '{}', '{"version":99}',
                                    '{"version":1,"kinds":[],"collections":["NOS"],"minimum_stock":0}'])
def test_missing_or_invalid_rules_never_authorize_unfiltered_raw(card_service, content):
    service, _ = card_service
    if content is None:
        service.hybrid.rules_path.unlink()
    else:
        service.hybrid.rules_path.write_text(content, encoding="utf-8")
    result = service.load("canonical")
    assert result.raw_state is ViewState.UNAVAILABLE and result.state is ViewState.UNAVAILABLE
    assert not result.raw_products and not result.products
    assert "Настройте правила" in result.raw_message
    assert result.profile is not None
    service.hybrid.purchases.get.assert_called_once()  # Independently checked, never read per item.


def test_rules_replaced_during_load_reject_mixed_snapshot(card_service, monkeypatch):
    service, _ = card_service
    original = backend.load_rules
    def changed(path):
        loaded = original(path)
        save_rules(replace(loaded, kinds=("Носки",)), path)
        return loaded
    monkeypatch.setattr(backend, "load_rules", changed)
    result = service.load("canonical")
    assert result.raw_state is ViewState.STALE and result.state is ViewState.STALE
    assert not result.raw_products and not result.products


@pytest.mark.parametrize("catalog_available", [False, True])
def test_rules_change_during_selection_rejects_result_before_return(card_service, monkeypatch, catalog_available):
    service, _ = card_service
    if not catalog_available:
        service.hybrid.catalog.path.unlink()
    original = backend.select_displayed_raw
    def changed(*args, **kwargs):
        selected = original(*args, **kwargs)
        save_rules(replace(kwargs["rules"], kinds=("Носки",)), service.hybrid.rules_path)
        return selected
    monkeypatch.setattr(backend, "select_displayed_raw", changed)
    result = service.load("canonical")
    assert result.raw_state is ViewState.STALE and result.state is ViewState.STALE
    assert not result.raw_products and not result.products
    assert not result.hybrid.displayed_raw


def test_rule_change_at_delivery_clears_raw_and_next_search_uses_new_kinds(card_service):
    service, _ = card_service
    raw, metadata = service.inference.get_candidate_sources.return_value
    metadata = {code: {**values, "ВидНоменклатуры": "Носки" if n < 10 else "Брюки"}
                for n, (code, values) in enumerate(metadata.items())}
    service.inference.get_candidate_sources.return_value = raw, metadata
    result = service.load("canonical")
    assert [p.recommendation.rank for p in result.raw_products] == list(range(11, 21))
    save_rules(RecommendationRules(("Носки",), ("NOS",), 0), service.hybrid.rules_path)
    stale = service.validate_result(result)
    assert stale.raw_state is ViewState.STALE and stale.state is ViewState.STALE
    assert not stale.raw_products and not stale.products and stale.profile == result.profile
    repeated = service.load("canonical")
    assert [p.recommendation.rank for p in repeated.raw_products] == list(range(1, 11))
    assert repeated.raw_state is ViewState.READY  # Current catalog need not contain this historical kind.
    assert service.inference.get_candidate_sources.call_count == service.hybrid.purchases.get.call_count == 2


def test_greedy_backfills_beyond_displayed_raw_and_keeps_full_normalization(card_service):
    service, _ = card_service
    catalog = write_catalog(service.catalog_path.parent, [(f"{n:06d}", {"Коллекция": "NOS"}) for n in range(12)])
    save_rules(rules(catalog), service.hybrid.rules_path)
    raw = raw_result(30)
    metadata = {rec.model_item_code: {**BASE, "ВидНоменклатуры": "Носки" if rec.rank <= 10 else "Брюки",
                                     "ПолНоменклатуры": "Женский" if rec.rank <= 20 else "Мужской"}
                for rec in raw.recommendations}
    service.inference.get_candidate_sources.return_value = raw, metadata
    service.hybrid.purchases.get.return_value = checked_purchases(raw)
    result = service.load("canonical")
    assert [p.recommendation.rank for p in result.raw_products] == list(range(11, 21))
    assert {item.sources[0].source_raw_rank for item in result.hybrid.items} == set(range(21, 31))
    assert len({item.target_item_code for item in result.hybrid.items}) == 10
    for item in result.hybrid.items:
        source, = item.sources
        calc = source.calculation
        assert calc.source_count == 30 and calc.position == source.source_raw_rank - 1
        assert calc.rank_score == 1 - calc.position / 29
        assert calc.normalized_bpr_score == (source.source_raw_score - raw.recommendations[-1].raw_score) / 29
        assert calc.minimum_bpr_score == raw.recommendations[-1].raw_score
        assert item.final_score == .30 * calc.rank_score + .30 * calc.normalized_bpr_score + .40 * source.similarity_score


def test_legacy_selector_call_without_rules_is_safe(card_service):
    service, _ = card_service
    raw, _ = service.inference.get_candidate_sources.return_value
    assert not backend.select_displayed_raw(raw, service.hybrid.purchases.get.return_value)


def test_cancellation_keeps_priority_when_rules_change_during_matching(card_service, monkeypatch):
    service, _ = card_service
    stopped = False
    def cancelled_matching(*args, **kwargs):
        nonlocal stopped
        save_rules(RecommendationRules(("Носки",), ("NOS",), 0), service.hybrid.rules_path)
        stopped = True
        return ()
    monkeypatch.setattr(backend, "compose", cancelled_matching)
    result = service.load("canonical", cancelled=lambda: stopped)
    assert result.state is ViewState.CANCELLED and not result.products and not result.raw_products
