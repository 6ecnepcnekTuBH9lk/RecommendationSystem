"""Actual composition explanations, including sources outside displayed Top-10."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Application.customer_card_data import CustomerCardService, ViewState
from Application.customer_profile_service import CustomerProfileResult, ProfileStatus
from Application.hybrid_recommendations import HybridRecommendationService, compose, similarity, similarity_breakdown
from Application.purchased_products import PurchaseEvidence
from Application.recommendation_catalog import item_from_metadata
from Application.recommendation_rules import save_rules
from Application.model.raw_bpr_inference import RawBprStatus
from tests.test_hybrid_recommendations import BASE, catalog as catalog, checked_purchases, raw_result, rules


def test_full_ranking_and_best_source_rank_37(catalog):
    raw = raw_result(40)
    items = compose(raw, {"source-36": BASE, "source-37": BASE}, catalog, rules(catalog), checked_purchases(raw))
    assert len(items) == 2
    for item in items:
        best = item.sources[0]
        c = best.calculation
        assert best.source_raw_rank in (37, 38) and c.position == best.source_raw_rank - 1 and c.source_count == 40
        low, high = raw.recommendations[-1].raw_score, raw.recommendations[0].raw_score
        assert c.minimum_bpr_score == low and c.maximum_bpr_score == high
        assert c.rank_score == 1 - c.position / 39
        assert c.normalized_bpr_score == (best.source_raw_score - low) / (high - low)
        assert c.rank_contribution == .30 * c.rank_score
        assert c.bpr_contribution == .30 * c.normalized_bpr_score
        assert c.similarity_contribution == .40 * best.similarity_score
        assert c.before_bonus == c.rank_contribution + c.bpr_contribution + c.similarity_contribution
        assert c.final_score == item.final_score == best.evidence_score == c.before_bonus
        assert c.divisor == 1 and c.bonus == 0 and len(c.features) == 10
        assert sum(f.contribution for f in c.features) == pytest.approx(best.similarity_score)
        assert len(item.sources) == 1
        assert item.final_score == max(s.evidence_score for s in item.sources)


def test_feature_breakdown_partial_categories_tokens_and_missing(catalog):
    source = item_from_metadata("source", {**BASE, "СтилеваяГруппа": ""})
    target = item_from_metadata("target", {**BASE, "КатегорияНаСайте": "jeans", "НазваниеНаСайте": "Брюки лен"})
    features = {f.feature: f for f in similarity_breakdown(source, target, catalog)}
    assert features["КатегорияНаСайте"].match == pytest.approx(1 / 3)
    assert features["НазваниеНаСайте"].match == pytest.approx(1 / 3)
    assert features["СтилеваяГруппа"].historical_value == ""
    assert features["СтилеваяГруппа"].contribution == 0
    assert sum(f.contribution for f in features.values()) == similarity(source, target, catalog)


@pytest.mark.parametrize("primary", [None, "NOS"])
def test_equal_raw_scores_bonus_and_no_intermediate_rounding(catalog, primary):
    raw = raw_result(3)
    raw = replace(raw, recommendations=tuple(replace(rec, raw_score=-1.12345678912345) for rec in raw.recommendations))
    result = compose(raw, {r.model_item_code: BASE for r in raw.recommendations}, catalog,
                     replace(rules(catalog), primary_collection=primary), checked_purchases(raw))
    for item in result:
        c = item.sources[0].calculation
        assert c.normalized_bpr_score == 1
        assert c.primary_collection == primary
        assert c.bonus == (.05 if primary == item.target_collection else 0)
        assert c.divisor == (1.05 if primary is not None else 1)
        assert c.final_score == (c.before_bonus + c.bonus) / c.divisor
        assert item.sources[0].source_raw_score == -1.12345678912345


@pytest.fixture
def card_service(tmp_path, catalog):
    inference = Mock()
    raw = raw_result(20)
    inference.get_candidate_sources.return_value = (raw, {rec.model_item_code: BASE for rec in raw.recommendations})
    purchases = Mock()
    purchases.get.return_value = PurchaseEvidence("READY", historical=PurchaseEvidence(
        "READY", checked_codes=frozenset(r.model_item_code for r in raw_result(20).recommendations)))
    purchases.is_current.return_value = True
    hybrid = HybridRecommendationService(catalog_path=catalog.path, categories_path=catalog.categories_path,
                                         rules_path=tmp_path / "rules.json", purchases=purchases)
    save_rules(rules(catalog), hybrid.rules_path)
    identity = SimpleNamespace(is_matched=True, model_user_id="model-user", canonical_customer_id="canonical", model_generation="g")
    bridge = Mock()
    bridge.resolve.return_value = identity
    bridge.is_current.return_value = True
    profile = Mock()
    profile.get_profile.return_value = CustomerProfileResult(ProfileStatus.NOT_FOUND, canonical_customer_id="canonical")
    profile.is_current.return_value = True
    service = CustomerCardService(bridge=bridge, inference=inference, hybrid=hybrid, profile_service=profile,
                                  model_dir=tmp_path / "model", raw_root=tmp_path / "raw", catalog_path=catalog.path)
    return service, catalog


@pytest.mark.parametrize("failure", ["empty", "catalog", "matching"])
def test_raw_top10_survives_later_failure_with_historical_metadata(card_service, monkeypatch, failure):
    service, catalog = card_service
    save_rules(rules(catalog), service.hybrid.rules_path)
    if failure == "empty":
        save_rules(replace(rules(catalog), minimum_stock=99999), service.hybrid.rules_path)
    elif failure == "catalog":
        service.hybrid.catalog.path.unlink()
    elif failure == "matching":
        monkeypatch.setattr("Application.hybrid_recommendations.compose", Mock(side_effect=ValueError()))
    result = service.load("canonical")
    assert result.raw_state is ViewState.READY and len(result.raw_products) == 10
    raw = service.inference.get_candidate_sources.return_value[0]
    assert tuple(p.recommendation for p in result.raw_products) == raw.recommendations[:10]
    assert result.raw_products[0].name == BASE["НазваниеНаСайте"]
    assert result.raw_products[0].collection == "Весна-Лето 2025"
    assert all(p.name == BASE["НазваниеНаСайте"] and p.collection == "Весна-Лето 2025" for p in result.raw_products)
    service.inference.get_candidate_sources.assert_called_once()
    service.inference.get_recommendations.assert_not_called()
    service.inference.get_source_metadata.assert_not_called()
    service.bridge.is_current.return_value = False
    stale = service.validate_result(result)
    assert stale.raw_state is ViewState.STALE and stale.raw_products == ()


def test_exact_current_photo_does_not_replace_historical_labels(card_service):
    service, _ = card_service
    service.catalog_path.write_text("КодНоменклатуры|НазваниеНаСайте|Коллекция|ТитульнаяФотография\n"
                                    "source-0|Современное имя|2026|/upload/synthetic.jpg\n", encoding="utf-8-sig")
    result = service.load("canonical")
    assert result.raw_products[0].photo_url == "/upload/synthetic.jpg"
    assert result.raw_products[0].name == BASE["НазваниеНаСайте"] and result.raw_products[0].collection == "Весна-Лето 2025"


def test_ambiguous_current_photo_does_not_contaminate_historical_fields(card_service):
    service, _ = card_service
    service.catalog_path.write_text("КодНоменклатуры|НазваниеНаСайте|Коллекция|ТитульнаяФотография\n"
                                    "source-0|Имя A|2026|/a.jpg\nsource-0|Имя B|2027|/b.jpg\n", encoding="utf-8-sig")
    result = service.load("canonical")
    assert result.raw_products[0].photo_url == ""
    assert result.raw_products[0].name == BASE["НазваниеНаСайте"]


def test_purchase_freshness_exception_invalidates_both_recommendation_sections(card_service):
    service, catalog = card_service
    save_rules(rules(catalog), service.hybrid.rules_path)
    result = service.load("canonical")
    assert result.state is ViewState.READY and result.raw_state is ViewState.READY
    service.hybrid.purchases.is_current.side_effect = RuntimeError()
    checked = service.validate_result(result)
    assert checked.state is ViewState.STALE and not checked.products
    assert not checked.raw_products and checked.raw_state is ViewState.STALE


@pytest.mark.parametrize("mode", ["not_in_model", "changed_generation", "empty"])
def test_missing_stale_or_empty_raw_cannot_produce_fictitious_cards(card_service, mode):
    service, catalog = card_service
    save_rules(rules(catalog), service.hybrid.rules_path)
    raw, metadata = service.inference.get_candidate_sources.return_value
    if mode == "not_in_model":
        raw = replace(raw, status=RawBprStatus.USER_NOT_IN_MODEL, recommendations=())
    elif mode == "changed_generation":
        raw = replace(raw, model_generation="other")
    else:
        raw = replace(raw, recommendations=())
    service.inference.get_candidate_sources.return_value = raw, metadata
    result = service.load("canonical")
    assert not result.raw_products and not result.products
    assert result.raw_state is {"empty": ViewState.EMPTY, "changed_generation": ViewState.STALE,
                               "not_in_model": ViewState.UNAVAILABLE}[mode]
