"""Evaluate the real compose() baseline, without models, files or mocked math."""

from dataclasses import replace
from decimal import Decimal
from types import MappingProxyType, SimpleNamespace

import pytest

from Application.hybrid_recommendations import compose, similarity
from Application.model.raw_bpr_inference import RawBprInferenceService, RawBprRecommendation, RawBprResult, RawBprStatus
from Application.purchased_products import PurchaseEvidence, PurchasedProductsService
from Application.recommendation_catalog import RecommendationCatalog, item_from_metadata
from Application.recommendation_rules import RecommendationRules
from tests.evaluation.test_hybrid_evaluation import UniverseMode, cohort, evaluate, make_case, purchase


X = {"ВидНоменклатуры": "Брюки", "ПолНоменклатуры": "Мужской", "КатегорияНаСайте": "category-x",
     "СтилеваяГруппа": "style-x", "Марка": "brand-x", "ГруппаСоставов": "fabric-x",
     "ВидАссортимента": "assortment-x", "НазваниеНаСайте": "namex", "СезонНоски": "season-x", "Коллекция": "NOS"}
Y = {"ВидНоменклатуры": "Брюки", "ПолНоменклатуры": "Мужской", "КатегорияНаСайте": "category-y",
     "СтилеваяГруппа": "style-y", "Марка": "brand-y", "ГруппаСоставов": "fabric-y",
     "ВидАссортимента": "assortment-y", "НазваниеНаСайте": "namey", "СезонНоски": "season-y", "Коллекция": "Всесезонный"}
A = {**Y, **{key: X[key] for key in ("КатегорияНаСайте", "Марка", "ВидАссортимента", "СезонНоски")}}


@pytest.fixture(autouse=True)
def no_production_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Production source reads are forbidden in these fixtures")
    monkeypatch.setattr(RawBprInferenceService, "_get_model", forbidden)
    monkeypatch.setattr(RecommendationCatalog, "load", forbidden)
    monkeypatch.setattr(PurchasedProductsService, "get", forbidden)


def catalog(rows):
    items = {code: replace(item_from_metadata(code, values), stock=Decimal(1)) for code, values in rows}
    return SimpleNamespace(items=MappingProxyType(items), kinds=frozenset(("Брюки",)),
                           collections=frozenset(("NOS", "Всесезонный")),
                           category_paths={"category-x": frozenset(("category-x",)),
                                           "category-y": frozenset(("category-y",))})


def raw(count):
    return RawBprResult("synthetic-generation", "synthetic-user", RawBprStatus.OK,
                        tuple(RawBprRecommendation(n + 1, f"source-{n}", float(count - n - 1)) for n in range(count)))


def checked(ranking, *, bought=(), uncertain=(), historical_bought=(), historical_uncertain=()):
    return PurchaseEvidence("READY", frozenset(bought), frozenset(uncertain), historical=PurchaseEvidence(
        "READY", frozenset(historical_bought), frozenset(historical_uncertain),
        checked_codes=frozenset(r.model_item_code for r in ranking.recommendations)))


RULES = RecommendationRules(("Брюки",), ("NOS", "Всесезонный"), 0)


def full_case(ranking, *, items, target, history=(), uncertain=()):
    return make_case(items=items, target=target, history=history, uncertain=uncertain,
                     mode=UniverseMode.FULL_ELIGIBLE,
                     bpr_items=tuple(rec.model_item_code for rec in ranking.recommendations))


def test_greedy_fewer_pairs_than_feasible_assignment_is_the_baseline():
    current = catalog((("X", X), ("Y", Y)))
    ranking = raw(2)
    metadata = {"source-0": A, "source-1": X}
    for values, expected in ((A, [.65, .57]), (X, [1., .22])):
        assert [similarity(item_from_metadata("source", values), item, current)
                for item in current.items.values()] == pytest.approx(expected)
    items = compose(ranking, metadata, current, RULES, checked(ranking))
    assert len(items) == 1 and items[0].target_item_code == "X"
    assert items[0].final_score == pytest.approx(.86)
    # Both alternative edges really exist under production similarity/thresholds.
    assert similarity(item_from_metadata("a", A), current.items["Y"], current) >= .55
    assert similarity(item_from_metadata("b", X), current.items["X"], current) >= .55
    result = evaluate(full_case(ranking, target="Y", items=("X", "Y")), tuple(i.target_item_code for i in items))
    assert result.mean_recommendations == 1 and result.hit_rate == 0
    assert result.incomplete_results == 1 and result.target_reachability == 1


def test_late_better_source_is_not_considered_after_ten_successes():
    current = catalog(tuple((f"T{n:02d}", Y) for n in range(11)))
    ranking = raw(11)
    metadata = {rec.model_item_code: A if rec.rank <= 10 else Y for rec in ranking.recommendations}
    items = compose(ranking, metadata, current, RULES, checked(ranking))
    assert len(items) == 10 and max(i.sources[0].source_raw_rank for i in items) == 10
    assert min(i.final_score for i in items) == pytest.approx(.288)
    assert all(i.sources[0].calculation.source_count == 11 for i in items)
    # Same complete ranking; masking the earlier sources exposes the actual late pair.
    late = compose(ranking, metadata, current, RULES, checked(
        ranking, historical_bought=tuple(rec.model_item_code for rec in ranking.recommendations[:10])))
    assert len(late) == 1 and late[0].sources[0].source_raw_rank == 11
    assert late[0].final_score == pytest.approx(.4)
    assert late[0].final_score > min(i.final_score for i in items)
    result = evaluate(full_case(ranking, target="T10", items=tuple(current.items)), tuple(i.target_item_code for i in items))
    assert result.full_topk_coverage == 1 and result.hit_rate == 0 and result.target_reachability == 1


def test_full_sources_normalization_purchases_uniqueness_and_final_order():
    ranking = raw(40)
    current = catalog((("T0", Y), ("T1", X), ("bought", X), ("uncertain", X)))
    metadata = {rec.model_item_code: {} for rec in ranking.recommendations}
    low = {**Y, **{key: "" for key in ("КатегорияНаСайте", "Марка", "ВидАссортимента", "СезонНоски")}}
    metadata.update({"source-35": X, "source-36": X, "source-37": low, "source-38": X})
    evidence = checked(ranking, bought=("bought",), uncertain=("uncertain",),
                       historical_bought=("source-35",), historical_uncertain=("source-36",))
    items = compose(ranking, metadata, current, RULES, evidence)
    assert len(items) == 2 and len({i.target_item_code for i in items}) == 2
    assert {i.sources[0].source_raw_rank for i in items} == {38, 39}
    assert [i.sources[0].source_raw_rank for i in items] == [39, 38]
    assert compose(replace(ranking, recommendations=ranking.recommendations[:10]), metadata, current, RULES, evidence) == ()
    assert [i.final_score for i in items] == sorted((i.final_score for i in items), reverse=True)
    for item in items:
        calc = item.sources[0].calculation
        assert calc.source_count == 40 and calc.minimum_bpr_score == 0 and calc.maximum_bpr_score == 39
        assert calc.rank_score == pytest.approx(1 - calc.position / 39)
        assert calc.normalized_bpr_score == pytest.approx(item.sources[0].source_raw_score / 39)
    codes = tuple(i.target_item_code for i in items)
    case = full_case(ranking, target=codes[-1], items=tuple(current.items),
                     history=(purchase("bought", hour=-1),), uncertain=("uncertain",))
    result = evaluate(case, codes)
    assert result.ndcg == pytest.approx(1 / 1.584962500721156)
    assert compose(ranking, metadata, current, RULES, evidence) == items


def test_production_empty_list_remains_empty_in_evaluation():
    ranking = raw(1)
    current = catalog((("X", X),))
    items = compose(ranking, {}, current, RULES, checked(ranking))
    assert items == ()
    result = evaluate(full_case(ranking, target="X", items=("X",)), tuple(i.target_item_code for i in items))
    assert result.empty_results == 1 and result.mean_recommendations == 0
    assert result.quality_cases == 1 and result.hit_rate == result.ndcg == 0


def test_evaluator_does_not_sort_or_regenerate_the_production_order():
    from Application.evaluation.hybrid import RecommendationList, evaluate_recommendations
    ranking = raw(2)
    current = catalog((("X", X), ("Y", Y)))
    items = compose(ranking, {"source-0": A, "source-1": Y}, current, RULES, checked(ranking))
    codes = tuple(i.target_item_code for i in items)
    case = full_case(ranking, target=codes[0], items=tuple(current.items))
    result = evaluate_recommendations(cohort(case), {case.customer_id: RecommendationList(codes)})
    assert result.ndcg == 1
    assert codes == tuple(i.target_item_code for i in items)


def test_shared_candidate_restriction_is_applied_before_matching():
    from Application.evaluation.hybrid_provenance import HybridEvaluationError
    ranking = replace(raw(2), recommendations=(RawBprRecommendation(1, "X", 1.), RawBprRecommendation(2, "Y", 0.)))
    metadata = {"X": X, "Y": Y}
    current = catalog((("A-outside-shared", X), ("X", X), ("Y", Y)))
    evidence = checked(ranking)
    case = make_case(target="X", items=("X", "Y"), bpr_items=("X", "Y"))
    unrestricted = compose(ranking, metadata, current, RULES, evidence)
    assert unrestricted[0].target_item_code == "A-outside-shared"
    with pytest.raises(HybridEvaluationError, match="OUTSIDE_CANDIDATES"):
        evaluate(case, tuple(i.target_item_code for i in unrestricted))
    # Restrict targets, retaining the complete source ranking and other catalog facts.
    restricted = SimpleNamespace(**{**vars(current), "items": MappingProxyType({
        code: current.items[code] for code in sorted(case.candidates)})})
    items = compose(ranking, metadata, restricted, RULES, evidence)
    assert tuple(i.target_item_code for i in items) == ("X", "Y")
    assert evaluate(case, tuple(i.target_item_code for i in items)).hit_rate == 1
    assert all(i.sources[0].calculation.source_count == 2 for i in items)
