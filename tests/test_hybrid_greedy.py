"""Sequential assignments, full-ranking normalization, no real model or index."""

from dataclasses import replace

import pytest

from Application import hybrid_recommendations as hybrid
from Application.purchased_products import PurchaseEvidence
from tests.test_hybrid_recommendations import BASE, checked_purchases, raw_result, rules, write_catalog


def scenario(tmp_path, source_count=15, target_count=12):
    catalog = write_catalog(tmp_path, [(f"{n:06d}", {"Коллекция": "NOS"}) for n in range(target_count)])
    raw = raw_result(source_count)
    metadata = {r.model_item_code: dict(BASE) for r in raw.recommendations}
    return raw, metadata, catalog, rules(catalog), checked_purchases(raw)


def assignments(items):
    return {item.sources[0].source_raw_rank: item.target_item_code for item in items}


def test_rank_order_distinct_pairs_and_early_stop_without_matrix(tmp_path, monkeypatch):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 5545)
    calls = []
    def quality(source, target, _catalog):
        calls.append((source.code, target.code))
        return .8
    monkeypatch.setattr(hybrid, "similarity", quality)
    # Deliberately reverse the container: rank, not container order, is priority.
    reversed_raw = replace(raw, recommendations=tuple(reversed(raw.recommendations)))
    result = hybrid.compose(reversed_raw, metadata, catalog, setting, evidence)
    assert assignments(result) == {n + 1: f"{n:06d}" for n in range(10)}
    assert len({i.target_item_code for i in result}) == len(result) == 10
    assert {source for source, _ in calls} == {f"source-{n}" for n in range(10)}
    assert len(calls) == sum(12 - n for n in range(10))  # Used targets are not rescored.
    assert raw.recommendations[0].rank == 1 and raw.recommendations[-1].rank == 5545


def test_one_source_never_creates_multiple_targets(tmp_path):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 1)
    result = hybrid.compose(raw, metadata, catalog, setting, evidence)
    assert len(result) == 1 and len(result[0].sources) == 1


def test_filtering_skips_purchases_and_uncertain_without_renormalizing(tmp_path):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path)
    evidence = replace(evidence, historical=replace(evidence.historical,
        purchased=frozenset(f"source-{n}" for n in range(5)), uncertain=frozenset(("source-5",))))
    result = hybrid.compose(raw, metadata, catalog, setting, evidence)
    assert len(result) == 9
    assert set(assignments(result)) == set(range(7, 16))
    for item in result:
        source = item.sources[0]
        calc = source.calculation
        assert calc.position == source.source_raw_rank - 1
        assert calc.source_count == 15
        assert calc.rank_score == 1 - calc.position / 14
        assert calc.normalized_bpr_score == (source.source_raw_score - raw.recommendations[-1].raw_score) / 14
        assert calc.minimum_bpr_score == raw.recommendations[-1].raw_score
        assert calc.maximum_bpr_score == raw.recommendations[0].raw_score
        assert calc.final_score == item.final_score == source.evidence_score


def test_occupied_best_target_uses_next_even_with_legacy_per_source_one(tmp_path, monkeypatch):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 3, 3)
    scores = {("source-0", "000000"): .7, ("source-1", "000000"): 1.,
              ("source-1", "000001"): .6, ("source-2", "000002"): .8}
    monkeypatch.setattr(hybrid, "similarity", lambda source, target, _: scores.get((source.code, target.code), 0.))
    result = hybrid.compose(raw, metadata, catalog, setting, evidence, per_source=1)
    assert assignments(result) == {1: "000000", 2: "000001", 3: "000002"}
    # Source 2 could score higher for target 0, but cannot reclaim it from source 1.
    assert next(i for i in result if i.target_item_code == "000000").sources[0].source_raw_rank == 1


def test_no_reassignment_even_when_global_sum_would_be_higher(tmp_path, monkeypatch):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 2, 2)
    matrix = {("source-0", "000000"): .9, ("source-0", "000001"): .8,
              ("source-1", "000000"): 1., ("source-1", "000001"): 0.}
    monkeypatch.setattr(hybrid, "similarity", lambda source, target, _: matrix[(source.code, target.code)])
    result = hybrid.compose(raw, metadata, catalog, setting, evidence)
    assert assignments(result) == {1: "000000"}
    assert len(result) == 1  # Global matching could assign two; greedy must not.


def test_skip_no_metadata_no_similarity_and_no_free_target_then_backfill(tmp_path, monkeypatch):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 14, 12)
    del metadata["source-1"]
    def quality(source, target, _):
        if source.code in ("source-2", "source-3"):
            return .8 if target.code == "000000" else 0.
        return .8
    monkeypatch.setattr(hybrid, "similarity", quality)
    result = hybrid.compose(raw, metadata, catalog, setting, evidence)
    # 2 lacks metadata, 3/4 only match the already claimed target; continue beyond 10.
    assert set(assignments(result)) == {1, *range(5, 14)}
    assert 14 not in assignments(result)
    assert len(result) == 10


def test_final_display_sorted_by_score_after_assignments_not_source_order(tmp_path, monkeypatch):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 100, 2)
    def quality(source, target, _):
        if source.code == "source-0":
            return .55 if target.code == "000000" else 0.
        return 1.
    monkeypatch.setattr(hybrid, "similarity", quality)
    result = hybrid.compose(raw, metadata, catalog, setting, evidence)
    assert [i.sources[0].source_raw_rank for i in result] == [2, 1]
    assert [i.final_rank for i in result] == [1, 2]
    assert result[0].final_score > result[1].final_score


def test_candidate_selection_uses_final_rating_with_primary_bonus(tmp_path, monkeypatch):
    raw, metadata, _, _, evidence = scenario(tmp_path, 1, 2)
    catalog = write_catalog(tmp_path, [("000000", {"Коллекция": "NOS"}), ("000001", {"Коллекция": "Всесезонный"})])
    setting = replace(rules(catalog), primary_collection="Всесезонный")
    monkeypatch.setattr(hybrid, "similarity", lambda _, target, _catalog: .9 if target.code == "000000" else .8)
    item, = hybrid.compose(raw, metadata, catalog, setting, evidence)
    assert item.target_item_code == "000001"  # .05 outweighs .40 * .10.
    assert item.final_score == (.3 + .3 + .4 * .8 + .05) / 1.05
    assert item.sources[0].calculation.final_score == item.final_score


def test_ties_use_code_for_candidate_and_final_order(tmp_path, monkeypatch):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 2, 2)
    raw = replace(raw, recommendations=tuple(replace(rec, raw_score=1.) for rec in raw.recommendations))
    # Explicitly make R equal so final ties are exact; normalization has separate real tests.
    monkeypatch.setattr(hybrid, "_normalized_scores", lambda *args: (1., 1.))
    monkeypatch.setattr(hybrid, "similarity", lambda *args: .8)
    first = hybrid.compose(raw, metadata, catalog, setting, evidence)
    assert [i.target_item_code for i in first] == ["000000", "000001"]
    assert assignments(first) == {1: "000000", 2: "000001"}
    assert hybrid.compose(raw, metadata, catalog, setting, evidence) == first


@pytest.mark.parametrize("quality,allowed", [(.549999, False), (.55, True)])
def test_similarity_threshold_unchanged(tmp_path, monkeypatch, quality, allowed):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 1, 1)
    monkeypatch.setattr(hybrid, "similarity", lambda *args: quality)
    assert bool(hybrid.compose(raw, metadata, catalog, setting, evidence)) is allowed


@pytest.mark.parametrize("status", ["INDEX_NOT_READY", "INCOMPLETE_COVERAGE", "UNRESOLVED_PRODUCTS"])
def test_missing_historical_validation_never_falls_back_to_current_scope(tmp_path, status):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path)
    assert not hybrid.compose(raw, metadata, catalog, setting, PurchaseEvidence("READY"))
    assert not hybrid.compose(raw, metadata, catalog, setting, replace(evidence,
        historical=replace(evidence.historical, status=status)))


def test_no_sources_no_analogs_or_fewer_targets_produce_no_fictitious_pairs(tmp_path):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 15, 2)
    result = hybrid.compose(raw, metadata, catalog, setting, evidence)
    assert len(result) == 2
    assert not hybrid.compose(raw, {}, catalog, setting, evidence)
    assert not hybrid.compose(raw, metadata, catalog, setting, replace(evidence,
        historical=replace(evidence.historical, purchased=evidence.historical.checked_codes)))


def test_actual_formula_features_and_full_precision(tmp_path):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path, 12, 12)
    setting = replace(setting, primary_collection="NOS")
    evidence = replace(evidence, historical=replace(evidence.historical, purchased=frozenset(("source-0",))))
    result = hybrid.compose(raw, metadata, catalog, setting, evidence)
    assert len(result) == 10
    for item in result:
        source, = item.sources
        calc = source.calculation
        assert source.source_raw_score == raw.recommendations[source.source_raw_rank - 1].raw_score
        assert calc.similarity_contribution == .40 * source.similarity_score
        assert sum(f.contribution for f in calc.features) == pytest.approx(source.similarity_score)
        assert calc.before_bonus == .30 * calc.rank_score + .30 * calc.normalized_bpr_score + .40 * source.similarity_score
        assert item.final_score == (calc.before_bonus + .05) / 1.05 == calc.final_score
        assert item.final_score != round(item.final_score, 3)


def test_same_historical_and_current_code_allowed_when_verified(tmp_path):
    raw, _, catalog, setting, _ = scenario(tmp_path, 1, 1)
    raw = replace(raw, recommendations=(replace(raw.recommendations[0], model_item_code="000000"),))
    item, = hybrid.compose(raw, {"000000": BASE}, catalog, setting, checked_purchases(raw))
    assert item.target_item_code == item.sources[0].source_model_item_code == "000000"


def test_cooperative_cancellation_never_returns_partial_assignment(tmp_path, monkeypatch):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path)
    stopped = False
    original = hybrid.similarity
    def score(*args):
        nonlocal stopped
        stopped = True
        return original(*args)
    monkeypatch.setattr(hybrid, "similarity", score)
    assert not hybrid.compose(raw, metadata, catalog, setting, evidence, cancelled=lambda: stopped)


def test_legacy_per_source_does_not_change_result_or_disable_candidates(tmp_path):
    raw, metadata, catalog, setting, evidence = scenario(tmp_path)
    first = hybrid.compose(raw, metadata, catalog, setting, evidence, per_source=0)
    assert first == hybrid.compose(raw, metadata, catalog, setting, evidence, per_source=1)
    assert len(first) == 10
    assert not hybrid.compose(raw, metadata, catalog, setting, evidence, k=0)
