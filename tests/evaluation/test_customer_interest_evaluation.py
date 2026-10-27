"""Offline invariants and bounded diagnostics, all data synthetic."""

from dataclasses import replace
from datetime import timedelta
import math
from types import SimpleNamespace

import pytest

from Application.evaluation.experiments.customer_interest import InterestInputError, build_profile, score_item
from Application.evaluation.experiments.customer_interest_evaluation import (
    PROFILE_WEIGHTS, InterestEvaluationError, PilotLimits, categorical_association, rerank_top10,
)
from Application.interactions import InteractionType
from scripts.demo_customer_interest import CUTOFF, action, fixtures
from scripts.evaluate_customer_interest import ablations, main, synthetic_baseline, synthetic_evaluation


def fixture(events=None, **kwargs):
    items, background, context = fixtures()
    events = [action(f"10000{i}", i, InteractionType.FAVORITE) for i in range(1, 4)] if events is None else events
    return synthetic_baseline(items), build_profile("synthetic", events, items, background, context, **kwargs)


@pytest.mark.parametrize("weight", PROFILE_WEIGHTS)
def test_same_candidates_and_recall_for_every_weight(weight):
    baseline, profile = fixture()
    original = tuple((i.target_item_code, i.final_rank, i.final_score) for i in baseline)
    for target in (baseline[0].target_item_code, baseline[-1].target_item_code, "outside"):
        zero = rerank_top10(baseline, profile, target=target)
        actual = rerank_top10(baseline, profile, weight=weight, target=target)
        assert set(i.target_item_code for i in actual.items) == set(i.target_item_code for i in baseline)
        assert actual.recall == zero.recall == actual.hit_rate == zero.hit_rate
    assert tuple((i.target_item_code, i.final_rank, i.final_score) for i in baseline) == original


def test_gamma_zero_neutral_and_ties_preserve_exact_baseline_objects():
    baseline, profile = fixture()
    tied = tuple(replace(i, final_score=.5) for i in baseline)
    assert rerank_top10(tied, profile).items == tied
    _, empty = fixture([])
    for weight in PROFILE_WEIGHTS:
        result = rerank_top10(tied, empty, weight=weight)
        assert not result.changed and all(a is b for a, b in zip(result.items, tied))
        assert result.recall is None and result.ndcg is None and result.hit_rate is None


def test_independent_metric_oracle_and_reordering_is_not_recall_gain():
    baseline, profile = fixture()
    tied = tuple(replace(i, final_score=.5) for i in baseline)
    target = next(i.target_item_code for i in tied if i.target_kind == "Пиджак" and i.final_rank > 1)
    result = rerank_top10(tied, profile, weight=.1, target=target)
    assert result.changed
    position = next(n for n, i in enumerate(result.items, 1) if i.target_item_code == target)
    assert result.ndcg == pytest.approx(1 / math.log2(position + 1))
    assert result.recall == result.hit_rate == 1


@pytest.mark.parametrize("change", ["duplicate", "too_many", "unsorted", "bad_rank", "nonfinite", "scale"])
def test_bad_or_incomparable_hybrid_snapshot_rejected(change):
    items, profile = fixture()
    if change == "duplicate":
        items = (items[0], replace(items[1], target_item_code=items[0].target_item_code))
    elif change == "too_many":
        items += (items[0],)
    elif change == "unsorted":
        items = (replace(items[0], final_score=.1), replace(items[1], final_score=.9))
    elif change == "bad_rank":
        items = (replace(items[0], final_rank=2),)
    elif change == "nonfinite":
        items = (replace(items[0], final_score=math.nan),)
    else:
        items = (replace(items[0], final_score=2.),)
    with pytest.raises(InterestEvaluationError):
        rerank_top10(items, profile)


@pytest.mark.parametrize("weight", [float("nan"), -.1, .2, True, "0.1"])
def test_only_predeclared_grid_accepted(weight):
    with pytest.raises(InterestEvaluationError, match="WEIGHT"):
        rerank_top10(*fixture(), weight=weight)


def test_categorical_association_known_oracles_missing_constant_and_bound():
    identical = categorical_association([("A", "X"), ("B", "Y")] * 4)
    independent = categorical_association([("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")])
    assert identical["nmi"] == pytest.approx(1.)
    assert identical["h_left_given_right_nats"] == pytest.approx(0.)
    assert independent["nmi"] == pytest.approx(0.)
    assert categorical_association([("", "X"), ("A", "")])["complete_pairs"] == 0
    assert categorical_association([("A", "X")] * 3)["nmi"] is None
    assert "A" not in repr(identical) and "X" not in repr(identical)
    with pytest.raises(InterestEvaluationError, match="ROW_BOUND"):
        categorical_association(iter([("A", "X")] * 10), max_rows=3)


@pytest.mark.parametrize("measurement,value", [("customers", 97), ("events", 20_001),
    ("rss_bytes", 1536 * 2**20 + 1), ("output_bytes", 16 * 2**20 + 1),
    ("free_bytes", 2 * 2**30 - 1), ("elapsed_seconds", 181)])
def test_every_resource_guard_is_enforced(measurement, value):
    measurements = dict(customers=1, events=1, rss_bytes=1, output_bytes=1, free_bytes=3 * 2**30, elapsed_seconds=1.)
    PilotLimits().check(**measurements)
    measurements[measurement] = value
    with pytest.raises(InterestEvaluationError, match="RESOURCE_BOUND"):
        PilotLimits().check(**measurements)


def test_available_future_event_never_admitted_even_exploratory():
    items, reference, context = fixtures()
    context = replace(context, mode="EXPLORATORY", identity_verified=False)
    events = [replace(action(f"10000{i}", i), available_at=CUTOFF + timedelta(days=1)) for i in range(1, 4)]
    profile = build_profile("synthetic", events, items, reference, context)
    assert profile.status == "EMPTY" and profile.diagnostics["unavailable_events"] == 3
    with pytest.raises(InterestInputError, match="SYNTHETIC"):
        replace(context, dataset_kind="REAL")


def test_repetition_saturates_without_changing_same_day_evidence():
    events = [action(f"10000{i}", i) for i in range(1, 4)]
    _, once = fixture(events)
    repeated = events + [replace(e, event_id=f"extra-{n}-{i}") for n in range(30) for i, e in enumerate(events)]
    _, many = fixture(repeated)
    assert score_item(once, "100006").score == score_item(many, "100006").score
    _, long = fixture([action(f"10000{i}", day) for i in range(1, 4) for day in range(1, 366)])
    assert long.diagnostics["total_mass"] <= 3 * 2  # Per-item VIEW channel cap.


def test_hierarchical_clone_fields_do_not_triple_weight():
    baseline, profile = fixture()
    full = rerank_top10(baseline, profile, weight=.1)
    for name, config in ablations().items():
        if name.startswith("ONLY_"):
            _, single = fixture(config=config)
            actual = rerank_top10(baseline, single, weight=.1)
            assert actual.profile_scores == pytest.approx(full.profile_scores)


def test_zero_reference_frequency_is_smoothed_not_infinite_and_sparse_remains_neutral():
    items, background, context = fixtures()
    distributions = {f: {v: float(n == 0) for n, v in enumerate(q)} for f, q in background.distributions.items()}
    background = replace(background, distributions=distributions)
    events = [action(f"20000{i}", i, InteractionType.FAVORITE) for i in range(1, 4)]
    profile = build_profile("synthetic", events, items, background, context)
    actual = score_item(profile, "200006")
    assert math.isfinite(actual.score) and 0 < actual.score <= actual.confidence <= 1
    for field in profile.fields:
        assert all(q > 0 for q in field.prior.values())
        assert all(abs(lift) <= 1 for lift in field.lift.values())
    sparse = build_profile("synthetic", events[:1], items, background, context)
    assert score_item(sparse, "200006").score == 0


def test_synthetic_evaluation_deterministic_paired_not_independent_evidence():
    first, second = synthetic_evaluation(), synthetic_evaluation()
    assert first == second and first["independent_customers"] == 0 and first["selected_weight"] is None
    assert len(first["rows"]) == 12 * 10 * 4
    groups = {}
    for row in first["rows"]:
        groups.setdefault((row["case"], row["ablation"]), set()).add(row["recall"])
    assert all(len(values) == 1 for values in groups.values())


def test_output_cannot_target_protected_sources():
    from scripts.evaluate_customer_interest import ROOT
    with pytest.raises(SystemExit):
        main(["--output", str(ROOT / "model")])


def test_null_on_empty_top10():
    _, profile = fixture()
    result = rerank_top10((), profile, weight=.1, target="outside")
    assert result.items == () and result.recall == result.ndcg == 0


def test_real_identity_size_gate_stops_before_reading_or_resolving(monkeypatch, tmp_path):
    from Application import customer_history_index as index
    from Application.mindbox import raw_reader
    from Application.model import customer_identity_bridge as bridge
    from scripts import evaluate_customer_interest as runner
    fake = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=33 * 2**20))
    monkeypatch.setattr(bridge, "_capture_sources", lambda *a: SimpleNamespace(
        mappings_path=fake, merges_directory=tmp_path))
    monkeypatch.setattr(raw_reader, "part_files", lambda *a: (fake,))
    monkeypatch.setattr(index, "CustomerHistoryIndexService", lambda: SimpleNamespace(
        get_status=lambda **k: SimpleNamespace(status=SimpleNamespace(value="READY"))))
    monkeypatch.setattr(runner, "rss_bytes", lambda: 1)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda *a: SimpleNamespace(free=3 * 2**30))
    def forbidden(*args, **kwargs):
        raise AssertionError("Identity or history must not be read after admission failure")
    monkeypatch.setattr(bridge.CustomerIdentityBridge, "resolve", forbidden)
    monkeypatch.setattr(index, "_read_index", forbidden)
    result = runner.real_diagnostics(tmp_path)
    assert result["status"] == "BLOCKED_IDENTITY_INPUT_BOUND"
    assert result["real_profiles_built"] == result["history_rows_read"] == 0
