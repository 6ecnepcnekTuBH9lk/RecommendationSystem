"""Small hand-calculated purchase/list/provenance cases; no training or real I/O."""

from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import math

import pytest

from Application.evaluation.hybrid import (
    CandidateUniverse, CaseExclusion, HybridEvaluationCase, HybridEvaluationCohort,
    Purchase, RecommendationList, RefusalReason, UniverseMode,
    assess_case, compare_recommendations, evaluate_recommendations,
)
from Application.evaluation.hybrid_provenance import (
    EvidenceBasis, EvidenceReview, HybridEvaluationError, InputEvidence, InputRole,
    Validity, assess_evidence,
)


CUTOFF = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = CUTOFF + timedelta(days=30)
ITEMS = tuple(f"item-{n:02d}" for n in range(12))


def purchase(item=ITEMS[0], *, hour=1, observed_hour=None, status="F", quantity=Decimal(1)):
    stamp = CUTOFF + timedelta(hours=hour)
    return Purchase(item, stamp, stamp if observed_hour is None else CUTOFF + timedelta(hours=observed_hour),
                    status, quantity)


def reviewed(record, basis=EvidenceBasis.CONTROLLED_SYNTHETIC):
    record = replace(record, review=None)
    return replace(record, review=EvidenceReview(basis, record.statement_digest(),
                                               "synthetic fixture construction", "unit-test"))


def with_evidence(case):
    # Bytes encode the fixture context; each role has a distinct artifact receipt.
    records = []
    for role in InputRole:
        content = json.dumps({"synthetic_role": role.value, "context": case.context_digest()}).encode()
        labels = role is InputRole.FUTURE_LABELS
        records.append(reviewed(InputEvidence(
            role, content, hashlib.sha256(content).hexdigest(), case.context_digest(), case.model_generation,
            case.future_end if labels else case.cutoff,
            available_at=case.future_end if labels else case.cutoff,
            latest_event_at=case.cutoff - timedelta(microseconds=1)
            if role in (InputRole.TRAINING_HISTORY, InputRole.PURCHASE_HISTORY) else None,
            coverage_start=case.cutoff if labels else None,
            coverage_end=case.future_end if labels else None,
        )))
    return replace(case, evidence=tuple(records))


def make_case(customer="synthetic-customer", *, target=ITEMS[0], items=ITEMS,
              history=(), future=None, uncertain=(), mode=UniverseMode.SHARED, bpr_items=None,
              exclusions=()):
    universe = CandidateUniverse(mode, frozenset(items), frozenset(items if bpr_items is None else bpr_items))
    case = HybridEvaluationCase(customer, CUTOFF, END, universe, "synthetic-generation",
                                tuple(history), tuple([purchase(target)] if future is None and target else future or ()),
                                frozenset(uncertain), exclusions=tuple(exclusions))
    return with_evidence(case)


def cohort(*cases):
    return HybridEvaluationCohort(CUTOFF, END, cases[0].universe.mode if cases else UniverseMode.SHARED, tuple(cases))


def changed_evidence(case, role, **changes):
    return replace(case, evidence=tuple(reviewed(replace(e, review=None, **changes)) if e.role is role else e
                                        for e in case.evidence))


def evaluate(case, items=(), *, mode=Validity.STRICT, k=10, refusal=None):
    return evaluate_recommendations(cohort(case), {case.customer_id: RecommendationList(items, refusal)}, k=k, mode=mode)


def test_empty_cohort_has_no_fabricated_zero_quality():
    m = evaluate_recommendations(cohort(), {})
    assert m.status is Validity.UNAVAILABLE
    assert m.cohort_cases == m.evaluated_cases == m.quality_cases == 0
    assert m.hit_rate is m.recall is m.ndcg is m.recommendation_coverage is m.target_reachability is None


@pytest.mark.parametrize("length", [0, 1, 9, 10])
def test_list_lengths_not_padded(length):
    case = make_case()
    codes = list(ITEMS[:length])
    result = RecommendationList(codes)
    codes.append("later-mutation")
    m = evaluate_recommendations(cohort(case), {case.customer_id: result})
    assert len(result.items) == length
    assert m.mean_recommendations == length
    assert m.recommendation_coverage == bool(length)
    assert m.full_topk_coverage == (length == 10)
    assert m.empty_results == (length == 0)
    assert m.incomplete_results == (0 < length < 10)
    assert m.empty_rate == (length == 0)
    assert m.incomplete_rate == (0 < length < 10)
    assert m.hit_rate == m.recall == bool(length)


@pytest.mark.parametrize("rank", [1, 10, None])
def test_single_target_manual_oracle(rank):
    case = make_case(target=ITEMS[10])
    codes = list(ITEMS[:10])
    if rank is not None:
        codes[rank - 1] = ITEMS[10]
    m = evaluate(case, codes)
    assert m.hit_rate == m.recall == (rank is not None)
    assert m.ndcg == pytest.approx(1 / math.log2(rank + 1) if rank else 0)
    assert m.quality_cases == m.reachable_targets == 1


def test_multiple_customers_denominators_and_no_success_only_selection():
    cases = [make_case(str(n), target=ITEMS[n] if n != 3 else None) for n in range(4)]
    predictions = {"0": RecommendationList((ITEMS[0],)),
                   "1": RecommendationList((ITEMS[0], ITEMS[1])),
                   "2": RecommendationList(refusal=RefusalReason.BACKEND_UNAVAILABLE),
                   "3": RecommendationList(ITEMS[:10])}
    m = evaluate_recommendations(cohort(*cases), predictions)
    assert m.evaluated_cases == 4 and m.quality_cases == 3 and m.no_target_cases == 1
    assert m.hit_rate == m.recall == pytest.approx(2 / 3)
    assert m.ndcg == pytest.approx((1 + 1 / math.log2(3)) / 3)
    assert m.recommendation_coverage == 3 / 4
    assert m.full_topk_coverage == 1 / 4
    assert m.mean_recommendations == 13 / 4
    assert m.empty_results == 1 and m.incomplete_results == 2
    assert dict(m.refusal_counts) == {"BACKEND_UNAVAILABLE": 1}


def test_first_unreachable_target_not_replaced_by_later_purchase():
    case = make_case(future=(purchase("unreachable", hour=0), purchase(ITEMS[0], hour=2)))
    m = evaluate(case, (ITEMS[0],))
    assert case.target == "unreachable"
    assert m.quality_cases == m.unreachable_targets == 1
    assert m.target_reachability == m.bpr_target_reachability == m.hit_rate == m.ndcg == 0


def test_first_qualified_purchase_status_quantity_novelty_and_boundaries():
    past = purchase("bought", hour=-1)
    records = (purchase("cancelled", hour=0, status="otmenen"), purchase("zero", hour=0, quantity=Decimal(0)),
               purchase("unknown", hour=0, status="UNKNOWN"), purchase("bought", hour=0),
               purchase("first", hour=0, status="CP"), purchase("later", hour=1, status="delivering"),
               purchase("end", hour=720))
    case = make_case(history=(past,), future=records)
    assert case.target == "first"  # Exactly cutoff is future; repeats of purchases are not targets.
    assert make_case(future=(purchase("end", hour=720),)).target is None
    assert make_case(future=(purchase("after", hour=721),)).target is None


def test_purchase_ties_and_input_order_are_deterministic():
    events = (purchase("z", hour=1), purchase("a", hour=1), purchase("a", hour=2))
    first, second = make_case(future=events), make_case(future=tuple(reversed(events)))
    assert first.target == second.target == "a"
    assert first.context_digest() == second.context_digest()
    assert evaluate(first) == evaluate(second)


def test_seen_but_not_purchased_remains_eligible_and_uncertain_is_excluded():
    case = make_case(history=(purchase(ITEMS[1], hour=-1),), uncertain=(ITEMS[2],))
    assert case.purchased_items == {ITEMS[1]}
    assert ITEMS[0] in case.candidates and ITEMS[1] not in case.candidates and ITEMS[2] not in case.candidates
    assert evaluate(case, (ITEMS[0],)).hit_rate == 1
    for code in (ITEMS[1], ITEMS[2], "unauthorized"):
        with pytest.raises(HybridEvaluationError, match="OUTSIDE_CANDIDATES"):
            evaluate(case, (code,))


@pytest.mark.parametrize("items,error", [(ITEMS[:11], "TOO_MANY"), ((ITEMS[0], ITEMS[0]), "DUPLICATE"),
                                         ({ITEMS[0]}, "ORDERED_LIST"), ("item-00", "ORDERED_LIST")])
def test_invalid_recommendations_raise_instead_of_becoming_empty(items, error):
    with pytest.raises(HybridEvaluationError, match=error):
        evaluate(make_case(), items)


@pytest.mark.parametrize("k", [0, -1, "10", True, 1.5])
def test_invalid_k(k):
    with pytest.raises(HybridEvaluationError, match="INVALID_EVALUATION"):
        evaluate(make_case(), k=k)


def test_top_one_and_explicit_refusal():
    case = make_case()
    assert evaluate(case, (ITEMS[0],), k=1).full_topk_coverage == 1
    assert evaluate(case, refusal=RefusalReason.MODEL_USER_MISSING).hit_rate == 0
    with pytest.raises(HybridEvaluationError, match="INVALID_REFUSAL"):
        RecommendationList((ITEMS[0],), RefusalReason.CANCELLED)


def test_all_no_target_cases_still_have_coverage_without_quality():
    m = evaluate(make_case(target=None), ITEMS[:9])
    assert m.evaluated_cases == m.no_target_cases == 1
    assert m.quality_cases == 0 and m.hit_rate is m.ndcg is m.target_reachability is None
    assert m.mean_recommendations == 9


def test_shared_and_full_universes_do_not_conflate_bpr_coverage():
    with pytest.raises(HybridEvaluationError, match="SHARED_UNIVERSE"):
        make_case(items=("new",), bpr_items=())
    case = make_case(items=("new", "old"), target="new", bpr_items=("old",), mode=UniverseMode.FULL_ELIGIBLE)
    m = evaluate(case, ("new",))
    assert m.target_reachability == m.hit_rate == 1 and m.bpr_target_reachability == 0
    paired = compare_recommendations(cohort(case), {case.customer_id: RecommendationList(("old",))},
                                      {case.customer_id: RecommendationList(("new",))})
    assert paired.hit_rate_delta == paired.ndcg_delta == 1


def test_pair_requires_every_eligible_customer_for_both_algorithms():
    cases = cohort(make_case("first"), make_case("second"))
    baseline = {c.customer_id: RecommendationList() for c in cases.cases}
    challenger = {"first": RecommendationList((ITEMS[0],))}
    with pytest.raises(HybridEvaluationError, match="OUTPUT_COHORT_MISMATCH"):
        compare_recommendations(cases, baseline, challenger)
    challenger["second"] = RecommendationList(refusal=RefusalReason.BACKEND_UNAVAILABLE)
    result = compare_recommendations(cases, baseline, challenger)
    assert result == compare_recommendations(cases, dict(reversed(list(baseline.items()))), challenger)
    assert result.baseline.quality_cases == result.challenger.quality_cases == 2
    assert result.hit_rate_delta == .5


def test_utc_normalization_and_cohort_cutoff_mismatch():
    offset = timezone(timedelta(hours=3))
    case = make_case()
    shifted = replace(case, cutoff=case.cutoff.astimezone(offset), future_end=case.future_end.astimezone(offset))
    assert shifted == case
    assert shifted.cutoff.tzinfo is timezone.utc
    assert evaluate(shifted) == evaluate(case)
    with pytest.raises(HybridEvaluationError, match="COHORT_WINDOW"):
        cohort(replace(case, cutoff=CUTOFF - timedelta(seconds=1)))
    with pytest.raises(HybridEvaluationError, match="AWARE_TIMESTAMP"):
        replace(case, cutoff=CUTOFF.replace(tzinfo=None))


@pytest.mark.parametrize("event,observed", [(0, 0), (1, 2), (-1, 1)])
def test_future_event_or_late_status_cannot_be_historical(event, observed):
    with pytest.raises(HybridEvaluationError, match="FUTURE_EVENT_OR_KNOWLEDGE"):
        make_case(history=(purchase(hour=event, observed_hour=observed),))


def test_case_and_nested_inputs_are_frozen():
    items = set(ITEMS)
    universe = CandidateUniverse(UniverseMode.SHARED, items, items)
    items.clear()
    assert len(universe.eligible_items) == 12
    case = make_case()
    with pytest.raises(FrozenInstanceError):
        case.customer_id = "changed"
    with pytest.raises(FrozenInstanceError):
        case.evidence[0].available_at = END


@pytest.mark.parametrize("role", [InputRole.CATALOG, InputRole.AVAILABILITY, InputRole.IDENTITY,
                                  InputRole.PURCHASE_HISTORY])
def test_unreviewed_inputs_are_proxy_and_never_strict(role):
    case = make_case()
    case = replace(case, evidence=tuple(replace(e, review=None) if e.role is role else e for e in case.evidence))
    strict = evaluate_recommendations(cohort(case), {})
    assert strict.status is Validity.UNAVAILABLE and strict.excluded_cases == 1
    assert dict(strict.exclusion_counts) == {"STRICT_EVIDENCE_REQUIRED": 1}
    proxy = evaluate(case, mode=Validity.PROXY)
    assert proxy.status is Validity.PROXY and proxy.proxy_cases == 1 and proxy.hit_rate == 0
    assert dict(proxy.evidence_issue_counts)[f"{role.value}:REVIEW_UNPROVEN"] == 1


@pytest.mark.parametrize("role", list(InputRole))
def test_every_required_input_must_exist_even_for_proxy(role):
    case = make_case()
    case = replace(case, evidence=tuple(e for e in case.evidence if e.role is not role))
    for mode in (Validity.STRICT, Validity.PROXY):
        m = evaluate_recommendations(cohort(case), {}, mode=mode)
        assert m.status is Validity.UNAVAILABLE and m.evaluated_cases == 0
        assert dict(m.evidence_issue_counts)[f"{role.value}:MISSING_INPUT"] == 1


def test_checksums_review_and_context_are_bound_not_boolean_flags_or_filenames():
    case = make_case()
    record = case.evidence[0]
    for changes, error in (({"content": b"tampered"}, "CHECKSUM_MISMATCH"),
                           ({"available_at": CUTOFF - timedelta(seconds=1)}, "REVIEW_BINDING"),
                           ({"review": True}, "INVALID_REVIEW")):
        with pytest.raises(HybridEvaluationError, match=error):
            replace(record, **changes)
    with pytest.raises(HybridEvaluationError, match="EVIDENCE_CONTEXT"):
        evaluate(replace(case, uncertain_items={ITEMS[3]}))
    with pytest.raises(HybridEvaluationError, match="EVIDENCE_CUTOFF"):
        evaluate(changed_evidence(case, InputRole.CATALOG, as_of=END))
    case = changed_evidence(case, InputRole.CATALOG, available_at=None)
    assert evaluate(case, mode=Validity.PROXY).proxy_cases == 1
    assert evaluate_recommendations(cohort(case), {}).excluded_cases == 1


@pytest.mark.parametrize("role", [InputRole.CATALOG, InputRole.MODEL, InputRole.AVAILABILITY,
                                  InputRole.HISTORICAL_ATTRIBUTES])
def test_late_publication_is_only_proxy_even_with_review(role):
    case = changed_evidence(make_case(), role, available_at=CUTOFF + timedelta(seconds=1))
    assert evaluate(case, mode=Validity.PROXY).proxy_cases == 1
    assert evaluate_recommendations(cohort(case), {}).status is Validity.UNAVAILABLE


@pytest.mark.parametrize("role", [InputRole.TRAINING_HISTORY, InputRole.PURCHASE_HISTORY])
def test_future_history_in_evidence_is_unavailable_not_proxy(role):
    case = changed_evidence(make_case(), role, latest_event_at=CUTOFF)
    m = evaluate_recommendations(cohort(case), {}, mode=Validity.PROXY)
    assert m.status is Validity.UNAVAILABLE
    assert dict(m.evidence_issue_counts)[f"{role.value}:FUTURE_HISTORY"] == 1


def test_generation_mismatch_and_incomplete_labels_cannot_be_proxy():
    case = changed_evidence(make_case(), InputRole.MODEL, model_generation="different-generation")
    assert evaluate_recommendations(cohort(case), {}, mode=Validity.PROXY).status is Validity.UNAVAILABLE
    case = changed_evidence(make_case(), InputRole.FUTURE_LABELS, coverage_end=END - timedelta(seconds=1))
    m = evaluate_recommendations(cohort(case), {}, mode=Validity.PROXY)
    assert dict(m.evidence_issue_counts)["FUTURE_LABELS:INCOMPLETE_LABEL_WINDOW"] == 1


def test_label_knowledge_and_real_vs_synthetic_attestation():
    case = make_case(future=(purchase(observed_hour=721),))
    assert evaluate_recommendations(cohort(case), {}).status is Validity.UNAVAILABLE
    case = changed_evidence(case, InputRole.FUTURE_LABELS, available_at=END + timedelta(hours=2))
    assert evaluate(case).status is Validity.STRICT
    real_cohort = replace(cohort(case), synthetic=False)
    assert evaluate_recommendations(real_cohort, {}).status is Validity.UNAVAILABLE
    # This verifies receipt structure only; it does not certify an actual real source.
    real_case = replace(case, evidence=tuple(reviewed(e, EvidenceBasis.REVIEWED_POINT_IN_TIME) for e in case.evidence))
    assert evaluate_recommendations(replace(real_cohort, cases=(real_case,)),
                                    {case.customer_id: RecommendationList()}).status is Validity.STRICT


def test_data_exclusion_is_identical_for_both_algorithms_and_counted():
    valid = make_case("valid")
    invalid = make_case("excluded", exclusions=(CaseExclusion.INCOMPLETE_HISTORY, CaseExclusion.UNRESOLVED_PRODUCTS))
    data = cohort(valid, invalid)
    first = {"valid": RecommendationList()}
    second = {"valid": RecommendationList((ITEMS[0],))}
    pair = compare_recommendations(data, first, second)
    assert pair.baseline.excluded_cases == pair.challenger.excluded_cases == 1
    assert pair.baseline.exclusion_counts == pair.challenger.exclusion_counts
    assert dict(pair.baseline.exclusion_counts) == {"INCOMPLETE_HISTORY": 1, "UNRESOLVED_PRODUCTS": 1}
    assert pair.baseline.quality_cases == pair.challenger.quality_cases == 1


def test_no_pii_in_aggregates_repr_errors_or_output(capsys, caplog):
    private = "private-customer@example.invalid"
    case = make_case(private, target="private-product")
    report = asdict(evaluate(case))
    serialized = json.dumps(report)
    assert private not in serialized and "private-product" not in serialized
    assert private not in repr(case) + repr(cohort(case)) + repr(case.evidence[0])
    with pytest.raises(HybridEvaluationError) as error:
        evaluate(case, (private,))
    assert private not in str(error.value)
    assert private not in caplog.text + str(capsys.readouterr())


def test_evidence_assessment_is_explicit_and_reproducible():
    case = make_case()
    result = assess_evidence(case.evidence, context_sha256=case.context_digest(),
                             model_generation=case.model_generation, cutoff=CUTOFF, future_end=END, synthetic=True)
    assert result.status is Validity.STRICT and result.issues == ()
    with pytest.raises(HybridEvaluationError, match="INVALID_EVIDENCE_SET"):
        evaluate(replace(case, evidence=case.evidence + case.evidence[:1]))


def test_empty_universe_is_reachable_zero_not_a_data_exclusion():
    case = make_case(items=())
    m = evaluate(case, refusal=RefusalReason.NO_CANDIDATES)
    assert m.evaluated_cases == m.quality_cases == m.unreachable_targets == 1
    assert m.hit_rate == m.target_reachability == 0
    assert dict(m.refusal_counts) == {"NO_CANDIDATES": 1}


def test_unknown_history_bound_is_proxy_but_contradicting_bound_is_unavailable():
    case = make_case(history=(purchase(hour=-1),))
    assert assess_case(case).status is Validity.STRICT
    unknown = changed_evidence(case, InputRole.PURCHASE_HISTORY, latest_event_at=None)
    assert assess_case(unknown).status is Validity.PROXY
    assert evaluate(unknown, mode=Validity.PROXY).proxy_cases == 1
    contradicted = changed_evidence(case, InputRole.PURCHASE_HISTORY, latest_event_at=CUTOFF - timedelta(hours=2))
    assert assess_case(contradicted).status is Validity.UNAVAILABLE
    assert "PURCHASE_EVIDENCE_BOUND_MISMATCH" in assess_case(contradicted).exclusions


def test_public_preflight_selects_one_shared_data_eligible_population():
    good, proxy = make_case("good"), make_case("proxy")
    proxy = replace(proxy, evidence=tuple(replace(e, review=None) for e in proxy.evidence))
    data = cohort(good, proxy)
    strict_keys = {c.customer_id for c in data.cases if assess_case(c).status is Validity.STRICT}
    proxy_keys = {c.customer_id for c in data.cases if assess_case(c).status is not Validity.UNAVAILABLE}
    assert strict_keys == {"good"} and proxy_keys == {"good", "proxy"}
    m = evaluate_recommendations(data, {key: RecommendationList() for key in strict_keys})
    assert m.evaluated_cases == m.excluded_cases == m.quality_cases == 1


def test_more_general_k_does_not_expand_or_pad_lists():
    case = make_case()
    m = evaluate(case, ITEMS, k=20)
    assert m.mean_recommendations == 12 and m.incomplete_results == 1 and m.full_topk_coverage == 0


@pytest.mark.parametrize("changes,error", [
    ({"cutoff": END}, "INVALID_CASE_WINDOW"),
    ({"exclusions": (CaseExclusion.INCOMPLETE_HISTORY,) * 2}, "DUPLICATE_EXCLUSION"),
    ({"future_purchases": (purchase(hour=-1),)}, "LABEL_BEFORE_CUTOFF"),
    ({"future_purchases": (True,)}, "INVALID_CASE_RECORDS"),
])
def test_invalid_case_contract_is_not_an_empty_recommendation(changes, error):
    with pytest.raises(HybridEvaluationError, match=error):
        replace(make_case(), **changes)


def test_duplicate_customer_or_mixed_universe_cohort_is_rejected():
    case = make_case()
    with pytest.raises(HybridEvaluationError, match="DUPLICATE_COHORT"):
        cohort(case, case)
    other = make_case("other", mode=UniverseMode.FULL_ELIGIBLE)
    with pytest.raises(HybridEvaluationError, match="COHORT_WINDOW_OR_UNIVERSE"):
        cohort(case, other)
