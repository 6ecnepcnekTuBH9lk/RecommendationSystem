"""Fixed-cohort first-future-purchase evaluation of ordered, possibly empty lists.

Independent of production inference, Qt and source loaders. No ranking scores,
training, matching, target substitution or list padding occurs here.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
import math

from Application.mindbox.selection import DEFAULT_PURCHASE_STATUSES
from Application.model.training_metrics import single_target_metrics
from .hybrid_provenance import (
    EvidenceIssue, HybridEvaluationError, InputEvidence, InputRole, Validity, assess_evidence, digest, identifier, utc,
)


class UniverseMode(str, Enum):
    SHARED = "SHARED"
    FULL_ELIGIBLE = "FULL_ELIGIBLE"


class CaseExclusion(str, Enum):
    INCOMPLETE_HISTORY = "INCOMPLETE_HISTORY"
    INCOMPLETE_LABELS = "INCOMPLETE_LABELS"
    IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
    UNRESOLVED_PRODUCTS = "UNRESOLVED_PRODUCTS"


class RefusalReason(str, Enum):
    EMPTY = "EMPTY"
    NO_CANDIDATES = "NO_CANDIDATES"
    MODEL_USER_MISSING = "MODEL_USER_MISSING"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    CANCELLED = "CANCELLED"


def _codes(values):
    if not isinstance(values, (tuple, list, set, frozenset)):
        raise HybridEvaluationError("INVALID_CODES")
    for value in values:
        identifier(value)
    return frozenset(values)


@dataclass(frozen=True, slots=True, repr=False)
class CandidateUniverse:
    mode: UniverseMode
    eligible_items: frozenset[str]
    bpr_items: frozenset[str]

    def __post_init__(self):
        if not isinstance(self.mode, UniverseMode):
            raise HybridEvaluationError("INVALID_UNIVERSE_MODE")
        for name in ("eligible_items", "bpr_items"):
            object.__setattr__(self, name, _codes(getattr(self, name)))
        if self.mode is UniverseMode.SHARED and not self.eligible_items <= self.bpr_items:
            raise HybridEvaluationError("SHARED_UNIVERSE_NOT_SCOREABLE")


@dataclass(frozen=True, slots=True, repr=False)
class Purchase:
    item_code: str
    occurred_at: datetime
    observed_at: datetime
    status: str
    quantity: Decimal = Decimal(1)

    def __post_init__(self):
        identifier(self.item_code)
        if not isinstance(self.status, str) or not isinstance(self.quantity, Decimal) or not self.quantity.is_finite():
            raise HybridEvaluationError("INVALID_PURCHASE")
        for name in ("occurred_at", "observed_at"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        if self.observed_at < self.occurred_at:
            raise HybridEvaluationError("OBSERVATION_PRECEDES_EVENT")

    @property
    def qualifies(self):
        return self.status in DEFAULT_PURCHASE_STATUSES and self.quantity > 0


@dataclass(frozen=True, slots=True, repr=False)
class HybridEvaluationCase:
    customer_id: str
    cutoff: datetime
    future_end: datetime
    universe: CandidateUniverse
    model_generation: str
    historical_purchases: tuple[Purchase, ...] = ()
    future_purchases: tuple[Purchase, ...] = ()
    uncertain_items: frozenset[str] = frozenset()
    evidence: tuple[InputEvidence, ...] = ()
    exclusions: tuple[CaseExclusion, ...] = ()

    def __post_init__(self):
        identifier(self.customer_id)
        identifier(self.model_generation)
        for name in ("cutoff", "future_end"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        if self.cutoff >= self.future_end or not isinstance(self.universe, CandidateUniverse):
            raise HybridEvaluationError("INVALID_CASE_WINDOW_OR_UNIVERSE")
        for name, expected in (("historical_purchases", Purchase), ("future_purchases", Purchase),
                               ("evidence", InputEvidence), ("exclusions", CaseExclusion)):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or any(not isinstance(v, expected) for v in values):
                raise HybridEvaluationError("INVALID_CASE_RECORDS")
            object.__setattr__(self, name, tuple(values))
        object.__setattr__(self, "uncertain_items", _codes(self.uncertain_items))
        if len(set(self.exclusions)) != len(self.exclusions):
            raise HybridEvaluationError("DUPLICATE_EXCLUSION")
        if any(p.occurred_at >= self.cutoff or p.observed_at > self.cutoff for p in self.historical_purchases):
            raise HybridEvaluationError("FUTURE_EVENT_OR_KNOWLEDGE_IN_HISTORY")
        if any(p.occurred_at < self.cutoff for p in self.future_purchases):
            raise HybridEvaluationError("LABEL_BEFORE_CUTOFF")

    @property
    def purchased_items(self):
        return frozenset(p.item_code for p in self.historical_purchases if p.qualifies)

    @property
    def candidates(self):
        return self.universe.eligible_items - self.purchased_items - self.uncertain_items

    @property
    def target(self):
        qualifying = (p for p in self.future_purchases if p.qualifies and p.occurred_at < self.future_end
                      and p.item_code not in self.purchased_items)
        first = min(qualifying, key=lambda p: (p.occurred_at, p.item_code), default=None)
        return first.item_code if first is not None else None

    def context_digest(self):
        """Private binding, never included in aggregate reports (contains identities)."""
        def purchases(values):
            return sorted((p.item_code, p.occurred_at.isoformat(), p.observed_at.isoformat(),
                           p.status, str(p.quantity)) for p in values)
        return digest({"protocol": "first_qualified_purchase_v1", "purchase_statuses": DEFAULT_PURCHASE_STATUSES,
                       "customer": self.customer_id,
                       "cutoff": self.cutoff.isoformat(), "future_end": self.future_end.isoformat(),
                       "generation": self.model_generation, "universe_mode": self.universe.mode.value,
                       "eligible": sorted(self.universe.eligible_items), "bpr": sorted(self.universe.bpr_items),
                       "history": purchases(self.historical_purchases), "future": purchases(self.future_purchases),
                       "uncertain": sorted(self.uncertain_items), "exclusions": sorted(e.value for e in self.exclusions)})


@dataclass(frozen=True, slots=True, repr=False)
class HybridEvaluationCohort:
    cutoff: datetime
    future_end: datetime
    universe_mode: UniverseMode
    cases: tuple[HybridEvaluationCase, ...]
    synthetic: bool = True

    def __post_init__(self):
        for name in ("cutoff", "future_end"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        if (self.cutoff >= self.future_end or not isinstance(self.universe_mode, UniverseMode)
                or type(self.synthetic) is not bool or not isinstance(self.cases, (tuple, list))):
            raise HybridEvaluationError("INVALID_COHORT")
        object.__setattr__(self, "cases", tuple(self.cases))
        if any(not isinstance(c, HybridEvaluationCase) for c in self.cases):
            raise HybridEvaluationError("INVALID_COHORT_CASE")
        if len({c.customer_id for c in self.cases}) != len(self.cases):
            raise HybridEvaluationError("DUPLICATE_COHORT_CUSTOMER")
        if any((c.cutoff, c.future_end, c.universe.mode) != (self.cutoff, self.future_end, self.universe_mode)
               for c in self.cases):
            raise HybridEvaluationError("COHORT_WINDOW_OR_UNIVERSE_MISMATCH")


@dataclass(frozen=True, slots=True, repr=False)
class RecommendationList:
    items: tuple[str, ...] = ()
    refusal: RefusalReason | None = None

    def __post_init__(self):
        if not isinstance(self.items, (tuple, list)):
            raise HybridEvaluationError("ORDERED_LIST_REQUIRED")
        object.__setattr__(self, "items", tuple(self.items))
        for code in self.items:
            identifier(code)
        if len(set(self.items)) != len(self.items):
            raise HybridEvaluationError("DUPLICATE_RECOMMENDATION")
        if self.refusal is not None and (not isinstance(self.refusal, RefusalReason) or self.items):
            raise HybridEvaluationError("INVALID_REFUSAL")


@dataclass(frozen=True, slots=True)
class HybridEvaluationMetrics:
    status: Validity
    mode: Validity
    universe_mode: UniverseMode
    k: int
    cohort_cases: int
    evaluated_cases: int
    excluded_cases: int
    proxy_cases: int
    quality_cases: int
    no_target_cases: int
    reachable_targets: int
    unreachable_targets: int
    hit_rate: float | None
    recall: float | None
    ndcg: float | None
    recommendation_coverage: float | None
    full_topk_coverage: float | None
    mean_recommendations: float | None
    target_reachability: float | None
    empty_results: int
    incomplete_results: int  # 1..K-1; empty is reported separately.
    empty_rate: float | None
    incomplete_rate: float | None
    bpr_target_reachability: float | None
    exclusion_counts: tuple[tuple[str, int], ...]
    evidence_issue_counts: tuple[tuple[str, int], ...]
    refusal_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class CaseAssessment:
    status: Validity
    exclusions: tuple[str, ...]
    evidence_issues: tuple[EvidenceIssue, ...]


def assess_case(case, *, synthetic=True):
    """Public preflight: choose output keys before running either recommender.

    This returns fixed reason codes only. STRICT mode also excludes PROXY cases;
    PROXY mode accepts both STRICT and PROXY, never UNAVAILABLE.
    """
    if not isinstance(case, HybridEvaluationCase) or type(synthetic) is not bool:
        raise HybridEvaluationError("INVALID_CASE_ASSESSMENT")
    assessment = assess_evidence(case.evidence, context_sha256=case.context_digest(),
                                 model_generation=case.model_generation, cutoff=case.cutoff,
                                 future_end=case.future_end, synthetic=synthetic)
    reasons = {reason.value for reason in case.exclusions}
    records = {e.role: e for e in case.evidence}
    labels = records.get(InputRole.FUTURE_LABELS)
    if labels is not None and labels.available_at is not None and any(
            p.observed_at > labels.available_at for p in case.future_purchases if p.occurred_at < case.future_end):
        reasons.add("LABEL_OBSERVED_AFTER_EVIDENCE")
    history = records.get(InputRole.PURCHASE_HISTORY)
    if history is not None and any(
            history.latest_event_at is not None and p.occurred_at > history.latest_event_at
            or history.available_at is not None and p.observed_at > history.available_at
            for p in case.historical_purchases):
        reasons.add("PURCHASE_EVIDENCE_BOUND_MISMATCH")
    return CaseAssessment(Validity.UNAVAILABLE if reasons else assessment.status,
                          tuple(sorted(reasons)), assessment.issues)


def _assess(cohort, mode):
    selected, exclusions, issues = [], Counter(), Counter()
    for case in cohort.cases:
        assessment = assess_case(case, synthetic=cohort.synthetic)
        reasons = set(assessment.exclusions)
        issues.update(f"{i.role.value}:{i.reason}" for i in assessment.evidence_issues)
        if assessment.status is Validity.UNAVAILABLE and not reasons:
            reasons.add("INPUT_UNAVAILABLE")
        elif mode is Validity.STRICT and assessment.status is Validity.PROXY:
            reasons.add("STRICT_EVIDENCE_REQUIRED")
        if reasons:
            exclusions.update(reasons)
        else:
            selected.append((case, assessment.status))
    return selected, exclusions, issues


def evaluate_recommendations(cohort, recommendations, *, k=10, mode=Validity.STRICT):
    """All data-eligible cases require explicit outputs, including algorithm failures.

    Data exclusions are algorithm independent. No-target cases remain in coverage;
    unreachable first targets and empty/refused lists remain in quality metrics.
    """
    if (not isinstance(cohort, HybridEvaluationCohort) or not isinstance(recommendations, Mapping)
            or type(k) is not int or k < 1 or mode not in (Validity.STRICT, Validity.PROXY)
            or not isinstance(mode, Validity)):
        raise HybridEvaluationError("INVALID_EVALUATION_ARGUMENTS")
    selected, exclusions, issues = _assess(cohort, mode)
    if set(recommendations) != {c.customer_id for c, _ in selected}:
        raise HybridEvaluationError("OUTPUT_COHORT_MISMATCH")
    lengths, contributions, refusals = [], [], Counter()
    reachable = unreachable = bpr_reachable = 0
    for case, _ in selected:
        result = recommendations[case.customer_id]
        if not isinstance(result, RecommendationList):
            raise HybridEvaluationError("INVALID_RECOMMENDATION_LIST")
        if len(result.items) > k:
            raise HybridEvaluationError("TOO_MANY_RECOMMENDATIONS")
        if not set(result.items) <= case.candidates:
            raise HybridEvaluationError("RECOMMENDATION_OUTSIDE_CANDIDATES")
        lengths.append(len(result.items))
        if not result.items:
            refusals[(result.refusal or RefusalReason.EMPTY).value] += 1
        target = case.target
        if target is not None:
            reachable += int(target in case.candidates)
            unreachable += int(target not in case.candidates)
            bpr_reachable += int(target in case.candidates and target in case.universe.bpr_items)
            rank = result.items.index(target) + 1 if target in result.items else None
            contributions.append(single_target_metrics(rank))
    n, q = len(selected), len(contributions)
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    hit = ratio(math.fsum(c[0] for c in contributions), q)
    empty, incomplete = lengths.count(0), sum(0 < length < k for length in lengths)
    return HybridEvaluationMetrics(
        status=mode if n else Validity.UNAVAILABLE, mode=mode, universe_mode=cohort.universe_mode, k=k,
        cohort_cases=len(cohort.cases), evaluated_cases=n, excluded_cases=len(cohort.cases) - n,
        proxy_cases=sum(status is Validity.PROXY for _, status in selected), quality_cases=q, no_target_cases=n - q,
        reachable_targets=reachable, unreachable_targets=unreachable, hit_rate=hit, recall=hit,
        ndcg=ratio(math.fsum(c[1] for c in contributions), q),
        recommendation_coverage=ratio(n - empty, n), full_topk_coverage=ratio(lengths.count(k), n),
        mean_recommendations=ratio(sum(lengths), n), target_reachability=ratio(reachable, q),
        empty_results=empty, incomplete_results=incomplete, empty_rate=ratio(empty, n),
        incomplete_rate=ratio(incomplete, n), bpr_target_reachability=ratio(bpr_reachable, q),
        exclusion_counts=tuple(sorted(exclusions.items())), evidence_issue_counts=tuple(sorted(issues.items())),
        refusal_counts=tuple(sorted(refusals.items())),
    )


@dataclass(frozen=True, slots=True)
class PairedEvaluation:
    baseline: HybridEvaluationMetrics
    challenger: HybridEvaluationMetrics
    hit_rate_delta: float | None
    ndcg_delta: float | None


def compare_recommendations(cohort, baseline, challenger, *, k=10, mode=Validity.STRICT):
    """One cohort/window/universe/eligibility policy for both algorithms; no tuning."""
    first = evaluate_recommendations(cohort, baseline, k=k, mode=mode)
    second = evaluate_recommendations(cohort, challenger, k=k, mode=mode)
    return PairedEvaluation(first, second,
                            second.hit_rate - first.hit_rate if first.hit_rate is not None else None,
                            second.ndcg - first.ndcg if first.ndcg is not None else None)
