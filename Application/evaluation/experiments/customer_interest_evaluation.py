"""Bounded offline diagnostics. Never called by production Hybrid or the GUI.

The membership-fixed experiment cannot improve Recall/HitRate. Profile utility
is already signed [-1, 1]; no per-list normalization or weight fitting occurs.
"""

from collections import Counter
from dataclasses import dataclass
import math

from Application.evaluation.experiments.customer_interest import score_item
from Application.model.training_metrics import single_target_metrics


PROFILE_WEIGHTS = (0., .025, .05, .1)
ACTIVE_FIELDS = ("ВидНоменклатуры", "ВидАссортимента", "КатегорияНаСайте")


class InterestEvaluationError(ValueError):
    """Static diagnostics; input identities and source values are never echoed."""


@dataclass(frozen=True, repr=False)
class RerankResult:
    items: tuple
    profile_scores: tuple[float, ...]  # In original baseline order.
    mixed_scores: tuple[float, ...]
    weighted_attribute_coverage: tuple[float, ...]
    changed: bool
    recall: float | None
    ndcg: float | None
    hit_rate: float | None


def rerank_top10(items, profile, *, weight=0., target=None):
    """Preserve exact eligible membership, baseline scores and baseline tie order.

    Target is optional: diagnostics without independent labels have no quality
    metrics. Callers own paired cohort/target/availability provenance. A profile
    with insufficient evidence supplies exact neutral fallback.
    """
    if type(weight) not in (int, float) or weight not in PROFILE_WEIGHTS:
        raise InterestEvaluationError("UNDECLARED_PROFILE_WEIGHT")
    if not isinstance(items, tuple) or len(items) > 10:
        raise InterestEvaluationError("TOP10_BOUND_REQUIRED")
    codes, baseline = [], []
    for rank, item in enumerate(items, 1):
        if (not isinstance(item.target_item_code, str) or not item.target_item_code
                or item.final_rank != rank or type(item.final_score) not in (int, float)
                or not math.isfinite(item.final_score) or not 0 <= item.final_score <= 1):
            raise InterestEvaluationError("INVALID_HYBRID_SNAPSHOT")
        codes.append(item.target_item_code)
        baseline.append(item.final_score)
    if len(set(codes)) != len(codes) or any(a < b for a, b in zip(baseline, baseline[1:])):
        raise InterestEvaluationError("INVALID_HYBRID_SNAPSHOT")
    matches = tuple(score_item(profile, code) for code in codes)
    scores = tuple(match.score for match in matches)
    mixed = tuple(value + weight * score for value, score in zip(baseline, scores))
    # Do not sort gamma=0/all-neutral: even baseline ties must remain identical.
    order = (tuple(sorted(range(len(items)), key=lambda i: -mixed[i]))
             if weight and any(scores) else tuple(range(len(items))))
    ranked = tuple(items[i] for i in order)
    recall = ndcg = None
    if target is not None:
        if not isinstance(target, str) or not target or target != target.strip():
            raise InterestEvaluationError("INVALID_TARGET")
        position = next((n for n, item in enumerate(ranked, 1) if item.target_item_code == target), None)
        recall, ndcg = (float(v) for v in single_target_metrics(position))
    return RerankResult(ranked, scores, mixed, tuple(m.attribute_coverage for m in matches),
                        order != tuple(range(len(items))), recall, ndcg, recall)


def categorical_association(pairs, *, max_rows=50_000):
    """NMI and directional conditional entropy on complete categorical pairs.

    No Pearson correlation on arbitrary category codes, and no identifiers or
    raw attribute values in the result. High-cardinality bias remains possible.
    """
    counts = Counter()
    seen = 0
    for left, right in pairs:
        seen += 1
        if seen > max_rows:
            raise InterestEvaluationError("ASSOCIATION_ROW_BOUND")
        if left and right:
            counts[left, right] += 1
    total = sum(counts.values())
    a, b = Counter(), Counter()
    for (left, right), count in counts.items():
        a[left] += count
        b[right] += count
    def entropy(distribution):
        return -math.fsum((n / total) * math.log(n / total) for n in distribution.values()) if total else 0.
    ha, hb = entropy(a), entropy(b)
    mutual = math.fsum(n / total * math.log(n * total / (a[x] * b[y]))
                       for (x, y), n in counts.items()) if total else 0.
    return {"rows": seen, "complete_pairs": total, "left_cardinality": len(a), "right_cardinality": len(b),
            "nmi": min(1., max(0., mutual / math.sqrt(ha * hb))) if ha and hb else None,
            "h_left_given_right_nats": max(0., ha - mutual),
            "h_right_given_left_nats": max(0., hb - mutual)}


@dataclass(frozen=True)
class PilotLimits:
    max_customers: int = 96
    max_events_per_customer: int = 400
    max_events: int = 20_000
    max_rss_bytes: int = 1536 * 2**20
    max_output_bytes: int = 16 * 2**20
    min_free_bytes: int = 2 * 2**30
    max_seconds: int = 180

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise InterestEvaluationError("INVALID_PILOT_LIMITS")

    def check(self, *, customers, events, rss_bytes, output_bytes, free_bytes, elapsed_seconds):
        values = (customers, events, rss_bytes, output_bytes, free_bytes, elapsed_seconds)
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise InterestEvaluationError("INVALID_RESOURCE_MEASUREMENT")
        if (customers > self.max_customers or events > self.max_events or rss_bytes > self.max_rss_bytes
                or output_bytes > self.max_output_bytes or free_bytes < self.min_free_bytes
                or elapsed_seconds > self.max_seconds):
            raise InterestEvaluationError("PILOT_RESOURCE_BOUND")
