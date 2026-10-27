"""Isolated interest-profile baseline. No I/O, ranking or GUI integration.

Signed scores express preference relative to a reference population, never a
purchase probability. Indexed retrospective research is explicitly exploratory;
it must never masquerade as a historical point-in-time forecast.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from itertools import islice
import math
from types import MappingProxyType

from Application.interactions import InteractionType
from Application.mindbox.records import ProductKey
from Application.model.item_feature_snapshot import FEATURE_FIELDS
from Application.product_resolution import CatalogDiagnostics, ProductCatalog, ProductResolver, ResolutionStatus
from Application.purchased_products import assess_order
from Application.recommendation_catalog import CatalogItem, item_from_metadata


class InterestInputError(ValueError):
    """Static diagnostics only; no customer, source or product values."""


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise InterestInputError("AWARE_TIME_REQUIRED")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ResearchContext:
    cutoff: datetime
    catalog_available_at: datetime | None
    identity_verified: bool = False
    event_availability_verified: bool = False
    mode: str = "STRICT"
    dataset_kind: str = "SYNTHETIC"
    # Event window and knowledge date are deliberately separate for retrospective
    # indexed descriptions. The default preserves all original as-of guards.
    knowledge_cutoff: datetime | None = None

    def __post_init__(self):
        object.__setattr__(self, "cutoff", _utc(self.cutoff))
        if self.catalog_available_at is not None:
            object.__setattr__(self, "catalog_available_at", _utc(self.catalog_available_at))
        if self.dataset_kind not in ("SYNTHETIC", "INDEX_EXPLORATORY") or self.mode not in ("STRICT", "EXPLORATORY"):
            raise InterestInputError("SYNTHETIC_RESEARCH_ONLY")
        knowledge = self.cutoff if self.knowledge_cutoff is None else _utc(self.knowledge_cutoff)
        if self.knowledge_cutoff is not None:
            object.__setattr__(self, "knowledge_cutoff", knowledge)
        if self.dataset_kind == "INDEX_EXPLORATORY":
            if self.mode != "EXPLORATORY" or knowledge < self.cutoff:
                raise InterestInputError("INDEX_REQUIRES_RETROSPECTIVE_EXPLORATORY")
        elif knowledge != self.cutoff:
            raise InterestInputError("SYNTHETIC_KNOWLEDGE_MUST_MATCH_CUTOFF")
        if type(self.identity_verified) is not bool or type(self.event_availability_verified) is not bool:
            raise InterestInputError("INVALID_VERIFICATION_FLAGS")
        if self.mode == "STRICT" and (not self.identity_verified or not self.event_availability_verified
                or self.catalog_available_at is None or self.catalog_available_at > self.cutoff):
            raise InterestInputError("STRICT_EVIDENCE_REQUIRED")


@dataclass(frozen=True)
class InterestConfig:
    action_weights: tuple = (("VIEW", 1.), ("FAVORITE", 3.), ("PURCHASE", 4.))
    half_lives: tuple = (("VIEW", 30.), ("FAVORITE", 90.), ("PURCHASE", 180.))
    field_weights: tuple = tuple((name, {"ВидНоменклатуры": .5, "ВидАссортимента": .3,
                                         "КатегорияНаСайте": .2}.get(name, 0.)) for name in FEATURE_FIELDS)
    lookback_days: int = 365
    item_channel_cap: float = 2.
    prior_mass: float = 3.
    uniform_mix: float = .05
    max_log_lift: float = math.log(4)
    min_items: int = 3
    min_events: int = 3
    min_mass: float = 1.
    full_effective_items: float = 5.
    min_reference_users: int = 20
    min_candidate_coverage: float = .5
    max_events: int = 10_000
    max_catalog_items: int = 10_000

    def __post_init__(self):
        for pairs, expected in ((self.action_weights, {"VIEW", "FAVORITE", "PURCHASE"}),
                                (self.half_lives, {"VIEW", "FAVORITE", "PURCHASE"}),
                                (self.field_weights, set(FEATURE_FIELDS))):
            if (not isinstance(pairs, tuple) or len(dict(pairs)) != len(pairs) or set(dict(pairs)) != expected
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for _, v in pairs)):
                raise InterestInputError("INVALID_RESEARCH_WEIGHTS")
        if any(v <= 0 for _, v in self.half_lives) or sum(dict(self.field_weights).values()) <= 0:
            raise InterestInputError("INVALID_RESEARCH_WEIGHTS")
        for value in (self.lookback_days, self.min_items, self.min_events, self.min_reference_users,
                      self.max_events, self.max_catalog_items):
            if type(value) is not int or not 1 <= value <= 10_000:
                raise InterestInputError("INVALID_RESEARCH_BOUND")
        for value in (self.item_channel_cap, self.prior_mass, self.max_log_lift, self.min_mass, self.full_effective_items):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise InterestInputError("INVALID_RESEARCH_SCALE")
        if not 0 < self.uniform_mix < 1 or not 0 <= self.min_candidate_coverage <= 1:
            raise InterestInputError("INVALID_RESEARCH_FRACTION")


@dataclass(frozen=True, repr=False)
class InterestAction:
    event_id: str
    customer_id: str
    when: datetime
    kind: InteractionType
    products: tuple[ProductKey, ...]
    available_at: datetime | None = None

    def __post_init__(self):
        if not isinstance(self.kind, InteractionType) or self.kind not in (InteractionType.VIEW, InteractionType.FAVORITE):
            raise InterestInputError("PURCHASE_REQUIRES_ORDER_CONSENSUS")
        if not isinstance(self.products, tuple) or any(not isinstance(p, ProductKey) for p in self.products):
            raise InterestInputError("INVALID_PRODUCT_REFERENCES")
        object.__setattr__(self, "when", _utc(self.when))
        if self.available_at is not None:
            object.__setattr__(self, "available_at", _utc(self.available_at))


@dataclass(frozen=True, repr=False)
class InterestLine:
    product: ProductKey | None
    status: str
    quantity: Decimal = Decimal(1)
    source_line_key: str | None = None

    def __post_init__(self):
        if (self.product is not None and not isinstance(self.product, ProductKey)) or not isinstance(self.status, str):
            raise InterestInputError("INVALID_ORDER_LINE")
        if not isinstance(self.quantity, Decimal) or not self.quantity.is_finite():
            raise InterestInputError("INVALID_ORDER_QUANTITY")
        if self.source_line_key is not None and not isinstance(self.source_line_key, str):
            raise InterestInputError("INVALID_ORDER_LINE_ID")


@dataclass(frozen=True, repr=False)
class InterestSnapshot:
    lines: tuple[InterestLine, ...]
    available_at: datetime | None

    def __post_init__(self):
        if not isinstance(self.lines, tuple) or any(not isinstance(line, InterestLine) for line in self.lines):
            raise InterestInputError("INVALID_ORDER_SNAPSHOT")
        if self.available_at is not None:
            object.__setattr__(self, "available_at", _utc(self.available_at))


@dataclass(frozen=True, repr=False)
class InterestOrder:
    event_id: str
    customer_id: str
    when: datetime
    snapshots: tuple[InterestSnapshot, ...]
    owner_conflict: bool = False

    def __post_init__(self):
        object.__setattr__(self, "when", _utc(self.when))
        if not isinstance(self.snapshots, tuple) or any(not isinstance(s, InterestSnapshot) for s in self.snapshots):
            raise InterestInputError("INVALID_ORDER_SNAPSHOTS")


@dataclass(frozen=True)
class Background:
    distributions: Mapping = field(repr=False)
    reference_users: int
    event_until: datetime
    availability_verified: bool = False

    def __post_init__(self):
        object.__setattr__(self, "event_until", _utc(self.event_until))
        if type(self.reference_users) is not int or self.reference_users < 0:
            raise InterestInputError("INVALID_REFERENCE_SUPPORT")
        if type(self.availability_verified) is not bool:
            raise InterestInputError("INVALID_VERIFICATION_FLAGS")
        frozen = {}
        for name, distribution in self.distributions.items():
            if name not in FEATURE_FIELDS or not distribution or any(
                    not isinstance(v, str) or not v or type(p) not in (int, float) or not math.isfinite(p) or p < 0
                    for v, p in distribution.items()) or not math.isclose(sum(distribution.values()), 1., abs_tol=1e-9):
                raise InterestInputError("INVALID_REFERENCE_DISTRIBUTION")
            frozen[name] = MappingProxyType(dict(distribution))
        object.__setattr__(self, "distributions", MappingProxyType(frozen))


@dataclass(frozen=True)
class AttributeInterest:
    name: str
    mass: float
    confidence: float
    effective_items: float
    prior: Mapping
    posterior: Mapping
    lift: Mapping
    counts: Mapping = field(default_factory=lambda: MappingProxyType({}), repr=False)


@dataclass(frozen=True, repr=False)
class InterestProfile:
    status: str
    mode: str
    fields: tuple[AttributeInterest, ...]
    diagnostics: Mapping
    day_confidence: float
    config: InterestConfig
    catalog: Mapping = field(repr=False)
    cutoff: datetime
    warnings: tuple[str, ...] = ()
    knowledge_cutoff: datetime | None = None


@dataclass(frozen=True)
class ProfileMatch:
    status: str
    score: float
    confidence: float
    attribute_coverage: float
    contributions: tuple[dict, ...]
    mode: str
    warnings: tuple[str, ...]
    as_of_utc: str
    knowledge_cutoff_utc: str | None = None


def _effective(values):
    values = tuple(values)
    total = math.fsum(values)
    return total * total / math.fsum(v * v for v in values) if total else 0.


def build_profile(customer_id, events, catalog, background, context, config=InterestConfig()):
    """Consume bounded, already canonical logical events; never resolve customer merges here."""
    if not isinstance(customer_id, str) or not customer_id or customer_id != customer_id.strip():
        raise InterestInputError("CANONICAL_CUSTOMER_REQUIRED")
    if not isinstance(context, ResearchContext) or not isinstance(background, Background):
        raise InterestInputError("INVALID_RESEARCH_CONTEXT")
    if background.event_until > context.cutoff or (context.mode == "STRICT" and not background.availability_verified):
        raise InterestInputError("REFERENCE_AS_OF_UNVERIFIED")
    known_until = context.knowledge_cutoff or context.cutoff
    items = {}
    for position, item in enumerate(islice(iter(catalog), config.max_catalog_items + 1), 1):
        if not isinstance(item, CatalogItem) or position > config.max_catalog_items:
            raise InterestInputError("INVALID_OR_UNBOUNDED_CATALOG")
        if not isinstance(item.code, str) or not item.code or item.code != item.code.strip() or not isinstance(item.attributes, Mapping):
            raise InterestInputError("INVALID_CATALOG_ITEM")
        frozen = item_from_metadata(item.code, dict(item.attributes))
        if item.code in items and items[item.code] != frozen:
            raise InterestInputError("AMBIGUOUS_CATALOG_CODE")
        items[item.code] = frozen
    resolver = ProductResolver(ProductCatalog(frozenset(items), CatalogDiagnostics()))
    diagnostics = Counter()
    unique, conflicts = {}, set()
    for count, event in enumerate(events, 1):
        if count > config.max_events:
            raise InterestInputError("BOUNDED_EVENTS_EXCEEDED")
        if not isinstance(event, (InterestAction, InterestOrder)) or event.customer_id != customer_id:
            raise InterestInputError("MIXED_OR_UNCONFIRMED_CUSTOMER")
        if not isinstance(event.event_id, str) or not event.event_id or event.event_id != event.event_id.strip():
            raise InterestInputError("INVALID_EVENT_ID")
        age = (context.cutoff - event.when).total_seconds() / 86400
        if not 0 < age <= config.lookback_days:
            diagnostics["outside_window_events"] += 1
            continue
        if isinstance(event, InterestAction):
            availability = (event.available_at,)
        else:
            availability = tuple(s.available_at for s in event.snapshots)
        if context.mode == "STRICT" and any(stamp is None for stamp in availability):
            raise InterestInputError("MISSING_EVENT_AVAILABILITY")
        if isinstance(event, InterestAction):
            if event.available_at is not None and event.available_at > known_until:
                diagnostics["unavailable_events"] += 1
                continue
        else:
            event = replace(event, snapshots=tuple(s for s in event.snapshots
                            if s.available_at is None or s.available_at <= known_until))
            if not event.snapshots:
                diagnostics["unavailable_events"] += 1
                continue
        key = (type(event), event.event_id)
        if key in unique:
            diagnostics["duplicate_events"] += 1
            if unique[key] != event:
                conflicts.add(key)
        else:
            unique[key] = event
    diagnostics["conflicting_event_ids"] = len(conflicts)
    buckets = {}
    active_days = set()
    for key, event in sorted(unique.items(), key=lambda pair: (pair[1].when, pair[1].event_id, pair[0][0].__name__)):
        if key in conflicts:
            continue
        references = (event.products if isinstance(event, InterestAction) else
                      tuple(line.product for snapshot in event.snapshots for line in snapshot.lines))
        resolved, unresolved = set(), set()
        for reference in references:
            match = resolver.resolve(reference, strict=False) if reference is not None else None
            if match is not None and match.status is ResolutionStatus.RESOLVED:
                resolved.add(match.item_id)
            else:
                unresolved.add(reference)
        denominator = len(resolved) + len(unresolved)
        diagnostics["unresolved_product_slots"] += len(unresolved)
        if isinstance(event, InterestOrder):
            rows = [[{"raw_status": line.status, "quantity": str(line.quantity), "source_line_key": line.source_line_key,
                      "ordinal": ordinal, "selected_namespace": line.product.namespace if line.product else None,
                      "selected_value": line.product.value if line.product else None}
                     for ordinal, line in enumerate(snapshot.lines)] for snapshot in event.snapshots]
            purchased, uncertain, _, _, _, unresolved_order = assess_order(rows, resolver, owner_conflict=event.owner_conflict)
            if unresolved_order:
                diagnostics["unresolved_order_events"] += 1
                purchased = set()
            resolved &= purchased - uncertain
            diagnostics["uncertain_order_codes"] += len(uncertain)
            kind = "PURCHASE"
        else:
            kind = event.kind.value
        if not resolved or not denominator:
            diagnostics["events_without_positive_products"] += 1
            continue
        diagnostics["evidence_events"] += 1
        diagnostics[kind.lower() + "_evidence_events"] += 1
        active_days.add(event.when.date())
        age = (context.cutoff - event.when).total_seconds() / 86400
        strength = 2 ** (-age / dict(config.half_lives)[kind]) / denominator
        for code in sorted(resolved):
            bucket = (code, kind, event.when.date())
            buckets[bucket] = max(buckets.get(bucket, 0.), strength)
    channel_mass = defaultdict(float)
    for (code, kind, _), value in buckets.items():
        channel_mass[code, kind] += value
    item_mass = defaultdict(float)
    for (code, kind), value in channel_mass.items():
        saturation = config.item_channel_cap * -math.expm1(-value / config.item_channel_cap)
        item_mass[code] = max(item_mass[code], dict(config.action_weights)[kind] * saturation)
    total_mass = math.fsum(item_mass.values())
    effective = _effective(item_mass.values())
    diagnostics.update(distinct_items=len(item_mass), active_days=len(active_days))
    diagnostics["total_mass"] = total_mass
    diagnostics["effective_items"] = effective
    attributes = []
    for name in FEATURE_FIELDS:
        distribution = background.distributions.get(name, {})
        q = {v: (1 - config.uniform_mix) * p + config.uniform_mix / len(distribution)
             for v, p in distribution.items()}
        counts, masses = defaultdict(float), []
        for code, mass in item_mass.items():
            value = items[code].get(name)
            if value and value in q:
                counts[value] += mass
                masses.append(mass)
        support = math.fsum(masses)
        p = {v: (counts[v] + config.prior_mass * prior) / (support + config.prior_mass) for v, prior in q.items()}
        lifts = {v: max(-1., min(1., math.log(p[v] / prior) / config.max_log_lift)) for v, prior in q.items()}
        confidence = support / (support + config.prior_mass) * min(1., _effective(masses) / config.full_effective_items)
        attributes.append(AttributeInterest(name, support, confidence, _effective(masses),
                          MappingProxyType(q), MappingProxyType(p), MappingProxyType(lifts),
                          MappingProxyType(dict(counts))))
    if not item_mass:
        status = "EMPTY"
    elif background.reference_users < config.min_reference_users:
        status = "INSUFFICIENT_BACKGROUND"
    elif (len(item_mass) < config.min_items or diagnostics["evidence_events"] < config.min_events
          or total_mass < config.min_mass or effective < 2):
        status = "INSUFFICIENT_HISTORY"
    elif not any(dict(config.field_weights)[a.name] > 0 and a.mass > 0 for a in attributes):
        status = "NO_ATTRIBUTE_EVIDENCE"
    elif not any(dict(config.field_weights)[a.name] > 0 and a.confidence > 0 and
                 any(abs(value) > 1e-12 for value in a.lift.values()) for a in attributes):
        status = "NO_DISTINCT_PREFERENCE"
    else:
        status = "READY"
    warnings = (("SYNTHETIC_ONLY",) if context.dataset_kind == "SYNTHETIC" else
                ("INDEX_EXPLORATORY", "RETROSPECTIVE_KNOWLEDGE_NOT_FORECAST"))
    return InterestProfile(status, context.mode, tuple(attributes), MappingProxyType(dict(diagnostics)),
                            min(1., len(active_days) / 3), config, MappingProxyType(items), context.cutoff,
                            warnings + (("AS_OF_UNVERIFIED",) if context.mode == "EXPLORATORY" else ()), known_until)


def score_item(profile, item_code):
    """Explain a single frozen-catalog item. Does not select/filter/rank recommendations."""
    item = profile.catalog.get(item_code)
    if item is None:
        return ProfileMatch("UNKNOWN_ITEM", 0., 0., 0., (), profile.mode, profile.warnings, profile.cutoff.isoformat(),
                            profile.knowledge_cutoff.isoformat() if profile.knowledge_cutoff else None)
    weights = dict(profile.config.field_weights)
    denominator = math.fsum(weights.values())
    explanations, coverage, confidence = [], 0., 0.
    for attribute in profile.fields:
        beta = weights[attribute.name]
        value = item.get(attribute.name)
        known = bool(value) and value in attribute.lift
        contribution = (beta / denominator * attribute.confidence * profile.day_confidence * attribute.lift[value]) if known else 0.
        if known:
            coverage += beta / denominator
            confidence += beta / denominator * attribute.confidence * profile.day_confidence
        explanations.append({"attribute": attribute.name, "value": value or None, "weight": beta,
                             "support_mass": attribute.mass, "confidence": attribute.confidence,
                             "background": attribute.prior.get(value), "smoothed_share": attribute.posterior.get(value),
                             "relative_lift": attribute.lift.get(value), "contribution": contribution,
                             "reason": "KNOWN" if known else "MISSING_OR_OUT_OF_REFERENCE"})
    status = profile.status if profile.status != "READY" else (
        "READY" if coverage >= profile.config.min_candidate_coverage else "INSUFFICIENT_ITEM_ATTRIBUTES")
    score = math.fsum(row["contribution"] for row in explanations) if status == "READY" else 0.
    # Explain potential evidence separately; final contribution must sum to the applied score.
    if status != "READY":
        for row in explanations:
            row["contribution"] = 0.
    return ProfileMatch(status, score, confidence if status == "READY" else 0., coverage,
                         tuple(sorted(explanations, key=lambda row: (-abs(row["contribution"]), row["attribute"]))),
                         profile.mode, profile.warnings, profile.cutoff.isoformat(),
                         profile.knowledge_cutoff.isoformat() if profile.knowledge_cutoff else None)
