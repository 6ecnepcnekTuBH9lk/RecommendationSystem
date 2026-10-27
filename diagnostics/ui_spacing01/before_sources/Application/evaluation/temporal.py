"""Shared next-novel-item benchmark from already resolved canonical interactions."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType

from Application.interactions import InteractionRecord, InteractionType
from Application.product_resolution import ResolvedInteraction


class TemporalProtocolError(ValueError):
    """Safe protocol error without customer, product or raw event values."""


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TemporalProtocolError("Temporal timestamps must be timezone-aware")
    try:
        return value.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise TemporalProtocolError("Invalid temporal timestamp") from None


@dataclass(frozen=True)
class TemporalConfig:
    validation_start: datetime
    test_start: datetime
    test_end: datetime | None = None
    min_history_events: int = 10

    def __post_init__(self):
        for name in ("validation_start", "test_start", "test_end"):
            value = getattr(self, name)
            if name != "test_end" or value is not None:
                object.__setattr__(self, name, _utc(value))
        if not self.validation_start < self.test_start or (
            self.test_end is not None and not self.test_start < self.test_end
        ):
            raise TemporalProtocolError("Temporal cutoffs must be strictly increasing")
        if type(self.min_history_events) is not int or self.min_history_events < 1:
            raise TemporalProtocolError("Minimum history must be a positive event count")


@dataclass(frozen=True, repr=False)
class TemporalCase:
    customer_id: str
    target_item: str
    target_timestamp: datetime
    interaction_type: InteractionType
    seen_items: frozenset[str]

    def __post_init__(self):
        object.__setattr__(self, "target_timestamp", _utc(self.target_timestamp))
        object.__setattr__(self, "seen_items", frozenset(self.seen_items))


@dataclass(frozen=True)
class TemporalDiagnostics:
    training_events: int
    training_users: int
    training_items: int
    future_events: int
    future_users: int
    users_passing_history_threshold: int
    eligible_warm_user_targets: int
    excluded_cold_users: int
    excluded_sparse_users: int
    excluded_cold_item_targets: int
    no_novel_target_users: int
    evaluation_targets: int
    view_targets: int
    favorite_targets: int
    purchase_targets: int
    history_event_count_distribution: tuple[tuple[int, int], ...]
    mean_target_delay_seconds: float | None


@dataclass(frozen=True, repr=False)
class TemporalSnapshot:
    cutoff: datetime
    future_end: datetime | None
    history: tuple[ResolvedInteraction, ...]
    cases: tuple[TemporalCase, ...]
    item_universe: tuple[str, ...]
    seen_at_cutoff: Mapping[str, frozenset[str]] = field(repr=False)
    diagnostics: TemporalDiagnostics

    def __post_init__(self):
        object.__setattr__(self, "cutoff", _utc(self.cutoff))
        if self.future_end is not None:
            object.__setattr__(self, "future_end", _utc(self.future_end))
            if self.future_end <= self.cutoff:
                raise TemporalProtocolError("Snapshot window must end after cutoff")
        # Shared consumers cannot mutate history, cases, item universe or the seen mapping.
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "cases", tuple(self.cases))
        object.__setattr__(self, "item_universe", tuple(self.item_universe))
        object.__setattr__(self, "seen_at_cutoff", MappingProxyType(
            {user: frozenset(items) for user, items in self.seen_at_cutoff.items()}))
        if any(event.interaction.event_datetime_utc >= self.cutoff for event in self.history):
            raise TemporalProtocolError("Snapshot history must precede prediction cutoff")
        if any(case.target_timestamp < self.cutoff or (
            self.future_end is not None and case.target_timestamp >= self.future_end
        ) for case in self.cases):
            raise TemporalProtocolError("Targets must belong to the snapshot future window")

    def candidates_for(self, case: TemporalCase) -> tuple[str, ...]:
        return tuple(item for item in self.item_universe if item not in case.seen_items)


@dataclass(frozen=True, repr=False)
class TemporalEvaluationProtocol:
    config: TemporalConfig
    validation: TemporalSnapshot
    test: TemporalSnapshot


def _snapshot(events, cutoff, future_end, min_history):
    history = tuple(e for e in events if e.interaction.event_datetime_utc < cutoff)
    future = tuple(e for e in events if e.interaction.event_datetime_utc >= cutoff
                   and (future_end is None or e.interaction.event_datetime_utc < future_end))
    counts = Counter(e.interaction.customer_id for e in history)
    seen = defaultdict(set)
    for event in history:
        seen[event.interaction.customer_id].add(event.item_id)
    items = tuple(sorted({e.item_id for e in history}))
    item_set = set(items)
    future_users = {e.interaction.customer_id for e in future}
    # The history is fixed at cutoff: repeated future events never update it.
    first_novel = {}
    for event in future:
        user = event.interaction.customer_id
        if user not in first_novel and event.item_id not in seen.get(user, set()):
            first_novel[user] = event
    cold = sparse = passing = novel = cold_item = no_novel = 0
    cases = []
    for user in sorted(future_users):
        if counts[user] == 0:
            cold += 1
        elif counts[user] < min_history:
            sparse += 1
        else:
            passing += 1
            event = first_novel.get(user)
            if event is None:
                no_novel += 1
            else:
                novel += 1
                if event.item_id not in item_set:
                    # Never replace a first cold novel target with a later scoreable target.
                    cold_item += 1
                else:
                    cases.append(TemporalCase(user, event.item_id, event.interaction.event_datetime_utc,
                                              event.interaction.interaction_type, frozenset(seen[user])))
    types = Counter(case.interaction_type for case in cases)
    delays = [(case.target_timestamp - cutoff).total_seconds() for case in cases]
    diagnostics = TemporalDiagnostics(
        len(history), len(counts), len(items), len(future), len(future_users), passing, novel,
        cold, sparse, cold_item, no_novel, len(cases), types[InteractionType.VIEW],
        types[InteractionType.FAVORITE], types[InteractionType.PURCHASE],
        tuple(sorted(Counter(counts.values()).items())), sum(delays) / len(delays) if delays else None,
    )
    return TemporalSnapshot(cutoff, future_end, history, tuple(cases), items, seen, diagnostics)


def build_temporal_protocol(
    events: Iterable[ResolvedInteraction], config: TemporalConfig,
) -> TemporalEvaluationProtocol:
    """Input must already use CustomerIdResolver and ProductResolver identities.

    Exact ties follow canonical type order PURCHASE, FAVORITE, VIEW and source order
    within type. Scoring backends receive only the selected snapshot/cases.
    """
    if not isinstance(config, TemporalConfig):
        raise TemporalProtocolError("Expected TemporalConfig")
    normalized = []
    priority = {InteractionType.PURCHASE: 0, InteractionType.FAVORITE: 1, InteractionType.VIEW: 2}
    for event in events:
        if not isinstance(event, ResolvedInteraction) or not isinstance(event.interaction, InteractionRecord):
            raise TemporalProtocolError("Expected canonical resolved interactions")
        record = event.interaction
        if any(not isinstance(value, str) or not value or value != value.strip()
               for value in (record.customer_id, event.item_id)):
            raise TemporalProtocolError("Expected canonical nonempty customer and item identities")
        if not isinstance(record.interaction_type, InteractionType):
            raise TemporalProtocolError("Invalid temporal interaction type")
        normalized.append(replace(event, interaction=replace(record, event_datetime_utc=_utc(record.event_datetime_utc))))
    # Python's stable sort preserves source order for exact timestamp/type ties.
    normalized.sort(key=lambda e: (e.interaction.event_datetime_utc, priority[e.interaction.interaction_type]))
    return TemporalEvaluationProtocol(
        config, _snapshot(normalized, config.validation_start, config.test_start, config.min_history_events),
        _snapshot(normalized, config.test_start, config.test_end, config.min_history_events),
    )
