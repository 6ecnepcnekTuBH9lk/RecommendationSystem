"""Historical order-placement research contracts and bounded per-customer reduction.

This is NOT a point-in-time feature store: canonical identities are retrospective.
No current KPI, profile, BPR checkpoint, money or customer export is consumed.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from enum import Enum
from typing import Iterable

from Application import customer_history_index as index
from Application.model.customer_propensity_history import EventFact, PropensityDataError


DAY = 86_400_000_000
HORIZON = 30
SCHEMA_VERSION = 2
TARGET = "order_placed_within_30_days"
AS_OF_LEVEL = "RETROSPECTIVE_EVENT_TIME"
STRICT_AS_OF_LEVEL = "POINT_IN_TIME_VERIFIED"
ORDER_WINDOWS = (30, 90, 180, 365)
ACTION_WINDOWS = {"VIEW": (7, 30, 90, 180), "FAVORITE": (7, 30, 90)}
LIMITATIONS = (
    "Current canonical merge graph; historical identity availability is not recoverable.",
    "Source observed_us is export publication, not initial event/status availability.",
    "firstAction time is a research placement proxy; completed checkout versus drafts/technical orders is not verified.",
    "Actions use event time; missing initial availability remains unverified even when creation time is absent.",
    "Absence is conditional on declared export coverage; completeness specifically by firstAction time is not independently verified.",
    "Retrospective order features use event time regardless of later source publication; status/lines are not consumed.",
)


def utc_day(value: datetime) -> datetime:
    if (not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0)
            or any((value.hour, value.minute, value.second, value.microsecond))):
        raise ValueError("Prediction and split dates must be UTC midnight")
    return value.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def at(micros: int) -> datetime:
    return index._EPOCH + timedelta(microseconds=micros)


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


@dataclass(frozen=True, slots=True)
class PropensityDatasetConfig:
    start: datetime | None = None
    end: datetime | None = None  # Inclusive prediction date, NOT target end.
    step_days: int = 30
    validation_start: datetime | None = None
    test_start: datetime | None = None
    as_of_level: str = AS_OF_LEVEL
    max_slices: int = 256
    purpose: str = "training"

    def __post_init__(self):
        for value in (self.start, self.end, self.validation_start, self.test_start):
            if value is not None:
                utc_day(value)
        if (type(self.step_days) is not int or self.step_days < 1
                or type(self.max_slices) is not int or not 3 <= self.max_slices <= 1024):
            raise ValueError("Invalid bounded date grid")
        if self.start and self.end and self.start > self.end:
            raise ValueError("Invalid prediction interval")
        if (self.validation_start is None) != (self.test_start is None):
            raise ValueError("Supply both split boundaries")
        if self.as_of_level == STRICT_AS_OF_LEVEL:
            raise PropensityDataError("POINT_IN_TIME_IDENTITY_UNVERIFIED")
        if self.as_of_level != AS_OF_LEVEL or self.purpose not in ("training", "inference"):
            raise ValueError("Unsupported temporal mode or dataset purpose")
        if self.purpose == "inference" and any(v is not None for v in (
                self.start, self.end, self.validation_start, self.test_start)):
            raise ValueError("Inference date is determined exclusively by common coverage")


class PropensityLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN_LABEL"


FeatureValue = int | str | None  # Exact Decimal quantities are JSON strings.


@dataclass(frozen=True, slots=True, repr=False)
class PropensityInferenceRow:
    customer_key: str
    prediction_utc: str
    target_until_utc: str
    split: str
    features: dict[str, FeatureValue]
    available: dict[str, bool]
    feature_issues: dict[str, tuple[str, ...]]
    def payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True, repr=False)
class PropensityFeatureRow(PropensityInferenceRow):
    label_state: PropensityLabel
    target: int | None
    label_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PropensitySourceSpan:
    component: str
    since_us: int
    until_us: int
    observed_us: int


@dataclass(frozen=True, slots=True)
class PropensityPlan:
    dates: tuple[datetime, ...]
    validation_start: datetime
    test_start: datetime
    purpose: str = "training"

    def split(self, date: datetime) -> str:
        if self.purpose == "inference":
            return "inference"
        end = date + timedelta(days=HORIZON)
        if date < self.validation_start:
            return "train" if end <= self.validation_start else "embargo"
        if date < self.test_start:
            return "validation" if end <= self.test_start else "embargo"
        return "test"


def covered(coverage: tuple, component: str, low: int, high: int) -> bool:
    """Half-open union; gaps, including partial days, are never filled."""
    position = low
    for span in sorted((c for c in coverage if c.component == component), key=lambda c: c.since_utc):
        start, end = index._micros(span.since_utc), index._micros(span.until_utc)
        if end <= position:
            continue
        if start > position:
            return False
        position = max(position, end)
        if position >= high:
            return True
    return False


def make_plan(config: PropensityDatasetConfig, coverage: tuple) -> PropensityPlan:
    if config.purpose == "inference":
        date = latest_prediction_date(coverage)
        return PropensityPlan((date,), date, date, "inference")
    orders = [c for c in coverage if c.component == "orders"]
    if not orders:
        raise PropensityDataError("NO_ORDERS_COVERAGE")
    first = min(c.since_utc for c in orders)
    # Round UP, never invent a fully observed partial first day.
    midnight = first.replace(hour=0, minute=0, second=0, microsecond=0)
    first_day = midnight + (timedelta(days=1) if first != midnight else timedelta())
    start = config.start or first_day + timedelta(days=30)
    last = max(c.until_utc for c in orders) - timedelta(days=HORIZON)
    end = min(config.end or last, last)
    dates = []
    while start <= end:
        if len(dates) >= config.max_slices:
            raise PropensityDataError("TOO_MANY_SLICES")
        dates.append(start)
        start += timedelta(days=config.step_days)
    if len(dates) < 3:
        raise PropensityDataError("INSUFFICIENT_CHRONOLOGICAL_SLICES")
    validation = config.validation_start or dates[min(len(dates) - 2, max(1, len(dates) * 3 // 5))]
    test = config.test_start or dates[min(len(dates) - 1, max(2, len(dates) * 4 // 5))]
    if not dates[0] < validation < test <= dates[-1]:
        raise PropensityDataError("INVALID_SPLIT_BOUNDARIES")
    plan = PropensityPlan(tuple(dates), validation, test)
    if any(not any(plan.split(t) == split for t in dates) for split in ("train", "validation", "test")):
        raise PropensityDataError("EMPTY_SPLIT_AFTER_EMBARGO")
    return plan


def latest_prediction_date(coverage: tuple) -> datetime:
    """Latest shared complete UTC-day boundary, requiring all feature lookbacks.

    Intersect actual interval unions; max/min of component ends alone misses gaps.
    Do not silently fall back to an old forecast if the latest common run is short.
    """
    def union(component):
        result = []
        for span in sorted((c for c in coverage if c.component == component), key=lambda c: c.since_utc):
            low, high = index._micros(span.since_utc), index._micros(span.until_utc)
            if result and low <= result[-1][1]:
                result[-1] = (result[-1][0], max(high, result[-1][1]))
            else:
                result.append((low, high))
        return result

    orders, actions = union("orders"), union("actions")
    i = j = 0
    boundary = None
    while i < len(orders) and j < len(actions):
        low, high = max(orders[i][0], actions[j][0]), min(orders[i][1], actions[j][1])
        midnight = high // DAY * DAY
        if low < midnight <= high:
            boundary = midnight
        if orders[i][1] <= actions[j][1]:
            i += 1
        else:
            j += 1
    if boundary is None:
        raise PropensityDataError("NO_COMMON_COMPLETE_DAY")
    if not all(covered(coverage, component, boundary - days * DAY, boundary)
               for component, days in (("orders", 365), ("actions", 180))):
        raise PropensityDataError("INFERENCE_HISTORY_INCOMPLETE")
    return utc_day(at(boundary))


def feature_schema() -> dict[str, dict]:
    result = {}
    for days in ORDER_WINDOWS:
        result[f"placed_orders_{days}d"] = {"type": "nullable_integer", "source": "orders", "window_days": days}
    for days in (30, 90):
        result[f"order_active_days_{days}d"] = {"type": "nullable_integer", "source": "orders", "window_days": days}
    result["days_since_last_order_365d"] = {"type": "nullable_integer", "source": "orders", "window_days": 365,
                                           "empty": "null with available=true means no observed order within lookback"}
    result["has_order_365d"] = {"type": "nullable_integer", "source": "orders", "window_days": 365}
    result["average_days_between_orders_365d"] = {"type": "nullable_decimal_string", "source": "orders", "window_days": 365,
                                                "empty": "null with available=true means fewer than two orders"}
    for kind, windows in ACTION_WINDOWS.items():
        name = "views" if kind == "VIEW" else "favorites"
        for days in windows:
            result[f"{name}_{days}d"] = {"type": "nullable_integer", "source": "actions", "window_days": days}
        result[f"{name}_active_days_30d"] = {"type": "nullable_integer", "source": "actions", "window_days": 30}
    for name, component, days in (("view", "actions", 180), ("favorite", "actions", 90)):
        result[f"days_since_{name}_{days}d"] = {"type": "nullable_integer", "source": component, "window_days": days,
                                               "empty": "null with available=true means observed absence within lookback"}
        result[f"has_{name}_{days}d"] = {"type": "nullable_integer", "source": component, "window_days": days}
    return result


@dataclass(slots=True)
class _Slice:
    cutoff: int
    cohort: bool = False
    counts: Counter = field(default_factory=Counter)
    days: dict[str, set[int]] = field(default_factory=lambda: {k: set() for k in ACTION_WINDOWS})
    order_days: dict[int, set[int]] = field(default_factory=lambda: {30: set(), 90: set()})
    first_order: int | None = None
    last: dict[str, int] = field(default_factory=dict)
    issues: dict[str, set[str]] = field(default_factory=dict)
    positive: bool = False
    label_issues: set[str] = field(default_factory=set)

    def issue(self, names: Iterable[str], reason: str) -> None:
        for name in names:
            self.issues.setdefault(name, set()).add(reason)


def _possible(event: EventFact, low: int, high: int) -> bool:
    return any(t is None or low <= t < high for t in event.times)


def _order_names(days: int) -> tuple[str, ...]:
    return (f"placed_orders_{days}d",) + ((f"order_active_days_{days}d",) if days in (30, 90) else ()) + (
        ("days_since_last_order_365d", "has_order_365d", "average_days_between_orders_365d") if days == 365 else ())


def _action_names(kind: str, days: int) -> tuple[str, ...]:
    plural, singular = ("views", "view") if kind == "VIEW" else ("favorites", "favorite")
    return (f"{plural}_{days}d",) + ((f"{plural}_active_days_30d",) if days == 30 else ()) + (
        (f"days_since_{singular}_{days}d", f"has_{singular}_{days}d") if days == max(ACTION_WINDOWS[kind]) else ())


def _consume(state: _Slice, customer: str, event: EventFact, *, labeled: bool) -> None:
    t = state.cutoff
    stamp = event.stamp()
    owned = event.owned_by(customer)
    # B-level cohort: observed event-time interactions, current canonical identities.
    if event.identity_valid and owned and stamp is not None and stamp < t:
        state.cohort = True
    if event.source == "ORDER":
        if labeled and _possible(event, t, t + HORIZON * DAY):
            reason = ("INVALID_ORDER_IDENTITY" if not event.identity_valid else
                      "AMBIGUOUS_OWNER" if not owned else "UNRELIABLE_ORDER_TIME" if stamp is None
                      else "ORDER_PROVENANCE_MISSING" if any(v.observed is None for v in event.variants) else None)
            if reason:
                state.label_issues.add(reason)
            else:
                state.positive = True
        # Explicit schema-v2 retrospective placement semantics, never status semantics.
        for window in ORDER_WINDOWS:
            if not _possible(event, t - window * DAY, t):
                continue
            names = _order_names(window)
            reason = ("INVALID_ORDER_IDENTITY" if not event.identity_valid else
                      "AMBIGUOUS_OWNER" if not owned else "UNRELIABLE_ORDER_TIME" if stamp is None
                      else "ORDER_PROVENANCE_MISSING" if any(v.observed is None for v in event.variants) else None)
            if reason:
                state.issue(names, reason)
                continue
            state.counts[names[0]] += 1
            if window in state.order_days:
                state.order_days[window].add(stamp // DAY)
            if window == 365:
                state.first_order = stamp if state.first_order is None else min(state.first_order, stamp)
                state.last["order"] = max(state.last.get("order", stamp), stamp)
        return
    kinds = {v.kind for v in event.variants}
    for kind, windows in ACTION_WINDOWS.items():
        if kind not in kinds:
            continue
        for window in windows:
            if not _possible(event, t - window * DAY, t):
                continue
            reason = ("AMBIGUOUS_OWNER" if not owned else "UNRELIABLE_ACTION_TIME" if stamp is None
                      else "ACTION_KIND_CONFLICT" if kinds != {kind}
                      else "ACTION_NOT_CREATED_BY_CUTOFF" if any(v.created is not None and v.created > t for v in event.variants)
                      else None)
            if reason:
                state.issue(_action_names(kind, window), reason)
                continue
            plural, singular = ("views", "view") if kind == "VIEW" else ("favorites", "favorite")
            state.counts[f"{plural}_{window}d"] += 1
            state.last[singular] = max(state.last.get(singular, stamp), stamp)
            if window == 30:
                state.days[kind].add(stamp // DAY)


def customer_rows(customer: str, events: Iterable[EventFact], plan: PropensityPlan, coverage: tuple,
                  pseudonym: str, cancelled: Callable[[], bool] = lambda: False,
                  publication: tuple[PropensitySourceSpan, ...] = ()) -> Iterable[PropensityFeatureRow | PropensityInferenceRow]:
    """O(events * bounded grid), memory O(grid * fixed windows + one event).

    Active-day sets have <=30 entries per kind/slice. Never retains customer events.
    """
    from Application.model.customer_propensity_history import PropensityCancelled

    states = [_Slice(index._micros(date)) for date in plan.dates]
    for event in events:
        if cancelled():
            raise PropensityCancelled
        for state in states:
            _consume(state, customer, event, labeled=plan.purpose == "training")
    schema = feature_schema()
    for date, state in zip(plan.dates, states):
        if not state.cohort:
            continue
        t = state.cutoff
        features: dict[str, FeatureValue] = dict(state.counts)
        features.update({f"order_active_days_{w}d": len(days) for w, days in state.order_days.items()})
        last = state.last.get("order")
        features["has_order_365d"] = int(last is not None)
        features["days_since_last_order_365d"] = None if last is None else (t - last) // DAY
        n = state.counts["placed_orders_365d"]
        with localcontext() as context:
            context.prec = 64
            features["average_days_between_orders_365d"] = (
                decimal_text(Decimal(last - state.first_order) / Decimal((n - 1) * DAY)) if n >= 2 else None)
        features.update({"views_active_days_30d": len(state.days["VIEW"]),
                         "favorites_active_days_30d": len(state.days["FAVORITE"])})
        for name, days in (("view", 180), ("favorite", 90)):
            latest = state.last.get(name)
            features[f"days_since_{name}_{days}d"] = None if latest is None else (t - latest) // DAY
            features[f"has_{name}_{days}d"] = int(latest is not None)
        for name, spec in schema.items():
            features.setdefault(name, 0)
            if not covered(coverage, spec["source"], t - spec["window_days"] * DAY, t):
                state.issue((name,), "INCOMPLETE_FEATURE_COVERAGE")
            if name in state.issues:
                features[name] = None
        common = (pseudonym, iso(date), iso(date + timedelta(days=HORIZON)), plan.split(date),
                  {k: features[k] for k in sorted(schema)}, {k: k not in state.issues for k in sorted(schema)},
                  {k: tuple(sorted(v)) for k, v in sorted(state.issues.items())})
        if plan.purpose == "inference":
            yield PropensityInferenceRow(*common)
            continue
        reasons = set(state.label_issues)
        if not covered(coverage, "orders", t, t + HORIZON * DAY):
            reasons.add("INCOMPLETE_TARGET_COVERAGE")
            target = None  # Even a positive is withheld when target coverage is incomplete.
        else:
            target = 1 if state.positive else None if reasons else 0
            if target == 1:
                reasons.clear()  # Another uncertain order cannot negate an established existence.
        label = PropensityLabel.UNKNOWN if target is None else PropensityLabel.POSITIVE if target else PropensityLabel.NEGATIVE
        yield PropensityFeatureRow(*common, label, target, tuple(sorted(reasons)))
