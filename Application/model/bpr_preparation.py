"""Independent event-level BPR preparation with characterized legacy ordering."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math

import numpy as np
import pandas as pd

from Application.interactions import InteractionType
from Application.product_resolution import ResolvedInteraction
from Application.model.training_data import Mappings as BprMappings, Splits as BprSplits, PreparedBprData


class BprPreparationError(ValueError):
    """Safe validation error, without event values."""


class DateMode(str, Enum):
    LEGACY_DATE = "LEGACY_DATE"
    FULL_TIMESTAMP = "FULL_TIMESTAMP"


@dataclass(frozen=True)
class BprWeightConfig:
    view_weight: float = 0.1
    favorite_weight: float = 2.0
    purchase_weight: float = 10.0
    min_purchase_quantity: float = 1.0
    max_purchase_quantity: float = 10.0

    def __post_init__(self):
        for value in vars(self).values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise BprPreparationError("Weight config requires finite positive numbers")
        if self.min_purchase_quantity > self.max_purchase_quantity:
            raise BprPreparationError("Invalid quantity bounds")


@dataclass(frozen=True)
class BprPreparationConfig:
    weights: BprWeightConfig = field(default_factory=BprWeightConfig)
    min_user_interactions_for_eval: int = 10
    date_mode: DateMode = DateMode.LEGACY_DATE

    def __post_init__(self):
        if not isinstance(self.weights, BprWeightConfig) or not isinstance(self.date_mode, DateMode):
            raise BprPreparationError("Invalid preparation config")
        if (type(self.min_user_interactions_for_eval) is not int
                or self.min_user_interactions_for_eval < 1):
            raise BprPreparationError("Evaluation threshold must be a positive integer")


@dataclass(frozen=True, repr=False)
class BprEvent:
    customer_id: str
    item_id: str
    timestamp: datetime
    interaction_type: InteractionType
    weight: float

    def __post_init__(self):
        if any(not isinstance(value, str) or not value or value != value.strip()
               for value in (self.customer_id, self.item_id)):
            raise BprPreparationError("Event requires canonical nonempty IDs")
        if (not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None
                or self.timestamp.utcoffset() is None):
            raise BprPreparationError("Event requires timezone-aware timestamp")
        if not isinstance(self.interaction_type, InteractionType):
            raise BprPreparationError("Invalid interaction type")
        if not isinstance(self.weight, (int, float)) or not math.isfinite(self.weight) or self.weight <= 0:
            raise BprPreparationError("Event requires finite positive weight")


def to_bpr_event(resolved: ResolvedInteraction, weights: BprWeightConfig = BprWeightConfig()) -> BprEvent:
    event = resolved.interaction
    if event.interaction_type is InteractionType.VIEW:
        weight = weights.view_weight
    elif event.interaction_type is InteractionType.FAVORITE:
        weight = weights.favorite_weight
    elif event.interaction_type is InteractionType.PURCHASE:
        # Same coercion/fill/float/clip order as legacy; no quantity rounding.
        qty = pd.to_numeric(pd.Series([event.quantity]), errors="coerce").fillna(1).astype(float)
        weight = weights.purchase_weight * qty.clip(weights.min_purchase_quantity, weights.max_purchase_quantity).iloc[0]
    else:
        raise BprPreparationError("Invalid interaction type")
    return BprEvent(event.customer_id, resolved.item_id, event.event_datetime_utc,
                    event.interaction_type, float(weight))


@dataclass(frozen=True)
class BprDiagnostics:
    events_total: int
    view_events: int
    favorite_events: int
    purchase_events: int
    unique_users: int
    unique_items: int
    eligible_eval_users: int
    train_events_before_aggregation: int
    train_pairs_after_aggregation: int
    eval_events: int
    total_train_weight: float


@dataclass(repr=False)
class BprPreparation(PreparedBprData):
    diagnostics: BprDiagnostics


def prepare_bpr(events: Iterable[BprEvent], config: BprPreparationConfig = BprPreparationConfig()) -> BprPreparation:
    # Preserve source row order within each type, even for interleaved input.
    grouped = {kind: [] for kind in InteractionType}
    for event in events:
        grouped[event.interaction_type].append(event)
    purchase, favorite, view = (grouped[kind] for kind in
                                (InteractionType.PURCHASE, InteractionType.FAVORITE, InteractionType.VIEW))
    users = list(dict.fromkeys(event.customer_id for group in (purchase, view, favorite) for event in group))
    items = list(dict.fromkeys(event.item_id for group in (purchase, favorite, view) for event in group))
    maps = BprMappings({user: index for index, user in enumerate(users)}, users,
                       {item: index for index, item in enumerate(items)}, items)
    ordered = purchase + favorite + view
    rows = []
    for row_id, event in enumerate(ordered):
        timestamp = event.timestamp.astimezone(timezone.utc)
        if config.date_mode is DateMode.LEGACY_DATE:
            timestamp = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        rows.append((maps.user2idx[event.customer_id], maps.item2idx[event.item_id], timestamp, event.weight, row_id))
    frame = pd.DataFrame(rows, columns=["u_idx", "i_idx", "ts", "w", "_row_id"])
    # Explicit dtypes also make an empty input produce [0,2] pairs and float weights.
    frame = frame.astype({"u_idx": "int64", "i_idx": "int64", "w": "float64", "_row_id": "int64"})
    ordered_frame = frame.sort_values(["u_idx", "ts", "_row_id"], kind="mergesort")
    counts = ordered_frame.groupby("u_idx").size()
    eligible = counts[counts >= config.min_user_interactions_for_eval].index
    last = ordered_frame[ordered_frame["ts"].notna()].groupby("u_idx").tail(1)
    last = last[last["u_idx"].isin(eligible)]
    train = ordered_frame.drop(index=last.index)
    # Use the existing pandas sum semantics, including float accumulation precision.
    aggregated = train.groupby(["u_idx", "i_idx"], as_index=False)["w"].sum()
    pairs = aggregated[["u_idx", "i_idx"]].to_numpy(dtype=np.int64)
    train_weights = aggregated["w"].to_numpy(dtype=float)
    positives = [set() for _ in users]
    for user, item in pairs:
        positives[int(user)].add(int(item))
    splits = BprSplits(pairs, train_weights, last["u_idx"].to_numpy(dtype=np.int64),
                       last["i_idx"].to_numpy(dtype=np.int64), positives)
    types = Counter(event.interaction_type for event in ordered)
    diagnostics = BprDiagnostics(len(ordered), types[InteractionType.VIEW], types[InteractionType.FAVORITE],
                                 types[InteractionType.PURCHASE], len(users), len(items), len(eligible),
                                 len(train), len(pairs), len(last), float(train_weights.sum()))
    return BprPreparation(maps, splits, diagnostics)
