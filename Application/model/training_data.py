"""Shared in-memory contracts. Controlled mutability, not immutable snapshots.

Containers/arrays are retained by reference for legacy compatibility. The owner
must not mutate them during training; consumers validate immediately before use.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .interaction_analytics import InteractionAnalytics


@dataclass(repr=False)
class Mappings:
    user2idx: dict[str, int]
    idx2user: list[str]
    item2idx: dict[str, int]
    idx2item: list[str]


@dataclass(repr=False)
class Splits:
    train_pairs: np.ndarray
    train_weights: np.ndarray
    eval_users: np.ndarray
    eval_items: np.ndarray
    user_pos_train: list[set[int]]


@dataclass(repr=False)
class PreparedBprData:
    mappings: Mappings
    splits: Splits
    analytics: "InteractionAnalytics | None" = field(default=None, kw_only=True, repr=False)


class PreparedDataError(ValueError):
    """Safe structural validation error, without IDs or input values."""


def validate_prepared_data(data: PreparedBprData) -> None:
    if not isinstance(data, PreparedBprData) or not isinstance(data.mappings, Mappings) or not isinstance(data.splits, Splits):
        raise PreparedDataError("Expected PreparedBprData with shared mappings/splits")
    maps, splits = data.mappings, data.splits
    for forward, reverse in ((maps.user2idx, maps.idx2user), (maps.item2idx, maps.idx2item)):
        if not isinstance(forward, dict) or not isinstance(reverse, list):
            raise PreparedDataError("Invalid mapping containers")
        if any(not isinstance(value, str) for value in reverse):
            raise PreparedDataError("Mapping IDs must be strings")
        if len(set(reverse)) != len(reverse) or len(forward) != len(reverse):
            raise PreparedDataError("Inconsistent mapping cardinality")
        if any(not isinstance(forward.get(value), (int, np.integer))
               or isinstance(forward.get(value), (bool, np.bool_)) or forward[value] != index
               for index, value in enumerate(reverse)):
            raise PreparedDataError("Mappings must be inverse contiguous indexes")
    users, items = len(maps.idx2user), len(maps.idx2item)
    pairs, weights = splits.train_pairs, splits.train_weights
    if not isinstance(pairs, np.ndarray) or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise PreparedDataError("train_pairs must have shape Nx2")
    if not isinstance(weights, np.ndarray) or weights.ndim != 1 or len(weights) != len(pairs):
        raise PreparedDataError("train_weights must have length N")
    if weights.dtype.kind not in "fiu" or not np.isfinite(weights).all() or (weights < 0).any():
        raise PreparedDataError("Weights must be finite and non-negative")

    def indexes(values, bound):
        if (not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype.kind not in "iu"
                or (values < 0).any() or (values >= bound).any()):
            raise PreparedDataError("Invalid user/item indexes")

    indexes(pairs[:, 0], users)
    indexes(pairs[:, 1], items)
    indexes(splits.eval_users, users)
    indexes(splits.eval_items, items)
    if len(splits.eval_users) != len(splits.eval_items):
        raise PreparedDataError("Evaluation arrays must have equal length")
    if not isinstance(splits.user_pos_train, list) or len(splits.user_pos_train) != users:
        raise PreparedDataError("user_pos_train must have one set per user")
    expected = [set() for _ in range(users)]
    for user, item in pairs:
        expected[int(user)].add(int(item))
    for actual, wanted in zip(splits.user_pos_train, expected):
        if (not isinstance(actual, set) or any(not isinstance(item, (int, np.integer))
                or isinstance(item, (bool, np.bool_)) for item in actual) or actual != wanted):
            raise PreparedDataError("user_pos_train must match train pairs")
    if len(pairs) == 0:
        raise PreparedDataError("Training set is empty")
