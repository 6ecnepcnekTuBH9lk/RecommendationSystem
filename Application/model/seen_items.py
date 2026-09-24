"""Source-neutral, immutable CSR index of all known training-event pairs."""

from dataclasses import dataclass, field
import numpy as np

if __package__:
    from .training_data import validate_prepared_data
else:
    from training_data import validate_prepared_data


class SeenItemsError(ValueError):
    """Invalid seen-index contract; messages contain no identities."""


@dataclass(frozen=True, eq=False)
class SeenItemsIndex:
    num_users: int
    num_items: int
    indptr: np.ndarray = field(repr=False)
    indices: np.ndarray = field(repr=False)

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.num_users, self.num_items)):
            raise SeenItemsError("Seen dimensions must be non-negative integers")
        for name in ("indptr", "indices"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype("int64") or value.ndim != 1:
                raise SeenItemsError("Seen arrays must be one-dimensional int64")
            # bytes backing prevents callers from re-enabling NumPy writes.
            object.__setattr__(self, name, np.frombuffer(value.tobytes(), dtype=np.int64))
        p, i = self.indptr, self.indices
        if (len(p) != self.num_users + 1 or p[0] != 0 or p[-1] != len(i)
                or np.any(p < 0) or np.any(p > len(i)) or np.any(p[1:] < p[:-1])):
            raise SeenItemsError("Invalid seen offsets")
        if np.any(i < 0) or np.any(i >= self.num_items):
            raise SeenItemsError("Seen item index out of bounds")
        for user in range(self.num_users):
            segment = self.items_for_user(user)
            if np.any(segment[1:] <= segment[:-1]):
                raise SeenItemsError("Seen segments must be sorted and unique")

    def items_for_user(self, u_idx):
        if isinstance(u_idx, (bool, np.bool_)) or not isinstance(u_idx, (int, np.integer)) or not 0 <= u_idx < self.num_users:
            raise SeenItemsError("Seen user index out of bounds")
        return self.indices[self.indptr[u_idx]:self.indptr[u_idx + 1]]

    def __repr__(self):
        return (f"SeenItemsIndex(num_users={self.num_users}, num_items={self.num_items}, "
                f"total_seen_pairs={len(self.indices)}, users_with_seen_items={np.count_nonzero(np.diff(self.indptr))})")


def build_seen_items_index(prepared_data):
    validate_prepared_data(prepared_data)
    splits = prepared_data.splits
    users = [set(items) for items in splits.user_pos_train]
    for user, item in zip(splits.eval_users, splits.eval_items):
        users[int(user)].add(int(item))
    indptr = [0]
    indices = []
    for items in users:
        indices.extend(sorted(items))
        indptr.append(len(indices))
    return SeenItemsIndex(len(prepared_data.mappings.idx2user), len(prepared_data.mappings.idx2item),
                          np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int64))


def seen_items_from_checkpoint(checkpoint):
    keys = ("seen_items_indptr", "seen_items_indices")
    if not any(key in checkpoint for key in keys):
        return None  # Old artifact only; corrupt new contracts never fall back.
    if not all(key in checkpoint for key in keys):
        raise SeenItemsError("Incomplete checkpoint seen fields")
    return SeenItemsIndex(checkpoint["num_users"], checkpoint["num_items"],
                          checkpoint[keys[0]], checkpoint[keys[1]])
