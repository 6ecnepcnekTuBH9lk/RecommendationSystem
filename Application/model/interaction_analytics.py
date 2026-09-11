"""Immutable mapping-index aggregates; no event history in the artifact."""

from dataclasses import dataclass, field
import math
import numpy as np


class AnalyticsError(ValueError):
    """Safe source-neutral contract error."""


def normalize_kind(value):
    return " ".join(str(value or "").replace("\ufeff", "").replace("\xa0", " ").split()).replace("–", "-").replace("—", "-")


@dataclass(frozen=True, eq=False)
class InteractionAnalytics:
    num_users: int
    num_items: int
    viewers: np.ndarray = field(repr=False)
    converted: np.ndarray = field(repr=False)
    purchase_activity: np.ndarray = field(repr=False)
    favorite_activity: np.ndarray = field(repr=False)
    view_activity: np.ndarray = field(repr=False)
    item_kinds: tuple[str, ...] = field(repr=False)
    window_days: int = 30
    prior_strength: float = 20.0

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.num_users, self.num_items, self.window_days)):
            raise AnalyticsError("Invalid analytics dimensions/window")
        if type(self.prior_strength) not in (int, float) or not math.isfinite(self.prior_strength) or self.prior_strength <= 0:
            raise AnalyticsError("Invalid analytics prior")
        for name in ("viewers", "converted", "purchase_activity", "favorite_activity", "view_activity"):
            a = getattr(self, name)
            count = name in ("viewers", "converted")
            dtype = np.dtype("int64" if count else "float64")
            size = self.num_items if count else self.num_users
            if (not isinstance(a, np.ndarray) or a.dtype != dtype or a.shape != (size,)
                    or not np.isfinite(a).all() or np.any(a < 0)):
                raise AnalyticsError("Invalid analytics array shape/type/range")
            object.__setattr__(self, name, np.frombuffer(a.tobytes(), dtype=dtype))
        if np.any(self.converted > self.viewers) or np.any(self.viewers > self.num_users):
            raise AnalyticsError("Invalid unique conversion counts")
        for name in ("favorite_activity", "view_activity"):
            if np.any(getattr(self, name) % 1 != 0):
                raise AnalyticsError("Activity counts must be integral")
        if not isinstance(self.item_kinds, (tuple, list)):
            raise AnalyticsError("Invalid analytics category container")
        kinds = tuple(self.item_kinds)
        if len(kinds) != self.num_items or any(not isinstance(k, str) or k != normalize_kind(k) for k in kinds):
            raise AnalyticsError("Invalid analytics category metadata")
        object.__setattr__(self, "item_kinds", kinds)

    def conversion(self, idx2item, current_kinds):
        """Return percentages using training kind priors and current-item fallback."""
        total = int(self.viewers.sum())
        global_rate = float(self.converted.sum()) / total if total else 0.0
        counts = {}
        for kind, v, c in zip(self.item_kinds, self.viewers, self.converted):
            if kind and v:
                old_v, old_c = counts.get(kind, (0, 0))
                counts[kind] = (old_v + int(v), old_c + int(c))
        rates = {k: c / v for k, (v, c) in counts.items()}
        result = {}
        for code, kind, v, c in zip(idx2item, self.item_kinds, self.viewers, self.converted):
            if v:
                prior = rates.get(kind, global_rate)
                result[code] = round(100 * (c + self.prior_strength * prior) / (v + self.prior_strength), 4)
            else:
                fallback_kind = normalize_kind(current_kinds.get(code, kind))
                result[code] = round(100 * rates.get(fallback_kind, global_rate), 4)
        for code, kind in current_kinds.items():
            result.setdefault(code, round(100 * rates.get(normalize_kind(kind), global_rate), 4))
        return result, round(100 * global_rate, 4)

    def rank_users(self, cfg, eligible_mask=None):
        p, f, v = self.purchase_activity, self.favorite_activity, self.view_activity
        score = p * cfg.w_purchase + f * cfg.w_favorite + v * cfg.w_view_item
        eligible = (p + f + v) > 0
        if eligible_mask is not None:
            eligible &= np.asarray(eligible_mask, dtype=bool)
        indices = np.flatnonzero(eligible)
        return indices[np.lexsort((-v[indices], -f[indices], -p[indices], -score[indices]))].astype(int).tolist()

    def to_checkpoint(self):
        return {"version": 1, **{name: getattr(self, name) for name in self.__dataclass_fields__}}


def analytics_from_checkpoint(checkpoint):
    if "interaction_analytics" not in checkpoint:
        return None
    block = checkpoint["interaction_analytics"]
    if (not isinstance(block, dict) or set(block) != {"version", *InteractionAnalytics.__dataclass_fields__}
            or type(block["version"]) is not int or block["version"] != 1):
        raise AnalyticsError("Invalid analytics checkpoint schema")
    result = InteractionAnalytics(**{k: v for k, v in block.items() if k != "version"})
    if (result.num_users, result.num_items) != (checkpoint["num_users"], checkpoint["num_items"]):
        raise AnalyticsError("Analytics dimensions differ from model")
    return result


class AnalyticsCollector:
    """One pass, order independent. State scales with unique pairs, not occurrences.

    Purchases contribute merged inclusive intervals of possible first-view days.
    For a yearly horizon and window=30, disjoint intervals per pair are bounded
    by roughly 12. This also handles orders arriving before views.
    """
    def __init__(self, window_days=30):
        if type(window_days) is not int or window_days < 0:
            raise AnalyticsError("Invalid conversion window")
        self.window_days = window_days
        self.first_views = {}
        self.purchase_windows = {}
        self.activity = {}

    def add(self, user, item, kind, timestamp, quantity=None):
        kind = getattr(kind, "value", kind)
        if kind not in ("VIEW", "FAVORITE", "PURCHASE"):
            raise AnalyticsError("Unsupported analytics interaction type")
        values = self.activity.setdefault(user, [0., 0., 0.])
        if kind == "PURCHASE":
            q = 1.0 if quantity is None else float(quantity)
            if not math.isfinite(q):
                raise AnalyticsError("Invalid purchase activity quantity")
            values[0] += min(10., max(1., q))
        else:
            values[1 if kind == "FAVORITE" else 2] += 1
        if timestamp is None:
            return  # Legacy undated events count as activity, not conversion.
        day = timestamp.date().toordinal()
        pair = (user, item)
        if kind == "VIEW":
            self.first_views[pair] = min(day, self.first_views.get(pair, day))
        elif kind == "PURCHASE":
            intervals = self.purchase_windows.setdefault(pair, [])
            intervals.append((day - self.window_days, day))
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            self.purchase_windows[pair] = merged

    def finalize(self, mappings, item_kinds):
        viewers = np.zeros(len(mappings.idx2item), dtype=np.int64)
        converted = viewers.copy()
        activity = np.zeros((3, len(mappings.idx2user)), dtype=np.float64)
        for user, values in self.activity.items():
            activity[:, mappings.user2idx[user]] = values
        for (user, item), day in self.first_views.items():
            i = mappings.item2idx[item]
            viewers[i] += 1
            if any(start <= day <= end for start, end in self.purchase_windows.get((user, item), ())):
                converted[i] += 1
        return InteractionAnalytics(len(mappings.idx2user), len(mappings.idx2item), viewers, converted, *activity,
            tuple(normalize_kind(item_kinds.get(i, "")) for i in mappings.idx2item), self.window_days)


def load_catalog_kinds(path):
    from pathlib import Path
    import pandas as pd
    if not Path(path).is_file():
        return {}
    frame = pd.read_csv(path, sep="|", encoding="utf-8-sig", dtype=str).fillna("")
    if not {"КодНоменклатуры", "ВидНоменклатуры"}.issubset(frame.columns):
        return {}
    frame = frame.drop_duplicates("КодНоменклатуры", keep="last")
    return dict(zip(frame["КодНоменклатуры"], frame["ВидНоменклатуры"]))
