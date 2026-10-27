"""Immutable analytical selection; no Qt, persistence or product identity rules."""

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from functools import cached_property


def values_sorted(values):
    return tuple(sorted(set(values), key=lambda v: (v is None, (v or "").casefold(), v or "")))


@dataclass(frozen=True)
class AnalysisFilter:
    # ISO calendar dates keep asdict/worker/cache payloads JSON-safe.
    start_date: str | None = None
    end_date: str | None = None
    nomenclature_types: tuple[str | None, ...] | None = None
    collections: tuple[str | None, ...] | None = None

    def __post_init__(self):
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("Укажите обе даты периода.")
        if self.start_date is not None:
            for value in (self.start_date, self.end_date):
                if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                    raise ValueError("Некорректная дата отбора.")
            if self.start_date > self.end_date or self.end_date == "9999-12-31":
                raise ValueError("Некорректный диапазон дат.")
        for field in ("nomenclature_types", "collections"):
            values = getattr(self, field)
            if values is None:
                continue
            if not isinstance(values, (tuple, list)) or not values or any(
                    value is not None and not isinstance(value, str) for value in values):
                raise ValueError("Выберите хотя бы одно значение в каждом списке.")
            object.__setattr__(self, field, values_sorted((v.strip() or None) if v is not None else None for v in values))

    @property
    def active(self):
        return self.start_date is not None or self.product_restricted

    @property
    def product_restricted(self):
        return self.nomenclature_types is not None or self.collections is not None

    @cached_property
    def interval(self):
        if self.start_date is None:
            return None
        start = datetime.combine(date.fromisoformat(self.start_date), time(), timezone.utc)
        end = datetime.combine(date.fromisoformat(self.end_date) + timedelta(days=1), time(), timezone.utc)
        return start, end

    def contains(self, timestamp):
        return self.interval is None or self.interval[0] <= timestamp < self.interval[1]

    def matches(self, metadata):
        if not self.product_restricted:
            return True
        return metadata is not None and all(selected is None or value in selected for selected, value in (
            (self.nomenclature_types, metadata.nomenclature_type), (self.collections, metadata.collection)))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Некорректный формат отбора.")
        result = cls(**payload)
        # Persistence/worker boundaries require already canonical values.
        if any(payload[k] is not None and tuple(payload[k]) != getattr(result, k)
               for k in ("nomenclature_types", "collections")):
            raise ValueError("Отбор не нормализован.")
        return result

    def summary(self):
        parts = []
        if self.start_date is not None:
            parts.append("период " + "–".join(date.fromisoformat(v).strftime("%d.%m.%Y")
                                            for v in (self.start_date, self.end_date)))
        for label, values in (("вид номенклатуры", self.nomenclature_types), ("сезон", self.collections)):
            if values is not None:
                text = ", ".join(v if v is not None else "Не указано" for v in values) if len(values) <= 3 else f"{len(values)} значений"
                parts.append(f"{label}: {text}")
        return "Отбор → " + "; ".join(parts) if parts else "Отбор не установлен"


@dataclass(frozen=True)
class AnalysisOptions:
    start_date: str
    end_date: str
    nomenclature_types: tuple[str | None, ...]
    collections: tuple[str | None, ...]

    def normalize(self, selection, *, reconcile=False):
        changes = {}
        for field in ("nomenclature_types", "collections"):
            selected, available = getattr(selection, field), getattr(self, field)
            if selected is not None:
                if reconcile:
                    selected = tuple(v for v in selected if v in available)
                changes[field] = None if (not selected or set(selected) == set(available)) else selected
        start, end = selection.start_date, selection.end_date
        if start is not None:
            if reconcile:
                start = min(max(start, self.start_date), self.end_date)
                end = min(max(end, self.start_date), self.end_date)
            if (start, end) == (self.start_date, self.end_date):
                start = end = None
        return replace(selection, start_date=start, end_date=end, **changes)


def options_from_snapshot(metadata, intervals):
    if not intervals:
        raise ValueError("Нет общего доступного периода статистики.")
    return AnalysisOptions(min(a for a, _ in intervals).date().isoformat(),
                           (max(b for _, b in intervals) - timedelta(microseconds=1)).date().isoformat(),
                           values_sorted(m.nomenclature_type for m in metadata.values()),
                           values_sorted(m.collection for m in metadata.values()))
