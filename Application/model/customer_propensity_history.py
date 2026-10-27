"""Streaming history facts for retrospective research; no current KPI lookup."""

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import groupby
import sqlite3

from Application.customer_activity_service import _add
from Application.customer_history_normalization import history_event_key
from Application.customer_history_records import HistorySource
from Application.purchased_products import classify


class PropensityDataError(ValueError):
    """Static, non-personal diagnostic category."""


class PropensityCancelled(Exception):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class LineFact:
    identity: str | None
    ordinal: int
    status: str
    quantity: str
    product: str | None


@dataclass(frozen=True, slots=True, repr=False)
class VariantFact:
    key: str
    owner: str
    kind: str
    stamp: int | None
    observed: int | None
    created: int | None
    lines: tuple[LineFact, ...]


@dataclass(frozen=True, slots=True, repr=False)
class EventFact:
    source: str
    variants: tuple[VariantFact, ...]
    identity_valid: bool = True

    @property
    def times(self) -> frozenset[int | None]:
        return frozenset(v.stamp for v in self.variants)

    def stamp(self) -> int | None:
        times = self.times
        return next(iter(times)) if len(times) == 1 and None not in times else None

    def owned_by(self, customer: str) -> bool:
        return {v.owner for v in self.variants} == {customer}


@dataclass(frozen=True, slots=True)
class OrderOutcome:
    purchase: bool | None
    units: Decimal | None
    reason: str | None = None


def order_outcome(variants: Sequence[VariantFact]) -> OrderOutcome:
    """Legacy schema-v1 purchase consensus, retained for regression checks only.

    The schema-v2 placement builder never calls this helper or reads order lines.
    """
    if not variants:
        return OrderOutcome(None, None, "STATUS_NOT_OBSERVED_BY_CUTOFF")
    groups = defaultdict(list)
    for n, variant in enumerate(variants):
        for line in variant.lines:
            groups[line.identity or (n, line.ordinal)].append((n, line))
    if not groups:
        return OrderOutcome(None, None, "MISSING_ORDER_LINES")
    bought, uncertain, units_unknown = False, False, False
    units = Decimal(0)
    for identity, rows in groups.items():
        states = {classify(line.status) for _, line in rows}
        if states <= {"CANCELLED", "RETURN"}:
            continue  # Neither is a new purchase, even when quantities differ.
        codes = {line.product for _, line in rows}
        stable = (len(rows) == len(variants) and len({n for n, _ in rows}) == len(variants)
                  and (len(variants) == 1 or (isinstance(identity, str) and len(codes) == 1 and None not in codes)))
        quantities = set()
        for _, line in rows:
            if classify(line.status) == "PURCHASE":
                try:
                    qty = Decimal(line.quantity)
                    quantities.add(qty if qty.is_finite() and qty > 0 else None)
                except (InvalidOperation, TypeError):
                    quantities.add(None)
        if not stable or not states <= {"PURCHASE", "RETURN"} or not quantities or None in quantities:
            uncertain = True
            continue
        bought = True
        if len(quantities) != 1:
            units_unknown = True
        else:
            units = _add(units, next(iter(quantities)))
    if bought:
        return OrderOutcome(True, None if uncertain or units_unknown else units,
                            "UNCERTAIN_ORDER_UNITS" if uncertain or units_unknown else None)
    if uncertain:
        return OrderOutcome(None, None, "UNCERTAIN_ORDER_STATUS_OR_SNAPSHOT")
    return OrderOutcome(False, Decimal(0))


# Only order identity/time/owner headers are needed for schema v2 propensity.
# Deliberately do not read status, quantity, price or product lines.
FACTS_SQL = """
SELECT ec.canonical_customer_id AS customer,e.event_key,e.source,e.variant_count,
       v.variant_key,v.canonical_customer_id AS owner,v.kind,v.event_us,
       s.order_us,s.snapshot_key,a.creation_us,e.source_event_id,o.order_id
FROM event_customers ec JOIN events e USING(event_key) JOIN variants v USING(event_key)
LEFT JOIN order_snapshots s USING(variant_key) LEFT JOIN actions a USING(variant_key)
LEFT JOIN orders o USING(event_key)
ORDER BY ec.canonical_customer_id,e.event_key,v.variant_key
"""
ORIGINS_SQL = """
SELECT ec.canonical_customer_id AS customer,v.event_key,v.variant_key,MIN(s.observed_us) AS observed
FROM event_customers ec JOIN variants v USING(event_key)
LEFT JOIN record_origins r USING(variant_key) LEFT JOIN sources s USING(source_key)
GROUP BY ec.canonical_customer_id,v.event_key,v.variant_key
ORDER BY ec.canonical_customer_id,v.event_key,v.variant_key
"""


def iter_facts(connection: sqlite3.Connection, cancelled: Callable[[], bool], *,
               before_us: int | None = None,
               customer_ids: tuple[str, ...] | None = None) -> Iterator[tuple[str, EventFact]]:
    """Yield one logical event; memory depends on one event, not all customers."""
    # For inference do not fetch reliable future events. NULL-consensus events
    # retain all variants, so contradictory timestamps cannot masquerade as clean history.
    conditions, parameters = [], []
    if customer_ids is not None:
        if not customer_ids or len(customer_ids) + 1 > connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER):
            raise PropensityDataError("INVALID_PROBE_CUSTOMER_SCOPE")
        conditions.append("ec.canonical_customer_id IN (" + ",".join("?" for _ in customer_ids) + ")")
        parameters.extend(customer_ids)
    if before_us is not None:
        conditions.append("(ec.event_us < ? OR ec.event_us IS NULL)")
        parameters.append(before_us)
    where = "WHERE " + " AND ".join(conditions) + " " if conditions else ""
    origins_sql = ORIGINS_SQL.replace("GROUP BY", where + "GROUP BY", 1)
    facts_sql = FACTS_SQL.replace("ORDER BY", where + "ORDER BY", 1)
    origins = groupby(connection.execute(origins_sql, parameters), lambda r: (r["customer"], r["event_key"]))
    for key, rows in groupby(connection.execute(facts_sql, parameters), lambda r: (r["customer"], r["event_key"])):
        if cancelled():
            raise PropensityCancelled
        origin_key, origin_rows = next(origins, (None, ()))
        if origin_key != key:
            raise PropensityDataError("HISTORY_ORIGINS_DISAGREE")
        observed = {r["variant_key"]: r["observed"] for r in origin_rows}
        variants = []
        valid_identity = True
        source = expected = None
        for head in rows:
            if cancelled():
                raise PropensityCancelled
            variant_key = head["variant_key"]
            source, expected = head["source"], head["variant_count"]
            stamp = head["event_us"]
            if source == "ORDER":
                identifier = head["source_event_id"]
                valid_identity &= bool(isinstance(identifier, str) and identifier.strip() == identifier and identifier
                                       and all(c.isprintable() for c in identifier)
                                       and head["order_id"] == identifier and head["kind"] == "ORDER"
                                       and head["snapshot_key"] == variant_key
                                       and history_event_key(HistorySource.ORDER, identifier) == key[1])
                if head["order_us"] != stamp:
                    stamp = None
            variants.append(VariantFact(variant_key, head["owner"], head["kind"], stamp,
                                        observed.get(variant_key), head["creation_us"], ()))
        if source not in ("ORDER", "ACTION") or len(variants) != expected:
            raise PropensityDataError("HISTORY_VARIANTS_DISAGREE")
        yield key[0], EventFact(source, tuple(variants), valid_identity)
    if next(origins, None) is not None:
        raise PropensityDataError("HISTORY_ORIGINS_DISAGREE")
