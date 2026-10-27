"""Read-only bounded indexed-history research. No merge parsing or model lookup.

Canonical IDs originate ONLY in the already-normalized event_customers index.
No public report contains IDs, event keys or individual histories/profiles.
"""

from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import random
import sqlite3
from time import monotonic

from Application import customer_history_index as index
from Application import customer_history_service as history
from Application.customer_history_records import HistoryKind, HistorySource
from Application.evaluation.experiments.customer_interest import (
    Background, InterestAction, InterestConfig, InterestLine, InterestOrder, InterestSnapshot,
    build_profile, score_item,
)
from Application.interactions import InteractionType
from Application.model.item_feature_snapshot import FEATURE_FIELDS
from Application.purchased_products import assess_order, classify


class PilotError(ValueError):
    """Only static, non-personal diagnostic categories may cross the runner boundary."""


@dataclass(frozen=True)
class Bounds:
    customers: int = 100
    events: int = 10_000  # Includes page-header lookahead, conservatively counted again on next page.
    events_per_customer: int = 200
    page_size: int = 25
    variants_per_event: int = 8
    origins_per_event: int = 32
    products_per_variant: int = 200
    max_catalog_items: int = 10_000
    seconds: int = 180
    private_bytes: int = 1536 * 2**20
    output_bytes: int = 4 * 2**20
    disk_floor_bytes: int = 20 * 2**30

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise PilotError("INVALID_BOUNDS")
        maximum = (100, 10_000, 200, 25, 8, 32, 200, 10_000, 180, 1536 * 2**20, 4 * 2**20, 20 * 2**30)
        if any(v > cap for v, cap in zip(vars(self).values(), maximum)) or self.disk_floor_bytes < 20 * 2**30:
            raise PilotError("BOUNDS_CANNOT_BE_WEAKENED")


class Budget:
    def __init__(self, bounds=Bounds(), *, poll=lambda: None, cancelled=lambda: False, clock=monotonic):
        self.bounds, self.poll, self.cancelled, self.clock = bounds, poll, cancelled, clock
        self.started, self.customers, self.events = clock(), 0, 0

    def check(self):
        if self.cancelled():
            raise PilotError("CANCELLED")
        if self.clock() - self.started >= self.bounds.seconds:
            raise PilotError("TIME_LIMIT")
        self.poll()

    def consume_events(self, count):
        self.check()
        if type(count) is not int or count < 0 or self.events + count > self.bounds.events:
            raise PilotError("EVENT_LIMIT")
        self.events += count

    def consume_customer(self):
        self.check()
        if self.customers >= self.bounds.customers:
            raise PilotError("CUSTOMER_LIMIT")
        self.customers += 1


def require_ready(status):
    if status.status is not index.HistoryIndexStatus.READY:
        raise PilotError("INDEX_" + status.status.value)


SEEK_SQL = ("SELECT canonical_customer_id FROM event_customers INDEXED BY event_customers_customer "
            "WHERE canonical_customer_id>=? ORDER BY canonical_customer_id LIMIT 1")


@contextmanager
def open_reader(service, raw_root, budget):
    """One validated snapshot; reuse backend cursor SQL and typed deserialization."""
    raw_root = Path(raw_root).resolve()
    ready = service.get_status(raw_root=raw_root, verify_integrity=False)
    require_ready(ready)
    with index._reader_lock(raw_root):
        captured = index._capture_sources(raw_root)
        path = service.index_path(raw_root=raw_root)
        before = index._database_version(path)
        with index._read_index(path) as connection:
            connection.execute("PRAGMA temp_store=MEMORY")  # Local read handle; no spill files.
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 2**20)
            def progress():
                try:
                    budget.check()
                    return 0
                except Exception:
                    return 1
            connection.set_progress_handler(progress, 5000)
            same = index._inspect_connection(connection, captured)
            if same.generation != ready.generation or same.source_version != ready.source_version:
                raise PilotError("INDEX_CHANGED")
            connection.row_factory = sqlite3.Row
            yield connection, ready
        if before != index._database_version(path) or index._capture_sources(raw_root).version != captured.version:
            raise PilotError("SOURCES_CHANGED")


def select_customers(connection, budget, *, seed=20261026, anchors=None):
    """Deterministic random lexicographic seeks: gap-weighted, NOT uniform users.

    No OFFSET, COUNT, random() sort or history scan. Duplicate seeks do not add
    customers. Non-numeric canonical scopes require explicit synthetic anchors.
    """
    rng, seen = random.Random(seed), set()
    anchors = (str(rng.randint(100_000, 999_999)) for _ in range(800)) if anchors is None else anchors
    for anchor in anchors:
        budget.check()
        row = connection.execute(SEEK_SQL, (anchor,)).fetchone()
        if row is None:
            continue
        customer = row[0]
        if customer in seen:
            continue
        if budget.customers == budget.bounds.customers:
            break
        budget.consume_customer()
        seen.add(customer)
        yield customer


def _payload_fits(connection, row, bounds):
    if row["variant_count"] > bounds.variants_per_event or row["occurrence_count"] > bounds.origins_per_event:
        return False
    variants = connection.execute("SELECT variant_key FROM variants WHERE event_key=? LIMIT ?",
                                  (row["event_key"], bounds.variants_per_event + 1)).fetchall()
    if len(variants) != row["variant_count"]:
        raise PilotError("VARIANT_COUNT_DISAGREES")
    for variant in variants:
        origins = connection.execute("SELECT 1 FROM record_origins WHERE variant_key=? LIMIT ?",
                                     (variant[0], bounds.origins_per_event + 1)).fetchall()
        if not origins or len(origins) > bounds.origins_per_event:
            return False
        for table in ("order_lines", "product_occurrences", "product_keys"):
            cap = bounds.products_per_variant * (8 if table == "product_keys" else 1)
            # Table names are fixed constants, never source strings.
            query = f"SELECT 1 FROM {table} WHERE variant_key=? LIMIT ?"
            if len(connection.execute(query, (variant[0], cap + 1)).fetchall()) > cap:
                return False
    return True


def read_customer(connection, ready, customer, since, until, budget):
    """Exhaust the bounded date window or return explicit unusable partial history.

    Missing/conflicting common timestamps remain included by the backend query;
    they are later excluded with diagnostics, never silently dated or assigned.
    """
    bounds, events, last, pages, partial = budget.bounds, [], None, 0, 0
    request = history._request(customer, None, since, until, bounds.page_size)
    if not history._coverage_complete(ready.coverage, request):
        return None, {"INCOMPLETE_SOURCE_COVERAGE": 1}
    while True:
        budget.check()
        if budget.events + bounds.page_size + 1 > bounds.events:
            return None, {"EVENT_BUDGET_EXHAUSTED": 1, "pages": pages}
        sql, args = history._page_query(request, last)
        rows = connection.execute(sql, args).fetchall()
        budget.consume_events(len(rows))
        pages += 1
        if len(events) + min(len(rows), bounds.page_size) > bounds.events_per_customer:
            return None, {"CUSTOMER_HISTORY_LIMIT": 1, "pages": pages}
        for row in rows[:bounds.page_size]:
            budget.check()
            if not _payload_fits(connection, row, bounds):
                return None, {"EVENT_PAYLOAD_LIMIT": 1, "pages": pages}
            events.append(history._logical_event(connection, row))
        partial += int(any(event.has_conflicts for event in events[-bounds.page_size:]))
        if len(rows) <= bounds.page_size:
            return tuple(events), {"pages": pages, "partial_conflict_pages": partial}
        if len(events) == bounds.events_per_customer:
            return None, {"CUSTOMER_HISTORY_LIMIT": 1, "pages": pages}
        # The same opaque generation/request-bound cursor as the public API.
        cursor = history._encode_cursor(ready.generation, request, rows[bounds.page_size - 1])
        decoded = history._decode_cursor(cursor)
        if decoded[1] != ready.generation:
            raise PilotError("STALE_CURSOR")
        last = decoded[3:]


def convert_events(customer, events, resolver):
    """Preserve all order versions, Decimal, earliest known publication per variant."""
    result, diagnostics = [], Counter()
    for logical in events:
        if (logical.owner_conflict or logical.canonical_customer_id != customer
                or logical.candidate_customer_ids != (customer,)):
            diagnostics["OWNER_CONFLICT"] += 1
            continue
        if logical.time_conflict or logical.kind_conflict or logical.event_datetime_utc is None:
            diagnostics["TIME_OR_KIND_CONFLICT"] += 1
            continue
        if logical.kind is HistoryKind.UNCLASSIFIED:
            diagnostics["UNCLASSIFIED"] += 1
            continue
        if not logical.variants or any(not v.origins or any(o.observed_at_utc is None for o in v.origins) for v in logical.variants):
            diagnostics["MISSING_PUBLICATION"] += 1
            continue
        if logical.source is HistorySource.ACTION:
            if len(logical.variants) != 1:
                diagnostics["ACTION_VARIANTS_CONFLICT"] += 1
                continue
            detail = logical.variants[0].event.details
            if not detail.products or any(o.reference.product is None for o in detail.products):
                diagnostics["MISSING_ACTION_PRODUCT"] += 1
                continue
            products = tuple(o.reference.product for o in detail.products)
            if logical.kind not in (HistoryKind.VIEW, HistoryKind.FAVORITE):
                diagnostics["UNSUPPORTED_ACTION_KIND"] += 1
                continue
            result.append(InterestAction(logical.event_key, customer, logical.event_datetime_utc,
                                         InteractionType(logical.kind.value), products,
                                         min(o.observed_at_utc for o in logical.variants[0].origins)))
        else:
            snapshots, rows = [], []
            for variant in logical.variants:
                lines = variant.event.details.lines
                snapshots.append(InterestSnapshot(tuple(InterestLine(line.reference.product, line.raw_status,
                    line.quantity, line.source_line_key) for line in lines), min(o.observed_at_utc for o in variant.origins)))
                rows.append([{"raw_status": line.raw_status, "quantity": str(line.quantity),
                              "source_line_key": line.source_line_key, "ordinal": line.ordinal,
                              "selected_namespace": line.reference.product.namespace if line.reference.product else None,
                              "selected_value": line.reference.product.value if line.reference.product else None} for line in lines])
                for line in lines:
                    diagnostics["STATUS_" + classify(line.raw_status)] += 1
            bought, doubtful, _, cancelled, conflict, unresolved = assess_order(rows, resolver)
            diagnostics["ORDER_CANCELLED_LINES"] += cancelled
            diagnostics["ORDER_CONFLICTS"] += int(conflict)
            diagnostics["ORDER_UNCERTAIN_CODES"] += len(doubtful)
            diagnostics["ORDER_UNRESOLVED"] += int(unresolved)
            diagnostics["ORDER_WITH_POSITIVE_CODES"] += int(bool(bought - doubtful) and not unresolved)
            result.append(InterestOrder(logical.event_key, customer, logical.event_datetime_utc, tuple(snapshots)))
        diagnostics["ADAPTED_" + logical.kind.value] += 1
    return tuple(result), diagnostics


def feature_configs():
    base = InterestConfig(max_events=200)
    values = {"KIND": {"ВидНоменклатуры": 1.}, "ASSORTMENT": {"ВидАссортимента": 1.},
              "BOTH": {"ВидНоменклатуры": .5, "ВидАссортимента": .3},
              "WITH_CATEGORY": {"ВидНоменклатуры": .5, "ВидАссортимента": .3, "КатегорияНаСайте": .2},
              "CATEGORY": {"КатегорияНаСайте": 1.}}
    return {name: replace(base, field_weights=tuple((f, weights.get(f, 0.)) for f in FEATURE_FIELDS))
            for name, weights in values.items()}


def reference_background(profiles, cutoff, *, minimum=20):
    """Each reference user has unit mass per supported field; no power-user vote."""
    counts, support = defaultdict(Counter), Counter()
    for profile in profiles:
        for field in profile.fields:
            if field.mass > 0:
                support[field.name] += 1
                for value, mass in field.counts.items():
                    counts[field.name][value] += mass / field.mass
    distributions = {f: {v: n / support[f] for v, n in values.items()} for f, values in counts.items()
                     if support[f] >= minimum}
    return Background(distributions, len(profiles), cutoff, False), dict(support)


def describe(values):
    """Aggregated measurements only; no individual vectors in persisted output."""
    values = sorted(float(v) for v in values)
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    def quantile(p):
        pos = p * (len(values) - 1)
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return {"n": len(values), "mean": math.fsum(values) / len(values), "median": quantile(.5),
            "p95": quantile(.95), "min": values[0], "max": values[-1]}


def evaluate_profiles(cohort, catalog, context, budget, *, panel_codes, reference_size=24):
    """Disjoint holdout plus reference leave-one-out; no future targets/quality claims."""
    if context.mode != "EXPLORATORY":
        raise PilotError("PILOT_IS_EXPLORATORY_ONLY")
    if (len(cohort) > budget.bounds.customers or len({row[0] for row in cohort}) != len(cohort)
            or any(not 0 <= len(row[1]) <= row[2] <= budget.bounds.events_per_customer for row in cohort)):
        raise PilotError("INVALID_COHORT_SCOPE")
    if (not isinstance(panel_codes, tuple) or not 1 <= len(panel_codes) <= 256
            or len(set(panel_codes)) != len(panel_codes)):
        raise PilotError("INVALID_DIAGNOSTIC_PANEL")
    vocabulary = {f: sorted({item.get(f) for item in catalog if item.get(f)}) for f in FEATURE_FIELDS}
    uniform = Background({f: {v: 1 / len(vs) for v in vs} for f, vs in vocabulary.items() if vs},
                         0, context.cutoff, False)
    configs = feature_configs()
    initial = []
    for customer, events, count, partial in cohort:
        budget.check()
        profile = build_profile(customer, events, catalog, uniform, context, configs["WITH_CATEGORY"])
        partial |= bool(profile.diagnostics.get("unresolved_product_slots") or profile.diagnostics.get("uncertain_order_codes"))
        initial.append((customer, events, count, partial, profile))
    references = [row for row in initial if row[4].diagnostics["total_mass"] > 0][:reference_size]
    reference_ids = {row[0] for row in references}
    background, support = reference_background([row[4] for row in references], context.cutoff)
    collected, statuses, score_counts, field_lifts = defaultdict(list), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    stability, diagnostics = [], Counter()
    partial_users, heldout_users, loo_users = 0, 0, 0
    for customer, events, count, partial, _ in initial:
        if customer in reference_ids:
            # Small sparse cohorts can have every positive-history customer in
            # reference. Evaluate those users descriptively with their entire
            # contribution removed from q; never reuse their own population vote.
            local_background, _ = reference_background([row[4] for row in references if row[0] != customer], context.cutoff)
            loo_users += 1
            role = "REFERENCE_LOO"
        else:
            local_background = background
            heldout_users += 1
            role = "HELDOUT"
        partial_users += int(partial)
        group = "0" if count == 0 else "1-5" if count <= 5 else "6-30" if count <= 30 else "31-200"
        scores = {}
        full = None
        for name, config in configs.items():
            budget.check()
            profile = build_profile(customer, events, catalog, local_background, context, config)
            matches = [score_item(profile, code) for code in panel_codes]
            scores[name] = [m.score for m in matches]
            for scope in (name, name + "/" + group, name + "/" + role):
                statuses[scope][profile.status] += 1
                for m in matches:
                    score_counts[scope]["positive" if m.score > 1e-12 else "negative" if m.score < -1e-12 else "neutral"] += 1
                collected[scope + "/mean_abs_score"].append(math.fsum(abs(m.score) for m in matches) / len(matches))
                collected[scope + "/attribute_coverage"].append(math.fsum(m.attribute_coverage for m in matches) / len(matches))
            if name == "WITH_CATEGORY":
                full = profile
                diagnostics.update(profile.diagnostics)
                for field in profile.fields:
                    if dict(config.field_weights)[field.name]:
                        for lift in field.lift.values():
                            field_lifts[field.name]["positive" if lift > 1e-12 else "negative" if lift < -1e-12 else "neutral"] += 1
        collected["joint_abs_minus_max_single"].extend(abs(c) - max(abs(a), abs(b))
                                                      for a, b, c in zip(scores["KIND"], scores["ASSORTMENT"], scores["BOTH"]))
        collected["joint_abs_exceeds_max_single"].extend(float(abs(c) > max(abs(a), abs(b)) + 1e-12)
                                                       for a, b, c in zip(scores["KIND"], scores["ASSORTMENT"], scores["BOTH"]))
        paired = [(a, b) for a, b in zip(scores["KIND"], scores["ASSORTMENT"]) if abs(a) > 1e-12 and abs(b) > 1e-12]
        if paired:
            collected["family_sign_disagreement"].append(sum(a * b < 0 for a, b in paired) / len(paired))
        a, b = scores["KIND"], scores["ASSORTMENT"]
        ma, mb = math.fsum(a) / len(a), math.fsum(b) / len(b)
        va, vb = math.fsum((x - ma)**2 for x in a), math.fsum((x - mb)**2 for x in b)
        if va > 1e-24 and vb > 1e-24:
            collected["family_score_pearson"].append(math.fsum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb))
        collected["category_absolute_score_delta"].extend(abs(a - b) for a, b in zip(scores["WITH_CATEGORY"], scores["BOTH"]))
        if full.status == "READY":
            for omitted in sorted(events, key=lambda e: hashlib.sha256(e.event_id.encode()).digest())[:3]:
                budget.check()
                other = build_profile(customer, [e for e in events if e is not omitted], catalog, local_background,
                                      context, configs["WITH_CATEGORY"])
                changed = [score_item(other, code).score for code in panel_codes]
                stability.append(math.fsum(abs(a - b) for a, b in zip(scores["WITH_CATEGORY"], changed)) / len(changed))
                collected["leave_one_out_neutral_fallback"].append(float(other.status != "READY"))
    return {"reference_users": len(references), "field_reference_support": support,
            "supported_background_fields": sorted(background.distributions), "evaluated_users": len(initial),
            "heldout_evaluation_users": heldout_users, "reference_loo_evaluation_users": loo_users,
            "partially_used_evaluation_users": partial_users, "statuses": {k: dict(v) for k, v in statuses.items()},
            "score_signs": {k: dict(v) for k, v in score_counts.items()}, "interest_signs": {k: dict(v) for k, v in field_lifts.items()},
            "measurements": {k: describe(v) for k, v in collected.items()}, "leave_one_out_mean_abs_delta": describe(stability),
            "profile_diagnostics": dict(diagnostics), "panel_size": len(panel_codes), "quality_metrics": None}
