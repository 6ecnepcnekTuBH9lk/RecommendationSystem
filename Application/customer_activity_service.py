"""Read-only, per-canonical-customer aggregates over one history snapshot.

Gross purchases use the agreed CP/delivering/F policy. This is neither a net
return balance nor a historical feature store for training an as-of model.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
from itertools import groupby
from pathlib import Path
import sqlite3
from typing import Callable

from Application import customer_history_index as index
from Application.customer_history_service import _coverage_complete, _request
from Application.mindbox.records import ProductKey
from Application.order_statistics import CURRENCIES, _currency
from Application.product_resolution import ResolutionStatus, normalize_product_key
from Application.purchased_products import classify


class ActivityStatus(str, Enum):
    READY = "READY"
    EMPTY = "EMPTY"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CurrencyActivity:
    currency: str
    orders: int
    amount: Decimal
    average_check: Decimal


@dataclass(frozen=True, slots=True)
class ActivityIssue:
    metric: str
    reason: str


@dataclass(frozen=True, slots=True, repr=False)
class CustomerActivity:
    status: ActivityStatus
    canonical_customer_id: str | None = None
    orders: int | None = None
    units: Decimal | None = None
    money: tuple[CurrencyActivity, ...] | None = None
    views: int | None = None
    favorites: int | None = None
    last_purchase_utc: datetime | None = None
    days_since_purchase: int | None = None
    unique_viewed_products: int | None = None
    coverage: tuple[index.HistoryCoverage, ...] = ()
    calculated_at_utc: datetime | None = None
    purchase_as_of_utc: datetime | None = None
    generation: str | None = None
    source_version: str | None = None
    orders_coverage_complete: bool = False
    actions_coverage_complete: bool = False
    issues: tuple[ActivityIssue, ...] = ()
    diagnostics: tuple[tuple[str, int], ...] = ()

    @property
    def complete(self):
        return self.status in (ActivityStatus.READY, ActivityStatus.EMPTY) and not self.issues


_ORDER_METRICS = ("orders", "units", "money", "recency")
_ACTION_METRICS = ("views", "favorites", "unique_viewed_products")

# Constant query count, indexed customer selection, one logical event in memory.
_ORDERS_SQL = """
SELECT e.event_key,e.owner_conflict,e.time_conflict,e.kind_conflict,
       v.variant_key,v.kind,v.event_us,s.order_us,
       l.source_line_key,l.ordinal,l.quantity,l.raw_status,l.price_of_line,
       p.selected_namespace,p.selected_value
FROM event_customers ec JOIN events e USING(event_key)
JOIN variants v USING(event_key)
LEFT JOIN order_snapshots s USING(variant_key)
LEFT JOIN order_lines l USING(variant_key)
LEFT JOIN product_occurrences p ON p.variant_key=l.variant_key AND p.role='line' AND p.ordinal=l.ordinal
WHERE ec.canonical_customer_id=? AND e.source='ORDER'
ORDER BY e.event_key,v.variant_key,l.ordinal
"""
_ACTIONS_SQL = """
SELECT e.event_key,e.owner_conflict,e.time_conflict,e.kind_conflict,
       v.variant_key,v.kind,v.event_us,p.ordinal,p.selected_namespace,p.selected_value
FROM event_customers ec JOIN events e USING(event_key)
JOIN variants v USING(event_key)
LEFT JOIN product_occurrences p ON p.variant_key=v.variant_key AND p.role='product'
WHERE ec.canonical_customer_id=? AND e.source='ACTION'
ORDER BY e.event_key,v.variant_key,p.ordinal
"""


def _product(namespace, value):
    key = ProductKey(namespace, value)
    result = normalize_product_key(key)
    if (result.status is not ResolutionStatus.RESOLVED or result.item_id is None or len(result.item_id) != 6
            or any(c.isspace() or not c.isprintable() for c in result.item_id)
            or any(not c.isprintable() for c in value)):
        return None
    return result.item_id


def _decimal(value, *, positive=False):
    if not isinstance(value, str):
        return None  # SQLite stores exact decimal text; never accept binary floats.
    try:
        amount = Decimal(value)
        if amount.is_finite() and (amount > 0 if positive else amount >= 0):
            return amount
    except InvalidOperation:
        pass
    return None


def _add(a, b):
    # Decimal addition is context-sensitive too. Preserve long source amounts
    # exactly; only a non-terminating average needs finite division precision.
    digits = max(a.adjusted(), b.adjusted()) - min(a.as_tuple().exponent, b.as_tuple().exponent) + 2
    with localcontext() as context:
        context.prec = max(64, digits)
        return a + b


def _mean(amount, count):
    with localcontext() as context:
        context.prec = max(64, len(amount.as_tuple().digits) + 4 * len(str(count)) + 16)
        return amount / count


def _order_product(namespace, value):
    code = _product(namespace, value)
    if code is not None:
        return ("normalized", code)
    # An opaque, unchanged unsupported key can align order lines without
    # guessing a catalog code. Its currency still remains unknown.
    if (namespace is not None and isinstance(value, str) and value and value == value.strip()
            and all(c.isprintable() for c in value)
            and normalize_product_key(ProductKey(namespace, value)).status is ResolutionStatus.UNSUPPORTED_NAMESPACE):
        return ("opaque", namespace, value)
    return None


def _period(coverage, component):
    spans = [c for c in coverage if c.component == component]
    if not spans:
        return None
    request = _request("coverage", ("ORDER",) if component == "orders" else ("VIEW",), None, None, 1)
    if not _coverage_complete(coverage, request):
        return None
    return min(c.since_utc for c in spans), max(c.until_utc for c in spans)


class _Aggregate:
    def __init__(self, customer, ready, now, cancelled):
        self.customer, self.ready, self.now = customer, ready, now
        self.cancelled = cancelled
        self.order_period = _period(ready.coverage, "orders")
        self.action_period = _period(ready.coverage, "actions")
        self.as_of = min(now, self.order_period[1] - timedelta(microseconds=1)) if self.order_period else None
        self.issues: set[tuple[str, str]] = set()
        self.counts = Counter()
        self.orders = self.views = self.favorites = 0
        self.units = Decimal(0)
        self.amounts = {c: Decimal(0) for c in CURRENCIES}
        self.currency_orders = Counter()
        self.last = None
        self.products = set()
        if not self.order_period:
            self.bad(_ORDER_METRICS, "coverage")
        if not self.action_period:
            self.bad(_ACTION_METRICS, "coverage")

    def bad(self, metrics, reason):
        self.issues.update((metric, reason) for metric in metrics)

    def valid_time(self, stamp, period):
        if type(stamp) is not int or period is None:
            return False
        try:
            dt = index._EPOCH + timedelta(microseconds=stamp)
        except OverflowError:
            return False
        return period[0] <= dt < period[1] and dt <= self.now

    def order(self, rows):
        self.counts["logical_orders"] += 1
        snapshots = {}
        groups = defaultdict(list)
        for row in rows:
            if self.cancelled():
                raise index._Cancelled
            self.counts["order_rows_read"] += 1
            snapshots[row["variant_key"]] = row
            if row["ordinal"] is not None:
                identity = row["source_line_key"] or (row["variant_key"], row["ordinal"])
                groups[identity].append(row)
        head = next(iter(snapshots.values()))
        if head["owner_conflict"] or head["kind_conflict"] or any(r["kind"] != "ORDER" for r in snapshots.values()):
            self.bad(_ORDER_METRICS, "order_conflict")
            return
        if not groups:
            self.bad(_ORDER_METRICS, "missing_order_lines")
            return
        purchased = False
        possible = False
        order_amount = Decimal(0)
        namespaces = set()
        for identity, lines in groups.items():
            states = {classify(r["raw_status"]) for r in lines}
            if states == {"CANCELLED"}:
                self.counts["cancelled_lines"] += len(lines)
                continue
            codes = {_order_product(r["selected_namespace"], r["selected_value"]) for r in lines}
            stable = (len(lines) == len(snapshots) and len({r["variant_key"] for r in lines}) == len(snapshots)
                      and (len(snapshots) == 1 or (isinstance(identity, str) and len(codes) == 1 and None not in codes)))
            buy = [r for r in lines if classify(r["raw_status"]) == "PURCHASE"]
            quantities = {_decimal(r["quantity"], positive=True) for r in buy}
            if not stable or not buy or not states <= {"PURCHASE", "RETURN"} or None in quantities:
                # Returns alone cannot establish a new purchase. Other unknown
                # outcomes can affect totals; never silently drop such lines.
                if stable and states == {"RETURN"}:
                    self.counts["return_only_lines"] += len(lines)
                    continue
                possible = True
                self.bad(("units", "money"), "order_conflict")
                self.counts["uncertain_line_groups"] += 1
                continue
            purchased = True
            if len(quantities) == 1:
                self.units = _add(self.units, next(iter(quantities)))
            else:
                self.bad(("units",), "quantity_conflict")
            prices = {_decimal(r["price_of_line"]) for r in buy}
            if len(prices) == 1 and None not in prices:
                order_amount = _add(order_amount, next(iter(prices)))
            else:
                self.bad(("money",), "amount_conflict")
            # Same product code across namespaces need not imply same currency.
            namespaces.update(r["selected_namespace"] for r in buy)
        if purchased:
            self.orders += 1
            currency = _currency(namespaces)
            if currency in CURRENCIES:
                self.amounts[currency] = _add(self.amounts[currency], order_amount)
                self.currency_orders[currency] += 1
            else:
                self.bad(("money",), "currency")
                self.counts["unassigned_currency_orders"] += 1
            dates = {r["order_us"] for r in snapshots.values()}
            if not all(self.valid_time(t, self.order_period) for t in dates):
                self.bad(_ORDER_METRICS, "purchase_time")
            if (len(dates) != 1 or head["time_conflict"]
                    or not all(self.valid_time(t, self.order_period) for t in dates)):
                self.bad(("recency",), "purchase_time")
            else:
                stamp = index._EPOCH + timedelta(microseconds=next(iter(dates)))
                self.last = max(self.last, stamp) if self.last else stamp
        elif possible:
            self.bad(("orders", "recency"), "order_conflict")

    def action(self, rows):
        self.counts["logical_actions"] += 1
        variants = {}
        damaged = set()
        for row in rows:
            if self.cancelled():
                raise index._Cancelled
            self.counts["action_rows_read"] += 1
            variant = variants.setdefault(row["variant_key"], (row, set()))
            if row["ordinal"] is not None and row["kind"] == "VIEW":
                code = _product(row["selected_namespace"], row["selected_value"])
                if code is None:
                    damaged.add(row["variant_key"])
                else:
                    variant[1].add(code)
        head = next(iter(variants.values()))[0]
        kinds = {row["kind"] for row, _ in variants.values()}
        metrics = []
        if "VIEW" in kinds:
            metrics.extend(("views", "unique_viewed_products"))
        if "FAVORITE" in kinds:
            metrics.append("favorites")
        if not metrics:
            return
        if head["owner_conflict"] or head["kind_conflict"] or len(kinds) != 1:
            self.bad(metrics, "action_conflict")
            return
        if not all(self.valid_time(row["event_us"], self.action_period) for row, _ in variants.values()):
            self.bad(metrics, "action_time")
            return
        if "VIEW" in kinds:
            self.views += 1
            product_sets = {frozenset(products) for _, products in variants.values()}
            if damaged or len(product_sets) != 1:
                self.bad(("unique_viewed_products",), "product_identity")
            else:
                self.products.update(next(iter(product_sets)))
        else:
            self.favorites += 1

    def result(self):
        bad = {metric for metric, _ in self.issues}
        money = tuple(CurrencyActivity(c, self.currency_orders[c], self.amounts[c], _mean(self.amounts[c], self.currency_orders[c]))
                      for c in CURRENCIES if self.currency_orders[c]) if "money" not in bad else None
        last = self.last if "recency" not in bad else None
        days = (self.as_of.date() - last.date()).days if last is not None and self.as_of is not None else None
        values = dict(orders=self.orders if "orders" not in bad else None,
                      units=self.units if "units" not in bad else None, money=money,
                      views=self.views if "views" not in bad else None,
                      favorites=self.favorites if "favorites" not in bad else None,
                      unique_viewed_products=len(self.products) if "unique_viewed_products" not in bad else None)
        available = any(v is not None for v in values.values())
        empty = not self.issues and not self.counts["logical_orders"] and not self.counts["logical_actions"]
        status = ActivityStatus.EMPTY if empty else ActivityStatus.READY if available else ActivityStatus.UNAVAILABLE
        return CustomerActivity(status, self.customer, **values, last_purchase_utc=last, days_since_purchase=days,
                                coverage=self.ready.coverage, calculated_at_utc=self.now, purchase_as_of_utc=self.as_of,
                                generation=self.ready.generation, source_version=self.ready.source_version,
                                orders_coverage_complete=self.order_period is not None,
                                actions_coverage_complete=self.action_period is not None,
                                issues=tuple(ActivityIssue(*issue) for issue in sorted(self.issues)),
                                diagnostics=tuple(sorted(self.counts.items())))


class CustomerActivityService:
    def __init__(self, *, cache_dir=index.DEFAULT_CACHE_DIR, clock: Callable[[], datetime] | None = None):
        self._index = index.CustomerHistoryIndexService(cache_dir=cache_dir)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def get(self, canonical_customer_id, *, raw_root=index.DEFAULT_RAW_ROOT, cancelled=lambda: False):
        """Caller supplies the confirmed canonical ID; no model/contact lookup."""
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            return CustomerActivity(ActivityStatus.ERROR)
        now = now.astimezone(timezone.utc)
        def failure(status):
            return CustomerActivity(status, calculated_at_utc=now)
        try:
            _request(canonical_customer_id, None, None, None, 1)
            if cancelled():
                return failure(ActivityStatus.CANCELLED)
            root = Path(raw_root).resolve()
            path = self._index.index_path(raw_root=root)
            if not path.exists():
                return failure(ActivityStatus.UNAVAILABLE)
            with index._reader_lock(root):
                sources = index._capture_sources(root)
                version = index._database_version(path)
                with index._read_index(path) as connection, localcontext() as decimals:
                    decimals.prec = 64
                    connection.set_progress_handler(lambda: int(cancelled()), 1000)
                    ready = index._inspect_connection(connection, sources)  # No verify_integrity.
                    connection.row_factory = sqlite3.Row
                    result = _Aggregate(canonical_customer_id, ready, now, cancelled)
                    for period, sql, aggregate in ((result.order_period, _ORDERS_SQL, result.order),
                                                   (result.action_period, _ACTIONS_SQL, result.action)):
                        if period is None:
                            continue
                        for _, rows in groupby(connection.execute(sql, (canonical_customer_id,)), lambda row: row["event_key"]):
                            if cancelled():
                                return failure(ActivityStatus.CANCELLED)
                            aggregate(rows)
                    output = result.result()
                if cancelled():
                    return failure(ActivityStatus.CANCELLED)
                if version != index._database_version(path) or sources.version != index._capture_sources(root).version:
                    return failure(ActivityStatus.STALE)
                return output
        except index._Stale:
            return failure(ActivityStatus.STALE)
        except Exception as exc:
            if cancelled():
                return failure(ActivityStatus.CANCELLED)
            return failure(ActivityStatus.UNAVAILABLE if index._unavailable(exc) else ActivityStatus.ERROR)

    def validate(self, result, *, raw_root=index.DEFAULT_RAW_ROOT):
        if result.status not in (ActivityStatus.READY, ActivityStatus.EMPTY):
            return result
        state = self._index.get_status(raw_root=raw_root)
        if (state.status is index.HistoryIndexStatus.READY and state.generation == result.generation
                and state.source_version == result.source_version):
            return result
        return CustomerActivity(ActivityStatus.STALE, calculated_at_utc=result.calculated_at_utc)
