"""Read-only purchase evidence for recommendation exclusion, not ML labels.

CP/delivering/F are the agreed operational definition, not proof of payment.
No source-order/creation-time heuristic selects a winning order snapshot.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
import sqlite3

from Application import customer_history_index as index
from Application.customer_history_service import _coverage_complete, _request
from Application.mindbox.records import ProductKey
from Application.mindbox.selection import DEFAULT_PURCHASE_STATUSES
from Application.product_resolution import CatalogDiagnostics, ProductCatalog, ProductResolver, ResolutionStatus


CANCELLED = frozenset(("Mindbox-Tech-Cancellation", "otmenen", "out-of-stock"))
RETURNED = frozenset(("Return", "klientskii-vozvrat"))


@dataclass(frozen=True, slots=True, repr=False)
class PurchaseEvidence:
    status: str
    purchased: frozenset[str] = frozenset()
    uncertain: frozenset[str] = frozenset()
    diagnostics: dict = field(default_factory=dict)
    generation: str | None = None
    source_version: str | None = None
    coverage: tuple = ()


def classify(status):
    if status in DEFAULT_PURCHASE_STATUSES:
        return "PURCHASE"
    if status in CANCELLED:
        return "CANCELLED"
    if status in RETURNED:
        return "RETURN"
    return "UNKNOWN"


def assess_order(snapshots, resolver, *, owner_conflict=False):
    """Return code evidence and aggregate diagnostics for ONE logical order.

    Missing/repeated cross-snapshot identities and changing product identities
    make their involved codes uncertain. Returns preserve an otherwise unanimous
    purchase but cannot establish a purchase on their own.
    """
    groups = defaultdict(list)
    counts = Counter()
    unresolved = False
    for snapshot_no, lines in enumerate(snapshots):
        for row in lines:
            status = row["raw_status"]
            counts[status] += 1
            state = classify(status)
            key = None if row["selected_namespace"] is None else ProductKey(row["selected_namespace"], row["selected_value"])
            match = None if key is None else resolver.resolve(key, strict=False)
            code = match.item_id if match is not None and match.status is ResolutionStatus.RESOLVED else None
            if code is None and state != "CANCELLED":
                unresolved = True
            identity = row["source_line_key"]
            # A single snapshot needs no cross-snapshot line alignment.
            identity = identity or ("local", snapshot_no, row["ordinal"])
            groups[identity].append((snapshot_no, code, state, Decimal(row["quantity"])))
    purchased, uncertain = set(), set()
    cancelled_lines = 0
    conflict = bool(owner_conflict)
    for identity, lines in groups.items():
        codes = {line[1] for line in lines if line[1] is not None}
        states = {line[2] for line in lines}
        stable = (not owner_conflict and len(codes) == 1
                  and len(lines) == len(snapshots)
                  and len({line[0] for line in lines}) == len(snapshots)
                  and (len(snapshots) == 1 or isinstance(identity, str))
                  and all(line[1] is not None for line in lines))
        if stable and states == {"CANCELLED"}:
            cancelled_lines += len(lines)
        elif stable and "PURCHASE" in states and states <= {"PURCHASE", "RETURN"} and all(
                qty.is_finite() and qty > 0 for _, _, state, qty in lines if state == "PURCHASE"):
            purchased.update(codes)
        else:
            uncertain.update(codes)
            conflict |= len(snapshots) > 1 or owner_conflict
    return purchased, uncertain, counts, cancelled_lines, conflict, unresolved


class PurchasedProductsService:
    def __init__(self, *, cache_dir=index.DEFAULT_CACHE_DIR):
        self._index = index.CustomerHistoryIndexService(cache_dir=cache_dir)

    def get(self, customer_id, catalog, *, raw_root=index.DEFAULT_RAW_ROOT, cancelled=lambda: False):
        request = _request(customer_id, ("ORDER",), None, None, 1)
        root = Path(raw_root).resolve()
        path = self._index.index_path(raw_root=root)
        try:
            if not path.exists():
                return PurchaseEvidence("INDEX_NOT_READY")
            with index._reader_lock(root):
                sources = index._capture_sources(root)
                version = index._database_version(path)
                resolver = ProductResolver(ProductCatalog(catalog.all_codes, CatalogDiagnostics()))
                purchased, uncertain, statuses = set(), set(), Counter()
                cancelled_lines = conflicts = 0
                unresolved = False
                with index._read_index(path) as connection:
                    connection.set_progress_handler(lambda: int(cancelled()), 1000)
                    ready = index._inspect_connection(connection, sources)  # No integrity scan.
                    if not _coverage_complete(ready.coverage, request):
                        return PurchaseEvidence("INCOMPLETE_COVERAGE", coverage=ready.coverage)
                    connection.row_factory = sqlite3.Row
                    events = connection.execute(
                        "SELECT e.event_key,e.owner_conflict FROM event_customers ec "
                        "JOIN events e USING(event_key) WHERE ec.canonical_customer_id=? AND e.source='ORDER'",
                        (customer_id,))
                    for event in events:
                        if cancelled():
                            return PurchaseEvidence("CANCELLED")
                        snapshots = []
                        for variant, in connection.execute("SELECT variant_key FROM variants WHERE event_key=?", (event["event_key"],)):
                            snapshots.append(connection.execute(
                                "SELECT l.source_line_key,l.ordinal,l.raw_status,l.quantity,"
                                "p.selected_namespace,p.selected_value FROM order_lines l "
                                "LEFT JOIN product_occurrences p ON p.variant_key=l.variant_key "
                                "AND p.role='line' AND p.ordinal=l.ordinal WHERE l.variant_key=?",
                                (variant,)).fetchall())
                        bought, doubtful, count, excluded, conflict, missing = assess_order(
                            snapshots, resolver, owner_conflict=bool(event["owner_conflict"]))
                        purchased.update(bought)
                        uncertain.update(doubtful)
                        statuses.update(count)
                        cancelled_lines += excluded
                        conflicts += int(conflict)
                        unresolved |= missing
                if version != index._database_version(path) or index._capture_sources(root).version != sources.version:
                    return PurchaseEvidence("INDEX_NOT_READY")
                diagnostics = {"rows_by_status": dict(sorted(statuses.items())), "purchased_codes": len(purchased),
                               "cancelled_lines_not_purchases": cancelled_lines,
                               "unknown_status_lines": sum(n for s, n in statuses.items() if classify(s) == "UNKNOWN"),
                               "conflicting_orders": conflicts, "uncertain_codes": len(uncertain),
                               "unresolved_non_cancelled_products": unresolved}
                return PurchaseEvidence("UNRESOLVED_PRODUCTS" if unresolved else "READY", frozenset(purchased),
                                        frozenset(uncertain), diagnostics, ready.generation, ready.source_version, ready.coverage)
        except Exception as exc:
            if cancelled():
                return PurchaseEvidence("CANCELLED")
            return PurchaseEvidence("SOURCE_UNAVAILABLE" if index._unavailable(exc) else "INDEX_NOT_READY")

    def is_current(self, evidence, *, raw_root=index.DEFAULT_RAW_ROOT):
        status = self._index.get_status(raw_root=raw_root)  # Metadata only.
        return (status.status is index.HistoryIndexStatus.READY and status.generation == evidence.generation
                and status.source_version == evidence.source_version)
