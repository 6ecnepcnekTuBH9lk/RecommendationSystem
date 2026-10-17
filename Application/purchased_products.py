"""Read-only purchase evidence for recommendation exclusion, not ML labels.

CP/delivering/F are the agreed operational definition, not proof of payment.
No source-order/creation-time heuristic selects a winning order snapshot.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
import sqlite3

from Application import customer_history_index as index
from Application.customer_history_service import _coverage_complete, _request
from Application.mindbox.records import ProductKey
from Application.mindbox.selection import DEFAULT_PURCHASE_STATUSES
from Application.product_resolution import (
    CatalogDiagnostics, ProductCatalog, ProductResolver, ResolutionStatus, normalize_product_key,
)


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
    # An independent resolver scope, evaluated over the SAME order snapshots.
    historical: "PurchaseEvidence | None" = None
    checked_codes: frozenset[str] = frozenset()


def classify(status):
    if status in DEFAULT_PURCHASE_STATUSES:
        return "PURCHASE"
    if status in CANCELLED:
        return "CANCELLED"
    if status in RETURNED:
        return "RETURN"
    return "UNKNOWN"


def assess_order(snapshots, resolver, *, owner_conflict=False, diagnostics=None, ambiguous_catalog_prefixes=frozenset()):
    """Return code evidence and aggregate diagnostics for ONE logical order.

    Missing/repeated cross-snapshot identities and changing product identities
    make their involved codes uncertain. Returns preserve an otherwise unanimous
    purchase but cannot establish a purchase on their own.
    """
    groups = defaultdict(list)
    counts = Counter()
    unresolved = False
    diagnostics = {} if diagnostics is None else diagnostics
    for snapshot_no, lines in enumerate(snapshots):
        for row in lines:
            status = row["raw_status"]
            counts[status] += 1
            state = classify(status)
            key = None if row["selected_namespace"] is None else ProductKey(row["selected_namespace"], row["selected_value"])
            match = None if key is None else resolver.resolve(key, strict=False)
            code = match.item_id if match is not None and match.status is ResolutionStatus.RESOLVED else None
            if match is not None and match.status is ResolutionStatus.UNKNOWN_CANDIDATE:
                normalized = normalize_product_key(key)
                # Only a complete, well-formed legacy prefix proves disjointness.
                # Printable size suffixes may contain spaces: the established
                # resolver uses ONLY the first six characters for identity.
                # Keep short IDs, damaged prefixes and control characters unsafe.
                if (normalized.item_id is not None and len(normalized.item_id) == 6
                        and normalized.item_id not in ambiguous_catalog_prefixes
                        and all(c.isprintable() and not c.isspace() for c in normalized.item_id)
                        and all(c.isprintable() for c in key.value)):
                    code = normalized.item_id
                    diagnostics.setdefault("historical_out_of_scope_codes", set()).add(code)
                    diagnostics["historical_out_of_scope_lines"] = diagnostics.get("historical_out_of_scope_lines", 0) + 1
                    if " " in key.value[6:]:
                        diagnostics["normalized_suffix_space_lines"] = diagnostics.get("normalized_suffix_space_lines", 0) + 1
            if code is None and state != "CANCELLED":
                unresolved = True
                diagnostics["unsafe_unresolved_identifier_lines"] = diagnostics.get("unsafe_unresolved_identifier_lines", 0) + 1
                # Fixed categories only; never put raw keys/namespaces in diagnostics.
                if match is None:
                    reason = "missing_reference"
                elif match.status is ResolutionStatus.UNSUPPORTED_NAMESPACE:
                    reason = "unsupported_namespace"
                elif match.status is ResolutionStatus.INVALID_ID:
                    reason = "invalid_identifier"
                elif len(key.value) < 6:
                    reason = "incomplete_identifier"
                elif any(not c.isprintable() or c.isspace() for c in key.value[:6]):
                    reason = "damaged_identity_prefix"
                elif any(not c.isprintable() for c in key.value):
                    reason = "nonprintable_identifier"
                else:
                    reason = "ambiguous_prefix"
                diagnostics.setdefault("unresolved_identifier_reasons", Counter())[reason] += 1
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


class _PurchaseScope:
    """Accumulate the existing consensus separately for each identity universe."""

    def __init__(self, codes):
        self.codes = frozenset(codes)
        self.resolver = ProductResolver(ProductCatalog(self.codes, CatalogDiagnostics()))
        self.ambiguous_prefixes = frozenset(code[:6] for code in self.codes if len(code) > 6)
        self.purchased, self.uncertain, self.statuses = set(), set(), Counter()
        self.cancelled_lines = self.conflicts = 0
        self.unresolved = False
        self.identities = {}

    def add(self, snapshots, owner_conflict):
        bought, doubtful, count, excluded, conflict, missing = assess_order(
            snapshots, self.resolver, owner_conflict=owner_conflict, diagnostics=self.identities,
            ambiguous_catalog_prefixes=self.ambiguous_prefixes)
        self.purchased.update(bought)
        self.uncertain.update(doubtful)
        self.statuses.update(count)
        self.cancelled_lines += excluded
        self.conflicts += int(conflict)
        self.unresolved |= missing

    def evidence(self, ready):
        # Keep outside identities throughout snapshot comparison; only clip the
        # final evidence. Never merge model codes into the current catalog scope.
        purchased = frozenset(self.purchased & self.codes)
        uncertain = frozenset(self.uncertain & self.codes)
        diagnostics = {"rows_by_status": dict(sorted(self.statuses.items())), "purchased_codes": len(purchased),
                       "cancelled_lines_not_purchases": self.cancelled_lines,
                       "unknown_status_lines": sum(n for s, n in self.statuses.items() if classify(s) == "UNKNOWN"),
                       "conflicting_orders": self.conflicts, "uncertain_codes": len(uncertain),
                       "unresolved_non_cancelled_products": self.unresolved,
                       "historical_out_of_scope_codes": len(self.identities.get("historical_out_of_scope_codes", ())),
                       "historical_out_of_scope_lines": self.identities.get("historical_out_of_scope_lines", 0),
                       "unsafe_unresolved_identifier_lines": self.identities.get("unsafe_unresolved_identifier_lines", 0),
                       "normalized_suffix_space_lines": self.identities.get("normalized_suffix_space_lines", 0),
                       "unresolved_identifier_reasons": dict(self.identities.get("unresolved_identifier_reasons", {})),
                       "coverage_complete": True}
        return PurchaseEvidence("UNRESOLVED_PRODUCTS" if self.unresolved else "READY", purchased, uncertain,
                                diagnostics, ready.generation, ready.source_version, ready.coverage,
                                checked_codes=self.codes)


class PurchasedProductsService:
    def __init__(self, *, cache_dir=index.DEFAULT_CACHE_DIR):
        self._index = index.CustomerHistoryIndexService(cache_dir=cache_dir)

    def get(self, customer_id, catalog, *, raw_root=index.DEFAULT_RAW_ROOT, cancelled=lambda: False,
            historical_codes=None):
        """One indexed read; optional historical evidence never changes target resolution.

        catalog=None permits historical-only validation when current CSV is unavailable.
        """
        request = _request(customer_id, ("ORDER",), None, None, 1)
        root = Path(raw_root).resolve()
        path = self._index.index_path(raw_root=root)
        try:
            if not path.exists():
                return PurchaseEvidence("INDEX_NOT_READY", diagnostics={"failure_reason": "missing_index"})
            with index._reader_lock(root):
                sources = index._capture_sources(root)
                version = index._database_version(path)
                current = _PurchaseScope(catalog.all_codes if catalog is not None else ())
                historical = _PurchaseScope(historical_codes) if historical_codes is not None else None
                if historical is not None:
                    # These model identities cannot be recovered unambiguously by
                    # the existing prefix-6 resolver, even in an otherwise empty history.
                    historical.uncertain.update(code for code in historical.codes if not code or len(code) > 6
                                                or any(not c.isprintable() or c.isspace() for c in code))
                with index._read_index(path) as connection:
                    connection.set_progress_handler(lambda: int(cancelled()), 1000)
                    ready = index._inspect_connection(connection, sources)  # No integrity scan.
                    if not _coverage_complete(ready.coverage, request):
                        return PurchaseEvidence("INCOMPLETE_COVERAGE", diagnostics={"coverage_complete": False}, coverage=ready.coverage)
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
                        current.add(snapshots, bool(event["owner_conflict"]))
                        if historical is not None:
                            historical.add(snapshots, bool(event["owner_conflict"]))
                if version != index._database_version(path) or index._capture_sources(root).version != sources.version:
                    return PurchaseEvidence("INDEX_NOT_READY", diagnostics={"failure_reason": "sources_changed"})
                result = current.evidence(ready)
                if historical is not None:
                    result = replace(result, historical=historical.evidence(ready))
                return result
        except Exception as exc:
            if cancelled():
                return PurchaseEvidence("CANCELLED")
            unavailable = index._unavailable(exc)
            reason = "source_unavailable" if unavailable else "stale_index" if isinstance(exc, index._Stale) else "index_error"
            return PurchaseEvidence("SOURCE_UNAVAILABLE" if unavailable else "INDEX_NOT_READY", diagnostics={"failure_reason": reason})

    def is_current(self, evidence, *, raw_root=index.DEFAULT_RAW_ROOT):
        status = self._index.get_status(raw_root=raw_root)  # Metadata only.
        return (status.status is index.HistoryIndexStatus.READY and status.generation == evidence.generation
                and status.source_version == evidence.source_version)
