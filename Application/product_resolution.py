"""Catalog identity boundary: validated legacy prefix-6, without event aggregation."""

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from Application.interactions import InteractionRecord, InteractionType
from Application.mindbox.records import ProductKey


DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[1] / "ВходныеДанные" / "Номенклатура.csv"
SUPPORTED_NAMESPACES = frozenset({"offline1C", "kanzlerKz"})


class CatalogError(Exception):
    """Safe catalog error; never includes file contents or paths."""


@dataclass(frozen=True)
class CatalogDiagnostics:
    rows_total: int = 0
    empty_code_rows: int = 0
    duplicate_code_rows: int = 0
    unique_items: int = 0


@dataclass(frozen=True, repr=False)
class ProductCatalog:
    item_ids: frozenset[str]
    diagnostics: CatalogDiagnostics

    def __post_init__(self):
        object.__setattr__(self, "item_ids", frozenset(self.item_ids))


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> ProductCatalog:
    """Read pipe UTF-8(-BOM) catalog; skip empty codes, count repeated codes."""
    codes = set()
    total = empty = duplicates = 0
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="|", strict=True)
            header = next(reader, None)
            if not header or any(not name.strip() for name in header) or len(set(header)) != len(header):
                raise CatalogError("Invalid catalog header")
            if "КодНоменклатуры" not in header:
                raise CatalogError("Missing required catalog column: КодНоменклатуры")
            column = header.index("КодНоменклатуры")
            for row in reader:
                total += 1
                if not row:
                    empty += 1
                    continue
                if len(row) != len(header):
                    raise CatalogError("Invalid catalog row width")
                code = row[column]
                if not code.strip():
                    empty += 1
                    continue
                if code != code.strip():
                    raise CatalogError("Outer whitespace in catalog code")
                duplicates += int(code in codes)
                codes.add(code)
    except FileNotFoundError:
        raise CatalogError("Catalog file not found") from None
    except (OSError, UnicodeError, csv.Error):
        raise CatalogError("Cannot read catalog: check file, UTF-8 encoding and CSV structure") from None
    return ProductCatalog(frozenset(codes), CatalogDiagnostics(total, empty, duplicates, len(codes)))


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    UNKNOWN_CANDIDATE = "unknown_candidate"
    UNSUPPORTED_NAMESPACE = "unsupported_namespace"
    INVALID_ID = "invalid_id"


class ProductResolutionError(Exception):
    def __init__(self, status: ResolutionStatus):
        self.status = status
        super().__init__(f"Product resolution failed: {status.value}")


@dataclass(frozen=True, repr=False)
class ProductResolution:
    status: ResolutionStatus
    item_id: str | None = None


@dataclass(frozen=True, repr=False)
class ResolvedInteraction:
    interaction: InteractionRecord
    item_id: str


@dataclass(frozen=True)
class ResolutionCounts:
    interactions_total: int = 0
    resolved: int = 0
    unresolved: int = 0
    unsupported_namespace: int = 0

    @property
    def resolution_rate_percent(self) -> float:
        return 100 * self.resolved / self.interactions_total if self.interactions_total else 0.0


@dataclass(frozen=True)
class ProductResolutionDiagnostics:
    total: ResolutionCounts
    by_type: Mapping[InteractionType, ResolutionCounts]
    by_namespace: Mapping[str, ResolutionCounts]
    unique_source_product_keys: int
    unique_resolved_catalog_items: int
    catalog_items_with_multiple_source_keys: int
    max_source_keys_per_catalog_item: int

    def __post_init__(self):
        for name in ("by_type", "by_namespace"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


class ProductResolver:
    """One sequential run per instance. Only resolve_interaction updates diagnostics."""

    def __init__(self, catalog: ProductCatalog):
        self._catalog = catalog
        self._total = Counter()
        self._by_type = {kind: Counter() for kind in InteractionType}
        # Unknown names go into a fixed bucket, never into printed diagnostics.
        self._by_namespace = {name: Counter() for name in (*sorted(SUPPORTED_NAMESPACES), "unsupported")}
        self._source_keys: set[ProductKey] = set()
        self._item_sources: dict[str, set[ProductKey]] = defaultdict(set)

    def resolve(self, product: ProductKey, *, strict: bool = True) -> ProductResolution:
        if product.namespace not in SUPPORTED_NAMESPACES:
            result = ProductResolution(ResolutionStatus.UNSUPPORTED_NAMESPACE)
        elif not isinstance(product.value, str) or not product.value or product.value != product.value.strip():
            result = ProductResolution(ResolutionStatus.INVALID_ID)
        else:
            candidate = product.value[:6]
            result = (ProductResolution(ResolutionStatus.RESOLVED, candidate)
                      if candidate in self._catalog.item_ids
                      else ProductResolution(ResolutionStatus.UNKNOWN_CANDIDATE))
        if strict and result.status is not ResolutionStatus.RESOLVED:
            raise ProductResolutionError(result.status)
        return result

    def resolve_interaction(self, interaction: InteractionRecord, *, strict: bool = True) -> ResolvedInteraction | None:
        result = self.resolve(interaction.product, strict=False)
        resolved = result.status is ResolutionStatus.RESOLVED
        namespace = interaction.product.namespace
        bucket = namespace if namespace in SUPPORTED_NAMESPACES else "unsupported"
        for counts in (self._total, self._by_type[interaction.interaction_type], self._by_namespace[bucket]):
            counts["interactions_total"] += 1
            counts["resolved" if resolved else "unresolved"] += 1
            counts["unsupported_namespace"] += int(result.status is ResolutionStatus.UNSUPPORTED_NAMESPACE)
        self._source_keys.add(interaction.product)
        if not resolved:
            if strict:
                raise ProductResolutionError(result.status)
            return None
        self._item_sources[result.item_id].add(interaction.product)
        return ResolvedInteraction(interaction, result.item_id)

    @property
    def diagnostics(self) -> ProductResolutionDiagnostics:
        sizes = [len(keys) for keys in self._item_sources.values()]
        return ProductResolutionDiagnostics(
            ResolutionCounts(**self._total),
            {kind: ResolutionCounts(**counts) for kind, counts in self._by_type.items()},
            {name: ResolutionCounts(**counts) for name, counts in self._by_namespace.items()},
            len(self._source_keys), len(sizes), sum(size > 1 for size in sizes), max(sizes, default=0),
        )
