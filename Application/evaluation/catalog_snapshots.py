"""Explicit external CSV archive for research. Never an automatic production source.

Original bytes are retained. Parsing/stock/duplicate/path semantics come from
RecommendationCatalog; no legacy DataFrame rounding or category-code rewriting.
Integrity and declared provenance do not certify point-in-time availability.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
import csv
from dataclasses import dataclass, fields
from datetime import date, datetime
from enum import Enum
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from types import MappingProxyType

from Application.paths import PROJECT_ROOT
from Application.recommendation_catalog import CatalogItem, RecommendationCatalog, read_rows, stock, text
from .hybrid_provenance import HybridEvaluationError, Validity, utc


class CatalogSnapshotError(ValueError):
    """Static codes only; no raw rows, source paths or product identifiers."""


class SourceOrigin(str, Enum):
    HISTORICAL_1C = "HISTORICAL_1C"
    CURRENT_1C = "CURRENT_1C"
    ARCHIVED_EXPORT = "ARCHIVED_EXPORT"
    SYNTHETIC = "SYNTHETIC"


def _utc(value):
    try:
        return utc(value)
    except HybridEvaluationError:
        raise CatalogSnapshotError("AWARE_TIMESTAMP_REQUIRED") from None


def _day(value):
    if type(value) is not date:
        raise CatalogSnapshotError("CALENDAR_DATE_REQUIRED")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("ascii")).hexdigest()


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CatalogSnapshotError("INVALID_MANIFEST")
        result[key] = value
    return result


def _file_spec(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


@dataclass(frozen=True, slots=True)
class SnapshotProvenance:
    inventory_date: date  # UTC date when exact time is known; declared date otherwise.
    inventory_as_of: datetime | None
    exported_at: datetime  # Nomenclature export creation, supplied explicitly.
    attributes_observed_at: datetime
    categories_observed_at: datetime
    inventory_origin: SourceOrigin
    attributes_origin: SourceOrigin
    categories_origin: SourceOrigin
    categories_exported_at: datetime | None = None

    def __post_init__(self):
        _day(self.inventory_date)
        for name in ("inventory_as_of", "exported_at", "attributes_observed_at", "categories_observed_at",
                     "categories_exported_at"):
            value = getattr(self, name)
            if value is None and name in ("inventory_as_of", "categories_exported_at"):
                continue
            object.__setattr__(self, name, _utc(value))
        if self.inventory_as_of is not None and self.inventory_date != self.inventory_as_of.date():
            raise CatalogSnapshotError("INVENTORY_DATE_MISMATCH")
        for name in ("inventory_origin", "attributes_origin", "categories_origin"):
            if not isinstance(getattr(self, name), SourceOrigin):
                raise CatalogSnapshotError("INVALID_SOURCE_ORIGIN")

    def to_dict(self):
        return {f.name: getattr(self, f.name).isoformat() if isinstance(getattr(self, f.name), (date, datetime))
                else getattr(self, f.name).value if isinstance(getattr(self, f.name), SourceOrigin)
                else None for f in fields(self)}

    @property
    def limitations(self):
        reasons = ["POINT_IN_TIME_UNVERIFIED"]
        if self.inventory_as_of is None:
            reasons.append("UNKNOWN_INVENTORY_TIME")
        else:
            for stamp, reason in ((self.attributes_observed_at, "LATER_ATTRIBUTES"),
                                  (self.categories_observed_at, "LATER_CATEGORIES"),
                                  (self.exported_at, "RETROSPECTIVE_EXPORT"),
                                  (self.categories_exported_at, "RETROSPECTIVE_CATEGORIES_EXPORT")):
                if stamp is not None and stamp > self.inventory_as_of:
                    reasons.append(reason)
        if self.categories_exported_at is None:
            reasons.append("CATEGORY_EXPORT_TIME_UNKNOWN")
        return tuple(reasons)

    @classmethod
    def from_dict(cls, value):
        try:
            if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
                raise ValueError
            decoded = dict(value)
            decoded["inventory_date"] = date.fromisoformat(decoded["inventory_date"])
            for name in ("inventory_origin", "attributes_origin", "categories_origin"):
                decoded[name] = SourceOrigin(decoded[name])
            for name in ("inventory_as_of", "exported_at", "attributes_observed_at", "categories_observed_at",
                         "categories_exported_at"):
                if decoded[name] is not None:
                    decoded[name] = datetime.fromisoformat(decoded[name])
            return cls(**decoded)
        except (ValueError, TypeError, KeyError):
            raise CatalogSnapshotError("INVALID_PROVENANCE") from None


@dataclass(frozen=True, slots=True)
class SnapshotSuitability:
    status: Validity
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SnapshotInfo:
    snapshot_id: str
    provenance: SnapshotProvenance


@dataclass(frozen=True, slots=True, repr=False)
class HistoricalCatalogSnapshot:
    info: SnapshotInfo
    known_codes: frozenset[str]
    items: Mapping[str, CatalogItem]  # Unambiguous, including zero/unknown stock.
    category_paths: Mapping[str, frozenset[str]]  # Code-only, no label aliases.
    diagnostics: Mapping[str, int]  # Aggregate counters only.
    exclusions: Mapping[str, tuple[str, ...]]  # Local research, not aggregate reports.
    manifest_json: str

    @property
    def validity(self):
        return Validity.PROXY

    @property
    def stocks(self) -> Mapping[str, Decimal | None]:
        return MappingProxyType({code: item.stock for code, item in self.items.items()})

    @property
    def in_stock_codes(self):
        return frozenset(code for code, item in self.items.items() if item.stock is not None and item.stock > 0)

    def assess_cutoff(self, cutoff):
        cutoff = _utc(cutoff)
        p = self.info.provenance
        if p.inventory_as_of is None:
            return SnapshotSuitability(Validity.UNAVAILABLE, ("UNKNOWN_INVENTORY_TIME",))
        if p.inventory_as_of != cutoff:
            return SnapshotSuitability(Validity.UNAVAILABLE, ("INVENTORY_CUTOFF_MISMATCH",))
        return SnapshotSuitability(Validity.PROXY, p.limitations)


def _analyze(nomenclature, categories, max_rows):
    """Streaming diagnostics followed by existing catalog loading, O(unique items) memory."""
    counts = Counter(nomenclature_rows=0, category_rows=0, invalid_stock_rows=0,
                     empty_category_rows=0, unknown_category_rows=0, invalid_category_path_rows=0,
                     missing_kind_rows=0, missing_gender_rows=0, missing_collection_rows=0,
                     multi_category_codes=0, missing_parent_refs=0, blank_code_rows=0,
                     duplicate_category_rows=0, ambiguous_category_codes=0)
    category_codes, parents, definitions, ambiguous_categories = set(), [], {}, set()
    try:
        for row in read_rows(categories, ("КодКатегории", "НазваниеКатегории", "КодРодительскойКатегории")):
            counts["category_rows"] += 1
            if counts["category_rows"] > max_rows:
                raise CatalogSnapshotError("ROW_LIMIT_EXCEEDED")
            code, parent = text(row["КодКатегории"]), text(row["КодРодительскойКатегории"])
            if code:
                definition = (parent, text(row["НазваниеКатегории"]))
                if code in definitions:
                    counts["duplicate_category_rows"] += 1
                    if definitions[code] != definition:
                        ambiguous_categories.add(code)
                else:
                    definitions[code] = definition
                category_codes.add(code)
                parents.append(parent)
        counts["missing_parent_refs"] = sum(bool(p) and p not in category_codes for p in parents)
        counts["ambiguous_category_codes"] = len(ambiguous_categories)
        catalog = RecommendationCatalog(nomenclature, categories)
        product_categories = defaultdict(set)
        required = ("КодНоменклатуры", "ВидНоменклатуры", "ПолНоменклатуры", "Коллекция", "Остаток", "КатегорияНаСайте")
        for row in read_rows(nomenclature, required):
            counts["nomenclature_rows"] += 1
            if counts["nomenclature_rows"] > max_rows:
                raise CatalogSnapshotError("ROW_LIMIT_EXCEEDED")
            code = row["КодНоменклатуры"]
            if not code:
                counts["blank_code_rows"] += 1
                continue
            if (not text(code) or code == "-" or any(c.isspace() or not c.isprintable() for c in code)):
                raise CatalogSnapshotError("INVALID_PRODUCT_CODE")
            category = text(row["КатегорияНаСайте"])
            if category:
                product_categories[code].add(category)
            counts["empty_category_rows"] += not category
            counts["unknown_category_rows"] += bool(category) and category not in category_codes
            counts["invalid_stock_rows"] += stock(row["Остаток"]) is None
            for name, field in (("missing_kind_rows", "ВидНоменклатуры"), ("missing_gender_rows", "ПолНоменклатуры"),
                                ("missing_collection_rows", "Коллекция")):
                counts[name] += not text(row[field])
        catalog.load()
        paths = {code: path for code, path in catalog.category_paths.items() if code in category_codes}
        # A second bounded stream avoids retaining all input rows just for path diagnostics.
        for row in read_rows(nomenclature, required):
            category = text(row["КатегорияНаСайте"])
            counts["invalid_category_path_rows"] += category in category_codes and category not in paths
        counts["multi_category_codes"] = sum(len(values) > 1 for values in product_categories.values())
        counts.update(catalog.diagnostics)
        counts["known_products"] = len(catalog.all_codes)
        counts["unambiguous_products"] = len(catalog.items)
        reasons = {}
        for code in sorted(catalog.all_codes):
            item = catalog.items.get(code)
            issues = []
            if item is None:
                issues.append("AMBIGUOUS_CODE")
            else:
                if item.stock is None:
                    issues.append("INVALID_STOCK")
                elif item.stock == 0:
                    issues.append("ZERO_STOCK")
                for field, reason in (("ВидНоменклатуры", "MISSING_KIND"), ("ПолНоменклатуры", "MISSING_GENDER"),
                                      ("Коллекция", "MISSING_COLLECTION")):
                    if not item.get(field):
                        issues.append(reason)
                if not item.get("КатегорияНаСайте"):
                    issues.append("EMPTY_CATEGORY")
                elif item.get("КатегорияНаСайте") not in category_codes:
                    issues.append("UNKNOWN_CATEGORY_CODE")
                elif item.get("КатегорияНаСайте") not in paths:
                    issues.append("INVALID_CATEGORY_PATH")
            if issues:
                reasons[code] = tuple(issues)
        projection = {"known_codes": sorted(catalog.all_codes), "items": {
            code: {"attributes": dict(item.attributes), "stock": str(item.stock) if item.stock is not None else None}
            for code, item in sorted(catalog.items.items())},
            "category_paths": {code: sorted(path) for code, path in sorted(paths.items())}}
        return catalog, MappingProxyType(paths), dict(sorted(counts.items())), MappingProxyType(reasons), _digest(projection)
    except CatalogSnapshotError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError, OSError, csv.Error):
        raise CatalogSnapshotError("INVALID_CSV") from None


class HistoricalCatalogArchive:
    def __init__(self, root, *, max_rows=250_000):
        self.root = Path(root).resolve()
        if self.root.is_relative_to(PROJECT_ROOT.resolve()):
            raise CatalogSnapshotError("EXTERNAL_ARCHIVE_REQUIRED")
        if type(max_rows) is not int or max_rows < 1:
            raise CatalogSnapshotError("INVALID_ROW_LIMIT")
        self.max_rows = max_rows
        self._check_root()

    def _check_root(self):
        if (self.root.resolve() != self.root or self.root.is_symlink() or self.root.is_junction()
                or self.root.is_relative_to(PROJECT_ROOT.resolve())):
            raise CatalogSnapshotError("UNSAFE_ARCHIVE_ROOT")
        if any((parent / ".git").exists() for parent in (self.root, *self.root.parents)):
            raise CatalogSnapshotError("ARCHIVE_INSIDE_GIT_WORKTREE")

    def _directory(self, snapshot_id):
        self._check_root()
        if not isinstance(snapshot_id, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot_id):
            raise CatalogSnapshotError("INVALID_SNAPSHOT_ID")
        path = self.root / snapshot_id
        if path.resolve() != path or path.is_symlink() or path.is_junction():
            raise CatalogSnapshotError("UNSAFE_SNAPSHOT_PATH")
        return path

    def _manifest(self, snapshot_id):
        path = self._directory(snapshot_id) / "manifest.json"
        try:
            if path.is_symlink() or path.stat().st_size > 64 * 1024:
                raise ValueError
            body = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields)
            if (set(body) != {"snapshot_id", "format_version", "parser_version", "provenance", "files", "normalized_sha256",
                              "diagnostics", "validity", "limitations"}
                    or type(body["format_version"]) is not int or body["format_version"] != 1
                    or type(body["parser_version"]) is not int or body["parser_version"] != 1
                    or body["validity"] != "PROXY" or body["snapshot_id"] != snapshot_id
                    or set(body["files"]) != {"nomenclature.csv", "site_categories.csv"}
                    or _digest({key: value for key, value in body.items() if key != "snapshot_id"}) != snapshot_id):
                raise ValueError
            provenance = SnapshotProvenance.from_dict(body["provenance"])
            if provenance.to_dict() != body["provenance"]:
                raise ValueError
            if body["limitations"] != list(provenance.limitations):
                raise ValueError
            return body, provenance
        except (ValueError, TypeError, KeyError, OSError):
            raise CatalogSnapshotError("INVALID_MANIFEST") from None

    def register(self, nomenclature, categories, provenance):
        """Explicitly copy only these two files; publish a content-addressed directory."""
        if not isinstance(provenance, SnapshotProvenance):
            raise CatalogSnapshotError("INVALID_PROVENANCE")
        self._check_root()
        self.root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root)).resolve()
        try:
            specs = {}
            for name, source in (("nomenclature.csv", nomenclature), ("site_categories.csv", categories)):
                source = Path(source)
                before = _file_spec(source)
                with source.open("rb") as original, (stage / name).open("xb") as target:
                    shutil.copyfileobj(original, target, 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
                if _file_spec(stage / name) != before or _file_spec(source) != before:
                    raise CatalogSnapshotError("SOURCE_CHANGED")
                specs[name] = before
            _, _, diagnostics, _, normalized = _analyze(stage / "nomenclature.csv", stage / "site_categories.csv", self.max_rows)
            body = {"format_version": 1, "parser_version": 1, "provenance": provenance.to_dict(),
                    "files": specs, "normalized_sha256": normalized, "diagnostics": diagnostics,
                    "validity": "PROXY", "limitations": list(provenance.limitations)}
            snapshot_id = _digest(body)
            body["snapshot_id"] = snapshot_id
            destination = self._directory(snapshot_id)
            with (stage / "manifest.json").open("x", encoding="utf-8") as stream:
                stream.write(_json(body))
                stream.flush()
                os.fsync(stream.fileno())
            if not destination.exists():
                try:
                    os.rename(stage, destination)  # Never overwrite an existing published snapshot.
                except OSError:
                    if not destination.exists():
                        raise
            return self.load(snapshot_id)
        finally:
            # Only the exact temporary directory created by this invocation is removable.
            if stage.exists():
                if stage.parent != self.root or not stage.name.startswith(".staging-") or stage.is_symlink() or stage.is_junction():
                    raise CatalogSnapshotError("UNSAFE_STAGING_PATH")
                shutil.rmtree(stage)

    def load(self, snapshot_id):
        body, provenance = self._manifest(snapshot_id)
        directory = self._directory(snapshot_id)
        for name, spec in body["files"].items():
            path = directory / name
            if path.is_symlink() or _file_spec(path) != spec:
                raise CatalogSnapshotError("CSV_CHECKSUM_MISMATCH")
        catalog, paths, diagnostics, reasons, normalized = _analyze(
            directory / "nomenclature.csv", directory / "site_categories.csv", self.max_rows)
        if diagnostics != body["diagnostics"] or normalized != body["normalized_sha256"]:
            raise CatalogSnapshotError("NORMALIZED_DATA_MISMATCH")
        # Detect ordinary concurrent file changes across parsing as well as before it.
        for name, spec in body["files"].items():
            if _file_spec(directory / name) != spec:
                raise CatalogSnapshotError("CSV_CHANGED_DURING_LOAD")
        return HistoricalCatalogSnapshot(SnapshotInfo(snapshot_id, provenance), catalog.all_codes, catalog.items,
                                         paths, MappingProxyType(diagnostics), reasons, _json(body))

    def list_snapshots(self, inventory_date=None):
        self._check_root()
        if inventory_date is not None:
            _day(inventory_date)
        if not self.root.exists():
            return ()
        found = []
        for folder in sorted(self.root.iterdir()):
            if re.fullmatch(r"[0-9a-f]{64}", folder.name):
                _, provenance = self._manifest(folder.name)
                if inventory_date is None or provenance.inventory_date == inventory_date:
                    found.append(SnapshotInfo(folder.name, provenance))
        return tuple(found)

    def available_dates(self):
        return tuple(sorted({info.provenance.inventory_date for info in self.list_snapshots()}))

    def load_for_date(self, inventory_date, *, snapshot_id=None):
        _day(inventory_date)
        if snapshot_id is not None:
            snapshot = self.load(snapshot_id)
            if snapshot.info.provenance.inventory_date != inventory_date:
                raise CatalogSnapshotError("SNAPSHOT_DATE_MISMATCH")
            return snapshot
        choices = self.list_snapshots(inventory_date)
        if len(choices) != 1:
            raise CatalogSnapshotError("SNAPSHOT_MISSING_OR_AMBIGUOUS")
        return self.load(choices[0].snapshot_id)

    def load_for_cutoff(self, cutoff, *, snapshot_id=None):
        cutoff = _utc(cutoff)
        snapshot = self.load_for_date(cutoff.date(), snapshot_id=snapshot_id)
        suitability = snapshot.assess_cutoff(cutoff)
        if suitability.status is Validity.UNAVAILABLE:
            raise CatalogSnapshotError(suitability.reasons[0])
        return snapshot
