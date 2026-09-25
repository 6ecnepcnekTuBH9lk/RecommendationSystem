"""Read-only canonical statistics. Memory scales with unique users/products/orders.

No event history, pandas, training preparation or Qt. The storage lock protects
the complete scan against partition replacement; only its coordination file is
touched. Customers SQLite is opened in mode=ro.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import csv
import json
from pathlib import Path
import sqlite3
from statistics import median

from Application.interactions import InteractionBuilder, InteractionBuildError, classify_action_system_name
from Application.mindbox.adapters import adapt_action, adapt_action_system_name, adapt_customer_merge, adapt_order
from Application.mindbox.adapters._common import identifier, objects
from Application.mindbox.canonical_customers import database
from Application.mindbox.canonical_storage import catalog, checked_directory, storage_lock
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.order_dedup import OrderSnapshots
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, iter_export, part_files
from Application.mindbox.selection import MindboxSelectionConfig
from Application.product_resolution import CatalogError, DEFAULT_CATALOG_PATH, ProductResolver, load_catalog


@dataclass(frozen=True)
class Coverage:
    source: str
    intervals: tuple[tuple[str, str], ...] = ()
    source_kinds: tuple[str, ...] = ()
    updated: str | None = None


@dataclass(frozen=True)
class DatasetStatistics:
    calculated_at: str
    coverage: tuple[Coverage, ...]
    actions: int
    orders: int
    order_lines: int
    customers: int | None
    action_customers: int
    order_customers: int
    interaction_users: int
    actions_with_product: int
    actions_without_product: int
    view_interactions: int
    favorite_interactions: int
    purchase_interactions: int
    purchase_quantity: str
    mean_interactions: float
    median_interactions: float
    unique_source_products: int
    unique_resolved_items: int
    resolved_interactions: int
    unresolved_interactions: int
    resolution_rate: float
    action_types: tuple[tuple[str, int], ...]
    line_statuses: tuple[tuple[str, int, bool], ...]
    namespaces: tuple[tuple[str, int, int, int, float], ...]
    top_products: tuple[tuple[str, str, int, int, int, int], ...]
    diagnostics: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]


def _entries(data, name):
    """All current coverage, including gaps and unpaired days.

    Match canonical manual precedence, without training's longest-run selection.
    """
    manual = data.get("manual_interactions")
    result = [manual[name]] if manual else []
    for entry in data[name].values():
        if manual and (datetime.fromisoformat(entry["since"]) < datetime.fromisoformat(manual["until"])
                       and datetime.fromisoformat(entry["until"]) > datetime.fromisoformat(manual["since"])):
            continue
        result.append(entry)
    return sorted(result, key=lambda entry: datetime.fromisoformat(entry["since"]))


def _coverage(name, entries):
    intervals = []
    for entry in entries:
        start, end = entry["since"], entry["until"]
        if intervals and datetime.fromisoformat(start) <= datetime.fromisoformat(intervals[-1][1]):
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return Coverage(name, tuple(intervals), tuple(sorted({e["source_kind"] for e in entries})),
                    max((e["updated"] for e in entries), default=None))


class _Interactions:
    """Aggregate builder output, including unresolved item interactions."""

    def __init__(self, resolver):
        self.resolver = resolver
        self.users = Counter()
        self.items = {}
        self.failures = Counter()
        self.quantity = Decimal(0)

    def add(self, interaction):
        self.users[interaction.customer_id] += 1
        if interaction.quantity is not None:
            self.quantity += interaction.quantity
        resolved = self.resolver.resolve_interaction(interaction, strict=False)
        if resolved is None:
            status = self.resolver.resolve(interaction.product, strict=False).status
            self.failures[status.value] += 1
        else:
            counts = self.items.setdefault(resolved.item_id, Counter())
            counts[interaction.interaction_type.value] += 1


def _customer_snapshot(root):
    path = database(root)
    if not path.exists():
        return None, Coverage("Customers")
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        with connection:
            count = connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
            row = connection.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()
            metadata = json.loads(row[0]) if row else {}
        kinds = metadata.get("source_kinds", [metadata["source_kind"]] if metadata.get("source_kind") else [])
        return count, Coverage("Customers", tuple(tuple(pair) for pair in metadata.get("intervals", ())),
                               tuple(kinds), metadata.get("updated"))
    finally:
        connection.close()


def _catalog_names(path):
    # Catalog has already passed load_catalog validation. Keep its pipe delimiter.
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return {row["КодНоменклатуры"]: row.get("НазваниеНаСайте") or row.get("Номенклатура", "")
                for row in csv.DictReader(stream, delimiter="|")}


def _catalog_snapshot(path):
    """Keep names and resolver IDs from the same reference-file generation."""
    def signature():
        stat = Path(path).stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    try:
        before = signature()
        products = load_catalog(Path(path))
        names = _catalog_names(path)
        if signature() != before:
            raise CatalogError("Номенклатура обновилась во время чтения. Повторите расчёт.")
        return products, names
    except OSError:
        raise CatalogError("Cannot read catalog file") from None


def calculate_statistics(*, raw_root=DEFAULT_RAW_ROOT, catalog_path=DEFAULT_CATALOG_PATH,
                         progress=None, cancelled=None):
    """Scan published canonical data once. No legacy fallback or silent skipping.

    Raw order counts/statuses include every saved snapshot. Item interactions and
    purchase quantity use the first snapshot, as OrderSnapshots(diagnose=True)
    does in training diagnostics. Conflicts are explicitly reported as incomplete.
    """
    root = Path(raw_root)
    if not (root / "canonical/catalog.json").is_file():
        raise ValueError("Нет canonical dataset. Сначала загрузите данные Mindbox.")

    def check():
        if cancelled and cancelled():
            raise InterruptedError("Расчёт статистики отменён")

    with storage_lock(root):
        check()
        data = catalog(root)
        selection = MindboxSelectionConfig(**data["selection"])
        entries = {name: _entries(data, name) for name in ("actions", "orders")}
        merge = data["customer_merges"]
        warnings = []
        if any(entries.values()) and not merge:
            raise ValueError("Нет canonical CustomerMerges: невозможно подтвердить identity клиентов.")
        if merge and any(datetime.fromisoformat(e["since"]) < datetime.fromisoformat(merge["since"])
                         or datetime.fromisoformat(e["until"]) > datetime.fromisoformat(merge["until"])
                         for group in entries.values() for e in group):
            warnings.append("CustomerMerges не покрывает весь период: canonical identity может быть неполной.")

        def records(name, sources):
            count = 0
            for entry in sources:
                check()
                directory = checked_directory(root, entry["directory"], name)
                if len(part_files(directory, name)) != entry["parts"]:
                    raise ValueError("Canonical parts mismatch")
                for raw in iter_export(name, input_dir=directory):
                    check()
                    count += 1
                    if progress and (count == 1 or count % 10000 == 0):
                        progress(f"{name}: {count:,}".replace(",", " "))
                    yield raw

        identities = CustomerIdResolver(adapt_customer_merge(raw)
                                        for raw in records("customer_merges", [merge] if merge else []))
        builder = InteractionBuilder(selection.interaction_rules())
        product_catalog, names = _catalog_snapshot(catalog_path)
        products = ProductResolver(product_catalog)
        collected = _Interactions(products)
        action_types, statuses = Counter(), Counter()
        action_users, order_users = set(), set()
        with_product = missing_action_customer = raw_lines = 0
        for raw in records("actions", entries["actions"]):
            name = adapt_action_system_name(raw)
            action_types[name] += 1
            with_product += bool(objects(raw, "products"))
            source = identifier(raw, "customer.ids.mindboxId", required=False)
            if source is not None:
                action_users.add(identities.resolve(source))
            else:
                missing_action_customer += 1
            if classify_action_system_name(name, builder.rules) is None:
                builder.record_unmapped_action(name)
                continue
            action = adapt_action(raw, identities, product_namespaces=selection.action_product_namespaces)
            try:
                interactions = builder.from_action(action)
            except InteractionBuildError:
                # Valid mapped event with no products: not an import error.
                continue
            for interaction in interactions:
                collected.add(interaction)

        snapshots = OrderSnapshots()
        for raw in records("orders", entries["orders"]):
            order_users.add(identities.resolve(identifier(raw, "customer.ids.mindboxId")))
            for line in objects(raw, "lines", required=True):
                raw_lines += 1
                statuses[identifier(line, "status.ids.externalId")] += 1
            if not snapshots.accept(raw, diagnose=True):
                continue
            for line in adapt_order(raw, identities, product_namespaces=selection.order_product_namespaces):
                interaction = builder.from_order_line(line)
                if interaction is not None:
                    collected.add(interaction)

        check()
        customers, customer_coverage = _customer_snapshot(root)
        interactions, resolution = builder.diagnostics, products.diagnostics
        if snapshots.conflicting:
            warnings.append("Есть конфликтующие снимки заказов: item interactions рассчитаны по первому снимку.")
        if missing_action_customer:
            warnings.append("Часть Actions не содержит customer ID; число клиентов Actions учитывает только известные ID.")
        if customers is None:
            warnings.append("Canonical Customers отсутствует: число профилей неизвестно.")
        top = sorted(collected.items.items(), key=lambda pair: (-sum(pair[1].values()), pair[0]))[:30]
        diagnostics = (
            ("mapped_view_actions", interactions.actions_view),
            ("mapped_favorite_actions", interactions.actions_favorite),
            ("unmapped_actions", interactions.actions_unmapped),
            ("mapped_without_product", interactions.actions_malformed),
            ("actions_without_customer_id", missing_action_customer),
            ("unknown_candidate", collected.failures["unknown_candidate"]),
            ("unsupported_namespace", collected.failures["unsupported_namespace"]),
            ("invalid_id", collected.failures["invalid_id"]),
            ("orders_unique", len(snapshots.fingerprints)),
            ("orders_duplicate_identical", snapshots.identical),
            ("orders_duplicate_conflicting", snapshots.conflicting),
            ("unique_order_lines", interactions.order_lines_total),
            ("purchase_lines", interactions.order_lines_purchase),
            ("filtered_by_status", interactions.order_lines_filtered_by_status),
        )
        return DatasetStatistics(
            calculated_at=datetime.now(timezone.utc).isoformat(),
            coverage=tuple(_coverage(name.title(), entries[name]) for name in ("actions", "orders"))
                     + (_coverage("CustomerMerges", [merge] if merge else []), customer_coverage),
            actions=interactions.actions_total, orders=snapshots.raw, order_lines=raw_lines, customers=customers,
            action_customers=len(action_users), order_customers=len(order_users), interaction_users=len(collected.users),
            actions_with_product=with_product, actions_without_product=interactions.actions_total - with_product,
            view_interactions=interactions.view_interactions, favorite_interactions=interactions.favorite_interactions,
            purchase_interactions=interactions.purchase_interactions, purchase_quantity=str(collected.quantity),
            mean_interactions=interactions.total_interactions / len(collected.users) if collected.users else 0.0,
            median_interactions=float(median(collected.users.values())) if collected.users else 0.0,
            unique_source_products=resolution.unique_source_product_keys,
            unique_resolved_items=resolution.unique_resolved_catalog_items,
            resolved_interactions=resolution.total.resolved, unresolved_interactions=resolution.total.unresolved,
            resolution_rate=resolution.total.resolution_rate_percent,
            action_types=tuple(sorted(action_types.items(), key=lambda pair: (-pair[1], pair[0]))),
            line_statuses=tuple((name, count, name in builder.rules.purchase_line_statuses)
                                for name, count in sorted(statuses.items())),
            namespaces=tuple((name, counts.interactions_total, counts.resolved, counts.unresolved,
                              counts.resolution_rate_percent) for name, counts in resolution.by_namespace.items()),
            top_products=tuple((code, names.get(code, ""), counts["VIEW"], counts["FAVORITE"],
                                counts["PURCHASE"], sum(counts.values())) for code, counts in top),
            diagnostics=diagnostics, warnings=tuple(warnings),
        )
