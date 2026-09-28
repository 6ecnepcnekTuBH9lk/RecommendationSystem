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
from Application.mindbox.adapters._common import AdapterError, birth_date, identifier, number, objects, timestamp
from Application.mindbox.canonical_customers import database
from Application.mindbox.canonical_storage import catalog, checked_directory, storage_lock
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.order_dedup import OrderSnapshots
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, iter_export, part_files
from Application.mindbox.selection import MindboxSelectionConfig
from Application.product_resolution import CatalogError, DEFAULT_CATALOG_PATH, ProductResolver, load_catalog
from Application.order_statistics import CURRENCY_WARNING, OrderAggregates, order_for_adapter
from Application.statistics_period import shared_intervals


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
    gender_distribution: tuple[tuple[str, int, float], ...]
    age_distribution: tuple[tuple[str, int, float], ...]
    mean_age: float | None
    median_age: float | None
    view_users: int
    favorite_users: int
    purchase_users: int
    view_purchase_users: int
    favorite_purchase_users: int
    all_interaction_type_users: int
    repeat_buyers: int
    interaction_activity_distribution: tuple[tuple[str, int, float], ...]
    purchase_order_distribution: tuple[tuple[str, int, float], ...]
    mean_orders_per_buyer: float
    median_orders_per_buyer: float
    repeat_buyer_rate: float
    active_buyer_rate: float
    purchase_orders: int
    mean_purchase_lines_per_order: float
    median_purchase_lines_per_order: float
    mean_purchase_units_per_order: str
    median_purchase_units_per_order: str
    purchase_basket_distribution: tuple[tuple[str, int, float], ...]
    # currency, orders, lines, quantity, amount, mean, median
    order_financials: tuple[tuple[str, int, int, str, str, str, str], ...]
    # UTC month, RUB orders/amount, KZT orders/amount
    order_monthly_dynamics: tuple[tuple[str, int, str, int, str], ...]
    # currency, store key (None = missing), name, orders, buyers, lines, units, amount, mean
    store_statistics: tuple[tuple[str, str | None, str, int, int, int, str, str, str], ...]
    ordering_method_distribution: tuple[tuple[str, int, float], ...]
    delivery_type_distribution: tuple[tuple[str, int, float], ...]
    payment_type_distribution: tuple[tuple[str, int, float], ...]
    delivery_financials: tuple[tuple[str, int, str, str, str], ...]
    mixed_currency_purchase_orders: int
    unknown_currency_purchase_orders: int


def _distribution(counts, labels):
    total = sum(counts.values())
    return tuple((label, counts[label], 100 * counts[label] / total if total else 0.0) for label in labels)


def _bucket(value, limits, labels):
    return next((label for limit, label in zip(limits, labels) if value <= limit), labels[-1])


def _histogram_median(counts):
    total = sum(counts.values())
    if not total:
        return None
    positions = ((total - 1) // 2, total // 2)
    seen, values = 0, []
    for value, count in sorted(counts.items()):
        values.extend(value for position in positions if seen <= position < seen + count)
        seen += count
    return sum(values) / 2


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
        self.user_types = {}
        self.purchase_orders = Counter()
        self.items = {}
        self.failures = Counter()
        self.quantity = Decimal(0)

    def add(self, interaction):
        self.users[interaction.customer_id] += 1
        bit = {"VIEW": 1, "FAVORITE": 2, "PURCHASE": 4}[interaction.interaction_type.value]
        self.user_types[interaction.customer_id] = self.user_types.get(interaction.customer_id, 0) | bit
        if interaction.quantity is not None:
            self.quantity += interaction.quantity
        resolved = self.resolver.resolve_interaction(interaction, strict=False)
        if resolved is None:
            status = self.resolver.resolve(interaction.product, strict=False).status
            self.failures[status.value] += 1
        else:
            counts = self.items.setdefault(resolved.item_id, Counter())
            counts[interaction.interaction_type.value] += 1


def _customer_snapshot(root, as_of, check, progress):
    path = database(root)
    genders, ages, groups = Counter(), Counter(), Counter()
    age_labels = ("До 18 лет", "18–25 лет", "26–35 лет", "36–45 лет", "46–55 лет", "56–65 лет",
                  "66 лет и старше", "Возраст не определён")
    count, coverage = None, Coverage("Customers")
    if not path.exists():
        return count, coverage, _distribution(genders, ("Мужчины", "Женщины", "Не указан")), _distribution(groups, age_labels), None, None
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        count = 0
        # Project only demographic fields; contacts/IDs/full profiles never enter memory.
        for birthday, sex in connection.execute(
                "SELECT json_extract(raw, '$.birthDate'), json_extract(raw, '$.sex') FROM profiles"):
            check()
            count += 1
            if progress and (count == 1 or count % 10000 == 0):
                progress(f"Клиенты: {count:,}".replace(",", " "))
            # Same supported source values as the existing male/female mapping.
            gender = "Мужчины" if sex == "male" else "Женщины" if sex == "female" else "Не указан"
            genders[gender] += 1
            try:
                born = birth_date({"birthDate": birthday})
            except AdapterError:
                born = None
            if born is None or born > as_of:
                groups["Возраст не определён"] += 1
            else:
                age = as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))
                ages[age] += 1
                groups[_bucket(age, (17, 25, 35, 45, 55, 65), age_labels[:-1])] += 1
        with connection:
            row = connection.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()
            metadata = json.loads(row[0]) if row else {}
        kinds = metadata.get("source_kinds", [metadata["source_kind"]] if metadata.get("source_kind") else [])
        coverage = Coverage("Customers", tuple(tuple(pair) for pair in metadata.get("intervals", ())),
                            tuple(kinds), metadata.get("updated"))
        mean_age = sum(age * n for age, n in ages.items()) / sum(ages.values()) if ages else None
        return (count, coverage, _distribution(genders, ("Мужчины", "Женщины", "Не указан")),
                _distribution(groups, age_labels), mean_age, _histogram_median(ages))
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
        order_aggregates = OrderAggregates()
        period = shared_intervals(*[[(entry["since"], entry["until"]) for entry in entries[name]]
                                    for name in ("actions", "orders")])
        outside_period = fractional_lines = 0
        for raw in records("orders", entries["orders"]):
            ordered_at = timestamp(raw, "firstAction.dateTimeUtc")
            if not any(start <= ordered_at < end for start, end in period):
                outside_period += 1
                continue
            eligible_lines = []
            for line in objects(raw, "lines", required=True):
                quantity = number(line, "quantity")
                if quantity != quantity.to_integral_value():
                    fractional_lines += 1
                    continue
                eligible_lines.append(line)
            order_users.add(identities.resolve(identifier(raw, "customer.ids.mindboxId")))
            for line in eligible_lines:
                raw_lines += 1
                statuses[identifier(line, "status.ids.externalId")] += 1
            if not snapshots.accept(raw, diagnose=True):
                continue
            buyer = None
            purchases = []
            eligible_order = {**raw, "lines": eligible_lines}
            for line in adapt_order(order_for_adapter(eligible_order), identities, product_namespaces=selection.order_product_namespaces):
                interaction = builder.from_order_line(line)
                if interaction is not None:
                    collected.add(interaction)
                    buyer = interaction.customer_id
                    purchases.append(line)
            if buyer is not None:
                collected.purchase_orders[buyer] += 1
            order_aggregates.add(raw, purchases)

        check()
        calculated_at = datetime.now(timezone.utc)
        customers, customer_coverage, genders, ages, mean_age, median_age = _customer_snapshot(
            root, calculated_at.astimezone().date(), check, progress)
        user_types = Counter(collected.user_types.values())
        view_users = sum(n for mask, n in user_types.items() if mask & 1)
        favorite_users = sum(n for mask, n in user_types.items() if mask & 2)
        purchase_users = len(collected.purchase_orders)
        repeat_buyers = sum(n >= 2 for n in collected.purchase_orders.values())
        activity_labels = ("1", "2–5", "6–10", "11–25", "26–50", "51–100", "101+")
        order_labels = ("1 заказ", "2 заказа", "3–5 заказов", "6–10 заказов", "11+ заказов")
        activity = Counter(_bucket(n, (1, 5, 10, 25, 50, 100), activity_labels) for n in collected.users.values())
        order_counts = Counter(_bucket(n, (1, 2, 5, 10), order_labels) for n in collected.purchase_orders.values())
        interactions, resolution = builder.diagnostics, products.diagnostics
        if snapshots.conflicting:
            warnings.append("Есть конфликтующие снимки заказов: item interactions рассчитаны по первому снимку.")
        if missing_action_customer:
            warnings.append("Часть Actions не содержит customer ID; число клиентов Actions учитывает только известные ID.")
        if customers is None:
            warnings.append("Canonical Customers отсутствует: число профилей неизвестно.")
        if order_aggregates.currency_counts["mixed"] or order_aggregates.currency_counts["unknown"]:
            warnings.append(CURRENCY_WARNING)
        if fractional_lines:
            warnings.append("Позиции заказов с нецелым количеством исключены из статистики.")
        top = sorted(collected.items.items(), key=lambda pair: (-sum(pair[1].values()), pair[0]))[:30]
        diagnostics = (
            ("orders_outside_statistics_period", outside_period),
            ("fractional_quantity_order_lines", fractional_lines),
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
            calculated_at=calculated_at.isoformat(),
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
            gender_distribution=genders, age_distribution=ages, mean_age=mean_age, median_age=median_age,
            view_users=view_users, favorite_users=favorite_users, purchase_users=purchase_users,
            view_purchase_users=sum(n for mask, n in user_types.items() if mask & 5 == 5),
            favorite_purchase_users=sum(n for mask, n in user_types.items() if mask & 6 == 6),
            all_interaction_type_users=user_types[7], repeat_buyers=repeat_buyers,
            interaction_activity_distribution=_distribution(activity, activity_labels),
            purchase_order_distribution=_distribution(order_counts, order_labels),
            mean_orders_per_buyer=sum(collected.purchase_orders.values()) / purchase_users if purchase_users else 0.0,
            median_orders_per_buyer=float(median(collected.purchase_orders.values())) if purchase_users else 0.0,
            repeat_buyer_rate=100 * repeat_buyers / purchase_users if purchase_users else 0.0,
            active_buyer_rate=100 * purchase_users / len(collected.users) if collected.users else 0.0,
            **order_aggregates.result(),
        )
