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
from types import MappingProxyType

from Application.analysis_filter import AnalysisFilter, options_from_snapshot
from Application.interactions import InteractionBuilder, InteractionBuildError, classify_action_system_name
from Application.mindbox.adapters import adapt_action, adapt_action_system_name, adapt_customer_merge, adapt_order
from Application.mindbox.adapters._common import AdapterError, birth_date, identifier, number, objects, timestamp
from Application.mindbox.canonical_customers import database
from Application.mindbox.canonical_storage import catalog, checked_directory, storage_lock, effective_entries as _entries
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.order_dedup import OrderSnapshots
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, iter_export, part_files
from Application.mindbox.selection import MindboxSelectionConfig
from Application.product_resolution import CatalogError, DEFAULT_CATALOG_PATH, ProductResolver, load_catalog
from Application.order_statistics import CURRENCY_WARNING, OrderAggregates, order_for_adapter
from Application.statistics_period import shared_intervals
from Application.action_statistics import ActionAggregates
from Application.product_statistics import ProductAggregates, ProductMetadata


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
    orders_without_purchase: int
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
    mean_views_per_viewer: float
    median_views_per_viewer: float
    mean_favorites_per_user: float
    median_favorites_per_user: float
    view_user_activity_distribution: tuple[tuple[str, int, float], ...]
    favorite_user_activity_distribution: tuple[tuple[str, int, float], ...]
    # key, name, VIEW interactions/users/rate, FAVORITE interactions/users/rate
    action_channel_statistics: tuple[tuple[str | None, str, int, int, float, int, int, float], ...]
    action_monthly_dynamics: tuple[tuple[str, int, int, int, int], ...]
    view_parameter_actions: int
    view_availability_distribution: tuple[tuple[str, int, float], ...]
    view_price_statistics: tuple[tuple[str, int, str, str], ...]
    products_with_views: int
    products_with_favorites: int
    products_with_purchases: int
    resolved_view_interactions: int
    resolved_favorite_interactions: int
    resolved_purchase_interactions: int
    resolved_purchase_quantity: str
    top_viewed_products: tuple[tuple[str, str, int, int, int, int], ...]
    top_favorited_products: tuple[tuple[str, str, int, int, int, int], ...]
    top_purchased_products: tuple[tuple[str, str, int, int, str, int, int], ...]
    # label, unique items, VIEW/FAVORITE/PURCHASE interactions, purchase quantity
    product_category_statistics: tuple[tuple[str | None, str, int, int, int, int, str], ...]
    product_gender_statistics: tuple[tuple[str, int, int, int, int, str], ...]
    product_season_statistics: tuple[tuple[str, int, int, int, int, str], ...]
    product_style_statistics: tuple[tuple[str, int, int, int, int, str], ...]
    analysis_filter: AnalysisFilter = AnalysisFilter()


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
        return resolved


def _customer_snapshot(root, as_of, check, progress, cohort=None, identities=None):
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
        # Stream identity and demographics only; never retain raw profiles/contacts.
        seen = set()
        for source_id, birthday, sex in connection.execute(
                "SELECT id, json_extract(raw, '$.birthDate'), json_extract(raw, '$.sex') FROM profiles ORDER BY id"):
            check()
            if cohort is not None:
                canonical_id = identities.resolve(source_id)
                if canonical_id not in cohort or canonical_id in seen:
                    continue
                seen.add(canonical_id)
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


def _catalog_metadata(path):
    """Project analytical fields only, preserving the validated catalog's last-row precedence."""
    def value(row, column):
        return (row.get(column) or "").strip() or None

    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|", strict=True)
        if not reader.fieldnames or "КодНоменклатуры" not in reader.fieldnames:
            raise CatalogError("Missing required catalog column: КодНоменклатуры")
        metadata = {}
        for row in reader:
            if None in row or any(cell is None for cell in row.values()):
                raise CatalogError("Invalid catalog row width")
            code = row["КодНоменклатуры"]
            if code.strip():
                metadata[code] = ProductMetadata(
                    value(row, "НазваниеНаСайте") or value(row, "Номенклатура") or "",
                    value(row, "КатегорияНаСайте"), value(row, "ПолНоменклатуры"),
                    value(row, "Коллекция"), value(row, "СтилеваяГруппа"), value(row, "ВидНоменклатуры"))
        return MappingProxyType(metadata)


def _category_names(path):
    """Read only exact category codes/names, never hierarchy or raw rows."""
    names = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|", strict=True)
        columns = reader.fieldnames or []
        if not {"КодКатегории", "НазваниеКатегории"}.issubset(columns) or len(set(columns)) != len(columns):
            raise CatalogError("Некорректные колонки справочника категорий.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise CatalogError("Некорректная структура строки справочника категорий.")
            code, name = row["КодКатегории"].strip(), row["НазваниеКатегории"].strip()
            if not code:
                continue
            if code in names and names[code] != name:
                raise CatalogError("Конфликт названий для кода категории.")
            names[code] = name
    return MappingProxyType({code: name or code for code, name in names.items()})


def _catalog_snapshot(path):
    """Keep resolver IDs, metadata and category names in one reference snapshot."""
    path = Path(path)
    categories_path = path.with_name("site_categories.csv")

    def signature():
        return tuple((stat.st_ino, stat.st_size, stat.st_mtime_ns)
                     for stat in (path.stat(), categories_path.stat()))

    try:
        before = signature()
        products = load_catalog(path)
        metadata = _catalog_metadata(path)
        categories = _category_names(categories_path)
        if signature() != before:
            raise CatalogError("Номенклатура или справочник категорий обновились во время чтения. Повторите расчёт.")
        if products.item_ids != metadata.keys():
            raise CatalogError("Метаданные не соответствуют товарам справочника.")
        return products, metadata, categories
    except (OSError, UnicodeError, csv.Error):
        raise CatalogError("Не удалось прочитать номенклатуру или справочник категорий.") from None


def _statistics_sources(root):
    """Shared source prerequisites for calculation and option discovery, under lock."""
    if not (root / "canonical/catalog.json").is_file():
        raise ValueError("Нет canonical dataset. Сначала загрузите данные Mindbox.")
    data = catalog(root)
    entries = {name: _entries(data, name) for name in ("actions", "orders")}
    if any(entries.values()) and not data["customer_merges"]:
        raise ValueError("Нет canonical CustomerMerges: невозможно подтвердить identity клиентов.")
    return data, entries


def _source_directory(root, name, entry):
    """Shared physical source check for readiness and the statistics reader."""
    directory = checked_directory(root, entry["directory"], name)
    if len(part_files(directory, name)) != entry["parts"]:
        raise ValueError("Canonical parts mismatch")
    return directory


def load_analysis_options(*, raw_root=DEFAULT_RAW_ROOT, catalog_path=DEFAULT_CATALOG_PATH):
    """Reference/manifest-only readiness snapshot; never scan Actions or Orders."""
    root = Path(raw_root)
    if not (root / "canonical/catalog.json").is_file():
        raise ValueError("Для установки отбора необходимо загрузить исходные данные и справочники.")
    with storage_lock(root):
        data, entries = _statistics_sources(root)
        if not all(entries.values()) or not data["customer_merges"] or not database(root).is_file():
            raise ValueError("Для установки отбора необходимо загрузить исходные данные и справочники.")
        for name, sources in (*entries.items(), ("customer_merges", [data["customer_merges"]])):
            for entry in sources:
                _source_directory(root, name, entry)
        _, metadata, _ = _catalog_snapshot(catalog_path)
        period = shared_intervals(*[[(e["since"], e["until"]) for e in entries[name]]
                                  for name in ("actions", "orders")])
        return options_from_snapshot(metadata, period)


def calculate_statistics(*, raw_root=DEFAULT_RAW_ROOT, catalog_path=DEFAULT_CATALOG_PATH,
                         progress=None, cancelled=None, analysis_filter=AnalysisFilter()):
    """Scan published canonical data once. No legacy fallback or silent skipping.

    Default counts/statuses retain the source-snapshot baseline. With a filter,
    business orders/statuses use matching lines of unique first snapshots.
    Source dedup diagnostics remain independent of product selection.
    """
    root = Path(raw_root)
    if not (root / "canonical/catalog.json").is_file():
        raise ValueError("Нет canonical dataset. Сначала загрузите данные Mindbox.")

    def check():
        if cancelled and cancelled():
            raise InterruptedError("Расчёт статистики отменён")

    with storage_lock(root):
        check()
        data, entries = _statistics_sources(root)
        selection = MindboxSelectionConfig(**data["selection"])
        merge = data["customer_merges"]
        warnings = []
        if merge and any(datetime.fromisoformat(e["since"]) < datetime.fromisoformat(merge["since"])
                         or datetime.fromisoformat(e["until"]) > datetime.fromisoformat(merge["until"])
                         for group in entries.values() for e in group):
            warnings.append("CustomerMerges не покрывает весь период: canonical identity может быть неполной.")

        def records(name, sources):
            count = 0
            for entry in sources:
                check()
                directory = _source_directory(root, name, entry)
                for raw in iter_export(name, input_dir=directory):
                    check()
                    count += 1
                    if progress and (count == 1 or count % 10000 == 0):
                        progress(f"{name}: {count:,}".replace(",", " "))
                    yield raw

        identities = CustomerIdResolver(adapt_customer_merge(raw)
                                        for raw in records("customer_merges", [merge] if merge else []))
        builder = InteractionBuilder(selection.interaction_rules())
        product_catalog, metadata, categories = _catalog_snapshot(catalog_path)
        period = shared_intervals(*[[(entry["since"], entry["until"]) for entry in entries[name]]
                                    for name in ("actions", "orders")])
        if period:
            analysis_filter = options_from_snapshot(metadata, period).normalize(analysis_filter)
        products = ProductResolver(product_catalog)

        def matches(product):
            if not analysis_filter.product_restricted:
                return True
            item = products.resolve(product, strict=False)
            return analysis_filter.matches(metadata.get(item.item_id))

        product_aggregates = ProductAggregates(metadata, categories)
        collected = _Interactions(products)
        accepted_counts = Counter()
        business_orders = business_lines = 0
        action_types, statuses = Counter(), Counter()
        action_users, order_users = set(), set()
        accepted_action_users, accepted_order_users = set(), set()
        with_product = missing_action_customer = raw_lines = 0
        action_aggregates = ActionAggregates()
        for raw in records("actions", entries["actions"]):
            if analysis_filter.start_date is not None:
                # Unmapped technical actions may have no timestamp; do not invent one.
                occurred_at = timestamp(raw, "dateTimeUtc", required=False)
                if occurred_at is not None and not analysis_filter.contains(occurred_at):
                    continue
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
            interactions = tuple(item for item in interactions if matches(item.product))
            if not interactions:
                continue
            action_aggregates.add(action, interactions)
            for interaction in interactions:
                accepted_action_users.add(interaction.customer_id)
                accepted_counts[interaction.interaction_type.value] += 1
                resolved = collected.add(interaction)
                if resolved is not None:
                    product_aggregates.add(resolved)

        snapshots = OrderSnapshots()
        order_aggregates = OrderAggregates()
        outside_period = fractional_lines = 0
        for raw in records("orders", entries["orders"]):
            ordered_at = timestamp(raw, "firstAction.dateTimeUtc")
            if not any(start <= ordered_at < end for start, end in period):
                outside_period += 1
                continue
            if not analysis_filter.contains(ordered_at):
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
                if not analysis_filter.active:
                    statuses[identifier(line, "status.ids.externalId")] += 1
            if not snapshots.accept(raw, diagnose=True):
                continue
            buyer = None
            purchases = []
            eligible_order = {**raw, "lines": eligible_lines}
            lines = adapt_order(order_for_adapter(eligible_order), identities, product_namespaces=selection.order_product_namespaces)
            lines = tuple(line for line in lines if matches(line.product))
            if analysis_filter.product_restricted and not lines:
                continue
            business_orders += 1
            business_lines += len(lines)
            accepted_order_users.add(identities.resolve(identifier(raw, "customer.ids.mindboxId")))
            for line in lines:
                if analysis_filter.active:
                    statuses[line.line_status] += 1
                interaction = builder.from_order_line(line)
                if interaction is not None:
                    accepted_counts[interaction.interaction_type.value] += 1
                    resolved = collected.add(interaction)
                    if resolved is not None:
                        product_aggregates.add(resolved)
                    buyer = interaction.customer_id
                    purchases.append(line)
            if buyer is not None:
                collected.purchase_orders[buyer] += 1
            order_aggregates.add(raw, purchases)

        check()
        calculated_at = datetime.now(timezone.utc)
        customers, customer_coverage, genders, ages, mean_age, median_age = _customer_snapshot(
            root, calculated_at.astimezone().date(), check, progress,
            cohort=set(collected.users) if analysis_filter.active else None, identities=identities)
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
        product_result = product_aggregates.result()
        if any(product_result[f"resolved_{kind.value.lower()}_interactions"] != counts.resolved
               for kind, counts in resolution.by_type.items()):
            raise ValueError("Product aggregates disagree with resolution diagnostics")
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
            ("ambiguous_product_view_actions", action_aggregates.ambiguous),
            ("unknown_currency_view_price_actions", action_aggregates.unknown_price_currency),
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
            analysis_filter=analysis_filter,
            actions=sum(accepted_counts.values()) if analysis_filter.active else interactions.actions_total,
            orders=business_orders if analysis_filter.active else snapshots.raw,
            order_lines=business_lines if analysis_filter.active else raw_lines, customers=customers,
            action_customers=len(accepted_action_users if analysis_filter.active else action_users),
            order_customers=len(accepted_order_users if analysis_filter.active else order_users), interaction_users=len(collected.users),
            actions_with_product=with_product, actions_without_product=interactions.actions_total - with_product,
            view_interactions=accepted_counts["VIEW"], favorite_interactions=accepted_counts["FAVORITE"],
            purchase_interactions=accepted_counts["PURCHASE"], purchase_quantity=str(collected.quantity),
            mean_interactions=sum(accepted_counts.values()) / len(collected.users) if collected.users else 0.0,
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
            top_products=tuple((code, metadata[code].name, counts["VIEW"], counts["FAVORITE"],
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
            orders_without_purchase=business_orders - order_aggregates.orders,
            **order_aggregates.result(),
            **action_aggregates.result(),
            **product_result,
        )
