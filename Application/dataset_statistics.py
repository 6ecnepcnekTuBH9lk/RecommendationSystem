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
from Application.customer_statistics_scope import CustomerScope


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
    source_actions: int  # Date-scoped canonical events, before product selection.
    total_interactions: int  # Accepted VIEW + FAVORITE + PURCHASE item interactions.
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
    # Optional snapshot metadata; legacy results have no paths. Seven category
    # aggregate fields above retain their positions and numerical semantics.
    product_category_paths: tuple[tuple[str, tuple[str, ...] | None], ...] | None = None
    customer_scope: dict | None = None


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


def _customer_snapshot(root, as_of, check, progress, cohort=None, identities=None, scope=None):
    from Application.customer_demographics import AGE_LABELS, age_group, calendar_age

    path = database(root)
    genders, ages, groups = Counter(), Counter(), Counter()
    age_labels = AGE_LABELS
    count, coverage = None, Coverage("Customers")
    if not path.exists():
        return count, coverage, _distribution(genders, ("Мужчины", "Женщины", "Не указан")), _distribution(groups, age_labels), None, None
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        count = 0
        if scope is not None:
            scope.profiles_available = True
        # Stream identity and demographics only; never retain raw profiles/contacts.
        seen = set()
        for source_id, birthday, sex in connection.execute(
                "SELECT id, json_extract(raw, '$.birthDate'), json_extract(raw, '$.sex') FROM profiles ORDER BY id"):
            check()
            canonical_id = identities.resolve(source_id) if scope is not None or cohort is not None else source_id
            if scope is not None:
                scope.profile(canonical_id)
            if cohort is not None:
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
                age = calendar_age(born, as_of)
                ages[age] += 1
                groups[age_group(age)] += 1
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


def _category_reference(path):
    """Names and ordered paths from one CSV read; never aggregate ancestors."""
    names, parents, ambiguous_parents = {}, {}, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|", strict=True)
        columns = reader.fieldnames or []
        if not {"КодКатегории", "НазваниеКатегории"}.issubset(columns) or len(set(columns)) != len(columns):
            raise CatalogError("Некорректные колонки справочника категорий.")
        has_parents = "КодРодительскойКатегории" in columns
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise CatalogError("Некорректная структура строки справочника категорий.")
            code, name = row["КодКатегории"].strip(), row["НазваниеКатегории"].strip()
            if not code:
                continue
            if code in names and names[code] != name:
                raise CatalogError("Конфликт названий для кода категории.")
            parent = row.get("КодРодительскойКатегории", "").strip()
            if code in parents and parents[code] != parent:
                ambiguous_parents.add(code)
            names[code] = name
            parents[code] = parent
    paths = {}
    for code in sorted(names):
        trail, seen, cursor = [], set(), code
        prefix = ()
        while cursor:
            if cursor in paths:
                prefix = paths[cursor]
                break
            if (not has_parents or cursor in seen or cursor not in names
                    or cursor in ambiguous_parents or not names[cursor]):
                prefix = None
                break
            seen.add(cursor)
            trail.append(cursor)
            cursor = parents[cursor]
        for node in reversed(trail):
            prefix = (*prefix, names[node]) if prefix is not None else None
            paths[node] = prefix
        paths.setdefault(code, None)
    return (MappingProxyType({code: name or code for code, name in names.items()}),
            MappingProxyType(paths))


def _category_names(path):
    """Compatibility helper for the existing exact code/name mapping."""
    return _category_reference(path)[0]


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
        categories, category_paths = _category_reference(categories_path)
        if signature() != before:
            raise CatalogError("Номенклатура или справочник категорий обновились во время чтения. Повторите расчёт.")
        if products.item_ids != metadata.keys():
            raise CatalogError("Метаданные не соответствуют товарам справочника.")
        return products, metadata, categories, category_paths
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
        _, metadata, _, _ = _catalog_snapshot(catalog_path)
        period = shared_intervals(*[[(e["since"], e["until"]) for e in entries[name]]
                                  for name in ("actions", "orders")])
        return options_from_snapshot(metadata, period)


def calculate_statistics(*, raw_root=DEFAULT_RAW_ROOT, catalog_path=DEFAULT_CATALOG_PATH,
                         progress=None, cancelled=None, analysis_filter=AnalysisFilter(),
                         backend="REFERENCE", index_dir=None, calculated_at=None):
    """Scan published canonical data once. No legacy fallback or silent skipping.

    Source Actions are date-scoped, independent of product selection; item totals
    always count accepted builder records. Default order counts/statuses retain
    the source-snapshot baseline. With a filter,
    business orders/statuses use matching lines of unique first snapshots.
    Source dedup diagnostics remain independent of product selection.
    """
    # One instant/local calendar date for the complete operation, including
    # provenance/cache callers that already captured it before expensive work.
    calculated_at = calculated_at or datetime.now(timezone.utc)
    if backend != "REFERENCE":
        if backend != "INDEXED":
            raise ValueError("Unknown statistics backend")
        from Application.statistics_query import calculate_indexed_statistics
        return calculate_indexed_statistics(
            raw_root=raw_root, catalog_path=catalog_path, index_dir=index_dir,
            analysis_filter=analysis_filter, progress=progress, cancelled=cancelled,
            calculated_at=calculated_at)
    root = Path(raw_root)
    if not (root / "canonical/catalog.json").is_file():
        raise ValueError("Нет canonical dataset. Сначала загрузите данные Mindbox.")

    def check():
        if cancelled and cancelled():
            raise InterruptedError("Расчёт статистики отменён")

    with storage_lock(root):
        return _calculate_statistics(root, catalog_path, progress, check, analysis_filter,
                                     calculated_at=calculated_at)


def _calculate_statistics(root, catalog_path, progress, check, analysis_filter, input_source=None, *, calculated_at=None,
                          customer_filter=None):
    """One shared calculation driver. Indexed inputs preserve occurrence order.

    The reference path retains the raw reader/adapters. The optional input source
    supplies validated projections, never an alternative implementation of metrics.
    Caller owns source locking/attestation and the lifetime of its connection.
    """
    calculated_at = calculated_at or datetime.now(timezone.utc)
    as_of = calculated_at.astimezone().date()
    check()
    data, entries = _statistics_sources(root)
    customer_filter = analysis_filter if customer_filter is None else customer_filter
    scope = CustomerScope(bool(entries['actions']), bool(entries['orders']), customer_filter)
    selection = MindboxSelectionConfig(**data["selection"])
    merge = data["customer_merges"]
    warnings = []
    if merge and any(datetime.fromisoformat(e["since"]) < datetime.fromisoformat(merge["since"])
                     or datetime.fromisoformat(e["until"]) > datetime.fromisoformat(merge["until"])
                     for group in entries.values() for e in group):
        warnings.append("CustomerMerges не покрывает весь период: canonical identity может быть неполной.")

    def raw_records(name, sources):
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

    records = input_source.records if input_source is not None else raw_records
    identities = input_source.identities if input_source is not None else CustomerIdResolver(adapt_customer_merge(raw)
                                    for raw in records("customer_merges", [merge] if merge else []))
    builder = InteractionBuilder(selection.interaction_rules())
    product_catalog, metadata, categories, category_paths = _catalog_snapshot(catalog_path)
    period = shared_intervals(*[[(entry["since"], entry["until"]) for entry in entries[name]]
                                for name in ("actions", "orders")])
    if period:
        analysis_filter = options_from_snapshot(metadata, period).normalize(analysis_filter)
    order_scope_period = shared_intervals([], [(entry['since'], entry['until']) for entry in entries['orders']])
    products = ProductResolver(product_catalog)

    def matches(product):
        if not analysis_filter.product_restricted:
            return True
        item = products.resolve(product, strict=False)
        if input_source is not None:
            return item.item_id in input_source.selected_items
        return analysis_filter.matches(metadata.get(item.item_id))

    product_aggregates = ProductAggregates(metadata, categories)
    collected = _Interactions(products)
    accepted_counts = Counter()
    business_orders = business_lines = 0
    action_types, statuses = Counter(), Counter()
    source_classification = Counter()
    action_users, order_users = set(), set()
    accepted_action_users, accepted_order_users = set(), set()
    with_product = missing_action_customer = raw_lines = 0
    action_aggregates = ActionAggregates()
    for raw in records("actions", entries["actions"]):
        if analysis_filter.start_date is not None:
            # Unmapped technical actions may have no timestamp; do not invent one.
            occurred_at = timestamp(raw, "dateTimeUtc", required=False)
            if occurred_at is None or not analysis_filter.contains(occurred_at):
                continue
        name = adapt_action_system_name(raw)
        action_types[name] += 1
        has_product = bool(objects(raw, "products"))
        with_product += has_product
        kind = classify_action_system_name(name, builder.rules)
        source_classification['unmapped' if kind is None else kind.value] += 1
        if kind is not None and not has_product:
            source_classification['itemless'] += 1
        source = identifier(raw, "customer.ids.mindboxId", required=False)
        if source is not None:
            canonical_id = identities.resolve(source)
            action_users.add(canonical_id)
            if (customer_filter.start_date is None or
                    (occurred := timestamp(raw, 'dateTimeUtc', required=False)) is not None and customer_filter.contains(occurred)):
                scope.action(canonical_id, kind, has_product)
        else:
            missing_action_customer += 1
        if kind is None:
            builder.record_unmapped_action(name)
            continue
        if source is None:
            # An event without identity remains in source diagnostics, never
            # creates a synthetic customer or an item interaction.
            continue
        action = (input_source.action(raw) if input_source is not None else
                  adapt_action(raw, identities, product_namespaces=selection.action_product_namespaces))
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

    snapshots = input_source.snapshots() if input_source is not None else OrderSnapshots()
    scope_only_orders = set()  # Only orders selected outside the legacy shared coverage.

    def scope_order(raw, lines):
        identity = identities.resolve(identifier(raw, 'customer.ids.mindboxId'))
        scope.order(identity, any(line.line_status in builder.rules.purchase_line_statuses for line in lines))

    def adapted_lines(raw, eligible):
        return (input_source.order(raw) if input_source is not None else
                adapt_order(order_for_adapter({**raw, 'lines': eligible}), identities,
                            product_namespaces=selection.order_product_namespaces))

    order_aggregates = OrderAggregates()
    outside_period = fractional_lines = 0
    for raw in records("orders", entries["orders"]):
        ordered_at = timestamp(raw, "firstAction.dateTimeUtc")
        scope_eligible = (customer_filter.contains(ordered_at)
                          and any(start <= ordered_at < end for start, end in order_scope_period))
        if not any(start <= ordered_at < end for start, end in period):
            outside_period += 1
            if scope_eligible:
                key = identifier(raw, 'ids.mindboxId')
                if key not in snapshots.fingerprints and key not in scope_only_orders:
                    eligible = [line for line in objects(raw, 'lines', required=True)
                                if (quantity := number(line, 'quantity')) == quantity.to_integral_value()]
                    scope_order(raw, adapted_lines(raw, eligible))
                    scope_only_orders.add(key)
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
        lines = adapted_lines(raw, eligible_lines)
        if scope_eligible and identifier(raw, 'ids.mindboxId') not in scope_only_orders:
            scope_order(raw, lines)
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
        if input_source is not None:
            input_source.add_order(order_aggregates, raw, purchases)
        else:
            order_aggregates.add(raw, purchases)

    check()
    customers, customer_coverage, genders, ages, mean_age, median_age = (input_source.customer_snapshot if input_source is not None else _customer_snapshot)(
        root, as_of, check, progress,
        cohort=set(collected.users) if analysis_filter.active else None, identities=identities, scope=scope)
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
        ("mapped_view_actions", source_classification['VIEW']),
        ("mapped_favorite_actions", source_classification['FAVORITE']),
        ("unmapped_actions", source_classification['unmapped']),
        ("mapped_without_product", source_classification['itemless']),
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
        customer_scope=scope.result(check),
        product_category_paths=tuple((code, category_paths.get(code))
                                     for code, *_ in product_result["product_category_statistics"] if code is not None),
        source_actions=sum(action_types.values()),
        total_interactions=sum(accepted_counts.values()),
        orders=business_orders if analysis_filter.active else snapshots.raw,
        order_lines=business_lines if analysis_filter.active else raw_lines, customers=customers,
        action_customers=len(accepted_action_users if analysis_filter.active else action_users),
        order_customers=len(accepted_order_users if analysis_filter.active else order_users), interaction_users=len(collected.users),
        actions_with_product=with_product, actions_without_product=sum(action_types.values()) - with_product,
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
