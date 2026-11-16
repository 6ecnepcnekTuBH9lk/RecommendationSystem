"""Canonical and schema 4 adapters for the isolated aggregate calculation."""

from collections.abc import Iterable
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
import sqlite3

from Application import dataset_statistics as legacy
from Application.interactions import classify_action_system_name
from Application.mindbox.adapters import adapt_action, adapt_action_system_name
from Application.mindbox.adapters._common import AdapterError, birth_date, identifier, objects, timestamp
from Application.mindbox.selection import MindboxSelectionConfig
from Application.product_resolution import ProductResolver
from Application.product_statistics import ProductMetadata
from Application.statistics_index import BuildGuard, GIB, manifest_digest
from Application.statistics_period import shared_intervals
from Application.statistics_storage_safety import StatisticsIndexError, reject_reparse, validate_storage_root
from Application.statistics_unified_aggregates import ActionEvidence, AggregateOrder, accumulate_aggregates
from Application.statistics_unified_facts import AnalyticsContext, reconcile_order
from Application.statistics_unified_index import (
    _configure, _identities, _inputs, _readonly, _records, _separate, _staging,
    open_unified_index, reference_order_reader,
)
from Application.statistics_unified_policy import PurchasePolicy
from Application.statistics_unified_projection import EPOCH


def _effective(provenance, name):
    return shared_intervals([], [(entry["since"], entry["until"])
                                 for kind, entry, _ in provenance["parts"] if kind == name])


def _period(value, coverage):
    if value is None:
        return (coverage[0][0], coverage[-1][1]) if coverage else None
    start, end = value
    if any(not isinstance(part, datetime) or part.utcoffset() is None for part in value) or start >= end:
        raise ValueError("Unified periods must be aware, nonempty half-open intervals")
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _contexts(reader, *, calculated_at, order_period, buyer_period, customer_period, selected_items):
    provenance = reader.manifest["binding"]["provenance"]
    if calculated_at.utcoffset() is None:
        raise ValueError("Unified calculation requires an aware timestamp")
    calculated_at = calculated_at.astimezone(timezone.utc)
    orders, actions = _effective(provenance, "orders"), _effective(provenance, "actions")
    def intervals(rows):
        return tuple((start.isoformat(), end.isoformat()) for start, end in rows)

    common = shared_intervals(intervals(actions), intervals(orders))
    identity = reader.identity_space
    digest = manifest_digest(provenance)
    filter_key = None if selected_items is None else manifest_digest(sorted(selected_items))

    def context(name, period, order_coverage, action_coverage, product=None):
        return AnalyticsContext(name, PurchasePolicy(), identity, calculated_at, None, period, product,
                                order_coverage, action_coverage, "DECLARED_ORDERS_PROXY_V1", False, False, digest)

    order_bounds = _period(order_period, common)
    return (
        context("raw_orders", None, orders, actions),
        context("order_product_slice", order_bounds, common if orders else (), common if actions else (), filter_key),
        context("item_interactions", order_bounds, common if orders else (), common if actions else (), filter_key),
        context("buyer_period", _period(buyer_period, orders), orders, (), None),
        context("customer_population", _period(customer_period, ()) if customer_period else None, orders, actions),
    )


def _orders(reader, predicate, scope, check, *, include_products=True, include_line_history=False):
    with closing(reader.orders()) as rows:
        for key, group in groupby(rows, key=lambda row: row.observation.order_key.value):
            check()
            if key is None:
                for row in group:
                    check()
                    yield AggregateOrder((row.observation,), None, None)
                continue
            observations = []
            rank = None
            for row in group:
                check()
                if rank is not None and row.order_sort_rank != rank:
                    raise StatisticsIndexError("INVALID_OBSERVATION_PROJECTION")
                rank = row.order_sort_rank
                observations.append(row.observation)
            whole = reconcile_order(observations, order_sort_rank=rank, include_products=include_products,
                                    include_line_history=include_line_history, check=check)
            date = whole.order_created_at.value
            in_slice = (predicate is not None and date is not None and scope.period is not None and scope.period[0] <= date < scope.period[1]
                        and any(start <= date < end for start, end in scope.orders_intervals))
            selected = whole if predicate is None or not in_slice else reconcile_order(
                observations, order_sort_rank=rank, product_predicate=predicate, include_products=True,
                include_line_history=include_line_history, check=check)
            yield AggregateOrder(tuple(observations), whole, selected)


def _action_evidence(kind, date, owner, items, *, invalid=False, details=None, has_product=None, system_name=None):
    if invalid:
        return ActionEvidence(owner, date, kind, (), "ACTION_INVALID", has_product=has_product, system_name=system_name)
    if kind is None:
        return ActionEvidence(owner, date, None, (), "ACTION_UNMAPPED", has_product=has_product, system_name=system_name)
    if owner is None:
        return ActionEvidence(None, date, kind, (), "ACTION_OWNER_MISSING", has_product=has_product, system_name=system_name)
    return ActionEvidence(owner, date, kind, items, None if items else "ACTION_ITEMLESS", details, has_product, system_name)


def _reference_actions(root, provenance, identities, resolver, selection, check):
    rules = selection.interaction_rules()
    for _, raw in _records(root, provenance, "actions", check):
        try:
            occurred = timestamp(raw, "dateTimeUtc", required=False)
        except AdapterError:
            occurred = None
        kind = owner = has_product = system_name = None
        try:
            system_name = adapt_action_system_name(raw)
            kind = classify_action_system_name(system_name, rules)
            has_product = bool(objects(raw, "products"))
            source = identifier(raw, "customer.ids.mindboxId", required=False)
            owner = identities.resolve(source) if source is not None else None
            if kind is None or owner is None:
                yield _action_evidence(kind, occurred, owner, (), has_product=has_product, system_name=system_name)
                continue
            action = adapt_action(raw, identities, product_namespaces=selection.action_product_namespaces)
            items = tuple(resolver.resolve(key, strict=False).item_id for key in action.products)
        except AdapterError:
            yield _action_evidence(kind, occurred, owner, (), invalid=True, has_product=has_product, system_name=system_name)
            continue
        yield _action_evidence(kind, occurred, owner, items, details=action, has_product=has_product, system_name=system_name)


def _indexed_actions(reader, selection, check, *, period=None, selected_items=None):
    rules = selection.interaction_rules()
    where, values = "", ()
    if period is not None:
        from Application.statistics_facts import microseconds
        where = "WHERE (a.event_us IS NULL OR (a.event_us>=? AND a.event_us<?))"
        values = tuple(map(microseconds, period))
    if selected_items is not None:
        # Keep all positions of each matched event for availability and price
        # evidence. Unresolved positions still affect product-scope quality.
        where = (where + " AND " if where else "WHERE ") + """(
            a.event_us IS NULL OR a.header_bad OR a.adapter_bad OR a.customer IS NULL OR EXISTS (
                SELECT 1 FROM action_products candidate
                WHERE candidate.event=a.id AND (candidate.item IS NULL OR candidate.item IN (
                    SELECT value FROM json_each(?)
                ))
            )
        )"""
        values += (json.dumps(sorted(selected_items)),)
    with closing(reader.db.execute(f"""
        SELECT a.id,a.event_us,a.name,a.customer,a.header_bad,a.adapter_bad,p.ordinal,p.item,a.details,p.namespace,a.has_product,p.value
        FROM actions a LEFT JOIN action_products p ON a.id=p.event {where} ORDER BY a.id,p.ordinal
    """, values)) as rows:
        for _, group in groupby(rows, key=lambda row: row[0]):
            check()
            first = next(group)
            occurred = None if first[1] is None else EPOCH + timedelta(microseconds=first[1])
            kind = classify_action_system_name(first[2], rules) if first[2] is not None else None
            owner = None if first[3] is None else str(first[3])
            items = ([first[7]] if first[6] is not None else [])
            keys = [(first[9], first[11])] if first[6] is not None else []
            for row in group:
                check()
                items.append(row[7])
                keys.append((row[9], row[11]))
            invalid = bool(first[4] or (first[5] and owner is not None and kind is not None))
            if invalid or kind is None or owner is None or not items or occurred is None:
                yield _action_evidence(kind, occurred, owner, tuple(items), invalid=invalid,
                                       has_product=None if first[4] else bool(first[10]), system_name=first[2])
                continue
            details = json.loads(first[8])
            action = SimpleNamespace(event_datetime_utc=occurred, products=tuple(
                SimpleNamespace(namespace=namespace, value=value) for namespace, value in keys),
                channel_system_name=details.get("channel_system_name"), channel_external_id=details.get("channel_external_id"),
                channel_name=details.get("channel_name"), product_view_is_available=details.get("available"),
                product_view_price=Decimal(details["price"]) if details.get("price") is not None else None)
            yield _action_evidence(kind, occurred, owner, tuple(items), invalid=invalid, details=action,
                                   has_product=None if first[4] else bool(first[10]), system_name=first[2])


def _indexed_action_headers(accumulator, reader, selection, *, period, customer_period, check):
    """Populate exact source counts and population independently of product evidence."""
    from Application.statistics_facts import microseconds

    rules = selection.interaction_rules()
    where, parameters = ("", ()) if period is None else (
        "WHERE event_us>=? AND event_us<?", tuple(map(microseconds, period)))
    for name, header_bad, has_product, count in reader.db.execute(f"""
        SELECT name,header_bad,has_product,COUNT(*) FROM actions {where}
        GROUP BY name,header_bad,has_product ORDER BY MIN(id)
    """, parameters):
        check()
        kind = classify_action_system_name(name, rules) if name is not None else None
        accumulator.source_actions["all"] += count
        if name:
            accumulator.event_names[kind.value if kind is not None else "UNMAPPED", name] += count
        if kind is None:
            accumulator.source_actions["classification_unknown" if header_bad else "unmapped"] += count
        else:
            accumulator.source_actions[kind.value] += count
            if header_bad:
                accumulator.source_actions["mapped_product_unknown"] += count
            elif not has_product:
                accumulator.source_actions["mapped_without_product"] += count

    where, parameters = ("", ()) if customer_period is None else (
        "AND event_us>=? AND event_us<?", tuple(map(microseconds, customer_period)))
    with closing(reader.db.execute(
            f"SELECT DISTINCT customer FROM actions WHERE customer IS NOT NULL {where}", parameters)) as owners:
        for (owner,) in owners:
            check()
            accumulator.members.add("population", str(owner))
    if customer_period is not None and reader.db.execute(
            "SELECT 1 FROM actions WHERE event_us IS NULL AND customer IS NOT NULL LIMIT 1").fetchone():
        from Application.statistics_unified_policy import EvidenceQuality
        accumulator.population_issues.add((EvidenceQuality.INSUFFICIENT_COVERAGE, "ACTION_DATE_UNRESOLVED"))


def _reference_profiles(root, identities, check):
    path = root / "canonical/customers.sqlite"
    if path.is_file():
        with closing(_readonly(path)) as db:
            for (source,) in db.execute("SELECT id FROM profiles ORDER BY id"):
                check()
                yield identities.resolve(source)


def _reference_demographics(root, identities, check):
    path = root / "canonical/customers.sqlite"
    if path.is_file():
        with closing(_readonly(path)) as db:
            for source, birthday, sex in db.execute(
                    "SELECT id,json_extract(raw,'$.birthDate'),json_extract(raw,'$.sex') FROM profiles ORDER BY id"):
                check()
                try:
                    born = birth_date({"birthDate": birthday})
                except AdapterError:
                    born = None
                yield identities.resolve(source), born, sex


def _indexed_profiles(reader):
    with closing(reader.profile_links()) as rows:
        for _, sid in rows:
            yield str(sid)


def _indexed_metadata(db):
    return {code: ProductMetadata(name, category, gender, collection, style, kind)
            for code, name, category, gender, collection, style, kind in db.execute(
                "SELECT i.code,m.name,m.category,m.sex,i.collection,m.style_group,i.kind "
                "FROM items i JOIN item_metadata m USING(code)")}


def _calculate(reader, root, scratch_dir, expected_bytes, cancelled, metadata, actions, profiles, *,
               calculated_at, order_period, buyer_period, customer_period, selected_items, observer):
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("Positive aggregation workspace estimate required")
    if selected_items is not None:
        selected_items = frozenset(selected_items)
    predicate = None if selected_items is None else selected_items.__contains__
    contexts = _contexts(reader, calculated_at=calculated_at, order_period=order_period, buyer_period=buyer_period,
                         customer_period=customer_period, selected_items=selected_items)
    scratch = reject_reparse(scratch_dir)
    _separate(scratch, root)
    with _staging(scratch, validate_storage_root(scratch)) as (folder, _, _):
        guard = BuildGuard(folder, cancelled, observer, 20 * GIB)
        guard.check(force=True, extra=expected_bytes)
        try:
            with closing(sqlite3.connect(folder / "statistics.sqlite")) as db:
                _configure(db, guard, cache_kib=32768)
                with closing(_orders(reader, predicate, contexts[1], guard.check)) as orders:
                    result = accumulate_aggregates(
                        db, contexts=contexts, orders=orders, actions=actions, profile_keys=profiles,
                        metadata=metadata, product_predicate=predicate, action_period=order_period, check=guard.check,
                    )
                guard.check(force=True)
                return result
        except sqlite3.Error:
            if guard.failure is not None:
                raise guard.failure from None
            raise StatisticsIndexError("UNIFIED_AGGREGATION_FAILED") from None


def aggregate_reference(*, raw_root, catalog_path, scratch_dir, expected_bytes, calculated_at,
                        order_period=None, buyer_period=None, customer_period=None, selected_items: Iterable[str] | None = None,
                        cancelled=None, observer=None):
    """REFERENCE reads canonical Actions independently and groups Orders in a private spool."""
    root, catalog = _inputs(raw_root, catalog_path)
    with reference_order_reader(raw_root=root, catalog_path=catalog, scratch_dir=scratch_dir,
                                expected_bytes=expected_bytes, cancelled=cancelled, observer=observer) as reader:
        provenance = reader.manifest["binding"]["provenance"]
        identities = _identities(root, provenance, reader.check)
        products, metadata, _, _ = legacy._catalog_snapshot(catalog)
        selection = MindboxSelectionConfig(**provenance["canonical"]["selection"])
        with closing(_reference_actions(root, provenance, identities, ProductResolver(products),
                                        selection, reader.check)) as actions, closing(
                _reference_profiles(root, identities, reader.check)) as profiles:
            return _calculate(reader, root, scratch_dir, expected_bytes, cancelled, metadata, actions, profiles,
                              calculated_at=calculated_at, order_period=order_period, buyer_period=buyer_period,
                              customer_period=customer_period, selected_items=selected_items, observer=observer)


def aggregate_indexed(*, raw_root, catalog_path, index_dir, scratch_dir, expected_bytes, calculated_at,
                      order_period=None, buyer_period=None, customer_period=None, selected_items: Iterable[str] | None = None,
                      cancelled=None, observer=None):
    """INDEXED reads schema 4 observations and the pinned Actions/catalog projection."""
    def check():
        if cancelled and cancelled():
            raise InterruptedError("Unified aggregation cancelled")

    check()
    with open_unified_index(raw_root=raw_root, catalog_path=catalog_path, index_dir=index_dir, check=check) as reader:
        provenance = reader.manifest["binding"]["provenance"]
        selection = MindboxSelectionConfig(**provenance["canonical"]["selection"])
        with closing(_indexed_actions(reader, selection, check)) as actions, closing(_indexed_profiles(reader)) as profiles:
            return _calculate(reader, Path(raw_root).absolute(), scratch_dir, expected_bytes, cancelled,
                              _indexed_metadata(reader.db), actions, profiles, calculated_at=calculated_at,
                              order_period=order_period, buyer_period=buyer_period, customer_period=customer_period,
                              selected_items=selected_items, observer=observer)
