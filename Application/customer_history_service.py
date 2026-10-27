"""Read-only, bounded logical-event pages from an explicitly built history index.

customer_id is already canonical: this API does not resolve aliases or infer that
an unknown customer exists. Dates accept aware datetime values, normalized to UTC;
the interval is [since, until). A logical event matches when ONE variant belonging
to the requested candidate customer matches ALL filters. All variants are then
returned, including other candidate owners, explicitly marked as uncertain.

Sort: common event time descending, then event_key descending. Unknown common
times sort last; their key order is technical, never a claimed historical date.
READY/EMPTY describe indexed coverage only, never lifetime activity. PARTIAL means
coverage is incomplete for the request or this page contains conflicting variants.
Normal source-quality flags are retained independently of status.

Cursor v1 is an opaque consistency token, not an authorization credential. It
binds generation, customer, normalized filters and limit. Page RAM scales with
limit and the complete payloads (variants/lines/origins) of those logical events,
not with the customer's total history. A single large order is not truncated.
"""

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3

from Application import customer_history_index as index
from Application import customer_history_records as records
from Application.mindbox.adapters._common import AdapterError, identifier
from Application.mindbox.records import ProductKey


MAX_PAGE_SIZE = 200


class HistoryStatus(str, Enum):
    READY = "READY"
    EMPTY = "EMPTY"
    PARTIAL = "PARTIAL"
    INVALID_INPUT = "INVALID_INPUT"
    INDEX_NOT_READY = "INDEX_NOT_READY"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SOURCE_INCONSISTENT = "SOURCE_INCONSISTENT"
    STALE_CURSOR = "STALE_CURSOR"


@dataclass(frozen=True, slots=True, repr=False)
class HistoryVariant:
    """Every source occurrence; event.provenance is the first deterministic origin."""
    event: records.HistoryEvent
    origins: tuple[records.HistoryProvenance, ...]


@dataclass(frozen=True, slots=True, repr=False)
class LogicalHistoryEvent:
    event_key: str
    source: records.HistorySource
    source_event_id: str
    canonical_customer_id: str | None
    candidate_customer_ids: tuple[str, ...]
    kind: records.HistoryKind | None
    event_datetime_utc: datetime | None
    variants: tuple[HistoryVariant, ...]
    occurrence_count: int
    owner_conflict: bool
    time_conflict: bool
    kind_conflict: bool
    quality: tuple[records.HistoryQuality, ...]

    @property
    def has_conflicts(self):
        return len(self.variants) > 1

    @property
    def membership_certain(self):
        return not self.owner_conflict


@dataclass(frozen=True, slots=True, repr=False)
class HistoryPage:
    status: HistoryStatus
    canonical_customer_id: str | None = None
    events: tuple[LogicalHistoryEvent, ...] = ()
    next_cursor: str | None = None
    coverage: tuple[index.HistoryCoverage, ...] = ()
    generation: str | None = None
    source_version: str | None = None
    coverage_complete: bool = False
    has_conflicts: bool = False
    quality: tuple[records.HistoryQuality, ...] = ()


def _time(value):
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("History range requires aware datetime")
    return index._micros(value)


def _datetime(value):
    return None if value is None else index._EPOCH + timedelta(microseconds=value)


def _decimal(value):
    return None if value is None else Decimal(value)


def _boolean(value, *, optional=False):
    if value is None and optional:
        return None
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Invalid stored history boolean")
    return bool(value)


def _quality(value):
    return tuple(records.HistoryQuality(item) for item in json.loads(value))


def _flags(events):
    return tuple(sorted({flag for event in events for flag in event.quality}, key=lambda flag: flag.value))


def _request(customer_id, kinds, since, until, limit):
    customer = identifier({"value": customer_id}, "value")
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError("Invalid page limit")
    if kinds is None:
        selected = tuple(kind.value for kind in records.HistoryKind)
    else:
        if not isinstance(kinds, (tuple, list, set, frozenset)) or not kinds:
            raise ValueError("Invalid history kinds")
        selected = tuple(sorted({records.HistoryKind(kind).value for kind in kinds}))
    selected = tuple(sorted(selected))
    start, end = _time(since), _time(until)
    if start is not None and end is not None and start >= end:
        raise ValueError("Invalid history range")
    return customer, selected, start, end, limit


def _encode_cursor(generation, request, row):
    payload = index._json([1, generation, request, row["event_us"], row["event_key"]]).encode("ascii")
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest() + payload).decode("ascii")


def _decode_cursor(cursor):
    if not isinstance(cursor, str) or not 1 <= len(cursor) <= 4096:
        raise ValueError("Invalid cursor")
    raw = base64.b64decode(cursor.encode("ascii"), altchars=b"-_", validate=True)
    payload = raw[32:]
    if hashlib.sha256(payload).digest() != raw[:32]:
        raise ValueError("Invalid cursor checksum")
    result = json.loads(payload)
    if not isinstance(result, list) or len(result) != 5 or type(result[0]) is not int or result[0] != 1:
        raise ValueError("Invalid cursor version")
    _, generation, request, stamp, key = result
    if (not isinstance(generation, str) or len(generation) != 32
            or any(c not in "0123456789abcdef" for c in generation)
            or not isinstance(request, list) or len(request) != 5
            or (stamp is not None and type(stamp) is not int)):
        raise ValueError("Invalid cursor shape")
    records._digest(key)
    if stamp is not None:
        _datetime(stamp)
    return result


def _page_query(request, last):
    customer, kinds, since, until, limit = request
    conditions, values = [], []
    variant_conditions = ["v.event_key=e.event_key", "v.canonical_customer_id=?"]
    variant_values = [customer]
    if len(kinds) != len(records.HistoryKind):
        marks = ",".join("?" for _ in kinds)
        conditions.append(f"(e.kind IN ({marks}) OR e.kind IS NULL)")
        values.extend(kinds)
        variant_conditions.append(f"v.kind IN ({marks})")
        variant_values.extend(kinds)
    for value, operator in ((since, ">="), (until, "<")):
        if value is not None:
            variant_conditions.append(f"v.event_us{operator}?")
            variant_values.append(value)
    if len(variant_conditions) > 2:
        # Date+kind+candidate owner must match on ONE variant.
        conditions.append("EXISTS (SELECT 1 FROM variants v WHERE " + " AND ".join(variant_conditions) + ")")
        values.extend(variant_values)
    suffix = " AND " + " AND ".join(conditions) if conditions else ""
    base = ("SELECT e.*,ec.event_us AS link_us FROM event_customers ec JOIN events e USING(event_key) "
            "WHERE ec.canonical_customer_id=?" + suffix)
    order = " ORDER BY ec.event_us DESC,ec.event_key DESC LIMIT ?"
    if last is None and since is None and until is None:
        return base + order, [customer, *values, limit + 1]
    # Separate known/unknown dates to retain B-tree seeks instead of a broad OR.
    # NULL belongs last in DESC SQLite ordering, with no invented timestamp.
    branches, args = [], []
    if last is None or last[0] is not None:
        extra, bounds = ["ec.event_us IS NOT NULL"], []
        for value, operator in ((since, ">="), (until, "<")):
            if value is not None:
                extra.append(f"ec.event_us{operator}?")
                bounds.append(value)
        if last is not None:
            extra.append("(ec.event_us,ec.event_key)<(?,?)")
            bounds.extend(last)
        branches.append("SELECT * FROM (" + base + " AND " + " AND ".join(extra) + order + ")")
        args.extend([customer, *values, *bounds, limit + 1])
    unknown = base + " AND ec.event_us IS NULL"
    bounds = []
    if last is not None and last[0] is None:
        unknown += " AND ec.event_key<?"
        bounds.append(last[1])
    branches.append("SELECT * FROM (" + unknown + order + ")")
    args.extend([customer, *values, *bounds, limit + 1])
    sql = "SELECT * FROM (" + " UNION ALL ".join(branches) + ") ORDER BY event_us DESC,event_key DESC LIMIT ?"
    return sql, [*args, limit + 1]


def _channel(row):
    return records.HistoryChannel(*(row[name] for name in
                                   ("channel_system_name", "channel_external_id", "channel_mindbox_id", "channel_name")))


def _references(connection, variant):
    keys = {}
    for row in connection.execute("SELECT * FROM product_keys WHERE variant_key=? ORDER BY role,ordinal,key_ordinal", (variant,)):
        keys.setdefault((row["role"], row["ordinal"]), []).append(ProductKey(row["namespace"], row["value"]))
    refs = {}
    for row in connection.execute("SELECT * FROM product_occurrences WHERE variant_key=? ORDER BY role,ordinal", (variant,)):
        selected = None if row["selected_namespace"] is None else ProductKey(row["selected_namespace"], row["selected_value"])
        refs[row["role"], row["ordinal"]] = records.HistoryProductReference(
            tuple(keys.pop((row["role"], row["ordinal"]), ())), selected, _quality(row["quality"]))
    if keys:
        raise ValueError("Orphaned history product keys")
    return refs


def _variant(connection, common, row):
    key = row["variant_key"]
    origins = tuple(records.HistoryProvenance(
        origin["source_kind"], origin["source_key"], record_ordinal=origin["record_ordinal"],
        observed_at_utc=_datetime(origin["observed_us"])) for origin in connection.execute(
            "SELECT s.source_kind,s.source_key,s.observed_us,r.record_ordinal FROM record_origins r "
            "JOIN sources s USING(source_key) WHERE r.variant_key=? ORDER BY s.source_key,r.record_ordinal", (key,)))
    if not origins:
        raise ValueError("Missing variant provenance")
    references = _references(connection, key)
    if common["source"] == "ACTION":
        if any(role not in ("product", "category") for role, _ in references):
            raise ValueError("Invalid action reference role")
        detail = connection.execute("SELECT * FROM actions WHERE variant_key=?", (key,)).fetchone()
        products = tuple(records.ActionProductOccurrence(ordinal, ref)
                         for (role, ordinal), ref in references.items() if role == "product")
        categories = tuple(records.ActionProductOccurrence(ordinal, ref)
                           for (role, ordinal), ref in references.items() if role == "category")
        details = records.ActionDetails(
            detail["action_id"], detail["system_name"], _datetime(detail["creation_us"]), _channel(detail),
            products, categories, detail["action_name"], detail["brand_system_name"], _decimal(detail["view_price"]),
            _boolean(detail["view_available"], optional=True))
    else:
        detail = connection.execute("SELECT * FROM order_snapshots WHERE variant_key=?", (key,)).fetchone()
        order = records.OrderDetails(common["source_event_id"], detail["retail_order_id"],
                                     _datetime(detail["order_us"]), _channel(detail),
                                     _decimal(detail["total_price"]), _decimal(detail["delivery_cost"]))
        lines = tuple(records.OrderLineDetails(
            line["line_key"], line["source_line_key"], line["line_id"], line["source_line_id"], _boolean(line["is_synthetic"]),
            line["ordinal"], line["line_number"], references["line", line["ordinal"]], line["product_name"],
            _decimal(line["quantity"]), line["raw_status"], _decimal(line["price_of_line"]),
            _decimal(line["base_price_per_item"]), _quality(line["quality"])) for line in connection.execute(
                "SELECT * FROM order_lines WHERE variant_key=? ORDER BY ordinal", (key,)))
        if len(lines) != len(references):
            raise ValueError("Order references disagree")
        details = records.OrderSnapshot(common["event_key"], detail["snapshot_key"], row["fingerprint"], order, lines, origins[0])
    event = records.HistoryEvent(
        common["event_key"], records.HistorySource(common["source"]), common["source_event_id"],
        row["canonical_customer_id"], row["source_customer_id"], records.HistoryKind(row["kind"]),
        _datetime(row["event_us"]), row["fingerprint"], origins[0], details, _quality(row["quality"]))
    return HistoryVariant(event, origins)


def _logical_event(connection, row):
    if row["link_us"] != row["event_us"]:
        raise ValueError("History pagination link date disagrees")
    variants = tuple(_variant(connection, row, variant) for variant in connection.execute(
        "SELECT * FROM variants WHERE event_key=? ORDER BY variant_key", (row["event_key"],)))
    candidates = tuple(item[0] for item in connection.execute(
        "SELECT canonical_customer_id FROM event_customers WHERE event_key=? ORDER BY canonical_customer_id", (row["event_key"],)))
    events = tuple(variant.event for variant in variants)
    owners = {event.canonical_customer_id for event in events}
    stamps = {event.event_datetime_utc for event in events}
    kinds = {event.kind for event in events}
    if (not variants or len(variants) != row["variant_count"] or set(candidates) != owners
            or _boolean(row["owner_conflict"]) != (len(owners) > 1)
            or _boolean(row["time_conflict"]) != (len(stamps) > 1)
            or _boolean(row["kind_conflict"]) != (len(kinds) > 1)
            or row["canonical_customer_id"] != (next(iter(owners)) if len(owners) == 1 else None)
            or _datetime(row["event_us"]) != (next(iter(stamps)) if len(stamps) == 1 else None)
            or row["kind"] != (next(iter(kinds)).value if len(kinds) == 1 else None)
            or row["occurrence_count"] != sum(len(variant.origins) for variant in variants)):
        raise ValueError("Logical history consensus disagrees")
    return LogicalHistoryEvent(
        row["event_key"], records.HistorySource(row["source"]), row["source_event_id"], row["canonical_customer_id"],
        candidates, None if row["kind"] is None else records.HistoryKind(row["kind"]), _datetime(row["event_us"]),
        variants, row["occurrence_count"], bool(row["owner_conflict"]), bool(row["time_conflict"]),
        bool(row["kind_conflict"]), _flags(events))


def _coverage_complete(coverage, request):
    _, kinds, since, until, _ = request
    components = set()
    if "ORDER" in kinds:
        components.add("orders")
    if any(kind != "ORDER" for kind in kinds):
        components.add("actions")
    for component in components:
        intervals = sorted((index._micros(c.since_utc), index._micros(c.until_utc))
                           for c in coverage if c.component == component)
        if not intervals:
            return False
        start = intervals[0][0] if since is None else since
        end = intervals[-1][1] if until is None else until
        if start >= end:
            return False
        reached = start
        for left, right in intervals:
            if right <= reached:
                continue
            if left > reached:
                break
            reached = right
            if reached >= end:
                break
        if reached < end:
            return False
    return True


class CustomerHistoryService:
    def __init__(self, *, cache_dir=index.DEFAULT_CACHE_DIR):
        self._index = index.CustomerHistoryIndexService(cache_dir=cache_dir)

    def get_customer_history(self, customer_id, *, kinds=None, since=None, until=None, limit=50,
                             cursor=None, raw_root=index.DEFAULT_RAW_ROOT) -> HistoryPage:
        try:
            request = _request(customer_id, kinds, since, until, limit)
            decoded = None if cursor is None else _decode_cursor(cursor)
            root = Path(raw_root).resolve()
        except (ValueError, TypeError, AdapterError, OverflowError, binascii.Error):
            return HistoryPage(HistoryStatus.INVALID_INPUT)
        customer = request[0]
        def failure(status):
            return HistoryPage(status, customer)
        path = self._index.index_path(raw_root=root)
        try:
            if not path.exists():
                return failure(HistoryStatus.INDEX_NOT_READY)
            with index._reader_lock(root):
                sources = index._capture_sources(root)
                try:
                    version = index._database_version(path)
                    with index._read_index(path) as connection:
                        ready = index._inspect_connection(connection, sources)
                        if decoded is not None and (decoded[1] != ready.generation
                                                    or decoded[2] != json.loads(index._json(request))):
                            return failure(HistoryStatus.STALE_CURSOR)
                        connection.row_factory = sqlite3.Row
                        sql, args = _page_query(request, None if decoded is None else decoded[3:])
                        rows = connection.execute(sql, args).fetchall()
                        events = tuple(_logical_event(connection, row) for row in rows[:limit])
                        next_cursor = _encode_cursor(ready.generation, request, rows[limit - 1]) if len(rows) > limit else None
                    if version != index._database_version(path):
                        return failure(HistoryStatus.INDEX_NOT_READY)
                except (sqlite3.Error, ValueError, TypeError, KeyError, IndexError, OverflowError, AdapterError,
                        ArithmeticError, OSError, index._Stale):
                    return failure(HistoryStatus.INDEX_NOT_READY)
                if index._capture_sources(root).version != sources.version:
                    return failure(HistoryStatus.INDEX_NOT_READY)
                complete = _coverage_complete(ready.coverage, request)
                conflicts = any(event.has_conflicts for event in events)
                status = HistoryStatus.PARTIAL if conflicts or not complete else (HistoryStatus.READY if events else HistoryStatus.EMPTY)
                return HistoryPage(status, customer, events, next_cursor, ready.coverage, ready.generation,
                                   ready.source_version, complete, conflicts, _flags(events))
        except Exception as exc:
            # External storage helpers wrap errors. Never echo paths/IDs/raw text.
            return failure(HistoryStatus.SOURCE_UNAVAILABLE if index._unavailable(exc) else HistoryStatus.SOURCE_INCONSISTENT)


_DEFAULT_SERVICE = CustomerHistoryService()


def get_customer_history(customer_id, *, kinds=None, since=None, until=None, limit=50, cursor=None,
                         raw_root=index.DEFAULT_RAW_ROOT) -> HistoryPage:
    return _DEFAULT_SERVICE.get_customer_history(customer_id, kinds=kinds, since=since, until=until,
                                                 limit=limit, cursor=cursor, raw_root=raw_root)
