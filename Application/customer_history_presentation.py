"""Bounded history pages as flat product rows; no Qt, SQL or ranking."""

from collections import Counter
from dataclasses import dataclass
from datetime import timezone

from Application.customer_history_records import HistoryKind, OrderSnapshot


CATEGORY_NAMES = {HistoryKind.ORDER: "Заказы", HistoryKind.VIEW: "Просмотры",
                  HistoryKind.FAVORITE: "Избранное", HistoryKind.UNCLASSIFIED: "Прочее"}


@dataclass(frozen=True, slots=True, repr=False)
class HistoryRow:
    values: tuple[str, ...]
    photo_url: str = ""
    notice: str = ""


@dataclass(frozen=True, slots=True, repr=False)
class HistoryDisplayPage:
    page: object
    events: tuple  # (logical event key, product rows); keys never become display text.


def history_date(value):
    return value.astimezone(timezone.utc).strftime("%d.%m.%Y %H:%M") if value is not None else "—"


def _product(reference, catalog, source_name=None):
    name, url, kind, collection = catalog.get(reference.product, ("", "", "", ""))
    code = reference.product.value if reference.product is not None else ", ".join(dict.fromkeys(key.value for key in reference.keys))
    return code or "—", source_name or name or "—", kind or "—", collection or "—", url


def _variant_rows(variant, kind, catalog, notice):
    event, detail = variant.event, variant.event.details
    stamp = history_date(event.event_datetime_utc)
    if isinstance(detail, OrderSnapshot):
        products = ((line.reference, line.product_name) for line in detail.lines)
    else:
        products = ((item.reference, None) for item in detail.products
                    if kind not in (HistoryKind.VIEW, HistoryKind.FAVORITE) or item.reference.keys)
    rows = []
    for reference, source_name in products:
        code, name, product_kind, collection, url = _product(reference, catalog, source_name)
        row = HistoryRow(("", code, name, product_kind, collection, stamp), url, notice)
        # Preserve namespaces when comparing versions, even if the displayed codes coincide.
        rows.append((reference.keys, row))
    if not rows and not isinstance(detail, OrderSnapshot) and kind not in (HistoryKind.VIEW, HistoryKind.FAVORITE):
        rows.append(((), HistoryRow(("", "—", "—", "—", "—", stamp), notice=notice)))
    return rows


def present_history(page, kind, catalog=None):
    catalog = catalog or {}
    output = []
    for event in page.events:
        notice = ""
        if event.owner_conflict:
            notice = "Принадлежность события клиенту неоднозначна."
        elif event.has_conflicts:
            notice = "Данные события различаются между источниками."
        rows, counts = [], Counter()
        for variant in event.variants:
            current = Counter()
            for identity, row in _variant_rows(variant, kind, catalog, notice):
                key = (identity, row)
                current[key] += 1
                if current[key] > counts[key]:
                    rows.append(row)
            # A multiset union of visible product rows across snapshots. Repeated lines
            # within one snapshot survive; identical lines in other versions add no rows.
            counts |= current
        output.append((event.event_key, tuple(rows)))
    return HistoryDisplayPage(page, tuple(output))
