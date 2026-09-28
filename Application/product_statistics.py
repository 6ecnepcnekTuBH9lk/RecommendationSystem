"""Resolved catalog-item aggregates; no resolution, file access or event history."""

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from Application.interactions import InteractionType
from Application.product_resolution import CatalogError


@dataclass(frozen=True)
class ProductMetadata:
    name: str = ""
    category: str | None = None
    gender: str | None = None
    season: str | None = None
    style_group: str | None = None


@dataclass
class _Item:
    counts: Counter = field(default_factory=Counter)
    users: dict = field(default_factory=lambda: {kind: set() for kind in InteractionType})
    quantity: Decimal = Decimal(0)


class ProductAggregates:
    def __init__(self, metadata, category_names):
        self.metadata = metadata
        self.category_names = category_names
        self.items = {}

    def add(self, resolved):
        if resolved.item_id not in self.metadata:
            raise CatalogError("Для распознанного товара отсутствуют метаданные справочника.")
        interaction = resolved.interaction
        kind = interaction.interaction_type
        item = self.items.get(resolved.item_id)
        if item is None:
            item = self.items[resolved.item_id] = _Item()
        item.counts[kind] += 1
        item.users[kind].add(interaction.customer_id)
        if kind == InteractionType.PURCHASE:
            # Eligibility and quantity validation belong to the existing Orders pipeline.
            item.quantity += interaction.quantity

    def result(self):
        view, favorite, purchase = InteractionType.VIEW, InteractionType.FAVORITE, InteractionType.PURCHASE
        viewed, favorited, purchased = [], [], []
        totals = Counter()
        quantity = Decimal(0)
        groups = {name: {} for name in ("category", "gender", "season", "style_group")}
        for code, item in self.items.items():
            metadata = self.metadata[code]
            v, f, p = (item.counts[kind] for kind in (view, favorite, purchase))
            totals.update(item.counts)
            quantity += item.quantity
            if v:
                viewed.append((code, metadata.name, v, len(item.users[view]), f, p))
            if f:
                favorited.append((code, metadata.name, f, len(item.users[favorite]), v, p))
            if p:
                purchased.append((code, metadata.name, p, len(item.users[purchase]), str(item.quantity), v, f))
            for attribute, values in groups.items():
                label = getattr(metadata, attribute)
                label = (label or "").strip() or None if attribute == "category" else label or "Не указано"
                row = values.setdefault(label, [0, 0, 0, 0, Decimal(0)])
                for index, value in enumerate((1, v, f, p, item.quantity)):
                    row[index] += value
        viewed.sort(key=lambda r: (-r[2], -r[3], -r[5], r[0]))
        favorited.sort(key=lambda r: (-r[2], -r[3], -r[5], r[0]))
        purchased.sort(key=lambda r: (-r[2], -Decimal(r[4]), -r[3], r[0]))
        result = dict(products_with_views=len(viewed), products_with_favorites=len(favorited),
                      products_with_purchases=len(purchased), resolved_view_interactions=totals[view],
                      resolved_favorite_interactions=totals[favorite], resolved_purchase_interactions=totals[purchase],
                      resolved_purchase_quantity=str(quantity), top_viewed_products=tuple(viewed[:20]),
                      top_favorited_products=tuple(favorited[:20]), top_purchased_products=tuple(purchased[:20]))
        for attribute, values in groups.items():
            rows = [(label, *values[:4], str(values[4])) for label, values in values.items()]
            if attribute == "category":
                rows = [(code, self.category_names.get(code, code) if code is not None else "Не указано", *rest)
                        for code, *rest in rows]
                rows.sort(key=lambda r: (-r[5], -r[3], -r[4], r[1], r[0] is not None, r[0] or ""))
            else:
                rows.sort(key=lambda r: (-r[4], -r[2], -r[3], r[0]))
            name = "style" if attribute == "style_group" else attribute
            result[f"product_{name}_statistics"] = tuple(rows)
        return result
