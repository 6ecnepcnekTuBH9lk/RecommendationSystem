"""Compact aggregates of already classified action interactions; no I/O or history."""

from collections import Counter, defaultdict
from datetime import timezone
from decimal import Decimal

from Application.interactions import InteractionType
from Application.order_statistics import CURRENCIES, CURRENCY_BY_NAMESPACE, histogram_median


VIEW_BUCKETS = ("1", "2–5", "6–10", "11–25", "26–50", "51–100", "101+")
FAVORITE_BUCKETS = ("1", "2", "3–5", "6–10", "11+")
AVAILABILITY_LABELS = ("Доступен", "Недоступен", "Не указано")


def _distribution(counts, labels, total):
    return tuple((label, counts[label], 100 * counts[label] / total if total else 0.) for label in labels)


class _Activity:
    def __init__(self):
        self.counts = Counter()
        self.users = {kind: set() for kind in (InteractionType.VIEW, InteractionType.FAVORITE)}

    def add(self, interactions):
        for interaction in interactions:
            self.counts[interaction.interaction_type] += 1
            self.users[interaction.interaction_type].add(interaction.customer_id)

    def values(self):
        return tuple(value for kind in (InteractionType.VIEW, InteractionType.FAVORITE)
                     for value in (self.counts[kind], len(self.users[kind])))


class ActionAggregates:
    def __init__(self):
        self.users = {kind: Counter() for kind in (InteractionType.VIEW, InteractionType.FAVORITE)}
        self.channels = defaultdict(_Activity)
        self.channel_names = defaultdict(Counter)
        self.months = defaultdict(_Activity)
        self.availability = Counter()
        self.prices = {currency: Counter() for currency in CURRENCIES}
        self.ambiguous = self.unknown_price_currency = 0

    def add(self, action, interactions):
        if not interactions:
            raise ValueError("Action aggregates require successful item interactions")
        kind = interactions[0].interaction_type
        if kind not in self.users or any(item.interaction_type != kind for item in interactions):
            raise ValueError("Action interactions must have one VIEW/FAVORITE type")
        for item in interactions:
            self.users[kind][item.customer_id] += 1
        key = next((value for value in (action.channel_system_name, action.channel_external_id)
                    if value and value.strip()), None)
        self.channels[key].add(interactions)
        if action.channel_name and action.channel_name.strip():
            self.channel_names[key][action.channel_name] += 1  # One vote per source action.
        month = action.event_datetime_utc.astimezone(timezone.utc)
        self.months[f"{month.year:04d}-{month.month:02d}"].add(interactions)
        if kind != InteractionType.VIEW:
            return
        if len(action.products) > 1:
            self.ambiguous += 1
            return
        if len(action.products) != 1:
            raise ValueError("Successful VIEW requires a product")
        available = action.product_view_is_available
        self.availability["Не указано" if available is None else "Доступен" if available else "Недоступен"] += 1
        price = action.product_view_price
        if price is not None:
            if not price.is_finite() or price < 0:
                raise ValueError("View price must be finite and non-negative")
            currency = CURRENCY_BY_NAMESPACE.get(action.products[0].namespace)
            if currency is None:
                self.unknown_price_currency += 1
            else:
                self.prices[currency][price] += 1

    def result(self):
        result = {}
        for kind, prefix, suffix, labels, limits in (
            (InteractionType.VIEW, "views", "viewer", VIEW_BUCKETS, (1, 5, 10, 25, 50, 100)),
            (InteractionType.FAVORITE, "favorites", "user", FAVORITE_BUCKETS, (1, 2, 5, 10)),
        ):
            users = self.users[kind]
            histogram = Counter(users.values())
            buckets = Counter()
            for count, n in histogram.items():
                label = next((label for limit, label in zip(limits, labels) if count <= limit), labels[-1])
                buckets[label] += n
            result[f"mean_{prefix}_per_{suffix}"] = sum(users.values()) / len(users) if users else 0.
            result[f"median_{prefix}_per_{suffix}"] = float(histogram_median(histogram) or 0)
            result[f"{kind.value.lower()}_user_activity_distribution"] = _distribution(buckets, labels, len(users))
        views, favorites = (sum(self.users[kind].values()) for kind in (InteractionType.VIEW, InteractionType.FAVORITE))
        channels = []
        for key, activity in self.channels.items():
            names = self.channel_names[key]
            name = min(names, key=lambda name: (-names[name], name)) if names else key or "Не указано"
            v, vu, f, fu = activity.values()
            channels.append((key, name, v, vu, 100 * v / views if views else 0., f, fu, 100 * f / favorites if favorites else 0.))
        channels.sort(key=lambda row: (-row[2], -row[5], row[1], row[0] is not None, row[0] or ""))
        prices = []
        for currency, histogram in self.prices.items():
            n = sum(histogram.values())
            total = sum((price * count for price, count in histogram.items()), Decimal(0))
            prices.append((currency, n, str(total / n if n else Decimal(0)), str(histogram_median(histogram) or Decimal(0))))
        parameters = sum(self.availability.values())
        return dict(result, action_channel_statistics=tuple(channels),
                    action_monthly_dynamics=tuple((month, *activity.values()) for month, activity in sorted(self.months.items())),
                    view_parameter_actions=parameters,
                    view_availability_distribution=_distribution(self.availability, AVAILABILITY_LABELS, parameters),
                    view_price_statistics=tuple(prices))
