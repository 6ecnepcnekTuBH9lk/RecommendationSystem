"""Internal single-pass aggregates for accepted purchase-order snapshots."""

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

from Application.mindbox.adapters._common import AdapterError, get, identifier, objects, text


CURRENCY_BY_NAMESPACE = {"offline1C": "RUB", "kanzlerKz": "KZT"}
CURRENCIES = ("RUB", "KZT")
BASKET_LABELS = ("1 позиция", "2 позиции", "3–5 позиций", "6–10 позиций", "11+ позиций")
CURRENCY_WARNING = ("Часть заказов с покупкой не удалось однозначно отнести к валюте; "
                    "их денежные показатели не включены в RUB/KZT.")


def _decimal(value, field, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise AdapterError(f"{field}: ожидается неотрицательное число")
    try:
        result = Decimal(str(value))
        if not result.is_finite() or result < 0:
            raise InvalidOperation
        return result
    except InvalidOperation:
        raise AdapterError(f"{field}: некорректное число") from None


def histogram_median(counts):
    total = sum(counts.values())
    if not total:
        return None
    positions = ((total - 1) // 2, total // 2)
    seen, values = 0, []
    for value, count in sorted(counts.items()):
        values.extend(value for position in positions if seen <= position < seen + count)
        seen += count
    return sum(values) / 2


def _currency(namespaces):
    currencies = {CURRENCY_BY_NAMESPACE.get(namespace) for namespace in namespaces}
    if len(currencies - {None}) > 1:
        return "mixed"
    if not currencies or None in currencies:
        return "unknown"
    return next(iter(currencies))


def _distribution(counts, total):
    return tuple((name, count, 100 * count / total if total else 0.0) for name, count in sorted(counts.items()))


def _category(raw, path):
    value = text(raw, path)
    return value.strip() if value and value.strip() else "Не указано"


def order_for_adapter(raw):
    """Only supply missing channel metadata required by the shared adapter.

    Dedup and store grouping always use the original raw snapshot. No storage,
    identity, product, status or interaction fields are changed.
    """
    key = identifier(raw, "firstAction.channel.ids.externalId", required=False)
    name = text(raw, "firstAction.channel.name")
    if key is not None and name is not None and name.strip():
        return raw
    channel = dict(get(raw, "firstAction.channel") or {})
    channel["ids"] = {**(channel.get("ids") or {}), "externalId": key if key is not None else "statistics-missing-channel"}
    channel["name"] = name if name and name.strip() else "Не указано"
    return {**raw, "firstAction": {**raw["firstAction"], "channel": channel}}


class _Totals:
    def __init__(self):
        self.orders = self.lines = 0
        self.quantity = self.amount = Decimal(0)
        self.buyers = set()

    def add(self, lines, quantity, amount, buyer=None):
        self.orders += 1
        self.lines += lines
        self.quantity += quantity
        self.amount += amount
        if buyer is not None:
            self.buyers.add(buyer)

    def values(self):
        return self.orders, self.lines, str(self.quantity), str(self.amount), str(self.amount / self.orders if self.orders else Decimal(0))


class OrderAggregates:
    """Retain counters/histograms and store buyer sets, never raw history."""

    def __init__(self):
        self.orders = 0
        self.lines = Counter()
        self.units = Counter()
        self.currency_counts = Counter()
        self.financials = {currency: _Totals() for currency in CURRENCIES}
        self.amounts = {currency: Counter() for currency in CURRENCIES}
        self.delivery = {currency: Counter() for currency in CURRENCIES}
        self.months = defaultdict(lambda: {currency: _Totals() for currency in CURRENCIES})
        self.stores = {}
        self.store_names = defaultdict(Counter)
        self.ordering = Counter()
        self.delivery_types = Counter()
        self.payments = Counter()

    def add(self, raw, purchases):
        if not purchases:
            return
        count = len(purchases)
        quantity = sum((_decimal(line.quantity, "quantity") for line in purchases), Decimal(0))
        currency = _currency(line.product.namespace for line in purchases)
        amount = Decimal(0)
        for line in purchases:
            price = _decimal(line.price_of_line, "priceOfLine")
            if currency in CURRENCIES:
                amount += price
        delivery = _decimal(get(raw, "deliveryCost"), "deliveryCost", optional=True)
        key = identifier(raw, "firstAction.channel.ids.externalId", required=False)
        name = _category(raw, "firstAction.channel.name") if key is not None else "Не указано"
        self.store_names[key][name] += 1
        self.orders += 1
        self.lines[count] += 1
        self.units[quantity] += 1
        self.currency_counts[currency] += 1
        self.ordering[_category(raw, "customFields.orderingMethod")] += 1
        self.delivery_types[_category(raw, "customFields.deliveryType")] += 1
        payment_types = {_category(payment, "type") for payment in objects(raw, "payments")
                         if text(payment, "type") and text(payment, "type").strip()}
        self.payments.update(payment_types or {"Не указано"})
        if currency not in CURRENCIES:
            return
        self.financials[currency].add(count, quantity, amount)
        self.amounts[currency][amount] += 1
        if delivery is not None:
            self.delivery[currency][delivery] += 1
        month = purchases[0].order_datetime_utc.strftime("%Y-%m")
        self.months[month][currency].add(count, quantity, amount)
        totals = self.stores.setdefault((currency, key), _Totals())
        totals.add(count, quantity, amount, purchases[0].customer_id)

    def result(self):
        financials, delivery_rows, stores = [], [], []
        for currency in CURRENCIES:
            financials.append((currency, *self.financials[currency].values(),
                               str(histogram_median(self.amounts[currency]) or Decimal(0))))
            histogram = self.delivery[currency]
            n = sum(histogram.values())
            total = sum((value * count for value, count in histogram.items()), Decimal(0))
            delivery_rows.append((currency, n, str(total), str(total / n if n else Decimal(0)),
                                  str(histogram_median(histogram) or Decimal(0))))
            rows = []
            for (unit, key), totals in self.stores.items():
                if unit != currency:
                    continue
                name = min(self.store_names[key], key=lambda name: (-self.store_names[key][name], name))
                n, lines, quantity, amount, mean = totals.values()
                rows.append((currency, key, name, n, len(totals.buyers), lines, quantity, amount, mean))
            rows.sort(key=lambda row: (-Decimal(row[7]), -row[3], row[2], row[1] is not None, row[1] or ""))
            stores.extend(rows)
        basket = Counter({label: 0 for label in BASKET_LABELS})
        for count, orders in self.lines.items():
            index = 0 if count == 1 else 1 if count == 2 else 2 if count <= 5 else 3 if count <= 10 else 4
            basket[BASKET_LABELS[index]] += orders
        return dict(
            purchase_orders=self.orders,
            mean_purchase_lines_per_order=sum(n * count for n, count in self.lines.items()) / self.orders if self.orders else 0.,
            median_purchase_lines_per_order=float(histogram_median(self.lines) or 0),
            mean_purchase_units_per_order=str(sum((n * count for n, count in self.units.items()), Decimal(0)) / self.orders
                                              if self.orders else Decimal(0)),
            median_purchase_units_per_order=str(histogram_median(self.units) or Decimal(0)),
            purchase_basket_distribution=tuple((label, basket[label], 100 * basket[label] / self.orders if self.orders else 0.)
                                               for label in BASKET_LABELS),
            order_financials=tuple(financials), delivery_financials=tuple(delivery_rows), store_statistics=tuple(stores),
            order_monthly_dynamics=tuple((month, values["RUB"].orders, str(values["RUB"].amount),
                                         values["KZT"].orders, str(values["KZT"].amount)) for month, values in sorted(self.months.items())),
            ordering_method_distribution=_distribution(self.ordering, self.orders),
            delivery_type_distribution=_distribution(self.delivery_types, self.orders),
            payment_type_distribution=_distribution(self.payments, self.orders),
            mixed_currency_purchase_orders=self.currency_counts["mixed"],
            unknown_currency_purchase_orders=self.currency_counts["unknown"],
        )
