"""Additional report breakdowns over the same accepted Unified evidence."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Context, Decimal, Inexact, InvalidOperation, MAX_EMAX, MIN_EMIN, Overflow, localcontext
import json

from Application.order_statistics import BASKET_LABELS, CURRENCIES, histogram_median
from Application.statistics_unified_facts import _issues, _result, _sum
from Application.statistics_unified_policy import EvidenceQuality as Quality


_ZERO = Decimal(0)
_NO_BASE = (Quality.UNKNOWN, "NO_FULLY_VALUED_BASE")
_EXACT_ADDITION = Context(prec=48, Emin=MIN_EMIN, Emax=MAX_EMAX, traps=[Inexact, InvalidOperation, Overflow])


def add_exact_decimal(left, right):
    """Use the native Decimal operations when the addition fits exactly."""
    try:
        value = _EXACT_ADDITION.add(left, right)
    except Inexact:
        return _sum([left, right])
    return _EXACT_ADDITION.normalize(value) if value else _ZERO


def _amount(values):
    return _sum(list(values))


def _mean(amount, count):
    if count == 0:
        return None
    with localcontext() as context:
        context.prec = max(28, len(amount.as_tuple().digits))
        return amount / count


def _label(value):
    return value.strip() if isinstance(value, str) and value.strip() else "Не указано"


@dataclass(slots=True)
class _Valuation:
    orders: int = 0
    lines: int = 0
    units: Decimal = _ZERO
    valued: int = 0
    valued_amount: Decimal = _ZERO
    known_subtotal: Decimal | None = None
    issues: set = field(default_factory=set)

    def add(self, fact, money):
        self.orders += 1
        self.lines += fact.purchase.known_positions
        self.units = add_exact_decimal(self.units, fact.purchase.known_nominal_units)
        self.issues.update(_issues(money.recognized_purchase))
        if money.valued_purchase_positions:
            self.known_subtotal = add_exact_decimal(self.known_subtotal or _ZERO, money.known_purchase_subtotal.value)
            self.valued += 1
            self.valued_amount = add_exact_decimal(self.valued_amount, money.recognized_purchase.value)

    def result(self, issues):
        issues = self.issues | issues
        return dict(orders=self.orders, positions=self.lines, nominal_units=self.units, valued_orders=self.valued,
                    unvalued_orders=self.orders-self.valued,
                    recognized=_result(self.known_subtotal, ()) if self.known_subtotal is not None
                    else _result(None, issues or {_NO_BASE}), known_subtotal=self.known_subtotal,
                    mean_valued_order=_result(_mean(self.valued_amount, self.valued), () if self.valued else (_NO_BASE,)))


class OrderBreakdowns:
    """Former basket, payment, delivery, month and store scopes on reconciled purchases."""

    def __init__(self):
        self.orders = 0
        self.lines, self.units = Counter(), Counter()
        self.methods, self.payment_types, self.delivery_types = Counter(), Counter(), Counter()
        self.attribute_issues = defaultdict(set)
        self.basket_issues = set()
        self.months, self.stores = defaultdict(_Valuation), defaultdict(_Valuation)
        self.currencies = {currency: _Valuation() for currency in CURRENCIES}
        self.store_names = defaultdict(Counter)
        self.delivery = {currency: Counter() for currency in CURRENCIES}
        self.delivery_issues = defaultdict(set)
        self.checks = {currency: Counter() for currency in CURRENCIES}

    def add(self, fact, add_member, *, whole_currency):
        if fact.purchase.observed.value is not True:
            return
        self.orders += 1
        purchase = fact.purchase
        self.lines[purchase.known_positions] += 1
        self.units[purchase.known_nominal_units] += 1
        self.basket_issues.update(_issues(purchase.positions))
        self.basket_issues.update(_issues(purchase.nominal_units))
        attributes = dict(fact.order_attributes)
        for attribute, counter in (("ordering_method", self.methods), ("delivery_type", self.delivery_types),
                               ("payment_types", self.payment_types)):
            metric = attributes[attribute]
            self.attribute_issues[attribute].update(_issues(metric))
            if metric.value is None:
                counter["Не указано"] += 1
            elif attribute == "payment_types":
                counter.update({_label(value) for value in metric.value} or {"Не указано"})
            else:
                counter[_label(metric.value)] += 1
        month = fact.order_created_at.value.strftime("%Y-%m")
        channel, name = fact.channel.value or (None, None)
        delivery = attributes["delivery_cost"]
        for money in fact.money:
            currency = money.currency
            if fact.order_currency.value != currency and not money.valued_purchase_positions:
                continue
            self.months[month, currency].add(fact, money)
            self.currencies[currency].add(fact, money)
            self.store_names[channel][_label(name)] += 1
            key = json.dumps((currency, channel), ensure_ascii=False)
            add_member("order_store", fact.customer_key.value, key)
            self.stores[currency, channel].add(fact, money)
            self.delivery_issues[currency].update(_issues(delivery))
            if delivery.value is not None:
                if whole_currency == currency:
                    self.delivery[currency][delivery.value] += 1
                else:
                    self.delivery_issues[currency].add((Quality.UNKNOWN, "DELIVERY_CURRENCY_UNVERIFIED"))
            if money.valued_purchase_positions:
                self.checks[currency][money.recognized_purchase.value] += 1

    def result(self, issues, members):
        denominator = sum(self.lines.values())
        baskets = Counter()
        for count, n in self.lines.items():
            index = 0 if count == 1 else 1 if count == 2 else 2 if count <= 5 else 3 if count <= 10 else 4
            baskets[BASKET_LABELS[index]] += n

        def histogram_stats(histogram):
            n = sum(histogram.values())
            # Exact weighted Decimal sum, independent of the ambient context.
            with localcontext() as context:
                context.prec = max((len(Decimal(value).as_tuple().digits) + len(str(count)) + 1
                                    for value, count in histogram.items()), default=28)
                subtotal = _amount(Decimal(value)*count for value, count in histogram.items())
            with localcontext() as context:
                context.prec = max(28, len(subtotal.as_tuple().digits) + 1)
                median = histogram_median(histogram)
            return dict(evaluated=n, subtotal=subtotal if n else None, mean=_mean(subtotal, n), median=median)

        def distribution(counter):
            return [(label, n, 100*n/self.orders if self.orders else None) for label, n in sorted(counter.items())]

        return dict(
            observed_orders=self.orders, basket=histogram_stats(self.lines), nominal_units=histogram_stats(self.units),
            basket_distribution=[(label, baskets[label], 100*baskets[label]/denominator if denominator else None)
                                 for label in BASKET_LABELS],
            basket_reasons=sorted({reason for _, reason in self.basket_issues | issues}),
            ordering_methods=distribution(self.methods), delivery_types=distribution(self.delivery_types),
            payment_types=distribution(self.payment_types),
            attribute_reasons={key: sorted({reason for _, reason in value | issues})
                               for key, value in sorted(self.attribute_issues.items())},
            median_check={currency: histogram_stats(self.checks[currency]) for currency in CURRENCIES},
            currencies={currency: value.result(issues) for currency, value in self.currencies.items()},
            delivery={currency: dict(**histogram_stats(self.delivery[currency]),
                       reasons=sorted({reason for _, reason in self.delivery_issues[currency] | issues}))
                      for currency in CURRENCIES},
            months=[dict(month=month, currency=currency, **values.result(issues))
                    for (month, currency), values in sorted(self.months.items())],
            stores=[dict(currency=currency, channel=channel,
                         name=min(self.store_names[channel], key=lambda s: (-self.store_names[channel][s], s)),
                         buyers=members.get(json.dumps((currency, channel), ensure_ascii=False), 0),
                         **values.result(issues)) for (currency, channel), values in
                    sorted(self.stores.items(), key=lambda row: (row[0][0], row[0][1] or ""))],
        )


def product_breakdowns(rows, metadata, views, favorites):
    groups = {key: defaultdict(lambda: [0, 0, 0, 0, _ZERO]) for key in ("category", "gender", "collection", "style_group")}
    viewers, saved = {}, {}
    for row in rows:
        code = row.item_id
        item = metadata[code]
        viewers[code], saved[code] = views.get(code, 0), favorites.get(code, 0)
        for name, bins in groups.items():
            key = _label(getattr(item, name))
            bucket = bins[key]
            for index, value in enumerate((1, row.views, row.favorites, row.purchase_positions.known)):
                bucket[index] += value
            bucket[4] = add_exact_decimal(bucket[4], row.nominal_units.known or _ZERO)
    return dict(viewers=viewers, favorite_users=saved,
                groups={field: [(label, *counts) for label, counts in sorted(bins.items())]
                        for field, bins in groups.items()})


class BuyerBreakdowns:
    """Disk-backed single-policy inputs to the existing buyer histogram formulas."""

    def __init__(self, db, check):
        self.db, self.check = db, check
        db.executescript("""
            CREATE TABLE buyer_breakdown(customer TEXT, stamp TEXT, rank INTEGER, lines INTEGER, units TEXT,
                rub TEXT, kzt TEXT, PRIMARY KEY(customer,stamp,rank)) WITHOUT ROWID;
            CREATE TABLE buyer_breakdown_profile(customer TEXT PRIMARY KEY, gender TEXT, age INTEGER) WITHOUT ROWID;
            CREATE TABLE buyer_breakdown_seen(customer TEXT PRIMARY KEY) WITHOUT ROWID;
        """)
        self.width = [1, 0]
        self.orders = 0

    def add(self, fact):
        amounts = {row.currency: format(row.recognized_purchase.value, "f")
                   for row in fact.money if row.valued_purchase_positions}
        units = format(fact.purchase.known_nominal_units, "f")
        for value in (units, *amounts.values()):
            integral, _, fraction = value.partition(".")
            self.width[0] = max(self.width[0], len(integral))
            self.width[1] = max(self.width[1], len(fraction))
        self.orders += 1
        self.db.execute("INSERT INTO buyer_breakdown VALUES(?,?,?,?,?,?,?)",
                        (fact.customer_key.value, fact.order_created_at.value.isoformat(timespec="microseconds"),
                         fact.order_sort_rank, fact.purchase.known_positions, units, amounts.get("RUB"), amounts.get("KZT")))

    def demographics(self, profiles, *, available, restricted, as_of):
        from Application.customer_demographics import AGE_LABELS, age_group, calendar_age
        from Application.statistics_buyer_aggregates import GENDERS, distribution

        sexes, ages, age_groups = Counter(), Counter(), Counter()
        count = 0
        for owner, birthday, sex in profiles:
            self.check()
            gender = GENDERS[0] if sex == "male" else GENDERS[1] if sex == "female" else GENDERS[2]
            age = calendar_age(birthday, as_of)
            self.db.execute("INSERT OR IGNORE INTO buyer_breakdown_profile VALUES(?,?,?)", (owner, gender, age))
            if restricted and (not self.db.execute("SELECT 1 FROM actions WHERE customer=?", (owner,)).fetchone()
                               or not self.db.execute("INSERT OR IGNORE INTO buyer_breakdown_seen VALUES(?)", (owner,)).rowcount):
                continue
            count += 1
            sexes[gender] += 1
            age_groups[age_group(age) if age is not None else AGE_LABELS[-1]] += 1
            if age is not None:
                ages[age] += 1
        self.db.commit()
        return dict(profiles=count if available else None, gender=distribution(sexes, GENDERS, count),
                    age=distribution(age_groups, AGE_LABELS, count),
                    mean_age=sum(age*n for age, n in ages.items()) / sum(ages.values()) if ages else None,
                    median_age=histogram_median(ages))

    def _buyers(self, amounts=None):
        from array import array
        from contextlib import closing
        from datetime import date
        from itertools import groupby
        from types import SimpleNamespace
        from Application.statistics_buyer_aggregates import _Purchases

        query = """SELECT b.customer,b.stamp,b.lines,b.units,b.rub,b.kzt,p.gender,p.age
                   FROM buyer_breakdown b LEFT JOIN buyer_breakdown_profile p USING(customer)
                   ORDER BY b.customer,b.stamp,b.rank"""
        with closing(self.db.execute(query)) as rows:
            for _, group in groupby(rows, key=lambda row: row[0]):
                state, days, codes = _Purchases(), [], array("I")
                profile = None
                for _, stamp, lines, units, rub, kzt, gender, age in group:
                    self.check()
                    day = date.fromisoformat(stamp[:10])
                    days.append(day.toordinal())
                    profile = (gender, age) if gender is not None else None
                    prices = [(unit, Decimal(amount)) for unit, amount in (("RUB", rub), ("KZT", kzt)) if amount is not None]
                    unit, price = prices[0] if prices else (None, None)
                    state.add(day.year, SimpleNamespace(line_count=lines, units=Decimal(units), currency=unit,
                                                       amount=price, monetary_available=bool(prices)))
                    annual = state.years[day.year]
                    for unit, price in prices[1:]:
                        annual.amounts[unit] = add_exact_decimal(annual.amounts.get(unit, _ZERO), price)
                        annual.counts[unit] += 1
                    if amounts is not None:
                        amounts.budget.reserve(8 * len(prices))
                        codes.extend(amounts.code(unit, price) for unit, price in prices)
                yield state, days, profile, codes
                if amounts is not None:
                    amounts.budget.release(8*len(codes))

    def result(self, coverage, profiles_available, *, period=None):
        from datetime import timedelta
        from Application.statistics_buyer_aggregates import GENDERS, _Summary, quintiles
        from Application.statistics_buyer_demography import DemographicSummary, OrderAmounts
        from Application.statistics_cohorts_core import _month, _month_key
        from Application.statistics_unified_policy import UNIFIED_PURCHASE_V2
        from types import SimpleNamespace

        histograms = defaultdict(Counter)
        monthly = {currency: {} for currency in CURRENCIES}
        for currency, column in (("RUB", "rub"), ("KZT", "kzt")):
            for month, buyers, orders in self.db.execute(
                    f"SELECT substr(stamp,1,7),COUNT(DISTINCT customer),COUNT(*) "
                    f"FROM buyer_breakdown WHERE {column} IS NOT NULL GROUP BY substr(stamp,1,7)"):
                monthly[currency][month] = [buyers, orders, _ZERO]
        for month, rub, kzt in self.db.execute("SELECT substr(stamp,1,7),rub,kzt FROM buyer_breakdown"):
            self.check()
            for currency, amount in (("RUB", rub), ("KZT", kzt)):
                if amount is not None:
                    price = Decimal(amount)
                    histograms[int(month[:4]), currency][price] += 1
                    row = monthly[currency][month]
                    row[2] = add_exact_decimal(row[2], price)
        months = sorted({_month_key(n) for start, end in ((period,) if period is not None else coverage)
                         for n in range(_month(start), _month(end - timedelta(microseconds=1)) + 1)})
        with localcontext() as context:
            context.prec = max(28, sum(self.width) + len(str(self.orders or 1)) + 4)
            summary = _Summary()
            repeats = {currency: Counter() for currency in CURRENCIES}
            for state, days, profile, _ in self._buyers():
                summary.add(state, days, profile, self.check)
                if state.orders >= 2:
                    for currency in CURRENCIES:
                        values = [year.amounts[currency] for year in state.years.values() if currency in year.amounts]
                        if values:
                            repeats[currency][_amount(values)] += 1
            output = summary.result(UNIFIED_PURCHASE_V2, histograms, Counter(), profiles_available, coverage)
            for currency in CURRENCIES:
                output["money"][currency]["quintiles"] = quintiles(summary.money[currency].known, repeats[currency])
            output["monthly_money"] = {currency: [(month, *rows.get(month, (0, 0, None))) for month in months]
                                       for currency, rows in monthly.items()}
            if profiles_available:
                amounts = OrderAmounts()
                # Intern only distinct fully valued order amounts. Partition
                # readers use the sealed dictionary without retaining orders.
                for (_, currency), histogram in histograms.items():
                    for amount in histogram:
                        amounts.code(currency, amount)
                demographics = DemographicSummary(amounts, GENDERS, self.check)
                for state, _, profile, _ in self._buyers():
                    demographics.add(state, profile)

                class Buyers:
                    def __iter__(other):
                        for state, _, profile, codes in self._buyers(amounts):
                            yield SimpleNamespace(strict=state, profile=profile, money_codes=codes)

                output["demographic_analytics"] = demographics.result(output["money"], Buyers())
        return output
