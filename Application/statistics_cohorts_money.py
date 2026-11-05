"""Compact monetary companions for existing first-observed cohort states."""
from dataclasses import dataclass,field
from decimal import Decimal,localcontext
from Application.order_statistics import CURRENCIES


def _add(a,b):
    """Exact, order-independent sum; do not change the caller's Decimal context."""
    if not a:return b if b else Decimal(0)
    if not b:return a
    with localcontext() as context:
        context.prec=max(context.prec,max(a.adjusted(),b.adjusted())-min(a.as_tuple().exponent,b.as_tuple().exponent)+2)
        value=a+b
    return value if value else Decimal(0)


def _mean(amount,count):
    if not count:return None
    with localcontext():
        return str(amount/count)


@dataclass(slots=True)
class _Amounts:
    rub_orders: int=0
    kzt_orders: int=0
    rub_amount: Decimal=Decimal(0)
    kzt_amount: Decimal=Decimal(0)

    def add(self,payload):
        if payload is None:return
        currency,amount=payload
        if currency=="RUB":
            self.rub_orders+=1;self.rub_amount=_add(self.rub_amount,amount)
        elif currency=="KZT":
            self.kzt_orders+=1;self.kzt_amount=_add(self.kzt_amount,amount)
        else:
            raise ValueError("Unsupported valued cohort currency")

    def merge(self,other):
        self.rub_orders+=other.rub_orders;self.kzt_orders+=other.kzt_orders
        self.rub_amount=_add(self.rub_amount,other.rub_amount)
        self.kzt_amount=_add(self.kzt_amount,other.kzt_amount)

    def currency(self,currency):
        return (self.rub_orders,self.rub_amount) if currency=="RUB" else (self.kzt_orders,self.kzt_amount)


@dataclass(slots=True)
class BuyerMoney(_Amounts):
    orders: int=0
    unvalued_orders: int=0
    first: tuple | None=None
    second: tuple | None=None

    def record(self,purchase):
        self.orders+=1
        payload=(purchase.currency,purchase.amount) if purchase.monetary_available else None
        if payload is None:self.unvalued_orders+=1
        else:_Amounts.add(self,payload)
        return payload


@dataclass(slots=True)
class SecondMoney(_Amounts):
    unvalued_orders: int=0

    def record(self,payload):
        if payload is None:self.unvalued_orders+=1
        else:_Amounts.add(self,payload)

    def merge(self,other):
        _Amounts.merge(self,other)
        self.unvalued_orders+=other.unvalued_orders

    def result(self,eligible,repeated):
        currencies={}
        for currency in CURRENCIES:
            n,amount=self.currency(currency)
            currencies[currency]=dict(known_orders=n,amount=str(amount),mean_order=_mean(amount,n))
        return dict(eligible=eligible,second_buyers=repeated,unvalued_orders=self.unvalued_orders,currencies=currencies)


@dataclass(slots=True)
class MonthMoney:
    orders: int=0
    unvalued_orders: int=0
    totals: _Amounts=field(default_factory=_Amounts)
    first: _Amounts=field(default_factory=_Amounts)
    horizons: list=field(default_factory=lambda:[SecondMoney() for _ in range(5)])

    def add(self,buyer):
        self.orders+=buyer.orders;self.unvalued_orders+=buyer.unvalued_orders
        self.totals.merge(buyer);self.first.add(buyer.first)

    def result(self,size):
        currencies={}
        for currency in CURRENCIES:
            n,amount=self.totals.currency(currency)
            first_n,first_amount=self.first.currency(currency)
            repeated=_add(amount,first_amount.copy_negate())
            with localcontext():
                share=float(100*repeated/amount) if amount else None
            currencies[currency]=dict(known_orders=n,amount=str(amount),mean_order=_mean(amount,n),
                first_orders=first_n,first_amount=str(first_amount),repeat_orders=n-first_n,
                repeat_amount=str(repeated),repeat_amount_rate=share)
        return dict(cohort_size=size,orders=self.orders,unvalued_orders=self.unvalued_orders,currencies=currencies)
