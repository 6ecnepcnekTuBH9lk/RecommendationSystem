"""Private compact monetary codes and fixed demographic groups, never customer/order DTOs."""
from array import array
from collections import Counter
from dataclasses import dataclass,field
from decimal import Decimal
from Application.customer_demographics import AGE_LABELS,age_group
from Application.order_statistics import CURRENCIES,histogram_median

MAX_ADDITIONAL_BYTES=256*1024**2


class _Budget:
    """Conservative accounting for added buffers, intern tables and sort scratch."""
    def __init__(self,limit=MAX_ADDITIONAL_BYTES,used=0):
        self.limit=limit;self.used=used;self.peak=used

    def reserve(self,size):
        if self.used+size>self.limit:
            raise MemoryError("Превышен бюджет памяти демографической аналитики покупателей.")
        self.used+=size;self.peak=max(self.peak,self.used)

    def release(self,size):
        self.used-=size


class OrderAmounts:
    """One interned Decimal per distinct currency/amount; four-byte order codes."""
    def __init__(self):
        self.budget=_Budget()
        self.indices={c:{} for c in CURRENCIES}
        self.values={c:[] for c in CURRENCIES}

    def code(self,currency,amount):
        mapping=self.indices[currency]
        index=mapping.get(amount)
        if index is None:
            self.budget.reserve(256)
            index=len(self.values[currency])
            if index>=2**31:
                raise MemoryError("Слишком много различных сумм демографической аналитики.")
            mapping[amount]=index;self.values[currency].append(amount)
        return index*2+CURRENCIES.index(currency)


@dataclass(slots=True)
class _Group:
    buyers: int=0
    orders: int=0
    single_buyers: int=0
    unvalued_orders: int=0
    known_buyers: dict=field(default_factory=lambda:{c:0 for c in CURRENCIES})
    amounts: dict=field(default_factory=lambda:{c:Decimal(0) for c in CURRENCIES})
    codes: dict=field(default_factory=lambda:{c:array("I") for c in CURRENCIES})

    def add(self,state,unvalued,money):
        self.buyers+=1;self.orders+=state.orders
        self.single_buyers+=int(state.orders==1)
        self.unvalued_orders+=unvalued
        for currency,known,amount in money:
            self.known_buyers[currency]+=known
            self.amounts[currency]+=amount


def _median(codes,values,check,budget):
    if not codes:return None
    histogram=len(values)*4<len(codes)
    scratch=192*len(values) if histogram else 64*len(codes)
    budget.reserve(scratch)
    n=0
    def key(code):
        nonlocal n
        n+=1
        if n%1024==0:check()
        return values[code]
    try:
        check()
        if histogram:
            def amounts():
                for code in codes:yield key(code)
            value=histogram_median(Counter(amounts()))
            check()
            return str(value)
        ordered=sorted(codes,key=key)
        check()
        midpoint=len(ordered)//2
        value=values[ordered[midpoint]]
        if len(ordered)%2==0:
            value=(values[ordered[midpoint-1]]+value)/2
        return str(value)
    finally:
        budget.release(scratch)


class DemographicSummary:
    """Finalize both partitions from existing buyer states; no source IO or SQL."""
    def __init__(self,amounts,genders,check):
        self.amounts=amounts;self.check=check
        self.budget=_Budget(amounts.budget.limit,amounts.budget.used)
        self.budget.reserve(16*1024)
        self.groups={"gender":{g:_Group() for g in genders},"age":{g:_Group() for g in AGE_LABELS}}

    def _label(self,dimension,profile):
        gender,age=profile if profile is not None else (next(reversed(self.groups["gender"])),None)
        return gender if dimension=="gender" else age_group(age) if age is not None else AGE_LABELS[-1]

    def add(self,state,profile):
        unvalued=sum(y.unvalued for y in state.years.values())
        money=tuple((currency,int(any(y.counts[currency] for y in state.years.values())),
                     sum((y.amounts.get(currency,Decimal(0)) for y in state.years.values()),Decimal(0)))
                    for currency in CURRENCIES)
        for dimension,groups in self.groups.items():
            groups[self._label(dimension,profile)].add(state,unvalued,money)

    def _partition(self,dimension,groups,buyers):
        size=0;ticks=0
        for buyer_index,buyer in enumerate(buyers):
            if buyer_index%1024==0:self.check()
            if buyer.strict is None:continue
            target=groups[self._label(dimension,buyer.profile)]
            for code in buyer.money_codes or ():
                if ticks%1024==0:self.check()
                ticks+=1
                self.budget.reserve(8);size+=8
                target.codes[CURRENCIES[code&1]].append(code//2)
        return size

    def result(self,global_money,buyers):
        def rate(n,d):return 100*n/d if d else None
        result={}
        for dimension,groups in self.groups.items():
            size=self._partition(dimension,groups,buyers)
            total=sum(g.buyers for g in groups.values());rows=[]
            for label,g in groups.items():
                self.check()
                money={}
                for currency in CURRENCIES:
                    n=len(g.codes[currency]);amount=g.amounts[currency]
                    money[currency]=dict(known_buyers=g.known_buyers[currency],known_orders=n,amount=str(amount),
                        mean_order=str(amount/n) if n else None,
                        median_order=_median(g.codes[currency],self.amounts.values[currency],self.check,self.budget),
                        amount_rate=float(rate(amount,Decimal(global_money[currency]["amount"])))
                                    if Decimal(global_money[currency]["amount"]) else None)
                repeated=g.buyers-g.single_buyers
                rows.append(dict(label=label,buyers=g.buyers,buyer_rate=rate(g.buyers,total),orders=g.orders,
                    mean_orders=g.orders/g.buyers if g.buyers else None,single_buyers=g.single_buyers,
                    repeat_buyers=repeated,repeat_rate=rate(repeated,g.buyers),
                    unvalued_orders=g.unvalued_orders,money=money))
            result[dimension]=rows
            # Reuse the same bounded workspace for the other dimension.
            for g in groups.values():
                for currency in CURRENCIES:g.codes[currency]=array("I")
            self.budget.release(size)
        self.check()
        return result
