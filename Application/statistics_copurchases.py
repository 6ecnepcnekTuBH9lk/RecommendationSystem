"""Exact within-order category co-purchases over the existing unique buyer facts."""
from dataclasses import dataclass,field
from itertools import combinations
from sys import getsizeof

from Application.statistics_purchase_facts import PurchaseOrderFact
from Application.statistics_cohorts_categories import (
    UNKNOWN_CATEGORY,category_reference,category_codes,category_key,
)

MAX_PAIRS_PER_ORDER = 4096
MAX_DISTINCT_PAIRS = 100000
MAX_ADDITIONAL_BYTES = 512*1024**2


class CoPurchaseLimitError(ValueError):
    """Static resource diagnostic; never includes individual source values."""


@dataclass(slots=True)
class _Count:
    orders: int = 0
    buyers: set = field(default_factory=set)


class CategoryCoPurchases:
    """Caller owns deduplication and CustomerMerges; no order IDs/history retained."""
    def __init__(self,identity_space,coverage,names,paths,check=lambda:None):
        self.identity_space=identity_space
        self.coverage=tuple(coverage)
        self.reference=category_reference(names,paths)
        self.check=check
        self.orders=self.multi_orders=self.unknown_orders=0
        self._buyers={}
        self._multi_buyers=set()
        self.categories={}
        self.pairs={}
        self._reserved=0
        self._failed=False
        for code,(name,path) in self.reference.items():
            self._reserve(512+getsizeof(code)+getsizeof(name)+getsizeof(path))

    def _reserve(self,size):
        if self._reserved+size>MAX_ADDITIONAL_BYTES:
            raise CoPurchaseLimitError('Превышен бюджет памяти совместных покупок категорий. Расчёт остановлен без усечения данных.')
        self._reserved+=size

    def _membership(self,buyers,sid):
        if sid not in buyers:
            self._reserve(96)  # Conservative amortized hash-set capacity, SID is shared.
            buyers.add(sid)

    def _add(self,fact,codes):
        self.check()
        if not isinstance(fact,PurchaseOrderFact) or fact.identity_space!=self.identity_space:
            raise ValueError('Co-purchase facts belong to a different identity space')
        if not any(a<=fact.order_datetime_utc<b for a,b in self.coverage):
            raise ValueError('Co-purchase fact is outside buyer observation')
        if not fact.strict.has_purchase:
            return
        buyer=fact.customer_id
        if not (type(buyer) is int and buyer>0 or isinstance(buyer,str) and buyer.strip() and buyer==buyer.strip()):
            raise ValueError('Invalid co-purchase buyer identity')
        codes=category_codes(codes,self.reference)
        if len(codes)*(len(codes)-1)//2>MAX_PAIRS_PER_ORDER:
            raise CoPurchaseLimitError('Слишком много пар категорий одного заказа. Расчёт остановлен без усечения данных.')
        if buyer not in self._buyers:
            self._reserve(160+getsizeof(buyer))
            self._buyers[buyer]=len(self._buyers)
        sid=self._buyers[buyer]
        self.orders+=1
        self.unknown_orders+=int(None in codes)
        if len(codes)>1:
            self.multi_orders+=1
            self._membership(self._multi_buyers,sid)
        for code in codes:
            self.check()
            if code not in self.categories:
                self._reserve(1024)
                self.categories[code]=_Count()
            counter=self.categories[code]
            counter.orders+=1
            self._membership(counter.buyers,sid)
        # Stream combinations of unique codes; A+A and reversed duplicates cannot arise.
        for pair in combinations(sorted(codes,key=category_key),2):
            self.check()
            if pair not in self.pairs:
                if len(self.pairs)>=MAX_DISTINCT_PAIRS:
                    raise CoPurchaseLimitError('Слишком много различных совместных пар категорий. Расчёт остановлен без усечения данных.')
                self._reserve(1024)
                self.pairs[pair]=_Count()
            counter=self.pairs[pair]
            counter.orders+=1
            self._membership(counter.buyers,sid)

    def add(self,fact,category_codes=None):
        if self._failed:
            raise ValueError('Co-purchase calculation is incomplete; restart is required')
        try:
            self._add(fact,category_codes)
        except BaseException:
            self._failed=True
            raise

    def result(self):
        if self._failed:
            raise ValueError('Co-purchase calculation is incomplete; restart is required')
        self.check()
        # Account for sorted keys and DTO scratch, without accumulating across repeated reads.
        if self._reserved+512*(len(self.categories)+len(self.pairs))>MAX_ADDITIONAL_BYTES:
            raise CoPurchaseLimitError('Превышен бюджет результата совместных покупок категорий. Расчёт остановлен без усечения данных.')
        categories=[]
        for code,value in sorted(self.categories.items(),key=lambda p:category_key(p[0])):
            self.check()
            name,path=self.reference[code] if code is not None else (UNKNOWN_CATEGORY,None)
            categories.append(dict(code=code,name=name,path=list(path) if path is not None else None,
                orders=value.orders,buyers=len(value.buyers),order_rate=100*value.orders/self.orders))
        pairs=[]
        for (a,b),value in sorted(self.pairs.items(),key=lambda p:(-p[1].orders,category_key(p[0][0]),category_key(p[0][1]))):
            self.check()
            pairs.append(dict(category1=a,category2=b,orders=value.orders,buyers=len(value.buyers),
                order_rate=100*value.orders/self.orders,
                given_first_rate=100*value.orders/self.categories[a].orders,
                given_second_rate=100*value.orders/self.categories[b].orders,incomplete=a is None or b is None))
        return dict(orders=self.orders,buyers=len(self._buyers),multicategory_orders=self.multi_orders,
                    multicategory_rate=100*self.multi_orders/self.orders if self.orders else None,
                    multicategory_buyers=len(self._multi_buyers),unknown_orders=self.unknown_orders,
                    categories=categories,pairs=pairs)
