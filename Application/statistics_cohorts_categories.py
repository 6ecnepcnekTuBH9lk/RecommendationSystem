"""Direct category sets of two observed orders; aggregate counters, no source I/O."""
from collections import Counter
from collections.abc import Mapping

from Application.statistics_purchase_policy import STRICT_COMPLETED, accepts_purchase_line

UNKNOWN_CATEGORY = 'Категория не определена'
MAX_ORDER_CATEGORIES = 256
MAX_PAIRS_PER_BUYER = 4096
MAX_AGGREGATE_PAIRS = 100000


class CategoryAnalyticsLimitError(ValueError):
    """Safe resource diagnostic; no order, buyer, product or category values."""


def category_key(code):
    return code is None, code or ''


def category_reference(names, paths):
    if not isinstance(names, Mapping) or paths is not None and not isinstance(paths, Mapping):
        raise ValueError('Invalid cohort category reference')
    result = {}
    for code, name in names.items():
        if (not isinstance(code, str) or not code.strip() or code != code.strip()
                or not isinstance(name, str) or not name.strip() or name != name.strip()):
            raise ValueError('Invalid cohort category reference')
        path = (paths or {}).get(code)
        if path is not None:
            if (not isinstance(path, (list,tuple)) or not path
                    or any(not isinstance(v,str) or not v.strip() or v != v.strip() for v in path)
                    or path[-1] != name):
                raise ValueError('Invalid cohort category path')
            path = tuple(path)
        result[code] = name, path
    return result


def category_codes(values, reference):
    if not isinstance(values, (set,frozenset,list,tuple)):
        raise ValueError('Cohort purchase category projection is unavailable')
    result = set()
    for code in values:
        if code is not None and (not isinstance(code,str) or code not in reference):
            raise ValueError('Invalid cohort purchase category code')
        result.add(code)
        if len(result) > MAX_ORDER_CATEGORIES:
            raise CategoryAnalyticsLimitError('Слишком много различных категорий одного заказа. Расчёт остановлен без усечения данных.')
    if not result:
        raise ValueError('Cohort purchase category projection is empty')
    return frozenset(result)


def order_category_codes(lines, products, metadata, names, check):
    """Read prepared STRICT lines, using the existing resolver without its diagnostics."""
    codes = set()
    for line in lines:
        check()
        if not accepts_purchase_line(line.line_status,line.quantity,policy=STRICT_COMPLETED):
            continue
        item = products.resolve(line.product,strict=False).item_id
        product = metadata.get(item)
        code = ((product.category or '').strip() or None) if product is not None else None
        codes.add(code if code in names else None)
        if len(codes) > MAX_ORDER_CATEGORIES:
            raise CategoryAnalyticsLimitError('Слишком много различных категорий одного заказа. Расчёт остановлен без усечения данных.')
    return frozenset(codes)


class CategoryCounts:
    def __init__(self):
        self.buyers = self.repeated = 0
        self.first = Counter()
        self.first_repeated = Counter()
        self.second = Counter()
        self.pairs = Counter()

    def add(self, first, second, check):
        if second is not None and len(first)*len(second) > MAX_PAIRS_PER_BUYER:
            raise CategoryAnalyticsLimitError('Слишком много переходов категорий у одного покупателя. Расчёт остановлен без усечения данных.')
        self.buyers += 1
        self.first.update(first)
        if second is None:
            return
        self.repeated += 1
        self.first_repeated.update(first)
        self.second.update(second)
        # Stream the Cartesian product; never allocate a per-buyer pair list.
        for a in first:
            for b in second:
                check()
                pair = a,b
                if pair not in self.pairs and len(self.pairs) >= MAX_AGGREGATE_PAIRS:
                    raise CategoryAnalyticsLimitError('Слишком много различных переходов категорий. Расчёт остановлен без усечения данных.')
                self.pairs[pair] += 1

    def result(self, reference, check):
        rows = []
        for code in sorted(set(self.first)|set(self.second)|{None},key=category_key):
            check()
            first, repeated, second = self.first[code], self.first_repeated[code], self.second[code]
            name, path = reference[code] if code is not None else (UNKNOWN_CATEGORY,None)
            rows.append(dict(code=code,name=name,path=list(path) if path is not None else None,
                first_buyers=first,repeat_buyers=repeated,observed_repeat_rate=100*repeated/first if first else None,
                second_buyers=second,second_rate=100*second/self.repeated if self.repeated else None))
        pairs = []
        for (a,b), n in sorted(self.pairs.items(),key=lambda p:(-p[1],category_key(p[0][0]),category_key(p[0][1]))):
            check()
            pairs.append(dict(first_code=a,second_code=b,buyers=n,repeat_rate=100*n/self.repeated))
        return dict(buyers=self.buyers,repeat_buyers=self.repeated,unknown_first_buyers=self.first[None],
                    unknown_second_buyers=self.second[None],categories=rows,transitions=pairs)
