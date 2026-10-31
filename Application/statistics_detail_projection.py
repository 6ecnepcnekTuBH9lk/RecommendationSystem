"""Pure preparation of real detail rows from adapted canonical records/facts."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import fields
from datetime import date
from decimal import Decimal, localcontext
import json

from Application.customer_demographics import calendar_age
from Application.mindbox.canonical_customers import encode
from Application.mindbox.records import CustomerRecord, OrderLineRecord, ProductKey
from Application.order_statistics import CURRENCY_BY_NAMESPACE
from Application.statistics_purchase_facts import PurchaseOrderFact
from Application.statistics_purchase_policy import (
    STRICT_COMPLETED, EXTENDED_COMPLETED, StatusGroup, classify_status, eligible_purchase_policies,
)
from Application.statistics_rfm_core import rfm_windows, SEGMENTS
from Application.statistics_detail_contract import (
    BuyerPolicyMetrics, ClientRow, CoverageStatus, DetailCells, DetailMetadata, DetailContractError,
    Identifier, Money, OrderBundle, OrderHeader, OrderLineRow, OrderMode, OrderPolicyRow, OrderRow,
    Permission, PersonalRfm, ProfileSelection, ProfileStatus, RfmAvailability,
    CLIENT_BASE_COLUMNS, METRIC_COLUMNS, ORDER_BASE_COLUMNS, ORDER_LINE_COLUMNS, ORDER_POLICY_COLUMNS,
    RFM_COLUMNS, RFM_WINDOWS, CURRENCIES, csv_cell, require, utc, valid_id,
)


POLICIES = (STRICT_COMPLETED, EXTENDED_COMPLETED)


def _check(check):
    require(check is None or callable(check))
    if check is not None:
        try:
            check()
        except InterruptedError:
            raise InterruptedError('Detail preparation cancelled') from None


def _fact(fact, metadata, customer_id=None):
    require(isinstance(metadata,DetailMetadata) and isinstance(fact,PurchaseOrderFact))
    require(fact.identity_space == metadata.identity_space and utc(fact.order_datetime_utc))
    require(valid_id(fact.customer_id) and valid_id(fact.order_id))
    require(customer_id is None or fact.customer_id == customer_id)


def _decimal(value, *, optional=False):
    require(value is None and optional or isinstance(value,Decimal) and value.is_finite())
    return value


def _clean(value):
    require(value is None or isinstance(value,str))
    return value.strip() or None if value is not None else None


def _payload(record):
    return tuple(getattr(record,f.name) for f in fields(CustomerRecord)
                 if f.name not in ('customer_id','source_customer_id','change_datetime_utc'))


def _preferred(a,b,canonical):
    return min((a,b),key=lambda p:(p.source_customer_id != canonical,p.source_customer_id))


def select_profile(customer_id, candidates, *, source_available=True, check=None):
    """One canonical key, bounded state; never merge by contact or break ambiguous ties."""
    require(valid_id(customer_id) and type(source_available) is bool)
    best = unknown = None
    known_conflict = unknown_conflict = False
    count = 0
    for record in candidates:
        _check(check)
        require(source_available and isinstance(record,CustomerRecord))
        require(record.customer_id == customer_id and valid_id(record.source_customer_id))
        require(record.change_datetime_utc is None or utc(record.change_datetime_utc))
        count += 1
        if record.change_datetime_utc is None:
            if unknown is None:
                unknown = record
            elif _payload(unknown) != _payload(record):
                unknown_conflict = True
            else:
                unknown = _preferred(unknown,record,customer_id)
        elif best is None or record.change_datetime_utc > best.change_datetime_utc:
            best, known_conflict = record, False
        elif record.change_datetime_utc == best.change_datetime_utc:
            if _payload(best) != _payload(record):
                known_conflict = True
            else:
                best = _preferred(best,record,customer_id)
    _check(check)
    if not count:
        return ProfileSelection(ProfileStatus.MISSING if source_available else ProfileStatus.SOURCE_UNAVAILABLE)
    if known_conflict or unknown_conflict or best is not None and unknown is not None and _payload(best) != _payload(unknown):
        return ProfileSelection(ProfileStatus.CONFLICT,candidates=count)
    return ProfileSelection(ProfileStatus.AVAILABLE,best if best is not None else unknown,count)


def _plain(value):
    if isinstance(value,Mapping):
        require(all(isinstance(k,str) for k in value))
        return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_plain(v) for v in value]
    require(value is None or isinstance(value,(str,int,float,Decimal,bool)))
    return value


def _subscriptions(value):
    try:
        return encode(_plain(value))
    except (ValueError,TypeError,OverflowError):
        raise DetailContractError() from None


def project_client(customer_id, selection, metadata, *, strict=None, extended=None, rfm=(), check=None):
    _check(check)
    require(valid_id(customer_id) and isinstance(selection,ProfileSelection) and isinstance(metadata,DetailMetadata))
    for policy,metrics in zip(POLICIES,(strict,extended)):
        if metrics is not None:
            require(isinstance(metrics,BuyerPolicyMetrics) and metrics.policy == policy and metrics.customer_id == customer_id)
            require(metrics.identity_space == metadata.identity_space and metrics.as_of_date == metadata.as_of_date)
    rfm = tuple(rfm)
    dimensions = set()
    for cell in rfm:
        require(isinstance(cell,PersonalRfm) and cell.customer_id == customer_id and cell.identity_space == metadata.identity_space)
        require(cell.as_of_date == metadata.as_of_date and cell.policy in POLICIES and cell.window in dict(RFM_WINDOWS))
        require(cell.currency in CURRENCIES and isinstance(cell.status,RfmAvailability))
        key = cell.policy,cell.window,cell.currency
        require(key not in dimensions)
        dimensions.add(key)
        require(cell.status is RfmAvailability.NOT_COMPUTED or valid_id(cell.thresholds_token))
        for count in (cell.r_days,cell.f_orders):
            require(count is None or type(count) is int and count >= 0)
        for score in (cell.r_score,cell.f_score,cell.m_score):
            require(score is None or type(score) is int and 1 <= score <= 5)
        require(cell.monetary is None or isinstance(cell.monetary,Decimal) and cell.monetary.is_finite() and cell.monetary >= 0)
        require(cell.segment is None or cell.segment in SEGMENTS)
        require(cell.recency_reliable is None or type(cell.recency_reliable) is bool)
        if cell.status in (RfmAvailability.NOT_COMPUTED,RfmAvailability.OUTSIDE_BASE,RfmAvailability.NOT_IN_CURRENCY_BASE):
            require(all(getattr(cell,name) is None for name in RFM_COLUMNS if name != 'status'))
        elif cell.status is RfmAvailability.AVAILABLE:
            require(all(v is not None for v in (cell.r_days,cell.f_orders,cell.monetary,cell.r_score,cell.f_score,cell.m_score)))
            require(cell.recency_reliable is True and cell.segment is not None and cell.segment != 'UNSCORABLE')
        else:
            require(cell.segment == 'UNSCORABLE' and None in (cell.r_score,cell.f_score,cell.m_score))
    record = selection.record
    if record is not None:
        require(record.customer_id == customer_id and valid_id(record.source_customer_id))
        require(record.change_datetime_utc is None or utc(record.change_datetime_utc))
    born = record.birth_date if record is not None else None
    require(born is None or type(born) is date)
    email, phone = (_clean(getattr(record,k)) if record is not None else None for k in ('email','mobile_phone'))
    present = None if selection.status is ProfileStatus.SOURCE_UNAVAILABLE else selection.candidates > 0
    changed = record.change_datetime_utc if record is not None else None
    _check(check)
    return ClientRow(Identifier(customer_id),Identifier(record.source_customer_id) if record is not None and
        record.source_customer_id != customer_id else None,present,selection.status,changed,
        changed >= metadata.cutoff_exclusive if changed is not None else None,
        *(_clean(getattr(record,k)) if record is not None else None for k in ('last_name','first_name','middle_name')),
        phone,email,_clean(record.sex) if record is not None else None,born,calendar_age(born,metadata.as_of_date),
        'MISSING' if born is None else 'FUTURE' if born > metadata.as_of_date else 'VALID',
        email is not None,phone is not None,_subscriptions(record.subscriptions) if record is not None else None,
        Permission.UNKNOWN,Permission.UNKNOWN,metadata.as_of_date,strict,extended,rfm)


def header_from_line(line):
    require(isinstance(line,OrderLineRecord))
    return OrderHeader(line.order_id,line.customer_id,line.source_customer_id,line.order_datetime_utc,
                       line.channel_external_id,line.channel_name,line.retail_order_id)


def project_line(line, ordinal, *, check=None):
    _check(check)
    require(isinstance(line,OrderLineRecord) and type(ordinal) is int and ordinal > 0)
    require(all(valid_id(v) for v in (line.order_id,line.customer_id,line.source_customer_id,line.line_id)))
    require(type(line.line_number) is int and utc(line.order_datetime_utc))
    require(isinstance(line.product,ProductKey) and valid_id(line.product.namespace) and valid_id(line.product.value))
    require(line.product_name is None or isinstance(line.product_name,str) and bool(line.product_name.strip()))
    _decimal(line.quantity)
    _decimal(line.price_of_line)
    _decimal(line.base_price_per_item,optional=True)
    policies = eligible_purchase_policies(line.line_status,line.quantity)
    require(valid_id(line.line_status))
    return OrderLineRow(line.order_id,line.customer_id,line.source_customer_id,ordinal,line.line_id,line.line_number,
        line.product.value,line.product.namespace,line.product_name,line.quantity,line.base_price_per_item,
        line.price_of_line,line.line_status,classify_status(line.line_status).value,
        STRICT_COMPLETED in policies,EXTENDED_COMPLETED in policies,CURRENCY_BY_NAMESPACE.get(line.product.namespace),
        'NEGATIVE_UNVALUED' if line.price_of_line < 0 else 'VALID_SOURCE_NUMBER')


def _order_policy(value):
    return OrderPolicyRow(value.has_purchase,value.line_count,value.units,value.amount,value.currency,
                          tuple(sorted(issue.value for issue in value.monetary_issues)))


def project_order(fact, lines, metadata, *, header=None, check=None):
    """An entire selected order or None; never emit a partial/filtered line bundle."""
    _check(check)
    _fact(fact,metadata)
    lines = tuple(lines)  # Bounded by one source order, never all Orders.
    require(header is not None or bool(lines))
    header = header_from_line(lines[0]) if header is None else header
    require(isinstance(header,OrderHeader) and utc(header.order_datetime_utc))
    require(all(valid_id(v) for v in (header.order_id,header.customer_id,header.source_customer_id,header.channel_external_id)))
    require(header.retail_order_id is None or valid_id(header.retail_order_id))
    require(isinstance(header.channel_name,str) and bool(header.channel_name.strip()))
    require((header.order_id,header.customer_id,header.order_datetime_utc) ==
            (fact.order_id,fact.customer_id,fact.order_datetime_utc))
    if not metadata.includes_event(fact.order_datetime_utc):
        return None
    projected = []
    for index,line in enumerate(lines,1):
        require(header_from_line(line) == header)
        projected.append(project_line(line,index,check=check))
    groups = Counter(row.status_group for row in projected)
    require(groups == dict((group.value,n) for group,n in fact.status_groups))
    require(sum(row.accepted_strict for row in projected) == fact.strict.line_count and
            sum(row.accepted_extended for row in projected) == fact.extended.line_count)
    _check(check)
    if metadata.order_mode is OrderMode.PURCHASE_EITHER and not fact.extended.has_purchase:
        return None
    returned, cancelled = groups[StatusGroup.RETURNED.value],groups[StatusGroup.CANCELLED.value]
    row = OrderRow(header.order_id,header.retail_order_id,header.customer_id,header.source_customer_id,
                   header.order_datetime_utc,header.channel_external_id,header.channel_name,len(lines),returned,cancelled,
                   len(lines)-returned-cancelled,_order_policy(fact.strict),_order_policy(fact.extended))
    return OrderBundle(row,tuple(projected))


def _exact_add(a,b):
    # Source amounts are already policy facts; add them without losing decimal
    # digits when the total exceeds the process's default precision.
    with localcontext() as context:
        context.prec = max(context.prec,max(a.adjusted(),b.adjusted())-min(a.as_tuple().exponent,b.as_tuple().exponent)+2)
        return a+b


def buyer_metrics(customer_id, facts, metadata, *, check=None):
    """One customer's already deduplicated full history; constant state, no RFM scoring."""
    require(valid_id(customer_id) and isinstance(metadata,DetailMetadata))
    windows = list(rfm_windows(metadata.as_of_date).values())[:3]
    coverage = metadata.effective_coverage
    states = {p:dict(orders=0,positions=0,units=Decimal(0),first=None,second=None,last=None,
                    windows=[0,0,0],money={c:Decimal(0) for c in CURRENCIES},valued=0,unknown=0,outside=0) for p in POLICIES}
    for fact in facts:
        _check(check)
        _fact(fact,metadata,customer_id)
        occurred = fact.order_datetime_utc
        if occurred >= metadata.cutoff_exclusive:
            continue
        covered = any(a <= occurred < b for a,b in coverage)
        key = occurred,fact.order_id
        for policy,purchase in ((STRICT_COMPLETED,fact.strict),(EXTENDED_COMPLETED,fact.extended)):
            if not purchase.has_purchase:
                continue
            s = states[policy]
            s['orders'] += 1
            s['positions'] += purchase.line_count
            s['units'] = _exact_add(s['units'],purchase.units)
            if s['first'] is None or key < s['first']:
                s['second'],s['first'] = s['first'],key
            elif s['second'] is None or key < s['second']:
                s['second'] = key
            s['last'] = max(s['last'],occurred) if s['last'] is not None else occurred
            for i,(a,b) in enumerate(windows):
                s['windows'][i] += int(a <= occurred < b)
            if purchase.monetary_available:
                require(purchase.currency in CURRENCIES)
                s['money'][purchase.currency] = _exact_add(s['money'][purchase.currency],purchase.amount)
                s['valued'] += 1
            else:
                s['unknown'] += 1
            s['outside'] += int(not covered)
    result = []
    for policy,s in states.items():
        _check(check)
        n = s['orders']
        first = s['first'][0] if n else None
        second = s['second'][0] if s['second'] is not None else None
        status = metadata.coverage_status
        if s['outside'] and status is CoverageStatus.FULL:
            status = CoverageStatus.PARTIAL
        complete = status is CoverageStatus.FULL and not s['unknown']
        rub,kzt = (Money(s['money'][c]) for c in CURRENCIES)
        result.append(BuyerPolicyMetrics(policy,Identifier(customer_id),metadata.identity_space,metadata.as_of_date,
            n,s['positions'],s['units'],first,s['last'],max(n-1,0),
            Decimal((s['last'].date()-first.date()).days)/Decimal(n-1) if n>1 else None,
            *s['windows'],second is not None,(second.date()-first.date()).days if second is not None else None,
            f'{first.year:04d}-{first.month:02d}' if first is not None else None,rub,kzt,s['valued'],s['unknown'],
            not s['unknown'],complete,rub if complete else None,kzt if complete else None,status,s['outside']))
    return tuple(result)


def row_cells(row):
    """Ordered typed cells, with a non-PII repr; accessing values is explicit."""
    if isinstance(row,ClientRow):
        values = [getattr(row,name) for name in CLIENT_BASE_COLUMNS]
        for metrics in (row.strict,row.extended):
            values.append('NOT_COMPUTED' if metrics is None else 'OBSERVED')
            values.extend(getattr(metrics,name) if metrics is not None else None for name in METRIC_COLUMNS)
        rfm = {(c.policy,c.window,c.currency):c for c in row.rfm}
        for policy in POLICIES:
            for window,_ in RFM_WINDOWS:
                for currency in CURRENCIES:
                    cell = rfm.get((policy,window,currency))
                    values.extend(getattr(cell,name) if cell is not None else
                                  RfmAvailability.NOT_COMPUTED if name=='status' else None for name in RFM_COLUMNS)
    elif isinstance(row,OrderRow):
        values = [getattr(row,name) for name in ORDER_BASE_COLUMNS]
        for policy in (row.strict,row.extended):
            values.extend(json.dumps(policy.monetary_issues) if name=='monetary_issues' else getattr(policy,name)
                          for name in ORDER_POLICY_COLUMNS)
    elif isinstance(row,OrderLineRow):
        values = [getattr(row,name) for name in ORDER_LINE_COLUMNS]
    else:
        raise DetailContractError()
    return DetailCells(tuple(values))


def csv_cells(row):
    return DetailCells(tuple(csv_cell(value) for value in row_cells(row)))
