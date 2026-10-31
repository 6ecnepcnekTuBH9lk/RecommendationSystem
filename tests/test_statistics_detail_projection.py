"""Prepared record projections with synthetic identities/contacts only."""

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
import json
from pathlib import Path

import pytest

from Application.mindbox.adapters.customers import adapt_customer
from Application.mindbox.adapters.orders import adapt_order
from Application.mindbox.adapters.customer_merges import adapt_customer_merge
from Application.mindbox.adapters._common import AdapterError
from Application.mindbox.identity import CustomerIdResolver
from Application.statistics_purchase_facts import build_purchase_order_fact
from Application.statistics_purchase_policy import STRICT_COMPLETED
from Application.statistics_detail_contract import (
    CLIENT_COLUMNS, ORDER_COLUMNS, ORDER_LINE_COLUMNS, DetailContractError, DetailMetadata, DetailPeriod,
    OrderHeader, OrderMode, ProfileStatus, Permission, CoverageStatus, PersonalRfm, RfmAvailability,
)
from Application.statistics_detail_projection import (
    select_profile, project_client, project_order, buyer_metrics, row_cells, csv_cells,
)
from Application.statistics_cohorts_core import CohortAggregates
from Application.statistics_rfm_core import RfmAggregates
from Application.statistics_buyer_aggregates import BuyerAggregates
from Application.analysis_filter import AnalysisFilter
from tests.test_dataset_statistics import order as raw_order
from tests.test_statistics_detail_contract import SPACE, AS_OF, FULL


CANONICAL = '000123456789012345678901234567890'
EMAIL = 'SYNTH_CONTACT@example.invalid'
PHONE = '+70000000000'


def resolver():
    return CustomerIdResolver([adapt_customer_merge({'id':'synthetic-merge','dateTimeUtc':'2026-01-01T00:00:00Z',
        'resultingCustomer':{'ids':{'mindboxId':CANONICAL}},
        'mergedCustomers':[{'ids':{'mindboxId':'alias-a'}},{'ids':{'mindboxId':'alias-b'}}]})])


def metadata(coverage=FULL,mode=OrderMode.ALL_SELECTED):
    return DetailMetadata(AS_OF,DetailPeriod(date(2026,1,1),AS_OF),SPACE,coverage,mode)


def profile(source='alias-a',email=EMAIL,changed='2026-07-01T00:00:00Z',**kwargs):
    raw={'ids':{'mindboxId':source},'firstName':'SYNTH_FIRST','lastName':'SYNTH_LAST','middleName':'SYNTH_MIDDLE',
         'email':email,'mobilePhone':PHONE,'sex':'female','birthDate':'2000-02-29',
         'changeDateTimeUtc':changed,'subscriptions':[{'channel':'email','isSubscribed':True}],**kwargs}
    return adapt_customer(raw,resolver())


def prepared(statuses=('F',),key='000000000000000000000000000001',day='2026-01-01',source='alias-a',namespace='offline1C',modify=None):
    raw=raw_order(statuses,key)
    raw['ids']['retailOrderId']='000000000099'
    raw['customer']['ids']['mindboxId']=source
    raw['firstAction']['dateTimeUtc']=day+'T12:00:00Z'
    for line in raw['lines']:
        line['product']['ids']={namespace:'000123456789012345678901234567890-size'}
    if modify is not None:
        modify(raw)
    lines=adapt_order(raw,resolver(),product_namespaces=tuple(dict.fromkeys((namespace,'offline1C','kanzlerKz'))))
    at=datetime.fromisoformat(day+'T12:00:00+00:00')
    fact=build_purchase_order_fact(lines,identity_space=SPACE,customer_id=resolver().resolve(source),order_id=key,order_datetime_utc=at)
    return fact,lines


def test_profile_keeps_real_string_values_and_subscriptions_without_inventing_consent():
    context=metadata()
    selected=select_profile(CANONICAL,[profile()])
    row=project_client(CANONICAL,selected,context)
    assert row.customer_id==CANONICAL and row.source_customer_id=='alias-a'
    assert row.profile_present and row.profile_status is ProfileStatus.AVAILABLE
    assert (row.email,row.mobile_phone,row.last_name,row.first_name,row.middle_name)==(EMAIL,PHONE,'SYNTH_LAST','SYNTH_FIRST','SYNTH_MIDDLE')
    assert row.age==26 and row.birth_date==date(2000,2,29)
    assert row.email_permission is row.sms_permission is Permission.UNKNOWN
    assert json.loads(row.subscriptions_json)==[{'channel':'email','isSubscribed':True}]
    assert row.has_email and row.has_phone and row.profile_after_analysis_date is False
    assert len(row_cells(row))==len(CLIENT_COLUMNS)
    assert len(csv_cells(row))==len(CLIENT_COLUMNS)
    assert dict(zip(CLIENT_COLUMNS,row_cells(row)))['strict_metrics_status']=='NOT_COMPUTED'
    assert dict(zip(CLIENT_COLUMNS,row_cells(row)))['strict_rfm_12m_rub_status'] is RfmAvailability.NOT_COMPUTED


@pytest.mark.parametrize('available,status',[(True,ProfileStatus.MISSING),(False,ProfileStatus.SOURCE_UNAVAILABLE)])
def test_missing_profile_keeps_order_owner_and_nulls_contacts(available,status):
    selected=select_profile(CANONICAL,[],source_available=available)
    row=project_client(CANONICAL,selected,metadata())
    assert row.customer_id==CANONICAL and row.profile_status is status
    assert row.profile_present is (False if available else None)
    assert row.email is row.mobile_phone is row.first_name is row.subscriptions_json is None
    assert row.age is None and not row.has_email and row.email_permission is Permission.UNKNOWN


def test_equal_email_never_resolves_two_distinct_customers():
    records=[profile(source='separate-a'),profile(source='separate-b')]
    rows=[project_client(p.customer_id,select_profile(p.customer_id,[p]),metadata()) for p in records]
    assert {r.customer_id for r in rows}=={'separate-a','separate-b'}
    assert rows[0].email==rows[1].email==EMAIL


def test_multiple_source_ids_are_one_canonical_buyer_with_latest_profile():
    a=profile('alias-a',email='OLD_SYNTH@example.invalid',changed='2025-01-01T00:00:00Z')
    b=profile('alias-b')
    assert select_profile(CANONICAL,[a,b])==select_profile(CANONICAL,[b,a])
    row=project_client(CANONICAL,select_profile(CANONICAL,[a,b]),metadata())
    assert row.email==EMAIL and row.source_customer_id=='alias-b'
    facts=[prepared(key='first')[0],prepared(key='second',day='2026-02-01',source='alias-b')[0]]
    strict,extended=buyer_metrics(CANONICAL,facts,metadata())
    row=project_client(CANONICAL,select_profile(CANONICAL,[b]),metadata(),strict=strict,extended=extended)
    assert row.strict.orders==2 and row.strict.repeat_orders==1


@pytest.mark.parametrize('unknown',[False,True])
def test_conflicting_contacts_never_choose_an_arbitrary_alias(unknown):
    a=profile('alias-a',email='OTHER_SYNTH@example.invalid',changed=None if unknown else '2026-07-01T00:00:00Z')
    b=profile('alias-b')
    for records in ([a,b],[b,a]):
        selected=select_profile(CANONICAL,records)
        assert selected.status is ProfileStatus.CONFLICT and selected.record is None
        row=project_client(CANONICAL,selected,metadata())
        assert row.profile_present and row.email is row.mobile_phone is None
        assert 'OTHER_SYNTH' not in repr(selected) and EMAIL not in repr(row)


def test_identical_profile_aliases_prefer_canonical_source_and_later_unambiguous_update():
    a=profile('alias-a')
    b=replace(a,source_customer_id=CANONICAL)
    assert select_profile(CANONICAL,[a,b]).record.source_customer_id==CANONICAL
    conflict=profile('alias-b',email='OTHER_SYNTH@example.invalid')
    latest=profile('alias-b',changed='2026-08-01T00:00:00Z')
    assert select_profile(CANONICAL,[a,conflict,latest]).record==latest


def test_empty_contacts_and_current_profile_after_analysis_date_are_explicit():
    row=project_client(CANONICAL,select_profile(CANONICAL,[profile(email='  ',mobilePhone=None,changed='2027-01-01T00:00:00Z')]),metadata())
    assert row.email is row.mobile_phone is None and not row.has_email and not row.has_phone
    assert row.profile_after_analysis_date is True and row.email_permission is Permission.UNKNOWN


@pytest.mark.parametrize('status,strict,extended,group',[
    ('F',True,True,'COMPLETED'),('sold',True,True,'COMPLETED'),('CP',False,True,'PARTIALLY_COMPLETED'),
    ('delivering',False,False,'PAID_OR_IN_PROGRESS'),('Return',False,False,'RETURNED'),
    ('CD',False,False,'CANCELLED'),('cancelled',False,False,'UNKNOWN')])
def test_order_and_lines_use_existing_policy_facts(status,strict,extended,group):
    fact,lines=prepared((status,))
    bundle=project_order(fact,lines,metadata())
    row,line=bundle.order,bundle.lines[0]
    assert row.strict.accepted is strict and row.extended.accepted is extended
    assert line.accepted_strict is strict and line.accepted_extended is extended and line.status_group==group
    assert row.order_id==line.order_id==fact.order_id
    assert row.customer_id==line.customer_id==CANONICAL and line.source_customer_id=='alias-a'
    assert line.product_id=='000123456789012345678901234567890-size'
    assert line.base_price_per_item==Decimal(10) and line.price_of_line==Decimal(20)
    assert row.strict.amount==(Decimal(20) if strict else None)  # Not totalPrice=999999 or quantity*price.
    assert len(row_cells(row))==len(ORDER_COLUMNS) and len(row_cells(line))==len(ORDER_LINE_COLUMNS)


@pytest.mark.parametrize('namespace,currency',[('offline1C','RUB'),('kanzlerKz','KZT'),('future_namespace',None)])
def test_original_products_and_currencies_need_no_current_catalog(namespace,currency):
    fact,lines=prepared(namespace=namespace)
    bundle=project_order(fact,lines,metadata())
    assert bundle.lines[0].product_namespace==namespace and bundle.lines[0].currency==currency
    assert bundle.order.strict.currency==currency if currency else bundle.order.strict.currency=='unknown'
    assert bundle.order.strict.amount==(Decimal(20) if currency else None)


def test_mixed_order_policy_currencies_can_differ_and_returns_are_not_subtracted():
    def mutate(raw):
        raw['lines'][1]['product']['ids']={'kanzlerKz':'ORIGINAL-KZT'}
    fact,lines=prepared(('F','CP','Return','CD','delivering'),modify=mutate)
    row=project_order(fact,lines,metadata()).order
    assert row.strict.currency=='RUB' and row.strict.amount==Decimal(20)
    assert row.extended.currency=='mixed' and row.extended.amount is None
    assert row.return_lines==row.cancelled_lines==1 and row.other_status_lines==3
    assert row.strict.units==Decimal(2) and row.extended.units==Decimal(4)


@pytest.mark.parametrize('quantity',[Decimal('1.5'),Decimal(0),Decimal(-1)])
def test_numeric_nonpurchase_quantities_are_retained_not_silently_filtered(quantity):
    fact,lines=prepared(modify=lambda raw:raw['lines'][0].update(quantity=quantity))
    bundle=project_order(fact,lines,metadata())
    assert len(bundle.lines)==bundle.order.line_count==1 and bundle.lines[0].quantity==quantity
    assert not bundle.order.strict.accepted and not bundle.lines[0].accepted_extended


def test_negative_numeric_source_price_is_retained_but_policy_money_stays_unavailable():
    fact,lines=prepared(modify=lambda raw:raw['lines'][0].update(priceOfLine=-5))
    bundle=project_order(fact,lines,metadata())
    assert bundle.lines[0].price_of_line==Decimal(-5) and bundle.lines[0].price_status=='NEGATIVE_UNVALUED'
    assert bundle.order.strict.accepted and bundle.order.strict.amount is None
    assert bundle.order.strict.monetary_issues==('INVALID_PRICE',)


@pytest.mark.parametrize('field,value',[('priceOfLine','SYNTH_SECRET_PRICE'),('quantity','SYNTH_SECRET_QUANTITY'),
                                       ('priceOfLine',None),('priceOfLine',float('nan'))])
def test_bad_raw_numbers_fail_closed_in_existing_adapter(field,value):
    with pytest.raises(AdapterError) as error:
        prepared(modify=lambda raw:raw['lines'][0].update({field:value}))
    assert 'SYNTH_SECRET' not in str(error.value)


def test_missing_product_fails_instead_of_creating_an_unlinked_or_synthetic_product():
    with pytest.raises(AdapterError):
        prepared(modify=lambda raw:raw['lines'][0].update(product={'ids':{}}))
    fact,lines=prepared()
    with pytest.raises(DetailContractError):
        project_order(fact,[replace(lines[0],product=None)],metadata())


def test_duplicate_line_ids_have_stable_ordinal_keys_and_zero_line_orders_require_header():
    fact,lines=prepared(('F','sold'),modify=lambda raw:raw['lines'][1].update(id='0',number=0))
    bundle=project_order(fact,lines,metadata())
    assert [line.line_id for line in bundle.lines]==['0','0']
    assert [line.line_ordinal for line in bundle.lines]==[1,2]
    empty=build_purchase_order_fact([],identity_space=SPACE,customer_id=CANONICAL,order_id='empty',
                                   order_datetime_utc=datetime(2026,1,1,tzinfo=timezone.utc))
    with pytest.raises(DetailContractError):
        project_order(empty,[],metadata())
    header=OrderHeader('empty',CANONICAL,'alias-a',empty.order_datetime_utc,'shop','Synthetic shop')
    assert project_order(empty,[],metadata(),header=header).order.line_count==0


def test_export_mode_keeps_or_excludes_nonpurchase_orders_as_a_whole():
    fact,lines=prepared(('Return',))
    assert project_order(fact,lines,metadata()) is not None
    assert project_order(fact,lines,metadata(mode=OrderMode.PURCHASE_EITHER)) is None


def test_per_customer_metrics_use_history_not_event_period_and_ignore_future_orders():
    facts=[prepared(key='old',day='2020-01-01')[0],prepared(key='second',day='2025-01-01')[0],
           prepared(key='current',day='2026-01-01')[0],prepared(key='future',day='2027-01-01')[0]]
    strict,extended=buyer_metrics(CANONICAL,reversed(facts),metadata())
    assert strict.orders==extended.orders==3 and strict.repeat_orders==2
    assert strict.first_observed_cohort=='2020-01' and strict.first_purchase_utc.year==2020
    assert strict.last_purchase_utc.year==2026 and (strict.orders_12m,strict.orders_24m,strict.orders_48m)==(1,2,2)
    assert strict.days_to_second_purchase==(date(2025,1,1)-date(2020,1,1)).days
    assert strict.known_amount_rub==Decimal(60) and strict.known_amount_kzt==0
    assert strict.mean_interval_days==Decimal((date(2026,1,1)-date(2020,1,1)).days)/2
    assert strict==buyer_metrics(CANONICAL,facts,metadata())[0]
    assert project_order(*prepared(day='2020-01-01'),metadata()) is None


def test_financial_unknowns_and_gaps_do_not_turn_into_complete_money():
    known=prepared(key='rub')[0]
    unknown=prepared(key='unknown',namespace='future_namespace')[0]
    kzt=prepared(key='kzt',namespace='kanzlerKz')[0]
    s,_=buyer_metrics(CANONICAL,[known,unknown,kzt],metadata())
    assert s.known_amount_rub==s.known_amount_kzt==Decimal(20)
    assert s.valued_orders==2 and s.unvalued_orders==1 and not s.financial_complete
    assert s.complete_amount_rub is s.complete_amount_kzt is None
    for coverage,status in (((),CoverageStatus.NONE),
        ((('2020-01-01T00:00:00Z','2021-01-01T00:00:00Z'),('2026-01-01T00:00:00Z','2026-09-01T00:00:00Z')),CoverageStatus.PARTIAL)):
        result=buyer_metrics(CANONICAL,[known],metadata(coverage))[0]
        assert result.coverage_status is status and not result.financial_complete
        assert result.known_amount_rub==Decimal(20) and result.complete_amount_rub is None


def test_money_sums_keep_decimal_digits_of_prepared_facts():
    fact=prepared()[0]
    with localcontext() as context:
        context.prec=80
        big=Decimal('123456789012345678901234567890.123456789')
        a=replace(fact,strict=replace(fact.strict,amount=big),extended=replace(fact.extended,amount=big))
    b=replace(fact,order_id='tiny',strict=replace(fact.strict,amount=Decimal('.000000001')),
              extended=replace(fact.extended,amount=Decimal('.000000001')))
    s,_=buyer_metrics(CANONICAL,[a,b],metadata())
    assert str(s.known_amount_rub)=='123456789012345678901234567890.123456790'


def test_individual_counts_money_cohort_and_windows_match_existing_cores():
    facts=[prepared(key='a',day='2025-01-01',statuses=('CP',))[0],
           prepared(key='b',day='2026-01-01',statuses=('F','Return'))[0],
           prepared(key='c',day='2026-02-01',statuses=('sold',))[0]]
    context=metadata()
    strict,extended=buyer_metrics(CANONICAL,facts,context)
    cohorts=CohortAggregates(AS_OF,context.orders_coverage,identity_space=SPACE)
    rfm=RfmAggregates(AS_OF,context.orders_coverage,identity_space=SPACE)
    buyers=BuyerAggregates(SPACE,AnalysisFilter(),context.orders_coverage,orders_available=True)
    for fact in facts:
        cohorts.add(fact)
        rfm.add(fact)
        buyers.add(fact)
    b=buyers.result(AS_OF)['policies']
    for metrics in (strict,extended):
        assert metrics.orders==b[metrics.policy]['orders']
        assert metrics.known_amount_rub==Decimal(b[metrics.policy]['money']['RUB']['amount'])
        row=cohorts.result()['policies'][metrics.policy]['cohorts'][metrics.first_observed_cohort]
        assert row['cohort_size']==1 and row['second_purchase_days']['distribution']==[[metrics.days_to_second_purchase,1]]
        windows=rfm.result()['policies'][metrics.policy]['windows']
        assert [metrics.orders_12m,metrics.orders_24m,metrics.orders_48m]==[
            windows[name]['behavior']['observed_orders'] for name in ('LAST_12M','LAST_24M','LAST_48M')]


def test_projections_have_no_io_resolution_fact_rebuild_or_pii_repr(monkeypatch):
    import builtins
    import sqlite3
    from Application import statistics_purchase_facts
    fact,lines=prepared(('F','CP'))
    context=metadata()
    selected=select_profile(CANONICAL,[profile()])
    def forbidden(*args,**kwargs):
        raise AssertionError('Unexpected I/O or repeated source work')
    with monkeypatch.context() as patch:
        patch.setattr(builtins,'open',forbidden)
        patch.setattr(Path,'open',forbidden)
        patch.setattr(sqlite3,'connect',forbidden)
        patch.setattr(CustomerIdResolver,'resolve',forbidden)
        patch.setattr(statistics_purchase_facts,'build_purchase_order_fact',forbidden)
        strict,extended=buyer_metrics(CANONICAL,[fact],context)
        client=project_client(CANONICAL,selected,context,strict=strict,extended=extended)
        bundle=project_order(fact,lines,context)
        objects=(client,selected,bundle,bundle.order,bundle.lines[0],row_cells(client),csv_cells(client),strict)
    for value in objects:
        assert all(secret not in repr(value) for secret in (CANONICAL,EMAIL,PHONE,'SYNTH_FIRST'))
    with pytest.raises(DetailContractError) as error:
        project_order(replace(fact,customer_id='SYNTH_SECRET_WRONG'),lines,context)
    assert 'SYNTH_SECRET_WRONG' not in repr(error.value)


def test_cancellation_and_identity_mismatch_do_not_return_partial_records():
    fact,lines=prepared(('F','sold'))
    def cancel():
        raise InterruptedError(EMAIL)
    with pytest.raises(InterruptedError) as error:
        project_order(fact,lines,metadata(),check=cancel)
    assert EMAIL not in str(error.value)
    with pytest.raises(DetailContractError):
        buyer_metrics('other-owner',[fact],metadata())
    with pytest.raises(DetailContractError):
        project_order(replace(fact,identity_space='INDEXED:synthetic'),lines,metadata())


def test_personal_rfm_slots_do_not_invent_scores_from_aggregate_segments():
    context=metadata()
    cell=PersonalRfm(STRICT_COMPLETED,'LAST_12M','RUB',CANONICAL,SPACE,AS_OF)
    row=project_client(CANONICAL,select_profile(CANONICAL,[]),context,rfm=(cell,))
    assert cell.status is RfmAvailability.NOT_COMPUTED and cell.segment is None
    cells=dict(zip(CLIENT_COLUMNS,row_cells(row)))
    assert cells['strict_rfm_12m_rub_r_score'] is None and cells['extended_rfm_48m_kzt_segment'] is None
    with pytest.raises(DetailContractError):
        project_client(CANONICAL,select_profile(CANONICAL,[]),context,rfm=(cell,cell))



def test_bad_computed_rfm_and_cross_customer_metrics_cannot_be_attached():
    context=metadata()
    selected=select_profile(CANONICAL,[])
    cell=PersonalRfm(STRICT_COMPLETED,'LAST_12M','RUB',CANONICAL,SPACE,AS_OF)
    for bad in (replace(cell,r_score=5),replace(cell,status=RfmAvailability.AVAILABLE,thresholds_token='population'),
                replace(cell,customer_id='different-owner')):
        with pytest.raises(DetailContractError):
            project_client(CANONICAL,selected,context,rfm=(bad,))
    metrics=buyer_metrics(CANONICAL,[prepared()[0]],context)[0]
    with pytest.raises(DetailContractError):
        project_client(CANONICAL,selected,context,strict=replace(metrics,customer_id='different-owner'))


def test_empty_history_is_distinct_from_unknown_coverage_and_single_purchase_has_no_interval():
    empty=buyer_metrics(CANONICAL,[],metadata())[0]
    assert empty.orders==0 and empty.first_purchase_utc is empty.first_observed_cohort is None
    assert empty.mean_interval_days is None and empty.complete_amount_rub==Decimal(0)
    missing=buyer_metrics(CANONICAL,[],metadata(()))[0]
    assert missing.complete_amount_rub is None and missing.coverage_status is CoverageStatus.NONE
    single=buyer_metrics(CANONICAL,[prepared()[0]],metadata())[0]
    assert single.orders==1 and single.mean_interval_days is None and not single.has_second_purchase


def test_profile_payload_unknown_timestamps_cannot_silently_override_contacts():
    a=profile('alias-a',changed=None)
    b=profile('alias-b',changed=None,email='OTHER_SYNTH@example.invalid')
    assert select_profile(CANONICAL,[a,b]).status is ProfileStatus.CONFLICT
    assert select_profile(CANONICAL,[a,profile('alias-b')]).status is ProfileStatus.AVAILABLE


def test_source_snapshot_conflicts_use_existing_safe_failure_not_latest_order_wins():
    from Application.mindbox.order_dedup import OrderSnapshots, OrderSnapshotConflict
    selector=OrderSnapshots()
    a=raw_order(('Return',),'SYNTH_PRIVATE_ORDER')
    b=raw_order(('F',),'SYNTH_PRIVATE_ORDER')
    assert selector.accept(a) and not selector.accept(a)
    with pytest.raises(OrderSnapshotConflict) as error:
        selector.accept(b)
    assert 'SYNTH_PRIVATE_ORDER' not in repr(error.value)



def test_blank_adapted_phone_and_email_are_null_not_permission():
    record=replace(profile(),mobile_phone='   ',email='')
    row=project_client(CANONICAL,select_profile(CANONICAL,[record]),metadata())
    assert row.mobile_phone is row.email is None and not row.has_phone and not row.has_email
    assert row.email_permission is row.sms_permission is Permission.UNKNOWN


def test_fresh_import_and_preparation_need_no_qt_database_files_or_network():
    import os
    import subprocess
    import sys
    script="""
import os,sys
from datetime import date
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if (isinstance(mode,str) and any(c in mode for c in 'wax+') or
                isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            raise AssertionError('Unexpected write')
        if isinstance(path,str) and (path.endswith('customers.sqlite') or 'input_data' in path):
            raise AssertionError('Unexpected data read')
    if event in ('os.mkdir','sqlite3.connect','socket.connect','subprocess.Popen'):
        raise AssertionError('Unexpected I/O')
sys.addaudithook(guard)
from Application.statistics_detail_contract import DetailMetadata, DetailPeriod
from Application.statistics_detail_projection import select_profile, project_client, buyer_metrics
context=DetailMetadata(date(2026,8,31),DetailPeriod(),'REFERENCE:synthetic',())
metrics=buyer_metrics('synthetic',[],context)
row=project_client('synthetic',select_profile('synthetic',[]),context,strict=metrics[0],extended=metrics[1])
assert row.email is None and row.customer_id=='synthetic'
assert not any(name.startswith(('PyQt','qt_material','torch')) for name in sys.modules)
"""
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parents[1],
                          env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
