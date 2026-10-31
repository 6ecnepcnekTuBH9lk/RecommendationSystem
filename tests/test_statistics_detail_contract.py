"""Detail schema/encoding only; fabricated values, no real exports or Git."""

from datetime import date, datetime, timezone
from decimal import Decimal
import math

import pytest

from Application.statistics_detail_contract import (
    CLIENT_COLUMNS, ORDER_COLUMNS, ORDER_LINE_COLUMNS, CLIENT_BASE_COLUMNS, METRIC_COLUMNS, RFM_COLUMNS,
    DetailCells, DetailContractError, DetailMetadata, DetailPeriod, CoverageStatus,
    CSV_ENCODING, CSV_DELIMITER, NULL_TOKEN, csv_cell, decode_csv_text,
)


SPACE = 'REFERENCE:detail-synthetic'
AS_OF = date(2026,8,31)
FULL = (('2020-01-01T00:00:00Z','2026-09-01T00:00:00Z'),)


def metadata(**kwargs):
    return DetailMetadata(**dict(as_of_date=AS_OF,event_period=DetailPeriod(date(2026,1,1),AS_OF),
                                 identity_space=SPACE,orders_coverage=FULL,**kwargs))


def test_stable_column_names_order_and_three_table_shapes():
    assert (len(CLIENT_COLUMNS),len(ORDER_COLUMNS),len(ORDER_LINE_COLUMNS)) == (177,23,18)
    assert CLIENT_BASE_COLUMNS == ('customer_id','source_customer_id','profile_present','profile_status',
        'profile_updated_utc','profile_after_analysis_date','last_name','first_name','middle_name','mobile_phone',
        'email','sex','birth_date','age','birth_date_status','has_email','has_phone','subscriptions_json',
        'email_permission','sms_permission','as_of_date')
    assert ORDER_LINE_COLUMNS == ('order_id','customer_id','source_customer_id','line_ordinal','line_id','line_number',
        'product_id','product_namespace','product_name','quantity','base_price_per_item','price_of_line','line_status',
        'status_group','accepted_strict','accepted_extended','currency','price_status')
    assert ORDER_COLUMNS[:11] == ('order_id','retail_order_id','customer_id','source_customer_id','order_datetime_utc',
        'channel_external_id','channel_name','line_count','return_lines','cancelled_lines','other_status_lines')
    for columns in (CLIENT_COLUMNS,ORDER_COLUMNS,ORDER_LINE_COLUMNS):
        assert len(columns)==len(set(columns)) and all(name.isidentifier() for name in columns)
    assert CLIENT_COLUMNS[21]=='strict_metrics_status'
    assert CLIENT_COLUMNS[22:22+len(METRIC_COLUMNS)] == tuple('strict_'+name for name in METRIC_COLUMNS)
    assert CLIENT_COLUMNS[-len(RFM_COLUMNS):] == tuple('extended_rfm_48m_kzt_'+name for name in RFM_COLUMNS)
    assert CSV_ENCODING=='utf-8-sig' and CSV_DELIMITER==';'


@pytest.mark.parametrize('text',['000000123456789012345678901234567890','+70000000000',
    'SYNTH_EMAIL@example.invalid','=SUM(A1:A2)',' +SUM(A1:A2)','@A1','-1+2','\t=1+2',
    "'original",'','=value;"quoted"\nnext line',r'\N','NULL'])
def test_text_encoding_is_reversible_formula_safe_and_keeps_long_ids(text):
    encoded = csv_cell(text)
    assert encoded.startswith("'") and decode_csv_text(encoded)==text
    assert encoded != NULL_TOKEN
    assert not encoded.startswith(('=','+','-','@','\t'))


def test_null_empty_datetime_boolean_decimal_are_distinct_without_float_conversion():
    assert csv_cell(None)==NULL_TOKEN and decode_csv_text(NULL_TOKEN) is None
    assert csv_cell('')=="'" and decode_csv_text("'")==''
    assert csv_cell(Decimal('12345678901234567890.123456789'))=='12345678901234567890.123456789'
    assert csv_cell(Decimal('1E+5'))=='100000'
    assert csv_cell(Decimal('-2.5'))=='-2.5'
    assert csv_cell(datetime(2026,8,31,23,59,tzinfo=timezone.utc))=='2026-08-31T23:59:00+00:00'
    assert csv_cell(date(2026,8,31))=='2026-08-31'
    assert [csv_cell(v) for v in (True,False,12)]==['1','0','12']
    assert 'SYNTH_EMAIL' not in repr(DetailCells(('SYNTH_EMAIL@example.invalid',)))


@pytest.mark.parametrize('value',[1.25,math.inf,Decimal('NaN'),object(),'secret\x00value',datetime(2026,1,1)])
def test_invalid_csv_values_fail_without_echoing_input(value):
    with pytest.raises(DetailContractError) as error:
        csv_cell(value)
    assert str(error.value)=='Invalid detail export contract'
    assert 'secret' not in repr(error.value)


def test_period_is_independent_of_full_history_and_coverage_is_normalized():
    context = metadata()
    assert context.event_period.start_date==date(2026,1,1)
    assert context.orders_coverage[0][0].year==2020
    assert context.coverage_status is CoverageStatus.FULL
    assert context.includes_event(datetime(2026,8,31,23,59,59,tzinfo=timezone.utc))
    assert not context.includes_event(datetime(2020,1,1,tzinfo=timezone.utc))
    assert not context.includes_event(datetime(2026,9,1,tzinfo=timezone.utc))
    gap = DetailMetadata(AS_OF,DetailPeriod(),SPACE,(('2020-01-01T00:00:00Z','2021-01-01T00:00:00Z'),
        ('2025-01-01T00:00:00Z','2026-09-01T00:00:00Z')))
    assert gap.coverage_status is CoverageStatus.PARTIAL
    assert DetailMetadata(AS_OF,DetailPeriod(),SPACE,()).coverage_status is CoverageStatus.NONE


@pytest.mark.parametrize('kwargs',[
    {'as_of_date':'2026-08-31'}, {'identity_space':'INDEXED:private-generation'}, {'identity_space':' secret '},
    {'orders_coverage':(('bad-secret','invalid'),)},
    {'orders_coverage':(('2026-09-01T00:00:00Z','2026-01-01T00:00:00Z'),)},
    {'orders_coverage':(('2020-01-01','2026-09-01'),)},
    {'order_mode':'ALL_SELECTED'}, {'profile_snapshot_at':datetime(2026,1,1)},
    {'event_period':DetailPeriod(date(2026,1,1),date(2026,9,1))},
])
def test_metadata_rejects_ambiguous_or_indexed_context_without_data_in_errors(kwargs):
    values=dict(as_of_date=AS_OF,event_period=DetailPeriod(),identity_space=SPACE,orders_coverage=FULL)
    values.update(kwargs)
    with pytest.raises(DetailContractError) as error:
        DetailMetadata(**values)
    assert 'private-generation' not in str(error.value) and 'bad-secret' not in repr(error.value)


@pytest.mark.parametrize('start,end',[(date(2026,1,1),None),(None,date(2026,1,1)),
    (date(2026,2,1),date(2026,1,1)),('2026-01-01','2026-02-01')])
def test_invalid_period_is_not_silently_repaired(start,end):
    with pytest.raises(DetailContractError):
        DetailPeriod(start,end)
