"""Typed, non-anonymized detail rows. No reader, writer, GUI or implicit clock."""

from dataclasses import dataclass, fields
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import NewType

from Application.mindbox.records import CustomerRecord
from Application.statistics_period import shared_intervals
from Application.statistics_rfm_core import rfm_windows


Identifier = NewType('Identifier', str)
Money = NewType('Money', Decimal)
FORMAT_VERSION = 1
NULL_TOKEN = r'\N'
CSV_ENCODING = 'utf-8-sig'
CSV_DELIMITER = ';'
CSV_TEXT_PREFIX = "'"
POLICY_PREFIXES = ('strict','extended')
RFM_WINDOWS = (('LAST_12M','12m'),('LAST_24M','24m'),('LAST_48M','48m'))
CURRENCIES = ('RUB','KZT')


class DetailContractError(ValueError):
    def __init__(self):
        super().__init__('Invalid detail export contract')


def require(condition):
    if not condition:
        raise DetailContractError()


def valid_id(value):
    return isinstance(value,str) and bool(value.strip()) and value == value.strip()


def utc(value):
    return isinstance(value,datetime) and value.tzinfo is not None and value.utcoffset() == timedelta(0)


class ProfileStatus(str, Enum):
    AVAILABLE = 'AVAILABLE'
    MISSING = 'MISSING'
    SOURCE_UNAVAILABLE = 'SOURCE_UNAVAILABLE'
    CONFLICT = 'CONFLICT'


class Permission(str, Enum):
    UNKNOWN = 'UNKNOWN'


class CoverageStatus(str, Enum):
    FULL = 'FULL'
    PARTIAL = 'PARTIAL'
    NONE = 'NONE'


class OrderMode(str, Enum):
    ALL_SELECTED = 'ALL_SELECTED'
    PURCHASE_EITHER = 'PURCHASE_EITHER'


class RfmAvailability(str, Enum):
    NOT_COMPUTED = 'NOT_COMPUTED'
    OUTSIDE_BASE = 'OUTSIDE_BASE'
    NOT_IN_CURRENCY_BASE = 'NOT_IN_CURRENCY_BASE'
    AVAILABLE = 'AVAILABLE'
    UNSCORABLE = 'UNSCORABLE'


@dataclass(frozen=True, slots=True, repr=False)
class DetailPeriod:
    start_date: date | None = None
    end_date: date | None = None

    def __post_init__(self):
        require((self.start_date is None) == (self.end_date is None))
        if self.start_date is not None:
            require(type(self.start_date) is date and type(self.end_date) is date and self.start_date <= self.end_date)

    def contains(self, occurred):
        require(utc(occurred))
        return self.start_date is None or self.start_date <= occurred.date() <= self.end_date


@dataclass(frozen=True, slots=True, repr=False)
class DetailMetadata:
    as_of_date: date
    event_period: DetailPeriod
    identity_space: str
    orders_coverage: tuple[tuple[datetime,datetime], ...]
    order_mode: OrderMode = OrderMode.ALL_SELECTED
    profile_snapshot_at: datetime | None = None

    def __post_init__(self):
        require(type(self.as_of_date) is date and isinstance(self.event_period,DetailPeriod))
        require(valid_id(self.identity_space) and not self.identity_space.startswith('INDEXED'))
        require(isinstance(self.order_mode,OrderMode))
        require(self.profile_snapshot_at is None or utc(self.profile_snapshot_at))
        require(self.event_period.end_date is None or self.event_period.end_date <= self.as_of_date)
        try:
            rfm_windows(self.as_of_date)  # Same representable calendar limits and windows.
            parsed = shared_intervals([],[(a.isoformat() if isinstance(a,datetime) else a,
                                           b.isoformat() if isinstance(b,datetime) else b) for a,b in self.orders_coverage])
        except (ValueError,TypeError,AttributeError,OverflowError):
            raise DetailContractError() from None
        object.__setattr__(self,'orders_coverage',parsed)

    @property
    def cutoff_exclusive(self):
        return datetime.combine(self.as_of_date+timedelta(days=1),time.min,timezone.utc)

    @property
    def effective_coverage(self):
        return tuple((a,min(b,self.cutoff_exclusive)) for a,b in self.orders_coverage if a < self.cutoff_exclusive)

    @property
    def coverage_status(self):
        intervals = self.effective_coverage
        return (CoverageStatus.NONE if not intervals else CoverageStatus.FULL if len(intervals)==1 and
                intervals[0][1]==self.cutoff_exclusive else CoverageStatus.PARTIAL)

    def includes_event(self, occurred):
        return self.event_period.contains(occurred) and occurred < self.cutoff_exclusive


@dataclass(frozen=True, slots=True, repr=False)
class ProfileSelection:
    status: ProfileStatus
    record: CustomerRecord | None = None
    candidates: int = 0

    def __post_init__(self):
        require(isinstance(self.status,ProfileStatus) and type(self.candidates) is int and self.candidates >= 0)
        require((self.status is ProfileStatus.AVAILABLE) == (self.record is not None))
        require(self.record is None or isinstance(self.record,CustomerRecord))
        require((self.candidates > 0) == (self.status in (ProfileStatus.AVAILABLE,ProfileStatus.CONFLICT)))


@dataclass(frozen=True, slots=True, repr=False)
class BuyerPolicyMetrics:
    policy: str
    customer_id: Identifier
    identity_space: str
    as_of_date: date
    orders: int
    accepted_lines: int
    units: Decimal
    first_purchase_utc: datetime | None
    last_purchase_utc: datetime | None
    repeat_orders: int
    mean_interval_days: Decimal | None
    orders_12m: int
    orders_24m: int
    orders_48m: int
    has_second_purchase: bool
    days_to_second_purchase: int | None
    first_observed_cohort: str | None
    known_amount_rub: Money
    known_amount_kzt: Money
    valued_orders: int
    unvalued_orders: int
    observed_amounts_complete: bool
    financial_complete: bool
    complete_amount_rub: Money | None
    complete_amount_kzt: Money | None
    coverage_status: CoverageStatus
    orders_outside_coverage: int


@dataclass(frozen=True, slots=True, repr=False)
class PersonalRfm:
    policy: str
    window: str
    currency: str
    customer_id: Identifier
    identity_space: str
    as_of_date: date
    thresholds_token: str | None = None
    status: RfmAvailability = RfmAvailability.NOT_COMPUTED
    r_days: int | None = None
    f_orders: int | None = None
    monetary: Money | None = None
    r_score: int | None = None
    f_score: int | None = None
    m_score: int | None = None
    segment: str | None = None
    recency_reliable: bool | None = None


@dataclass(frozen=True, slots=True, repr=False)
class ClientRow:
    customer_id: Identifier
    source_customer_id: Identifier | None
    profile_present: bool | None
    profile_status: ProfileStatus
    profile_updated_utc: datetime | None
    profile_after_analysis_date: bool | None
    last_name: str | None
    first_name: str | None
    middle_name: str | None
    mobile_phone: str | None
    email: str | None
    sex: str | None
    birth_date: date | None
    age: int | None
    birth_date_status: str
    has_email: bool
    has_phone: bool
    subscriptions_json: str | None
    email_permission: Permission
    sms_permission: Permission
    as_of_date: date
    strict: BuyerPolicyMetrics | None = None
    extended: BuyerPolicyMetrics | None = None
    rfm: tuple[PersonalRfm,...] = ()


@dataclass(frozen=True, slots=True, repr=False)
class OrderHeader:
    order_id: Identifier
    customer_id: Identifier
    source_customer_id: Identifier
    order_datetime_utc: datetime
    channel_external_id: str
    channel_name: str
    retail_order_id: Identifier | None = None


@dataclass(frozen=True, slots=True, repr=False)
class OrderPolicyRow:
    accepted: bool
    accepted_lines: int
    units: Decimal
    amount: Money | None
    currency: str | None
    monetary_issues: tuple[str,...]


@dataclass(frozen=True, slots=True, repr=False)
class OrderRow:
    order_id: Identifier
    retail_order_id: Identifier | None
    customer_id: Identifier
    source_customer_id: Identifier
    order_datetime_utc: datetime
    channel_external_id: str
    channel_name: str
    line_count: int
    return_lines: int
    cancelled_lines: int
    other_status_lines: int
    strict: OrderPolicyRow
    extended: OrderPolicyRow


@dataclass(frozen=True, slots=True, repr=False)
class OrderLineRow:
    order_id: Identifier
    customer_id: Identifier
    source_customer_id: Identifier
    line_ordinal: int
    line_id: Identifier
    line_number: int
    product_id: Identifier
    product_namespace: str
    product_name: str | None
    quantity: Decimal
    base_price_per_item: Money | None
    price_of_line: Money
    line_status: str
    status_group: str
    accepted_strict: bool
    accepted_extended: bool
    currency: str | None
    price_status: str


@dataclass(frozen=True, slots=True, repr=False)
class OrderBundle:
    order: OrderRow
    lines: tuple[OrderLineRow,...]


@dataclass(frozen=True, slots=True, repr=False)
class DetailCells:
    values: tuple

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


CLIENT_BASE_COLUMNS = tuple(f.name for f in fields(ClientRow) if f.name not in ('strict','extended','rfm'))
METRIC_COLUMNS = tuple(f.name for f in fields(BuyerPolicyMetrics) if f.name not in ('policy','customer_id','identity_space','as_of_date'))
RFM_COLUMNS = tuple(f.name for f in fields(PersonalRfm) if f.name not in ('policy','window','currency','customer_id','identity_space','as_of_date','thresholds_token'))
ORDER_BASE_COLUMNS = tuple(f.name for f in fields(OrderRow) if f.name not in POLICY_PREFIXES)
ORDER_POLICY_COLUMNS = tuple(f.name for f in fields(OrderPolicyRow))
ORDER_LINE_COLUMNS = tuple(f.name for f in fields(OrderLineRow))
CLIENT_COLUMNS = (*CLIENT_BASE_COLUMNS,
    *(prefix+'_'+name for prefix in POLICY_PREFIXES for name in ('metrics_status',*METRIC_COLUMNS)),
    *(f'{prefix}_rfm_{label}_{currency.lower()}_{name}' for prefix in POLICY_PREFIXES
      for _,label in RFM_WINDOWS for currency in CURRENCIES for name in RFM_COLUMNS))
ORDER_COLUMNS = (*ORDER_BASE_COLUMNS,*(prefix+'_'+name for prefix in POLICY_PREFIXES for name in ORDER_POLICY_COLUMNS))


def csv_cell(value):
    """Pure future-writer encoding; text is reversibly Excel-safe, never masked."""
    if value is None:
        return NULL_TOKEN
    if isinstance(value,Enum):
        value = value.value
    if isinstance(value,str):
        require('\x00' not in value)
        return CSV_TEXT_PREFIX + value
    if type(value) is bool:
        return '1' if value else '0'
    if type(value) is int:
        return str(value)
    if isinstance(value,Decimal):
        require(value.is_finite())
        return format(value,'f')
    if isinstance(value,datetime):
        require(utc(value))
        return value.isoformat()
    if type(value) is date:
        return value.isoformat()
    raise DetailContractError()


def decode_csv_text(value):
    require(isinstance(value,str))
    if value == NULL_TOKEN:
        return None
    require(value.startswith(CSV_TEXT_PREFIX))
    return value[1:]
