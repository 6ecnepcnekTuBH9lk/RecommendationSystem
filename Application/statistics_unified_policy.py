"""Status and quantity rules for the isolated UNIFIED_PURCHASE_V1 core."""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Generic, TypeVar
from unicodedata import category

UNIFIED_PURCHASE_V1 = "UNIFIED_PURCHASE_V1"
MINDBOX_STATUS_CATALOG_V1 = "MINDBOX_STATUS_CATALOG_V1"
ORDER_CREATED_PURCHASE_PROXY_V1 = "ORDER_CREATED_PURCHASE_PROXY_V1"
T = TypeVar("T")


class EvidenceQuality(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"
    INVALID = "INVALID"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"


@dataclass(frozen=True, slots=True)
class Measured(Generic[T]):
    value: T | None
    quality: EvidenceQuality
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.quality, EvidenceQuality):
            raise TypeError("Invalid evidence quality")
        if self.quality is EvidenceQuality.KNOWN:
            if self.value is None:
                raise ValueError("Known measurements require a value")
        elif self.value is not None or not self.reasons:
            raise ValueError("Unavailable measurements require reasons and no value")
        object.__setattr__(self, "reasons", tuple(sorted(set(self.reasons))))


@dataclass(frozen=True, slots=True)
class PurchasePolicy:
    policy_id: str = UNIFIED_PURCHASE_V1
    status_catalog_id: str = MINDBOX_STATUS_CATALOG_V1
    time_basis_id: str = ORDER_CREATED_PURCHASE_PROXY_V1
    snapshot_policy_id: str = "ORDER_OBSERVATION_RECONCILIATION_V1"
    money_policy_id: str = "OBSERVED_LINE_VALUE_V1"


class StatusGroup(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    PAID_OR_IN_PROGRESS = "PAID_OR_IN_PROGRESS"
    ORDER_IN_PROGRESS = "ORDER_IN_PROGRESS"
    DELIVERY_IN_PROGRESS = "DELIVERY_IN_PROGRESS"
    CANCELLED = "CANCELLED"
    RETURNED = "RETURNED"
    UNKNOWN = "UNKNOWN"

    @property
    def in_progress(self) -> bool:
        return self.value.endswith("_IN_PROGRESS")


@dataclass(frozen=True, slots=True)
class StatusDefinition:
    external_id: str
    name: str | None
    mindbox_category: str | None
    group: StatusGroup
    metadata_confirmation: str

    @property
    def metadata_source(self) -> str | None:
        return "STAT-ORDER-STATUS-DIAGNOSTICS-01" if self.name is not None else None

    @property
    def purchase_candidate(self) -> bool:
        return self.group in (StatusGroup.COMPLETED, StatusGroup.PARTIALLY_COMPLETED)

    @property
    def cancelled(self) -> bool:
        return self.group is StatusGroup.CANCELLED

    @property
    def returned(self) -> bool:
        return self.group is StatusGroup.RETURNED


# User-confirmed inventory (42 codes) and the technical cancellation code.
# The inventory CSV is a test reference, not a runtime input.
_CATALOG_ROWS = (
    ('CD', 'Отменен', 'Отменена', StatusGroup.CANCELLED),
    ('CP', 'Выполнен частично', 'Доставлена', StatusGroup.PARTIALLY_COMPLETED),
    ('DA', 'Комплектация заказа', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('DB', 'Готов к отправке', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('DD', 'Комплектация заказа (некомплект)', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('DG', 'Заказ отправлен', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('DS', 'Доставлен в пункт самовывоза', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('F', 'Выполнен', 'Доставлена', StatusGroup.COMPLETED),
    ('Mindbox-Tech-Cancellation', None, None, StatusGroup.CANCELLED),
    ('N', 'Принят', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('RJ', 'Отказ', 'Отменена', StatusGroup.CANCELLED),
    ('Return', 'Возврат', 'Возврат', StatusGroup.RETURNED),
    ('already-buyed', 'Купил в другом месте', 'Отменена', StatusGroup.CANCELLED),
    ('arrived-in-pickup-point', 'Прибыл в ПВЗ', 'Доставлена', StatusGroup.DELIVERY_IN_PROGRESS),
    ('assembling', 'Комплектуется', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('assembling-complete', 'Укомплектован', 'Оплачена', StatusGroup.PAID_OR_IN_PROGRESS),
    ('availability-confirmed', 'Наличие подтверждено', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('client-confirmed', 'Согласовано с клиентом', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('de', 'Доставка по графику', 'Доставлена', StatusGroup.DELIVERY_IN_PROGRESS),
    ('delivering', 'Доставляется', 'Оплачена', StatusGroup.PAID_OR_IN_PROGRESS),
    ('delyvery-did-not-suit', 'Не устроила доставка', 'Отменена', StatusGroup.CANCELLED),
    ('gotov-otpravka', 'Готов к отправке (crm)', 'Оплачена', StatusGroup.PAID_OR_IN_PROGRESS),
    ('klientskii-vozvrat', 'Клиентский возврат', 'Возврат', StatusGroup.RETURNED),
    ('kurerskii-vozvrat', 'Курьерский возврат', 'Возврат', StatusGroup.RETURNED),
    ('kv', 'К выполнению', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('new', 'Новый', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('no-call', 'Недозвон', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('no-product', 'Нет в наличии (crm)', 'Отменена', StatusGroup.CANCELLED),
    ('offer-analog', 'Предложить замену', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('opr-perem', 'Оприходовать перемещения', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('otmenen', 'Отменён', 'Отменена', StatusGroup.CANCELLED),
    ('out-of-stock', 'Нет в наличии', 'Отменена', StatusGroup.CANCELLED),
    ('prepayed', 'Предоплата поступила', 'Оплачена', StatusGroup.PAID_OR_IN_PROGRESS),
    ('ready-for-self-pickup', 'Готов к самовывозу', 'Доставлена', StatusGroup.DELIVERY_IN_PROGRESS),
    ('ready-to-wait', 'Готов ждать', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('redirect', 'Доставка перенесена', 'Оплачена', StatusGroup.PAID_OR_IN_PROGRESS),
    ('send-to-assembling', 'Передано в комплектацию', 'Оплачена', StatusGroup.PAID_OR_IN_PROGRESS),
    ('send-to-delivery', 'Передан в доставку', 'Доставлена', StatusGroup.DELIVERY_IN_PROGRESS),
    ('sh', 'Собран в магазине', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('soglasovanie', 'Согласование', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('sold', 'Продан', 'Доставлена', StatusGroup.COMPLETED),
    ('tovar-naiden', 'Товар найден', 'Оформлена', StatusGroup.ORDER_IN_PROGRESS),
    ('Оплачен', 'Оплачен', 'Оплачена', StatusGroup.PAID_OR_IN_PROGRESS),
)

STATUS_CATALOG = MappingProxyType({
    code: StatusDefinition(code, name, category_name, group,
                           "OFFICIAL_METADATA_UNCONFIRMED" if name is None else
                           "USER_REFERENCE_CONFIRMED; CURRENT_TENANT_COMPLETENESS_UNVERIFIED")
    for code, name, category_name, group in _CATALOG_ROWS
})


def classify_status_code(value: object) -> tuple[Measured[str], StatusGroup]:
    if value is None or isinstance(value, str) and not value:
        return Measured(None, EvidenceQuality.UNKNOWN, ("MISSING_CODE",)), StatusGroup.UNKNOWN
    if (not isinstance(value, str) or value != value.strip()
            or any(category(char) in {"Cc", "Cf", "Cs"} for char in value)):
        return Measured(None, EvidenceQuality.INVALID, ("INVALID_CODE",)), StatusGroup.UNKNOWN
    definition = STATUS_CATALOG.get(value)
    if definition is None:
        return Measured(value, EvidenceQuality.KNOWN, ("UNKNOWN_CODE",)), StatusGroup.UNKNOWN
    return Measured(value, EvidenceQuality.KNOWN), definition.group


def _decimal_input(value: object) -> Decimal | None:
    if type(value) not in (int, float, Decimal):
        return None
    number = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    if not number.is_finite():
        return None
    # Canonicalize without Decimal.normalize(), which rounds under the ambient context.
    if not number:
        return Decimal(0)
    sign, digits, exponent = number.as_tuple()
    last = len(digits)
    while digits[last - 1] == 0:
        last -= 1
    return Decimal((sign, digits[:last], exponent + len(digits) - last))


def qualify_quantity(value: object) -> Measured[Decimal]:
    """Validate source quantity, including the diagnostic known zero."""
    if value is None:
        return Measured(None, EvidenceQuality.UNKNOWN, ("MISSING_QUANTITY",))
    quantity = _decimal_input(value)
    if quantity is None:
        return Measured(None, EvidenceQuality.INVALID, ("INVALID_QUANTITY",))
    if quantity < 0:
        return Measured(None, EvidenceQuality.INVALID, ("NEGATIVE_QUANTITY",))
    if quantity != quantity.to_integral_value():
        return Measured(None, EvidenceQuality.INVALID, ("FRACTIONAL_QUANTITY",))
    return Measured(quantity, EvidenceQuality.KNOWN, ("ZERO_QUANTITY",) if not quantity else ())


def is_purchase_candidate(status: object, quantity: object) -> bool:
    definition = STATUS_CATALOG.get(status) if isinstance(status, str) else None
    if definition is None or not definition.purchase_candidate:
        return False
    units = qualify_quantity(quantity).value
    return units is not None and units > 0
