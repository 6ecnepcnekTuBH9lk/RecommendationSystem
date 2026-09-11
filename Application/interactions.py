"""Typed records -> item interactions. Без I/O, разрешения identity и весов."""

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType

from Application.mindbox.records import ActionRecord, OrderLineRecord, ProductKey


class InteractionType(str, Enum):
    VIEW = "VIEW"
    FAVORITE = "FAVORITE"
    PURCHASE = "PURCHASE"


class InteractionSource(str, Enum):
    ACTION = "ACTION"
    ORDER = "ORDER"


@dataclass(frozen=True, repr=False)
class InteractionRecord:
    source_customer_id: str
    customer_id: str
    product: ProductKey
    interaction_type: InteractionType
    event_datetime_utc: datetime
    source: InteractionSource
    source_event_id: str
    quantity: Decimal | None = None


class InteractionConfigError(ValueError):
    """Неоднозначные или некорректные правила классификации."""


class InteractionBuildError(Exception):
    """Malformed business event; сообщение не содержит ID и значений данных."""


VIEW_ACTION_SYSTEM_NAMES = frozenset({"ProsmotrProdukta", "ProsmotrProduktaVApiMethod"})
FAVORITE_ACTION_SYSTEM_NAMES = frozenset({
    "DobavlenieProduktaVSpisokVOperaciiDobavlenie",
    "DobavlenieProduktaVSpisokVOperaciiDobavlenieTovara",
})
PURCHASE_LINE_STATUSES = frozenset({"CP", "delivering", "F"})


@dataclass(frozen=True)
class InteractionRules:
    view_action_system_names: frozenset[str] = VIEW_ACTION_SYSTEM_NAMES
    favorite_action_system_names: frozenset[str] = FAVORITE_ACTION_SYSTEM_NAMES
    purchase_line_statuses: frozenset[str] = PURCHASE_LINE_STATUSES

    def __post_init__(self) -> None:
        for name in ("view_action_system_names", "favorite_action_system_names", "purchase_line_statuses"):
            values = getattr(self, name)
            if not isinstance(values, (set, frozenset)) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise InteractionConfigError("Правила должны быть наборами непустых строк")
            object.__setattr__(self, name, frozenset(values))
        if self.view_action_system_names & self.favorite_action_system_names:
            raise InteractionConfigError("VIEW и FAVORITE mappings не должны пересекаться")


DEFAULT_RULES = InteractionRules()


def classify_action(action: ActionRecord, rules: InteractionRules = DEFAULT_RULES) -> InteractionType | None:
    """Только точное совпадение. Классификация сама по себе не создаёт interactions."""
    if action.action_system_name in rules.view_action_system_names:
        return InteractionType.VIEW
    if action.action_system_name in rules.favorite_action_system_names:
        return InteractionType.FAVORITE
    return None


@dataclass(frozen=True)
class InteractionDiagnostics:
    actions_total: int = 0
    actions_view: int = 0
    actions_favorite: int = 0
    actions_unmapped: int = 0
    actions_malformed: int = 0
    view_interactions: int = 0
    favorite_interactions: int = 0
    order_lines_total: int = 0
    order_lines_purchase: int = 0
    order_lines_filtered_by_status: int = 0
    purchase_interactions: int = 0
    unmapped_action_system_names: Mapping[str, int] = field(default_factory=dict)
    malformed_action_system_names: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unmapped_action_system_names",
                           MappingProxyType(dict(self.unmapped_action_system_names)))
        object.__setattr__(self, "malformed_action_system_names",
                           MappingProxyType(dict(self.malformed_action_system_names)))

    @property
    def total_interactions(self) -> int:
        return self.view_interactions + self.favorite_interactions + self.purchase_interactions


class InteractionBuilder:
    """Счётчики накапливаются в одном экземпляре. Новый запуск — новый builder.

    Сохраняем каждое occurrence продукта, включая повторы внутри одного action.
    Полный поток не материализуется. Диагностика — снимок уже обработанной части.
    """

    def __init__(self, rules: InteractionRules = DEFAULT_RULES) -> None:
        self._rules = rules
        self._counts: Counter[str] = Counter()
        self._unmapped: Counter[str] = Counter()
        self._malformed: Counter[str] = Counter()

    @property
    def rules(self) -> InteractionRules:
        return self._rules

    @property
    def diagnostics(self) -> InteractionDiagnostics:
        return InteractionDiagnostics(
            **self._counts, unmapped_action_system_names=dict(sorted(self._unmapped.items())),
            malformed_action_system_names=dict(sorted(self._malformed.items())),
        )

    def from_action(self, action: ActionRecord) -> tuple[InteractionRecord, ...]:
        self._counts["actions_total"] += 1
        kind = classify_action(action, self.rules)
        if kind is None:
            self._counts["actions_unmapped"] += 1
            self._unmapped[action.action_system_name] += 1
            return ()
        counter_name = "view" if kind is InteractionType.VIEW else "favorite"
        self._counts["actions_" + counter_name] += 1
        if not action.products:
            self._counts["actions_malformed"] += 1
            self._malformed[action.action_system_name] += 1
            raise InteractionBuildError(f"{kind.value} action: требуется хотя бы один product")
        records = tuple(InteractionRecord(
            source_customer_id=action.source_customer_id,
            customer_id=action.customer_id,
            product=product,
            interaction_type=kind,
            event_datetime_utc=action.event_datetime_utc,
            source=InteractionSource.ACTION,
            source_event_id=action.action_id,
        ) for product in action.products)
        self._counts[counter_name + "_interactions"] += len(records)
        return records

    def from_order_line(self, line: OrderLineRecord) -> InteractionRecord | None:
        self._counts["order_lines_total"] += 1
        if line.line_status not in self.rules.purchase_line_statuses:
            self._counts["order_lines_filtered_by_status"] += 1
            return None
        record = InteractionRecord(
            source_customer_id=line.source_customer_id,
            customer_id=line.customer_id,
            product=line.product,
            interaction_type=InteractionType.PURCHASE,
            event_datetime_utc=line.order_datetime_utc,
            source=InteractionSource.ORDER,
            source_event_id=line.line_id,
            quantity=line.quantity,
        )
        self._counts["order_lines_purchase"] += 1
        self._counts["purchase_interactions"] += 1
        return record

    def iter_interactions(
        self, actions: Iterable[ActionRecord] = (), order_lines: Iterable[OrderLineRecord] = (),
    ) -> Iterator[InteractionRecord]:
        """Сначала actions, затем order lines; без сортировки, дедупликации и агрегации."""
        for action in actions:
            yield from self.from_action(action)
        for line in order_lines:
            record = self.from_order_line(line)
            if record is not None:
                yield record
