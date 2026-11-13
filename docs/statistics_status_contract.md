# STAT-STATUS-CONTRACT-01 — минимальный контракт UNIFIED_PURCHASE_V1

12.11.2026 · feature/hybrid-ranking · HEAD 2b56bb8. Нормативный контракт первого релиза. Статусы и источники решений приведены в [таблице контрактных решений](statistics_status_contract_decisions.md).

## 1. Цель и границы первого релиза

Единый поток наблюдаемых Orders покрывает покупки, товары, клиентов, отмены, возвратные состояния, номинальные суммы, RFM и когорты по дате оформления, справочник и общий отчёт GUI/HTML/XLSX. REFERENCE и INDEXED поставляют одни и те же входные факты. Первая версия использует зафиксированный набор доступных выгрузок. Событийный жизненный цикл, точное историческое состояние и финансовые операции вынесены за пределы этого релиза.

Правила опираются на [AUDIT](statistics_status_audit.md), [инвентарь](statistics_status_inventory.csv), [DESIGN](statistics_status_design.md), [реестр D01–D32](statistics_status_decisions.md), [DATA](statistics_status_data_evidence.md) и [матрицу возможностей](statistics_status_data_capabilities.csv). Договорённость о proxy от 12.11.2026 заменяет открытый D13 прежнего реестра для этой версии. DATA-вывод о неподтверждённом времени исполнения сохраняет силу. Новый сбор доказательств для реализации минимального контракта не требуется.

## 2. Утверждённые бизнес-правила

| Наблюдаемый статус | Класс | Вклад при пригодном quantity |
|---|---|---|
| F, sold | COMPLETED | Свидетельство покупки, пригодная цена строки может войти в денежную оценку |
| CP | PARTIAL | Свидетельство частичной покупки, номинал доступен по исходным полям, выкуп и его стоимость неизвестны |
| delivering и прочие стадии | IN_PROGRESS | Статус без свидетельства покупки |
| Коды отмены | CANCELLED | Отдельное состояние отмены |
| Возвратные коды | RETURNED | Отдельное возвратное состояние |
| Неизвестный/отсутствующий/невалидный код | UNKNOWN | Диагностика, покупочный статус не присваивается |

Единица поведенческой покупки — уникальный заказ. CP→F, повторы и несколько строк оставляют максимум одну покупку заказа. Установленное положительное свидетельство сохраняется при наличии наблюдения с возвратом. Return-only означает ноль подтверждённых покупок в полностью обработанном наблюдаемом наборе этого заказа и неизвестную предшествующую продажу. Номинал возвратного состояния и денежный refund имеют разные поля.

## 3. Дата оформления как proxy-time

**D13 — APPROVED, 12.11.2026.** ID временного правила **ORDER_CREATED_PURCHASE_PROXY_V1**. У наблюдаемой покупки датой для анализа служит валидное firstAction.dateTimeUtc заказа. Заказ оформлен 5 января и в доступной выгрузке есть пригодное F. Поведенческая аналитика относит его к 5 января. Точное время исполнения остаётся UNKNOWN.

Поля расчёта хранятся раздельно. order_created_at — UTC из firstAction. purchase.evidence_sources — ссылки на наблюдения в фиксированном срезе. calculated_at — время расчёта. executed_at — UNKNOWN с причиной EXECUTION_TIME_UNVERIFIED. Верхняя граница окна ограничивает order_created_at. Она не отбрасывает наблюдение по дате файла и не доказывает, что покупка была известна к этой дате. Публикуется ретроспективный результат с привязкой к provenance входного набора и явной подписью «Дата покупки — дата оформления заказа с наблюдаемым покупочным статусом».

Использовать действующий UTC parser, включая ISO без offset для полей Utc. Все пригодные даты одного заказа должны согласоваться. Разные даты дают CONFLICT, отсутствие/невалидность диагностируются независимо от наличия покупки. Если дата одной версии непригодна, а остальные согласованы, известная дата остаётся диагностическим значением, но датированные метрики этого заказа недоступны. Клиентские окна сохраняют существующие cutoff-функции и полуоткрытые границы (§11). Будущие записи за пределами cutoff не входят в временной срез. При непригодной дате сохраняется свидетельство заказа вне датированных итогов с отдельным счётчиком исключения. [Парсер UTC](../Application/mindbox/adapters/_common.py) · [Правило cutoff](../Application/statistics_rfm_validation.py)

## 4. Квалификация покупок

1. Нормализовать все доступные Orders в effective_entries. Сформировать приватный order key в одном identity space. Отсутствующий/невалидный ключ оставляет raw диагностику. Одинаковый строковый Mindbox ID из разных частей одной области означает тот же заказ. Retail ID не замещает основной ключ и участвует только в диагностике сопоставления.
2. На каждой валидно прочитанной строке отдельно классифицировать статус и quantity. qualified = статус F/sold/CP и положительное конечное целое quantity. Цена, валюта и разрешение товара в справочнике влияют на свои показатели. Они не отменяют квалификацию.
3. После согласования purchase.observed = KNOWN true, если существует квалифицированная строка с пригодным order key. KNOWN false означает отсутствие пригодных свидетельств в полностью измеренных наблюдениях этого заказа. При структурном пропуске состава и без положительного свидетельства — INSUFFICIENT_COVERAGE. Unknown/invalid status и qty остаются в диагностиках, а явно прочитанная непригодная строка сама не создаёт purchase.
4. В счётчик покупочных заказов без привязки к клиентам заказ может войти при надёжном order key даже с неразрешённым владельцем. В клиентские метрики заказ входит при надёжном canonical customer key. При конфликте order key группа исключается из бизнес-агрегатов. Scope и число исключённых наблюдений публикуются рядом с известными агрегатами. При конфликте владельца сохраняются неперсональные факты заказа и причины исключения из клиентского среза.
5. Фильтр применяется к результату согласования. Для продуктового среза доказательством служит пригодное положительное наблюдение этой же строки и товара. Запрещено переносить свидетельство одной версии строки на другой product key. Правила областей см. в §12.

## 5. Согласование снимков

**ORDER_OBSERVATION_RECONCILIATION_V1** — отдельный reconciler статистики. Группировать по scoped order key **до** отсечения по дате или товару. В первой версии источники order evidence — эффективные Orders. Связанный order в Actions не включается из-за непроверенной семантики истории. Физический порядок частей и время выгрузки сохраняют значение provenance. Авторитетную последовательность состояний они не задают.

| Шаг | Нормативное правило |
|---|---|
| Повторы | Считать raw occurrences. Одинаковые полные нормализованные JSON-снимки по действующему semantic fingerprint образуют одно наблюдение для расчёта. Хранить multiplicity и provenance. Сравнение полных снимков и сравнение бизнес-проекции дают отдельные счётчики |
| Состав | Стабильный line key действует внутри заказа. Строки с валидным уникальным ID сортируются по ключу. Для отсутствующего ID допустим локальный ordinal внутри одной версии. Бизнес-эквивалентность двух проекций с пропущенным ID допускается при полном поэлементном равенстве включая локальный ordinal, без вывода о связи между различающимися версиями |
| Бизнес-равенство | Сравнить canonical owner, created_at, полноту списка, множество/локальный порядок ключей строк, код и качество статуса, quantity, price, product/currency и используемые поля аналитики канала/доставки/оплаты. Значения Decimal сравниваются численно, object key order и unused metadata не влияют. Пропуск и INVALID отличимы. Равные версии считаются один раз |
| Согласованная часть | Для различающихся версий определить строки со стабильным ключом, встречающимся ровно один раз в каждой пригодной версии. Согласовать каждое поле по совпадению. Полное состояние известно при полной структуре, согласованном составе и пригодных группах статуса всех строк. При неизвестном составе доступен только доказанный поднабор |
| Историческое свидетельство | OR по квалифицированным наблюдениям заказа. При нескольких версиях точное число покупочных позиций доступно только если состав строк установлен по уникальным стабильным ключам во всех версиях и все списки полны. Тогда одна логическая строка учитывается максимум один раз, включая F и Return по одному ID. При различающемся составе считаются только положительные строки с однозначным общим ключом. Остальные дают причины неполноты и не суммируются по версиям |
| Независимые конфликты | Разные canonical owners дают OWNER_CONFLICT, отсутствующий/невалидный owner при согласованных известных значениях всё равно ограничивает персональную атрибуцию. Разные составы или повтор line ID дают LINE_SET_CONFLICT/LINE_ID_DUPLICATE. Разные точные коды дают STATUS_CONFLICT. Для quantity, цены, product key, валюты, даты и канала отдельные поля и причины |
| Победитель | Ни первая, ни последняя версия не назначается актуальной при отсутствии подтверждённого порядка. При конфликте статуса доступен purchase OR и неизвестно согласованное состояние затронутой строки. При конфликте только цены статусные счётчики сохраняют точность. Состав заказа не склеивается из разных корзин |

Однозначный общий line key означает присутствие в каждой версии с уникальным ID, а не только совпадение номера/товара. Повтор line ID внутри версии не раздувает позиции и денежные суммы. Положительный код остаётся свидетельством уровня заказа. Если неполный/невалидный список наблюдений не даёт сопоставить часть строк, сохраняется результат для общих однозначных строк, остальные остаются в raw диагностике. Состояния и агрегаты инвариантны к перестановке источников. Глобальный OrderSnapshots остаётся прежней зависимостью Legacy/ML. [Текущая реализация](../Application/mindbox/order_dedup.py)

## 6. Количества

| quantity | Качество nominal_units | Квалификация |
|---|---|---|
| Числовой конечный Decimal/int/float, положительное целое | KNOWN Decimal | Пригодно |
| Числовой ноль | KNOWN Decimal(0), ZERO_QUANTITY | Непригодно |
| Отрицательное, положительное дробное | INVALID с NEGATIVE_QUANTITY или FRACTIONAL_QUANTITY | Непригодно, значение диагностируется в нормализованном входе |
| Нет/null | UNKNOWN с MISSING_QUANTITY | Непригодно |
| Bool, строка, объект, NaN, Infinity | INVALID с INVALID_QUANTITY | Непригодно |

Декодировать JSON числа в Decimal без промежуточного float. При входе Python float использовать конечное Decimal(str(value)). Статус учитывается в своей стадии даже при непригодном quantity. Никакого округления, clipping или инверсии знака. Counts позиций и заказов — int. При совпавших пригодных quantities одной квалифицированной позиции nominal units известны. Разные значения между пригодными версиями дают CONFLICT. Исторический subtotal единиц включает лишь однозначно согласованные позиции и явно отмечается неполным. Поле fulfilled_units в первой версии UNKNOWN с FULFILLMENT_UNVERIFIED, в том числе для F/sold без отдельного подтверждения учётом. CP никогда не подставляет номинал. Факт единицы заказа отделён от этих оценок.

## 7. Денежные показатели

Цена исходной строки — конечный числовой Decimal ≥ 0. Отсутствие/null даёт UNKNOWN, отрицательная и нечисловая цена — INVALID. Валюта разрешается по действующему CURRENCY_BY_NAMESPACE из идентификаторов товара. offline1C даёт RUB, kanzlerKz даёт KZT. Несколько противоречивых поддерживаемых пространств у строки или явно противоположная проверяемая валюта дают CONFLICT. Нет установленного namespace — UNKNOWN. Исходная priceOfLine учитывается один раз независимо от количества. [Правила валюты и Decimal](../Application/order_statistics.py)

| Поле | Правило первой версии |
|---|---|
| source_nominal | Цена наблюдаемой строки и её валюта. Согласованный итог строится по согласованным строкам. Пропуск валюты оставляет цену доступной на уровне входа и недоступной в валютном итоге |
| known_purchase_subtotal | Сумма известных значений F/sold по однозначным покупочным позициям, один вклад на line key. Допускается при неполной оценке заказа. Включать только при согласованной цене и валюте во всех пригодных F/sold наблюдениях этой линии и отсутствии квалифицированного CP для неё. Противоречие цены/валюты убирает вклад этой строки, оставляя соседние известные суммы |
| recognized_purchase | Полная наблюдаемая стоимость F/sold. Известна, если состав и покупочная квалификация позиций разрешены, каждая квалифицированная строка имеет согласованные quantity, цену и валюту и CP с неизвестной суммой отсутствует. Смена F↔sold при неизменных quantity/цене/валюте не мешает оценке. F↔Return, CP↔F или неопределённая квалификация не дают полного итога, даже если отдельная положительная цена известна |
| partial_nominal | Цена согласованной CP-строки при пригодной цене и валюте. Выкуп и признанная стоимость остаются неизвестными |
| cancelled_nominal | Цена согласованной отменённой строки. Это номинал состояния |
| returned_nominal | Цена согласованной строки с RETURNED. Это номинал состояния |
| confirmed_refund, net_after_refunds | UNKNOWN, LEDGER_UNAVAILABLE. Платёж, цена и возвратный статус не подтверждают денежную операцию |

Если совместимые F/sold значения подтверждены, конфликт количества при двух положительных целых значениях не меняет сам вклад priceOfLine в известный подытог. Неполный состав или неопределённая квалификация не разрешают полный признанный итог. Недопустимые quantity у F/sold/CP сохраняют причину неполноты для признанной стоимости, если уже есть другие принятые позиции заказа. При ясной известной пустой совокупности допустим полный 0. Если есть неоценённые строки и ни одного оценённого вклада, сумма неизвестна, а счётчик оценённых равен 0.

Суммы публикуются раздельно по RUB и KZT. Смешанный заказ может давать полные валютные компоненты, если все позиции оценены, но не одновалютную сумму или полностью оценённый заказ для действующей одновалютной базы RFM. Неизвестная валюта делает полный валютный итог недоступным для затронутой группы. Известные подытоги остаются видимыми. Доли неоценённых считать на подходящем scope. Для средней или медианы чека знаменатель — полностью оценённые сопоставимые заказы валюты. Пропущенные цены не входят как 0. Счётчики unvalued positions/orders/buyers разделяются. Округление выполняет существующий форматтер при представлении. Оценка по статусу продажи не измеряет денежное поступление. CP allocation, банковские суммы и net не влияют на стоимость F/sold под видом корректировки.

## 8. Отмены и возвратные состояния

По согласованным наблюдаемым строкам публиковать позиции и уникальные заказы с отменой и возвратным статусом, полностью отменённые заказы, смешанные с отменой и смешанные с возвратом, уникальных canonical клиентов с каждым состоянием и известные номиналы. Заказ с двумя одинаковыми статусами учитывается в заказном счётчике этого кода один раз. Положительный вклад по известной строке допустим и для заказа с неполным составом, полное значение счётчика тогда отмечается отдельно.

Полностью отменённый означает непустой полный согласованный состав и все строки CANCELLED. RETURNED_ONLY по тому же правилу описывает все наблюдаемые строки. Смешанный — минимум две согласованные разные основные бизнес-группы из §2. Разные подгруппы исполнения сами по себе образуют IN_PROGRESS_ONLY. При частичной видимости можно достоверно показать наличие двух известных групп, но полная классификация заказа остаётся недоступной. Приоритет типа отображения CONFLICT → UNAVAILABLE → EMPTY → однотипный или MIXED. CP имеет отдельный тип PARTIAL_ONLY. В группах отмены и возврата показывать качество и долю заказов с неизвестным или конфликтующим состоянием от количества пригодных уникальных заказов в том же scope. Подтверждённые ранее покупки остаются в поведенческой истории. Отмена, возвратный снимок и повторное его появление не создают refund event.

## 9. Справочник статусов

**MINDBOX_STATUS_CATALOG_V1** содержит все 43 строки из инвентаря. Для 42 строк использовать точные name/code/category из пользовательского справочника. Mindbox-Tech-Cancellation относится к CANCELLED с официальными name/category = null и отдельной UI-подписью «Техническая отмена». Бизнес-группы берутся из столбца current_status_group, COMPLETED/PARTIALLY_COMPLETED образуют F/sold/CP, стадии *_IN_PROGRESS отображаются как IN_PROGRESS с сохранением подгруппы. Источник и степень подтверждения сохраняются отдельно. [43 строки каталога](statistics_status_inventory.csv)

Сопоставление externalId точно, включая регистр и Unicode. Нет/null/пустая строка → MISSING_CODE. Объект, число, bool, пробельная строка, краевые пробелы и управляющие символы → INVALID_CODE. Иной пригодный точный код → UNKNOWN_CODE со своим агрегированным безопасным отображением. Невалидные raw значения не публиковать. Все известные коды показываются всегда с названием, externalId, категорией, бизнес-группой, числом позиций и уникальных заказов.

| Стадия | Единица и область |
|---|---|
| raw occurrences | Каждая строка исходного выбранного источника до dedup, включая плохой quantity. Технический scope источников, с отдельной маркировкой без продуктового фильтра |
| resolved current positions | Основное распределение. Одна согласованная строка на order/line в объявленном order или product scope, даже при непригодном quantity |
| orders_with_status | Уникальный order key с хотя бы одной согласованной строкой данного кода в том же scope. Несколько кодов могут относиться к одному заказу |
| historical purchase positions | Однозначные положительные позиции по §5, отдельная стадия и область. Никакого сложения с current |

При неизвестном составе, коде или попадании строки в фильтр показать известный счётчик и причину неполноты. Измеренный ноль возможен при полном измерении данной области и стадии. Raw счётчик не подменяет основное распределение после включения фильтра. Известный ноль наблюдений не описывает частоту во всём tenant.

## 10. Доменные структуры

Образцы ниже задают поля и типы будущего Python-контракта. Структуры immutable. Приватные ключи принадлежат одной закреплённой области источников. Обозначение str для ключа включает непрозрачное внутреннее представление SID в читателе INDEXED после унификации типа. Публичные поля перечислены в §13.

```python
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Generic, TypeVar

T = TypeVar("T")
Interval = tuple[datetime, datetime]

class EvidenceQuality(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"
    INVALID = "INVALID"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"

class OrderKind(str, Enum):
    COMPLETED_ONLY = "COMPLETED_ONLY"
    PARTIAL_ONLY = "PARTIAL_ONLY"
    CANCELLED_ONLY = "CANCELLED_ONLY"
    RETURNED_ONLY = "RETURNED_ONLY"
    IN_PROGRESS_ONLY = "IN_PROGRESS_ONLY"
    MIXED = "MIXED"
    EMPTY = "EMPTY"
    UNAVAILABLE = "UNAVAILABLE"
    CONFLICT = "CONFLICT"

@dataclass(frozen=True, slots=True)
class Measured(Generic[T]):
    value: T | None
    quality: EvidenceQuality
    reasons: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class PurchasePolicy:
    policy_id: str = "UNIFIED_PURCHASE_V1"
    status_catalog_id: str = "MINDBOX_STATUS_CATALOG_V1"
    time_basis_id: str = "ORDER_CREATED_PURCHASE_PROXY_V1"
    snapshot_policy_id: str = "ORDER_OBSERVATION_RECONCILIATION_V1"
    money_policy_id: str = "OBSERVED_LINE_VALUE_V1"

@dataclass(frozen=True, slots=True, repr=False)
class OrderLineState:
    stable_line_key: Measured[str]
    ordinal: int | None
    status_code: Measured[str]
    status_group: str
    input_quantity: Decimal | None
    nominal_units: Measured[Decimal]
    fulfilled_units: Measured[Decimal]
    source_price: Measured[Decimal]
    currency: Measured[str]
    product_key: Measured[str]
    product_token: str | None

@dataclass(frozen=True, slots=True, repr=False)
class OrderObservation:
    source_ref: tuple[int, int, int]
    order_key: Measured[str]
    customer_key: Measured[str]
    order_created_at: Measured[datetime]
    raw_fingerprint: bytes | None
    lines_quality: EvidenceQuality
    lines: tuple[OrderLineState, ...]
    channel: Measured[tuple[str, str | None]]
    order_attributes: tuple[tuple[str, Measured[str | Decimal | tuple[str, ...]]], ...]

@dataclass(frozen=True, slots=True, repr=False)
class PurchaseEvidence:
    observed: Measured[bool]
    evidence_sources: tuple[tuple[int, int, int], ...]
    known_positions: int
    positions: Measured[int]
    known_nominal_units: Decimal
    nominal_units: Measured[Decimal]
    fulfilled_units: Measured[Decimal]
    executed_at: Measured[datetime]

@dataclass(frozen=True, slots=True)
class MonetaryEvidence:
    currency: str
    source_nominal: Measured[Decimal]
    known_purchase_subtotal: Measured[Decimal]
    recognized_purchase: Measured[Decimal]
    partial_nominal: Measured[Decimal]
    cancelled_nominal: Measured[Decimal]
    returned_nominal: Measured[Decimal]
    confirmed_refund: Measured[Decimal]
    net_after_refunds: Measured[Decimal]
    known_nominals: tuple[tuple[str, Decimal], ...]
    valued_purchase_positions: int
    unvalued_purchase_positions: int

@dataclass(frozen=True, slots=True, repr=False)
class OrderBusinessFact:
    order_key: str
    customer_key: Measured[str]
    order_sort_rank: int
    order_created_at: Measured[datetime]
    current_kind: OrderKind
    state_quality: EvidenceQuality
    state_groups: frozenset[str]
    resolved_lines: tuple[OrderLineState, ...]
    purchase: PurchaseEvidence
    money: tuple[MonetaryEvidence, MonetaryEvidence]
    order_currency: Measured[str]
    channel: Measured[tuple[str, str | None]]
    order_attributes: tuple[tuple[str, Measured[str | Decimal | tuple[str, ...]]], ...]
    diagnostics: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class AnalyticsContext:
    context_id: str
    policy: PurchasePolicy
    identity_space: str
    calculated_at: datetime
    as_of_date: date | None
    period: Interval | None
    product_filter_key: str | None
    orders_intervals: tuple[Interval, ...]
    actions_intervals: tuple[Interval, ...]
    coverage_basis: str
    created_history_verified: bool
    status_history_verified: bool
    provenance_digest: str
```

| Тип | Источник, инварианты и сериализация |
|---|---|
| PurchasePolicy | Один набор указанных неизменяемых ID. В DTO объект policy с пятью строками. Таблица статусов версионируется с ним, бизнес-версия и SQLite schema независимы |
| EvidenceQuality и Measured | KNOWN требует value, включая false и 0. Все остальные значения требуют null и минимум одну причину. Reasons — уникальные отсортированные коды. KNOWN допускает информационную причину ZERO_QUANTITY. При нескольких проблемах одного поля приоритет CONFLICT → INVALID → INSUFFICIENT_COVERAGE → UNKNOWN. Остальные причины сохраняются. Сериализация как {value,quality,reasons}, enum строкой |
| OrderLineState | На входе exact status.ids.externalId, lines[].id, ordinal, quantity, priceOfLine и product.ids. В согласованном факте ordinal заполнен только для локальной позиции одной эквивалентной проекции. status_group берётся из §9, UNKNOWN при неизвестном или конфликтном статусе. При пригодном незнакомом коде status_code KNOWN, группа UNKNOWN. Stable key обязателен для связи между различающимися версиями. Валюто-товарное разрешение использует закреплённый каталог. input_quantity хранит конечное исходное число, включая ноль, отрицательное и дробное, для сравнения и round-trip. product_token — внутреннее точное нормализованное пространство и ключ товара для сравнения неразрешённых позиций. fulfilled_units UNKNOWN. Строка в DTO доступна лишь в разрешённой детали с локальным ключом и без source IDs |
| OrderObservation | Один Orders occurrence. source_ref — индексы manifest entry, part и ordinal записи. Входные owner и order ID валидируются, owner канонизируется общим CustomerIdResolver. Отсутствие массива lines отдельно от известного пустого. Channel из firstAction.channel.ids.externalId и name. order_attributes содержит delivery_cost из deliveryCost, delivery_type и ordering_method из customFields.deliveryType и customFields.orderingMethod, payment_types из payments[].type. Сохраняется разница между известным пустым списком и непригодным. При сравнении типов оплаты используется отсортированный tuple с кратностью. Непригодный fingerprint при повреждённой структуре не разрешает exact dedup. Приватный формат projection, в публичный DTO целиком не сериализуется |
| PurchaseEvidence | Формируется по §4–6. observed независим от стоимости и текущего статуса. evidence_sources — уникальные приватные ссылки на положительные наблюдения. known_positions и known_nominal_units обозначают сумму доказанной части, positions и nominal_units полную оценку. executed_at UNKNOWN. В aggregate DTO только счётчики, качество и причины. В деталях без evidence_sources |
| MonetaryEvidence | Ровно два экземпляра на факт в порядке RUB, KZT. Отдельная строка диагностики для неразрешённой валюты. known_nominals содержит известные части source_nominal/partial_nominal/cancelled_nominal/returned_nominal, ключи сортируются. Счётчики unvalued по валюте считаются только для атрибутируемых позиций, неизвестные по валюте отдельно. Полный recognized по валюте не подменяется subtotal. Денежные value — десятичные строки без экспоненты. confirmed_refund и net null с причиной LEDGER_UNAVAILABLE |
| OrderBusinessFact | Один на валидный scoped order key. order_sort_rank положительный, соответствует лексическому порядку нормализованных исходных order key внутри identity space. resolved_lines — доказанная часть, поле отдельного атрибута может быть недоступно. order_currency KNOWN для единой определимой валюты покупки даже при неизвестной цене CP. Для смешанной покупки CONFLICT с MIXED_ORDER_CURRENCY, для неизвестной UNKNOWN. Категория состояния относится к наблюдению. Чувствительные ключи и raw fingerprint исключены из публичного DTO |
| AnalyticsContext | Полуоткрытые UTC intervals, period=None при недоступном периоде. identity_space связывает приватные ключи внутри расчёта. coverage_basis = DECLARED_ORDERS_PROXY_V1, created_history_verified=false и status_history_verified=false на нынешнем источнике. declared completeness для формул §11 хранится отдельно. В DTO context_id, даты, фильтр, coverage, capabilities, provenance. Приватное пространство ключей и пути не сериализуются |

Уточнения реализации. Reason registry включает ORDER_ID_INVALID, OWNER_MISSING, OWNER_CONFLICT, CREATED_TIME_MISSING, CREATED_TIME_INVALID, CREATED_TIME_CONFLICT, LINE_SET_CONFLICT, LINE_ID_DUPLICATE, LINE_ID_MISSING, LINES_INCOMPLETE, STATUS_CONFLICT, UNKNOWN_CODE, MISSING_CODE, INVALID_CODE, причины §6, QUANTITY_CONFLICT, PRICE_MISSING, PRICE_INVALID, PRICE_CONFLICT, CURRENCY_UNKNOWN, CURRENCY_CONFLICT, MIXED_ORDER_CURRENCY, PRODUCT_CONFLICT, CHANNEL_CONFLICT, CP_ALLOCATION_UNKNOWN, FULFILLMENT_UNVERIFIED, EXECUTION_TIME_UNVERIFIED, LEDGER_UNAVAILABLE и COVERAGE_GAP. Добавление кода требует записи в валидаторе и общей карте подписей. DTO 11 использует эти коды без raw строк ошибок. Decimal должен оставаться точным, 0 и -0 в JSON нормализуются в "0". UTC datetime сериализуется ISO 8601 с +00:00, даты ISO YYYY-MM-DD. Сортировка списков справочника, причин, валют и ключей детерминирована.

Алгоритм scope сохраняет нормализованные наблюдения заказа до проекции. При необходимости продуктового среза reconcile_order сначала формирует общий факт, затем применяет product predicate к тем же группам позиций и положительным наблюдениям. Категории/каналы допускают unknown при конфликте. Промежуточный массив удерживается в пределах группы с существующей проверкой ресурса. RefundEvidence остаётся будущей возможностью со стабильным ключом операции, связью с заказом, суммой, валютой и часами. Production-типа ledger в первой версии нет.

## 11. RFM и когорты

Для одного scope все входы происходят из одного OrderBusinessFact. Клиентские агрегаторы получают один экземпляр на пригодный order key и canonical customer key. Время — order_created_at по §3. Дата расчёта фиксирована для всех читателей. Окна и cutoff сохраняют rfm_cutoff/cohort_cutoff. Явный end_date задаёт as_of_date. При его отсутствии берётся последний покрытый день до UTC-даты расчёта включительно. Общий конец — полночь следующего дня UTC. rfm_windows сдвигает этот конец на −12/−24/−48 месяцев с существующим правилом конца месяца. PREVIOUS_12M — [start24, start12). Окна полуоткрытые.

**RFM.** База покупателей — 48 месяцев пригодных дат в действующей Orders eligibility. R использует последнюю наблюдаемую покупку этой базы до as_of_date, для валютной базы последнюю покупку той валюты. F — уникальные заказы выбранного окна, а для валютного вида заказы с однозначной валютой. CP входит в общий F и в F своей однозначной валюты. M — сумма полных recognized_purchase для заказов по дате оформления в окне, только при полном покрытии окна по действующей формуле и полной оценке всех покупочных заказов клиента в этом окне. Как и сейчас, любой неоценённый или смешанный заказ в окне закрывает полную M обеих валют. При этом известные подытоги публикуются и для неоценённых заказов. Счётчик valued orders учитывает только полную одновалютную оценку. Нулевой M допустим при достоверном нуле. Применяются прежние reliable_since, nearest-rank quintiles, правила ties, rfm_score и rfm_segment. Missing M → UNSCORABLE. Клиент с покупкой без определимой валюты остаётся в общей поведенческой базе и отдельном unassigned_currency счётчике. [Ядро RFM](../Application/statistics_rfm_core.py)

**Когорты.** По всей доступной истории Orders до cutoff, с текущим gate по order_scope_period, найти первые два различных заказа, упорядоченные по (order_created_at, order_sort_rank). Период пользовательского отбора ограничивает месяцы когорт после определения первой покупки, товарный фильтр здесь не ограничивает покупки. Месяц первой — UTC месяц оформления. Разность дат, включая 0 при одинаковом дне, задаёт интервал. Для H30/60/90/180/365 событие — вторая покупка через ≤ H календарных дней. Интервал eligibility — [полночь first_day, полночь first_day+H+1). M0 — размер когорты. Для M1–M12 учитывается наличие заказа в календарном месяце, interval eligibility — границы этого месяца UTC. Censored при end > cutoff. Incomplete при пробеле покрытия в наблюдаемом участке [start,min(end,cutoff)). eligible = not censored and not incomplete. Rate = qualified/eligible, при 0 eligible значение недоступно. Наблюдаемые повторы вне eligible выводятся отдельно. Последующий возврат второй покупки не убирает F=2, second и активный месяц. Сам Return не добавляет активность. [Ядро когорт](../Application/statistics_cohorts_core.py)

Сохранить денежные таблицы когорт и горизонтов на новой единице заказа, включая known subtotals и unvalued. Цены привязаны к дате оформления заказа. Полнота суммы и eligibility независимы. Каналы первой/второй покупки из согласованного firstAction.channel. Категории из достоверных квалифицированных строк этих заказов по текущему прямому товарному справочнику. При неполной позиции, неразрешённом товаре или конфликте атрибуции сохранять известные категории и отдельную неизвестную часть. Переходы каналов/категорий используют текущие знаменатели и unknown buckets. Факт пары двух известных категорий не доказывает полноту категорий корзины. [Деньги когорт](../Application/statistics_cohorts_money.py) · [Каналы](../Application/statistics_cohorts_channels.py) · [Категории](../Application/statistics_cohorts_categories.py)

**Покрытие proxy.** Для воспроизведения eligibility используется действующий расчёт объявленных effective Orders intervals. FULL/PARTIAL/NONE относится к календарному перекрытию объявленного scope. Покрытие оформления и покрытие изменений публикуются отдельными свойствами. DATA не подтвердил полноту по дате оформления или полный журнал статусов. В provenance сохраняются часы MANUAL=UNKNOWN, API Orders=ORDER_CHANGED по документированному основанию и флаги непроверенной полноты. На этой базе RFM и когорты вычисляются как согласованная proxy-аналитика. Известные даты за пределами order_scope_period и непригодные даты считаются отдельными исключениями. При дыре покрытия выводятся действующие частичные оценки и причины неполноты. Не переименовывать «первая наблюдаемая» в первую за всю историю. [DATA05](statistics_status_data_evidence.md)

## 12. Аналитические контексты

| context_id | Источники и scope | Единицы и потребители |
|---|---|---|
| raw_orders | Эффективные Orders в закреплённом наборе до фильтрации и dedup | Raw orders/lines/statuses, дубликаты, причины повреждений |
| order_product_slice | Действующий общий интервал Actions/Orders, пользовательские даты и предикат товаров. Группировка до фильтра | Уникальные order/line/product, состояния выбранных строк, статусный каталог, номиналы. Для отмены всего заказа проверяется полный состав, в продуктовом срезе отдельная подпись |
| item_interactions | Тот же продуктовый срез. Существующие VIEW/FAVORITE из Actions и PURCHASE из исторических квалифицированных позиций | Товарные рейтинги, пересечения пользователей, покупки по товарам. Заказный счётчик отделён от числа PURCHASE-позиций |
| customer_population | Действующий customer scope, включая доступные профили, действия и заказы без разрешённого товара | Идентифицированная клиентская база, доля наблюдаемых покупателей и неизвестная атрибуция |
| buyer_period | Свой интервал Orders с customer date filter, все товары | Покупатели, повторы, деньги, корзины и пары прямых категорий |
| rfm_48m | Полная заданная 48M база до cutoff, окна §11, все товары | R/F/M, сегменты, валютные и quality счётчики |
| cohorts_history | Доступная Orders история до cutoff, all-products. Фильтр месяца когорты после first/second | Повторность, M0–M12, каналы и категории, деньги и переходы |
| detail | Идентичный объявленный scope соответствующего результата | Связанные строки заказа/позиции/клиента с независимой диагностикой |

Ключи и фильтры фиксируются в AnalyticsContext. Одни и те же значения не объявляются равными при разных scope. Если наблюдение покупки содержит пригодный товар, который конфликтует между версиями одной линии, допускается заказное свидетельство в целевом товарном срезе от подходящей версии. Позицию нельзя суммировать по двум товарам как полную атрибуцию. При конфликте товарной идентичности публикуются диагностический счётчик и известные позиции с однозначной атрибуцией. Диагностика unknown/invalid сохраняется и при применении фильтра. Случай, когда фильтр невозможно проверить из-за неполного состава, оставляет неполную оценку.

## 13. DTO и старые результаты

Проверенные текущие значения в коде — statistics_cache schema 10, statistics_slices envelope 1, detail format 1 и INDEXED schema 3. Утверждены **DTO 11 и основной cache 11** с отдельным артефактом UNIFIED, **slice envelope 2** для нового пространства срезов и **detail format 2** для нового содержимого ZIP. Slice 1 и detail 1 остаются читаемыми в исходных путях. [Основной кеш](../Application/statistics_cache.py) · [Кеш срезов](../Application/statistics_slices.py) · [Детальный формат](../Application/statistics_detail_contract.py)

Форма сериализации — {"schema_version":11,"result":{...}}. Поля result обязательны. Ниже имена и типы границ DTO.

| Поле result | Тип и содержание |
|---|---|
| policy | PurchasePolicy из §10 с точными пятью ID |
| calculated_at | UTC ISO datetime |
| contexts | map context_id → объект из §10 с provenance и declared coverage. Каждый раздел ссылается на существующий context_id |
| capabilities | map str→bool. order_created_proxy=true, execution_time=false, historical_as_of=false, cp_allocation=false, split_lineage=false, refund_ledger=false. Версия проекции отражается отдельно |
| source | {reader,projection_schema,provenance_digest}, reader REFERENCE/INDEXED, projection_schema null/4 соответственно |
| orders | {context_id,unique_orders,positions,purchase_orders,purchase_positions,buyers,state_distribution,cancellations,return_states,mixed_states}. Счётчики — Count, state_distribution map OrderKind→Count. Cancellations/return_states содержат positions, orders, only_orders, mixed_orders, customers; mixed_states содержит orders и unresolved_state_share |
| money | {context_id,currencies,unassigned_currency_orders}. currencies содержит RUB и KZT. Каждая строка включает поля MonetaryEvidence, totals/known parts и valued_orders/unvalued_orders/unvalued_buyers |
| status_catalog | {context_id,catalog_id,rows}. row = {external_id,name,mindbox_category,business_group,metadata_confirmation,positions,orders}. Count для positions/orders, missing/invalid отдельные строки с external_id=null. Unknown exact code отдельная строка |
| actions, products, customers | {context_id,data,quality}. В data сохраняются действующие наборы таблиц соответствующих разделов, row definitions и вычисляемые поля VIEW/FAVORITE. PURCHASE меняет источник на §12. Поля, обозначавшие обе старые политики, заменяются одним рядом UNIFIED и явными quality/known subtotal |
| rfm | {context_id,as_of_date,base_buyers,windows,comparison,quality}. windows keyed четырьмя ID, каждый содержит coverage, behavior, currencies, quality. Поведенческие, валютные, score и segment поля сохраняют формы v10 ветки одной политики, known_amount включает подытоги §7 |
| cohorts | {context_id,as_of_date,cohort_bounds,coverage,months,summary,money,channels,categories,quality}. Строки месяцев и сводки H/Mn сохраняют v10 формы одной политики и включают качество денежной/товарной атрибуции |
| diagnostics | {context_id,counters,reason_counts}. Counters raw_orders/raw_lines/exact_duplicates/business_equivalent/snapshot_conflicts/undated_orders/orders_outside_coverage/unassigned_buyers. Reason counts map reason→int. Обязательна маркировка scope и стадии |

Примеры ячеек DTO. Измеренный ноль заказов — {"known":0,"value":0,"quality":"KNOWN","reasons":[]}. При двух известных заказах и неполном измерении — {"known":2,"value":null,"quality":"INSUFFICIENT_COVERAGE","reasons":["LINES_INCOMPLETE"]}. Для CP полная сумма — {"value":null,"quality":"UNKNOWN","reasons":["CP_ALLOCATION_UNKNOWN"]}. В известном F/sold-подытоге этого же заказа значение 100 кодируется строкой "100".

Count = {known:int,value:int/null,quality:str,reasons:list[str]}. known — доказанная нижняя часть. При KNOWN value=known. Известный ноль требует полной применимой области. Для сумм и долей используется Measured с отдельным известным подытогом там, где сумма неполная. Формы действующих таблиц, окон, сегментов и cohort cells задаются текущими валидаторами. В новом валидаторе 11 остаются инварианты знаменателей, дат, распределений и валют, новые независимые поля проверяются по §10. Старый валидатор не используется для reinterpretation. [Валидатор RFM](../Application/statistics_rfm_validation.py) · [Валидатор когорт](../Application/statistics_cohorts_validation.py)

Диспетчер читает версию до разбора result. Schema 10 проходит прежний валидатор и renderer с Legacy/STRICT/EXTENDED и исходными именами. Служебный viewer tag LEGACY_MULTIPOLICY_V1 допустим вне сохранённого result. Открытие старого отчёта не запускает расчёт или сохранение. Schema 11 проверяет все ID, context references, Decimal/quality и инварианты. Неизвестная версия даёт ошибку совместимости. Новые файлы основного кеша и срезов используют отдельный путь/ключ policy, schema и fingerprint. Fingerprint включает source manifest, каталог товаров, merges, интервалы/фильтр, cutoff, все policy IDs, версии projection/cache и код нового расчёта. Прежние артефакты и ключи в них сохраняются. Для вычислительного reuse старого кеша и v3 остаётся штатная проверка совместимости с версией кода и источников. Cache miss новой политики не является основанием для чтения schema 10 под новым заголовком.

Detail ZIP format 2 сохраняет clients.csv, orders.csv, order_lines.csv, manifest.json, действующий CSV encoding и правила защиты ячеек. Manifest содержит format_version=2, policy, context_id, generated_at и provenance_digest. В строках сохранить существующие разрешения на поля, связи внутри файла через export-local keys. Двойные policy columns заменяются едиными observed_purchase, purchase_quality и величинами §7 с quality/reasons. Для order_lines публиковать current и historical flags отдельно, номиналы и качество. Недоступный value — пустая ячейка с quality, известный ноль — "0". Сохраняются валидация связности, атомарная публикация и отсутствие raw ID в общих diagnostics. Формат 1 читается своим контрактом. Новых обещаний по refund не требуется.

## 14. INDEXED и REFERENCE

**A. Schema 3 недостаточна.** Хранятся вхождения orders, но порядок строки INDEXED выдаёт как синтетический line_id. Для nonintegral и ошибочных строк часть status, namespace, price, product теряется. Нельзя сопоставить состав и статусы версий или сохранить одинаковую диагностику непригодных полей. Дата оформления и order SID в схеме уже есть. Fingerprint не восстанавливает утраченные значения. [DDL и Projector](../Application/statistics_facts.py) · [IndexedInput](../Application/statistics_query.py) · [DATA15](statistics_status_data_evidence.md)

**B. Утверждена schema 4, STATISTICS_OBSERVATIONS_V1.** Сохранять существующие action/product/profile возможности, необходимые для VIEW/FAVORITE и клиентской базы. Проекция Orders хранит каждое наблюдение, до принятия quantity, статуса или winner. Таблица ниже задаёт логические колонки. SQLite хранит enum/reasons и полезную metadata в ограниченном каноническом JSON или скалярных столбцах, Decimal как TEXT, даты INTEGER UTC microseconds.

| Таблица | Минимальный набор |
|---|---|
| metadata и manifest | schema=4, projection_id, generation, capabilities, source entries/parts/clock coverage, content provenance, catalog/merges digest и версия нормализации |
| order_observations | occurrence_id PK, source_entry/part/source_ordinal, order_sid nullable + quality, customer_sid nullable + quality, cohort_order_rank, created_us nullable + quality, raw_fingerprint nullable, lines_quality, channel key/name + quality и нормализованные используемые order_attributes |
| order_line_observations | occurrence_id FK, ordinal, PK(occurrence_id,ordinal), scoped line_sid nullable + ID quality, status_code nullable + code quality, quantity_text nullable + type/reason, price_text nullable + quality/reason, namespace mask и выбранное пространство с currency quality, normalized product token nullable, resolved product key nullable + resolution quality |
| Существующие технические данные | Actions без новых order links, action_products, items, отдельная demographics база, соответствие поколению. Сортировка по (order_sid,occurrence_id) для группировки и индекс по created_us для допустимых поисков. Индексы не заменяют выбор всех версий для попавшего в scope ключа |

Finite decimal text доступен и для нулевых, отрицательных и дробных количеств. Для invalid raw type достаточно классификатора. Полный raw объект и лишние поля не сохраняются. Stable line SID строится из (order_sid, original line ID). line_number для ключа и минимального релиза не нужен. При недостающем stable ID ordinal остаётся local. Retail ID, payload Actions, split/fulfillment/revision/refund и банковские таблицы не требуются для этого контракта. Возможность детального экспорта разрешённых полей проверяется отдельно на старых ограничениях доступа. Служебные таблицы raw-ID→SID живут в приватном staging и удаляются до публикации штатным безопасным механизмом. Ранги для tie-break выводятся из исходного лексического порядка, а не из SID insertion order. Не сохранять исходные ID/контакты в отчёте или диагностике.

**C. Раздельная публикация.** В текущем приватном statistics_index на D: schema 3 сохраняет current.json. Schema 4 получает отдельный указатель current-unified-v1.json и свои generation directories под тем же корнем. Обе записи пользуются согласованным writer lock. Читатель явно выбирает указатель по policy, сверяет manifest, PRAGMA user_version, метаданные двух SQLite, поколение, source_version/provenance, ACL и identity путей. Несовместимость возвращает STALE/ошибку. V3 не открывается для UNIFIED. Если будущая правка общих модулей меняет проверяемый algorithm для V3, штатный STALE остаётся в силе. Нужен будущий rebuild из canonical для новых полей, миграция существующего SQLite без источника невозможна. Существующие ограничения памяти/диска, disk-backed staging, отмена, integrity check и атомарный pointer обязательны. Ресурсы измеряются на синтетике и отдельно разрешённом пилоте на D:. Оценка реального бюджета и rebuild относятся к D29. [Guard и публикация](../Application/statistics_index.py)

**D. Паритет.** REFERENCE нормализует raw через тот же normalizer. INDEXED восстанавливает OrderObservation без повторных бизнес-правил. Оба поставляют группы одного order key, сортировка которой не влияет на факт. Группировка REFERENCE при необходимости disk-backed на D: с действующими guards, bounded groups и проверкой отмены. Порядок групп не влияет на Decimal, buckets, диагностику и tie rank. Паритет сравнивает весь DTO за исключением source.reader, source.projection_schema, технических timing и приватного представления SID. Между приватными пространствами проверяется bijection ключей. Детали сравниваются с одинаковой локальной привязкой. Не допускается source schema, потерявшая признаки invalid/absence. При schema mismatch fallback на REFERENCE возможен только как явный режим текущего интерфейса с тем же scope и provenance.

## 15. GUI/HTML/XLSX

Одна модель представления на основе DTO 11. В заказах видны покупки, полностью и частично выполненные наблюдаемые состояния, отмены, возвратные состояния, смешанные/конфликтные заказы, полная таблица статусов и раздельные денежные значения. Сохраняются аналитические таблицы товаров, клиентов, RFM и когорт. Раскрывать можно пояснения, таблицы остаются видимыми. Технические raw counts имеют отдельную подпись и scope. Для каждого денежного показателя обозначить валюту и полноту. При недоступности показывать причину и известный подытог, при измеренном нуле — 0. Суммы форматируются общим форматтером, нулевой subtotal без оценённых строк не выдаётся за измеренную сумму.

Единая методологическая подпись proxy и declared coverage видна в GUI, HTML и XLSX. Не использовать подписи «выполнено на дату», «фактически выручено» и «сумма возврата» для наблюдаемой даты, номинала и возвратного статуса соответственно. Метаданные отчёта несут IDs policy. Открытие schema 10 сохраняет её labels и доступные тогда показатели. Неизвестные строки статусов экранируются в HTML и защищаются от интерпретации как формула в XLSX/CSV. Экспорт паритетен UI по значениям и качеству. [Общее представление](../Application/statistics_report_presentation.py)

## 16. Совместимость ML

Общий InteractionBuilder, MindboxSelectionConfig, Legacy snapshot dedup и правила BPR/hybrid остаются действующими. Новая статистическая PURCHASE-проекция порождает входы только для consumers UNIFIED. VIEW/FAVORITE используют свои существующие адаптеры. Нормализатор, reconciler и агрегаторы UNIFIED не импортируются обучением, exclusion и propensity. Изоляция проверяется сравнением старых результатов на синтетических данных и графа импортов. Изменение состава нового отчёта не запускает переобучение или миграцию артефактов. [Статистический драйвер](../Application/dataset_statistics.py) · [Строитель взаимодействий](../Application/interactions.py)

## 17. Release-critical сценарии

Синтетика. Если в строке не указано иначе, id и владелец валидны, состав полный, qty=1, priceOfLine=100 RUB, дата входит в полностью объявленный тестовый scope. Обозначение P/O — известные покупочные позиции и уникальные покупочные заказы. Непокупочные позиции сохраняются в статусах. Заказы с недоступным owner не входят в клиентские значения. Полные значения и known parts проверяются отдельно. Источник сценариев — T01–T80 DESIGN, ожидания времени ниже заменены утверждённым D13. [Исходная матрица](statistics_status_design.md)

| ID / DESIGN | Вход | Факт, позиции/заказы | Деньги | Качество и клиентские последствия |
|---|---|---|---|---|
| R01 / T01–04 | По одному F, sold, CP, delivering | F/sold/CP каждый 1/1, delivering 0/0 | F/sold 100, CP nominal 100 и recognized unknown, delivering без покупки | Known статусы. CP входит в F, delivering отсутствует в покупках |
| R02 / T05–08, T48 | Параметризовать все коды отмены и три возвратных кода как единственную строку | 0/0, одна строка соответствующего состояния, один уникальный order со статусом | Номинал состояния 100, refund/net unknown | Полный наблюдаемый CANCELLED_ONLY или RETURNED_ONLY, нет новой истории |
| R03 / T09–11 | X-unknown, отсутствующий код, bool, пробелы | 0/0 в полном входе, отдельные raw и диагностические buckets | Исходная пригодная цена остаётся номиналом | UNKNOWN_CODE, MISSING_CODE, INVALID_CODE различаются, согласованный вид заказа unavailable |
| R04 / T14, T33 | F100 + CP quantity=5 price=1000 | 2/1, MIXED, исходные единицы 6 | subtotal 100, partial_nominal 1000, recognized unknown | CP_ALLOCATION_UNKNOWN и fulfilled unknown. F=1, M недоступна |
| R05 / T12–13, T15 | F+CD, F+Return, CP+Return | В каждом 1/1, MIXED, один order в соответствующем счётчике отмен/возврата | F100 или CP unknown, отдельный nominal 100, refund unknown | Две независимые группы состояния, отсутствие денежного refund не влияет на R/F |
| R06 / T24–27 | F с qty 0, −1, 1.5, missing, bool, numeric string, NaN | 0/0 по отдельности, статус F сохраняется в каждой диагностике | Известная исходная цена не превращается в покупочную стоимость | ZERO, INVALID, UNKNOWN различимы. Нет новых F/retention |
| R07 / T28–30, T40 | F qty=2, цены 100, 0, missing, −1 | Каждый 1/1, nominal units 2, fulfilled unknown | 100, 0, UNKNOWN, INVALID соответственно | Известная 0 допускает M=0. Непригодная цена не меняет F |
| R08 / T31–32, T76 | F100 RUB + F200 KZT, затем вариант с неизвестной валютой второй строки | 2/1 | Подытоги 100/200 отдельно. В варианте unknown полный валютный итог закрыт | MIXED_ORDER_CURRENCY или CURRENCY_UNKNOWN. Общий F=1, full RFM M unavailable |
| R09 / T41, T45, T80 | Идентичные копии и версия с изменением неиспользуемой metadata | 1/1, одна согласованная строка | 100 один раз | Exact duplicate и business equivalent различимы. Перестановка частей не меняет бизнес-результат |
| R10 / T42–44 | F100 и Return100 одного line ID, без порядка | 1/1 исторически. Current status CONFLICT | F subtotal 100, полный recognized unknown, refund unknown | Покупка сохраняется. Точная current статистика строки недоступна |
| R11 / T42, T44 | Две F одной строки с ценой 100 и 120 | 1/1, состояние F согласовано | Цена CONFLICT, вклад строки недоступен | Current count известен. F=1, M unavailable |
| R12 / T43, T53 | Разные наборы stable line ID, повтор ID, пропущенный ID | Order purchase может быть 1. Точный P unknown, считать только доказанную общую часть. Одиночная версия допускает local ordinal | Различные корзины не суммируются | LINE_SET_CONFLICT, LINE_ID_DUPLICATE, LINE_ID_MISSING раздельно. M неполная |
| R13 / T77 | CP и F одного ID при одинаковой дате оформления | 1/1, состояния conflicting, одна покупка | CP allocation unknown, полный итог unknown | F=1. Не два события |
| R14 / T66 | Один order и два разных canonical owners после merges | Order evidence 1/1, 0 атрибутированных покупочных заказов клиентов | Неперсональный известный номинал сохраняется | OWNER_CONFLICT, клиентские RFM/cohorts по заказу отсутствуют. Два алиаса одного canonical owner входят один раз |
| R15 / T47, T50–51, T60 | 05.01 оформление, F виден в позднем снимке. Варианты двух разных firstAction и missing date | Известный 1/1 вне зависимости от даты | Согласованная цена остаётся известной | Proxy включает 05.01 при попадании в scope, время исполнения неизвестно. Conflict/unknown дата исключает только датированный результат |
| R16 / T55–58, T61 | Один клиент, разные заказы 05.01 CP и 04.02 F. Второй также имеет снимок всех строк Return | 2 заказа, по одному уникальному положительному line key на заказ. Второй повтор | CP unknown, F subtotal сохраняется, refund/net unknown | F=2, first=05.01, second=04.02, январская когорта и активность M1. Состояние второго конфликтно без winner |
| R17 / T59–60 | В одном окне у клиента F100 и CP100, обе даты пригодны | 2/2, F=2 | Known subtotal 100, full M unavailable | R по более поздней дате, UNSCORABLE. После исключения CP из другого окна M последнего проверяется независимо |
| R18 / T62–65 | Вторая покупка в день H. Отдельно cutoff накануне полной границы и пробел до observed_end. Повтор в другом календарном месяце | Два уникальных заказа при пригодности дат. H и Mn измеряются отдельно | Valued/known части сохраняют свой scope | Полный H включает день H, цензура и gap дают ineligible. M0=size, при eligible=0 rate null |
| R19 / T75, T09 | Полный прочитанный scope без sold и scope с неполным составом, плюс неизвестный код | Все 43 строки всегда. Для sold KNOWN 0 только в полном scope | Номиналы отдельно | Current и raw счётчики разделены до и после фильтра. Unknown code дополнительная строка |
| R20 / T64 | F у товара A, CD у B, фильтр B | Product purchase 0/0. Whole-order mixed и all-products purchase 1/1 | Cancelled nominal 100 в срезе B, buyer gross 100 | Фильтр не меняет RFM/cohorts all-products и историю order |
| R21 / T67–68 | Сохранённый schema 10, валидный schema 11, неизвестная schema, неправильный context | Старые счётчики и labels без пересчёта, новые показатели только при валидном contract | Legacy без reinterpretation, v11 quality не теряется | Legacy read-only. Unknown/mismatch отклоняются |
| R22 / T69–70 | Один набор REFERENCE и schema 4 с дробным qty, пропуском ID и конфликтом снимков. Затем попытка читать v3 под UNIFIED | В двух путях одинаковые факты и счётчики | Exact Decimal equality и те же недоступности | Parity по §14, schema 3 → STALE/ошибка. Никаких synthetic stable line IDs |
| R23 / T71 | Один DTO в GUI, HTML, XLSX и detail ZIP | Совпадают scope, покупки, статусы и связи | Decimal, zero, unknown, валюты не меняются | Proxy подпись и 43+ коды видимы, приватные ключи исключены |
| R24 / T72–74 | Cancel/read error, неразбираемый JSON-поток, диск/память/ACL, смена provenance во время работы | Неполный расчёт не публикуется | Частичные суммы не публикуются под видом результата | Предыдущий кеш/указатель/ZIP сохранены. Fail-closed и собственное staging cleanup |
| R25 / изоляция ML | Один синтетический Orders набор, содержащий sold/CP/delivering и отмены, до и после подключения UNIFIED | Новая статистика применяет R01–02. Legacy/BPR labels и exclusion прежние | Старые признаки и суммы прежние | Сравнить training inputs, weights, candidates и импортные границы. Нет нового обучения |

R16 проверяет наблюдавшийся Return всех строк как синтетический пример последующего возврата. Он сохраняет известную историю и не устанавливает факт денежного возмещения. Кейсы R01–15 и соответствующие части R20/R24/R25 предназначены CORE, остальные проверяются на уровне читателей, агрегаторов, представления и итоговой интеграции.

## 18. Ограничения первой версии

D09/D12/D14/D16/D27/D28 сохраняют ограничения из DATA. Техническое название и категория техотмены не подтверждены. Split-lineage, исполненное количество, CP allocation, verified execution time, authoritative revision, historical as-of, проверенный knowledge cutoff, refund ledger и net отложены. Для RFM/когорт используется утверждённая proxy-методология с действующей календарной eligibility. Результаты зависят от полноты доступной истории и могут отличаться от старой статистики вследствие смены популяции. Фактическое подтверждение и дополнительный источник понадобятся только для расширения перечисленных возможностей. [Доказательства и ограничения](statistics_status_data_evidence.md)

## 19. План технической реализации

| Задача | Прикладная граница | Критерий завершения |
|---|---|---|
| STAT-STATUS-CORE-01 | Новые типы, каталог, нормализация, reconciliation и единые факты на синтетических входах | Нижеприведённый backlog и проверки R01–15, продуктовый срез R20, локальная отмена R24, изоляция R25 |
| STAT-STATUS-INDEX-01 | Schema 4, staging и оба readers на синтетике. Никакого rebuild реальных данных | R22/R24, наблюдения эквивалентны и schema 3 остаётся отдельной |
| STAT-STATUS-AGGREGATES-01 | Общий факт в Orders, Actions PURCHASE, товары, клиенты, категории и деньги. DTO 11 и совместимый reader | R19–21, корректность scope и независимой полноты |
| STAT-STATUS-RFM-01 | Однополитический вход, прежняя методология и unknown M | R15–18 на RFM, прежние score/ranks/windows |
| STAT-STATUS-COHORTS-01 | Proxy first/second, когорты, H/Mn, деньги, каналы/категории | R15–18 для когорт, соблюдение eligibility |
| STAT-STATUS-UI-01 | Представление и форматы GUI/HTML/XLSX, ZIP2 | R21/R23/R24, один DTO и совместимые старые отчёты |
| STAT-STATUS-ACCEPTANCE-01 | Все release checks, измерения ресурсов, затем отдельное решение D29 | Синтетический PASS и план ограниченного пилота с бюджетом. Реальный пилот/rebuild по отдельному разрешению |

### Прикладной backlog STAT-STATUS-CORE-01

1. Изучить statistics_purchase_policy.py, statistics_purchase_facts.py, mindbox/adapters/orders.py и _common.py, mindbox/order_dedup.py, mindbox/identity.py, order_statistics.py и места вызова в dataset_statistics.py. Учитывать действующую нормализацию, безопасность исключений и границы adapter errors.
2. Ввести изолированные модули Application/statistics_unified_policy.py и Application/statistics_unified_facts.py. В первом PurchasePolicy, enum и неизменяемый каталог из 43 точных строк, classify_status_code(value), qualify_quantity(value) и is_purchase_candidate(status, quantity). Каталог проверяется против inventory fixture. CSV исследовательского отчёта не становится runtime-зависимостью. Во втором структуры §10, normalize_order_observation(raw, source_ref, identities, resolve_product), reconcile_order(observations, *, order_sort_rank, product_predicate=None, check), evaluate_money(agreed_lines, qualifying_evidence, *, completeness) и scope projection. Чистый нормализатор отделяет причины непригодности, wrapper отвечает за I/O и cancel. Reconciler принимает одну заранее собранную группу, не читает источники и не управляет кешем.
3. Параметризованные синтетические тесты на R01–15 и R20. Отдельно перестановка версий, сбой нормализации, cancel callback и защита repr от IDs. Smoke импорта и старые Legacy/ML контракты по R25. Проверить отсутствие новых сторонних зависимостей.
4. Глобальные OrderSnapshots.accept, InteractionBuilder.from_order_line, текущий build_purchase_order_fact, покупочные списки Legacy/STRICT/EXTENDED, CustomerIdResolver, training и exclusion не изменяются. Допустимо вызвать существующие чистые функции адаптации с сохранением их семантики. Новый путь не включается в штатный расчёт до этапов интеграции.
5. PASS CORE — все требуемые локальные синтетические проверки и целевые pytest/Ruff проходят. Факты повторяемы, unknown и 0 различимы, ни raw данные, ни индекс, ни кеш, ни штатная отчётность не меняются. Исследование Mindbox для этих задач не требуется.

## 20. Критерии готовности к разработке

Политика, D13, обработка quantities, состав reconciler, независимое качество и деньги заданы в §2–10. Формулы и входы поведенческой аналитики закреплены в §11–12. Версии, совместимость, паритет и безопасная публикация определены в §13–16. Release tests и backlog §17–19 достаточны для отдельного CORE на синтетике.

Незакрытых бизнес-решений, блокирующих CORE, нет. D29 относится к реальному пилоту и будущей сборке. Ограничения данных определяют unavailable capabilities, а не задерживают минимальный релиз. Контракт не вводит изменения в действующий код, формат или расчёт. Работа заканчивается двумя документами и передачей backlog.
