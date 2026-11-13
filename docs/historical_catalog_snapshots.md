# HYBRID-CATALOG-SNAPSHOTS-01

`Application.evaluation.catalog_snapshots` — отдельный research API, без GUI,
автоматического импорта, обучения, inference и второго evaluator. Он принимает
только явно указанные файлы и внешний root. Production продолжает читать свои
`input_data/nomenclature.csv` и `site_categories.csv`.

## Формат входа и политика качества

UTF-8 или UTF-8 BOM, разделитель `|`, уникальные непустые заголовки, одинаковая
ширина строк. Обязательные колонки номенклатуры: `КодНоменклатуры`,
`ВидНоменклатуры`, `ПолНоменклатуры`, `Коллекция`, `Остаток`, `КатегорияНаСайте`.
Остальные существующие поля сохраняются в оригинальном CSV; известные атрибуты
доступны через `CatalogItem`. Для категорий обязательны `КодКатегории`,
`НазваниеКатегории`, `КодРодительскойКатегории`.

`read_rows`, Decimal `stock()` и `RecommendationCatalog.load()` переиспользуются.
Production `import_reference()` намеренно не вызывается: его DataFrame-путь
округляет остатки, заменяет пропуски нулём и переписывает категориальные коды.
Исследовательский архив сохраняет исходные bytes и ведущие нули. Автоопределения
разделителя/кодировки, prefix-6 обрезания и исправления кодов здесь нет.

* Неправильная структура/кодировка и непригодные непустые товарные коды приводят
  к отказу регистрации. Пустые коды считаются отдельно и не становятся товарами.
* Пустые, отрицательные, нечисловые и бесконечные остатки диагностируются;
  stock=None, без импутации и включения в `in_stock_codes`. Нулевой stock допустим,
  но также не входит в это множество. Положительные Decimal не округляются.
* Совпадающие повторные строки не усиливают товар. Различающиеся строки одного
  кода дают `AMBIGUOUS_CODE`, исключаются из `items`, остаются в `known_codes`.
  Политика соответствует production. Остатки конфликтующих строк не суммируются.
* Несколько разных непустых категорий одного товара — отдельный счётчик
  `multi_category_codes`. Это наблюдение, не доказанная ошибка данных; до
  согласования иного правила код остаётся неоднозначным по общей политике.
* Категория номенклатуры разрешается **только по коду**, без fallback на название.
  `category_paths` содержит code-only множества из существующего resolver.
  Неизвестные/пустые категории, отсутствующие родители и невалидные пути
  диагностируются отдельно. Циклы и конфликтующие категории не получают пути.
* Пустые вид/пол/коллекция диагностируются, не заполняются предположениями.

`diagnostics` — безопасные агрегированные counters. `exclusions` — причины по
кодам для локального исследования; это не готовый агрегированный отчёт.
Некоторые причины относятся к availability, другие предупреждают о missing
attributes/category; это не окончательное решение о recommendation eligibility.

`known_codes`, `items`, `stocks`, `in_stock_codes`, `category_paths`, `diagnostics`
и `exclusions` возвращаются как неизменяемые структуры. Положительный остаток
не означает выполнение правил видов, коллекций, минимального наличия, покупок,
identity или matching. Эти правила здесь не дублируются.

## Даты и происхождение

`SnapshotProvenance` явно хранит:

* `inventory_date` — UTC-дата точного состояния; если время неизвестно, заявленная
  календарная дата без предположения о часовом поясе или начале суток;
* `inventory_as_of` — точный aware datetime состояния остатков, либо None;
* `exported_at` — время фактического создания номенклатурной выгрузки;
* `attributes_observed_at` — время получения используемых характеристик;
* `categories_observed_at` — время получения категорий;
* `categories_exported_at` — время создания категориальной выгрузки, либо None;
* отдельные `inventory_origin`, `attributes_origin`, `categories_origin` из
  `SourceOrigin`: HISTORICAL_1C / CURRENT_1C / ARCHIVED_EXPORT / SYNTHETIC.

Все известные timestamps нормализуются в UTC. При известном inventory_as_of его
UTC-дата должна совпасть с inventory_date. Эти даты не выводятся друг из друга,
из имени файла или filesystem mtime. Если точный момент остатков неизвестен,
он остаётся None, даже когда известен день.

## Хранение, целостность и атомарность

```text
<explicit external root>/
    <snapshot_id, 64 hex>/
        nomenclature.csv
        site_categories.csv
        manifest.json
```

Root внутри проекта, включая input_data, или внутри Git worktree запрещён.
Конструктор ничего не создаёт. Регистрация явно копирует только переданную пару
в собственный `.staging-*`, проверяет файлы, fsync-ит их и manifest, затем публикует
весь каталог одним `os.rename` на том же filesystem. Существующие версии не
перезаписываются. При ошибке до публикации удаляется только собственный staging.
Остатки staging после аварий не считаются опубликованными снимками; автоматической
очистки чужих каталогов нет. Атомарность rename не является гарантией сохранения
directory entry при внезапном отключении питания на любом filesystem.

Manifest v1 / parser v1 содержит snapshot_id, provenance, SHA-256 и byte size
каждого оригинального файла, `normalized_sha256` проекции, диагностику, validity
и ограничения. Snapshot ID — SHA-256 канонического manifest без самого ID.
Хеш нормализованной проекции относится к разобранным безопасным товарам/остаткам,
известным кодам и разрешённым путям; файлы при этом не переписываются.

Идентичные bytes и provenance дают идемпотентную регистрацию с тем же ID.
Изменённые bytes или provenance дают другую версию, даже в тот же день.
Повреждённая существующая версия с прежним ID вызывает отказ, а не замену.
Проверяются checksums до/после разбора и согласованность diagnostics/проекции.
Duplicate JSON keys, подмена manifest, пути обхода и symlink snapshots отклоняются.
Хеши удостоверяют связность данных, но не их внешнее историческое происхождение.

Чтение CSV потоковое, копирование/хеширование блоками 1 MiB. По умолчанию не более
250 000 строк в каждом CSV; лимит можно явно изменить. Память resolver зависит
от уникальных товаров, атрибутов и категорий. Проверка делает несколько проходов
по выбранной паре; все исторические годы или чужие CSV не сканируются.

## Выбор и статус PROXY

`list_snapshots()` читает только manifests; `available_dates()` перечисляет даты.
`load(id)` проверяет только выбранную пару и не открывает другие snapshots.
`load_for_date(day)` допустим при единственной версии. Иначе нужен snapshot_id.
`load_for_cutoff(cutoff, snapshot_id=...)` требует точного совпадения известного
inventory_as_of с cutoff. Нет автоматического nearest-past, выбора «последней»
версии или предположения, что утренний stock действовал вечером.

`snapshot.assess_cutoff()` возвращает существующий `Validity`: всегда PROXY при
совпадении момента, UNAVAILABLE при неизвестном времени/несовпадении cutoff.
`load_for_cutoff()` в последнем случае отклоняет выбор. Некорректный/naive cutoff
также вызывает ошибку. Snapshot не может получить STRICT через параметр.

Manifest хранит `POINT_IN_TIME_UNVERIFIED` и вычисленные ограничения:
UNKNOWN_INVENTORY_TIME, LATER_ATTRIBUTES, LATER_CATEGORIES, RETROSPECTIVE_EXPORT,
RETROSPECTIVE_CATEGORIES_EXPORT, CATEGORY_EXPORT_TIME_UNKNOWN — по применимости.
Даже одинаковые даты и origin=ARCHIVED_EXPORT не доказывают STRICT. Проверенный
point-in-time источник и evidence для существующего hybrid_provenance потребуют
отдельной реализации. Этот архив не создаёт положительные EvidenceReview.

## Пример: регистрация маленькой синтетической пары

Пути должны указывать на явно подготовленные fixtures/выгрузки; код ниже не
запускается автоматически и не создаёт фиктивные исторические данные.

```python
from datetime import date, datetime, timezone
from Application.evaluation.catalog_snapshots import (
    HistoricalCatalogArchive, SnapshotProvenance, SourceOrigin,
)

archive = HistoricalCatalogArchive(r"D:\KANZLER-research\historical_catalog")
inventory_time = datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc)
attributes_time = datetime(2026, 10, 28, 8, tzinfo=timezone.utc)
metadata = SnapshotProvenance(
    inventory_date=date(2026, 6, 1), inventory_as_of=inventory_time,
    exported_at=attributes_time, attributes_observed_at=attributes_time,
    categories_observed_at=attributes_time, categories_exported_at=attributes_time,
    inventory_origin=SourceOrigin.SYNTHETIC,
    attributes_origin=SourceOrigin.SYNTHETIC,
    categories_origin=SourceOrigin.SYNTHETIC,
)
snapshot = archive.register(
    r"D:\fixtures\june\nomenclature.csv",
    r"D:\fixtures\june\site_categories.csv", metadata,
)
selected = archive.load_for_cutoff(inventory_time, snapshot_id=snapshot.info.snapshot_id)
print(selected.validity.value, dict(selected.diagnostics))  # PROXY, aggregate counters
```

## Как передать реальные летние выгрузки позже

После отдельного согласования подготовить по паре CSV для июня, июля, августа.
Номенклатура может сочетать исторический stock с сегодняшними атрибутами, но это
должно быть явно заявлено в provenance. Для каждой пары нужны известный момент
stock либо честное «время неизвестно», реальное время создания выгрузок и даты
получения атрибутов/категорий. Указать внешний root и зарегистрировать каждую пару
отдельным вызовом, сохранив возвращённые IDs. Новые версии существующий архив не
заменяют. Неподдерживаемую кодировку сначала согласованно преобразовать отдельно,
не переписывая оригинал и не скрывая происхождение.

Наличие кода в сегодняшней номенклатуре не доказывает существование товара на
прошлую дату. Исторический stock также требует уточнения семантики: какие склады,
магазины, единицы, резервы и доступность покупателю он отражает.

Никакого benchmark здесь нет: опубликованная BPR, обученная после cutoff, не
становится исторически допустимой от добавления каталога. Для сравнения нужны
разрешённая training history, признаки, identity, purchase evidence и независимые
future labels в уже существующем evaluator. Его контракты не изменены.
