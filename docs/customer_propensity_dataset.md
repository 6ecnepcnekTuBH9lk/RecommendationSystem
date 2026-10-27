# CUSTOMER-PROPENSITY-SAFETY-01 — дисковый резерв и SQLite temp

Полный build и probe теперь имеют единый обязательный минимум **20 ГиБ**
(`MIN_FREE_BYTES`, default `build(min_free_bytes=...)`). Явные значения ниже
минимума отклоняются, а не молча повышаются. CLI `--min-free-gib 1`/`0`/`19`
завершается до создания builder/чтения ключа/тяжёлого SQL. Можно увеличить резерв.
Это намеренное ужесточение прежнего API с default 1 ГиБ; output/schema v2 не меняются.

## Где SQLite создаёт временные файлы

Новый `Application/model/customer_propensity_resources.py` определяет расположение
через **SQLITE_FCNTL_TEMPFILENAME** работающего SQLite VFS. `TEMP`/`TMP`, системный
`tempfile.gettempdir()` и выходной каталог не используются как предположение.

Для поддерживаемого Windows runtime проверяется:

1. PE import table стандартного `_sqlite3` действительно импортирует `sqlite3.dll`.
2. В процессе есть единственная такая уже загруженная DLL; новую копию через PATH
   не загружаем. Её version/source_id совпадают с Python-linked SQLite.
3. Отдельный краткоживущий native handle открывает индекс **READONLY**, без CREATE.
4. VFSNAME должен быть `win32`/`win32-longpath`. TEMPFILENAME запрашивается у VFS;
   извлекается существующий абсолютный parent, строка освобождается sqlite3_free,
   собственный native handle закрывается в finally. Сам temp-файл не создаётся.

Не используются private pointers/ABI структуры Python sqlite3.Connection,
`PRAGMA temp_store_directory`, запись sqlite3_temp_directory или изменение ENV.
`Application/customer_history_index.py` и его **temp_store=FILE** остаются прежними.
SQLite-соединение `:memory:` внутри проверки служит только для чтения source_id,
не содержит историю и не заменяет FILE storage реального reader.

Binding выполняется один раз на build, текущий parent от VFS проверяется при
каждом дисковом poll. Если расположение изменилось, запуск останавливается с
`SQLITE_TEMP_CHANGED`, чтобы не потерять контроль старых temp-файлов. Неизвестный
runtime, static/custom extension, несколько DLL, неподдержанный VFS/платформа,
ошибка определения каталога — `SQLITE_TEMP_UNVERIFIED`, до тяжёлого history SQL.
Отсутствие native Windows поддержки не блокирует metadata inspect, но блокирует build/probe.

## Какие тома контролируются

Output и подтверждённый SQLite temp проверяются по реальной Windows volume GUID:
`Path.resolve` → `GetVolumePathNameW` → `GetVolumeNameForVolumeMountPointW`.
Это учитывает junction/mounted-volume пути; текстовые префиксы букв дисков не
являются идентичностью. Если GUID одинаковы, disk_usage выполняется один раз и
нужен один общий резерв 20 ГиБ, а не два по 20. Если тома разные — каждый должен
сохранять свой минимум. Ошибка определения тома/free bytes — `DISK_INFO_UNAVAILABLE`;
недостаток места — `INSUFFICIENT_FREE_SPACE`. Неизвестное temp-расположение не заменяется output.

Free bytes уже отражают занятые текущими output/temp файлами объёмы. После их роста
должно оставаться не менее floor; проверка не предсказывает точный будущий peak и
не гарантирует, что вся сборка поместится. До отдельного full запуска нужно учесть
ожидаемый output и запас на SQL, чей точный peak по одному EXPLAIN неизвестен.

`_RunGuard` сохраняет проверки отмены каждого события/строки, периодический disk
poll раз в секунду или каждые 8 МиБ физической записи, принудительный poll до
history SQL и перед публикацией. SQL progress handler вызывается через 10 000
VM-инструкций, возвращает прерывание и сохраняет диагностическую категорию.
Не вызывает системные disk API на каждой строке. Проверки действуют и при
валидации/SHA256, включая checksum manifest перед переключением указателя.

Прерывание закрывает Python/native readers и gzip writers, удаляет только свой
UUID staging и сохраняет прежний CURRENT/CURRENT_INFERENCE. Ошибка после rename
может оставить **завершённую проверенную версию без указателя**, как и раньше;
это не публикация неполного gzip. Старые версии не удаляются.

Границы: это периодический кооперативный контроль, а не эксклюзивная резервация
диска. Внешний процесс может записать между poll, а отдельная SQLite VM-операция
или блокирующий I/O могут задержать callback. Автотаймера для полного build нет;
probe сохраняет свои отдельные ограничения. Глобальные настройки других частей
приложения и других процессов не меняются.

Metadata `inspect()` остаётся лёгким: `minimum_free_bytes` сообщает политику,
`disk_safety=NOT_CHECKED_BY_METADATA_INSPECT` прямо указывает, что native temp/disk
preflight выполняется при build/probe, а не подтверждён одним index READY.

## SQL-планы и проверка

FACTS_SQL/ORIGINS_SQL не изменялись. На реальном read-only индексе выполнены только
EXPLAIN QUERY PLAN и metadata/VFS diagnostics; глобальные SELECT не исполнялись.
FACTS использует covering scan event_customers_customer, lookup по PK/UNIQUE и
TEMP B-TREE FOR RIGHT PART OF ORDER BY. ORIGINS использует тот же covering scan,
record_origins_variant и TEMP B-TREE FOR GROUP BY. SCAN ожидаем для полной истории;
стоимость и temp peak по нему точно не определяются. Переписывание SQL — отдельная задача.

Отчёт SAFETY-01: `filtered_data/.propensity-safety01-20261020/REPORT.md`.
Математика, 22 features, UNKNOWN, split/embargo, HMAC и storage contract не менялись.
Ни полного build, ни inference, ни обучения/повторного реального probe в SAFETY-01 не было.
Ниже сохранены предыдущие этапы; минимальный резерв 1 ГиБ в старых описаниях больше не действует.

---

# CUSTOMER-PROPENSITY-OPT-01 — потоковый gzip и bounded probe

Для будущих сборок формат по умолчанию — **JSONL.gz** (`PropensityStorage()`),
явный отладочный вариант — `PropensityStorage("jsonl")` / `--storage-format jsonl`.
ML schema остаётся **v2**, target `order_placed_within_30_days`, все 22 признака,
правила UNKNOWN и chronological split/embargo сохраняются. Глобальный SQL-план
полной сборки и математический reducer не переписывались. Ниже сохранён документ 01B.

## Контракт хранения и публикация

Новый `Application/model/customer_propensity_storage.py` содержит конфигурацию,
потоковые writer/reader и общую проверку строк. По умолчанию gzip level=6;
`--gzip-level 0..9` задаёт уровень явно. Заголовок имеет `mtime=0`, пустое filename,
UTF-8 и LF; порядок строк/JSON-полей прежний. Несжатой копии при gzip-записи нет.
Воспроизводимость физических gzip-байтов проверена в одном Python/zlib runtime;
одинаковость с другим runtime не обещается. Контрольные суммы входят в version.

Manifest дополняется `storage_contract_version=1`, `storage_format`, `compression`,
`file_names`, `file_sizes`. Прежние `counts.files` содержат число строк по разделам,
`checksums` — **SHA256 физических файлов**, включая gzip header/trailer.
`counts.uncompressed_bytes` измеряет размер эквивалентного UTF-8 JSONL без создания
копии. Это размер, не второй хеш. `metrics` — диагностические времена фаз;
они не входят в content identity и не мешают детерминированному reuse.

Сначала закрываются все gzip-потоки (включая trailer), затем flush/fsync физических
файлов, потоковая проверка CRC/JSON/schema/counts, SHA256 и manifest. Только затем
freshness и rename staging в версию; CURRENT/CURRENT_INFERENCE обновляются последними.
Отмена/ошибка до публикации очищает только собственный staging; предыдущие версии
не удаляются. При ошибке после rename может остаться завершённая версия без указателя,
как и раньше. Полная сборка не получила автоматического ограничения длительности.

## Потоковое чтение

```python
from Application.model.customer_propensity_storage import iter_propensity_rows

for row in iter_propensity_rows(version_directory, "train", purpose="training",
                               source_version=expected_source, generation=expected_generation):
    consume(row)
```

Reader принимает старый schema-v2 manifest без storage-полей как обычный JSONL,
а новый читает строго по manifest. Schema v1, другой target/purpose/horizon,
повреждённый manifest, неправильные размеры/SHA256 и несоответствия строк отклоняются.
Нельзя читать v1 как v2. Reader сначала проверяет SHA256 выбранного физического файла,
затем построчно распаковывает и проверяет содержимое; временной распакованной копии нет.
Для окончательной проверки CRC и количества строк итератор нужно исчерпать.
При раннем выходе вызывающий код должен закрыть генератор (`contextlib.closing`).
Путь версии передаётся явно: reader сам не переключает CURRENT и не строит ML pipeline.
Inference читается с `purpose="inference"`, его строки не содержат target/label_state/label_reasons.

## Probe: отдельные исследовательские артефакты

`--probe` и `--build` взаимно исключают друг друга. Без них CLI по-прежнему выполняет
только metadata preflight. Probe использует тот же `customer_rows`, writer и validator;
при этом вычисляются только одна/две явно выбранные даты из актуальной полной сетки.
Calendar preflight может перечислять 55 дат; features/labels для остальных не считаются.

Пример ограниченного запуска (не команда для полного build):

```powershell
.venv312\Scripts\python.exe -B -X utf8 scripts/build_customer_propensity_dataset.py `
  --probe --allow-retrospective --ephemeral-probe-key `
  --probe-date 2023-01-26T00:00:00Z --probe-date 2026-07-09T00:00:00Z `
  --probe-max-customers 2000 --probe-max-events 200000 `
  --probe-max-seconds 600 --probe-max-mib 64 --min-free-gib 20
```

Для стабильных псевдонимов вместо ephemeral-флага передать `--key-file <existing key>`.
Эти варианты несовместимы; ключи не печатаются и не записываются в dataset. Временный
ключ разрешён только для probe и живёт в памяти процесса; прежние ключи не меняются.

Ограничения API `PropensityProbeConfig`:

- 1–2 упорядоченные различные UTC-даты, strict subset grid; training purpose.
- До 10 000 клиентов, default 2 000; детерминированно первые canonical IDs в SQLite
  BINARY order. Не отбор по метке; выборка нерепрезентативна, баланс классов не оценивается.
- До 1 000 000 logical events/вариантов (default 200 000); до grouped SQL отдельно
  проверяется вход вариантов и origins (origins максимум 2*event limit).
- До 1 800 секунд, default 600; deadline проверяется и внутри SQL progress handler,
  при feature/JSON/write/validation/hash и перед публикацией.
- До 2 ГиБ физических выходных файлов, default 128 МиБ; hard cap включает gzip
  headers/trailers и manifest. Это лимит output, а не обещание точного пика SQLite temp.
- Минимум 20 ГиБ свободного диска, даже если передан меньший reserve.

SQL выбирает ограниченные IDs по индексу, затем оба штатных SQL-потока ограничиваются
этими IDs **до** GROUP BY. Все variants выбранных logical events сохраняются, включая
conflicting owner. Сортировка полного build не менялась; N+1 SQL не добавлен.
Лимит/отмена очищает незавершённую попытку, автоматического перехода к full или retry нет.

Probe сохраняется в `<output>/probes/<version>/` со статусом `RESEARCH_PROBE_ONLY`,
`probe` constraints и `scan_scope=bounded_probe`; CURRENT и CURRENT_INFERENCE не создаёт.
CLI default output для него — отдельный `filtered_data/customer_propensity_research`.
Reader требует `allow_probe=True`; такие результаты нельзя использовать как полноценный
training dataset. Область полного build и его указатели остаются отдельными.

## Контроль ресурсов и оставшиеся затраты

Ручная отмена проверяется на каждом событии/строке и в SQL. `disk_usage` выполняется
на старте/перед публикацией и не реже раза в секунду либо каждых 8 МиБ физической
записи; это устраняет системный вызов на каждую строку, сохраняя резерв. Эта защита
не резервирует диск против произвольной внешней записи другими процессами.

Кэширование coverage/event инвариантов и moving windows пока **не реализованы**:
это отдельный следующий change с differential tests. Полный reader по-прежнему имеет
два SQL-потока, global GROUP BY provenance с temp B-tree и partial ORDER BY facts.
Gzip решает расход output, но не вычислительную сложность O(events*slices) или temp
глобальной сортировки. У gzip нет column projection, поэтому для ML нужен streaming
parser/batches; Parquet и новые зависимости в этой задаче не добавлялись.

Реальные измерения OPT-01 и итоговые тесты приведены отдельно в
`filtered_data/.propensity-opt01-20261020/REPORT.md`. Полный build и обучение не запускались.
Бизнес-ограничения 01B/preflight остаются: REQUIRES BUSINESS VALIDATION.

---

# CUSTOMER-PROPENSITY-01B — schema v2: оформление заказа

Текущая цель: `target = order_placed_within_30_days`. Семантика схемы v1
«подтверждённая покупка» несовместима с v2. Существующий сборщик, locks,
потоковая группировка, HMAC, отмена, freshness и атомарная публикация сохранены.
Никакие BPR/KPI/purchased-products/SQLite-schema/GUI правила не изменены.
Реальные данные в задаче 01B не обрабатывались; аудит статический, тесты синтетические.

## Определение факта и бизнес-ограничения

Один логический ORDER — один идентификатор `ids.mindboxId`, согласованный с
`orders.order_id`, `events.source_event_id` и вычисленным `event_key`.
Проверяются связь snapshot/variant, достоверность canonical owner и согласованность
`variants.event_us` с `order_snapshots.order_us` и между всеми снимками.
Дата — нормализованный в UTC `firstAction.dateTimeUtc`. Повторные origins,
товарные строки, fingerprints и обновления статусов не создают новые заказы.

Статусы N/new/CP/F/delivering/sold/CD/RJ/Return/otmenen и любые другие не влияют
на этот факт. В v2 SQL-проекция вообще не читает строки, raw_status, quantity,
цены, товары и платежи. Расхождения этих полей не вызывают UNKNOWN оформления.
Старый helper `order_outcome` оставлен для регрессии v1 и не вызывается v2 builder.
`DEFAULT_PURCHASE_STATUSES` остаётся `(CP, delivering, F)`; справочник по скриншотам
не расширяет глобальные статусы PURCHASE. Реальные значения статусов не профилировались.

**Не подтверждено, что firstAction означает завершённое оформление.** Локальный
адаптер просто сохраняет первое действие. В индексированной модели нет признака,
который надёжно отделяет черновик/технический заказ от оформленного. Полноценность
checkout не выводится из наличия товарных строк или финального статуса.
Поэтому согласованный заголовок используется только как исследовательский proxy
«логический заказ OrdersAPI по firstAction», с явным ограничением, а повреждённая
идентичность получает UNKNOWN. Не добавляется произвольная эвристика по статусу.

Второе ограничение: `Application/mindbox/CANONICAL_STORAGE.md` и `MANUAL_IMPORT.md`
прямо указывают, что ручной период — metadata пользователя; Orders не разделяются
по firstAction, дата создания может быть вне периода экспорта. Полнота экспорта
именно по времени оформления независимо не доказана. Нули условны на заявленной
полноте покрытия, а не на сверке с Mindbox. До бизнес-подтверждения определения
оформления/отсутствия drafts и фильтра полноты результат требует
**REQUIRES BUSINESS VALIDATION**. Это записано и в manifest.

Метки: `1` — достоверный logical order в `[t,t+30d)`; `0` — нет заказа или
неопределённого кандидата при полном заявленном Orders coverage. Конфликт owner,
времени или identity и отсутствующее provenance потенциального заказа дают UNKNOWN.
Положительный заказ перекрывает другой неопределённый кандидат. Прежняя осторожная
политика неполного будущего окна сохранена: оно даёт UNKNOWN даже с найденным заказом.
Ни первый, ни последний снимок не назначается победителем.

## Режим достоверности и 22 признака

Выбран **B — RETROSPECTIVE_EVENT_TIME**. Для build по-прежнему обязательно явное
`allow_retrospective=True` / `--allow-retrospective`. Схема v2 намеренно разрешает
оформленные заказы по event time даже при более поздней публикации выгрузки.
Все их event times строго `< t`. Статусы, состав, количество и цены не используются.
Исторические `purchase_*`/units убраны; под прежними именами значения не меняются.

Строгий запрос `POINT_IN_TIME_VERIFIED` явно отклоняется с
`POINT_IN_TIME_IDENTITY_UNVERIFIED`: текущий индекс не хранит историческую доступность
merge graph. Нет скрытого перехода в B или набора, ошибочно помеченного уровнем A.
Поздняя публикация и отсутствие event availability также остаются ограничениями.

| Признаки | Количество |
|---|---|
| `placed_orders_{30,90,180,365}d` | 4 |
| `days_since_last_order_365d`, `has_order_365d` | 2 |
| `order_active_days_{30,90}d` | 2 |
| `average_days_between_orders_365d` | 1 |
| `views_{7,30,90,180}d`, `favorites_{7,30,90}d` | 7, сохранены |
| `views_active_days_30d`, `favorites_active_days_30d` | 2, сохранены |
| `days_since_view_180d`, `has_view_180d`, `days_since_favorite_90d`, `has_favorite_90d` | 4, сохранены |

Окна `[t-window,t)`, активные дни календарные UTC, давность в полных сутках.
Средний интервал = `(last-first)/(n-1)` в сутках по distinct logical orders;
вычисляется с Decimal precision 64, хранится decimal string без float. При n<2
он null с available=true; при неизвестном окне null с available=false и причиной.
То же различие сохранено для давности при наблюдаемом отсутствии событий.
День оформления ровно в t попадает в label, а не в признаки. Дубли snapshots
не меняют n, интервалы или active days. Отмена позднее не уменьшает число оформлений.
Недоступные/неполные окна остаются null, деньги/quantity не подставляются.

Когорта — клиенты с достоверным взаимодействием до t, включая только просмотры
или избранное. Нынешняя Customers/профиль/KPI не являются источниками когорты или
признаков; поздние current merges — явно сохранённое B-ограничение.

## Даты обучения и inference

Training-сетка и chronological split/embargo сохранены: полное будущее 30 дней,
цели train заканчиваются <= validation_start, validation <= test_start. При
step_days<30 внутри split допустимо пересечение; классы не балансируются.

`latest_prediction_date(coverage)` объединяет реальные интервалы Orders и Actions,
пересекает их и выбирает последнюю общую границу полностью загруженного UTC-дня.
Проверяет непрерывные 365 дней Orders и 180 дней Actions до этой границы — это
максимальные окна текущего набора признаков. Одинокая более поздняя partition не
сдвигает дату за обязательный источник. Если последний общий участок недостаточен,
возвращается INFERENCE_HISTORY_INCOMPLETE, а не молчаливый откат к старому прогнозу.
При отсутствии общего полного дня — NO_COMMON_COMPLETE_DAY. Частичный день не
объявляется полностью загруженным. Системная дата не используется.

Для ранее подтверждённого покрытия `[2022-01-01,2026-09-01)` пример даёт:
последний день 31.08.2026, t=01.09.2026, интервал `[01.09,01.10)`.
Это пример на имеющихся метаданных, а не новая реальная проверка в 01B.
После расширения обоих источников граница пересчитывается автоматически; расширение
только одного источника не обязано сдвинуть t.

`PropensityDatasetConfig(purpose="inference")` и `make_plan` дают один срез.
`customer_rows` использует тот же reducer, но возвращает `PropensityInferenceRow`
**без target, label_state, label_reasons**. Будущее не требуется и не оценивается.
Дату нельзя задать через start/end или обучающие split-границы.
Надёжные события >=t не запрашиваются SQL. События с NULL time consensus сохраняют
все конфликтующие заголовки, чтобы будущий вариант не скрывал неоднозначную историю.

Строка inference: customer_key, prediction_utc, target_until_utc, split=inference,
features, available, feature_issues. Training `PropensityFeatureRow` добавляет
target/label_state/label_reasons. Таким образом отсутствие будущих заказов в
последнем срезе не превращается в отрицательную метку.

## Артефакты, CLI, проверка

Общий атомарный pipeline сохранён: локальный ignored output, HMAC, writer/shared
locks, source_version/generation/schema проверки, проверка выходных строк, SHA256,
fsync, собственный staging, rename и указатель. Manifest schema_version=2,
target=order_placed_within_30_days, purpose, label_fields_present, as-of, business
validation, feature schema, config/cutoffs/горизонт/coverage, определения logical
order/time, причины UNKNOWN, ограничения, counters/distributions и checksums.
Старые каталоги v1 не удаляются и не переиспользуются как v2.

В версии сохраняются шесть JSONL: train, validation, test, unknown, embargo,
inference. Для training последний пустой; для inference первые пять пустые.
Training публикуется через CURRENT.json, inference через CURRENT_INFERENCE.json,
поэтому один не заменяет другой. Оба указателя содержат SHA256 manifest.
Будущий читатель обязан проверять schema=2, target, purpose, as-of, checksums
и актуальность source_version/generation; одного наличия CURRENT недостаточно.

Полная training-сборка сверяет logical/variant/membership counters с метаданными.
Inference читает только граф событий до среза (scan_scope=pre_cutoff_inference),
проверяет его memberships/variant counts; не сравнивает отфильтрованный объём с
полным числом событий индекса. Это не замена full SQLite integrity check.
Два SQL-потока/один read snapshot и ограниченная память сохранены. В v2 память
текущего события зависит от количества вариантов, а не его товарных строк.
Максимальные наборы active days фиксированы: 30/90 дней заказов и 30 дней actions.
Сортировки SQLite по-прежнему могут потребовать временного диска. Авто-таймера нет.

Только metadata preflight (команды подготовлены, на реальных данных в 01B не запускались):

```powershell
.venv312\Scripts\python.exe -B -X utf8 scripts/build_customer_propensity_dataset.py
.venv312\Scripts\python.exe -B -X utf8 scripts/build_customer_propensity_dataset.py --purpose inference
```

Для будущего отдельно разрешённого запуска к нужному режиму добавить
`--build --allow-retrospective --key-file <секретный бинарный ключ>`.
Доступны прежние параметры диска/путей/training-сетки, а также `--as-of-level`.
Ключ нельзя хранить в версии датасета/публиковать. CLI Ctrl+C отменяет штатно;
при ошибке старые указатели сохраняются, собственного staging не остаётся.

Контрольная выборка прежней задачи с отменённым CD: теперь **9 наблюдений,
2 positive, 7 negative, 0 UNKNOWN** (прежде 1/7/1). Новая частота 2/9 естественно
следует из изменения цели. Неопределённые owner/time/identity/provenance проверены
отдельно; произвольные UNKNOWN не превращаются в 0. Все изменения проверяются
synthetic-only тестами, включая inference, clock independence, coverage/gaps,
statuses/dedup/snapshots, privacy/freshness/atomic/cancel и прежние регрессии.

Итог 01B: 88 propensity-тестов прошли (финальный прогон 6,21 с). Расширенный
целевой прогон с индексом/нормализацией/lookup, KPI, hybrid/purchase evidence,
BPR preparation и Mindbox preparation: 509 passed, 2 deselected, 44,04 с.
Два training-core integration tests намеренно исключены, обучение не запускалось.
Ruff и проверки diff, включая новые незакоммиченные файлы, проходят.

Следующий этап требует бизнес-подтверждения firstAction/черновиков и полноты
по датам оформления, review и отдельного разрешения на реальный build. Вердикт
**REQUIRES BUSINESS VALIDATION**, без заявления о готовой модели или эксплуатации.

<details>
<summary>Архив: постановка и отчёт CUSTOMER-PROPENSITY-01, schema v1 (покупка), не текущий контракт</summary>

# CUSTOMER-PROPENSITY-01 (архив v1)

Результат: **RESEARCH DATASET ONLY**. Реализован сборщик исследовательского
набора. Модель не обучается, интерфейс и существующие алгоритмы не меняются.
Полный расчёт на реальных данных требует отдельного согласования и не запускался.

## Аудит временной достоверности

Метаданные реального v2-индекса проверены 19.10.2026 без чтения событий и без
`verify_integrity=True`. Статус READY, generation `1fdf4490fba047e6af4e4b0a70163da7`.
Это наблюдение на дату аудита, не константа сборщика.

| Сведения | Доступно | Ограничение |
|---|---|---|
| Actions event time | `variants.event_us` из `dateTimeUtc` | Не доказывает первоначальную доступность записи |
| Action creation | `actions.creation_us`, может отсутствовать | Если creation позже t, признаки затронутого окна неизвестны |
| Orders event time | `order_snapshots.order_us`, `firstAction.dateTimeUtc` | Не дата обновления статуса или оплаты |
| Publication | `sources.observed_us` из catalog `updated`, через `record_origins` | Публикация выгрузки, не момент покупки/перехода статуса |
| Snapshots | Все различающиеся fingerprints, исходные строки | Нет надёжной последовательности обновлений; победитель не выбирается |
| Identity | Текущий canonical merge graph | Версионированная доступность объединений в индексе отсутствует |

Основания: `customer_history_records.py` (AS_OF_UNVERIFIED,
ORDER_UPDATE_TIME_UNKNOWN), `customer_history_normalization.py` (времена,
fingerprints, source_line_key), `customer_history_index.py` (sources/origins),
`mindbox/identity.py` (текущий resolver). Применён scoped-аудит навыка
`data-analytics:analyze-data-quality`, без исследования всей базы.

Покрытие Actions и Orders непрерывно: `[2022-01-01, 2026-09-01)` UTC.
Публикации Actions: 12.10.2026 12:32:42.410980–14:19:13.652488 UTC;
Orders: 12:33:37.339153–14:19:13.652488 UTC; merges: 12:30:00.252230 UTC.
Все публикации позже допустимых исторических дат прогнозирования.

Сохранённые счётчики: raw Actions 10 112 202, raw Orders 1 365 775;
логические Actions 10 112 198, Orders 1 359 846; action variants 10 112 199,
order snapshots 1 361 399; идентичные повторы 4 379, конфликтующие события 1 554.
Конфликты владельца/времени/категории в метаданных равны нулю. Это не новый
профиль качества строк: полного прохода и integrity scan не было.

**Уровень B — RETROSPECTIVE_EVENT_TIME.** Point-in-time reconstruction недоступна;
требование A в config отклоняется. Текущие merges могут изменить принадлежность
старых событий. VIEW/FAVORITE используют event time; при отсутствии creation
первоначальная доступность не подтверждена. Ранний creation тоже не устраняет
ограничение поздней публикации.

Метка означает наличие заказа из целевого периода с согласованной покупочной
строкой в **наблюдаемом сейчас** наборе снимков. Не установлено, что переход
в покупочный статус произошёл именно в этом периоде. Это ретроспективный proxy
по времени заказа, не подтверждённый факт исторической оплаты/статусного перехода.
Нельзя выдавать такой эксперимент за production-оценку прогноза момента покупки.

## Контракты и полный список признаков

JSONL-строка: `customer_key` (HMAC-SHA256), `prediction_utc`, `target_until_utc`,
`split`, `features`, `available`, `feature_issues`, `label_state`, `target`,
`label_reasons`. Типизированные контракты: PropensityDatasetConfig, PropensityPlan,
PropensitySourceSpan, PropensityFeatureRow, PropensityLabel, PropensityProgress,
PropensityDatasetSummary. Времена — aware UTC, t — строго полночь.

23 признака (фигурные скобки раскрываются в отдельные имена):

| Имена | Смысл |
|---|---|
| `purchase_orders_{30,90,180,365}d` | Число логических заказов с подтверждённой покупочной строкой |
| `purchase_units_{30,90,180,365}d` | Точная сумма quantity купленных строк, JSON decimal string |
| `views_{7,30,90,180}d` | Число логических VIEW, не product occurrences |
| `favorites_{7,30,90}d` | Число логических FAVORITE |
| `views_active_days_30d` | Число календарных UTC-дней с VIEW |
| `favorites_active_days_30d` | Число календарных UTC-дней с FAVORITE |
| `days_since_purchase_365d`, `has_purchase_365d` | Давность/наличие покупки внутри 365 дней |
| `days_since_view_180d`, `has_view_180d` | Давность/наличие VIEW внутри 180 дней |
| `days_since_favorite_90d`, `has_favorite_90d` | Давность/наличие FAVORITE внутри 90 дней |

Интервалы признаков `[t-window,t)`. Давность — целое число полных суток.
`None` с `available=false` означает недостоверный признак; причины в feature_issues.
Для давности `None` с `available=true` означает наблюдаемое отсутствие события
внутри lookback; соответствующий `has_*` равен 0. Это не пожизненная давность.

Проверяется объединение интервалов покрытия каждого источника, включая разрывы
и неполные дни. Для Orders дополнительно требуется покрытие, опубликованное
не позже t; используются только варианты с observed_us <= t. Поздний снимок
не заменяет ранее известное состояние; будущая backfill-запись не меняет даже
индикаторы доступности при уже наблюдавшемся покрытии. Поздний/отсутствующий источник не позволяет
объявить исторический ноль даже без найденных заказов. На текущей реальной выгрузке
**все 10 признаков Orders будут недоступны** во всех исторических срезах.
Actions остаются исследовательскими event-time признаками.

CP/delivering/F — согласованное операционное определение покупки, не утверждение
об оплате. Статусы проверяются у товарных строк. Несколько снимков сопоставляются
по стабильному source_line_key и согласованному коду товара. Искусственный/повторный
ID не доказывает межснимочную идентичность. PURCHASE + Return сохраняет покупку;
return-only и cancellation-only не дают новой покупки; PURCHASE + cancellation,
unknown или несопоставимость дают неопределённость. Достоверная покупочная строка
доказывает существование покупки заказа, но другие неопределённые строки делают
сумму units неизвестной. Snapshots/товарные строки не увеличивают число заказов.
Невалидные/неположительные quantity не доказывают покупку. Decimal не проходит
через float; каноническое строковое представление формируется без округления.

Денежные признаки исключены: неизвестные историческое состояние статусов/валюта
и различия нетто/брутто затрудняют достоверный baseline. Цены не читаются SQL;
RUB/KZT не объединяются, неизвестные деньги не подменяются нулём.

## Цель, когорта, даты, разделение

`target=1` при доказанной покупке в `[t,t+30d)`; `0` при полном заявленном покрытии
Orders, отсутствии покупки и способной изменить ответ неопределённости.
UNKNOWN_LABEL, target=null: неполное будущее покрытие, неоднозначный владелец,
время, статус/снимки или отсутствующее происхождение потенциального заказа.
При неполном будущем окне даже positive получает UNKNOWN. При полном окне другая
неопределённость не отменяет доказанное существование покупки. Причины UNKNOWN
могут пересекаться, поэтому сумма счётчиков причин может превышать число строк.

Когорта: клиенты с достоверным interaction event time < t и согласованным canonical
owner. Список не берётся из нынешних Customers, KPI, профиля или BPR checkpoint
и не определяется будущими покупками. Действует B-ограничение текущих merges
и ретроспективного event time при определении состава исторической когорты.

Начало сетки по умолчанию — первая полная UTC-дата Orders + 30 дней, шаг 30 суток.
Конец ограничен max(until Orders)-30d, не сегодняшней датой. Дыры покрытия не
прячутся: соответствующие строки UNKNOWN. Максимум 256 срезов (Python config до
1024); требуется минимум три.

Для реальных метаданных: 55 срезов 31.01.2022–09.07.2026. Последняя допустимая
календарная дата 02.08.2026; следующий шаг сетки позже неё. Train 33 среза,
validation 11, test 11; границы 17.10.2024 и 12.09.2025. Число клиентов/строк
без прохода по событиям не выдаётся за установленный факт.

Границы split задаются явно либо выбираются около индексов 60% и 80% сетки:
train t < validation_start и target_until <= validation_start;
validation t >= validation_start, t < test_start и target_until <= test_start;
test t >= test_start. Пересекающие границу строки сохраняются в embargo.jsonl.
После purge в каждом разделе должен остаться потенциальный срез, иначе ошибка.
Наличие реальных строк/обоих классов в каждом разделе проверяется перед обучением.
Внутри split окна могут пересекаться при step_days < 30, что отмечается в manifest.
Повторные клиенты разрешены хронологически; балансировки классов нет.

## Чтение, ресурсы, хранение, свежесть

Два упорядоченных SQL-потока в одной read-only transaction: события/товарные строки
и отдельно агрегированные origins, без раздувания lines × origins. Группировка
по клиенту/событию; в Python только один logical event и фиксированные аккумуляторы
срезов текущего клиента. Aliases уже разрешены существующим индексом. Нет N+1 SQL
по клиентам/датам и нет чтения исходных JSON сборщиком.

Для E клиентских членств событий, R строк вариантов/товаров, O origins, S дат,
P выходных строк, C интервалов источников и фиксированного W окон верхняя оценка
Python O((E+R)*S*W + P*W*C*log(C+1)); второй член — проверки объединения покрытий.
На проверенных реальных метаданных C=4; при тысячах daily-компонентов эти повторные
проверки могут стать отдельным узким местом. SQL-сортировки в худшем случае
O((R+O) log(R+O)). Конфликт владельца может дать несколько memberships одного события.
RAM Python O(S*W + C + размер одного события), active-day sets <=30 дней на вид/срез.
Нет списка всех клиентов/событий или всех событий одного клиента. Tracemalloc-тест
проверяет reducer, не полный native RSS SQLite. Очень крупный заказ задаёт размер
текущего события. Используется существующий read helper: cache 8 МиБ, temp_store=FILE.

Сортировки могут потребовать значительного временного диска до первого результата.
Скорость/итоговый размер/RSS полного реального расчёта не измерены, ETA не обещается.
55 срезов при множестве клиентов дают большой JSONL; перед согласованным запуском
нужна отдельная оценка дискового запаса. Два прохода по output для проверки и хешей
не являются повторными проходами по истории.

Прогресс каждые 100 000 memberships и перед проверкой output; только агрегаты.
Отмена callback/CLI Ctrl+C проверяется в SQL handler, при чтении/выдаче строк,
валидации, хешировании и перед CURRENT. Автоматического лимита времени нет.
Резерв диска по умолчанию 1 ГиБ проверяется при выдаче строк, каждые 10 000 событий
и примерно раз в 2 секунды во время SQL. Проверяется том output; свободное место
SQLite temp на другом томе не гарантируется, ошибка SQLite прекращает сборку.

По умолчанию `filtered_data/customer_propensity/` уже игнорируется Git:

```
CURRENT.json
<content-version>/
    manifest.json
    train.jsonl
    validation.jsonl
    test.jsonl
    unknown.jsonl
    embargo.jsonl
```

Только три split-файла содержат известные обучающие target. Manifest содержит
схему/версию реализации, config/grid, source_version/generation/schema v2,
coverage/publication, семантику, as-of уровень, признаки, счётчики классов/
исключений/пропусков, распределения по датам и split, repeated customers/число
срезов/positive rate, SHA256 файлов и дату/продолжительность. Распределение по
датам позволяет исследовать сезонность без индивидуальных историй.

Стабильный секрет >=32 байт задаётся извне, его нельзя публиковать/хранить рядом
с датасетом. Customer IDs псевдонимизированы domain-separated HMAC-SHA256; manifest
содержит только непрямой key ID. Это псевдонимизация, не полная анонимность.

OS writer lock сериализует сборки output; shared canonical lock удерживается до
публикации. Source_version, версия файла SQLite, generation и схема проверяются
до чтения и перед публикацией без integrity scan. Потоковые счётчики logical events,
variants и клиентских связей сверяются с метаданными без дополнительного прохода.
Ошибки прочитанных вариантов/
SQL прекращают сборку; это не полная проверка всех ограничений SQLite.

Сначала собственный `.building-UUID`: JSONL → flush/fsync → проверка схемы/числа
строк → SHA256 → manifest → свежесть → атомарный rename → атомарный CURRENT
с SHA256 manifest. Частичный результат не становится текущим. Ошибка/отмена
очищает только собственный staging и сохраняет прежний CURRENT. После ошибки
между rename и CURRENT может остаться завершённая, но не выбранная версия;
после аварийного завершения возможен staging. Чужие каталоги не удаляются.
UNKNOWN-only сохраняет diagnostic-версию BLOCKED_BY_DATA_QUALITY без CURRENT.
Повторные сборки детерминированы по строкам/content version при тех же реализации,
источниках, config и ключе; timestamps не входят в identity. Совпавшая версия
повторно проверяется и используется без перезаписи.

## CLI и проверки

Только metadata preflight, без build:

```powershell
.venv312\Scripts\python.exe -B -X utf8 scripts/build_customer_propensity_dataset.py
```

Для будущего **отдельно согласованного** полного запуска (не выполнялся):

```powershell
.venv312\Scripts\python.exe -B -X utf8 scripts/build_customer_propensity_dataset.py --build --allow-retrospective --key-file <путь-к-секретному-бинарному-ключу>
```

Параметры: --raw-root, --cache-dir, --output-dir, --start, --end,
--validation-start, --test-start, --step-days, --min-free-gib. Даты ISO UTC,
например `2024-01-01T00:00:00Z`. Импорт не запускает build, API или обучение.
CLI выдаёт агрегаты/безопасную категорию ошибки и код 130 при отмене.
Использовать отдельным процессом, не в главном Qt-потоке.

Контрольная синтетическая выборка: 3 клиента × 3 среза = 9 наблюдений;
1 positive, 7 negative, 1 UNKNOWN (UNCERTAIN_ORDER_STATUS_OR_SNAPSHOT),
positive rate известных меток 1/8 = 12,5%. Отдельная сквозная проверка настоящих
adapters/canonical readers: позднее объединение alias, 3 строки, 1 positive,
2 negative, исходные JSON не меняются. Остальные проверки: границы/UTC,
отмены/возвраты/unknown, версии/строки, Decimal/quantity/RUB/KZT без денег,
покрытие/разрывы, окна/давность, invariance от будущего/профиля, embargo,
privacy/manifest/checksums, детерминизм, cancel/disk/source/generation/output
errors, metadata-only CLI и память reducer. Используются только synthetic fixtures.

Контракт готов к исследовательскому этапу CUSTOMER-PROPENSITY-02. Сначала нужны
согласованный реальный build и отчёт о фактических классах/UNKNOWN/пропусках,
достаточности split; preprocessing fit только по train. Для production нужны
историческая доступность identities/events, времена изменения статусов или иной
согласованный временной target, актуальные источники и проверка online freshness.
Будущий читатель должен сверять SHA256 manifest из CURRENT, checksums файлов,
schema/as-of/config и сравнивать source_version/generation с новым metadata inspect;
наличие CURRENT само по себе не доказывает актуальность после обновления источников.

</details>
