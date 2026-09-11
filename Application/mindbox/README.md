# Mindbox → raw JSON

Изолированный синхронный транспорт. Пакет не импортирует PyQt, pandas и модули
обработки файлов приложения. Импорт не читает `.env`, не создаёт HTTP-сессии,
не запускает запросы и не создаёт каталоги выгрузок.

## Локальная конфигурация

Команды выполняются из корня репозитория в PowerShell с Python 3.10:

```powershell
python -m pip install -r requirements-dev.txt
Copy-Item -LiteralPath .env.example -Destination .env
```

Копируйте шаблон только если своего `.env` ещё нет. Откройте `.env` локально
в редакторе и заполните `MINDBOX_SECRET_KEY`. Не передавайте ключ через
аргументы командной строки, исходники или сообщения. Остальные значения
четырёх операций уже указаны в шаблоне.

Обязательны все семь переменных из `.env.example`. Переменные окружения
имеют приоритет над файлом, включая пустое значение. Чтение не изменяет
`os.environ` и не подставляет `${...}` внутри значений. Поддерживается UTF-8 с BOM.
По умолчанию `.env` ищется в корне проекта, независимо от текущего каталога.
`MindboxConfig.from_env(None)` использует только environment.

`.env`, `.env.*` и `ВходныеДанные/MindboxRaw/` исключены из Git;
`.env.example` остаётся доступным для версионирования.

## Первый ручной запуск

```powershell
python scripts/mindbox_export_smoke.py actions
```

Для `actions` без дополнительных аргументов берётся последний час UTC,
округлённый до минут. Это правило только ручного скрипта; клиент не добавляет
периоды в payload. Для явного небольшого периода:

```powershell
python scripts/mindbox_export_smoke.py actions --since "2026-09-09 10:00" --until "2026-09-09 11:00" --poll-interval 5 --timeout 600
```

Замените даты нужным периодом. `--since` и `--until` задаются вместе.
Период передаётся полями `sinceDateTimeUtc` и `tillDateTimeUtc`.
В контракте действий это время создания действия в Mindbox; верхняя граница
не включается. Это может отличаться от времени совершения действия.
Источник: [экспорт действий клиентов](https://developers.mindbox.ru/docs/export-customer-actions).

Другие экспорты запускаются без автоматически добавленных периодов:

```powershell
python scripts/mindbox_export_smoke.py orders
python scripts/mindbox_export_smoke.py customers
python scripts/mindbox_export_smoke.py customer_merges
```

Для произвольных параметров операции предусмотрен `--payload-file` с путём
к локальному JSON-объекту. Он заменяет payload целиком, в том числе период
по умолчанию у actions. Совместное использование с `--since/--until` запрещено.
Другой конфигурационный файл можно передать через `--env-file`.

Ожидаемый вывод (идентификатор, число частей и пути будут другими):

```text
Mindbox export: actions
Operation: CustomerActionsAPI
Export started: 123456
Waiting for export...
Export ready. Parts: 2
Saved:
.../ВходныеДанные/MindboxRaw/actions/20260909_153500/actions_part_001.json
.../ВходныеДанные/MindboxRaw/actions/20260909_153500/actions_part_002.json
```

Скрипт возвращает 0 при успехе, 1 при ошибке, 130 при Ctrl+C.
Ссылки на скачивание и ключ не выводятся. Скрипт не подключён к CI.

## Использование в коде

```python
from Application.mindbox import MindboxClient, MindboxConfig

config = MindboxConfig.from_env()
with MindboxClient(config) as client:
    paths = client.export("actions", payload={
        "sinceDateTimeUtc": "2026-09-09 10:00",
        "tillDateTimeUtc": "2026-09-09 11:00",
    }, poll_interval=5, timeout=600)
```

Этапы можно вызывать отдельно: `start_export(operation, payload)` возвращает
строковый `exportId`; `get_export_status(operation, export_id)` проверяет
один ответ; `wait_for_export(...)` возвращает все URL после Ready;
`download_export(export_name, urls)` возвращает пути опубликованных файлов.
При сохранённом `exportId` можно продолжить ожидание без повторного старта.
`RawExportStorage(root=...)` позволяет явно выбрать корень хранения.

Контракт: POST `/v3/operations/sync`, параметры `endpointId`/`operation`,
JSON headers и `Authorization: SecretKey ...`. Проверка статуса отправляет
только `{"exportId": "..."}`. `NotReady` продолжает ожидание, `Ready`
проверяет список URL, `Cancelled` выбрасывает исключение с очищенной причиной.
Ошибочный `status` операции немедленно прекращает polling.
Источник: [общий принцип экспортов Mindbox](https://developers.mindbox.ru/docs/exports-overview).

## Надёжность и ограничения

- HTTP: по умолчанию 3 попытки, connect timeout 10 с и read timeout 30 с.
  Повторяются Timeout, ConnectionError, ошибки передачи/декодирования потока,
  HTTP 429 и 5xx. Паузы 1 и 2 с; `Retry-After` поддерживает секунды и HTTP-date,
  с общим ограничением паузы 60 с. Значения переопределяются в конструкторе.
- Прочие HTTP 4xx, ошибка TLS, ошибка операции, некорректный JSON или контракт
  завершаются без бессмысленных повторов. Redirect запрещён как для API,
  так и для файлов: ожидаются прямые HTTPS URL, полученные от Mindbox.
- Polling: интервал 5 с, бюджет 600 с. Дедлайн на `time.monotonic()` включает
  запросы, retry и паузы. Timeout запроса сокращается по оставшемуся бюджету;
  после его истечения новый запрос не запускается и поздний Ready не принимается.
- Скачивание: отдельный бюджет 600 с на часть, включая retry; поток читается
  блоками 64 КиБ. Повторное скачивание начинает временный файл заново.
  Дедлайны проверяются между блоками/запросами. Ограничения `requests` остаются:
  read timeout ограничивает бездействие сокета, а не жёстко прерывает весь вызов;
  DNS или медленно поступающий блок могут задержать возврат управления.
- Старт — POST. При потере ответа нельзя гарантировать ровно одну постановку
  задачи. Mindbox документирует `isDuplicate` для уже формируемого экспорта;
  клиент использует полученный `exportId`, но не обещает общей идемпотентности.
- API и скачивание имеют отдельные сессии. Ключ добавляется только в headers
  API-запроса. `.netrc` не подменяет авторизацию; стандартные proxy/CA настройки
  requests через environment сохраняются. Используйте контекстный менеджер
  или `close()`. Один экземпляр клиента не предназначен для совместного
  использования несколькими потоками.
- Ключ исключён из repr конфигурации. Диагностика операции и отмены очищается
  от ключа; HTTP-ошибки содержат код или класс причины без тела запроса,
  signed URL и небезопасной цепочки исключений requests. Пакет не пишет логи
  запросов, конфигурацию или ключ на диск.

## Gzip и атомарная публикация

Скачивание через `iter_content()` может уже снять `Content-Encoding: gzip`.
Далее storage проверяет именно первые байты файла: `1f 8b` означает оставшийся
gzip, который распаковывается потоково с проверкой целостности библиотекой gzip.
Иначе байты копируются без преобразования. Нет `json.load → json.dump`,
нормализации, смены кодировки или проверки структуры бизнес-JSON. Ответы API
разбираются как JSON только для управления экспортом. Пустые файлы отвергаются.

Весь запуск сначала пишется в `.<staging>` (имя начинается с `.staging-`)
внутри каталога выбранного экспорта. Каждая часть скачивается в `.download`,
распаковывается/копируется в `.json.tmp`, проходит flush/fsync и публикуется
через `os.replace()` в `.json` внутри staging. Только после всех частей
каталог целиком атомарно переименовывается в `YYYYMMDD_HHMMSS` на том же диске.
При коллизии секунды добавляется `_001`, `_002` и т.д.; mkdir резервирует имя
между процессами. Предыдущие успешные выгрузки не перезаписываются.

При исключении или Ctrl+C собственный staging удаляется насколько возможно.
После принудительного завершения процесса/отказа удаления могут остаться
`.staging-*` или `.reserve` каталоги: они не считаются успешной выгрузкой.
Атомарная видимость не равнозначна гарантии сохранности при отказе диска/питания.
На диске временно требуются транспортная и распакованная копии текущей части.

## Модули и проверки

- `config.py`: локальная конфигурация и соответствие четырёх имён операций.
- `client.py`: HTTP, retry, запуск, статусы, polling, скачивание и общий flow.
- `storage.py`: gzip, временные файлы и атомарная публикация каталога.
- `exceptions.py`: специализированные ошибки с общим `MindboxError`.
- `__init__.py`: публичные импорты пакета.

```powershell
python -m pytest tests/mindbox -q
python -m pytest
python -m ruff check .
```

Тесты используют синтетические данные, in-memory HTTP adapter и подменённые
часы. Обращения стандартного HTTP adapter к сети в этих тестах запрещены.
Проверяются четыре операции, ошибки HTTP/контракта, retry/deadline, gzip,
многосоставные экспорты, сбои публикации, одновременные сохранения, секреты
и ручной CLI через имитацию транспорта. Живого API-теста в pytest нет.

На следующий этап оставлены интерпретация raw JSON, DataFrame/CSV,
просмотры/избранное, объединения клиентов, статистика, рекомендации и UI.

## M02-01: профилирование фактической схемы

`schema_profiler.py` читает только локальные raw JSON. Он не вызывает API,
не читает `.env`, не импортирует PyQt/pandas и не изменяет исходные файлы.
Новых зависимостей нет. CLI работает из любой текущей папки:

```powershell
python scripts/mindbox_schema_report.py actions
python scripts/mindbox_schema_report.py orders
python scripts/mindbox_schema_report.py customers
python scripts/mindbox_schema_report.py customer_merges
python scripts/mindbox_schema_report.py all
```

Без параметров выбирается последний timestamp-каталог соответствующего типа
в `ВходныеДанные/MindboxRaw/`. Суффиксы `_001` и далее сравниваются численно.
`.staging-*`, `.reserve` и произвольные имена исключены. Последняя выгрузка
с пропущенными частями/неверным JSON вызывает ошибку: возврата к старой нет.
Part-файлы должны иметь последовательные номера начиная с 001.

```powershell
python scripts/mindbox_schema_report.py actions --input-dir "ВходныеДанные/MindboxRaw/actions/20260909_153500"
python scripts/mindbox_schema_report.py all --input-dir "ВходныеДанные/MindboxRaw"
```

Для одного типа `--input-dir` — непосредственно каталог с part-файлами.
Для `all` это общий корень с подкаталогами `actions`, `orders`, `customers`,
`customer_merges`, из каждого выбирается последний timestamp.

Результат — `<export_name>_schema.json` и `<export_name>_schema.md`
в `ВходныеДанные/MindboxReports/`, папка исключена из Git. Параметр
`--output-dir` меняет каталог отчётов; такой каталог следует самостоятельно
исключить из Git, если он расположен в репозитории. Отчёты нельзя размещать
внутри raw-каталога. Повторный запуск обновляет отчёты выбранных типов.
Каждый файл публикуется атомарно, но JSON/Markdown не являются одной
файловой транзакцией. В режиме `all` все входы анализируются до первой записи;
ошибка чтения любого экспорта не обновляет отчёты остальных типов.

JSON содержит `schema_version=1`, число файлов/объектов, `paths`,
`variable_paths`, `sections`, `custom_fields`, `identifier_namespaces`,
`action_templates` и счётчики действий без строкового systemName.

Семантика статистики:

- `present_records` и `presence_percent` относятся к уникальным исходным
  объектам экспорта. Десять позиций в одном заказе не дают десять заказов.
- `occurrences`, `type_counts` и `null_count` считают все вхождения path,
  включая повторяющиеся элементы вложенных массивов. `null_records` считает
  исходные объекты с хотя бы одним null по данному path. Отсутствие не равно null.
- Для корневого массива `scope=part_files`: знаменатель — число частей.
  Для остальных paths `scope=export_objects`: знаменатель — число объектов.
- Пустой массив присутствует; path элементов `[]` возникает только при
  наличии элементов. Для массивов сохраняются count, empty_count, min/max длина.
  Структура элементов находится по path с `[]`, включая вложенные массивы.
- `variation_reasons` показывает optional-поля, разные типы, различающиеся
  наборы ключей объектов и длины массивов. Неодинаковые типы/ключи элементов
  одного массива тоже считаются вариативностью. Это не классификация ошибок
  данных и не интерпретация бизнес-смысла полей.
- Типы: object, array, string, integer, number, boolean, null. Boolean не
  объединяется с integer. Неоднозначные ключи, например `a.b`, записываются
  через квадратные скобки с JSON-экранированием, отдельно от вложенного `a.b`.

Защита данных строится на разрешённом наборе: сохраняются имена полей,
paths, типы и счётчики. Scalar values не переносятся в отчёт. Единственное
исключение — строки строго по `customerActions[].actionTemplate.ids.systemName`.
Смысл этих имён не интерпретируется. Распределение merge method намеренно не
выводится: его безопасность как enum не предполагается. Сообщения об ошибках
не содержат исходный JSON, неожиданные root names или пользовательские пути.
Markdown экранирует HTML, разделители таблиц и переводы строк.
Имена JSON-ключей и systemName считаются разрешёнными метаданными согласно
контракту задачи; инструмент не может определить их смысл, если источник
сам помещает персональные данные в эти разрешённые места.

Каждый part загружается через стандартный `json.load`, обрабатывается и
освобождается до чтения следующего. Все части не объединяются в общий список.
Память зависит от крупнейшей части, числа paths/имён полей и уникальных
actionTemplate systemName. Пустой root-массив допустим; отсутствие part-файлов,
неверный root, не-объекты в root-массиве, duplicate keys, NaN/Infinity и слишком
глубокая структура завершают анализ понятной ошибкой. Контракт root: один
ключ выбранного экспорта с массивом объектов; лишние корневые поля не игнорируются.

```powershell
python -m pytest tests/mindbox/test_schema_profiler.py -q
```

## M02-02: domain adapters и canonical customer identity

Новый путь: `raw_reader.iter_export()` → отдельный adapter → frozen dataclass.
Адаптеры не обращаются к API, не сохраняют records на диск, не используют
pandas/PyQt и не подключены к приложению. Старая CSV-обработка не меняется.

- `records.py`: ProductKey, ActionRecord, OrderLineRecord, CustomerRecord,
  CustomerMergeRecord. Records имеют `repr=False`, чтобы обычный repr
  не раскрывал ID/профиль. Доступ к полям остаётся явным.
- `raw_reader.py`: общий код выбора timestamp, проверки part-файлов и JSON,
  выделенный из profiler. `iter_export(name, input_dir=...)` поддерживает явный
  каталог, `raw_root=...` — последний опубликованный экспорт выбранного типа.
  Читается одна часть за раз; дробные числа через Decimal без промежуточного float.
  Ошибки дают RawExportError без raw-значений. Генератор может выдать предыдущие
  корректные объекты до ошибки в следующей части: обработку нельзя считать
  успешной до полного завершения итерации.
- `schema_profiler.py` использует тот же reader, сохраняя SchemaProfileError,
  прежние float-типы и формат отчётов M02-01.
- `identity.py`: CustomerIdResolver строится из CustomerMergeRecord. Сначала
  собирает прямые связи, затем проверяет весь граф. Работает с транзитивностью,
  несколькими источниками, любым порядком событий и path compression.
  Customers export для resolver не нужен. Неизвестный ID разрешается в себя.
- `adapters/actions.py`, `orders.py`, `customers.py`, `customer_merges.py`
  отвечают каждый за свою сущность. `_common.py` содержит общие проверки.

Identity policy: повтор идентичной прямой связи допустим; разные прямые targets
одного source — ошибка даже при последующем схождении цепочек. Цикл, включая
self-merge, пустой список источников или некорректный ID — ошибка. Порядок
событий и дата не используются для произвольного выбора «последней» связи.
После создания граф не дополняется. Для обновлённого экспорта создаётся новый
resolver. Кеш растёт только с числом aliases, а не с неизвестными клиентами.
В records `source_customer_id` сохраняет исходный mindboxId, `customer_id`
содержит результат resolve — канонический ID.

Правила required/optional:

| Record | Обязательные данные |
|---|---|
| ActionRecord | ids.mindboxId, actionTemplate.ids.systemName, dateTimeUtc, creationDateTimeUtc, customer.ids.mindboxId |
| OrderLineRecord | order/customer mindboxId, firstAction.dateTimeUtc, channel externalId/name, lines; в каждой позиции id, number, quantity, basePricePerItem, priceOfLine, поддержанный product ID, status.ids.externalId |

M02-09A: Orders `product.ids` остаётся required identity; `product.name` — optional
metadata (`OrderLineRecord.product_name: str | None`). Отсутствие/null даёт None;
присутствующая непустая строка сохраняется без преобразования. Неверный тип,
пустая или whitespace-only строка по-прежнему вызывает AdapterError. Отсутствие
названия не меняет PURCHASE, quantity, resolution или completeness preparation.
Исторические результаты schema profiler M02-01 не изменяются.

InteractionDiagnostics.malformed_action_system_names — immutable snapshot counts
по техническим systemName mapped VIEW/FAVORITE без products. Counter обновляется
перед прежним InteractionBuildError; normal/unmapped actions в него не входят.
Snapshot передаётся в MindboxPreparationDiagnostics. Batch CLI prepare печатает
`Malformed action types` с systemName/count без event/customer/product IDs и PII.
Strict/diagnostic policy и completeness не изменены.
| CustomerRecord | ids.mindboxId; остальные поля профиля optional |
| CustomerMergeRecord | id, dateTimeUtc, resultingCustomer mindboxId и непустой mergedCustomers со всеми mindboxId |

Отсутствие/null у optional scalar даёт None, у optional массивов — пустой tuple,
у customFields — пустой immutable mapping. Непустые malformed optional values
вызывают AdapterError; пустая строка timestamp/birthDate/identifier тоже ошибка.
Optional текстовые значения сохраняются как есть, включая пустую строку.
Ошибки не приводят к silent drop. Пустой массив orders.lines допустим и даёт
пустой tuple; отсутствие/null lines недопустимо. adapt_order строит весь tuple
до возврата, поэтому ошибка позиции не выдаёт частично нормализованный заказ.
retailOrderId, totalPrice и deliveryCost optional; суммы и quantity — Decimal,
line_number — integer. Никакой бизнес-фильтрации отрицательных значений,
расчёта скидок или определения валюты нет.

ProductKey сохраняет namespace и полный ID. Orders допускает ровно один ключ
из offline1C/kanzlerKz; наличие обоих (даже если один null) — ambiguity error.
Нет поддержанного ID или его значение некорректно — ошибка. В actions.products
и productCategories поддержан offline1C. ID типа integer переводится в строку;
строки, включая ведущие нули, не преобразуются и не усекаются. Boolean, float,
пустые IDs и строки с краевыми пробелами отклоняются, а не исправляются.

Timestamps принимаются в форме YYYY-MM-DD[T или пробел]HH:MM:SS с optional
дробной частью до 6 знаков и Z/±HH:MM. Поля *Utc без offset трактуются как UTC
по контракту источника; явно заданный offset переводится в UTC. Результат —
timezone-aware datetime. Неверные дни/время/offset и лишняя точность вызывают
ошибку. birthDate — строго YYYY-MM-DD → date; отсутствие/null → None,
malformed значение не исправляется. Возраст не вычисляется.

Customers сохраняет профиль из CustomersAPI. customFields, lastActivatedCard,
segmentations, balances, subscriptions и discountCards представлены глубоко
неизменяемыми JSON-структурами: mappingproxy для объектов, tuple для массивов.
Неизвестные поля этих структур сохраняются. Это независимый снимок, изменение
исходного dict не меняет record. Вложенные значения не интерпретируются;
в частности status может быть строкой либо объектом в разных карточных структурах.
mobilePhone сохраняется строкой; для исходного integer используется только str(),
без математики или восстановления отсутствующего в raw знака «+».

Actions/Orders берут из вложенного customer только ids.mindboxId. Профиль из
этих копий не переносится. customerActions.order игнорируется при создании
покупок — единственный источник OrderLineRecord это OrdersAPI.
action_system_name хранится буквально, без substring heuristics, VIEW/FAVORITE/
CART, фильтрации маркетинга, weights или interactions.

Пример использования без персистентности:

```python
from Application.mindbox.raw_reader import iter_export
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.adapters import adapt_customer_merge, adapt_action

resolver = CustomerIdResolver(
    adapt_customer_merge(raw) for raw in iter_export("customer_merges")
)
for raw in iter_export("actions"):
    record = adapt_action(raw, resolver)
    # Следующий бизнес-слой будет добавлен отдельно.
```

Безопасная проверка локальных exports (нужен customer_merges даже для одного типа):

```powershell
python scripts/mindbox_adapter_smoke.py all
python scripts/mindbox_adapter_smoke.py actions
python scripts/mindbox_adapter_smoke.py all --raw-root "ВходныеДанные/MindboxRaw"
python -m pytest tests/mindbox/test_adapters.py tests/mindbox/test_identity.py -q
```

CLI выводит только counts merges/aliases/customers/actions/orders/order lines
и число канонизированных customer references. Для orders это число заказов,
а не позиций. В режиме одного типа остальные counts остаются нулевыми.
При ошибке возвращается код 1, тип экспорта, порядковый номер raw-объекта и
статический путь/причина без значения; частичные counts не выводятся.
Отдельные prepared-файлы и fixtures из production данных не создаются.

## M02-03: interaction classification

`Application/interactions.py` — отдельный business layer над готовыми
ActionRecord / OrderLineRecord. Он не читает raw, не выполняет merge resolution,
не импортирует pandas/PyQt и не добавляет weights. Использует существующий
ProductKey без изменения namespace/value; ID клиентов и даты переносит из records.

InteractionRecord — frozen dataclass с полями source_customer_id, customer_id,
product, interaction_type, event_datetime_utc, source, source_event_id и optional
quantity. Типы: VIEW/FAVORITE/PURCHASE; источники: ACTION/ORDER. repr не выводит IDs.

InteractionRules неизменяемы, наборы копируются в frozenset. Defaults:

- VIEW: только `ProsmotrProdukta` и `ProsmotrProduktaVApiMethod`.
- FAVORITE: `DobavlenieProduktaVSpisokVOperaciiDobavlenie` и
  `DobavlenieProduktaVSpisokVOperaciiDobavlenieTovara` (подтверждено в M02-04).
- PURCHASE statuses: `CP`, `delivering`, `F`.

Все сравнения точные, без substring/regex/изменения регистра/strip. Список
favorite_action_system_names можно переопределить явно.
Пересечение VIEW/FAVORITE mappings вызывает InteractionConfigError вместо
неявного выбора. Разрешённые статусы покупок тоже можно задать конфигурацией.

M02-04 фиксирует FAVORITE по бизнес-правилу прежней специализированной CSV-выгрузки
«Добавление товаров в избранное»: после фильтрации использовались именно два
указанных имени. `DobavlenieProduktaVSpisokVOperaciiUstanovka` остаётся unmapped,
в том числе не становится VIEW. `UstanovkaSpiskaProduktov`,
`UstanovkaSpiskaProduktovV`, `UdalenieProduktaIzSpiskaVOperaciiUdalenieTovara`,
`OchistkaSpiskaProduktov`, `OchistkaSpiskaProduktovV` также остаются unmapped.
Удаления и очистки не создают отрицательных interactions. VIEW, PURCHASE и
политика malformed VIEW без products не изменены.

`classify_action(action, rules)` отдельно возвращает VIEW/FAVORITE либо None.
`InteractionBuilder.from_action()` возвращает tuple взаимодействий одного action;
`from_order_line()` — PURCHASE либо None при отфильтрованном статусе.
`iter_interactions(actions, order_lines)` лениво обрабатывает сначала actions,
затем позиции заказов. Весь поток не собирается в список.

Политики построения:

- Одно классифицированное действие с N products создаёт N interactions.
  Каждое occurrence сохраняется, включая одинаковые ProductKey внутри одного
  action и повторные ActionRecords. Дедупликации/агрегации/сортировки нет.
- VIEW или FAVORITE без products — InteractionBuildError.
  Categories не заменяют products. Неклассифицированное действие, в том числе
  без products, просто учитывается в unmapped.
- PURCHASE создаётся только из OrderLineRecord с разрешённым статусом.
  source_event_id = line_id, quantity переносится без пересчёта,
  event_datetime_utc = order_datetime_utc. Отрицательная/нулевая quantity
  не фильтруется неявно. Действия оформления заказов не создают покупки.
- Для ACTION source_event_id = action_id, дата = event_datetime_utc действия,
  quantity = None. Используется дата события, а не creation_datetime_utc.

`builder.diagnostics` возвращает неизменяемый снимок счётчиков:
actions_total/view/favorite/unmapped/malformed, view_interactions,
favorite_interactions, order_lines_total/purchase/filtered_by_status,
purchase_interactions, total_interactions, unmapped_action_system_names.
Последнее поле содержит только разрешённые технические имена и counts.
Значения неизвестных line statuses не сохраняются в diagnostics.

Счётчики накапливаются в экземпляре: для новой выгрузки создаётся новый builder.
Они отражают обработанную часть входа и созданные records; для окончательного
итога нужно полностью потребить iterator. Внутри одного action interactions
создаются до первого yield. При malformed action учитываются total,
классифицированный тип и malformed, но interactions для него не создаются.
Один экземпляр builder предназначен для последовательного использования.

```python
from Application.interactions import InteractionBuilder, InteractionRules

rules = InteractionRules()
builder = InteractionBuilder(rules)
for interaction in builder.iter_interactions(action_records, order_line_records):
    pass  # Следующий бизнес-слой появится в отдельной задаче.
stats = builder.diagnostics
```

Безопасный локальный smoke использует raw reader, resolver и adapters M02-02:

```powershell
python scripts/mindbox_interaction_smoke.py all
python scripts/mindbox_interaction_smoke.py all --raw-root "ВходныеДанные/MindboxRaw"
python -m pytest tests/test_interactions.py -q
```

Нужны локальные exports customer_merges, actions и orders. Customers export
для классификации не требуется. CLI использует default rules.
После полного успеха печатает только агрегированные counts, включая число
заказов (отдельно от позиций). При ошибке — код 1, тип источника, порядковый
номер raw-объекта и безопасная причина; частичные totals не выводятся.
Ни API-запросов, ни записи prepared dataset, ни подключения к BPR-MF/UI нет.

Для полного диагностического прохода при malformed VIEW/FAVORITE без products:

```powershell
python scripts/mindbox_interaction_smoke.py all --diagnose
```

Этот явный режим продолжает после InteractionBuildError, показывает
`Malformed actions` и полные counts остальных событий/позиций. При наличии
malformed events возвращается код 1 и предупреждение: TOTAL относится только
к взаимодействиям из корректных событий. Это не успешный prepared dataset.
Required-field ошибки адаптеров, raw reader и resolver по-прежнему останавливают
проверку. Правила builder и default fail-fast поведение не ослабляются.

## M02-05: product identity / catalog resolver

`Application/product_resolution.py` отделяет исходные ProductKey от catalog identity.
`load_catalog()` читает `ВходныеДанные/Номенклатура.csv` стандартным csv module:
UTF-8 с optional BOM (`utf-8-sig`), разделитель `|`, обязательный уникальный header
`КодНоменклатуры`. Ошибки файла, кодировки, CSV, header и ширины строки вызывают
безопасный CatalogError. Пустые/whitespace-only коды и пустые строки не создают item
и учитываются в empty_code_rows. Повторные коды считаются в duplicate_code_rows
(каждая строка после первого появления), но дают один canonical ID, независимо
от других колонок. Файл не изменяется. Непустой код с outer whitespace вызывает
ошибку: значимые ID не нормализуются молча. Каталог только с header допустим;
любой product в нём останется unresolved.

`ProductResolver.resolve(ProductKey)` применяет историческое правило **только**
для точных namespaces `offline1C` и `kanzlerKz`: `candidate = product.value[:6]`.
Resolution успешен только при наличии candidate в catalog.item_ids. ID короче
6 символов проверяется целиком, без padding. Full-ID matching не заменяет prefix-6,
даже если длинный full ID присутствует в каталоге. Регистр, ведущие нули и внутренние
символы сохраняются; числового преобразования нет. Пустой, нестроковый ID или ID
с outer whitespace получает invalid_id; неизвестный namespace — unsupported_namespace
без усечения. Отсутствующий candidate — unknown_candidate, без fallback на full ID.

Default `strict=True`: ProductResolutionError с безопасным статусом при любой
неудаче. При `strict=False` метод resolve возвращает ProductResolution со статусом
и `item_id=None` при неудаче. Он не изменяет interaction diagnostics.

`resolve_interaction(interaction, strict=True)` возвращает immutable
`ResolvedInteraction(interaction, item_id)`, сохраняя **тот же** InteractionRecord:
полный ProductKey, customer IDs, дату, type, source, source event ID, quantity.
При неудаче strict mode обновляет diagnostics и вызывает специализированную ошибку;
diagnostic mode (`strict=False`) обновляет diagnostics и возвращает None, который
не является готовым downstream interaction. Никакой user-item агрегации,
дедупликации, weights или изменения temporal split нет. Categories не передаются
resolver. Обычный repr каталога, resolution и resolved interaction не показывает ID.

Diagnostics — immutable snapshot на обработанные вызовы resolve_interaction:
total и разрезы by_type (VIEW/FAVORITE/PURCHASE), by_namespace
(offline1C/kanzlerKz/безопасная группа unsupported). Каждый содержит
interactions_total, resolved, unresolved, unsupported_namespace, resolution_rate_percent.
Unsupported входит в unresolved: total = resolved + unresolved. При нуле событий
rate = 0%. Неизвестные имена namespaces не выводятся.

Collapse diagnostics считают уникальные пары (namespace, full value):
unique_source_product_keys включает все рассмотренные keys, в том числе unresolved;
unique_resolved_catalog_items — только успешно найденные items;
catalog_items_with_multiple_source_keys — items с более чем одним уникальным key;
max_source_keys_per_catalog_item — максимальное число таких keys (0 без resolved).
Повторные события одного key увеличивают event counts, но не collapse counts.
Это диагностика many-to-one legacy policy, а не ошибка identity. Resolver хранит
множества уникальных keys, а не весь поток событий; память пропорциональна числу
уникальных товаров. Для нового последовательного прохода нужен новый экземпляр.

```python
from Application.product_resolution import ProductResolver, load_catalog

resolver = ProductResolver(load_catalog())
resolved = resolver.resolve_interaction(interaction)  # strict по умолчанию
stats = resolver.diagnostics
```

Локальный CLI читает только raw exports customer_merges/actions/orders и каталог:

```powershell
python scripts/product_resolution_smoke.py --diagnose
python scripts/product_resolution_smoke.py --raw-root "ВходныеДанные/MindboxRaw" --catalog "ВходныеДанные/Номенклатура.csv"
python -m pytest tests/test_product_resolution.py -q
```

Без --diagnose первая ошибка останавливает проход без частичных totals. С --diagnose
известные malformed VIEW/FAVORITE без products учитываются прежним builder и
пропускаются; unresolved products также считаются, но не превращаются в resolved
interactions. Любой malformed или unresolved даёт exit 1 и явное предупреждение.
Ошибки каталога/raw/adapters/customer identity по-прежнему останавливают проход.
При успешном проходе exit 0; при прерывании 130. Вывод содержит только безопасные
агрегаты, безопасные причины ошибок нового слоя и имена типов upstream ошибок,
без ID/PII, значений неизвестных namespaces и путей.
CLI не выполняет API-запросов, не создаёт CSV/Parquet/prepared dataset и не подключает
результат к BPR-MF, PyQt или старому CSV processing.

## M02-06: BPR preparation parity boundary

`Application/model/bpr_preparation.py` — независимый слой над ResolvedInteraction.
Он использует уже установленные pandas/NumPy для совместимой числовой обработки,
не импортирует BPRMF, PyQt, files_processing или Mindbox API/schema. Production
training flow не переключён. Никаких CSV/Parquet/prepared datasets не создаётся.

`to_bpr_event()` создаёт один immutable BprEvent на каждый resolved interaction:
canonical customer_id, catalog item_id, исходный timezone-aware timestamp,
interaction_type и float weight. Full ProductKey, source customer/event ID не
переносятся. repr событий, mappings и splits не раскрывает IDs.

Immutable BprWeightConfig: VIEW=0.1, FAVORITE=2.0, PURCHASE=10.0;
min/max purchase quantity=1.0/10.0. PURCHASE проходит legacy последовательность
`pd.to_numeric(errors='coerce') -> fillna(1) -> float -> clip(1, 10) -> *10`.
Нет округления: None/0/1 дают 10, 1.5 даёт 15, 2 даёт 20, 15 даёт 100.
Quantity VIEW/FAVORITE игнорируется. Конфигурация требует конечные положительные
веса и корректные границы. Значения defaults остаются и в TrainConfig;
regression test сравнивает их непосредственно, а quantity tests проверяют legacy
hardcoded clip bounds через private prep functions. Это защита от drift без
runtime импорта тяжёлого BPRMF и без изменения его production config.

`prepare_bpr(events, BprPreparationConfig())` возвращает mappings, splits и
безопасные diagnostics. Сначала события сохраняются отдельно, без дедупликации.
Порядок внутри каждого interaction type сохраняется из входного iterable:

- users: first occurrence в PURCHASE -> VIEW -> FAVORITE;
- items: first occurrence в PURCHASE -> FAVORITE -> VIEW;
- event row order: PURCHASE -> FAVORITE -> VIEW.

IDs не сортируются. Это различные порядки, соответствующие _build_mappings и
_collect_user_item_events существующего BPRMF. Категорий на этой границе уже нет.

Default date_mode=LEGACY_DATE: исходный timestamp остаётся в BprEvent, но для split
используется календарная дата UTC (00:00). FULL_TIMESTAMP явно сохраняет точное
время UTC при split; новый слой не принимает naive/missing timestamps. Legacy
поддерживает NaT, но корректный typed interaction domain их не создаёт.
min_user_interactions_for_eval=10, также проверяется против TrainConfig.

Split: сортировка по u_idx, timestamp и стабильному row_id; event count определяет
eligible users (>=10). Последнее датированное событие каждого eligible user уходит
в evaluation; только после его удаления train events группируются по (u_idx, i_idx)
и weights суммируются pandas groupby.sum с legacy float precision и порядком пар.
Выход: user2idx/idx2user/item2idx/idx2item, train_pairs, train_weights,
eval_users/eval_items и user_pos_train. Память пропорциональна числу событий;
это batch preparation для temporal split, не streaming aggregation.

Parity tests строят эквивалентные synthetic CSV-like DataFrames и resolved events,
вызывают реальные private functions BPRMF и сравнивают mappings, event weights,
eval targets, train pairs/weights и user_pos_train. Проверяются 9/10/13 событий,
несколько users/items, повторы, mixed types, одинаковые timestamps, стабильность
порядка, fractional/missing/clipped quantity, custom weights и UTC chronology.

**DATE vs FULL_TIMESTAMP:** старый _parse_interaction_date действительно вызывает
dt.normalize(). Characterization test: 9 просмотров в 08:00 и покупка другого item
в 22:00 того же дня. LEGACY_DATE выбирает последний VIEW из-за порядка типов;
FULL_TIMESTAMP выбирает PURCHASE. Полное время меняет evaluation semantics, поэтому
FULL_TIMESTAMP — только opt-in для анализа, default сохраняет legacy date behavior.
Этот выбор явно предусмотрен заданием M02-06; само обучение не изменено.

Граница parity — одинаковые canonical events и одинаковый порядок строк внутри
каждого типа. Старые process_* дополнительно sort_values по дате перед записью CSV,
без явного stable sort; raw API exports могут иметь иной порядок. Тождественность
независимых CSV/API выгрузок (в том числе tie order и timezone календарной даты)
не доказана этими тестами. При migration нужно отдельно сверить порядок/часовой пояс
источников; текущий слой не меняет время на локальную зону и не угадывает CSV row order.

**Known evaluation issue:** item последнего события может остаться в user_pos_train
из более раннего события того же item. Synthetic characterization test сохраняет
этот эффект. Исправление evaluation design отложено до отдельной задачи после parity.

**Catalog duplicates:** текущие 45436 rows / 44598 codes, 1 empty и 837 duplicate rows
не размножают BPR events, поскольку resolver использует уникальные IDs. Перед migration
item side-features нужен отдельный анализ conflicting duplicate catalog rows;
он не входит в M02-06.

```powershell
python scripts/bpr_preparation_smoke.py --diagnose
python scripts/bpr_preparation_smoke.py --diagnose --date-mode FULL_TIMESTAMP
python -m pytest tests/model/test_bpr_preparation.py tests/model/test_bpr_preparation_smoke.py -q
```

CLI читает локальные raw exports и catalog через существующие слои. --diagnose
продолжает после malformed upstream actions и unresolved products, показывает
полную статистику корректных событий и возвращает 1 при любой такой проблеме.
Без --diagnose первая ошибка останавливает проход без частичных totals. Ошибки
raw/adapters/catalog/customer identity не подавляются. Выводятся только counts:
events по типам, unique users/items, eligible users, eval events, train events до
агрегации, train pairs после неё, total train weight, upstream malformed/unresolved.
API-запросов, обучения и вывода ID/PII нет.

## M02-07: prepared training input boundary

`Application/model/training_data.py` содержит общие Mappings, Splits,
PreparedBprData(mappings, splits) и validate_prepared_data. BPRMF.Mappings/Splits
остаются совместимыми re-exports. BprMappings/BprSplits в bpr_preparation также
являются aliases; BprPreparation наследует PreparedBprData и добавляет diagnostics.
Таким образом, результат M02-06 можно передать напрямую в training core в памяти.

Legacy production flow: `_train_in_this_process(cfg)` устанавливает seed,
вызывает `prepare_training_data_from_csv(cfg)`, затем
`train_prepared_data(cfg, prepared, device)` и прежнюю публикацию artifacts.
CSV preparation выполняет проверки файлов/header, чтение трёх interaction CSV,
старые mapping/weight helpers и temporal split. Сообщения об отсутствующих файлах,
пустых events и ошибках header сохраняются; технические ошибки не подавляются.
Private prep helpers остаются. `train_bprmf(maps, events, cfg, device)` сохранён
как совместимая обёртка: старый split -> PreparedBprData -> новый core.
Прямой запуск BPRMF.py из legacy CLI поддерживается импортом sibling contract.

Prepared path: `train_prepared_data(cfg, prepared, device)` сначала валидирует
контракт, затем использует готовые weights/splits без пересчёта. Ни наличие, ни
чтение Заказы/Просмотры/Избранное.csv ему не нужны. Инициализация seed остаётся
обязанностью вызывающей стороны, как у прежнего train_bprmf. TrainConfig weights
сохраняются для legacy preparation/UI; они не заменяют веса prepared input.
В будущем orchestration явно передаст выбранные веса в BprWeightConfig.

Validation проверяет обратимость и непрерывность mapping indexes, размеры Nx2/1D,
целочисленные train/eval indexes и их диапазоны, длины weights/eval arrays,
конечные неотрицательные weights, наличие user_pos_train для каждого user и его
соответствие train pairs. Нулевые веса допускаются текущим loss. Пустой train set
вызывает понятный PreparedDataError до инициализации training. Никакого repair,
изменения индексов или фильтрации ошибочных записей нет. Ошибки не раскрывают IDs.

**Mutability policy:** shared contracts намеренно mutable для совместимости с
существующими list/dict/set/NumPy consumers. Arrays/containers хранятся по ссылке,
без defensive copy и write-protect; полной immutability не обещается. Владелец
обязан не изменять данные во время обучения и не разделять их с writer-потоком.
Core не меняет вход; validation выполняется перед каждым запуском. Тест показывает
aliasing и повторное отклонение данных после изменения weights на NaN.

Item feature loader не изменён: при use_item_features=True он может читать только
Номенклатура.csv по canonical mappings; при False prepared training работает без
любых CSV. Model/loss/negative sampling/evaluation/early stopping/publication
не изменены. LEGACY_DATE остаётся default M02-06, known evaluation overlap сохранён.
Mindbox production wiring не добавлен.

Тесты сравнивают CSV preparation со старыми helpers, проверяют новую orchestration
границу, validation и backwards aliases. Короткое двухэпоховое CPU-обучение через
старую обёртку и prepared core при одинаковом seed даёт совпадающие splits и близкие
параметры модели (torch.testing.assert_close). Тесты отключают повторную глобальную
настройку interop threads для двух запусков в одном процессе; production policy
потоков не меняется. Все CSV fixtures синтетические в tmp_path; artifacts не пишутся.

Существующий bpr_preparation_smoke дополнен validate_prepared_data и строкой
`Prepared input validation: OK`. Новый CLI не нужен. Smoke не вызывает обучение,
показывает агрегаты и по-прежнему возвращает 1 при четырёх malformed upstream VIEW.

## M02-08: Mindbox orchestration

Public API в `Application/model/mindbox_training_preparation.py`:

```python
result = prepare_training_data_from_mindbox(
    actions_export_dir=actions_directory,
    orders_export_dir=orders_directory,
    customer_merges_export_dir=merges_directory,
    catalog_path=catalog_file,
    train_config=cfg,
    diagnose=False,
)
```

Library требует конкретные export directories, не выбирает latest. None/пустой
путь отклоняется. Вызывающая сторона отвечает за согласованность периодов/запусков
этих exports: наличие explicit paths само по себе её не доказывает.
Сначала полностью строится CustomerIdResolver, затем адаптируются Actions/Orders.
Customers export и КатегорииСайта.csv не читаются. Product identity, классификация,
quantity и ordering делегированы существующим слоям, business rules не копируются.

TrainConfig передаётся по структурному Protocol (library не импортирует тяжёлый
BPRMF): w_view_item -> view_weight, w_favorite -> favorite_weight,
w_purchase -> purchase_weight, min_user_interactions_for_eval -> eval threshold.
BprPreparationConfig явно получает LEGACY_DATE. FULL_TIMESTAMP здесь недоступен.
Внутри каждого типа сохраняется исходный порядок raw; grouping/mappings/holdout
остаются обязанностью M02-06, без новых сортировок по IDs/systemName.

По умолчанию strict: malformed item action вызывает InteractionBuildError,
unresolved/unsupported ProductKey — ProductResolutionError. При diagnose=True
только эти recoverable случаи считаются и исключаются из дальнейшей подготовки;
результат имеет complete=False. Raw JSON/структура/adapter/merge conflict/cycle/
catalog/invalid prepared data остаются фатальными в обоих режимах. Если после
исключений train set пуст, validation вызывает PreparedDataError, результата нет.
Unmapped actions учитываются, но сами по себе не делают результат incomplete:
они намеренно находятся вне подтверждённой interaction classification.

Особенность существующих adapters: неизвестные raw product namespaces отклоняются
ещё как AdapterError (Actions сейчас поддерживает offline1C). Это structural error,
а не recoverable resolver failure. Unsupported typed ProductKey, достигший resolver,
обрабатывается strict/diagnostic policy; тест проверяет эту границу отдельно без
изменения adapters.

MindboxPreparationResult — frozen wrapper с prepared_data, безопасными diagnostics
и complete. Сам PreparedBprData сохраняет controlled mutability M02-07; wrapper не
обещает глубокую immutability массивов. repr result не показывает IDs. Diagnostics
композируют существующие resolution/BPR snapshots и безопасные interaction counts;
unmapped systemName values в них не включаются. Ничего не сериализуется на диск.
complete=False необходимо явно учитывать будущему caller: это не полноценный
production dataset. Training core сам происхождение/complete не проверяет.

CLI (только локальная preparation + validation, без обучения):

```powershell
python scripts/mindbox_training_preparation_smoke.py --diagnose
python scripts/mindbox_training_preparation_smoke.py --actions-export-dir "..." --orders-export-dir "..." --customer-merges-export-dir "..." --catalog "..."
```

Только CLI может выбирать latest для неуказанных каталогов через существующий
selector. --raw-root меняет корень convenience поиска. CLI использует TrainConfig
defaults и печатает только агрегаты; complete=False даёт exit 1 и предупреждение,
complete=True — 0. Фатальная ошибка даёт 1 без частичных totals и без сырых значений
исключения. Ни API-запросов, ни model artifacts, ни prepared файлов нет.

Synthetic integration tests: полный raw-vs-processed-CSV parity по всем mappings/
splits, равные даты разных типов, repeated items, >10 событий, quantity=2.5;
отдельно custom weights 0.7/3.5/6 и eval threshold=4. Merge A->B до mappings —
ожидаемое улучшение customer identity, а не CSV parity case. Проверены fatal и
recoverable ошибки, безопасный CLI/complete, и raw -> prepared -> training core
на один CPU epoch с выключенными features, без публикации artifacts.
Production UI/CLI training source остаётся legacy CSV; wiring от API экспорта
до explicit directories и orchestration будет отдельной задачей.

## M02-09: coordinated training batch

`Application/mindbox/training_batch.py` фиксирует связанный набор exports:
TrainingBatchWindow, TrainingBatchExport и MindboxTrainingBatch — frozen records.
Batch/export repr скрыт. Window содержит timezone-aware UTC interaction_since,
interaction_until и обязательный merge_since: `merge_since <= since < until`.
Окна полуоткрытые [since, until), для merges — [merge_since, until).
Как в M01 smoke transport, payload dates имеют формат YYYY-MM-DD HH:MM;
subminute input отклоняется вместо неявного усечения. Offset-aware input приводится
к UTC. Нет скрытого lookback/default merge_since.

`create_training_batch(client, raw_root=..., window=...)` последовательно вызывает
существующие start_export/wait_for_export/download_export для customer_merges,
actions, orders. Actions/Orders получают одинаковый payload периода; merges —
свой since и общий until. Export IDs берутся из start_export. ExportOutput и
exportId в initial payload не добавляются. Retry/polling/gzip/raw atomic storage
не изменены. Customers не экспортируется.

После успешной публикации всех raw exports создаётся
`training_batches/<uuid hex>/manifest.json`. Только manifest означает опубликованный
batch; пустой каталог после ошибки публикации batch не представляет. При API/download
failure manifest отсутствует, готовые raw artifacts не удаляются. Публикация:
exclusive UUID directory -> manifest.tmp -> flush -> fsync -> os.replace(manifest.json).
При ошибке временный manifest удаляется; raw не откатываются. Повторный запуск создаёт
новый batch, а не перезаписывает предыдущий. Это атомарная видимость manifest;
содержимое raw не хешируется и filesystem artifacts должны оставаться неизменными.

Manifest schema v1 — только metadata: schema_version, batch_id, created_at_utc,
window и exports (name/export_id/operation/relative_directory/parts_count).
Три ссылки относительны raw root: `<export type>/<timestamp>`.
URLs, exportResult, headers, credentials, raw fragments и IDs клиентов/товаров
не передаются serializer. Operation names и export IDs — разрешённая metadata.

`load_training_batch(manifest_path, raw_root=...)` и `validate_training_batch(batch,
raw_root)` проверяют version, required fields, уникальные JSON keys, batch ID,
периоды, ровно три различных ссылки, тип каталогов, границы resolved paths,
наличие и последовательность parts и совпадение parts_count. Absolute paths,
Windows drive/UNC/backslash и traversal запрещены. Symlink targets за raw root
отклоняются. Raw JSON содержимое не читается: это следующий raw_reader этап.
Manifest — локальная запись происхождения; validator не может доказать, что
кто-то вручную не изменил metadata периода или raw files после публикации.

`prepare_training_data_from_batch(batch, raw_root=..., catalog_path=...,
train_config=..., diagnose=False)` валидирует batch и делегирует M02-08 с точными
directory paths. Library никогда не ищет latest и не копирует preparation logic.
Transport-complete batch может дать training-data complete=False из-за malformed
VIEW/unresolved products. Это разные состояния; prepare не запускает обучение.

CLI `scripts/mindbox_training_batch.py`: export — единственная live-команда,
читает .env; validate/prepare полностью offline, не читают .env. Ошибки выводят
только тип исключения. Export печатает batch ID, manifest path и counts;
prepare — безопасные aggregates и training-data complete flag (False -> exit 1).
Credentials и download URLs не логируются.

Пример самостоятельного запуска в PowerShell из корня проекта:

```powershell
& .\.venv310aboba\Scripts\python.exe scripts/mindbox_training_batch.py export `
  --since "2026-08-01 00:00" `
  --until "2026-09-01 00:00" `
  --merge-since "2025-01-01 00:00"
```

merge-since в примере выбран явно: caller должен указать подходящее начало истории
merges для своих данных. После успеха использовать напечатанный manifest path:

```powershell
python scripts/mindbox_training_batch.py validate --manifest "ВходныеДанные/MindboxRaw/training_batches/<batch_id>/manifest.json"
python scripts/mindbox_training_batch.py prepare --manifest "ВходныеДанные/MindboxRaw/training_batches/<batch_id>/manifest.json" --diagnose
```

Тесты используют mocked API и настоящий M01 raw storage на синтетических байтах:
окна, последовательность, multipart, failure без manifest, fsync/replace, загрузка,
повреждённые metadata/paths, bridge exact dirs и отсутствие synthetic secrets в
manifest/CLI. Live API во время реализации не вызывается. Старые exports из разных
окон не маркируются coherent batch задним числом; новый batch создаёт caller явно.

## M02-10: daily chunked / resumable batches

Новый `daily_training_batch.py` и отдельный CLI `scripts/mindbox_daily_batch.py`
расширяют M02-09, не заменяя single-window API и CLI. `split_daily_windows(since,
until)` — pure function: UTC-aware даты приводятся к UTC, требуют midnight alignment
и since < until, возвращают последовательные полуоткрытые сутки без gaps/overlaps.
Offset-aware значения допустимы только если обозначают UTC midnight. Произвольные
minute-aligned окна остаются в прежнем single-window API.

`create_chunked_training_batch(client, raw_root=..., window=TrainingBatchWindow(...))`
создаёт batch ID и durable state до первого API request. Один batch-level merges
export использует [merge_since, interaction_until). Затем строго последовательно:
Actions(day1), Orders(day1), Actions(day2), Orders(day2), ... . День — логический
сегмент из двух компонентов; каждый export может иметь любое число parts >=1.
Parts скачивает/распаковывает/публикует существующий M01 client/storage.

Schema v2 хранит created UTC, batch ID, общее window, fingerprint endpoint/operations
и упорядоченные components: name, since, until, operation, status и optional export
metadata с относительным directory/parts_count. Компоненты идут merges, затем пары
actions/orders на каждый день. Final records frozen; snapshots состояния создаются
через replace. READY образуют последовательный prefix; PENDING/FAILED не имеют
export reference. Loader проверяет точное покрытие дней, порядок/периоды, статусы,
JSON duplicate keys, schema, batch identity/path, metadata и все READY part references.
Validation не читает содержимое raw parts. Paths проверяются относительно raw root
по правилам M02-09, включая traversal и resolved symlink boundaries.

`training_batches/<batch_id>/state.json` обновляется после каждого подтверждённого
READY: новый temp -> flush -> fsync -> os.replace. Ошибки компонента сохраняют FAILED;
process kill оставляет последний durable checkpoint, незавершённый компонент может
остаться PENDING. Нет удаления/rollback опубликованных raw exports. Temp files,
оставленные аварийным завершением, не блокируют resume: следующая запись использует
новое уникальное temp name. OS writer lock одного batch автоматически освобождается
при завершении процесса; существование .writer.lock само по себе не означает lock.
Один batch не может иметь двух concurrent writers. Разные batches caller должен
запускать последовательно с учётом общего лимита Mindbox.

`resume_chunked_training_batch(client, state_path=..., raw_root=...)` загружает периоды
из state, повторно проверяет READY references и продолжает только PENDING/FAILED.
API URL/endpoint/operations должны совпадать с сохранённым fingerprint; SecretKey
не хешируется/не сохраняется, поэтому его ротация разрешена. Изменение общего периода,
нарушающее coverage компонентов, отклоняется. State — доверенная локальная metadata,
не криптографически подписанный документ: согласованную ручную подмену всех периодов
loader доказать не может. Не редактировать state вручную.

Ограничение resume: server-side exportId не checkpoint-ится до локальной публикации.
После timeout незавершённый export может быть создан заново; предыдущий server-side
job может ещё выполняться, и caller должен учитывать сервисный лимит. Падение между
raw publication и durable READY также оставляет неподтверждённый компонент, который
будет повторён; orphan raw artifact сохраняется. Уже checkpointed READY никогда не
экспортируется повторно. Повреждённый/пропавший READY artifact вызывает ошибку вместо
скрытой повторной выгрузки.

manifest.json появляется атомарно только после всех READY и имеет
transport_complete=True. state сохраняется как audit с READY-компонентами;
manifest — commit marker. При сбое финализации resume публикует manifest без новых
exports. Повторный resume finalized batch возвращает тот же manifest без API calls.

`prepare_training_data_from_chunked_batch(...)` требует final contract и вызывает
общую внутреннюю path-sequence boundary M02-08. Старый public API M02-08 оборачивает
singleton dirs. Читаются ВСЕ actions directories в порядке дней, ЗАТЕМ ВСЕ orders
directories; внутри каждого сохраняется part order. Физического склеивания JSON нет.
Один CustomerIdResolver, InteractionBuilder, ProductResolver и BPR preparation на
весь batch обеспечивают общие mappings/split и накопительные diagnostics. Добавлен
raw orders count; malformed_action_system_names суммируется прежним builder.
LEGACY_DATE, weights, eval threshold, aggregation и known evaluation issue сохранены.

Transport complete и training-data complete независимы: recoverable malformed
actions не мешают экспорту, но diagnostic preparation вернёт complete=False.
Quality gate и shadow/production training здесь не добавлены.

CLI команды: export-daily/resume — LIVE; status/validate/prepare — offline. Только
live-команды читают .env. При прерывании экспортов CLI печатает batch ID, state path,
последний завершённый component и текущие статусы, без response text/URLs/IDs/PII.
Если checkpoint ещё не удалось записать, CLI явно сообщает, что state недоступен.
Status показывает дни/READY/FAILED и читает final manifest при его наличии.
Prepare выводит cumulative safe diagnostics, но не обучает и не пишет prepared files.

Первый ручной live test на три дня (PowerShell из корня проекта):

```powershell
& .\.venv310aboba\Scripts\python.exe scripts/mindbox_daily_batch.py export-daily `
  --since "2026-08-01" `
  --until "2026-08-04" `
  --merge-since "2025-01-01 00:00" `
  --timeout 3600
```

После вывода batch ID подставить его в путь:

```powershell
$batchDir = ".\ВходныеДанные\MindboxRaw\training_batches\<batch_id>"
& .\.venv310aboba\Scripts\python.exe scripts/mindbox_daily_batch.py status --state "$batchDir\state.json"
& .\.venv310aboba\Scripts\python.exe scripts/mindbox_daily_batch.py resume --state "$batchDir\state.json" --timeout 3600
& .\.venv310aboba\Scripts\python.exe scripts/mindbox_daily_batch.py validate --manifest "$batchDir\manifest.json"
& .\.venv310aboba\Scripts\python.exe scripts/mindbox_daily_batch.py prepare --manifest "$batchDir\manifest.json" --diagnose
```

Тесты только synthetic/mocked: календарные границы/leap day, failure merges/actions/
orders/download, restart с PENDING, READY skip, config mismatch, повреждение state,
atomic updates/finalization, OS writer lock, отсутствие secrets/URLs в metadata/CLI.
Single large exports vs daily multipart проходят полный pipeline и дают одинаковые
mappings/splits/diagnostics, в том числе для malformed событий разных дней.
Ни live export, ни model training при реализации M02-10 не запускались.

## M02-11: training data quality gate

`Application/model/training_quality.py` — отдельная source-neutral policy после
successful preparation. Зависит от shared training_data validation, не от Mindbox,
API/raw parsing или PyQt. Immutable contracts: QualityLevel(PASS/WARN/BLOCK),
TrainingQualityDiagnostics, TrainingQualityConfig, QualityIssue, TrainingQualityReport.
Report хранит только counts/rates и разрешённый технический systemName breakdown;
не хранит PreparedBprData, customer/item/event IDs или PII. Mapping snapshots копируются
в MappingProxyType, issues — tuple. training_allowed вычисляется из level.

`evaluate_training_quality(prepared_data, diagnostics, config=...)` переиспользует
validate_prepared_data. Любой PreparedDataError даёт BLOCK/INVALID_PREPARED_DATA;
сложная validation logic не дублируется. Пустой train, users/items, неверные indexes,
NaN/inf/negative weights блокируются тем же validator. Summary sizes для invalid
prepared input равны None (не выдаются за валидные измерения). Некорректные counters
или config вызывают ValueError. Structural preparation exceptions не перехватываются
и не превращаются в WARN: gate вызывается только после возврата preparation result.

PASS — policy не нашла issues; WARN — валидные данные с recoverable потерями, training
allowed; BLOCK — training запрещён. Unresolved products >0 и unsupported products >0
дают отдельные BLOCK issues, даже если diagnostic preparation исключила их и вернула
валидный набор. Unsupported также входит в unresolved по resolver contract, поэтому
оба issue могут описывать частично одни и те же события; их counts не суммируются.
Unmapped actions — только informational metric, не issue.

Default mapped_action_malformed_warn_rate=0.0: любое malformed mapped action даёт
WARN/MAPPED_ACTION_WITHOUT_PRODUCT; BLOCK percentage не существует. Настраиваемый
warn threshold сравнивается строго `rate > threshold`, лежит в [0,1]. При default
0 malformed — PASS, >0 — WARN, если нет независимых BLOCK conditions.

Denominator: `actions_view + actions_favorite`, включая mapped actions без products.
InteractionBuilder увеличивает эти counters после classification ДО проверки products.
M02-08 diagnostics теперь передаёт их отдельно от view/favorite_interactions.
`malformed_rate = malformed / mapped_actions`; при нулевом denominator — 0.0.
Malformed > mapped отклоняется как некорректная диагностика. Все Actions (включая
unmapped) и число InteractionRecord не используются в denominator.

Daily CLI prepare после successful preparation строит quality diagnostics явно,
печатает level, training_allowed, counts/rate/issues и прежний complete flag.
Новая exit policy только этой команды: PASS/WARN -> 0, BLOCK/preparation error -> 1.
`--diagnose` по-прежнему требуется для продолжения после recoverable malformed;
strict preparation не ослаблена. Возможен честный вывод:
`Training data complete: False`, `Training quality: WARN`, `Training allowed: True`.
Сумма весов форматируется через .12g без изменения underlying float.

Transport complete проверяет final batch loader до preparation; gate не принимает
transport decisions. TrainingQualityReport — snapshot policy decision, а не изменение
core training API или автоматический запуск обучения. В будущей production orchestration
нужно явно применять training_allowed; существующий training core не переключён.
Мутации PreparedBprData после проверки требуют повторной validation/quality evaluation.
Single-window CLI сохраняет прежнюю exit policy, core preparation.complete не меняется.

Тесты: clean/unmapped PASS, VIEW/FAVORITE/mixed malformed WARN, правильный denominator
при нескольких products на action, identity loss/invalid prepared BLOCK, immutable
snapshots и CLI exit 0 для WARN при complete=False. Проверяется отсутствие IDs/PII
в report repr и CLI. Никаких новых API-запросов или model training.

### M02-12: explicit offline shadow training

`scripts/mindbox_shadow_train.py` validates a final daily manifest, prepares with
`diagnose=True`, evaluates the unchanged M02-11 gate, and trains only for PASS/WARN.
Legacy CSV remains the production source. No PyQt/inference integration is added.
The same TrainConfig instance supplies preparation weights/eval eligibility and all
training settings. CLI overrides only epochs and data_dir (catalog parent); CPU is
the supported device. The catalog must be named Номенклатура.csv and match data_dir
so product resolution and item features read the same file. Existing feature math,
including duplicate catalog keep="last", remains unchanged.

Source-neutral `shadow_training.py` evaluates quality again on supplied prepared
data, seeds as the legacy process does, then calls the shared structured training
API. BLOCK yields training_started=False, training_completed=False, QUALITY_BLOCK.
Training exceptions yield training_started=True, training_completed=False,
TRAINING_FAILED; exception text is excluded. Preparation/manifest failures propagate
to the CLI, which prints a safe error and does not train. CLI exit: 0 for completed
PASS/WARN, 1 for BLOCK/preparation/training/report failure. Complete=False is retained.

`train_prepared_data(cfg, prepared_data, device)` still returns `(model, splits)`.
`train_prepared_data_with_metrics` returns `(model, splits, TrainingRunMetrics)`.
Both delegate to one implementation; sampling, loss, evaluation, early stopping,
best-state restoration and console output are preserved. Immutable epoch metrics
contain epoch/loss/recall/ndcg; run metrics contain history, epochs_completed,
best_epoch, best_recall, best_ndcg, best_metric_name and early_stopped. Best metrics
are those of the existing selected epoch, not independent maxima. No stdout parsing.
The existing process-level PyTorch interop initialization is unchanged: CLI runs in
a fresh process; synthetic multi-run tests isolate that initialization.

ShadowTrainingResult is frozen; it contains a copied immutable preparation summary,
quality report, metrics, status flags and optional report path. Model remains in
memory and is excluded from repr (as is report path); no mappings/IDs enter reports.
The model itself remains an ordinary mutable PyTorch model.

Reports are written only below project `ВходныеДанные/MindboxReports/shadow_training/`
at `<run_id>/report.json`. JSON sections: run_id/timestamp/batch_id/interaction_window,
quality (level, allowed, metrics, issue counts/rates/systemName breakdown), dataset
(safe counts, complete, total weight), config (allowlisted scalar training parameters
and standard feature column names), training (status/error code/structured metrics),
production_model_published=false. Arbitrary config attributes, paths, custom feature
column names and raw data are excluded; nonstandard column names contribute only a
count. JSON disallows NaN/Infinity. Writer uses a same-directory temporary file,
flush, fsync and replace, cleaning up the temporary file on failure. BLOCK and
training failure also get reports; failed preparation has no fabricated summary.

Shadow never calls `_save_artifacts`, saves checkpoints, or enters the production
publication orchestration. Synthetic regression tests forbid publication/network,
verify sentinel bytes and directory listings under Модель (current/runs/.staging),
exercise actual feature-enabled/disabled CPU training, deterministic API parity,
early stopping, full synthetic manifest CLI, safe JSON and atomic replace failure.
The real three-day batch is NOT trained automatically.

Manual first smoke, from project root in PowerShell:

```powershell
.\.venv310aboba\Scripts\python.exe scripts/mindbox_shadow_train.py `
  --manifest '.\ВходныеДанные\MindboxRaw\training_batches\dfe6c68e1109410aa47aeb3342877629\manifest.json' `
  --catalog '.\ВходныеДанные\Номенклатура.csv' `
  --epochs 2 `
  --device cpu
```

### M02-13: source-neutral seen items

`Application/model/seen_items.py` defines SeenItemsIndex and SeenItemsError.
`build_seen_items_index(prepared_data)` reuses prepared validation and constructs
`user_pos_train[u] UNION eval_items for u`. Eval-only items remain seen; repeated
train/eval targets deduplicate. Evaluation and temporal holdout are unchanged.
CSR layout: int64 indptr[num_users+1], int64 indices, sorted unique per-user segments.
Dimensions, offsets, bounds, order and uniqueness are validated. Arrays are copied
onto immutable bytes backing; items_for_user returns a read-only view whose write
flag cannot be re-enabled. Repr contains only users/items/pair/covered-user counts.

Checkpoint optional fields: `seen_items_indptr`, `seen_items_indices`. Existing
num_users/num_items and mappings supply dimensions. Both fields absent means OLD
artifact and permits legacy fallback; any partial/corrupt contract is rejected,
never silently replaced by CSV. `_save_artifacts(..., seen_items=index)` validates
dimensions and stores only integer mapping indices. Its three-argument compatibility
call remains supported. New legacy CSV training explicitly builds and saves the
index from PreparedBprData. No separate identity-to-products JSON is written.

For an embedded index, print_recommendations neither requires nor reads interaction
CSV for seen filtering. Номенклатура.csv can still supply display names. Train and
eval-only seen items are excluded, including when fewer than k unseen items exist.
The print inference function now uses no_grad: the synthetic full-path test exposed
an existing requires-grad Tensor.numpy error. Output formatting remains unchanged.
Old artifacts continue using the legacy CSV seen helper/schema requirements.

Mass export also selects embedded seen when available and never invokes either
legacy seen helper for that branch. filter_seen=False leaves scores unmasked.
It still validates/reads legacy interaction sources for other responsibilities;
this is NOT a complete recommendation-source migration. Remaining M02-14/M02-15 work:

- Historical conversion: Просмотры.csv + Заказы.csv.
- Loyalty ranking: Заказы.csv + Избранное.csv + Просмотры.csv.
- Contact/profile fields (phone, email, discount card): currently Заказы.csv.

Stock handling, seasonal mapping, recommendation output formats, customer sources,
quality policy, BPR math and PyQt are unchanged. M02-12 shadow still publishes no
checkpoint or production artifacts. Tests cover independent pre-holdout pair parity,
legacy CSV parity (views only ТипТовара == Номенклатура), immutable/invalid layouts,
artifact roundtrip/corruption, print without interaction CSV, mass-export helper spies
and filter_seen=False. Artifact tests use temporary directories only.

### M02-14: source-neutral interaction analytics

NEW published training models embed `interaction_analytics` version 1 in the torch
checkpoint. The block contains dimensions, int64 per-item unique viewers/converted
counts, float64 per-user purchase/favorite/view activity, item-kind metadata in mapping
order, window_days=30 and prior_strength=20. No user IDs, item event lists, event
timestamps, contacts, raw records or extra ID mappings are included. Kind/global
counts and rates are derived from per-item counts (no inconsistent duplicate totals).
Arrays have immutable bytes backing; repr exposes dimensions/config only. Validation
checks the exact schema/version, dtypes/shapes, finite nonnegative values, integral
view/favorite counts, converted <= viewers <= num_users and model dimension parity.
An absent block means OLD fallback; a partial/corrupt block is an error.

`PreparedBprData.analytics` is an optional keyword-only extension, so existing
two-positional-argument callers and the shared training validator remain compatible.
MindboxPreparationResult exposes the same value via `.analytics`. Mindbox collects
resolved canonical interactions in its existing single raw-export pass, before
to_bpr_event/aggregation/holdout. Malformed, unmapped and unresolved records do not
enter analytics; quality diagnostics/policy remain unchanged. Legacy CSV preparation
collects from its already loaded frames using mapped valid rows and nomenclature
views, before temporal split. New legacy training explicitly passes analytics to
artifact publication. Older three-argument save callers remain compatible.

Conversion uses the first calendar date of VIEW for each canonical user-item pair,
then any purchase on that date through date+30 inclusive. Time of day is discarded
(LEGACY_DATE); purchase-before-view and later-than-window do not convert. Repeated
views/purchases do not multiply conversion counts. Bayesian smoothing is unchanged:
`(converted + 20 * prior_rate) / (viewers + 20)`, with kind prior then global prior.
Percentages retain legacy rounding to four decimals. Training-kind priors are frozen
in the artifact. At export, a seasonally mapped code with no own view history uses
its current catalog kind's stored rate, then global. Catalog duplicates retain
keep="last". Catalog/settings/seasonal/stock handling are not migrated.

Analytics memory is incremental: activity counters per canonical user, one first-view
day per unique pair, and merged intervals `[purchase_day-30, purchase_day]` per pair.
Repeated purchases do not accumulate occurrences; intervals merge when overlapping
or adjacent. For a year, a pair has roughly at most 12 separated intervals at the
default window. This handles orders before or after views without depending on
stream arrival order. Finalization produces only O(users+items) aggregate arrays.
This avoids an additional giant event DataFrame; existing BPR preparation's own
memory behavior is unchanged and is not claimed to be a streaming annual trainer.

Raw activity: purchase quantity (missing -> 1, clip 1..10), favorite count, mapped
view count. Export uses checkpoint TrainConfig weights and preserves the existing
lexsort priorities: score, purchases, favorites, views, stable mapping order for ties.
Phone eligibility is still applied independently by the existing contact path.

NEW models need no interaction CSV for seen filtering, historical conversion or
loyalty/activity calculation. OLD models retain each legacy fallback independently.
The remaining legacy recommendation dependency is customer/profile/contact fields
from Заказы.csv: phone, email, discount card (M02-15). Catalog and settings inputs
also remain. Mass export no longer requires interaction schemas when both embedded
contracts suffice; reading orders for contacts is still permitted and explicit.
Print recommendations and M02-12 shadow publication isolation are unchanged.

Synthetic tests cover dates/boundary/order, repeated pairs, quantity clipping,
kind/global/new-item fallback, canonical merges, skipped malformed/unmapped actions,
single raw parse, legacy conversion/ranking parity, safe immutable roundtrip and
corrupt blocks. A full export spy test removes views/favorites, forbids legacy
conversion/seen helpers and chunked activity reads, and observes only the existing
orders contact read. No API requests or live model training are required.
