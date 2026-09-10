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
| OrderLineRecord | order/customer mindboxId, firstAction.dateTimeUtc, channel externalId/name, lines; в каждой позиции id, number, quantity, basePricePerItem, priceOfLine, product.name, поддержанный product ID, status.ids.externalId |
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
