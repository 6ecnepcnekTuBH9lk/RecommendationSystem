# STAT-INDEX-PREFLIGHT-01 — финальная подготовка индекса на D:

31.10.2026; измерения 08:09–08:19 UTC (11:09–11:19 Europe/Moscow). Реальные источники проверены только через metadata, schema и справочники.

## Решение

**BLOCKED_PREPARATION.** Production root отсутствует; три существующих родителя не проходят штатную ACL-политику. В унаследованном окружении ACL helper также возвращает ACL_INSPECTION_FAILED. Нужна отдельная подготовка namespace и окружения.

Дополнительно **PILOT_REQUIRED**: обоснованного expected_gib нет. API-сутки 31.08.2026 проходят metadata-план, но маленький пилот не подтверждает безопасность загрузки исторических JSON-частей размером до 1,44 GiB.

READY_FOR_BUILD не выдан. Блокирующих проблем доступности canonical на разрешённом уровне проверки не обнаружено. Однозначно установленного дефекта, доказывающего невозможность build при подготовленном окружении, нет; существенный RAM-риск остаётся непроверенным.

Production-код, canonical, ACL, TEMP/TMP, настройки Windows и резервная копия на C: не менялись. Реальные индекс, пилотные файлы, расчёт статистики и детальный экспорт не выполнялись. Единственный новый файл в репозитории — этот отчёт.

## 1. Git и версии

| Параметр | Результат |
| --- | --- |
| Проект | D:\RecommendationSystem |
| Ветка | feature/hybrid-ranking |
| HEAD | abb9cd8e98bbcd06c9906409e50ed98ea1fd4166 |
| Commit | feat(statistics): finalize buyer cohorts and detailed export |
| Python / SQLite | 3.12.3 / 3.45.1 |
| Python executable | D:\RecommendationSystem\.venv312\Scripts\python.exe |
| SCHEMA_VERSION | 2 |
| Tracked-изменения после Freeze | Нет |
| Контроль 43 файлов Freeze | Побайтовые SHA-256 совпали с сохранённым manifest Freeze |

Локальный HEAD совпал с явно заданным и ранее независимо проверенным Freeze, поэтому fetch не требовался. Повторный аудит удалённого GitHub не проводился. Git mutations не выполнялись.

Три исторических untracked-файла catalog_snapshots и девять диагностических каталогов сохранены; содержимое не исследовалось. Они не являются блокерами.

## 2. Canonical, периоды и физические объёмы

Корень: D:\RecommendationSystem\input_data\MindboxRaw.
Catalog: D:\RecommendationSystem\input_data\MindboxRaw\canonical\catalog.json, schema_version=2, 2 301 byte.

Использованы штатные catalog/effective_entries, part_files, capture_sources и stat. Orders/Actions/CustomerMerges JSON не открывались. При дополнительном capture_sources/plan_pilot audit guard запрещал открывать raw *_part_*.json; попыток — 0. Справочники CSV читались штатным capture-механизмом.

| Источник | Effective partitions | Объявлено / найдено частей | Byte | GiB | Заявленный период UTC |
| --- | --- | --- | ---: | ---: | --- |
| actions | 2 (1 MANUAL + 1 API) | 14 / 14 | 14,754,974,446 | 13.742 | [2022-01-01, 2026-09-01) |
| orders | 2 (1 MANUAL + 1 API) | 6 / 6 | 5,474,380,105 | 5.098 | [2022-01-01, 2026-09-01) |
| customer_merges | 1 (1 API) | 1 / 1 | 138,098,677 | 0.129 | [2022-01-01, 2026-09-01) |

Пути доступны, несовпадений числа частей — 0; внутренних разрывов заявленного покрытия не обнаружено. Число событий, заказов и товарных позиций **NOT_READ**. Содержимое частей, их JSON-корректность и качество данных не проверялись.

Actions и Orders: MANUAL [2022-01-01, 2026-08-31) плюс API [2026-08-31, 2026-09-01). CustomerMerges: один API-интервал [2022-01-01, 2026-09-01).

Минимальная заявленная дата Orders — 01.01.2022, верхняя исключающая граница — 01.09.2026; последняя покрытая дата — 31.08.2026. Это границы metadata, а не timestamps реально прочитанных заказов. Непрерывное заявленное покрытие — 1 704 суток. После 31.08.2026 покрытие не заявлено; для позднейших периодов нужны обновлённые источники.

Сохранённый анализ: 01.01–31.08.2026, продуктовый фильтр активен; его значения не публикуются. RFM cutoff — 31.08.2026, LAST_48M начинается 01.09.2022 UTC. Есть восемь месяцев Orders до этого окна, важных для первой наблюдаемой покупки Cohorts. Это не доказывает полноту lifetime-истории клиента.

### Customers и справочники

| Объект | Metadata / read-only результат |
| --- | --- |
| D:\RecommendationSystem\input_data\MindboxRaw\canonical\customers.sqlite | 7 856 758 784 byte, примерно 7,317 GiB |
| SQLite header / journal | SQLite format 3, read/write version=1, journal_mode=delete |
| Страницы | 1 918 154 × 4 096 byte |
| profiles schema | id TEXT PRIMARY KEY, changed TEXT, raw TEXT |
| metadata.summary | MANUAL; заявлено 829 500 профилей; intervals=[] |
| Прочитано строк profiles | 0 |
| -journal / -wal / -shm | Нет до и после проверки |
| D:\RecommendationSystem\input_data\nomenclature.csv | 8 041 664 byte |
| D:\RecommendationSystem\input_data\site_categories.csv | 47 457 byte |

Customers открыт через URI mode=ro с query_only=ON/trusted_schema=OFF; диагностический authorizer разрешал только schema/metadata. Штатный _customers также успешно выполнил read-only attestation. 829 500 — число из metadata, не COUNT(*) и не независимо подтверждённое количество строк. Пустой intervals не подтверждает историческую полноту профилей. Полный integrity_check исходной базы не запускался.

CustomerMerges доступен по metadata. Граф идентичностей, проверка циклов/неоднозначностей и подсчёт merges не выполнялись.

## 3. Диск D:, ресурсы и бюджет

Оба диска — local fixed NTFS, без readonly-флага тома.

| Диск | Всего, byte | Свободно до тестов, byte | Свободно после тестов, byte | Свободно после тестов, GiB |
| --- | ---: | ---: | ---: | ---: |
| D: | 512,092,008,448 | 439,687,208,960 | 439,667,126,272 | 409.472 |
| C: | 511,037,140,992 | 52,919,963,648 | 52,920,795,136 | 49.286 |

Последний замер 2026-10-31T08:19:27Z. Рекурсивного сканирования проекта не было. Production root отсутствует; существующих поколений — 0, суммарный объём — 0.

BuildGuard: reserve_bytes >=20 GiB; начальное условие free >= reserve + 2*expected_bytes.

**Следующие E — сценарии, не измеренные размеры реального индекса.**

| E=expected_gib | Бюджет 2E, GiB | Нужно 20+2E, GiB | Остаток free−2E, GiB | Запас free−20−2E, GiB | Disk-check |
| ---: | ---: | ---: | ---: | ---: | --- |
| 10 | 20 | 40 | 389.47 | 369.47 | PASS |
| 25 | 50 | 70 | 359.47 | 339.47 | PASS |
| 50 | 100 | 120 | 309.47 | 289.47 | PASS |
| 100 | 200 | 220 | 209.47 | 189.47 | PASS |
| 150 | 300 | 320 | 109.47 | 89.47 | PASS |
| 200 | 400 | 420 | 9.47 | -10.53 | FAIL |

Остаток после 2E — условный остаток, не прогноз фактических файлов. Последний запас доступен иным процессам сверх обязательного резерва 20 GiB. Алгебраическая граница E при текущем free/reserve — 194.74 GiB; это **не рекомендация** --expected-gib.

В runtime guard обычно проверяет extra=0: минимум 20 GiB, а двойной начальный бюджет не является постоянной квотой. Старые поколения сохраняются; их размер, будущие pilot copies, обе SQLite, journal и временные структуры должны учитываться совместно. Пока pilot-based оценки нет, нельзя обоснованно выбрать E или гарантировать достаточность примерно 409,47 GiB.

### Память Windows

| Показатель | Byte | GiB |
| --- | ---: | ---: |
| Total physical | 34,116,624,384 | 31.774 |
| Available physical | 17,395,281,920 | 16.201 |
| Commit limit | 36,264,108,032 | 33.774 |
| Available commit | 13,599,543,296 | 12.666 |

Доступные величины измерены после тестов, total physical/commit limit — близким по времени отдельным Windows snapshot. Private usage диагностического процесса — 28.73 MiB; это не peak индексатора.

BuildGuard использует GlobalMemoryStatusEx/GetProcessMemoryInfo. Условия остановки: private_usage >3 GiB, available physical <1 GiB, available commit <1 GiB; disk reserve минимум 20 GiB. Обычная проверка — не чаще раза в секунду, дополнительно force на границах этапов и SQL progress handler каждые 10 000 VM operations. Это кооперативный мониторинг, **не OS hard memory cap**.

Существенные риски:

- raw_reader.load_export_part вызывает json.load, материализуя всю часть до выдачи записей. Максимальная Actions часть — **1 550 962 072 byte (1,444 GiB)**, Orders — **1 298 443 510 byte (1,209 GiB)**. Python dict/list/строки/Decimal увеличивают RAM. Внутри json.load нет guard/cancel callback: лимит может быть превышен до проверки, отмена задержана.
- CustomerMerges — 138 098 677 byte JSON; размер Python-графа CustomerIdResolver по нему неизвестен.
- При проекции одновременно живут часть JSON, граф, product metadata и SQLite. Commit каждые 1 000 событий не ограничивает память уже загруженного JSON.
- Перед SQL-индексами projector/identities/products/catalog/metadata явно освобождаются. Однако сортировка с temp_store=MEMORY сама может повышать private memory.
- Параллельные приложения расходуют RAM/commit. Текущие показатели выше floor, но не доказывают успех полного build.

Реальный benchmark не выполнялся, лимиты не увеличивались.

## 4. Root, ACL и штатный статус

Цель: D:\RecommendationSystem\filtered_data\statistics_index.

| Проверка | Результат |
| --- | --- |
| Root / current.json / generation | Отсутствуют |
| Путь вне canonical | Да |
| Local fixed writable NTFS | Да |
| Reparse/junction в проверенной цепочке | Не обнаружены, reject_reparse прошёл |
| Под настроенными OneDrive/OneDriveCommercial/OneDriveConsumer | Нет |
| validate_storage_root | PRIVATE_INDEX_ROOT_REQUIRED |
| Реальная запись SQLite в production root | Не проверялась; создавать root запрещено |

Отсутствие root ожидаемо перед первым build и не является дефектом приложения. Индексатор требует заранее созданный приватный root.

Read-only inspect_acls в унаследованном окружении: ACL_INSPECTION_FAILED. В отдельном дочернем процессе со стандартным PSModulePath Windows PowerShell 5.1 тот же helper успешно прочитал ACL. Изменялась только копия окружения этого процесса; системные/user settings и TEMP/TMP не менялись. Helper запускает powershell.exe, поэтому окружение запуска нужно проверять отдельно от ACL.

| Существующий родитель | check_acl_facts(private=False) | Owner trusted | Unsupported ACE | Foreign allow ACE |
| --- | --- | --- | --- | ---: |
| D:\RecommendationSystem\filtered_data | UNSAFE_STORAGE_ACL | Да | Нет | 4 |
| D:\RecommendationSystem | UNSAFE_STORAGE_ACL | Да | Нет | 4 |
| D:\ | UNSAFE_STORAGE_ACL | Да | Нет | 4 |

Имена пользователей и их SID не публикуются. Есть foreign allow-права, запрещённые консервативной политикой namespace. Защита только leaf недостаточна: validator проверяет всех родителей. Политика не обходилась и произвольным AccessCheck не заменялась. Приватность отсутствующего root пока не подтверждена.

Штатный CLI из D:\RecommendationSystem:

~~~powershell
& 'D:\RecommendationSystem\.venv312\Scripts\python.exe' -B scripts\build_statistics_index.py --status
~~~

Exit **0**, точный результат:

~~~json
{"state":"INDEX_MISSING"}
~~~

Соответствует отсутствию current.json. index_state сначала делает metadata capture, затем ищет поколение; INDEX_MISSING не доказывает безопасные ACL/готовность build. Отдельный storage validator возвращает PRIVATE_INDEX_ROOT_REQUIRED.

До capture подтверждены существующий непустой canonical writer-lock, rollback header Customers и отсутствие sidecars. Штатный lock не нуждался в создании/записи начального байта. Контроль catalog/lock/stat после проверок показал отсутствие изменений.

## 5. Provenance, публикация и отмена

_algorithm_version учитывает все восемь обязательных зависимостей: statistics_purchase_policy, statistics_purchase_facts, statistics_buyer_aggregates, statistics_buyer_validation, statistics_rfm_core, statistics_rfm_validation, statistics_cohorts_core, statistics_cohorts_validation. Также включены основные статистические модули и рекурсивно Python-код Application\mindbox. Detail Export не добавлялся: он не определяет агрегированный INDEXED.

Algorithm hash: b679eb7d028436f3f74ef7902c862017c94200c411b8f3dcb4cab1a03904253e.

Digest source_version metadata snapshot: 491d5017a06c038d615f1e55f8f529d4c516096ecc77f0cc8969c97744e6cc01. Это provenance источников, не fingerprint построенного реального поколения.

Builder сохраняет source_version в manifest/metadata и demographics binding, повторно проверяет источники перед публикацией. Reader сравнивает manifest.provenance и source_version, несовпадение => STALE_SOURCE. Изменение алгоритмического hash инвалидирует индекс.

source_version исключает filter, as_of, timezone, customer_period, rfm_as_of. Изменение этих полей в копии реального snapshot сохранило source_version: пользовательская дата/фильтр меняют срез, а не требуют полного rebuild. Синтетические dependency/stale/scope/parity проверки прошли. Реальной генерации для эксперимента не создавалось.

| Механизм | Подтверждено кодом и синтетическими тестами |
| --- | --- |
| Locks | Index writer, затем canonical writer; конкурирующий builder отвергается |
| Staging | Собственный UUID; сверка идентичности root/folder |
| Проекция | Сверка counts Actions/Orders с projected rows, counts связей |
| Источники | Capture до/после проекции и перед publication; mismatch => STALE_SOURCE |
| Целостность | integrity_check обеих БД; foreign_key_check основной |
| Manifest | schema/generation/complete/provenance/counts, demographics hash/bytes и binding |
| Publication | commit/close/fsync SQLite, atomic_json manifest и затем current.json |
| ACL | Повторная проверка до публикации; root не ремонтируется |
| Отмена | callbacks, SQL progress handler; CLI SIGINT/SIGTERM/SIGBREAK или cancel-file |
| Ошибка | Закрытие connections, cleanup только собственного staging |
| Старые поколения/чужие файлы | Сохраняются, автоматической очистки нет |
| Retry | Автоматического повторного build нет |

Регрессия покрыла отмену на разных этапах, source change, competing lock, disk exhaustion, corrupt staging/manifest/pointer, publication failure, staging substitution и сохранность прежней генерации.

Внезапное завершение процесса/ОС не является кооперативной отменой: finally не гарантирован, может остаться неопубликованный staging. Автоматически удалять такие каталоги нельзя. Загрузка большой JSON-части может задержать отмену. Реальный crash/power-loss эксперимент не выполнялся.

## 6. Контролируемый пилот и обоснование оценки

**PILOT_REQUIRED**, ESTIMATE_READY не достигнут.
Read-only штатный plan_pilot дал **METADATA_PLAN_READY** для 31.08.2026; record/profile limits ещё не проверены.

| Компонент | Byte | MiB |
| --- | ---: | ---: |
| Actions API, полные UTC-сутки | 89 674 896 | 85,52 |
| Orders API, те же сутки | 4 491 608 | 4,28 |
| Actions+Orders | 94 166 504 | 89,80 |
| Полный CustomerMerges | 138 098 677 | 131,70 |

Это единственная общая API-дата в effective catalog. Обе границы точно midnight UTC, байтовые лимиты соблюдены. Произвольный фрагмент MANUAL или неполного дня не выбирался.

PilotLimits по коду: <=7 последовательных суток, Actions+Orders <=128 MiB, merges <=200 MiB, Actions <=250 000, Orders <=20 000, profile candidates <=100 000. Количества реальных записей/позиций/aliases неизвестны. Если будущая штатная подготовка превысит record/profile limit, она должна остановиться; тогда возможен PILOT_SELECTION_BLOCKED, а не автоматическое повышение лимита.

Реальные prepare_pilot/build_pilot не вызывались, TEST_ONLY-источник не создавался. Внешний JSON содержит только агрегатную сводку плана. Одноимённые функции в unit-тестах работали только с маленькими синтетическими fixtures.

Методика отдельного следующего этапа:

1. После инфраструктурной подготовки повторить capture/plan: sources и provenance могли измениться. Сейчас выбрать можно только полные paired API-сутки 31.08.2026.
2. В новом приватном TEST_ONLY-каталоге D: выполнить отдельно разрешённую ограниченную подготовку; при превышении byte/record/profile budgets остановиться.
3. Измерить подготовку и build раздельно: peak private/working set/commit, minimum free D:/C:, peak staging+journal, размеры обеих БД, время этапов; counts Actions/action_products/Orders/OrderLines/profiles и alias candidates. Персональные IDs/контакты не публиковать.
4. Разделить стоимость фиксированных справочников/графа, событий Actions, связей, заказов, товарных позиций, профилей и SQL-индексов. Не масштабировать единственным коэффициентом SQLite/raw-bytes либо по числу файлов.
5. Для всей MANUAL-истории counts неизвестны; один API-день не подтверждает репрезентативность. При отсутствии достоверных metadata counts после пилота потребуется отдельно согласованное ограниченное измерение/исследование. До него full-size экстраполяция остаётся предварительной.
6. Отдельно подтвердить безопасность больших MANUAL-частей и identity graph. API-часть 85,52 MiB не воспроизводит загрузку исторической части 1,44 GiB.
7. Только после обоснования expected_gib повторить ACL/resources/provenance checks и принять новое решение о полном build.

Численная оценка полного индекса и времени построения сейчас не обоснована.

## 7. Потенциальный расход C:

statistics.sqlite на D: получает DELETE/FULL, temp_store=MEMORY, cache_size=-32768 (~32 MiB). Rollback-journal расположен рядом с БД на D:. temp_store не является гарантией отсутствия всех journal/temp операций.

demographics.sqlite получает FULL, cache_size=-8192 (~8 MiB); отдельный temp_store=MEMORY не установлен. У runtime измерены default temp_store=0 и compile TEMP_STORE=1. Это допускает файловые временные структуры там, где их требует SQL-план. Большой реальный spill demographics не наблюдался и не утверждается.

TEMP/TMP и Python tempfile.gettempdir() указывают на %LOCALAPPDATA%\Temp на C:. SQLITE_TMPDIR unset. На Windows SQLite выбирает временный путь через Windows VFS/GetTempPath(); установка SQLITE_TMPDIR сама по себе не является подтверждённым решением. См. [SQLite Temporary Files](https://www.sqlite.org/tempfiles.html) и [PRAGMA temp_store](https://www.sqlite.org/pragma.html#pragma_temp_store).

Read-only CIM: C:\pagefile.sys, allocated 2 048 MiB, used 79 MiB, peak 103 MiB; automatic managed pagefile=true. Рост commit может приводить к подкачке/росту pagefile на C:. Это snapshot, не ограничение будущего расхода. C: свободно примерно 49,29 GiB.

В исследованном build нет явного Python tempfile для staging: SQLite/manifest/current.json/atomic_json привязаны к D:. Для CSV/product processing по коду не выявлен обязательный отдельный большой spill, но общий TEMP остаётся на C:. Model/training импортов ради проверки не было.

В будущей задаче пилота мониторить free C:, temp и commit/pagefile одновременно с D:. Если требуется temp на D:, отдельно подготовить приватный TEMP/TMP нового процесса и проверить фактический SQLite temp-path; это не переносит pagefile. Здесь TEMP/TMP, pagefile и настройки Windows не менялись. Гарантия «индексатор не затронет C:» не выдаётся.

## 8. Тесты и контроль сохранности

Python на D:, -B -m pytest -q --tb=short -p no:cacheprovider; basetemp и junitxml в D:\CodexTestTemp\statistics-index-preflight-01e3a17b6a4d4d56aa254a305ed92ae7.

Проверены:

- tests/test_statistics_index.py;
- tests/test_statistics_storage_safety.py;
- tests/test_statistics_slices.py;
- tests/test_statistics_pilot.py;
- tests/test_statistics_rfm_integration.py;
- tests/test_statistics_cohorts_integration.py.

Результат: **218 passed, 1 deselected in 59.02s**, exit 0. Падений, xfail и skip нет.

Точный deselect: tests/test_statistics_index.py::test_private_directory_protection_on_new_directory. Этот тест реально вызывает icacls/protect_directory; Windows ACL менять запрещено. Он не считается пройденным, код тестов не изменялся. Остальные index/pilot fixtures подменяют ACL operations и используют только синтетические tmp_path. Фактическое создание приватной Windows ACL этим запуском не проверялось.

Дочерний pytest использовал PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, QT_QPA_PLATFORM=offscreen и стандартный Windows PowerShell PSModulePath. TEMP/TMP не менялись; cacheprovider отключён. Реальные Mindbox данные и многотысячный suite не запускались.

Ruff: python -B -m ruff check --no-cache для 7 исследованных production/CLI-модулей и 6 указанных тестовых файлов — **All checks passed**, exit 0.

git diff --check / финальный список изменений: PASS: tracked diff пуст; единственный новый файл задачи — этот отчёт, прежние untracked-объекты сохранены.

Артефакты в D:\CodexTestTemp\statistics-index-preflight-01e3a17b6a4d4d56aa254a305ed92ae7: metadata-only JSON, pytest.log/xml/summary, ruff.log/summary, final_verification.json и синтетические tmp_path. Они не являются реальным пилотом, не удалялись.

Контроль после тестов: 43 файла Freeze совпали по hash; catalog побайтово и lock byte/stat неизменны; inventories частей по имени/размеру/mtime/inode неизменны (без чтения/хеширования raw contents); Customers stat/header и stat справочников прежние, sidecars не появились. Production root/current.json/реальные поколения отсутствуют. Ранее существовавшие untracked-объекты сохранены.

## 9. Существенные проблемы

### High — namespace/root

Файл: Application/statistics_storage_safety.py, validate_storage_root/check_acl_facts. Сейчас код требует приватный leaf и доверенную цепочку родителей. Root отсутствует, родители UNSAFE_STORAGE_ACL: безопасный build блокируется. Рекомендуется согласованная подготовка namespace администратором, затем нового leaf; ожидается штатный validator PASS. Риск изменения родительских ACL High из-за других пользователей/программ; запрещено слепо применять /T, /reset или удалять права на всём D:. Расчётное поведение сохраняется, права доступа изменятся по назначению. Новый unit-test для отчёта не нужен; требуется повторная read-only инспекция и отдельная разрешённая проверка leaf ACL.

### Medium — окружение helper

Файл: Application/statistics_storage_safety.py, inspect_acls. Унаследованное окружение даёт ACL_INSPECTION_FAILED, стандартный PSModulePath отдельного процесса позволяет инспекцию. Это fail-closed блокер окружения. Рекомендуется согласовать launch environment Windows PowerShell, без автоматического изменения глобальных настроек/кода. Ожидается воспроизводимая инспекция. Риск Low для отдельного процесса, но его поиск сторонних PS-модулей нужно учитывать. Расчёты сохраняются; нужна повторная read-only проверка фактического окружения запуска, а не новый тест изменения ACL.

### High — загрузка больших JSON и RAM

Файлы/функции: Application/mindbox/raw_reader.py (load_export_part/iter_export), statistics_index.py (BuildGuard/build_statistics_index), mindbox/identity.py (CustomerIdResolver). Сейчас части материализуются целиком до guard callback; до 1,444 GiB на диске плюс Python amplification/граф. Это риск памяти, стабильности и задержки отмены, не измеренное падение build. Рекомендуется отдельная ограниченная проверка memory-budget; при подтверждении — отдельно согласованное точечное исправление reader с regression tests. Ожидается доказанный предел RAM либо обоснованная остановка до full build. Полный неподготовленный запуск High risk, синтетическая проверка Low risk. В preflight поведение сохранено. Перед возможной реализацией нужны тесты строгого JSON-контракта, Decimal, порядка/дубликатов и ошибок.

### Medium — бюджет и системный диск

Файлы: scripts/build_statistics_index.py (--expected-gib), Application/statistics_index.py, окружение TEMP/pagefile. Сейчас byte metadata не заменяет counts событий/позиций, TEMP/pagefile на C:, второй connection допускает file temp. Нельзя гарантировать место/отсутствие расхода C:. Рекомендуются отдельно разрешённый пилот, компонентные измерения и мониторинг C:/commit. Ожидается обоснованный expected_gib/условия остановки. Риск пилота ограничен PilotLimits и private TEST_ONLY, но не нулевой; full build без оценки High risk. Расчётное поведение сохраняется. Для отчёта unit-регрессии достаточно, реальных измерений она не заменяет.

## 10. Следующая подготовка — команды НЕ ВЫПОЛНЕНЫ

Сначала администратор должен проверить запрещённые allow ACE всех трёх родителей и согласовать безопасный namespace. Универсальная команда удаления этих ACE не предлагается: она может нарушить доступ других пользователей/программ. Требование точное: каждый родитель проходит штатный check_acl_facts(private=False). Один приватный leaf недостаточен.

Блок ниже — только для отдельной согласованной подготовки. На сегодняшних ACL он обязан остановиться ДО создания root:

~~~powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\RecommendationSystem'
$python = 'D:\RecommendationSystem\.venv312\Scripts\python.exe'
$indexRoot = 'D:\RecommendationSystem\filtered_data\statistics_index'

# Только окружение будущего процесса, не user/system settings.
$env:PSModulePath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\Modules'

# Read-only admission родителей по существующей политике.
@'
from pathlib import Path
from Application import statistics_storage_safety as safety
root = Path(r"D:\RecommendationSystem\filtered_data\statistics_index")
safety.reject_reparse(root)
safety.local_volume(root.parent)
parents = tuple(root.parents)
for path, fact in zip(parents, safety.inspect_acls(parents)):
    safety.check_acl_facts(fact, private=False)
    print(str(path) + ": PASS")
'@ | & $python -B -
if ($LASTEXITCODE -ne 0) {
    throw 'Parent admission failed; stop and review parent ACL separately.'
}

# Только после PASS. Существующий root не исправлять/не заменять.
if (Test-Path -LiteralPath $indexRoot) {
    throw 'Root exists; inspect it separately without repair/cleanup.'
}
New-Item -ItemType Directory -Path $indexRoot -ErrorAction Stop | Out-Null
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
# Только новый leaf; без /T и без ACL родителей.
& icacls.exe $indexRoot /inheritance:r /grant:r "*$($sid):(OI)(CI)F" '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F'
if ($LASTEXITCODE -ne 0) {
    throw 'Leaf ACL failed; stop, no automatic cleanup/retry.'
}
& $python -B -c 'import sys; from Application.statistics_storage_safety import validate_storage_root; validate_storage_root(sys.argv[1]); print("PRIVATE_ROOT_VALIDATED")' $indexRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Root validation failed; stop, no build.'
}
~~~

Если leaf уже существует, grant/cleanup команды не выполнять без отдельной проверки владельца, ACL, содержимого и происхождения. После подготовки повторить read-only validator, --status, disk/memory/provenance checks в фактическом launch environment. INDEX_MISSING до первого build остаётся допустимым.

Следующие отдельные задачи: подготовка namespace/окружения; затем ограниченный пилот и проверка RAM больших исторических частей; затем обоснование expected_gib и новое решение о full build. Здесь подготовка/пилот/индексация не запускались. **Preflight завершён, автоматического продолжения нет.**
