# STAT-FINAL-ACCEPTANCE-01 — финальная техническая приёмка

Дата: **2026-10-31**. Проект: D:/RecommendationSystem.
Ветка: **feature/hybrid-ranking**. Python: .venv312/Scripts/python.exe.

**FUNCTIONAL ACCEPTANCE: PASS WITH CAVEATS.**
В выполненных проверках не осталось подтверждённых незакрытых P0/P1; production-код
не менялся. Найденные дефекты тестовой изоляции и устаревшие ожидания исправлены.
Единственный широкий запуск был прерван, поэтому единого полностью зелёного
результата всех 37 модулей после исправлений нет. Исправления и ближайшие
зависимые проверки выполнены отдельно. Это ограничение не скрывается суммой
перекрывающихся результатов.

**REAL-DATA ACCEPTANCE: NOT TESTED.**
Реальные canonical Orders/Customers, история Mindbox, модели и существующие
индексы не использовались для расчётов/экспорта/построения. Нет подтверждения
производительности на реальной истории. В прерванном запуске выявлен побочный
загрузчик с обычными путями к CSV-справочникам рекомендаций; он мог читать
локальную номенклатуру, категории и настройки рекомендаций. Этот запуск нельзя считать строго
изолированным от всех локальных справочников. Загрузчик не читает Orders/Customers,
не загружает веса модели и в данном режиме не сохраняет правила. После исправления
GUI-фикстуры её CSV и настройки находятся только в pytest tmp_path на D:.

## 1. Объём и среда

Проверены Purchase Policy/Facts, BuyerAggregates, агрегированный/персональный RFM,
Cohorts, REFERENCE/INDEXED, кеш v10, GUI, XLSX/offline HTML, детальные CSV/ZIP,
worker/IPC, ACL, отмена/атомарность и provenance.

Команды выполнялись через Python -B -m pytest и Python -B -m ruff --no-cache,
а не через старые Scripts/*.exe. Применялись процессные переменные
PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1,
QT_QPA_PLATFORM=offscreen и PSModulePath штатных модулей Windows PowerShell 5.1.
Это предотвращает конфликт Get-Acl с наследованными модулями PowerShell 7;
системные настройки не менялись. Реальные каталоги назначения/индекса на D:
и их ACL в данной задаче не проверялись и не исправлялись.

Все контролируемые наборы данных и результаты находятся в отдельной операции
D:/CodexTestTemp вне Git. Исторические diagnostics не исследовались, не очищались.
Изначальное дерево содержало 13 изменённых tracked-файлов и новые Cohorts/Detail
исходники; несвязанные catalog snapshots не включены в список фиксации статистики.

## 2. Фактические результаты

Числа ниже получены из JUnit XML; наборы перекрываются, их нельзя суммировать
как число уникальных тестов. failed и errors приведены отдельно.

| Запуск | passed | failed | errors | skipped |
|---|---:|---:|---:|---:|
| Пять ранее проблемных GUI-тестов, с параметризацией | 6 | 0 | 0 | 0 |
| 19 целевых наборов | 699 | 0 | 0 | 0 |
| Новые проверки heartbeat/PII и двух cohort dependencies | 3 | 0 | 0 | 0 |
| Единственный расширенный: 37 модулей, 1547 собранных случаев | итог отсутствует | итог отсутствует | прерывание | итог отсутствует |
| Узкая диагностика Qt с обязательным error при необработанном исключении | 2 | 0 | 1 | 0 |
| Первая проверка новой GUI-фикстуры/регрессионного теста | 2 | 1 | 0 | 0 |
| Полный модуль GUI-отбора после исправления | 39 | 0 | 0 | 0 |
| Неисполненный хвост и зависимая GUI-регрессия: 20 модулей | 644 | 7 | 0 | 0 |
| **Последний тестовый запуск: демография, scope, cohorts integration, slices** | **118** | **0** | **0** | **0** |

Прерванный широкий запуск вернул код 1 и не создал итоговый extended.xml.
Частичный вывод с точками не принят за точный счётчик passed. Прерывание
локализовано около test_selection_errors_use_global_status_and_readiness_stays_separate
в tests/tabs/test_analysis_filter_tab.py; узкое воспроизведение подтвердило
RuntimeError в recommendation_rules_widget._Task.run при отправке Qt-сигнала.

В первой версии нового регрессионного теста ожидание _task is None срабатывало
до первого QTimer callback. Оно исправлено на _initialized and _task is None;
продуктовый код для этого не менялся. Далее весь GUI-модуль дал 39 passed.

Семь последующих падений в тестах демографии/scope разобраны ниже и закрыты
последним запуском 118 passed. В 20-модульном прогоне после исправления изоляции
не зарегистрировано необработанных Qt-исключений.

Полный многотысячный набор проекта не запускался. Полный повтор 37 модулей
после исправлений не выполнялся: повторены исправленные тесты и ближайшие
зависимые регрессии; общий расчётный контракт не менялся.

## 3. Закрытие прежних трёх падений и двух расхождений 01B2

Все пять функций находятся в tests/tabs/test_dataset_statistics_tab.py:

| Тест | Причина прежнего падения / проверенное изменение |
|---|---|
| test_five_pages_technical_content_scroll_and_no_tooltips | Clients содержит 12 таблиц, включая ранее добавленную когортную заглушку; ожидалось 11. |
| test_customer_analytics_render_themes_scroll_and_no_duplication | В ожиданиях отсутствовали когортный заголовок и одна строка таблицы. |
| test_customer_demographic_layout_order_and_missing_profiles | Ожидание 11 таблиц не учитывало когортный раздел; обе параметризации проверены. |
| test_export_contains_every_visible_table_and_card | Проверка видимых таблиц не включала существующий отдельный XLSX-лист «Когортный анализ». |
| test_final_chart_set_preserves_tables_and_section_titles | Устарел итоговый count таблиц Clients; количество и положение графиков сохранены. |

В этой задаче эти ожидания повторно не менялись. Сверены текущий diff и
cohort_sections/export_statistics: таблица действительно существует, XLSX
содержит собственный лист. Проверки полного состава, отсутствия дублирования,
графиков и наличия каждой таблицы/карточки в экспорте сохранены. Первый обязательный
запуск дал 6 passed; все эти сценарии также прошли в последующей GUI-регрессии.
Прежние 736 passed, 3 failed не используются как зелёный итог текущей приёмки.

## 4. Подтверждённые исправления текущей задачи

### P1 / High — изоляция и завершение GUI-фикстуры

Файл: tests/tabs/test_analysis_filter_tab.py.
Затронуты fixture filter_tab и самостоятельные GUI-сценарии; добавлена
synthetic_recommendation_sidepanel.

До исправления вложенный RecommendationRulesWidget запускал настоящий _Task
с обычными путями CSV/настроек. QApplication теста мог завершиться раньше worker;
его _Signals.finished уже был удалён, что прерывало pytest. Последствие:
нестабильная приёмка и выход за изоляцию pytest-данных. Это не подтверждённая
ошибка расчёта статистики или модели.

Выполнено минимальное исправление только теста: две маленькие синтетические CSV,
временный путь правил, оригинальный QRunnable с оригинальной логикой/сигналами,
ожидание QThreadPool и обработка queued/deferred событий до уничтожения QApplication.
Ничего не заменено no-op worker и ни один тест не исключён.
Добавлен test_nested_recommendation_panel_uses_synthetic_catalog_and_finishes.
Риск низкий; production-поведение полностью сохранено. Исправление сначала
основано на воспроизведённом исключении, затем проверено отдельным regression test.

### P1 / Medium — устаревший размер DTO в демографической parity

Файл: tests/test_statistics_demography.py, helper exact.
Пять падений останавливались на len(asdict(actual)) == 92. Действующий
DatasetStatistics содержит 93 поля, включая cohort_statistics; это соответствует
существующей проверке в test_statistics_index.golden.

Ожидание исправлено на 93 и добавлена проверка наличия cohort_statistics.
Следующее полное asdict(actual) == asdict(expected) не удалено и не ослаблено.
Риск низкий, изменён только тест; приложение/схема кеша не менялись.
После исправления проверены все сочетания дат, дней рождения, leap/year boundaries,
профилей и фильтров, включая пересоздание индекса после изменения источника.

### P1 / Medium — две предпосылки старых customer-scope тестов

Файл: tests/test_statistics_customer_scope.py.

test_orders_own_coverage_original_period_and_cache_keys_do_not_change_legacy_metrics
сравнивал legacy-показатели, исключая Buyer/RFM/scope, но оставлял новый Cohorts.
Когорты намеренно имеют собственные cutoff/display bounds. Cohorts добавлен
в перечень аналитических расширений, а вместо скрытого исключения результата
добавлены точные проверки: cutoff 02.01 против 01.01, bounds и базы EXTENDED
2 против 1. Последующие полные REFERENCE/INDEXED равенства сохранены.

test_customer_period_is_attested_in_slices_not_only_the_normalized_legacy_filter
должен изолировать аттестацию периода scope. Оставленный cohort_statistics
закономерно вызывал более ранний StatisticsCacheError при несовместимом периоде.
При подготовке изолированного старого v10 DTO удаляется также optional Cohorts;
regex period и все проверки load/save сохранены. Дополнительно добавлен
test_cohort_snapshot_rejects_customer_period_mismatch_before_slice_attestation:
исходный DTO проходит валидацию, тот же DTO с подменённым периодом и сохранёнными
когортами обязательно отклоняется.

Риск низкий: изменены только тестовые предпосылки, не формулы/fallback/валидаторы.
Регрессионный тест добавлен; ожидаемый результат — обе ступени защиты проверяются
отдельно. Совместно с демографией, Cohorts integration и slices: 118 passed.

## 5. Результаты по компонентам

| Компонент | Подтверждено на выполненных фикстурах |
|---|---|
| Purchase Policy/Facts | STRICT = F+sold; EXTENDED = F+sold+CP; delivering/возвраты не признаются покупками. Положительное целое количество, цена не определяет факт покупки; неизвестные деньги не становятся нулём. Один общий факт, без независимых определений. |
| Production PURCHASE | Сохранён прежний набор CP/delivering/F и его отдельная семантика quantity. Аналитическая политика не подменила рекомендательную. |
| BuyerAggregates | Buyers/orders/positions/units, первая/последняя наблюдаемая покупка, интервалы, frequency/repeat, распределения, годы, деньги/неполнота, демография. Исходный период пользователя, независимость от товарного фильтра, последовательные merges, покупки без профиля. |
| Агрегированный RFM | 12/24/48 и PREVIOUS_12M, UTC cutoff, weighted nearest ranks с ties, R/F/M, валютные базы/dual currency, сегменты, UNSCORABLE, сравнение 12M и неполное покрытие. F=0 не делает покупателя автоматически недостоверным, неизвестная M не заменяется 0. |
| Персональный RFM | Совпадение порогов/population, распределений R/F/M, score_counts/segments и доступности на полной сопоставимой базе; 12 размерностей политик/окон/валют. Export event period не ограничивает историческую базу. Golden-проверка агрегата сохранена. |
| Cohorts | Первая наблюдаемая покупка по всей истории, в том числе до 48M; границы отображения не пересоздают когорту. Merges, первая допустимая snapshot, порядок/ties/cutoff. Горизонты 30/60/90/180/365, eligible/repeated/censored/incomplete, repeat_rate от своей eligible базы; observed positives вне полной базы сохранены. |
| Retention | UTC месяцы M0–M12; собственные знаменатели; incomplete/unfinished месяцы недоступны, а не фиктивные нули. Проверяется монотонность наблюдаемых повторов по контракту, не произвольная монотонность разных eligible числителей. |
| REFERENCE/INDEXED | Полные семантические DTO сравниваются, техническое время нормализуется либо фиксируется. Старые Orders для когорт доступны через тот же cursor; расширение касается Orders, не дополнительной истории Actions. Повторного полного чтения Orders нет, exceptional rows/legacy filters сохранены. |
| Provenance | Cohort core/validation входят в алгоритмический digest. Новый end-to-end параметризованный тест меняет каждый digest и получает STALE_SOURCE у готового синтетического индекса; current.json не меняется. Реальные старые индексы не открывались. |
| Кеш v10 | Старые v10 без optional Buyer/RFM/Cohorts и новые DTO проходят предусмотренную совместимость без изготовления данных. Другие schema отвергаются без переписывания файла. Некорректные вложенные значения, периоды, counts/PII поля отклоняются; открытие кеша не запускает расчёт. |
| GUI/XLSX/HTML | Пять страниц, Buyer/RFM/Cohorts/матрица, отсутствие данных, длинные таблицы, округление. Отдельный XLSX-лист когорт и соответствующий раздел Clients offline HTML, escaped content, DTO-only экспорт без пересчёта, прежний displayed_result при ошибке. |
| Detail | Ровно 177/23/18 колонок и четыре ZIP-entry, исходные строковые ID/нули/aliases, FK связность, профиль/конфликт/отсутствие, UNKNOWN permission. Decimal, RUB/KZT, NULL, Unicode и CSV formula protection проверены. |
| Потоковость/атомарность | Один canonical Orders проход, компактное временное состояние/пакеты, умеренная 1500-order fixture; SQLite/cleanup/лимиты/место, ошибки записи, отмена до публикации и сохранность прежнего destination. Ни одного реального клиентского ZIP. |
| Qt-worker/PII | Текущий Python, QProcess, bounded JSON Lines, exit codes/terminal, heartbeat, private stdin cancel, Close/Quit, кнопки/конфликты/кеш. Новый тест подавляет одновременно stdout/stderr backend и требует heartbeat до finished. Manifest/progress/errors не содержат синтетические контакты/ID. |
| ACL | Проверены policy-факты и реальные private ACL нового пустого тестового файла, в том числе после os.replace. Функциональные fixtures изолируют ACL subprocess; это не проверка будущего пользовательского/сетевого каталога. |

## 6. Дополнительные узкие проверки

В tests/test_statistics_detail_worker.py добавлен
test_heartbeat_and_terminal_keep_backend_stdout_stderr_private: реальный heartbeat
worker, terminal последним, оба потока backend подавлены, служебный вывод не содержит
тестовый контакт.

В tests/test_statistics_cohorts_integration.py добавлен
test_existing_index_rejects_changed_cohort_dependency_without_publication
для core и validation. Маленький индекс создаётся только в pytest tmp_path на D:.
Этот индекс не является реальным INDEXED проекта.

## 7. Незакрытые замечания и границы

P2 / Low: docs/statistics_detail_backend.md:100 упоминает сегмент INACTIVE,
которого нет в statistics_rfm_core.SEGMENTS. Реальные сегменты и расчёты совпадают;
ошибка только в пояснении. По указанию «P2 не выполнять автоматически» исходный
документ не исправлялся. Рекомендуется отдельно уточнить формулировку по действующим
rfm_segment/SEGMENTS, не добавляя сегмент или новую методику. Риск исправления
документа низкий; production-поведение полностью сохраняется, новый расчётный тест
для такого текста не требуется.

Нет проверки реальной полной истории, полноты реальных статусов/слияний, фактических
контактов, повреждений настоящих файлов или времени/диска на многомиллионном объёме.
Тест памяти умеренный и измеряет Python allocations, не весь RSS/SQLite/Windows.
Потребление памяти зависит также от крупнейшего заказа и карты CustomerMerges.
Контакты Customers — текущий snapshot, не реконструкция профиля на as_of.

Реальные ACL/private root, свободное место, переносимость PowerShell окружения и
совместимость конкретного сохранённого индекса должны проверяться отдельным
STAT-INDEX-PREFLIGHT-01. При недоступной проверке прав backend отказывает, не
обещает конфиденциальность на любом диске. Kill/crash не гарантируют cleanup;
обычная кооперативная отмена проверена. Удаление файла не обещает физического
стирания на SSD. Не проверялись сетевые/облачные хранилища как допустимое назначение.

Не запускались training/BPR-MF/Hybrid, исторические pilot/candidate-prototype
диагностические наборы и остальные несвязанные тесты проекта. Production-модель
и её архитектура не менялись.

## 8. Изменения текущего этапа и готовность

Изменены только пять тестовых модулей:
- tests/tabs/test_analysis_filter_tab.py — synthetic sidepanel и worker teardown.
- tests/test_statistics_demography.py — актуальный размер DTO при полной parity.
- tests/test_statistics_customer_scope.py — корректное разделение legacy/optional
  аналитики, дополнительные точные cohort assertions и негативная cross-validation.
- tests/test_statistics_cohorts_integration.py — фактический отказ stale index.
- tests/test_statistics_detail_worker.py — heartbeat/stdout/stderr privacy.

Добавлен этот отчёт. Production, кеш schema, публичные сигнатуры, UI/вычислительные
формулы и контракты экспорта не менялись. Предыдущие незакоммиченные файлы сохранены:
40 других зафиксированных исходников/тестов/документов совпадают с исходными SHA-256.
Несвязанные catalog snapshots и исторические diagnostics не включены в подготовку.

Ruff: 71 файл, 0 замечаний. git diff --check без замечаний; новые/untracked-файлы
отдельно проверяются на whitespace. Контактные литералы в source-only списке
относятся только к зарезервированным синтетическим доменам; значения не выводятся.
В список фиксации не входят CSV/ZIP/SQLite, индексы, окружения, пользовательские
выгрузки или диагностики. Это проверка выбранных исходников, не DLP-аудит всего диска.

**Git: READY FOR SOURCE REVIEW**, с явным P2 замечанием и ограничением широкого
прогона. Подготовлен только перечень исходников ниже; staging/commit/push/reset/clean
не выполнялись.

**INDEXED preflight: READY FOR SEPARATE PREFLIGHT**, а не подтверждение готовности
реальных данных/диска к построению. Индексация, реальный расчёт/экспорт и обучение
не запускались. Следующий этап требует отдельного согласования.

## 9. Проверяемые свидетельства
- [Точный план 19/37 модулей](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/run_plan.json)
- [Итоги завершённых запусков](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/verified_summary.json)
- [Первичная проверка прошлых падений](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/previous-regressions.xml)
- [Целевые наборы](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/target.xml)
- [Новые узкие проверки](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/supplementary.xml)
- [GUI после исправления изоляции](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/gui-isolation-verified.xml)
- [Хвост/GUI с семью разобранными падениями](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/followup.xml)
- [Последняя регрессия 118 passed](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/period-demography-verified.xml)
- [Ruff](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/ruff-final.json)
- [Сохранность исходных файлов](D:/CodexTestTemp/statistics-final-acceptance-912dc30fdca342558bea0891aafc52ae/preservation.json)

Свидетельства находятся вне Git; вместе с исходниками не фиксируются.

## 10. Полный перечень релевантных модулей плана

```text
tests/test_statistics_purchase_policy.py
tests/test_statistics_purchase_facts.py
tests/test_statistics_buyer_aggregates.py
tests/test_statistics_buyer_presentation.py
tests/test_statistics_rfm_core.py
tests/test_statistics_rfm_integration.py
tests/test_statistics_rfm_presentation.py
tests/test_statistics_cohorts_core.py
tests/test_statistics_cohorts_integration.py
tests/test_statistics_cohorts_presentation.py
tests/test_statistics_detail_contract.py
tests/test_statistics_detail_projection.py
tests/test_statistics_detail_rfm.py
tests/test_statistics_detail_backend.py
tests/test_statistics_detail_worker.py
tests/tabs/test_dataset_statistics_detail_export.py
tests/test_statistics_export.py
tests/test_statistics_html_export.py
tests/test_statistics_bundle_export.py
tests/test_dataset_statistics.py
tests/test_analysis_filter.py
tests/tabs/test_dataset_statistics_tab.py
tests/tabs/test_analysis_filter_tab.py
tests/test_statistics_index.py
tests/test_statistics_storage_safety.py
tests/test_statistics_slices.py
tests/test_statistics_presentation.py
tests/test_statistics_demography.py
tests/test_statistics_customer_scope.py
tests/test_statistics_categories.py
tests/test_statistics_charts.py
tests/test_raw_purchase_presentation.py
tests/test_purchase_identity_boundary.py
tests/tabs/test_purchase_identity_boundary_ui.py
tests/mindbox/test_adapters.py
tests/mindbox/test_identity.py
tests/test_customer_profile_service.py
```

## 11. Source-only перечень для последующей Git-фиксации

43 файла: предыдущие изменения статистики плюс тестовые исправления и этот отчёт. Это не staging.

Application (19):

```text
Application/dataset_statistics.py
Application/statistics_cache.py
Application/statistics_cohorts_core.py
Application/statistics_cohorts_presentation.py
Application/statistics_cohorts_validation.py
Application/statistics_detail_backend.py
Application/statistics_detail_contract.py
Application/statistics_detail_job.py
Application/statistics_detail_process.py
Application/statistics_detail_projection.py
Application/statistics_detail_rfm.py
Application/statistics_detail_storage.py
Application/statistics_export.py
Application/statistics_html_export.py
Application/statistics_query.py
Application/statistics_rfm_core.py
Application/statistics_slices.py
Application/tabs/dataset_statistics_tab.py
Application/tabs/statistics_detail_dialog.py
```

scripts (1):

```text
scripts/statistics_detail_export.py
```

tests (17):

```text
tests/tabs/test_analysis_filter_tab.py
tests/tabs/test_dataset_statistics_detail_export.py
tests/tabs/test_dataset_statistics_tab.py
tests/test_statistics_buyer_aggregates.py
tests/test_statistics_cohorts_core.py
tests/test_statistics_cohorts_integration.py
tests/test_statistics_cohorts_presentation.py
tests/test_statistics_customer_scope.py
tests/test_statistics_demography.py
tests/test_statistics_detail_backend.py
tests/test_statistics_detail_contract.py
tests/test_statistics_detail_projection.py
tests/test_statistics_detail_rfm.py
tests/test_statistics_detail_worker.py
tests/test_statistics_html_export.py
tests/test_statistics_index.py
tests/test_statistics_rfm_presentation.py
```

docs (6):

```text
docs/statistics_cohorts_contract.md
docs/statistics_cohorts_integration.md
docs/statistics_detail_backend.md
docs/statistics_detail_export_contract.md
docs/statistics_detail_gui.md
docs/statistics_final_acceptance.md
```
