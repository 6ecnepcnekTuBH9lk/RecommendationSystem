# Текущие данные Mindbox

## Операции

«Получить данные» обновляет только CustomerMerges, Actions и Orders.
«Обновить клиентов» получает только Customers, с отдельным периодом.
Все даты API — UTC, верхняя граница exclusive. Один процесс выполняет один
экспорт за раз. Общая межпроцессная блокировка исключает конкурирующие writers.
Секреты не входят в metadata; endpoint и конфигурация операций привязаны SHA-256.

## Дневное хранилище

`MindboxRaw/canonical/catalog.json` — атомарный каталог текущих источников.
Для Actions и Orders ключ — UTC-день. Значение содержит границы, время публикации,
transport/source kind, export ID, число частей и ссылку на raw-каталог.
Физические JSON parts находятся в `canonical/objects/<uuid>/<source>/`.
Новый запуск всегда переэкспортирует выбранные дни, включая существующие.

Экспорт скачивается в `.transport-*`, распаковывается существующим RawExportStorage,
потоково проверяется общими reader/adapters и переносится в новый объект. Только
потом `os.replace` переключает catalog.json. До этого старый источник доступен.
После commit старый canonical object удаляется; при прерывании cleanup повторяется
при следующем API job. Legacy-каталоги не участвуют в garbage collection.

Единица commit — компонент дня (Actions либо Orders). Если Orders дня не получен,
Actions этого дня может быть сохранён, но новый день не входит в training range
до появления обоих компонентов. Уже сохранённые дни не откатываются.

CustomerMerges — один текущий объект. При refresh диапазон равен объединению
старого и запрошенного: min(since), max(until). Покрытие не сужается. Ошибка новой
выгрузки сохраняет старый объект. История версий не накапливается.

## Ручная историческая база

Первичная загрузка: получить через API CustomerMerges с достаточно ранним merge_since
(для interactions можно выбрать только текущую неделю), затем вручную импортировать
большую пару Actions + Orders и при необходимости полный Customers. В дальнейшем
обновлять interactions и merges через API еженедельно, Customers — периодически.

Catalog schema v2 добавляет одну current manual_interactions пару. Catalog v1 читается
без изменения файла и нормализуется с manual_interactions=None; v2 записывается при
следующей mutation. Legacy raw не мигрирует и не удаляется.

Пара хранится в одном canonical/objects/<uuid> с подкаталогами actions и orders.
В каждом один part; source_kind=MANUAL, export_id=null, operation=MANUAL.
Период задаёт пользователь независимо от API-полей: UTC-midnight [since, until).
Длительность не ограничена одним днём. JSON не дробится по timestamps, в частности
firstAction.dateTimeUtc не определяет период Orders. Selection общий для каталога.

Под storage_lock оба файла копируются в .manual-interactions-*, затем каждый полностью
проверяется общими streaming reader/adapters и canonical CustomerIdResolver.
Текущее canonical CustomerMerges должно целиком покрывать ручной период.
Новый UUID object публикуется одной атомарной заменой catalog.json со всей парой.
Старый manual object удаляется только после commit. При cancel/error до commit старая
пара сохраняется; после гибели процесса confined staging и unreferenced objects
собираются следующей безопасной canonical операцией. Исходники пользователя не меняются.
Manual resume нет; pending API resume от независимой ручной операции не теряется.

Подробнее: [ручной импорт](MANUAL_IMPORT.md).

## Current training dataset

`canonical/training.json` — стабильная точка входа schema v5; содержит указатель
на canonical storage, а не копию изменяемого списка partitions. Loader строит
ChunkedTrainingBatch из согласованного catalog.json и сохраняет revision в batch_id.

Выбирается самый длинный непрерывный диапазон из одного manual interval и общих
дневных API Actions/Orders, целиком покрытых CustomerMerges. При равной длительности
выбирается более поздний диапазон. Дни за дыркой остаются на диске и не включаются
в заявленный период.

Внутри manual range приоритет MANUAL: перекрытые API partitions остаются на диске
и в каталоге, но не читаются training pipeline. Смежные API дни до/после manual могут
его продолжить. Например manual [01.01, 01.09) + API [01.09, 08.09) даёт [01.01, 08.09);
API только с 03.09 не закрывает разрыв после 01.09.

Manual Actions/Orders — одна длинная пара компонентов, без fake daily components.
Общая preparation получает каждый manual directory ровно один раз и отдельные
выбранные API directories. Диагностика дней учитывает длительность компонентов.
Summary показывает фактический current range и максимум updated используемых
источников отдельно для Actions и Orders.

Подготовка проверяет revision и удерживает store lock до завершения чтения raw.
Поэтому cleanup не удаляет файлы из-под training reader. Selection передаётся в
прежнюю общую preparation; order dedup, QC, product/identity resolution и BPR не заменены.
Пока API job удерживает блокировку, конкурирующая preparation получает ошибку занятости;
повторить её можно после окончания job.

## Customers

`canonical/customers.sqlite` — единственное актуальное хранилище профилей.
SQLite входит в Python: новые production-зависимости не нужны. JSON каждой записи
сохранён без потери точности Decimal. Primary key — Mindbox identity, разрешённая
существующим CustomerIdResolver. Новое состояние заменяет прежнее, если его
changeDateTimeUtc не старее сохранённого; повтор старого диапазона не откатывает
более новый профиль. При равной дате применяется последняя полученная запись.

Период разбивается на календарные месяцы; первый и последний могут быть неполными.
Каждый месяц целиком скачивается и затем применяется одной SQLite-транзакцией.
Вместе с upsert фиксируются summary и receipt для resume. Ошибка parsing/validation
откатывает только текущий месяц. Предыдущие месяцы остаются. Удалённые страницы БД
повторно используются; ежемесячной перезаписи полного 7 GB JSON нет.

Месяцы выгружаются последовательно. Timeout 14 400 секунд применяется к каждому
экспорту отдельно и меняется через `--timeout`. Это выбранный запас для эксплуатации,
а не заявленный Mindbox SLA. Ошибка передаётся в GUI отдельной безопасной категорией
с источником и периодом, без URL и payload.

При обновлении клиентов новые объединения применяются и к уже сохранённым ключам:
просматриваются ID порциями по 1024, raw читается только для затронутых объединением
профилей. Interaction job самостоятельно Customers DB не изменяет. Счётчик в summary
относится к профилям на момент последнего customer commit; новые объединения будут
учтены в нём при следующем обновлении клиентов. При построении contact index всегда
применяется актуальный resolver.

Количество клиентов — COUNT(*) по уникальным ключам, сохранённое в summary при commit.
Startup GUI читает только summary, не профили. Покрытие — объединение успешно
обработанных окон; разрывы выводятся отдельными интервалами, а не сплошным диапазоном.
Это покрытие запросов по последнему изменению, не исторические слепки клиентов.

Ручной full Customers JSON поступает в тот же SQLite representation. Новая база
строится отдельно и полностью проверяется; после закрытия транзакции атомарно
заменяет предыдущую. Неизвестный период ручного файла не выдумывается. Требование
валидного сохранённого API CustomerMerges сохранено, включая legacy manifests.

## Resume и остановка

`canonical/jobs/<uuid>/state.json` хранит запрос, selection и committed-компоненты.
State публикуется до первого запроса. После каждого commit записывается checkpoint.
Receipt в каталоге/SQLite позволяет пережить остановку между commit и checkpoint:
уже опубликованный компонент не скачивается повторно. Новый job, в отличие от resume,
заново получает выбранный диапазон. Customers также поддерживает resume.

SQLite rollback journal восстанавливает незавершённый месяц после завершения процесса.
Read connections допускают hot-journal recovery, но не создают отсутствующую БД.
Служебные state/receipts малы; старые гигантские raw snapshots не сохраняются.

## GUI и совместимость

Четыре read-only поля отображают persisted summary; во время job меняются только
связанные с ним runtime-поля. После любого исхода metadata перечитывается фоновым
worker. QProcess/cancel/close eventFilter сохранены. Progress bar — idle «Не запущено»
или busy «Идёт загрузка», без процентов. Журнал получает только проверенные события;
State/Manifest paths, UUID, traceback и произвольный stdout туда не попадают.

Старые training manifests v2/v3/v4 и profile manifests v1/v2 по-прежнему читаются.
Существующий resume старого daily batch сохраняется. Автомиграции и удаления legacy
JSON нет: они не считаются canonical partitions без явного нового API получения.
Для legacy-наборов используется их явный manifest; canonical summary их не подменяет.
Старые низкоуровневые snapshot-функции оставлены для совместимости, новые GUI/CLI jobs
их не используют. Canonical profiles доступны существующему contact-index loader
по пути `canonical/customers.sqlite` (metadata projection v3).

Новые ручные Actions/Orders пишутся только в canonical storage одной interval-парой.
Старый ручной interaction manifest по-прежнему можно передать явно в training pipeline.
GUI имеет отдельный manual period; Customers к нему не привязан. Во время ручной
загрузки interactions меняются только runtime-поля Actions/Orders, после любого исхода
перечитывается persisted summary.

## Официальный контракт и ограничения

Проверено 18.09.2026:

- [Рекомендации API](https://developers.mindbox.ru/docs/api-usage-recommendations):
  стандартные HTTP/JSON библиотеки, DNS/TLS, совместимость с дополнительными полями.
- [Экспорты](https://developers.mindbox.ru/docs/exports-overview): очередь, polling,
  NotReady/Ready, все части из exportResult.urls; одновременно до двух задач.
- [Customers](https://developers.mindbox.ru/docs/export-customers): фильтр по последнему
  изменению, since включительно и till исключительно; создание тоже является изменением.
- [CustomerMerges](https://developers.mindbox.ru/docs/export-customer-merges): отдельные
  события объединений; поля клиента соответствуют моменту объединения.

В изученном контракте обычного Customers export нет подтверждённого deletion tombstone.
Адаптеры проекта также его не обрабатывают. Отсутствие клиента в месяце не означает
удаление. Incremental update поэтому не удаляет отсутствующие профили; ручной полный
snapshot заменяет dataset целиком. Поля _isDeleted из аналитических datasets другого
API нельзя автоматически переносить в этот контракт.

Реальные API-запросы и 7 GB импорт при разработке не выполнялись. Synthetic tests
проверяют публикацию, rollback, resume, coverage, identity/upsert и GUI. Для месячного
commit требуется место под скачанные parts и rollback journal. В SQLite остаются
свободные страницы для последующих обновлений, отдельные исторические базы не создаются.
