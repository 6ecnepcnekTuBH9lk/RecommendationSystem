# Ручная загрузка Mindbox JSON

## Первичная загрузка и дальнейшее обновление

1. Получить CustomerMerges через API с достаточно ранним «Началом истории объединений».
   Период API Actions/Orders при этом может быть только текущей неделей.
2. Вручную выгрузить из Mindbox исторические Actions.json и Orders.json за одинаковый
   большой период, например 01.01.2025 → 01.09.2026, и импортировать оба файла вместе.
3. При необходимости импортировать полный Customers.json отдельной кнопкой.

Далее получать новые Actions/Orders и актуализацию CustomerMerges через API, например
раз в неделю, а Customers обновлять через API периодически. Новые API interactions
по-прежнему сохраняются по UTC-дням. См. [canonical storage](CANONICAL_STORAGE.md).

## Период и GUI

В блоке «Ручная загрузка из Mindbox» выберите оба JSON и укажите **отдельный**
«Период выгрузки Actions + Orders». Он не связан с API interaction dates, началом
истории объединений или периодом Customers. Один набор selection controls остаётся
общим для API и manual.

Период — доверенная metadata пользователя: [since 00:00 UTC, until 00:00 UTC).
Укажите те же границы, с которыми формировали экспорт. Верхняя граница исключительная.
В интерфейсе Mindbox UTC+03 полночь UTC соответствует 03:00.
Ограничения в один день нет. Период не вычисляется из JSON, Orders не разбивается
по firstAction.dateTimeUtc: дата создания заказа может быть вне периода экспорта.

Требуется текущий **canonical** CustomerMerges, целиком покрывающий ручной период.
Legacy API manifests не заменяют его для нового interaction import.
Если покрытия нет, сначала обновите объединения через API; отдельной кнопки
«только merges» нет. Предварительная проверка metadata работает в GUI worker,
окончательная проверка повторяется в процессе импорта под блокировкой хранилища.

Во время manual interactions только поля «Действия» и «Заказы» показывают выполнение.
После успеха, ошибки или отмены все persisted summaries перечитываются.
Общий progress остаётся indeterminate; технические paths/markers не попадают в журнал.
Незавершённый API resume сохраняется после независимой ручной операции.

## Хранение и атомарность пары

Новый импорт создаёт одну пару:

~~~text
canonical/objects/<uuid>/actions/actions_part_001.json
canonical/objects/<uuid>/orders/orders_part_001.json
~~~

В catalog.json schema v2 поле manual_interactions содержит since/until/updated
и metadata обоих источников: directory, parts=1, source_kind=MANUAL,
export_id=null, operation=MANUAL. Selection остаётся общим полем каталога.
Одновременно существует только одна текущая manual-пара. Стабильная точка входа
training — canonical/training.json; новые training_batches не создаются.

Под storage_lock файлы копируются порциями по 1 MiB в .manual-interactions-*.
Каждый файл полностью проверяется строгим потоковым reader и теми же adapters,
что API. CustomerIdResolver строится по canonical CustomerMerges. Никакой физической
фильтрации raw по selection нет. JSON не материализуется целиком: сохраняется
одна текущая запись; память resolver зависит от числа идентификаторов объединений.

После проверки обоих файлов каталог staging переносится в новый UUID object.
**Единственный логический commit — атомарная замена catalog.json** со ссылками сразу
на оба файла. Только после commit удаляется прежний manual object.
Новый явно заданный диапазон может отличаться от старого, включая сужение.

Ошибка Orders, повреждённый JSON, ошибка commit или отмена до переключения каталога
сохраняет прежнюю пару и metadata. Исходные пользовательские JSON только читаются,
никогда не перемещаются и не изменяются. При обычной ошибке staging удаляется;
после принудительного завершения процесса свои confined staging/unreferenced objects
собираются при следующей безопасной canonical операции под той же блокировкой.
После commit новая пара уже опубликована, даже если процесс не успел сообщить об успехе.
Manual import не имеет resume: повторный запуск снова читает выбранные файлы.

На время замены требуется место под старую и новую пару. История больших manual
снимков не накапливается. Пользовательские исходники и legacy directories не собираются.

## Manual + API, перекрытия и разрывы

Manual interval считается одним непрерывным источником. Внутри него приоритет manual:
API partitions сохраняются, но исключаются из current training dataset.
Смежные общие API Actions/Orders дни до или после manual range могут его продолжить.

Пример: manual [01.01, 01.07) + API [01.07, 03.07) → current [01.01, 03.07).
Если API начинается только 03.07, промежуток не считается покрытым.
Выбирается самый длинный непрерывный диапазон, при равной длительности — более поздний.
Компоненты вне покрытия CustomerMerges не используются.

Preparation получает manual Actions directory один раз, manual Orders directory один раз
и отдельные API directories выбранного диапазона. Виртуальных дневных manual components
нет. Общие selection, order dedup, identity/product resolution, QC и BPR сохранены.
Диагностика canonical batch считает длительность покрытия в днях, а не количество пар.
Summary показывает выбранный непрерывный период и максимальное время публикации
реально используемых источников отдельно для Actions и Orders.

Общий order dedup сохраняет прежнюю семантику: идентичный повтор заказа пропускается;
различающийся snapshot одного order_id вызывает strict error либо QC BLOCK в diagnose.
Raw IDs и payload в журнал не выводятся.

## Customers и совместимость

Manual Customers использует canonical_customers.import_full(): отдельная SQLite-база
строится и проверяется, затем атомарно заменяет canonical/customers.sqlite.
Период полного ручного снимка неизвестен и не привязывается к manual interaction dates.
Эта архитектура не меняется. Canonical merges предпочтительны; прежняя совместимость
Customers с сохранёнными legacy источниками объединений остаётся.

Canonical catalog v1 читается без записи на диск, с manual_interactions=None.
Следующая mutation записывает v2. Старые training manifests v2/v3/v4 и customer profile
snapshots v1/v2 остаются читаемыми; explicit legacy manifest можно передать в прежний
training pipeline. Автомиграции и смешивания legacy с canonical нет.

## CLI без сети

~~~text
python scripts/mindbox_manual_import.py interactions --actions Actions.json --orders Orders.json --since 2025-01-01 --until 2026-09-01
python scripts/mindbox_manual_import.py customers --customers Customers.json
~~~

Для interactions доступны те же repeated selection arguments, что у API export-daily.
--merge-since больше не требуется и не принимается. Оба режима поддерживают --raw-root.
CLI не загружает credentials и не обращается к API. После успеха interactions печатает
Manifest: <raw-root>/canonical/training.json для существующего GUI protocol.
