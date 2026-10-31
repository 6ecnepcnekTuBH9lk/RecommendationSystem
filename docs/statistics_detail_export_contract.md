# STAT-DETAIL-EXPORT-01A — контракт детальных данных покупателей

Текущая реализация этапа 01B1 описана в [statistics_detail_backend.md](statistics_detail_backend.md).
Ниже сохранён исходный контракт и отчёт этапа 01A; пометки «01B не начат»
и `NOT_COMPUTED` относятся к состоянию 01A. Backend 01B1 заполняет RFM.

## Назначение и граница этапа

Контракт FORMAT_VERSION=1 предназначен для последующей разрешённой выгрузки
заказчику реальных Mindbox ID и доступных контактов. Обезличивание, хеширование
ID вместо исходных значений и восстановление покупателей из агрегированного DTO
не применяются. Проекция сохраняет реальные строки и Decimal.

01A добавляет только типы, pure projection, синтетические тесты и этот документ.
Чтения реальной истории/Customers, создания CSV/XLSX/ZIP, постоянной базы,
статистического индекса, изменения GUI/кеша/расчётных ядер нет. 01B не начат.

## Подтверждённые источники и типы

CustomerRecord и adapt_customer уже предоставляют source_customer_id,
customer_id, change_datetime_utc, first_name/last_name/middle_name, email,
mobile_phone, birth_date (date), sex. Дополнительно доступны iana_time_zone,
time_zone_source, custom_fields, last_activated_card, segmentations, balances,
discount_cards, subscriptions. Динамические структуры глубоко замораживаются.
Из дополнительных структур в этот контракт включены только subscriptions;
остальные произвольно не интерпретируются и не выгружаются.

Реальные raw-пути: ids.mindboxId, changeDateTimeUtc, firstName, lastName,
middleName, email, mobilePhone, birthDate, sex, subscriptions. ID адаптером
identifier переводится в str без усечения/удаления ведущих нулей; телефон —
также через identifier. Поле email текстовое; его наличие не подтверждает
валидность адреса или разрешение на коммуникацию.

OrderLineRecord/adapt_order содержат order_id (ids.mindboxId),
retail_order_id (ids.retailOrderId), order_datetime_utc
(firstAction.dateTimeUtc), customer_id/source_customer_id,
channel_external_id/channel_name (firstAction.channel), line_id/line_number,
product.namespace/value, product_name (product.name), quantity,
base_price_per_item, price_of_line, line_status. Числа — Decimal; базовая цена
nullable. order_total_price/delivery_cost присутствуют в адаптированной записи,
но не подменяют сумму принятых позиций и не включены в данный словарь.

Отсутствующий line.id существующий адаптер заменяет внутренним fallback ID.
По одному OrderLineRecord нельзя доказать, был ли ID исходным или fallback;
это ограничение не маскируется. Для уникальности строк добавлен line_ordinal.

Текущий statistics_query.IndexedInput использует локальные customer/order SID,
не сохраняет полный исходный OrderLineRecord и контакты. Он не является
источником этой выгрузки. DetailMetadata отклоняет пространство INDEXED,
проекции не принимают integer-ID и не восстанавливают реальные ID через модели.
01B должен использовать canonical-источники и существующие source adapters.

## Типы и программный API

statistics_detail_contract.py определяет Identifier(str), Money(Decimal),
DetailPeriod, DetailMetadata, ProfileSelection, BuyerPolicyMetrics,
PersonalRfm, ClientRow, OrderHeader, OrderPolicyRow, OrderRow, OrderLineRow,
OrderBundle и DetailCells. Записи frozen/slots, repr=False. Прямое чтение
полей/ячеек раскрывает разрешённые данные, автоматический repr их не раскрывает.
Это не защита от намеренного логирования .values/asdict — оно запрещено.

statistics_detail_projection.py предоставляет:

- select_profile(canonical_id, adapted_candidates, source_available=..., check=...);
- project_client(canonical_id, selection, metadata, strict=..., extended=..., rfm=...);
- header_from_line(adapted_line);
- project_line(adapted_line, ordinal, check=...);
- project_order(prepared_fact, adapted_lines, metadata, header=..., check=...);
- buyer_metrics(canonical_id, unique_history_facts, metadata, check=...);
- row_cells(row) и csv_cells(row) — последовательные ячейки в стабильном порядке.

Ни одна функция не читает файлы или сеть и не пишет результат. project_order
не строит новый PurchaseOrderFact; вызывающий код обязан передать факт именно
этих неизменяемых адаптированных строк. Проверяются ID/пространство/UTC,
одинаковый заголовок всех позиций, число принятых строк и классифицированные
группы. Денежные итоги копируются из факта, не пересчитываются по новой формуле.
Переиспользовать факт от другого снимка с совпадающим ID запрещено контрактом.

OrderBundle возвращается только после проверки всего заказа; ошибки не дают
частично подготовленной таблицы позиций. При нуле позиций обязателен явный
OrderHeader из уже проверенного raw-заголовка: adapt_order возвращает пустой
кортеж и сам не сохраняет отдельный header. Источник 01B использует существующие
identifier/timestamp/text и CustomerIdResolver для этой ветки, не выдумывает ID.

## Три связанные таблицы

### Clients — одна строка на канонического владельца выгружаемых заказов

PK customer_id. Набор клиентов определяется выбранными Orders после режима
включения и периода событий, а не наличием профиля/контактов. При ALL_SELECTED
сюда входят и владельцы заказов без принятой покупки; их наблюдаемые STRICT/
EXTENDED показатели равны нулю, если полной истории покупок действительно нет.
Customers без выгружаемых заказов автоматически не добавляются.

177 колонок: 21 базовая, две группы персональных показателей по 24 колонки и
12 RFM-групп по 9 колонок (2 политики * 3 окна * 2 валюты). Это широкая CSV-
таблица фиксированного формата, не новые поля обычного агрегированного DTO.
Имена, порядок, nullable-типы и фактическая готовность перечислены ниже.

### Orders — одна строка на уникальный выбранный заказ

PK order_id. FK customer_id -> Clients.customer_id. Строка содержит источник
customer ID, UTC firstAction, канал, retail ID, общее число исходных позиций,
RETURN/CANCELLED/прочие статусы и отдельные результаты STRICT/EXTENDED.
Валюты раздельны по политикам: F в RUB и CP в KZT дают STRICT=RUB и
EXTENDED=mixed. Единую валюту всего заказа выдумывать нельзя.

ALL_SELECTED — все выбранные заказы событийного периода, включая Return,
отмены, промежуточные статусы и пустые заказы. PURCHASE_EITHER — только заказы,
принятые хотя бы EXTENDED. Исключение применяется ко всему OrderBundle:
нельзя оставить его позиции без Orders или удалить покупателя-владельца.

### OrderLines — все исходные адаптированные позиции выбранного заказа

PK (order_id, line_ordinal), FK order_id -> Orders.order_id; customer_id также
совпадает с Orders.customer_id. line_id/line_number сохраняются, но не считаются
уникальными: источник допускает повторы, а ID может быть fallback адаптера.
line_ordinal начинается с 1, следует исходному порядку выбранного снимка.

product_id — исходное product.value с namespace, включая вариант/размер и
ведущие нули. Каталог, актуальный артикул и ProductResolver не используются.
Неизвестный товар не исчезает. Если существующий адаптер не допускает
отсутствующий/неподдержанный product ID, подготовка завершается ошибкой, а не
создаёт фиктивный товар. Если уже адаптированная запись имеет явно допустимый
альтернативный namespace без валютного соответствия, currency=NULL.

Никаких товарных фильтров и фильтрации строк до формирования детали нет.
Строковые/NaN/отсутствующие обязательные числа отклоняются исходным адаптером.
Конечные дробные, нулевые и отрицательные quantity он допускает: они сохраняются,
но покупкой не признаются существующей политикой. Конечная отрицательная цена
также сохраняется как исходное число с NEGATIVE_UNVALUED; факт помечает сумму
INVALID_PRICE. Это не молчаливое исправление/удаление строки. Факт должен быть
построен по всем адаптированным позициям, в отличие от статистического драйвера,
который отдельно исключает дробные строки из своих raw-диагностик.

## CustomerMerges и выбор профиля

Единственный механизм identity — существующий CustomerIdResolver по
зафиксированному CustomerMerges. Адаптеры уже разрешают customer_id; pure
projection не выполняет повторное разрешение. Заказы разных source IDs
суммируются по canonical ID. Совпадение телефона/email не объединяет клиентов.
Граф является снимком выбранного источника Merges, а не восстановленной
исторической identity на дату анализа.

canonical/customers.sqlite: profiles(id TEXT PRIMARY KEY, changed TEXT, raw TEXT).
_upsert ключует профиль текущим resolver.resolve(ids.mindboxId); обновляет запись
при excluded.changed >= profiles.changed. Поэтому последняя дата выигрывает,
при равной дате — последнее поступление. База хранит один текущий raw, не историю
контактов и не все прежние конфликтующие профили. Уже перезаписанный конфликт
нельзя обнаружить или восстановить. AVAILABLE означает выбранную доступную
запись, а не доказанное отсутствие конфликтов за всё время.

У базы нет аттестованной версии графа Merges. После отдельного обновления Merges
ключи могут ещё требовать реканонизации; массовому reader нельзя слепо считать
поиск только canonical PK полным. План 01B ниже учитывает alias-ключи.

Для нескольких предоставленных кандидатов одного canonical ID действует
строгая детерминированная политика:

1. Выбирается максимальный известный change_datetime_utc.
2. Различающийся payload при равной максимальной дате -> CONFLICT, контакты NULL.
3. Неизвестная дата при отличающемся payload также -> CONFLICT: неизвестная запись
   могла быть новее. Несовместимые записи без дат не выбираются произвольно.
4. Идентичный payload допускает выбор; при равенстве предпочитается исходный ID,
   равный canonical, затем лексикографически меньший source ID. Значения контактов
   в таком tie не различаются. Payload включает адаптированные дополнительные
   структуры, не только email; это консервативное правило.
5. Более позднее однозначное обновление разрешает конфликт более старых
   кандидатов с известными датами. Неизвестные конфликтующие даты его не разрешают.

Алгоритм select_profile потоковый: держит только лучших known/unknown кандидатов
и признаки конфликтов, не список всех профилей/алиасов.

MISSING — Customers доступен, но профиль отсутствует; SOURCE_UNAVAILABLE —
источника нет, profile_present=NULL; CONFLICT — записи есть, но выбрать нельзя,
profile_present=True и контактные поля NULL. Во всех случаях канонический ID и
покупательская активность сохраняются. Ошибка/повреждение доступной базы не
маскируется отсутствием: 01B должен остановиться безопасно.

Данные Customers текущие. Если change_datetime_utc позже анализа, контакты
сохраняются с profile_after_analysis_date=True; утверждения «такими были
контакты на дату анализа» нет. Age рассчитывается по calendar_age на as_of_date;
будущая дата рождения сохраняется с FUTURE и age=NULL. Некорректная дата,
отклоняемая adapt_customer, в 01A не исправляется произвольно.

## Контакты и разрешения

Пустые/пробельные текстовые контакты становятся NULL, внешние пробелы убираются
по действующей практике profile service. Номер, регистр email и имена не
хешируются и не нормализуются до другой identity. has_email/has_phone означают
доступность непустого значения в подготовленной строке, не его валидность,
подтверждение или наличие у покупателя при недоступном/конфликтном профиле.

subscriptions_json — точная JSON-сериализация доступной структуры subscriptions
через существующий canonical encoder, без потери Decimal. Произвольные
isSubscribed/channel/brand из этой структуры не трактуются как согласие.
Документированного правила email/SMS-разрешений в изученном контракте нет:
email_permission=sms_permission=UNKNOWN даже при контактах и непустой подписке.
ALLOWED/DENIED в этой версии не выдумываются. Будущий документированный mapping
нуждается в отдельной версии/тестах источника и не следует из этих флагов.

## Периоды и персональные показатели

DetailMetadata требует явные as_of_date, event_period, identity_space и
orders_coverage; системная дата не используется. event_period — обе включительные
даты либо без ограничения. Конечная дата событий не может быть позже отсечения.
Дата отсечения исключает все события после конца этого UTC-дня из показателей.

orders_coverage — source интервалы [since,until), нормализованные в UTC и
объединённые shared_intervals; effective_coverage обрезано отсечением. FULL
означает один непрерывный доступный интервал до конца as_of_date; начало —
начало доступной истории, не начало lifetime. PARTIAL — разрывы/обрыв до
отсечения, NONE — нет подтверждённых интервалов. Состояния не заменяются нулём.

buyer_metrics принимает поток уникальных фактов ОДНОГО покупателя по всей
предоставленной истории, а не только event_period. Контекст/канонический ID
проверяются. Факты после отсечения не учитываются. Для каждой политики хранятся
счётчики, первые два ключа (timestamp, order_id), последний timestamp, суммы и
три счётчика окон. Все полные заказы клиента не сохраняются; порядок не важен.

Подтверждённые персональные поля 01A:

- orders — число принятых уникальных заказов; accepted_lines/units — из fact;
- первая/последняя наблюдаемые покупки; repeat_orders=max(orders-1,0);
- mean_interval_days=(last_utc_date-first_utc_date).days/(orders-1) при orders>1.
  Это телескопическая сумма всех соседних календарных интервалов после сортировки,
  эквивалентная BuyerAggregates, включая два заказа в один день;
- orders_12m/24m/48m используют rfm_windows, включая точные календарные границы;
- has_second_purchase/days_to_second_purchase и first_observed_cohort используют
  первые две наблюдаемые покупки по каждой политике, независимо от event_period;
- valued/unvalued orders и известные суммы отдельных RUB/KZT — из policy facts.

Нет accepted покупок: first/last/cohort/interval=NULL, наблюдаемые counts=0.
Это не доказанное отсутствие покупок вне покрытия. Известные суммы — подытоги
только фактов с monetary_available; mixed/unknown/INVALID_PRICE-заказы не
разбиваются заново по валютам позиций. Возвраты автоматически не вычитаются.
Не использовать totalPrice, не умножать priceOfLine на quantity, не называть
эти суммы чистой выручкой/LTV.

observed_amounts_complete означает отсутствие unvalued среди наблюдаемых
заказов; financial_complete требует ещё FULL покрытия и отсутствия заказов вне
сертифицированных интервалов. complete_amount_rub/kzt=NULL при неполноте.
known_amount_* остаётся достоверным наблюдаемым подытогом даже при неполноте.
Покупки вне интервалов не выбрасываются: orders_outside_coverage показывает их,
FULL понижается до PARTIAL. При NONE полнота также не заявляется.

Суммирование готовых Decimal расширяет локальную точность при необходимости,
без округления малых слагаемых большого подытога. Уже сделанное выше по цепочке
округление PurchaseOrderFact восстановить нельзя. Для parity с существующими
Decimal-агрегатами фиксируется общий контекст; если он у агрегатора приводит к
округлению, расхождение нужно диагностировать, а не искусственно округлять
детальную выгрузку. Денежная сериализация не использует float.

## Индивидуальный RFM — точная архитектура 01B

RfmAggregates.result() содержит только групповые результаты, не mapping
customer->score. Из него нельзя восстановить индивидуальные сегменты. В 01A
PersonalRfm лишь определяет тип и выходные места: NOT_COMPUTED и NULL, без
фиктивных баллов/UNSCORABLE вместо отсутствующего расчёта. Переданные готовые
RFM-строки связываются с ID, датой, identity_space и thresholds_token; проверяются
размерности, диапазоны и согласованность статуса доступности. Сами баллы pure
projection не вычисляет.

Предложенная export-only финализация:

1. Во временном состоянии сохранить компактные принятые policy facts. Обойти их
   по canonical ID и времени; один покупатель в памяти. База RFM — все покупатели
   политики за LAST_48M, не только выгружаемые владельцы событий/профилей/контактов.
2. Получить те же окна rfm_windows и сырые наблюдения R/F/M по существующей
   _observation/_Buyer/_Window-методике, через небольшой export-only адаптер.
   Не вызывать result() отдельного одно-клиентского RfmAggregates ради баллов:
   это даст квантили одного клиента вместо общей базы. В 01B выбрать стабильную
   точку доступа к этим pure-примитивам, сохранив определения и parity-тесты;
   текущие private API не объявляются новым публичным контрактом.
3. R — календарные дни от последней покупки в 48 месяцах, в валютной базе —
   последней покупки своей валюты. Достоверность R определяется непрерывным
   суффиксом покрытия до отсечения и последней датой, как reliable_since ядра.
4. F — заказы соответствующего окна, включая 0 у покупателей базы 48M;
   M — полная сумма своей валюты только при полном покрытии окна и отсутствии
   ЛЮБОГО unvalued policy-заказа в этом окне. Неполный заказ другой валюты тоже
   консервативно делает M недоступной. Неизвестная валюта не создаёт RUB/KZT-базу.
5. Для policy/window/currency сформировать дисковые weighted histograms: R —
   только достоверные recency, F — полная валютная база, M — только доступные M.
   При неполном окне окончательные баллы недоступны, как в ядре.
6. Общие пороги — nearest ranks ceil(k*N/5), k=1..4; сохранять совпадающие пороги.
   Для ограниченной RAM применим отсортированный по точным значениям дисковый
   histogram и накопленные counts; первый cumulative >= rank даёт cut. Денежная
   сортировка/группировка должна использовать Decimal-collation/canonical Decimal,
   не CAST AS REAL. Эта дисковая реализация обязана пройти parity с действующей
   empirical_thresholds, включая ties, нулевые значения и разные базы.
7. Применять действующие rfm_score и rfm_segment, сравнение строго > cut,
   направление R противоположно F/M. Сохранять None и UNSCORABLE согласно ядру.
   За пределами 48M/валютной базы использовать OUTSIDE_BASE/NOT_IN_CURRENCY_BASE,
   не создавать выдуманные оценки. thresholds_token связывает общий population,
   coverage, дату, политику и версию алгоритма даже при пустых порогах.
8. После финализации выдавать строки Clients только для владельцев экспортных
   событий. Пороги остаются от полной исторической базы, а не пересчитываются
   после отбора клиентов с email/телефоном.

Инварианты 01B: одинаковые окна/базы/coverage/пороги дают индивидуальные значения,
которые при группировке воспроизводят score_counts/segments ядра; одинаковые
сырые значения не разделяются квантилями; missing M не превращается в 0 или
балл 1; полное пустое короткое окно отличается от неизвестного наблюдения.
В 01A такой финализатор и дисковые histograms не создаются.

## Согласованность и выбор уникального заказа

Для будущего detail reader принимается строгий режим существующего
OrderSnapshots.accept(raw,diagnose=False): одинаковые дубли игнорируются,
конфликтующие снимки останавливают подготовку безопасным OrderSnapshotConflict.
Нельзя выбирать «последний статус» без отдельного контракта. В отличие от
обычной статистики с diagnose=True, изменившийся снимок здесь не замалчивается
предупреждением. Уникальность проверяется на всей допустимой истории до
отсечения ПЕРЕД отбором event_period/политики. Первый Return не подменяется
поздним F того же ID. Детальная выгрузка конфликтной истории не публикуется.

Сравнения с агрегированной статистикой допустимы только при одинаковых:
источнике/поколении, выбранных уникальных снимках, историческом и событийном
периодах, политике, валюте и отсутствии товарных ограничений. Совпадают число
принятых заказов/позиций, units, известные суммы при том же Decimal-контексте,
RFM-окна и месяцы первых наблюдаемых покупок. Профиль/контакты на эти числа не
влияют. Не равнять всю историю с пользовательским срезом BuyerAggregates или
текущими товарными legacy-показателями.

## Потоковый конвейер 01B и ресурсы

Чтение -> расчёт/временное состояние -> последовательная запись разделяются.
Существующий агрегированный DTO и его кеш не расширяются контактами.

1. Создать отдельный worker/process и закрытый временный job-каталог. Проверить
   его размещение, ACL/права и бюджет ДО записи ID. Использовать применимые
   существующие storage-safety helpers; никакой постоянной второй canonical базы.
2. Зафиксировать catalog/selection/Merges/Customers snapshot и удерживать
   соответствующую reader-защиту; прерывать результат при изменении источника.
   Существующий CustomerIdResolver держит O(A) графа алиасов — это явный базовый
   расход, не обещание полностью постоянной RAM.
3. Один проход canonical Orders через iter_export/adapt_order. Дисковый mapping
   вместо dictionary OrderSnapshots.fingerprints сохраняет существующие
   semantic_fingerprint и conflict semantics, а не реализует новый dedup/parser.
   Для отбора держать один raw order/tuple строк; использовать все адаптированные
   строки и один build_purchase_order_fact, без каталога/товарных фильтров.
4. Временный SQLite только задания: seen IDs/fingerprints, компактные policy
   facts по customer/order/date, владельцы event_orders, заголовки и позиции
   выбранных событий, персональные features/RFM thresholds. Raw JSON заказов и
   контакты туда не копируются. Записи пакетами, ограниченный cache/temp sorting.
   Индексы здесь — внутреннее временное состояние задания, не STATISTICS INDEXED;
   в 01A оно не создаётся.
5. Обход facts ORDER BY canonical customer/time/order ID даёт buyer_metrics для
   одного покупателя и first-observed cohort по всей истории. Дополнительные
   проходы только по временному состоянию, не по canonical Orders.
6. Из-за отсутствия версии Merges в profiles сначала потоково пройти только
   profiles.id, разрешить каждый ключ текущим resolver и записать временную
   ссылку (canonical_id, source_db_key) для нужных владельцев событий. Затем
   сгруппировать ссылки по canonical ID и читать raw по существующему PK
   ограниченными пакетами, адаптировать через adapt_customer, подтвердить
   соответствие raw identity/ключа и применить select_profile. Это учитывает
   ещё не переканонизированные aliases без полного списка raw профилей в RAM.
   Для каждого клиента записи освобождаются сразу после подготовки/записи.
7. Customers читать mode=ro, query_only, транзакцией под reader lock. connect()
   из canonical_customers использует mode=rw для восстановления writer и для
   этого reader не подходит. Переиспользовать гарантии _database_version/
   _reader_lock profile service: не игнорировать WAL/journal или обновление
   через immutable=1. Длительное удержание защиты может задержать writer;
   отмена должна быстро освобождать соединения и lock, GUI-поток не участвует.
8. ZIP создаётся приватным .partial, его entries пишутся последовательно из
   временных строк: clients.csv, orders.csv, order_lines.csv, manifest.json.
   Одновременная запись нескольких ZipFile entries не предполагается. Clients
   выдаются после общих порогов; порядок ID/исходных ordinal детерминирован.
   Только после успешных проверок ссылок, counts и повторной аттестации источника
   файл атомарно публикуется. Это не архив, создаваемый текущим этапом.

Память: O(A) существующего resolver + ограниченный DB cache/batch + максимальный
размер одного заказа/профиля + компактные features одного покупателя. Не хранить
список заказов/позиций/готовых клиентов целиком. Группа aliases обрабатывается
потоково, а не собирается в list.

Диск: O(U) seen/facts + O(D) позиций событий + O(B) features/owners + O(P) ссылок
нужных profiles + сортировочные страницы + закрытый итоговый ZIP. U — уникальные
исторические заказы, D — строки событий, B — покупатели, P — ссылки профилей.
Точный байтовый бюджет зависит от длин ID/полей и должен оцениваться 01B до
запуска; гигабайты на реальных данных сейчас не измерялись. Секретные контакты
не должны попадать в stderr, SQL trace, прогресс или имена временных файлов.

На success/cancel/error: закрыть ZIP/DB/reader lock, удалить только файлы своего
проверенного приватного job-каталога и SQLite sidecars, не трогать canonical,
прошлый успешный архив или диагностические каталоги проекта. Перед удалением
проверять resolved root/отсутствие перенаправляющих ссылок. Очистка — будущая
реализация 01B; текущая задача ничего подобного не создаёт и не удаляет.

## Будущий wire-format и конфиденциальность

Архив: clients.csv, orders.csv, order_lines.csv, manifest.json. UTF-8 BOM
(utf-8-sig), разделитель ;, CRLF, стандартное CSV quoting для ;/кавычек/переносов.
NULL — литерал \N, bool 1/0, даты ISO, UTC datetime с +00:00, Decimal fixed-point
без exponent/float. NUL отклоняется. Формат версии 1 сохраняет 177/23/18 колонок.

**CSV quoting само по себе не сохраняет текстовый тип в Excel и не защищает
формулы.** Поэтому csv_cell обратимо добавляет один апостроф ко ВСЕМ строковым
полям, включая ID, контакты, enum и JSON. Числовой Mindbox ID не превращается в
число, строка =.../пробел+... не становится формулой. При машинном чтении нужно
снять ровно один префикс по manifest/schema (decode_csv_text), не преобразовывать
ID в числа. Исходный апостроф удваивается таким транспортным представлением,
исходный \N отличается от NULL. В Excel апостроф может быть видимым.

Это обратимое транспортное экранирование, не обезличивание. ClientRow/OrderRow/
OrderLineRow содержат исходные ID. Альтернативный необработанный CSV для
машинного импорта потребует отдельного явно указанного режима/инструкции Excel
«импортировать ID как текст»; нельзя молча обещать типизацию обычного CSV.

Manifest содержит format_version, схему/nullable/колонки, кодировку/разделитель,
NULL/text prefix, periods/as_of/фактическое coverage, snapshot/provenance tokens,
контекст Decimal/RFM population, число строк и общие признаки качества.
Индивидуальные ID, контакты, raw-профили, примеры строк и конфликтующие значения
в manifest/логах/техническом отчёте запрещены. Исключения projection содержат
только статическую фразу; отмена также не переносит сообщение callback с PII.
Нельзя логировать .values, asdict(row), SQL parameters, locals или repr исходного
raw словаря. Тесты только синтетические, с доменом example.invalid.

## Открытые технические вопросы перед 01B

- Зафиксировать стабильный export-only доступ к RFM observations и дисковым
  общим порогам с parity, не менять текущие определения/private API скрытно.
- Настроить бюджет/путь защищённого job-state, стратегию длительного snapshot
  и отмены, восстановление после аварийного закрытия процесса.
- Подтвердить транспортное экранирование в downstream-импортере; схема уже
  определена, ID нельзя автоматически приводить к int.
- При необходимости historical contacts/consent нужен другой документированный
  источник: текущие Customers хранят latest raw, согласие сейчас UNKNOWN.
- Выбрать для reader строгий отказ на некорректном профиле (текущий контракт)
  либо отдельное согласованное переиспользование quality-обработки birthDate
  из profile service. Нельзя молча чинить дату/терять остальные поля.
- Уточнить поведение на конфликтующих snapshots, если заказчик захочет режим
  first-wins вместо принятого строгого отказа. Такой режим требует явной
  спецификации, а не автоматического ослабления текущего контракта.

## Полный словарь колонок FORMAT_VERSION=1

Следующие таблицы сгенерированы из определённых типов/порядка колонок, без
чтения источников. Тип Identifier — реальный str; Money — Decimal. NULL для
неподготовленных metrics/RFM не заменяется нулём. Поля RFM после готовой
финализации могут сохранять NULL согласно UNSCORABLE/OUTSIDE_BASE.

### Clients — 177 колонок

| Колонка | Тип и NULL | Источник / готовность |
|---|---|---|
| `customer_id` | Identifier(str) | CustomerRecord/выбор профиля; as_of из metadata |
| `source_customer_id` | Identifier(str) / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `profile_present` | bool / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `profile_status` | ProfileStatus | CustomerRecord/выбор профиля; as_of из metadata |
| `profile_updated_utc` | datetime / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `profile_after_analysis_date` | bool / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `last_name` | str / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `first_name` | str / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `middle_name` | str / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `mobile_phone` | str / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `email` | str / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `sex` | str / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `birth_date` | date / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `age` | int / NULL | calendar_age/качество birth_date на as_of |
| `birth_date_status` | str | calendar_age/качество birth_date на as_of |
| `has_email` | bool | CustomerRecord/выбор профиля; as_of из metadata |
| `has_phone` | bool | CustomerRecord/выбор профиля; as_of из metadata |
| `subscriptions_json` | str / NULL | CustomerRecord/выбор профиля; as_of из metadata |
| `email_permission` | Permission | UNKNOWN; документированного mapping нет |
| `sms_permission` | Permission | UNKNOWN; документированного mapping нет |
| `as_of_date` | date | CustomerRecord/выбор профиля; as_of из metadata |
| `strict_metrics_status` | str | OBSERVED/NOT_COMPUTED |
| `strict_orders` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_accepted_lines` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_units` | Decimal / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_first_purchase_utc` | datetime / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_last_purchase_utc` | datetime / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_repeat_orders` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_mean_interval_days` | Decimal / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_orders_12m` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_orders_24m` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_orders_48m` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_has_second_purchase` | bool / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_days_to_second_purchase` | int / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_first_observed_cohort` | str / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_known_amount_rub` | Money(Decimal) / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_known_amount_kzt` | Money(Decimal) / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_valued_orders` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_unvalued_orders` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_observed_amounts_complete` | bool / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_financial_complete` | bool / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_complete_amount_rub` | Money(Decimal) / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_complete_amount_kzt` | Money(Decimal) / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_coverage_status` | CoverageStatus / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_orders_outside_coverage` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_metrics_status` | str | OBSERVED/NOT_COMPUTED |
| `extended_orders` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_accepted_lines` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_units` | Decimal / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_first_purchase_utc` | datetime / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_last_purchase_utc` | datetime / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_repeat_orders` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_mean_interval_days` | Decimal / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_orders_12m` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_orders_24m` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_orders_48m` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_has_second_purchase` | bool / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_days_to_second_purchase` | int / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_first_observed_cohort` | str / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_known_amount_rub` | Money(Decimal) / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_known_amount_kzt` | Money(Decimal) / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_valued_orders` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_unvalued_orders` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_observed_amounts_complete` | bool / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_financial_complete` | bool / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_complete_amount_rub` | Money(Decimal) / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_complete_amount_kzt` | Money(Decimal) / NULL / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_coverage_status` | CoverageStatus / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `extended_orders_outside_coverage` | int / NULL при NOT_COMPUTED | buyer_metrics по полной истории своей политики |
| `strict_rfm_12m_rub_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_rub_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_rub_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_rub_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_rub_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_rub_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_rub_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_rub_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_rub_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_12m_kzt_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_rub_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_24m_kzt_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_rub_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `strict_rfm_48m_kzt_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_rub_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_12m_kzt_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_rub_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_24m_kzt_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_rub_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_status` | RfmAvailability | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_r_days` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_f_orders` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_monetary` | Money(Decimal) / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_r_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_f_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_m_score` | int / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_segment` | str / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |
| `extended_rfm_48m_kzt_recency_reliable` | bool / NULL | 01B: общая RFM-финализация; сейчас NOT_COMPUTED/NULL |

### Orders — 23 колонки

| Колонка | Тип и NULL | Источник / готовность |
|---|---|---|
| `order_id` | Identifier(str) | OrderHeader / полный набор строк и групп статусов |
| `retail_order_id` | Identifier(str) / NULL | OrderHeader / полный набор строк и групп статусов |
| `customer_id` | Identifier(str) | OrderHeader / полный набор строк и групп статусов |
| `source_customer_id` | Identifier(str) | OrderHeader / полный набор строк и групп статусов |
| `order_datetime_utc` | datetime | OrderHeader / полный набор строк и групп статусов |
| `channel_external_id` | str | OrderHeader / полный набор строк и групп статусов |
| `channel_name` | str | OrderHeader / полный набор строк и групп статусов |
| `line_count` | int | OrderHeader / полный набор строк и групп статусов |
| `return_lines` | int | OrderHeader / полный набор строк и групп статусов |
| `cancelled_lines` | int | OrderHeader / полный набор строк и групп статусов |
| `other_status_lines` | int | OrderHeader / полный набор строк и групп статусов |
| `strict_accepted` | bool | Готовый PurchaseOrderFact соответствующей политики |
| `strict_accepted_lines` | int | Готовый PurchaseOrderFact соответствующей политики |
| `strict_units` | Decimal | Готовый PurchaseOrderFact соответствующей политики |
| `strict_amount` | Money(Decimal) / NULL | Готовый PurchaseOrderFact соответствующей политики |
| `strict_currency` | str / NULL | Готовый PurchaseOrderFact соответствующей политики |
| `strict_monetary_issues` | JSON(str) | Готовый PurchaseOrderFact соответствующей политики |
| `extended_accepted` | bool | Готовый PurchaseOrderFact соответствующей политики |
| `extended_accepted_lines` | int | Готовый PurchaseOrderFact соответствующей политики |
| `extended_units` | Decimal | Готовый PurchaseOrderFact соответствующей политики |
| `extended_amount` | Money(Decimal) / NULL | Готовый PurchaseOrderFact соответствующей политики |
| `extended_currency` | str / NULL | Готовый PurchaseOrderFact соответствующей политики |
| `extended_monetary_issues` | JSON(str) | Готовый PurchaseOrderFact соответствующей политики |

### OrderLines — 18 колонок

| Колонка | Тип и NULL | Источник / готовность |
|---|---|---|
| `order_id` | Identifier(str) | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `customer_id` | Identifier(str) | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `source_customer_id` | Identifier(str) | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `line_ordinal` | int | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `line_id` | Identifier(str) | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `line_number` | int | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `product_id` | Identifier(str) | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `product_namespace` | str | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `product_name` | str / NULL | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `quantity` | Decimal | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `base_price_per_item` | Money(Decimal) / NULL | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `price_of_line` | Money(Decimal) | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `line_status` | str | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `status_group` | str | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `accepted_strict` | bool | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `accepted_extended` | bool | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `currency` | str / NULL | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |
| `price_status` | str | OrderLineRecord; status/acceptance из существующей политики; ordinal из порядка |


## Метаданные и внутренние структуры

DetailPeriod: start_date/end_date (date либо оба NULL). DetailMetadata:
as_of_date, event_period, identity_space, orders_coverage, order_mode,
profile_snapshot_at; вычисляемые cutoff_exclusive/effective_coverage/coverage_status.
Это metadata будущего manifest, не дополнительные таблицы с контактами.

ProfileSelection: status, record (nullable CustomerRecord), candidates.
BuyerPolicyMetrics привязан к policy/customer_id/identity_space/as_of_date;
служебные привязки не размножаются в колонках, значение клиента задаётся PK.
PersonalRfm дополнительно содержит policy/window/currency, customer_id,
identity_space/as_of_date, thresholds_token и девять полей RFM, перечисленных
в каждой из 12 размерностей. Бизнес-данные из группового DTO не восстанавливаются.

OrderBundle содержит один OrderRow и кортеж всех OrderLineRow. DetailCells
содержит упорядоченные typed/encoded values и безопасный repr; сам вызов
row_cells/csv_cells не создаёт никакой CSV-файл или ZIP.


## Проверка 01A

79 новых синтетических сценариев входят в успешно выполненный расширенный
релевантный набор: 542 passed. Проверены профили/контакты/конфликты, отсутствие
профиля, aliases и одинаковый email разных покупателей, политики и валюты,
целые/дробные/невалидные числа, исходные product ID, связи таблиц, порядок
колонок, длинные ID, обратимое CSV-экранирование, точность Decimal, периоды,
покрытие, отмена, безопасные repr/исключения и отсутствие I/O/повторной identity.
Индивидуальные counts/суммы/окна/когорта сопоставлены с существующими ядрами
на одинаковых синтетических фактах. Fresh-process проверка подтверждает
отсутствие чтения реальных данных, записи, DB/network операций и Qt.

Первое целевое падение было в новом тесте смешанной валюты: фикстура передавала
адаптеру только один разрешённый namespace. Список namespaces исправлен;
production-адаптер и определения валют/политик не менялись.

18 существующих модулей источников/статистики/GUI/экспортов сверены по SHA-256
с началом задачи: изменений нет. Добавлены только два detail-модуля, два
синтетических тестовых файла и этот документ. Реальные данные не открывались,
CSV/XLSX/ZIP и временный индекс не создавались. Git-команды, кроме явно
предписанного read-only git diff --check, не выполнялись. Ruff новых Python
файлов и проверки whitespace выполнены; untracked-файлы дополнительно
проверены средствами Python, без дополнительных Git-вызовов.
