# STAT-INDEX-ACL-SETUP-01B — решение по хранилищу и пакет административного согласования

Дата проверки: 31.10.2026. Основной ресурсный снимок 09:58 UTC / 12:58 Europe/Moscow.
Проект D:\RecommendationSystem; ветка feature/hybrid-ranking; HEAD e4b603d1924fe247172bb181a201ccc440c04075.

**Основной статус: ADMIN_DECISION_REQUIRED.**
**Административное согласование: NOT APPROVED.**

## 1. Конкретная рекомендация

Для имеющейся машины выбрать **B как практический вариант для согласования**:

- INDEXED: D:\KANZLER-Statistics-Private\statistics_index;
- новый private parent: D:\KANZLER-Statistics-Private;
- небольшой private control/backup namespace вне изменяемого D: — кандидат C:\KANZLER-Statistics-Control;
- единственное предполагаемое изменение существующего DACL — explicit effective Allow конкретного одобренного OTHER_PRINCIPAL на D:\: 0x001301BF → 0x001200AF;
- DACL проекта, filtered_data и canonical **не менять**; отдельный InheritOnly-шаблон D:\ оставить прежним;
- для D:\ использовать только отдельно проверенный, ограниченный данным объектом native setter без propagation, описанный ниже;
- путь B согласовать между builder/status/query/GUI через один источник конфигурации, без скрытого поиска старого индекса.

Это рекомендация проекта решения, **не разрешение на применение**. Допустимость снятия прав у конкретного principal, backup/restore и инструмент root-only записи ещё не подтверждены. Выполнять setup сейчас нельзя.

A отклонён как первый выбор из-за работы с inherited ACL существующего проекта. C лучше изолирует storage, но дополнительного готового тома/ёмкости нет; он не является сегодняшним практически доступным решением. Если администратор не подтвердит B, потребуется отдельно предоставленное контролируемое хранилище C. VHDX на прежнем небезопасном D: не считается автоматической заменой такого решения.

## 2. Что уже завершено и что проверено сейчас

Git Freeze e4b603d содержит исправленный child PSModulePath, прямой вызов icacls с actual ACL attestation и identity guard pilot cleanup. Предыдущая финальная синтетическая регрессия — 197 passed, 1 deselected. Ошибки программных helpers не остаются основанием блокировки; реальное назначение прав ими ещё не испытывалось.

Исходные отчёты прочитаны как техническая база: statistics_index_preflight.md, statistics_index_storage_plan.md, statistics_index_acl_helper.md, statistics_index_acl_setup_plan.md, statistics_index_protect_directory.md. Исторический вывод preflight о целиком загружаемой raw JSON-части уже исправлен последующими отчётами: reader потоковый.

Сведения об истории 01.01.2022–31.08.2026 и кандидате пилота 31.08.2026 взяты из прежних metadata-проверок, повторно здесь не аттестовались. expected_gib полного индекса всё ещё неизвестен. Перед будущим пилотом источники/план нужно проверить заново.

Сейчас проверены только Git, изменчивые ресурсы, critical paths, доступные disk/partition/volume metadata, backup namespace candidates, актуальное применение ACL-политики и in-memory delta. Orders/Actions/Customers не читались; пилот и статистика не запускались.

Tracked working tree и staging исходно чисты. Все 12 прежних unrelated untracked-путей (catalog snapshots и diagnostics) сохранены; их содержимое не исследовалось. В текущей задаче создаётся только данный документ.

## 3. Фактическая инфраструктура

| Объект | Факты | Вывод |
| --- | --- | --- |
| Disk 0, D: | SATA, GPT, online/writable, размер диска 512 110 190 592 byte; NTFS D: 512 092 008 448 byte; свободно 439 628 652 544 byte (~409,44 GiB) | Основной доступный ресурс для B |
| Disk 1, C: | NVMe, GPT, boot/system, online/writable; NTFS C: 511 037 140 992 byte; свободно 52 896 976 896 byte (~49,26 GiB) | Отдельный физический диск для небольшого control/backup; полный INDEXED сюда не предлагается |
| LargestFreeExtent | По 1 048 576 byte на каждом физическом диске | Значимой свободной неразмеченной области для нового большого тома нет |
| Остальные partitions | Reserved, EFI/System и скрытый Recovery; Recovery NTFS ~0,884 GiB | Системные служебные разделы не являются свободным storage |
| S: | По ранее установленному типу network; содержимое не исследовалось | Не использовать для INDEXED |
| Additional local volume | Get-Disk/Get-Partition/Get-Volume не обнаружили другого готового подходящего хранилища | C требует отдельного предоставления ресурса |
| Стандартный root, private parent B и root B | Отсутствуют, reparse в проверенной цепочке не обнаружены | Не создавать в этой задаче |
| C:\KANZLER-Statistics-Control | Отсутствует; existing C:\ проходит parent-policy, NTFS/local | Возможен новый private backup/control кандидат, но ещё не готов |

Числа свободного места — моментальный снимок. Сокращение C:/D:, создание разделов, томов, VHDX, подключение образов и изменение BitLocker не выполнялись и этим пакетом не разрешаются.

Свободных ~409,44 GiB нельзя объявлять гарантией полного build: reserve>=20 GiB и начальная проверка free>=reserve+2*expected_bytes сохраняются; E пока не измерен. Пилот, итоговые БД, старые поколения и journal/temp headroom должны бюджетироваться совместно. Для динамического VHDX отдельно нужен host/guest бюджет; логический размер гостя не добавляет физических byte.

## 4. ACL-блокер и влияние на пространство имён

Повторный read-only вызов действующих inspect_acls/check_acl_facts:

| Existing parent | Результат |
| --- | --- |
| D:\ | UNSAFE_STORAGE_ACL |
| D:\RecommendationSystem | UNSAFE_STORAGE_ACL |
| D:\RecommendationSystem\filtered_data | UNSAFE_STORAGE_ACL |
| C:\ — parent предлагаемого control root | PARENT PASS |

По подтверждённой предыдущей диагностике на D:\ отдельны effective explicit Modify и inheritable template того же principal; у проекта/filtered_data — inherited правила, без своих explicit ACE. Текущая ошибка инспекции не связана с загрузкой Get-Acl. Owner допустим, причина — реальные foreign Allow, а не helper failure.

Разница 0x001301BF → 0x001200AF снимает **0x00010110**: DELETE самого объекта, WRITE_ATTRIBUTES, WRITE_EA. Добавленных битов нет. Сохраняются list/read, read attributes/EA/control, traverse/execute, SYNCHRONIZE и создание дочерних файлов/каталогов. FILE_DELETE_CHILD, WRITE_DAC, WRITE_OWNER в исходном выбранном Modify не было.

Права создания внутри D:\ не равны праву удалить или изменить уже существующий защищённый дочерний объект. Однако изменяемые ancestors и inherited Modify проекта позволяют namespace-операции с этими каталогами с учётом Windows access checks. Ни приватный leaf под ними, ни один /inheritance:r на leaf не удовлетворяют validator всех родителей.

Семантика прав: [Microsoft File Access Rights Constants](https://learn.microsoft.com/en-us/windows/win32/fileio/file-access-rights-constants). Политика приложения сохраняется без исключений для диска/папки.

### Уточнённая модель B

В памяти построена копия DACL D:\ с заменой одной explicit non-inheritable effective ACE:

- check_acl_facts(private=False) для модели — PASS;
- owner, group, protection state — прежние;
- **все inheritable правила идентичны** до/после модели;
- физический descriptor D:\ не изменился.

Это уточняет пригодность delta, но не устанавливает роль principal или зависимости приложений. Полные SID/имена/DACL не публиковались и не сохранялись как backup.

После такого delta existing project/filtered_data сохраняют inherited Modify и продолжат FAIL как storage ancestors. Для B это допустимо: они больше не находятся в цепочке INDEXED. Их обычные working-file permissions предполагается оставить прежними, а не «исправлять заодно».

### Как ограничить запись одним existing object

Обычный Set-Acl/SetNamedSecurityInfo на D:\ не выбран: отсутствие /T не гарантирует отсутствия propagation.

Документирован вариант native **SetSecurityInfo по handle, открытому с MAXIMUM_ALLOWED**: Microsoft указывает, что при этом ACE не распространяются на детей. См. [SetSecurityInfo, Remarks](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-setsecurityinfo). Это не ослабление admission policy, а ограничение области административной записи.

Требования к будущему оператору:

1. Открыть именно существующий D:\, MAXIMUM_ALLOWED (0x02000000), OPEN_EXISTING, directory/reparse-safe flags; держать handle до verification.
2. Сверить volume/file identity и live descriptor с одобренным backup. Убедиться, что реально предоставлен WRITE_DAC; MAXIMUM_ALLOWED сам не повышает права.
3. Передать только DACL_SECURITY_INFORMATION (4), SE_FILE_OBJECT (1), валидный ненулевой proposed DACL; owner/group/SACL не устанавливать. Не менять protected/inheritance flags отдельными обходными командами.
4. Проверить native return и reread exact delta по тому же handle и пути; проверить control points проекта.
5. Закрыть handle в finally. При ошибке не переходить на Set-Acl/SetNamedSecurityInfo или рекурсивный способ.

Этот механизм **не выполнялся и не испытан на реальном изменении ACL**. Перед D:\ обязательно отдельное явно разрешённое испытание оператора на изолированном Windows test tree: подтвердить actual root-only изменение и unchanged children. Если контракт/реализацию нельзя подтвердить — NO-GO, пересогласование C.

При таком подтверждённом операторе intended existing write-set B — **один объект D:\**, существующие project/filtered_data/файлы не записываются. Для произвольного setter число potentially affected descendants неизвестно; они не подсчитывались и массового обхода D: не было. Не выдавать такой неизвестный scope за «0 объектов».

## 5. A/B/C: итоговое сравнение

| Критерий | A: штатный | B: отдельный D: | C: отдельный том |
| --- | --- | --- | --- |
| Реально доступен | Parent dirs есть, root нет | D: есть; два новых private dirs отсутствуют | Готового подходящего тома нет |
| Проходит ACL сейчас | Нет | Нет, D:\ FAIL | Не подтверждено |
| Изменение D:\ | Да | Да, одна согласованная effective ACE | Не требуется только при независимой безопасной всей цепочке |
| Влияние на проект | Три existing parent, inherited/protected semantics; descendants scope требует анализа | DACL проекта/filtered_data сохраняются при проверенном root-only setter | Можно полностью избежать ACL рабочего D: |
| Другие приложения | Риск изменения ancestor и наследования | Снимаются root D: DELETE/EA/attributes у конкретного principal; обязанности программ ещё неизвестны | Изоляция лучше, но инфраструктурные изменения отдельны |
| Путь INDEXED | Штатный | Нужен общий configured index_dir | Также нужен общий configured index_dir |
| Администратор | Нужен для D:\ / rollback | Нужен для D:\, backup/control и утверждения runtime identity | Нужен для предоставления storage |
| ACL backup | Да, включая реально затрагиваемый scope | Root D:\ + control baselines; только при доказанном no-propagation | Controlled namespace и отдельно его backing/storage chain |
| Сложность | Выше риск для существующего дерева | Умеренная: ограниченный admin оператор и один путь в приложении | Сейчас выше: нет готовой ёмкости |
| Риск потери доступа | Existing project и future inheritance | Root metadata операции иных программ; изменение их обязанностей надо согласовать | Зависит от предоставленного тома; mounted/backing failures |
| Восстановление | Частичное, с учётом descendant changes | DACL-only root rollback; новые private dirs не удалять автоматически | Отдельное; не означает удаление/форматирование тома |
| Рекомендуется | Не первый выбор | **Практический кандидат для согласования** | Если B не одобрен либо администратор предоставит независимый ресурс |

Для A минимально надо подготовить D:\, project и filtered_data, а затем leaf. Локальное ограничение inherited ACE может потребовать защиты двух DACL/конверсии правил и изменить будущие обновления наследования. Изменение root inheritable template затрагивает ещё более широкий existing/future scope. Это может влиять на PyCharm/Git/.venv/Python, canonical directories и другие working files; утверждать их доступность без анализа нельзя. Точное число descendants не измерялось. B избегает этой работы с working-tree DACL.

Для B обычные операции с уже существующим проектом, созданиями/изменениями его файлов должны опираться на прежние дочерние ACL. Но не доказано, что какое-либо приложение не использует запись EA/attributes самого корня D:\. Инвентаризация такой зависимости и фактическая роль OTHER_PRINCIPAL остаются задачей администратора. Без согласования применять даже root-only delta нельзя.

## 6. Реальная осуществимость C и риск VHDX

Два имеющихся физических диска уже разделены; LargestFreeExtent ~1 MiB не является ресурсом для статистического тома. Recovery/EFI/Reserved не перепрофилировать. Новый physical/admin-managed volume требует предоставления ёмкости; shrink/format существующих дисков не выбран как простая операция. Сетевой S: и перенос INDEXED на системный C: не предлагаются.

VHDX с отдельной буквой технически является отдельным guest namespace, но имеет host image file. [Microsoft CreateVirtualDisk](https://learn.microsoft.com/en-us/windows/win32/api/virtdisk/nf-virtdisk-createvirtualdisk) отдельно описывает path/security descriptor образа. Это не исчезающий слой защиты.

Для VHDX нужны одновременно:

- безопасные owner/DACL backing-файла и всей его host цепочки;
- отсутствие foreign write/delete/rename/подмены backing path, включая offline/unmount/reboot периоды;
- доверенное управление mount, правильная private DACL гостя и проверка его root;
- host/guest reserve, размер/growth образа, recovery после detach/host-full и безопасный backup.

Возможный file lock подключённого VHDX не является постоянной защитой offline файла. Администратор/сырой доступ и доверенная ОС остаются отдельной частью модели. Гостевой validate_storage_root не проверяет host backing path автоматически.

Образ в обычном inherited каталоге D: получает прежнюю проблему; даже новый protected host folder не позволяет автоматически объявить всю D:\ chain безопасной по текущим требованиям. Образ на D: не добавляет capacity; dynamic guest free может скрывать нехватку host space. Образ на C: ограничен системным ресурсом и конкурирует с TEMP/pagefile.

Поэтому VHDX сейчас **не рекомендуется как доказанная изоляция**, не создавался/не подключался. Для C необходимо отдельно предоставить и подтвердить и storage, и backing chain, если виртуализация используется.

## 7. Backup вне изменяемого namespace

Проверка улучшена относительно 01A:

- На C: есть безопасный parent C:\: local fixed NTFS, trusted owner, parent policy PASS. Это другой физический диск, не изменяемый D: namespace.
- Новый C:\KANZLER-Statistics-Control пока отсутствует. Он выбран как **кандидат** для маленьких ACL packages/config, не INDEXED и не резервного проекта C:.
- Альтернатива под текущим %LOCALAPPDATA% проверена обезличенно: два ближайших ancestors дают UNSAFE_STORAGE_ACL, хотя CurrentUser имеет FullControl. Она не объявляется безопасной только по доступу владельца; её права не менялись.
- Готового private backup leaf, пакета, проверенных ключей/доступа к восстановлению **нет**. Backup не создан. Одного safe parent недостаточно для GO.

Предлагаемая структура после отдельного одобрения:

~~~text
C:\KANZLER-Statistics-Control\                   protected private control directory
    index-storage.json                         только schema/index_dir, без SID/ACL
    acl-backup\                                 protected packages/manifest
D:\KANZLER-Statistics-Private\                  новый private namespace
    statistics_index\                          INDEXED root
~~~

Control/private grants должны соответствовать фактическому runtime пользователя приложения: CURRENT_USER FullControl, SYSTEM и ADMINISTRATORS. Не добавлять explicit grant для другого индивидуального оператора, который для runtime helper является OTHER_PRINCIPAL. Owner — runtime user либо допустимый системный principal. Создание под другим admin account и DPAPI user context требуют отдельного планирования, а не подстановки «текущего администратора» вместо app user.

До D:\ apply:

1. Подготовить новый control root атомарно с explicit protected DACL под заранее одобренными identity. Существующий одноимённый каталог не ремонтировать/не захватывать.
2. Проверить C:\ chain, leaf/files ACL, отсутствие reparse, exact IDs и доступ runtime/rollback оператора. Записей в source tree, diagnostics и C: backup project не делать.
3. Сохранить owner, group, DACL, per-ACE inheritance/propagation и control flags, volume/file identity, hashes, tool/OS version. Полные identifiers неизбежны внутри backup, поэтому пакет шифруется и не логируется.
4. Для выбранного root-only DACL delta owner/group/SACL не менять. SACL/audit/mandatory-label capture, если недоступен, явно отметить NOT_CAPTURED и не называть пакет полным backup всех секций. При необходимости такого восстановления нужны отдельные read/restore privileges и capture.
5. Пакет создать через CreateNew, flush/close/read-back, проверить количество records, hashes, расшифровку и dry-run restore. Подтвердить независимый доступ rollback-оператора **до** изменения прав.
6. DPAPI CurrentUser — возможный способ шифрования, не готовое подтверждение recovery. Требуются подходящий user context/ключи, проверенное восстановление и согласованное резервирование ключей/оператора. Копия ciphertext без доступных ключей бесполезна. Изменяемый пользовательский профиль и потеря его ключей остаются риском доступности.
7. Защищённый manifest должен фиксировать exact before/post IDs и hashes. Наличие пакета или выставленный произвольный flag BackupVerified не заменяет проверки.

При доказанном root-only setter обязательный before snapshot — D:\; project/filtered_data и agreed canonical metadata paths — контрольные identity/ACL baselines. При использовании другого механизма scope может расшириться до descendants: такой вариант этим пакетом не разрешён. Предварительно массовый обход/backup реальных sources здесь не выполнялся.

## 8. index_dir: точная схема следующей небольшой кодовой задачи

### Как сейчас

| Точка | Текущий путь |
| --- | --- |
| statistics_index.DEFAULT_INDEX_DIR | PROJECT_ROOT/filtered_data/statistics_index |
| build_statistics_index / index_state | Default параметра связан со старой константой |
| Builder/status CLI | --index-dir, default DEFAULT_INDEX_DIR |
| statistics_query | index_dir or DEFAULT_INDEX_DIR |
| dataset_statistics | INDEXED передаёт index_dir в query; REFERENCE не использует индекс |
| statistics_slices | Передаёт index_dir при INDEXED/AUTO, но сначала может вернуть valid slice cache |
| Statistics CLI | --index-dir default=None, --backend default=AUTO |
| GUI QProcess | Запускает Statistics CLI без --index-dir |
| Pilot protection boundaries | _forbidden/pilot_paths также используют DEFAULT_INDEX_DIR |

Следовательно, одно применение builder --index-dir не исправляет GUI и protection boundaries.

### Предлагаемая схема B

Один общий lightweight resolver в существующем Application.paths, без Qt и самостоятельного ACL manager:

~~~text
approved control file (C:\KANZLER-Statistics-Control\index-storage.json)
       -> get_statistics_index_dir(explicit=None)
       -> R, один абсолютный путь, snapshot на начало операции

builder CLI main -> R -> build_statistics_index(index_dir=R)
status CLI main  -> R -> index_state(index_dir=R)
GUI QProcess -> scripts/dataset_statistics.py main -> R
             -> calculate_cached_statistics(index_dir=R)
             -> calculate_statistics(INDEXED,index_dir=R)
             -> calculate_indexed_statistics(index_dir=R)
             -> open_generation(R)
~~~

Пример **будущей** конфигурации, файл не создан:

~~~json
{"schema_version":1,"index_dir":"D:\\KANZLER-Statistics-Private\\statistics_index"}
~~~

Правила следующего implementation stage:

- CLI explicit --index-dir — документированный override и валидируется; иначе единственный утверждённый control file. R определяется один раз, дальше передаётся явно. Никакого поиска альтернативных roots.
- Для настроенной B/C конфигурации отсутствие/повреждение control file — явная configuration error, не тихий возврат к PROJECT_ROOT. Не использовать auto-fallback на другой directory.
- DEFAULT_INDEX_DIR может остаться legacy compatibility constant для импортов/явных тестов; операционные defaults builder/status переводятся на None+resolver, query перестаёт подставлять старую константу. Legacy path не используется как скрытый operational fallback после выбранного B rollout.
- REFERENCE-only расчёту не требуется index storage. AUTO сохраняет существующее явно сообщаемое замедление в REFERENCE при runtime index error после корректной конфигурации; configuration errors разрешать до cache/fallback block и не маскировать под этот случай.
- GUI не должен выполнять Get-Acl/storage I/O в основном Qt-потоке. Минимальный вариант — сохранить QProcess, а тот же CLI-resolver выполняется в дочернем процессе. GUI не нужен отдельный settings UI или собственная копия пути.
- Control path определяется из проверенного системного расположения Windows; показанный C: — результат этой машины, не новый hardcode пользовательского профиля в production.

Минимальные точки будущего изменения: Application.paths (resolver/control location); statistics_index.build_statistics_index/index_state (resolve default); statistics_query.calculate_indexed_statistics; statistics_slices.calculate_cached_statistics; scripts/build_statistics_index.py::main; scripts/dataset_statistics.py::main. dataset_statistics.calculate_statistics уже передаёт аргумент, его контракт сохранить. GUI dataset_statistics_tab — проверка QProcess integration, обязательная смена UI не нужна.

Дополнительно statistics_pilot._forbidden и pilot_paths должны запрещать configured production root **и прежний reserved index path**, а не продолжать защищать только старый DEFAULT_INDEX_DIR. Это расширение запретной области для безопасности, не fallback поиска индекса. PilotLimits и бизнес-алгоритмы не менять.

### Кеш и provenance

В актуальном calculate_cached_statistics cache lookup выполняется **до** проверки backend/index; capture_sources не включает index_dir/backend в provenance. Поэтому корректный cached aggregate может быть выдан без открытия физического индекса. Cache hit не доказывает использование нового R. Это установленное поведение, не новая реализация или доказательство дефекта расчётов.

В config stage resolver должен применяться до cache lookup для INDEXED/AUTO, чтобы не скрыть configuration error. Приёмка physical location выполняется отдельными validate_storage_root(R) и index_state(index_dir=R), а не GUI-числами или наличием cached slice.

Storage path не является новым набором событий. Не добавлять его в source provenance ради искусственного пересчёта; actual generation/manifest/source validation сохраняется. Кодовые algorithm fingerprints могут измениться штатно при реализации resolver. Кеш, даты/фильтры и результаты Buyer/RFM/Cohorts не переписывать.

Нужны synthetic tests единого R во всех входных точках, missing/bad config без directory fallback, cache-hit configuration behavior, pilot forbidden configured root и GUI child arguments. Эта задача код не реализует.

## 9. Административный runbook B — NOT EXECUTED

| Шаг | Исполнитель / действие | Контроль допуска |
| --- | --- | --- |
| 1 | Администратор подтверждает B, exact paths, конкретного OTHER principal, delta и runtime/rollback accounts | Отдельное реальное одобрение; кодовые разрешения прошлых задач его не заменяют |
| 2 | Проверить elevated token/WRITE_DAC, recovery context | Не повышать права автоматически; без прав apply/restore — stop |
| 3 | Зафиксировать D:\ и C:\ targets, device/file IDs, отсутствие существующих private dirs/reparse | Любая ambiguity/существующий чужой target — stop |
| 4 | Утвердить root-only write-set, dependencies программ на D-root DELETE/EA/attributes, storage/key риски | Назначение OTHER и приложение-зависимости не угадывать |
| 5 | В отдельно разрешённом test tree проверить native MAXIMUM_ALLOWED setter и no-propagation | Не испытывать впервые на D:\; никакого fallback к обычному Set-Acl |
| 6 | Подготовить private C: control/backup и before snapshot | Outside D/project/diagnostics, runtime ACL, encrypted packages |
| 7 | Подтвердить decrypt/restore, hashes/IDs/sections и доступ rollback-оператора | До этого D:\ apply запрещён |
| 8 | Согласовать maintenance window и остановить конкурирующие ACL/admin изменения; учесть открытые handles программ | Не завершать процессы автоматически |
| 9 | Повторить before descriptor/identity/inheritables и review approved delta | Drift/unknown ACE/несовпадение scope — stop |
| 10 | Применить ровно одобренную DACL delta по pinned handle без propagation | Проверять native ERROR_SUCCESS, остальные sections/flags не менять |
| 11 | Reread root/namespace, require parent-policy PASS и expected post-hash | Любой неожиданный result — no retry, расследование/controlled rollback |
| 12 | Сверить DACL/identity проекта/filtered_data и agreed canonical controls; проверить обычный доступ PyCharm/Git/Python | Без массового изменения canonical, полного статистического расчёта или удаления данных |
| 13 | Атомарно создать private B parent/root с начальным protected ACL; validate из фактического app user | CurrentUser F, owner trusted, no OTHER/Deny, local NTFS, no reparse |
| 14 | В отдельно одобренном новом test child выполнить ограниченный реальный smoke protect_directory, затем exact ACL/identity check | Это MUTATING smoke, здесь не выполняется; не запускать pytest всего suite без согласия |
| 15 | Принять результат либо выполнить controlled restore; настроить общий R в отдельной кодовой задаче перед pilot/build | Storage admission не означает наличие индекса/обоснованный expected_gib |

Обычный пользователь сейчас может выполнить read-only проверки и подготовить сведения о зависимостях. Текущий unelevated token не является административным одобрением. Создание private control, изменение D:\ и разрешённый mutating smoke требуют отдельного согласования; право создания некоторых папок само по себе не заменяет его.

## 10. Команды

### Read-only verification — можно повторять

Используют локальный helper/Windows API, не выполняют статистику и не создают roots:

~~~powershell
Set-Location -LiteralPath 'D:\RecommendationSystem'
@'
from pathlib import Path
import json, shutil
from Application import statistics_storage_safety as s
for label in ("C:/", "D:/"):
    root = Path(label)
    s.reject_reparse(root)
    s.local_volume(root)
    print(json.dumps({"volume": label, "free_bytes": shutil.disk_usage(root).free}))
for path in map(Path, ("D:/", "D:/RecommendationSystem",
                       "D:/RecommendationSystem/filtered_data", "C:/")):
    fact = s.inspect_acls([path])[0]
    try:
        s.check_acl_facts(fact, private=False)
        state = "PARENT_PASS"
    except s.StatisticsIndexError as error:
        state = str(error)
    print(json.dumps({"path":str(path),"state":state,
                      "owner_trusted":fact["owner_trusted"],
                      "protected":fact["protected"]}))
for label in ("D:/RecommendationSystem/filtered_data/statistics_index",
              "D:/KANZLER-Statistics-Private", "D:/KANZLER-Statistics-Private/statistics_index",
              "C:/KANZLER-Statistics-Control"):
    path=Path(label)
    s.reject_reparse(path)
    print(json.dumps({"path":label,"exists":path.exists()}))
'@ | & 'D:\RecommendationSystem\.venv312\Scripts\python.exe' -B -
if ($LASTEXITCODE -ne 0) { throw 'Read-only check failed. Stop.' }
~~~

При необходимости администратор может повторить Get-Disk/Get-Partition/Get-Volume с полями размера/FS/role без serial/account/volume label. Запускать из отдельного Windows PowerShell с доверенным built-in module environment; не импортировать пользовательские modules ради storage probe.

### Proposed changes — NOT EXECUTED

Параметризованный dry proposal для одной ACE. $ApprovedPlan загружается из отдельного защищённого административного плана; реальных SID в примере нет. Никакой записи DACL в блоке:

~~~powershell
# NOT EXECUTED. In-memory proposal only, no ACL write/backup/root creation.
$ErrorActionPreference = 'Stop'
if ($null -eq $ApprovedPlan) { throw 'A private reviewed administrative plan is required.' }
if ($ApprovedPlan.Target -cne 'D:\') { throw 'Unexpected target.' }
$Principal = $ApprovedPlan.PrincipalObject
$ExpectedBeforeHash = $ApprovedPlan.BeforeDescriptorSha256
$Before = Get-Acl -LiteralPath $ApprovedPlan.Target
$bytes = $Before.GetSecurityDescriptorBinaryForm()
$hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($bytes)).Replace('-','').ToLowerInvariant()
if ($hash -ne $ExpectedBeforeHash) { throw 'Descriptor drift; stop.' }
$selectedRules = @($Before.Access | Where-Object {
 (-not $_.IsInherited) -and $_.AccessControlType -eq 'Allow' -and
 $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -eq $Principal.Value -and
 ([long]$_.FileSystemRights -band 4294967295L) -eq 0x001301BF -and
 $_.InheritanceFlags -eq 'None' -and $_.PropagationFlags -eq 'None'
})
if ($selectedRules.Count -ne 1) { throw 'Rule is missing or ambiguous.' }
$Proposed = [Security.AccessControl.DirectorySecurity]::new()
$Proposed.SetSecurityDescriptorBinaryForm($bytes)
$Proposed.RemoveAccessRuleSpecific($selectedRules[0])
$Proposed.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
 $Principal, [Security.AccessControl.FileSystemRights]0x001200AF,
 [Security.AccessControl.InheritanceFlags]::None,
 [Security.AccessControl.PropagationFlags]::None,
 [Security.AccessControl.AccessControlType]::Allow))
# Do not print/serialize Proposed or bytes into console/repository.
# NOT an apply script. Do not substitute Set-Acl/SetNamedSecurityInfo here.
# A separately tested native operator must:
# 1. open the exact D:\ object by handle with MAXIMUM_ALLOWED (0x02000000),
#    OPEN_EXISTING and directory/reparse-safe flags;
# 2. attest handle file ID, volume ID, live owner/group/control/DACL hashes,
#    required WRITE_DAC, private plan and verified backup/restore;
# 3. use SetSecurityInfo(handle, SE_FILE_OBJECT=1,
#    DACL_SECURITY_INFORMATION=4, NULL, NULL, nonnull_valid_proposed_dacl, NULL);
# 4. check native result=ERROR_SUCCESS, reread same handle/namespace,
#    require the expected delta only, and close handle in finally.
# Any inability to prove the no-propagation contract is NO-GO.
~~~

Будущий backup/setup contract, **не исполняемый apply-script**:

~~~powershell
# NOT EXECUTED. Contract for the future backup/setup operator.
$IndexParent = 'D:\KANZLER-Statistics-Private'
$IndexRoot = 'D:\KANZLER-Statistics-Private\statistics_index'
$ControlRoot = 'C:\KANZLER-Statistics-Control'
$ConfigPath = 'C:\KANZLER-Statistics-Control\index-storage.json'
$BackupDirectory = 'C:\KANZLER-Statistics-Control\acl-backup'
# Before creating anything, require approved runtime account, operator,
# owner/ACL policy, exact absent targets and safe C:\ parent.
# Create NEW directories atomically with an explicit protected DACL.
# Grants: runtime CURRENT_USER FullControl, SYSTEM, ADMINISTRATORS.
# No arbitrary operator SID grant if it is OTHER_PRINCIPAL for runtime user.
# Existing directories are never reset/taken over.
# Snapshot D:\ owner/group/DACL/control flags/file+volume identity.
# Add project/filtered_data and approved canonical metadata paths as read-only
# control hashes/identities; do not read Orders/Actions records.
# Encrypt descriptors into CreateNew backup packages, flush/close/read back,
# verify every record/hash and independent decrypt/restore capability.
# Do not apply D:\ delta until all prerequisites and admin approval hold.
# Never write security descriptors into Git, docs, diagnostics or plaintext logs.
~~~

После отдельно согласованной настройки — read-only приёмка из runtime account:

~~~powershell
# FUTURE READ-ONLY verification, AFTER separately approved setup. NOT RUN HERE.
Set-Location -LiteralPath 'D:\RecommendationSystem'
@'
from pathlib import Path
from Application.statistics_storage_safety import validate_storage_root
for target in (Path("C:/KANZLER-Statistics-Control"),
               Path("D:/KANZLER-Statistics-Private"),
               Path("D:/KANZLER-Statistics-Private/statistics_index")):
    validate_storage_root(target)
print("PRIVATE_STORAGE_ADMISSION_PASS")
'@ | & 'D:\RecommendationSystem\.venv312\Scripts\python.exe' -B -
if ($LASTEXITCODE -ne 0) { throw 'Storage admission failed. No pilot/build.' }
# Run as the actual application user. Status alone does not replace this check.
~~~

Штатный --status с явным R допустим в следующем этапе; INDEX_MISSING у нового пустого root ожидаем и не заменяет preceding private-root admission. Ни --build, ни prepare_pilot/build_pilot команда в текущей задаче не запускалась.

## 11. Восстановление и stop conditions

Восстановление D:\: pinned identity/handle, проверенный encrypted before snapshot, сверка live expected post-hash. Тот же проверенный **непропагирующий DACL-only setter**; не перезаписывать owner/group/SACL, если они не менялись. Не применять старый descriptor к другому object ID или поверх concurrent changes. Проверить исходную parent-policy (она снова может быть FAIL), доступ программ и unchanged project controls.

Новые private dirs, test child и данные не удалять автоматически. Состояние backup/config отдельно фиксируется; rollback root ACL не должен уничтожать единственную recovery copy. Ошибку cleanup не подавлять. Данные, созданные после change, раскрытие информации или выполненные операции не «откатываются» восстановлением ACL.

Немедленный NO-GO:

- NOT APPROVED, неизвестные principal/runtime/rollback identities или приложение-зависимости;
- backup/private leaf/keys/decrypt/restore не подтверждены;
- unknown ACE, before hash/identity drift, новые reparse/junction;
- выбранный setter не доказанно root-only, расширяет scope или требует массового обходного reset;
- unexpected owner/group/control/inheritance delta, failed native result, failed admission;
- потерян доступ проекта, canonical metadata controls, backup или rollback operator;
- небезопасный backing path VHDX, нехватка host/guest capacity, неподходящий том;
- valid cache используется как единственное доказательство actual index location.

Никакого автоматического повторения, force ownership, широких Deny, /reset /T или скрытого directory fallback.

## 12. Критерии Go/No-Go и следующий этап

| Критерий | Сегодня |
| --- | --- |
| Git/исправленные helpers | Подтверждены на e4b603d |
| Практический путь B | Определён как кандидат |
| In-memory D:\ delta / unchanged inheritable rules | PASS |
| Допустимость изменения для OTHER и иных программ | Не подтверждена |
| Native root-only setter real test | Не выполнен |
| Private control/backup leaf и restore | Не готовы |
| Private B dirs / root admission | Отсутствуют |
| Общая конфигурация R | Ещё не реализована |
| Реальный protect smoke | Не выполнен |
| expected_gib полного build | Не обоснован |
| Административное согласование | **NOT APPROVED** |

**Сейчас NO-GO для setup.** Это не выбор небезопасного B ради скорости.

Точное следующее решение администратора: принять/отклонить B и root-only delta, назначить runtime/rollback identities и проверку зависимости root metadata operations, одобрить private control/backup вне D: и отдельное тестирование bounded setter. Если D:\ изменение не допускается — предоставить контролируемое C; текущих готовых ресурсов для него нет.

После такого согласования — отдельный этап проверки административного инструмента/backup/recovery на разрешённом test namespace, затем отдельно gated operation по runbook B с post-validation и ограниченным mutating smoke. Общая конфигурация index_dir — отдельная небольшая разработческая задача; она должна быть завершена до реального pilot/full build.

## 13. Проверки текущей задачи и сохранность

Production-код не изменялся; полный pytest/предыдущие 197 security cases не повторялись. Достаточны целевые read-only probes и in-memory модель. Не запускался test_private_directory_protection_on_new_directory или настоящий icacls/protect_directory mutation.

Проверены metadata дисков/разделов/томов, free space, current paths, helper ACL admission, C: control parent, непригодность выбранного LocalAppData candidate и exact model invariants. DPAPI package/keys не создавались, crypto/restore test не имитируется как выполненный.

Исходные пять документов сохраняются как история. Новый файл только D:\RecommendationSystem\docs\statistics_index_acl_decision.md. Финальная проверка выполнена: git diff --check без ошибок; SHA-256 всех 14 исходных файлов и исторических отчётов совпадают с начальным снимком; HEAD, branch, tracked tree и staging не изменены. Список untracked содержит прежние 12 путей и только один новый документ этой задачи. Все 4 PowerShell-блока разобраны штатным parser без ошибок; 2 встроенных Python-фрагмента прошли ast.parse без исполнения будущих команд. В отчёте не обнаружены private SID или персональные profile paths. Повторный read-only admission: три existing D: parent — UNSAFE_STORAGE_ACL, C:\ — PARENT PASS; все четыре планируемых root/control пути отсутствуют.

Технические результаты: D:\CodexTestTemp\statistics-acl-decision-c5e9dd5434ad4044a7c5a1e163a63bad; только обезличенные facts/metadata, без raw ACL/security backup. Диагностика проекта, catalog snapshots и резервный проект C: не затрагивались. Никаких ACL/томов/VHDX/переносов/канонических данных/production roots/статистики/Git mutations.

**ADMIN_DECISION_REQUIRED. NOT APPROVED. После отчёта остановиться.**
