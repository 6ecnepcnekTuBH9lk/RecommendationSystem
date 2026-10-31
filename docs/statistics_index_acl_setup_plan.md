# STAT-INDEX-ACL-SETUP-01A — план безопасной подготовки ACL INDEXED

Дата: 31.10.2026. Проект D:\RecommendationSystem, ветка feature/hybrid-ranking, HEAD abb9cd8e98bbcd06c9906409e50ed98ea1fd4166.

## Решение

**ADMIN_DECISION_REQUIRED.** Ни один вариант на имеющейся инфраструктуре пока не подтверждён как безопасный для применения. PLAN_READY_FOR_APPROVAL не выдаётся.

**Дополнительно: PROTECT_DIRECTORY_FIX_REQUIRED.** В существующем втором helper воспроизведено ложное сообщение об успехе при неудачном запуске native tool. До setup нужен отдельный программный фикс; код здесь не менялся.

Приоритет архитектуры — отдельное контролируемое локальное NTFS-хранилище (вариант C), если администратор сможет его предоставить. Такого дополнительного локального тома сейчас не обнаружено. До его предоставления это предложение для согласования, не выбранный существующий путь. Варианты A/B требуют решения о правах общего D:\ и влиянии на другие приложения.

В этой задаче создан только данный отчёт. ACL, root, production-код, системные переменные, Registry, тома, canonical и резервный проект на C: не менялись. ACL backup не создавался. Пилот, индексирование, статистика и экспорт не запускались.

## 1. Git и локальная версия

Проект открыт с D:. Текущий helper содержит локальный фикс STAT-INDEX-ACL-HELPER-01; инспекция фактически работает без ручного изменения глобального PSModulePath.

Сохранены существовавшие до задачи изменения:

- modified D:\RecommendationSystem\Application\statistics_storage_safety.py;
- modified D:\RecommendationSystem\tests\test_statistics_storage_safety.py;
- untracked D:\RecommendationSystem\docs\statistics_index_acl_helper.md;
- предыдущие preflight/storage-plan документы, исторические справочники и diagnostics.

Tracked diff уже содержал 45 добавлений/2 удаления production helper и 217 добавлений тестов. Это изменения предыдущей задачи, не текущего анализа. Fetch, staging, commit, push и другие Git mutations не выполнялись. Не исследовалось содержимое diagnostics и исторических справочников. Финальная побайтовая проверка сохранности — в разделе проверок.

## 2. Точная security policy

| Объект | Действующие требования |
| --- | --- |
| Корень тома D:\ и остальные родители | Trusted owner, понятный DACL, без effective foreign Allow с битами 0xD0150; read-only/traverse допускаются; protected DACL и отдельная CurrentUser FullControl ACE для родителей не обязательны |
| Private index root | Protected DACL; CurrentUser Full Control; owner trusted; никаких OTHER_PRINCIPAL Allow, включая read и InheritOnly; никаких Deny |
| UUID generation | Builder создаёт собственный каталог, вызывает protect_directory и повторно проверяет как private child |
| current.json, manifest.json, SQLite | Создаются в private namespace, наследуют приватные права; при проверке перечисленные children проверяются как private, без reparse/чужого тома |
| Namespace / filesystem | Лексическая проверка reparse до resolve, локальный fixed writable NTFS, отсутствие синхронизируемого расположения, root отдельно от canonical |

Trusted principals: CURRENT_USER, SYSTEM, ADMINISTRATORS, TRUSTED_INSTALLER. Членство пользователя в посторонней группе не делает всю такую группу trusted. Неподдерживаемый/null DACL или owner => UNTRUSTED_STORAGE_OWNER_OR_ACL.

В parent-политике запрещённые биты 0xD0150: DELETE, FILE_DELETE_CHILD, WRITE_DAC, WRITE_OWNER, FILE_WRITE_ATTRIBUTES, FILE_WRITE_EA. Foreign InheritOnly не действует на текущий parent и сам по себе его не блокирует; на private root такое разрешение всё равно запрещено. Parent Deny не компенсирует опасную Allow: это строгая allowlist admission, не полная симуляция AccessCheck.

Generic mapping сохранён: All→0x1F01FF, Write→0x120116, Read→0x120089, Execute→0x1200A0. Маски не ослаблялись, исключений для D: или проекта нет.

### Различие прав и риск подмены

| Право | Что разрешает |
| --- | --- |
| READ_DATA/LIST_DIRECTORY, READ_EA/ATTRIBUTES, READ_CONTROL, EXECUTE/TRAVERSE | Чтение/просмотр/проход; READ_CONTROL не даёт изменить DACL |
| WRITE_DATA/ADD_FILE 0x2, APPEND_DATA/ADD_SUBDIRECTORY 0x4 | Для directory — создание дочерних объектов; сами по себе не дают право менять уже существующий защищённый дочерний файл |
| DELETE 0x10000 | Удаление/операции rename для самого объекта с учётом других требований Windows; не тождественно DeleteChild родителя |
| FILE_DELETE_CHILD 0x40 | Удаление дочернего объекта через родителя; нельзя считать приватный DACL ребёнка универсальной защитой от этого parent-права |
| WRITE_DAC 0x40000 / WRITE_OWNER 0x80000 | Изменение DACL / владельца, это отдельные полномочия |
| WRITE_ATTRIBUTES 0x100 / WRITE_EA 0x10 | Изменение metadata объекта; строгая политика запрещает foreign bits и на parent |

Rename/move также требует доступа к destination namespace, проверки sharing/open handles и условий ОС. Для существующих project/filtered_data разрешение DELETE на самом каталоге в сочетании с правами создания в destination создаёт риск перемещения/замены звена пути. Это не утверждение, что один DELETE позволяет удалить любой непустой каталог или корень тома.

У наблюдаемого Modify нет FILE_DELETE_CHILD, WRITE_DAC и WRITE_OWNER. Поэтому нельзя говорить «родитель всегда может переписать ACL/private child». Угроза включает изменяемые ancestors и inherited Modify на существующих каталогах; root тома дополнительно имеет системные ограничения. Изменение только будущего private leaf не удовлетворяет текущему validator.

Семантика прав сверена с [Microsoft File Access Rights Constants](https://learn.microsoft.com/en-us/windows/win32/fileio/file-access-rights-constants) и [File security and access rights](https://learn.microsoft.com/en-us/windows/win32/fileio/file-security-and-access-rights).

## 3. Повторная реальная диагностика

Использованы действующий локальный inspect_acls и check_acl_facts, а также отдельное read-only получение обезличенных inheritance flags. Никакие security descriptors на диск не экспортировались.

| Каталог | Owner category | Protected | Explicit / inherited ACE | Блокирующая Allow | Результат |
| --- | --- | --- | --- | --- | --- |
| D:\ | SYSTEM | Нет | 8 / 0 | OTHER_PRINCIPAL, 0x001301BF, explicit effective | UNSAFE_STORAGE_ACL |
| D:\RecommendationSystem | CURRENT_USER | Нет | 0 / 8 | OTHER_PRINCIPAL, 0x001301BF, inherited | UNSAFE_STORAGE_ACL |
| D:\RecommendationSystem\filtered_data | CURRENT_USER | Нет | 0 / 8 | OTHER_PRINCIPAL, 0x001301BF, inherited | UNSAFE_STORAGE_ACL |

Все три пути существуют, resolve соответствует лексическому пути, reparse/junction не обнаружены. Owner trusted=True, unsupported=False. Посторонние Allow имеются; из четырёх OTHER_PRINCIPAL форм только одна effective Modify ACE блокирует каждый parent. Deny нет. Наличие Get-Acl и успешного JSON не означает безопасный storage.

На D:\ у одного OTHER_PRINCIPAL есть две отдельные explicit формы:

- 0x001301BF, InheritanceFlags=None, PropagationFlags=None — действует на сам D:\.
- 0xE0010000, ContainerInherit|ObjectInherit, InheritOnly — передаёт потомкам effective Modify и сохраняемый шаблон наследования.

У проекта/filtered_data нет своих explicit правил. Их effective Modify сопоставлен с единственным source template на D:\ по реальному principal внутри процесса, Allow/type, нормализованной specific mask и inheritance flags. Полные идентификаторы не выводились. Совпадение одной маски не использовалось как единственное доказательство.

В результате имеется общая конфигурация предка, а не три независимых явных дефекта. Но effective D:\ ACE не наследуется сама: дочерние Modify происходят от отдельной InheritOnly ACE. Исправление только effective D:\ ACE не исправит inherited Modify существующего проекта.

0x001301BF пересекается с parent-blocking mask по **0x00010110**: DELETE + WRITE_ATTRIBUTES + WRITE_EA. Read/Execute 0x001200A9 для parent допустим. GenericRead|Execute InheritOnly 0xA0000000 также не вызывает текущий parent-отказ. Foreign InheritOnly на private root всё равно запрещён.

## 4. Реально доступная инфраструктура

Read-only GetLogicalDrives/GetDriveType/GetVolumeInformation; содержимое дисков не сканировалось, labels не извлекались.

| Расположение | Тип | Filesystem | Свободно при проверке | Пригодность |
| --- | --- | --- | ---: | --- |
| C: | Local fixed | NTFS writable | 52 911 112 192 byte (~49,28 GiB) | Системный том; перенос INDEXED сюда не предлагается |
| D: | Local fixed | NTFS writable | 439 650 652 160 byte (~409,46 GiB) | Текущий источник/проект; ACL namespace не проходят |
| S: | Network | Не исследовалась | Не измерялось | Не подходит для private INDEXED по контракту |

Других доступных logical drives не обнаружено. Это не инвентаризация неразмеченных физических носителей, offline дисков или скрытых partition. Наличие дополнительного готового локального хранилища не предполагается.

Число свободных byte — моментальный снимок. Для любого будущего storage сохраняются reserve>=20 GiB и начальный free>=reserve+2*expected_bytes, все существующие поколения и peak journal/temp должны быть учтены. Обоснованного expected_gib нет; пилот остаётся отдельным этапом.

## 5. Сравнение A/B/C

| Критерий | A: штатный root | B: новый private namespace на D: | C: отдельное контролируемое локальное NTFS-хранилище |
| --- | --- | --- | --- |
| Путь | D:\RecommendationSystem\filtered_data\statistics_index | D:\KANZLER-Statistics-Private\statistics_index | Точный путь ещё не предоставлен |
| Безопасность сейчас | Нет | Нет: общий D:\ FAIL | Не проверена: такого готового хранилища нет |
| Изменение ACL D:\ | Требуется для текущей политики | Также требуется | Не требуется, только если реальная цепочка нового тома не проходит через D:\ |
| Влияние на проект | Родительские ACL двух существующих каталогов, риск наследования | Можно сохранить ACL проекта/filtered_data | Можно сохранить ACL рабочего D: и проекта |
| Администратор | Практически нужен для D:\ и проверенного rollback | Нужен для D:\; создание новых dirs после admission | Нужен для выделения/подготовки инфраструктуры |
| Путь INDEXED в приложении | Штатный default сохраняется | Нужна отдельная согласованная конфигурация всех входных точек | Также нужна общая конфигурация пути |
| Риск для других программ | Наибольший: существующие project subtree и D: | Меньше затрагиваемых project ACL, но D: общий и propagation надо оценить | Изоляция рисков лучше; изменения инфраструктуры требуют отдельного согласования |
| Обратимость | Частичная по snapshots, данные/новые файлы не откатываются автоматически | Аналогично для D:; новые private dirs отдельно осматриваются | Не подразумевает автоматическое удаление/форматирование тома |
| Готовность | ADMIN_DECISION_REQUIRED | ADMIN_DECISION_REQUIRED, не обход D: | Только архитектурный кандидат, нет готового тома |

### Предпочтение

С учётом приоритета сохранения рабочего проекта предпочтительно согласовать **C** — предоставление изолированной local fixed NTFS цепочки, не зависящей от ACL D:\. Никакие VHD/VHDX, тома, разделы, BitLocker или носители этой задачей не менялись. Точка монтирования внутри D:\ при нынешнем validator не является доказанным обходом: reparse/mount-point также проверяются. Нужен отдельный проверяемый локальный путь.

Если отдельное хранилище недоступно, администратор должен решить, допустим ли минимальный root-D delta и вариант B. B рекомендуется рассматривать только **вместе** с подготовкой D:\ и общей конфигурацией GUI/CLI; один новый каталог переносит проблему. A пока не предпочтителен из-за необходимости работы с inherited ACL существующих project/filtered_data.

До решения ни один путь не назначается новым DEFAULT_INDEX_DIR и ни один setup script не объявляется готовым к применению.

### Что понадобится A

Подготовить D:\ и обе существующие parent-директории. Для проекта/filtered_data конфликтующие ACE inherited: их нельзя представить как обычное удаление локальной explicit записи. Возможные механизмы требуют отдельного выбора администратора: локальная защита DACL с точным сохранением нужных унаследованных правил либо изменение source inheritance. Второй вариант затрагивает большой существующий D: namespace и здесь не рекомендуется.

Защита двух отдельных DACL не равна рекурсивному отключению наследования всего проекта, но всё равно меняет получение будущих обновлений прав от предков. Инструмент/API может обрабатывать descendants. Ни /inheritance:r на всём проекте, ни массовый /reset, ни широкие Deny, ни takeown не предлагаются.

### Минимальный кандидат для D:\ в B

Только explicit effective Allow конкретного одобренного OTHER_PRINCIPAL: 0x001301BF → 0x001200AF, тот же principal/type, InheritanceFlags=None/PropagationFlags=None. Убираются DELETE/WRITE_ATTRIBUTES/WRITE_EA для самого D:\; чтение/проход/создание дочерних объектов остаются.

Модель копии DACL построена **только в памяти**, без Set-Acl/записи ACL. Она проходит check_acl_facts(private=False); физический descriptor D:\ до/после совпал. Это доказывает соответствие масок текущей политике, не допустимость изменения для других приложений.

InheritOnly 0xE0010000 при этом не менять: existing project/filtered_data остаются со своим Modify и продолжают FAIL, но в B они не являются ancestors индекса. Новый private parent сразу получает изолированную DACL.

**Без /T не означает нулевого влияния на descendants.** Windows security API может распространять inheritable ACE при записи DACL. Для предлагаемого delta их intended semantics сохраняются, но реальная область обработки должна быть подтверждена отдельно и входить в backup/verification scope. Если scope нельзя ограничить/проверить — операцию не разрешать. См. [Microsoft SetNamedSecurityInfo](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-setnamedsecurityinfow).

## 6. Согласованность GUI/CLI/cache

- DEFAULT_INDEX_DIR — PROJECT_ROOT/filtered_data/statistics_index.
- Builder CLI принимает --index-dir; default штатный.
- statistics_query использует index_dir or DEFAULT_INDEX_DIR.
- statistics_slices передаёт index_dir в INDEXED/AUTO.
- scripts/dataset_statistics.py принимает --index-dir с default=None.
- GUI dataset_statistics_tab запускает этот CLI без --index-dir.

Поэтому B/C требуют отдельной маленькой задачи общей конфигурации/передачи index_dir для builder, query и GUI subprocess. Одного builder flag недостаточно. Кеш агрегатов не обнаруживает расположение индекса сам: он получает путь через общий calculation route; путь кеша и расчётные контракты здесь не менялись. Изменение DEFAULT_INDEX_DIR только в одном месте или перенос кеша без задачи не предлагаются.

## 7. protect_directory(): отдельный подтверждённый дефект

**PROTECT_DIRECTORY_FIX_REQUIRED.**

Текущий код запускает powershell.exe -NoProfile -NonInteractive -EncodedCommand, наследует env, не задаёт timeout. В скрипте .NET WindowsIdentity, native icacls.exe, Out-Null и exit $LASTEXITCODE. Get-Acl/ConvertTo-Json не используются. Read-only примитивы работают с загрязнённым module path; изоляция inspect_acls автоматически на этот subprocess не распространяется.

Обычный nonzero native exit отвергается PRIVATE_INDEX_ACL_FAILED. Ошибка запуска PowerShell поднимается как исключение и прерывает операцию. Но отсутствуют ErrorActionPreference=Stop и надёжное различение неудачного запуска icacls от его native exit.

Безопасное воспроизведение: захвачена реальная command string через mock subprocess, имя icacls заменено на заведомо отсутствующую утилиту. **Настоящий icacls не запускался.** Windows PowerShell 5.1 выдал CommandNotFoundException в захваченный stderr и process exit=0; последняя native-команда не исполнилась, $LASTEXITCODE не получил код отказа. Сам protect_directory проверяет только returncode и принимает такой результат.

На этой машине icacls.exe найден; текущая проблема не означает, что он отсутствует сейчас. Воспроизведён дефект error-path существующего helper, а не реальный отказ назначения прав проекту. Семантика $LASTEXITCODE: [Microsoft automatic variables](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_automatic_variables?view=powershell-5.1).

Builder немедленно вызывает validate_storage_root после protect_directory до создания SQLite, а затем повторно перед publication. Это дополнительная защита: дефект helper не доказывает, что небезопасный generation был бы опубликован. Тем не менее контракт второго helper не fail-closed при показанном сценарии, поэтому setup нужно отложить до исправления.

Приоритет Medium. Файл: D:\RecommendationSystem\Application\statistics_index.py, protect_directory. Риск: ложный успех, ухудшенная диагностика и возможное зависание без timeout. Рекомендация — отдельная узкая задача: Stop/try-catch для ошибок запуска команды, проверенный native executable, явный валидный native exit, ограниченный timeout/безопасные исключения и mock regression. Не считать любое наличие stderr универсальным критерием ошибки и не чинить данный дефект простым его игнорированием/парсингом. При необходимости согласовать изоляцию child env, сохранив текущую ACL policy и post-check.

Expected result: отсутствие ложного успеха, прежние коды безопасного отказа. Риск изменения Low/Medium; успешное штатное поведение можно сохранить, error-path намеренно станет строгим. Сначала нужен regression test — он уже воспроизводится во внешнем временном модуле этой задачи. Production-код не исправлялся.

## 8. Предварительное резервирование — ПЛАН, BACKUP НЕ СОЗДАН

До будущего apply:

1. Зафиксировать точные target paths, volume serial/file IDs, owner/group, DACL, inheritance/protection/control flags и SHA-256 descriptors. Сопоставить runtime-пользователя приложения и административный token без публикации идентификаторов.
2. Scope A: D:\, D:\RecommendationSystem, D:\RecommendationSystem\filtered_data, а также каждый existing object в фактической области возможной propagation. Scope B: D:\ и все existing objects, которые будет обрабатывать выбранный write API; project/filtered_data как контрольные точки. Для C — только предоставленный namespace и его контролируемая область.
3. Если выбранный инструмент может затронуть descendants, snapshots трёх директоров недостаточно для обещания восстановления всего дерева. Администратор должен установить scope в отдельном maintenance window. Если его нельзя надёжно инвентаризировать/верифицировать, не применять изменение на общем D:.
4. Сохранять Owner+Group+DACL и control flags; владельца не менять. SACL/audit/mandatory label не менять. Если требуется полнота этих секций, отдельно получить необходимые read privileges и сохранить их; иначе отметить NOT_CAPTURED и не объявлять snapshot полным security descriptor backup.
5. $ApprovedBackupRoot — заранее существующее проверенное защищённое место **вне репозитория, canonical и резервного проекта C:**. Конкретное место ещё не утверждено. Пока оно/доступ к restore не подтверждены, apply запрещён. Git ignore не заменяет хранение вне проекта.
6. Пакет с фактическими identifiers шифровать, например DPAPI для утверждённого оператора с проверенным восстановлением ключей либо организационным способом. Plaintext SDDL не писать в docs/logs. DPAPI CurrentUser не означает автоматическую расшифровку другим администратором; потеря ключей/профиля — риск восстановления.
7. Создать пакет как новый файл, без перезаписи; закрыть/синхронизировать запись. Проверить обратную расшифровку, hash всех записей, наличие нужных sections, identities и количество объектов; проверить read-back/restore dry-run в памяти. Integrity/checksum хранить в доверенном manifest, а не только рядом с изменяемым файлом.
8. До apply повторить baseline hashes/IDs и подтвердить отдельный доступ rollback-оператора. Подготовить maintenance window, inventory зависимых приложений/служб, наблюдение за доступом и запрет параллельных ACL changes.

Использование encrypted backup не подтверждает безопасность его path namespace или восстановления. Условия хранения/целостности/доступности должны быть проверены отдельно. В этой задаче не создавался ни plaintext, ни encrypted ACL backup; только обезличенные facts и технические результаты.

## 9. Восстановление — последовательность и пределы

Для уже изменённых объектов:

1. Остановить последующие setup/build операции. Установить текущие path→file ID соответствия и сравнить live descriptor с ожидаемым post-change manifest. При подмене объекта, concurrent admin edits или неизвестных новых файлах не выполнять blind restore.
2. Убедиться, что rollback token имеет WRITE_DAC на targets. Для D:\ обычно потребуется elevated administrator; owner/group restore требует дополнительных полномочий и отдельного решения. Принудительное владение не предлагается.
3. При полном согласованном baseline восстановить parent DACL/control flags сверху вниз; затем конкретные changed descendant exceptions из inventory. После каждого этапа учитывать возможную propagation и повторно проверять hashes/flags/доступ. Для A восстановление D:\ не заменяет восстановления protected project/filtered_data.
4. Owner/group не перезаписывать, если они не менялись. Не заменять неизвестную SACL пустой секцией. SACL/mandatory labels восстанавливать только при отдельном сохранённом baseline и полномочиях.
5. Проверить original access зависимых программ, runtime-пользователя и администратора. Проверить существующие и новые объекты отдельно.
6. Новые private каталоги/поколения не удалять автоматически. Их корректный rollback — отдельное решение после проверки точного пути, ownership/identity, содержимого и необходимости данных. Для C не подразумевается удаление тома/форматирование.

ACL rollback не отменяет операции с данными, уже случившееся раскрытие информации, смену владельца вне snapshot или права файлов, созданных после изменения. Восстановление parent inheritance может изменить такие файлы; это не «полный откат системы».

Параметризованный DACL-only эскиз для отдельно подготовленного Windows PowerShell 5.1 с доверенным child module environment, **не выполнялся**:

~~~powershell
# RESTORE TEMPLATE ONLY. NOT EXECUTED.
$ErrorActionPreference = 'Stop'
# BeforeBytes из проверенного расшифрованного snapshot; actual path/identity
# и live post-change hash уже сверены по отдельной rollback-процедуре.
if (-not $RollbackPreconditionsVerified) { throw 'Rollback verification is required.' }
$Restore = Get-Acl -LiteralPath $ApprovedRestoreTarget
$Restore.SetSecurityDescriptorBinaryForm($BeforeBytes, [Security.AccessControl.AccessControlSections]::Access)
# Отдельно одобренная запись только DACL, не массовый owner/SACL reset:
# Set-Acl -LiteralPath $ApprovedRestoreTarget -AclObject $Restore
# После неё повторить read-only сравнение с baseline и validator.
# Никакого автоматического удаления файлов/поколений/томов.
~~~

$BeforeBytes должен быть из проверенного backup нужного объекта; $RollbackPreconditionsVerified — результат перечисленных проверок, а не свободно выставленный флаг ради запуска. Полная процедурная реализация restore выбирается после административного решения.

## 10. Read-only verification

Действующий Python helper, NTFS/reparse и фактические safe codes; блок выполнен без изменения прав:

~~~powershell
Set-Location -LiteralPath 'D:\RecommendationSystem'
@'
from pathlib import Path
import json
from Application import statistics_storage_safety as s
paths = [Path("D:/"), Path("D:/RecommendationSystem"),
         Path("D:/RecommendationSystem/filtered_data")]
for path, fact in zip(paths, s.inspect_acls(paths)):
    s.reject_reparse(path)
    s.local_volume(path)
    try:
        s.check_acl_facts(fact, private=False)
        state = "PASS"
    except s.StatisticsIndexError as error:
        state = str(error)
    print(json.dumps({"path": str(path), "state": state,
                      "owner_trusted": fact["owner_trusted"],
                      "protected": fact["protected"],
                      "unsupported": fact["unsupported"],
                      "other_allow_masks": [
                          hex(rule["mask"] & 0xffffffff) for rule in fact["rules"]
                          if rule["allow"] and not rule["trusted"]]}))
root = Path("D:/RecommendationSystem/filtered_data/statistics_index")
print(json.dumps({"index_root_exists": root.exists()}))
if root.exists():
    s.validate_storage_root(root)
'@ | & 'D:\RecommendationSystem\.venv312\Scripts\python.exe' -B -
if ($LASTEXITCODE -ne 0) { throw 'Read-only verification failed; stop.' }
~~~

Обезличенные owner/current token/inheritance/masks и доступность Windows PowerShell. Не печатает имён/полных SID; подготовлен и проверен как read-only:

~~~powershell
# Запустить блок в отдельном Windows PowerShell 5.1; не использовать профиль.
# Нормализуется только окружение этого дочернего диагностического процесса.
$ps51 = Join-Path ([Environment]::SystemDirectory) 'WindowsPowerShell\v1.0\powershell.exe'
$code = @'
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$env:PSModulePath=Join-Path ([Environment]::SystemDirectory) 'WindowsPowerShell\v1.0\Modules'
try {
 $me=[Security.Principal.WindowsIdentity]::GetCurrent()
 $system=[Security.Principal.SecurityIdentifier]::new([Security.Principal.WellKnownSidType]::LocalSystemSid,$null)
 $admins=[Security.Principal.SecurityIdentifier]::new([Security.Principal.WellKnownSidType]::BuiltinAdministratorsSid,$null)
 $ti=$null
 try {$ti=([Security.Principal.NTAccount]::new('NT SERVICE\TrustedInstaller')).Translate([Security.Principal.SecurityIdentifier])} catch {}
 function Category($id) {
  if($id.Value -eq $me.User.Value){return 'CURRENT_USER'}
  if($id.Value -eq $system.Value){return 'SYSTEM'}
  if($id.Value -eq $admins.Value){return 'ADMINISTRATORS'}
  if($ti -and $id.Value -eq $ti.Value){return 'TRUSTED_INSTALLER'}
  return 'OTHER_PRINCIPAL'
 }
 $rows=@(foreach($p in @('D:\','D:\RecommendationSystem','D:\RecommendationSystem\filtered_data')) {
  $a=Get-Acl -LiteralPath $p
  $rules=@(foreach($r in $a.Access) {
   $id=$r.IdentityReference.Translate([Security.Principal.SecurityIdentifier])
   if((Category $id) -eq 'OTHER_PRINCIPAL'){
    @{category='OTHER_PRINCIPAL';mask=('0x{0:X8}' -f ([long]$r.FileSystemRights -band 4294967295L));
      allow=($r.AccessControlType -eq 'Allow');inherited=$r.IsInherited;
      inheritance=$r.InheritanceFlags.ToString();propagation=$r.PropagationFlags.ToString()}
   }
  })
  @{path=$p;owner_category=(Category ($a.GetOwner([Security.Principal.SecurityIdentifier])));
    protected=$a.AreAccessRulesProtected;explicit_count=@($a.Access|Where-Object {-not $_.IsInherited}).Count;
    inherited_count=@($a.Access|Where-Object {$_.IsInherited}).Count;other_rules=$rules}
 })
 ConvertTo-Json -InputObject @{powershell_version=$PSVersionTable.PSVersion.ToString();
  identity_available=($null -ne $me.User);
  elevated_admin_token=([Security.Principal.WindowsPrincipal]::new($me)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator);
  rows=$rows} -Depth 7 -Compress
} catch { [Console]::WriteLine('READ_ONLY_ACL_DIAGNOSTIC_FAILED');exit 1 }
'@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($code))
& $ps51 -NoProfile -NonInteractive -EncodedCommand $encoded
if ($LASTEXITCODE -ne 0) { throw 'Read-only ACL diagnostic failed.' }
~~~

Результат current token probe: WindowsIdentity доступен, elevated_admin_token=False. Это не полный расчёт effective permissions всех principals; административные полномочия текущего процесса не запрашивались и не повышались.

После будущей подготовки обязательно вызвать validate_storage_root из фактического runtime account приложения, а не только из аккаунта администратора. CurrentUser в policy относится к запускающему процессу. Успешная проверка от другого пользователя может не подтвердить нужный GUI FullControl.

## 11. Proposed changes — NOT EXECUTED

| Операция | Target / полномочия | Проблема и эффект | Existing / future scope | Backup, проверка, stop и rollback |
| --- | --- | --- | --- | --- |
| P0: исправить protect_directory | Один production helper и тесты, отдельная разработческая задача | Устранить ложный success и подтвердить fail-closed | Не требует ACL операций | Regression first, mock native errors; при падении оставить setup заблокированным |
| P1 A/B: минимальный D:\ delta | Только одобренная explicit effective ACE D:\; WRITE_DAC, обычно elevated admin | Ограничить её до 0x1200AF; не менять owner/другие правила | Intended change — права на сам D:\; наследуемый шаблон не меняется, но API propagation scope надо подтвердить | Protected verified backup, exact principal/hash/IDs, post-check D:\ parent. Любое расхождение/неограниченный scope => stop; baseline restore по разделу 9 |
| P2 A: подготовить project/filtered_data | Два точных parent, владелец/WRITE_DAC; root D:\ отдельно | Разрешить inherited conflict после одобренной локальной конструкции DACL | Меняется inheritance этих каталогов, возможны effects будущих обновлений/descendants | Требуется отдельный inherited-rule план и snapshots scope; готовая mutating команда здесь не сформулирована |
| P2 B: новый private parent | D:\KANZLER-Statistics-Private; create только после D:\ PASS | Отделить namespace от существующего проекта | Только новый namespace, если parents безопасны; не ремонтировать существующий одноимённый каталог | Atomic private creation, exact approved runtime user, SYSTEM/Admins; validate/reparse/empty/owner checks; никакого cleanup при ошибке |
| P2 C: предоставить storage | Администратором утверждённый local fixed NTFS path | Избежать ACL рабочего D: | Только отдельно согласованная инфраструктура | Отдельное approval ресурсов/путей, никаких томов/VHD/format этим планом |
| P3: private index root | Точный путь выбранного варианта, безопасные parents | Protected DACL runtime CURRENT_USER FullControl + SYSTEM/Admins, без OTHER/Deny | Новый пустой leaf; последующие generations защищаются приложением | Перед созданием root отсутствует; использовать явную private DACL при создании, без промежуточного общего inherited доступа; validate из runtime account. При отказе не строить и не чинить автоматически |
| P4: согласовать lookup | GUI/CLI/query route для B/C, отдельная кодовая задача | Один index_dir во всех входных точках | Путь нового index, без изменения бизнес-контрактов | Проверить readonly status по общему пути; не запускать статистику/build ради конфигурации |

Для P1 обязательно выяснить фактическую роль конкретного OTHER_PRINCIPAL в приватном административном плане: какие группы/программы зависят от DELETE/EA/attribute прав D:\. Не подставлять «все пользователи» вместо неизвестного principal.

Параметризованный эскиз **только построения proposed DACL в памяти** для P1. Его будущий административный процесс должен иметь проверенный Windows PowerShell 5.1 module environment:

~~~powershell
# PARAMETERIZED PROPOSAL ONLY. NOT EXECUTED.
$ErrorActionPreference = 'Stop'
# ApprovedChange загружается из отдельного защищённого административного плана.
# Полные identifiers/SDDL не выводить и не сохранять в репозиторий.
if ($null -eq $ApprovedChange) { throw 'Approved private parameters are required.' }
$Target = $ApprovedChange.Target
if ($Target -cne 'D:\') { throw 'This proposal supports only the reviewed D: root.' }
$Principal = $ApprovedChange.PrincipalObject   # SecurityIdentifier, не guessed OTHER group
$ExpectedBeforeHash = $ApprovedChange.BeforeDescriptorSha256
$Before = Get-Acl -LiteralPath $Target
$bytes = $Before.GetSecurityDescriptorBinaryForm()
$hash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($bytes)).Replace('-','').ToLowerInvariant()
if ($hash -ne $ExpectedBeforeHash) { throw 'Descriptor changed; stop and review again.' }
$selectedRules = @($Before.Access | Where-Object {
 (-not $_.IsInherited) -and $_.AccessControlType -eq 'Allow' -and
 $_.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -eq $Principal.Value -and
 ([long]$_.FileSystemRights -band 4294967295L) -eq 0x001301BF -and
 $_.InheritanceFlags -eq 'None' -and $_.PropagationFlags -eq 'None'
})
if ($selectedRules.Count -ne 1) { throw 'Explicit rule is missing or ambiguous.' }
$Proposed = [Security.AccessControl.DirectorySecurity]::new()
$Proposed.SetSecurityDescriptorBinaryForm($bytes)
$Proposed.RemoveAccessRuleSpecific($selectedRules[0])
$rule = [Security.AccessControl.FileSystemAccessRule]::new(
 $Principal, [Security.AccessControl.FileSystemRights]0x001200AF,
 [Security.AccessControl.InheritanceFlags]::None,
 [Security.AccessControl.PropagationFlags]::None,
 [Security.AccessControl.AccessControlType]::Allow)
$Proposed.AddAccessRule($rule)
# Только in-memory объект. Ничего не выводить как raw DACL.
# Следующая запись ДОПУСТИМА лишь в отдельно согласованном apply script:
# Set-Acl -LiteralPath $Target -AclObject $Proposed
# Перед ней обязательны verified backup/restore, identity, dependency,
# inheritance-impact и anti-drift проверки из разделов плана.
~~~

В эскизе нет применения Set-Acl. Однострочная запись показана лишь как место будущей операции; её нельзя раскомментировать без отдельной проверенной apply-процедуры. $ApprovedChange должен содержать проверенные target identity, конкретный principal, baseline hash, verified backup/rollback record, dependency/inheritance-impact review и runtime-user mapping. Скрипт с одними произвольно выставленными booleans не является подтверждением этих условий.

Не меняется InheritOnly 0xE0010000. P1 не исправляет A целиком. Для A не выдаётся небезопасный универсальный RemoveAccessRule на inherited ACE. У новых private parent/root должна сразу применяться явная защищённая DACL (например, отдельная проверенная Windows Directory.CreateDirectory(path, DirectorySecurity) процедура), без force ownership и без обращения к существующим пользовательским каталогам.

Условия прекращения для всех операций: нет подтверждённого backup/restore; неизвестен principal/runtime account; hash/identity drift; reparse/не-NTFS; изменения scope/inheritance не оценены; root уже существует; policy FAIL; новый protect helper не проверен. Не выполнять автоматический retry, rollback/delete или расширение ACL ради PASS.

## 12. Критерии приёмки отдельной будущей настройки

- Каждое parent-звено проходит check_acl_facts(private=False); root и generation children — private policy.
- Выбранный runtime CURRENT_USER имеет необходимый FullControl; owner trusted, SYSTEM/Admins допускаются.
- Нет чужого управления namespace/reparse; путь не canonical/network/sync; том fixed writable NTFS.
- Atomic private creation и post-check не оставляют непроверенный root для build.
- Новый protect_directory не принимает запуск tool failure за success.
- После настройки отдельно подтверждаются SQLite create/close/journal в разрешённом synthetic private fixture, writer lock/staging identity, atomic current.json и прежняя cancellation/integrity regression. Это не основание строить production индекс в задаче настройки ACL.
- Сначала read-only storage admission, затем отдельно одобренный pilot/expected_gib и resource preflight. Get-Acl success не заменяет любой из этих пунктов.

## 13. Проверки текущей задачи

**79 passed, 1 failed, 9 deselected in 0.74s**, pytest exit 1. Падение не скрывалось через skip/xfail и не «исправлялось» ослаблением теста.

- 72 существующих policy/environment cases прошли, включая локальный helper fix.
- Внешний временный модуль содержит 7 прошедших mock/model checks и один failing regression: test_protect_missing_tool_error_must_not_be_reported_as_success — DID NOT RAISE.
- Он требует отказа protect_directory при смоделированном zero-exit/CommandNotFound. Это ранее существовавший дефект текущего неизменённого production-кода, а не новая регрессия реализации в данной задаче.
- Настоящая missing-tool PowerShell проверка с безопасной заменой имени независимо дала exit=0 и CommandNotFound; настоящий icacls не исполнялся.
- 9 deselected — нерелевантные storage fixtures; реальные ACL-mutating tests не выбирались. Не было production build, файлового pilot или прохода реальных Orders/Actions.
- In-memory candidate DACL прошёл policy, физический D:\ descriptor сохранился. Это dry-run проверки масок, не backup и не изменение ACL.
- Обе read-only команды проверены. Синтаксис proposed/restore fragments проверяется парсером без исполнения.

Артефакты этой задачи: D:\CodexTestTemp\statistics-acl-setup-ce3d614d3b5049ccb2c5f9e3ea89a423. Только sanitized metadata, pytest/log/XML, temporary synthetic tests и результаты проверок; security descriptor backup отсутствует.

Финальные checks production/tests/history/canonical metadata, Git diff и неожиданных файлов: PASS: initial tracked diff и hashes всех проверенных source/test/history файлов неизменны; canonical metadata прежние, sidecars отсутствуют; только один новый repository-файл — этот отчёт; root не создан; ACL facts прежние; 4 PowerShell-блока без parse errors; git diff --check чист; Ruff прошёл.

## 14. Что требует отдельного решения

1. Сначала отдельная узкая задача исправления protect_directory и regression tests, без реальных ACL операций.
2. Администратор выбирает предоставление изолированного NTFS storage C либо, при его недоступности, утверждает допустимость конкретного D:\ delta/варианта B. Для A дополнительно нужно явно принять влияние изменения inherited parent ACL проекта.
3. До любого apply выбрать защищённое backup место, restore-оператора/ключи, maintenance window и ограниченный проверяемый scope распространения правил.
4. Для B/C отдельно согласовать единый путь GUI/CLI. Никаких самостоятельных изменений DEFAULT_INDEX_DIR.
5. Лишь после этих решений подготовить конкретный исполняемый apply script и провести его отдельное одобрение. Затем другая задача — фактическая подготовка и read-only validation.

Планирование завершено: **ADMIN_DECISION_REQUIRED + PROTECT_DIRECTORY_FIX_REQUIRED**. Готовность к административному применению не объявлена. Ничего из предложенных ACL/backup/restore/setup операций не выполнялось.
