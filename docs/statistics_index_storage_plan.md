# STAT-INDEX-STORAGE-01 — Windows ACL и план подготовки INDEXED

Проверка: 31.10.2026. Проект D:\RecommendationSystem, ветка feature/hybrid-ranking, HEAD abb9cd8e98bbcd06c9906409e50ed98ea1fd4166. Production-код совпадает с HEAD. Новое содержимое репозитория в этой задаче — только этот документ.

## Решение

**Основной storage-статус: BLOCKED_SECURITY.** Причина отказа установлена, но безопасность изменения общих ACL для других пользователей/программ нельзя подтвердить по разрешённой обезличенной диагностике. Подготовка только нового private leaf проблему не устраняет. Готовность READY_FOR_STORAGE_SETUP сейчас не заявляется.

**Helper: ACL_HELPER_FIX_REQUIRED.** Установлена отдельная воспроизводимая проблема унаследованного PSModulePath; ниже предложен точечный фикс только окружения дочернего процесса, без реализации.

**RAM: STREAMING_CONFIRMED.** Блокирующая загрузка целой JSON-части не подтверждена; прежний вывод отозван. Статус RAM_REVIEW_REQUIRED по этой причине не требуется. Реальный peak RAM не измерялся; риски отдельных больших записей, merge graph и SQLite остаются.

Windows ACL, production root, пути приложения, TEMP/TMP, Registry, настройки PyCharm, canonical и резервная копия на C: не менялись. Реальный пилот, build, статистика и экспорт не запускались.

## 1. Пути и состояние

| Путь | Существует | Reparse/junction | Результат |
| --- | --- | --- | --- |
| D:\ | Да | Нет | Local fixed writable NTFS |
| D:\RecommendationSystem | Да | Нет | Local fixed writable NTFS |
| D:\RecommendationSystem\filtered_data | Да | Нет | Local fixed writable NTFS |
| D:\RecommendationSystem\filtered_data\statistics_index | Нет | Не обнаружены в лексической цепочке | PRIVATE_INDEX_ROOT_REQUIRED |
| D:\KANZLER-Statistics-Private | Нет | Не обнаружены в лексической цепочке | Только кандидат |
| D:\KANZLER-Statistics-Private\statistics_index | Нет | Не обнаружены в лексической цепочке | Только кандидат |

Существующие каталоги совпадают с resolve(), проверены lstat/identity; подмены пути не обнаружены. reject_reparse проверяет лексические звенья до resolve и запрещает '..'; local_volume требует fixed writable NTFS. Стандартный root остаётся вне canonical. Отсутствие root ожидаемо до первого build.

Штатный preflight INDEX_MISSING остаётся историческим результатом; --status и capture canonical в этой задаче повторно не запускались: для текущего исследования это не нужно.

## 2. Точные причины ACL-отказов

Получены только категории, маски, флаги наследования и результаты политики. Имена пользователей/доменов, полные SID и raw DACL не выводились и не сохранялись в документацию.

| Каталог | Владелец | Protected DACL | Явных ACE | Конфликтующий Allow | Наследуется | Namespace admission |
| --- | --- | --- | ---: | --- | --- | --- |
| D:\ | SYSTEM, trusted | Нет | 8 | OTHER_PRINCIPAL, 0x001301BF | Нет, applies to this folder | UNSAFE_STORAGE_ACL |
| D:\RecommendationSystem | CURRENT_USER, trusted | Нет | 0 | OTHER_PRINCIPAL, 0x001301BF | Да, от D:\ | UNSAFE_STORAGE_ACL |
| D:\RecommendationSystem\filtered_data | CURRENT_USER, trusted | Нет | 0 | OTHER_PRINCIPAL, 0x001301BF | Да, от D:\ через проект | UNSAFE_STORAGE_ACL |

Во всех трёх DACL: восемь Allow ACE, Deny нет, unsupported ACE нет; четыре Allow относятся к OTHER_PRINCIPAL, но **только одна эффективная ACE на каждом родителе вызывает parent-отказ**. Отказ не связан с недоверенным владельцем или отсутствующим Get-Acl.

### Что означает 0x001301BF

Это набор Modify-подобных прав, в том числе чтение, создание/запись данных, выполнение/проход, DELETE и запись атрибутов/EA. Пересечение с запрещённой parent-маской 0x000D0150 равно **0x00010110**:

| Бит | Право | Почему важно |
| --- | --- | --- |
| 0x00010000 | DELETE | Удаление самого объекта по его DACL, где операция допустима ОС |
| 0x00000100 | FILE_WRITE_ATTRIBUTES | Изменение атрибутов каталога |
| 0x00000010 | FILE_WRITE_EA | Изменение extended attributes |

FILE_DELETE_CHILD (0x40), WRITE_DAC (0x40000), WRITE_OWNER (0x80000) в этой конкретной ACE не установлены. Их отсутствие не отменяет три установленных запрещённых бита. Разрешение не является простым чтением.

Для дочерних project/filtered_data эффективный Modify допускает операции изменения содержимого и удаления самого каталога согласно Windows access checks. Для корня тома конкретные операции дополнительно ограничены ОС; политика всё равно консервативно отвергает эти маски. Это не доказательство фактического взлома или проверка токена произвольного пользователя. Полный effective AccessCheck с группами/привилегиями не выполнялся.

### Источник наследования — важное уточнение

На D:\ имеются **две отдельные явные Allow ACE одного OTHER_PRINCIPAL**:

1. 0x001301BF, InheritanceFlags=None, PropagationFlags=None — действует на сам D:\.
2. 0xE0010000, ContainerInherit|ObjectInherit, InheritOnly — шаблон для потомков. Generic-биты дают потомкам эффективный Modify 0x001301BF и отдельную наследуемую форму шаблона.

Сопоставление выполнялось внутри процесса по реальному principal, типу Allow и нормализованной specific mask; идентификатор не выводился. Для inherited Modify каждого дочернего каталога найден ровно один исходный шаблон на D:\. Также подтверждено, что effective ACE и шаблон на D:\ относятся к одному principal.

Таким образом, это **одна проблема конфигурации общего предка с двумя формами правил**, а не три независимые локальные настройки. Но effective ACE на D:\ сама не наследуется: дочерние Modify получены от отдельного InheritOnly-шаблона. Исправление только effective ACE D:\ оставит существующие project/filtered_data небезопасными.

Другие OTHER_PRINCIPAL Allow:

- 0x001200A9 — Read/Execute; для parent разрешено, для private root запрещено.
- 0xA0000000 — GenericRead|GenericExecute, ContainerInherit|ObjectInherit/InheritOnly; на текущий parent не действует.
- Указанный 0xE0010000/InheritOnly также не вызывает parent-отказ сам по себе, но передаёт Modify потомкам.

Нельзя судить о допустимости ACE только по количеству foreign Allow; важны mask, InheritOnly и режим private.

## 3. Политика check_acl_facts

Generic mappings в текущем коде:

| Generic bit | Добавляемая file-specific mask |
| --- | --- |
| GENERIC_ALL 0x10000000 | 0x001F01FF |
| GENERIC_WRITE 0x40000000 | 0x00120116 |
| GENERIC_READ 0x80000000 | 0x00120089 |
| GENERIC_EXECUTE 0x20000000 | 0x001200A0 |

Сначала mask приводится к unsigned 32-bit, specific bits добавляются через OR; generic bits не удаляются. Например, 0xE0010000 расширяется до 0xE01301BF. Для сопоставления источника наследования сравнивались specific bits, что учитывает преобразование generic template в effective ACE.

Допустимые principals: текущий пользователь, SYSTEM, ADMINISTRATORS и TRUSTED_INSTALLER. Остальные считаются OTHER_PRINCIPAL даже если текущий пользователь сам входит в такую группу.

Общее требование: owner_trusted=True и отсутствие unsupported/null DACL. В противном случае UNTRUSTED_STORAGE_OWNER_OR_ACL.

- **Parent:** не требуется protected DACL или отдельная CurrentUser FullControl ACE; foreign Allow отвергается только при !InheritOnly и пересечении с 0xD0150. Чтение/проход разрешены. Биты добавления файла/подкаталога 0x2/0x4 сами по себе не дают отказа.
- **Private root/children:** запрещён любой foreign Allow, включая read-only и InheritOnly; запрещены Deny ACE. Root должен быть protected. Применимые CurrentUser Allow должны совокупно давать 0x1F01FF, иначе OWNER_FULL_CONTROL_REQUIRED.
- **Deny:** на parent отдельная Deny не компенсирует опасную Allow. Это намеренная allowlist-политика, а не полная симуляция AccessCheck. На private объектах Deny => UNSUPPORTED_PRIVATE_DENY_ACL.

Наблюдаемый отказ соответствует текущей документированной политике; ложное срабатывание не установлено. Замена на os.access, доверие всем группам или пропуск D:\ не предлагаются.

Семантика прав и generic mapping сверена с [Microsoft File security and access rights](https://learn.microsoft.com/en-us/windows/win32/fileio/file-security-and-access-rights) и [Access Mask](https://learn.microsoft.com/en-us/windows-hardware/drivers/ifs/access-mask). Политика приложения строже обычного Windows открытия отдельного дочернего файла.

## 4. Windows PowerShell helper

Исполняемый файл: C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe.
Windows PowerShell **5.1.26100.9711, Desktop**. Python — D:\RecommendationSystem\.venv312\Scripts\python.exe.

Во всех случаях -NoProfile/-NonInteractive/-EncodedCommand, UTF-16LE command. Системные переменные и Registry не изменялись.

| Сценарий | Результат |
| --- | --- |
| Обычный startup Windows PowerShell без навязанного PSModulePath | OK |
| Python проекта, унаследованное окружение | Get-Acl: CouldNotAutoloadMatchingModule, CommandNotFoundException |
| Python, отдельная копия env со стандартным Windows PowerShell module path | OK |
| Штатный inspect_acls, унаследованное окружение | ACL_INSPECTION_FAILED |
| Штатный inspect_acls, минимально корректный child env | JSON/facts успешно получены |

Первый сценарий — новый Windows PowerShell процесс с обычным построением module path после удаления унаследованной переменной в **его env**, а не проверка интерактивного профиля пользователя. Живой PyCharm и его настройки не исследовались. Нельзя приписывать происхождение переменной именно PyCharm.

Read-only import Microsoft.PowerShell.Security в унаследованном окружении дополнительно дал FormatXmlUpdateException,Microsoft.PowerShell.Commands.ImportModuleCommand. Важен не текст сообщения, а воспроизводимое различие env: без унаследованного module path или со стандартным путём ошибка исчезает.

При корректном child env подтверждены Get-Acl, WindowsIdentity.GetCurrent(), RawSecurityDescriptor, преобразование FileSystemRights в int, чтение ACL временного каталога с русским именем и пробелом, Unicode JSON round-trip. Диагностический Unicode-каталог создан только внутри внешнего D:\CodexTestTemp, без установки ACL и реальных данных. Штатный inspect_acls также успешно прочитал ACL этого Unicode-каталога (1 fact). Helper выдаёт технический JSON без имён; его декодирование UTF-8 проверено отдельно.

Ошибки разделены:

| Категория | Наблюдение |
| --- | --- |
| Реально запрещённые ACL | UNSAFE_STORAGE_ACL после успешной инспекции |
| Autoload/module environment | Воспроизведён, ACL_INSPECTION_FAILED |
| PowerShell не запускается | Не воспроизведено; executable доступен; synthetic FileNotFoundError проверен |
| Read access denied descriptor | Не наблюдалось; synthetic PermissionError проверен |
| Unsupported structure | Не обнаружена; synthetic unsupported => UNTRUSTED_STORAGE_OWNER_OR_ACL |
| Некорректный JSON/encoding/count | Native успех проверен; synthetic ошибки fail closed |
| Несуществующий путь | GetAcl_PathNotFound / ItemNotFoundException, не UNSAFE_STORAGE_ACL |
| Timeout | Реального timeout не было; synthetic TimeoutExpired проверен |

Диагностические probes перехватывают исключение для безопасного JSON, поэтому их process exit=0 сам по себе не означает status=OK. Штатный helper без такого probe-wrapper возвращает безопасный ACL_INSPECTION_FAILED при ненулевом exit. validate_storage_root также оборачивает OSError/SubprocessError в этот код; fail-open fallback нет.

### Предлагаемый минимальный фикс — НЕ РЕАЛИЗОВАН

Application/statistics_storage_safety.py::inspect_acls: создать копию os.environ, удалить case-insensitive ключи PSModulePath из копии, задать только встроенный путь Windows PowerShell SystemRoot\System32\WindowsPowerShell\v1.0\Modules и передать env= в subprocess.run. Не изменять os.environ вызывающего процесса, Registry, PATH или настройки IDE.

Сохранить текущие timeout, Unicode command encoding, безопасный JSON и fail-closed поведение; не делать повторную попытку при неудаче с ослабленными ACL. В отдельной задаче добавить repository regression tests: загрязнённый parent env, неизменность родительского env, Unicode, отсутствующий executable/module, timeout, invalid JSON, denied access. Сейчас production/tests репозитория не менялись.

Это узкий обход дефектного окружения загрузки модулей, не исправление реальных ACL. Причина таких конфликтов module search описана в [Microsoft about_PSModulePath](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_psmodulepath?view=powershell-7.5).

## 5. Варианты размещения и согласованность путей

| Вариант | Существующие родители | Нужные условия | Преимущества / ограничения |
| --- | --- | --- | --- |
| Штатный D:\RecommendationSystem\filtered_data\statistics_index | D:\, проект, filtered_data — все FAIL | Согласованная подготовка всех трёх namespace, затем private leaf | GUI/CLI согласованы автоматически; затрагиваются существующие общие каталоги |
| D:\KANZLER-Statistics-Private\statistics_index | D:\ — FAIL; новый parent отсутствует | Согласованная подготовка D:\, новый private parent и leaf | Можно сохранить ACL проекта/filtered_data, но нужен отдельный маленький этап согласования путей GUI/CLI |

Новый путь на том же D: не обходит общий предок. Подтверждённого безопасного варианта «создать только private leaf под уже безопасным parent» сейчас нет.

**Рекомендация:** до решения вопроса общих ACL сохранить штатный index_dir как действующую конфигурацию; не создавать root. Если администратор подтвердит допустимость узкого изменения только effective non-inheritable ACE D:\, отдельный private namespace становится предпочтительным кандидатом по меньшей области затрагиваемых существующих каталогов. Он требует отдельно согласованного общего пути в GUI/CLI. Это условная альтернатива, не готовое безопасное размещение.

Самый узкий предмет согласования для D:\: пересмотреть только OTHER_PRINCIPAL Allow 0x001301BF, applies-to-this-folder. Удаление из неё parent-blocking bits даёт 0x001200AF (Read/Execute + AddFile/AddSubdirectory), что проходит текущую parent-политику. Это подтверждено синтетически, **не доказывает допустимость изменения для всех программ**.

Не менять наследуемый 0xE0010000 на всём D:\: это может затронуть наследование существующего дерева. После изменения только effective D:\ ACE проект и filtered_data сохранят inherited Modify и останутся FAIL; альтернативный новый private parent позволяет их не ремонтировать. На D:\ не предлагается универсальная исполняемая команда Set-Acl/icacls: без оценки назначения OTHER_PRINCIPAL и зависимостей нельзя подтвердить безопасность операции. Даже отсутствие /T не гарантирует отсутствие обработки наследования Windows API. Это отдельный BLOCKED_SECURITY, а не разрешение на автоматический ремонт.

### Кто должен искать индекс

| Точка входа | Текущее поведение |
| --- | --- |
| statistics_index.DEFAULT_INDEX_DIR | PROJECT_ROOT/filtered_data/statistics_index |
| scripts/build_statistics_index.py | --index-dir, default DEFAULT_INDEX_DIR |
| statistics_query.calculate_indexed_statistics | index_dir or DEFAULT_INDEX_DIR |
| statistics_slices.calculate_cached_statistics | Передаёт index_dir в INDEXED/AUTO |
| scripts/dataset_statistics.py | --index-dir, default None |
| GUI, Application/tabs/dataset_statistics_tab.py | Запускает dataset_statistics.py без --index-dir; используется штатный default |

Альтернативный CLI --index-dir технически поддержан. Но build в альтернативном месте не будет автоматически найден GUI. Потребуется отдельное согласованное изменение общей конфигурации/передачи пути; нельзя менять только builder. Новый параметр, пути или GUI в этой задаче не добавлялись. Индекс не переносится на C:, сеть или синхронизируемый storage.

## 6. Исправление вывода о RAM

**STREAMING_CONFIRMED для фактического build path.** Предыдущий preflight ошибочно связал наличие load_export_part/json.load с индексатором. Все утверждения прежнего документа о необходимости загрузить в RAM целый файл 1,44 GiB при build и о задержке отмены именно внутри такого json.load следует считать отозванными. Размер файла на диске может быть верным, вывод о его целиком материализуемом объёме — нет. Исходный docs/statistics_index_preflight.md оставлен побайтово неизменным.

Подтверждённая цепочка:

~~~text
build_statistics_index
  records(name, effective sources)
    raw_reader.iter_export
      customers_stream.iter_json_records
        ijson.basic_parse(use_float=False, buf_size=65536)
          ObjectBuilder для текущего export record
~~~

Actions и Orders идут через records в Projector.action/order. CustomerMerges идёт через тот же records, adapt_customer_merge и CustomerIdResolver. iter_export перечисляет части и yield from iter_json_records; вызова load_export_part нет.

Поиск ссылок в Application/scripts: load_export_part используется schema_profiler.py и определён в raw_reader.py. Profiler не вызывается исследованным build path. json.load внутри metadata/ref helpers не является загрузкой raw Orders/Actions/CustomerMerges частей.

Runtime ijson **3.4.0.post0**, backend **yajl2_c**. Синтетически для всех трёх export types первая запись выдана до invalid tail за 200 000 пробелов, с json.load/load_export_part, заменёнными на немедленный fail. Это проверяет фактический iter_export, не только customers-specific оболочку. Decimal и отложенная ошибка хвоста сохранены.

| Объект | Что действительно может занимать RAM |
| --- | --- |
| Streaming parser | Буфер 64 KiB и внутренние parser structures; это не hard limit размера записи |
| Raw record | ObjectBuilder создаёт целую одну запись с вложенными массивами; на переходах возможны одновременно ссылки на предыдущую и новую запись |
| Один Order | Полный lines/payments внутри записи плюс rows/eligible/adapted и временные структуры Projector; очень большой заказ может быть дорогим |
| Fingerprint | Рекурсивный обход одного raw snapshot, сортировка его ключей; не накопление всех raw orders в builder |
| CustomerIdResolver | Полные _parents/_canonical dictionaries по merge aliases, временные trail/seen при обходе |
| ProductResolver | Каталог item IDs/metadata и структуры resolver; index path использует resolve, не общий накопительный report pipeline |
| Customer/order SID mappings | В builder disk-backed таблицы SQLite customer_ids/order_ids, не Python dict всех ID |
| Customers SQLite | mode=ro, cursor по SELECT id/json_extract birthDate/sex ORDER BY id; строки обрабатываются по очереди, JSON parsing одного profile выполняет SQLite |
| SQLite | Page cache основной БД ~32 MiB, demographics ~8 MiB; main temp_store=MEMORY, SQL sorting/index build/ANALYZE/integrity working memory |

Projector не собирает массив всех событий или всех order lines всей истории. Его per-record буферы, адаптированные данные и JSON fields освобождаются по мере обработки. По завершении проекции builder явно освобождает identities/products/catalog/metadata/projector до создания основных SQL-индексов.

Остающиеся риски: отдельная огромная/глубоко вложенная запись, большой merge graph, resolver post-validation, SQL sorts/indexing и внешнее давление на available memory/commit. BuildGuard остаётся кооперативным мониторингом, а не OS hard cap; внутри разбора одной большой записи и graph validation отмена/проверка могут задержаться. Peak RAM и полный граф на реальных данных не измерялись. Streaming устраняет прежнее основание RAM-блокера, но не доказывает ресурсную готовность полного build. Пилот и обоснование expected_gib по-прежнему нужны.

## 7. Read-only verification

Следующий блок можно повторять без создания index root и без установки ACL. Нормализация PSModulePath действует только в дочернем Python/PowerShell env. NAMESPACE_PASS проверяет parent-политику; для существующего private root дополнительно нужен validate_storage_root, не только этот результат.

~~~powershell
Set-Location -LiteralPath 'D:\RecommendationSystem'
@'
import os, subprocess, sys
from pathlib import Path
env = os.environ.copy()
for key in list(env):
    if key.casefold() == "psmodulepath":
        del env[key]
env["PSModulePath"] = str(Path(os.environ["SYSTEMROOT"]) / "System32/WindowsPowerShell/v1.0/Modules")
code = r"""
import json
from pathlib import Path
from Application import statistics_storage_safety as s
targets = [Path("D:/"), Path("D:/RecommendationSystem"),
           Path("D:/RecommendationSystem/filtered_data"),
           Path("D:/RecommendationSystem/filtered_data/statistics_index"),
           Path("D:/KANZLER-Statistics-Private"),
           Path("D:/KANZLER-Statistics-Private/statistics_index")]
for p in targets:
    s.reject_reparse(p)
    if not p.exists():
        print(json.dumps({"path":str(p),"state":"MISSING"}))
        continue
    s.local_volume(p)
    fact = s.inspect_acls([p])[0]
    try:
        s.check_acl_facts(fact, private=False)
        state = "NAMESPACE_PASS"
    except s.StatisticsIndexError as exc:
        state = str(exc)
    print(json.dumps({"path":str(p),"namespace":state,
                      "owner_trusted":fact["owner_trusted"],
                      "protected":fact["protected"],"unsupported":fact["unsupported"]}))
"""
raise SystemExit(subprocess.run([sys.executable, "-B", "-c", code], env=env).returncode)
'@ | & 'D:\RecommendationSystem\.venv312\Scripts\python.exe' -B -
if ($LASTEXITCODE -ne 0) { throw 'Read-only helper failure; stop.' }
~~~

Если root будет подготовлен в отдельной задаче, проверка результата: из того же дочернего env вызвать штатный validate_storage_root точного пути; отдельно проверить, что он пуст, owner trusted, protected, без foreign Allow/Deny, CurrentUser FullControl. При любом отказе не строить индекс.

## 8. Proposed changes — NOT EXECUTED

**Это не разрешение исполнять блок сейчас.** Он подготавливает только штатный новый leaf, после отдельно одобренной подготовки родителей. При текущих ACL должен остановиться до записей. Общие parent ACL он не меняет.

До любого изменения общих ACL администратор должен сохранить их защищённый snapshot и определить точные роли OTHER_PRINCIPAL без помещения идентификаторов в репозиторий. Без этого конкретная команда изменения D:\ не считается безопасно подготовленной.

Блок ниже сохраняет snapshot родителей перед созданием leaf в DPAPI CurrentUser ciphertext. Он не печатает raw descriptors/SID. Backup-файл создаётся исключительно как новый, без перезаписи; ключи пользователя необходимы для расшифровки. Каталог backup — внешний технический D:\CodexTestTemp текущей диагностики, не production. До фактической операции нужно повторно проверить его существование/происхождение.

Новый leaf создаётся сразу с protected DACL CurrentUser/SYSTEM/ADMINISTRATORS, через Windows PowerShell 5.1/.NET Framework Directory.CreateDirectory(path, DirectorySecurity). Наличие overload проверено reflection без его вызова. Родитель уже должен существовать: недостающую цепочку создавать нельзя. Подготовка private ACL до создания исключает промежуточный leaf с унаследованными общими Allow. См. [Microsoft Directory.CreateDirectory](https://learn.microsoft.com/en-us/dotnet/api/system.io.directory.createdirectory?view=netframework-4.8.1).

~~~powershell
# Только после отдельного одобрения и подготовки ВСЕХ родителей.
# Запускать в Windows PowerShell 5.1 (Desktop).
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSEdition -ne 'Desktop') { throw 'Use Windows PowerShell 5.1.' }
Set-Location -LiteralPath 'D:\RecommendationSystem'
$python = 'D:\RecommendationSystem\.venv312\Scripts\python.exe'
$target = 'D:\RecommendationSystem\filtered_data\statistics_index'
$backupFile = 'D:\CodexTestTemp\statistics-index-storage-b021cf92eda54013a34689e407206508\parents-before-storage-setup.dpapi'
$env:PSModulePath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\Modules'

# Read-only admission до любых записей. Сейчас здесь ожидается отказ.
@'
import os
from pathlib import Path
from Application import statistics_storage_safety as s
root = Path(r"D:\RecommendationSystem\filtered_data\statistics_index")
s.reject_reparse(root)
s.local_volume(root.parent)
if os.path.lexists(root):
    raise SystemExit("Root exists: inspect separately, never repair/overwrite.")
for name in ("OneDrive","OneDriveCommercial","OneDriveConsumer"):
    if os.environ.get(name) and root.is_relative_to(Path(os.environ[name]).absolute()):
        raise SystemExit("Synchronized namespace is unsupported.")
parents = tuple(root.parents)
if not all(p.is_dir() for p in parents):
    raise SystemExit("Missing parent: stop, do not create the chain.")
for fact in s.inspect_acls(parents):
    s.check_acl_facts(fact, private=False)
s.reject_reparse(Path(r"D:\CodexTestTemp\statistics-index-storage-b021cf92eda54013a34689e407206508"))
'@ | & $python -B -
if ($LASTEXITCODE -ne 0) { throw 'Parent admission failed; no changes performed.' }

# Сохранить snapshot родителей до подготовки leaf. Raw DACL не печатается.
# DPAPI CurrentUser: backup нельзя прочитать без ключей этого пользователя.
Add-Type -AssemblyName System.Security
$snapshot = @('D:\', 'D:\RecommendationSystem', 'D:\RecommendationSystem\filtered_data') | ForEach-Object {
    $acl = Get-Acl -LiteralPath $_
    [pscustomobject]@{path=$_;descriptor=[Convert]::ToBase64String($acl.GetSecurityDescriptorBinaryForm())}
}
$plain = [Text.Encoding]::UTF8.GetBytes(($snapshot | ConvertTo-Json -Compress))
$entropy = [Text.Encoding]::UTF8.GetBytes('KANZLER statistics storage ACL snapshot v1')
$cipher = [Security.Cryptography.ProtectedData]::Protect($plain, $entropy, [Security.Cryptography.DataProtectionScope]::CurrentUser)
$stream = [IO.File]::Open($backupFile, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try { $stream.Write($cipher, 0, $cipher.Length); $stream.Flush($true) } finally { $stream.Dispose() }

# Новый leaf сразу с private DACL; родитель существует, цепочку не создаём.
# Никаких Set-Acl общих родителей, /T, /reset или изменения владельца.
if (Test-Path -LiteralPath $target) { throw 'Root appeared; stop without repair.' }
$me = [Security.Principal.WindowsIdentity]::GetCurrent().User
$system = [Security.Principal.SecurityIdentifier]::new([Security.Principal.WellKnownSidType]::LocalSystemSid, $null)
$admins = [Security.Principal.SecurityIdentifier]::new([Security.Principal.WellKnownSidType]::BuiltinAdministratorsSid, $null)
$private = [Security.AccessControl.DirectorySecurity]::new()
$private.SetAccessRuleProtection($true, $false)
$inherit = [Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit
foreach ($identity in @($me, $system, $admins)) {
    $rule = [Security.AccessControl.FileSystemAccessRule]::new($identity, [Security.AccessControl.FileSystemRights]::FullControl, $inherit, [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow)
    $private.AddAccessRule($rule)
}
$null = [IO.Directory]::CreateDirectory($target, $private)

# Проверка штатным validator; при отказе не удалять и не ремонтировать автоматически.
& $python -B -c 'import sys; from Application.statistics_storage_safety import validate_storage_root; validate_storage_root(sys.argv[1]); print("PRIVATE_ROOT_VALIDATED")' $target
if ($LASTEXITCODE -ne 0) { throw 'Validation failed; stop without cleanup/build.' }
if (@(Get-ChildItem -LiteralPath $target -Force).Count -ne 0) { throw 'Unexpected files; stop without cleanup/build.' }
~~~

Область изменения при всех PASS: один новый пустой D:\RecommendationSystem\filtered_data\statistics_index и один encrypted technical backup. Ни D:\, ни project/filtered_data/input_data/canonical ACL не меняются. Команды фактического создания/установки ACL здесь не исполнялись и как операция среды не тестировались; проверен синтаксис, наличие API и read-only preconditions.

Обратимость: у нового leaf нет прежнего DACL; при отказе он остаётся для отдельного осмотра, автоматического cleanup нет. Удаление допустимо лишь после отдельной проверки точного пути/владельца/идентичности и отсутствия данных, в отдельной задаче. Snapshot родителей — диагностическая/восстановительная основа, а не команда безусловного restore: нельзя перезаписывать параллельные изменения. Восстановление ACL не отменяет последствия уже выполненных операций или раскрытия данных. Для альтернативного namespace нужен отдельный проверенный script после согласования его общего пути GUI/CLI; подставлять путь в этот блок без новой проверки нельзя.

## 9. Целевые тесты и сохранность

**67 passed in 0.97s**, exit 0. Существующие целевые сценарии: private/parent/generic/deny/owner ACL policy, отсутствующий root без создания, synchronized root, reparse/'..'; customers_stream и reader multipart/decimal/invalid-envelope.

Внешний временный test_storage_diagnostic.py дополнительно проверил read-only helper protocol и Unicode literal, invalid JSON/UTF-8/shape/count, nonzero exit, missing executable/access/timeout, unsupported descriptor, различие Modify и read/create mask, отсутствие build/ACL repair при missing root и progressive reading Actions/Orders/Merges. Production index/build на настоящих данных не запускался; тест missing root прекращается до SQLite. ACL-mutating tests, включая test_private_directory_protection_on_new_directory, не выбраны. Установка ACL даже на временном каталоге не выполнялась.

Точный список pytest nodes и команды сохранён в pytest_summary.json внешнего каталога. Python на D:, -B -m pytest -q --tb=short -p no:cacheprovider, PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; basetemp/junitxml вне проекта на D:. TEMP/TMP не менялись.

Ruff для 14 исследованных source/CLI/test-файлов, включая внешний diagnostic test: All checks passed. Проверки синтаксиса команд, git diff --check, неизменности preflight/production и повторных ACL: PASS: 2 PowerShell-блока без parse errors, встроенный Python компилируется; git diff --check чист; production/preflight/ACL неизменны, production и alternative root отсутствуют.

Артефакты: D:\CodexTestTemp\statistics-index-storage-b021cf92eda54013a34689e407206508. Sanitized ACL/env/path JSON, inheritance proof, временные synthetic tests и logs. Нет raw DACL, имён учётных записей или исходных записей покупателей. Исходный preflight сохранён как историческое свидетельство, SHA-256: c892ded087f43a721de339cb3903ad806e740b0247f8d148fa0f3ddd5d851e5f.

## 10. Существенные проблемы и следующий этап

| Приоритет | Файл/функция | Проблема / влияние | Рекомендация, результат, риск и тест |
| --- | --- | --- | --- |
| High | statistics_storage_safety.py::validate_storage_root/check_acl_facts, общая ACL D:\ | Общий source Modify даёт три отказа namespace; новый leaf не поможет | Отдельное решение администратора о parent rights. Ожидается безопасная цепочка; риск parent ACL High. Расчёты сохранятся, доступ программ может измениться. Новый unit-test не нужен до реализации; фактический результат требует read-only повторной проверки |
| Medium | statistics_storage_safety.py::inspect_acls | Унаследованный module path ломает Get-Acl, даже корректные ACL недоступны helper | Узкий child env fix с repository regression tests. Ожидается воспроизводимая инспекция; риск Low при неизменных timeout/fail-closed. Публичное API/расчёты сохраняются. Сначала добавить tests загрязнённого env и протокола |
| Low, исправление документации | Предыдущий statistics_index_preflight.md, вывод RAM | Ошибочно приписан build вызов load_export_part; это могло привести к ненужному reader rewrite | Исправлено выводом в новом отчёте, исходный документ сохранён. Поведение приложения не меняется, риск Low; streaming подтверждён synthetic regression |

Неустранённые блокеры: согласование общих ACL D:\ и, для штатного пути, project/filtered_data; реализация child env fix; создание/проверка private root в отдельной задаче. При выборе альтернативы — ещё согласованная конфигурация GUI/CLI. Пилот не начинать до подготовки storage.

Следующий конкретный этап: отдельная маленькая задача исправления env только inspect_acls с regression tests; параллельно пользователь/администратор принимает решение о применимости узкого parent ACL изменения и выборе штатного или отдельного private namespace. Затем отдельная задача фактической подготовки с проверками и защищённым backup. После неё можно заново рассматривать ограниченный пилот.

Диагностика завершена. ACL, root, production-код и индекс не изменены; автоматического продолжения нет.
