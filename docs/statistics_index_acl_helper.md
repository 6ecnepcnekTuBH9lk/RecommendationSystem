# STAT-INDEX-ACL-HELPER-01 — изоляция PSModulePath

Дата: 31.10.2026. Проект D:\RecommendationSystem, ветка feature/hybrid-ranking.
Исходный HEAD: abb9cd8e98bbcd06c9906409e50ed98ea1fd4166. Git commit/staging/push не выполнялись.

**PASS_HELPER_FIXED.** Windows PowerShell успешно читает ACL при исходном и намеренно загрязнённом окружении Python. Прежняя политика продолжает отвергать все три небезопасных родителя. Общий storage-статус остаётся **BLOCKED_SECURITY**; production index root не создан.

## 1. Исходная проблема и сравнение вариантов

Штатный inspect_acls запускал powershell.exe -NoProfile -NonInteractive -EncodedCommand, наследуя PSModulePath целиком. На этой машине Windows PowerShell 5.1 не мог автоматически загрузить Get-Acl: CouldNotAutoloadMatchingModule. Сам executable и Windows security API доступны. Ранее отдельно получен FormatXmlUpdateException при импорте Microsoft.PowerShell.Security — см. исторический D:\RecommendationSystem\docs\statistics_index_storage_plan.md.

Повторный read-only эксперимент до исправления:

| Вариант | Get-Acl | Только встроенные пути поиска после startup |
| --- | --- | --- |
| Исходный Python environment | Ошибка автозагрузки | Нет |
| A: удалить PSModulePath из копии child env | Работает | Нет |
| B: передать только встроенный путь в child env | Работает | Нет: PowerShell дополняет список при startup |
| B плюс закрепление встроенного пути внутри child script до Get-Acl | Работает | Да |

Вариант A на текущей машине восстановил загрузку, но возвращает обычный поиск, включающий user/all-user locations. По этой причине одного удаления переменной недостаточно для ограничения доверенными встроенными модулями.

Выбран **B с закреплением внутри дочернего скрипта**. Автоматическое добавление путей при startup проверено фактически; оно также описано в [Microsoft about_PSModulePath](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_psmodulepath?view=powershell-7.5). Приватные значения env и пользовательские paths в отчёт не включены.

Причина установлена на уровне inherited module-search environment. Отдельный аудит происхождения каждого его элемента и настроек живого PyCharm не проводился; ошибка не приписывается исключительно IDE.

## 2. Что изменено

| Файл | Изменение |
| --- | --- |
| D:\RecommendationSystem\Application\statistics_storage_safety.py | Маленький _powershell_environment, отдельный env subprocess, закрепление module lookup до автозагрузки, безопасные ошибки запуска и проверка структуры JSON |
| D:\RecommendationSystem\tests\test_statistics_storage_safety.py | 46 новых параметризованных случаев для environment/protocol/fail-closed/policy invariance |
| D:\RecommendationSystem\docs\statistics_index_acl_helper.md | Этот отчёт |

Другие production-модули, GUI, canonical, кеш и расчёты Buyer/RFM/Cohorts не менялись. Публичные сигнатуры сохранены, новые зависимости не добавлены.

### Формирование child environment

1. Получить системный каталог Windows через kernel32.GetSystemDirectoryW. Возврат 0 или переполнение буфера означает ACL_INSPECTION_FAILED. Значения SystemRoot из родительского env для выбора модулей не используются.
2. Сформировать путь <Windows system directory>\WindowsPowerShell\v1.0\Modules.
3. Проверить отсутствие reparse в его цепочке и наличие встроенных manifests Microsoft.PowerShell.Security и Microsoft.PowerShell.Utility. При отсутствии/недоступности/небезопасном пути отказать, не искать пользовательский fallback.
4. Создать отдельный dict из os.environ, исключив все варианты регистра ключа PSModulePath. Добавить единственное нормализованное значение PSModulePath.
5. Передать dict как env= в subprocess.run. Остальные переменные, включая PATH, TEMP и TMP, сохранить.
6. Первой строкой закодированного PowerShell-script закрепить этот же PSModulePath. Так дополнительные startup paths не участвуют в дальнейшей автозагрузке Get-Acl и ConvertTo-Json.

GetSystemDirectoryW возвращает расположение системного каталога ОС; здесь выполняется только read-only запрос. См. [Microsoft GetSystemDirectoryW](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getsystemdirectoryw). Файлы системных модулей не создавались и не менялись.

Сохранены powershell.exe, -NoProfile, -NonInteractive, UTF-16LE -EncodedCommand, capture_output=True и timeout=30. ExecutionPolicy Bypass не добавлен. Нормализованный путь корректно экранируется как PowerShell literal; Unicode/пробелы/апостроф и сохранение текста проверяемого пути покрыты тестом.

os.environ родительского Python не меняется. Закрепление переменной в PowerShell действует только в уже созданном дочернем процессе и не записывает системные/user settings, Registry или настройки IDE.

### Безопасные ошибки и протокол

Ошибки подготовки child environment, запуска executable, доступа, timeout и ненулевой exit дают ACL_INSPECTION_FAILED без вывода stderr и исходного исключения. Неверный UTF-8/JSON, тип результата, количество facts или структура fact/rule также дают безопасный отказ.

Проверяются фиксированные поля протокола, настоящие bool вместо truthy строк/чисел, rules как list и mask как 32-bit int (signed или unsigned; bool не допускается). Это не признание ACL безопасным: структурно корректные facts далее проходят прежний check_acl_facts. Unknown principals, неподдерживаемый descriptor, небезопасные маски и недоверенный owner не получают послаблений.

Никакого os.access, пустого списка при сбое, отключения проверок или retry с более слабыми условиями нет.

## 3. Неизменность security policy

check_acl_facts, validate_storage_root, reject_reparse, local_volume и проверка принадлежности staging не изменены; неизменность их исходного кода относительно HEAD проверена отдельно.

Сохранены:

- CURRENT_USER/SYSTEM/ADMINISTRATORS/TRUSTED_INSTALLER allowlist;
- OTHER_PRINCIPAL как недоверенная категория;
- owner validation и generic mask mappings;
- private root protected DACL и CurrentUser Full Control;
- запрет foreign Allow и Deny на private объектах;
- различие private root и родителей, учёт InheritOnly;
- проверки всех родителей, local NTFS и reparse/junction;
- отказ PRIVATE_INDEX_ROOT_REQUIRED для отсутствующего root.

При одинаковых синтетических facts результат политики до и после прохождения через исправленный helper одинаков. Покрыты допустимый private root, Modify/Delete 0x001301BF, foreign read, owner, unsupported descriptor, protected/inherited flag, owner read-only и Deny.

## 4. Read-only Windows smoke test

Среда: Windows PowerShell 5.1.26100.9711 Desktop; Python D:\RecommendationSystem\.venv312\Scripts\python.exe, 3.12.3.

Штатный inspect_acls без ручной подмены его env вернул три структурированных fact. Значения фактов совпали с первым smoke snapshot; security policy дала:

| Путь | Инспекция Get-Acl/JSON | Результат политики |
| --- | --- | --- |
| D:\ | Успех | UNSAFE_STORAGE_ACL |
| D:\RecommendationSystem | Успех | UNSAFE_STORAGE_ACL |
| D:\RecommendationSystem\filtered_data | Успех | UNSAFE_STORAGE_ACL |

Во всех трёх имеются OTHER_PRINCIPAL Allow с технической маской 0x001301BF. Владелец проходит прежнюю политику; unsupported descriptor не обнаружен. Имена, SID и полный DACL не выводятся. Исправление helper намеренно **не устраняет** этот отказ.

Отсутствующий D:\RecommendationSystem\filtered_data\statistics_index по-прежнему даёт PRIVATE_INDEX_ROOT_REQUIRED. Каталог не создан.

Дополнительные проверки:

| Запуск | Результат |
| --- | --- |
| Обычный новый Windows PowerShell со штатным startup env → Python проекта → helper | 3 facts, три UNSAFE_STORAGE_ACL |
| Python проекта с намеренно неверным PSModulePath → helper | 3 facts, три UNSAFE_STORAGE_ACL |
| Текущий Python с ранее проблемным inherited env → helper | Успех инспекции, прежний отказ политики |
| Unicode-каталог во внешнем техническом D:\CodexTestTemp | Успех реального inspect_acls |
| Mocked environment внутри pytest | 46 новых cases в прошедшей регрессии |

В прямых/nested запусках сравнение dict(os.environ) до и после дало полное совпадение. PATH, TEMP/TMP родителя сохранены. Для терминального сценария запущен реальный Windows PowerShell subprocess; живой UI PyCharm не запускался. Совместимость с IDE проверена через релевантное загрязнённое окружение Python и mocks, а не через утверждение о проверенной интерактивной сессии.

## 5. Связанный protect_directory

D:\RecommendationSystem\Application\statistics_index.py::protect_directory оставлен без изменения. Он использует WindowsIdentity.GetCurrent, внешний icacls.exe и Out-Null, а не Get-Acl/ConvertTo-Json.

Read-only probe только WindowsIdentity и Out-Null при намеренно неверном module path прошёл. Подтверждённой аналогичной ошибки автозагрузки в этих примитивах нет. Настоящий icacls/protect_directory не вызывался: операция меняет ACL и запрещена этой задачей. Её фактическая успешность в будущей подготовке root этим тестом не подтверждена; повторно использовать helper во всех PowerShell subprocess без необходимости не стали.

## 6. Тесты и качество

Финальная целевая регрессия:

**126 passed, 1 deselected in 32.79s**, exit 0.

Запуск через D:\RecommendationSystem\.venv312\Scripts\python.exe -B -m pytest:

~~~text
-q --tb=short -p no:cacheprovider
--basetemp <external D: technical directory>/final-tests
--junitxml <external D: technical directory>/final-tests.xml
--deselect tests/test_statistics_index.py::test_private_directory_protection_on_new_directory
tests/test_statistics_storage_safety.py
tests/test_statistics_index.py
~~~

Deselect один: существующий test_private_directory_protection_on_new_directory реально устанавливает Windows ACL. Он не запускался и не считается пройденным. Остальные builds в тестах — только маленькие синтетические fixtures с mocked protection/validation; настоящих данных и production root в них нет.

46 новых cases покрывают:

- отсутствие PSModulePath и различные регистры его имени, загрязнённые и дублирующиеся ключи;
- отдельный env и неизменность parent env, сохранение несвязанных переменных, игнорирование ложного SYSTEMROOT при выборе модулей;
- встроенные manifests, отсутствие fallback, сбои API/переполнение буфера и reparse module location;
- Unicode/апостроф, protocol flags, pin до автозагрузки и неизменность проверяемого path;
- missing executable, denied access, timeout, nonzero Get-Acl/module errors без stderr;
- malformed JSON/UTF-8, shape/count, типы bool/rules/mask, signed/unsigned masks;
- сохранение результатов прежней политики на тех же facts.

Существующие storage/index-тесты дополнительно проверили reparse, unsafe parents, отсутствие root без создания, конкуренцию, отмену и fail-closed индекса на синтетике. Полный многотысячный suite не запускался.

Pytest child env: PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, QT_QPA_PLATFORM=offscreen. TEMP/TMP не менялись; cacheprovider выключен.

Ruff по изменённому production-модулю и тестам: All checks passed. Финальный git diff --check, список файлов и сохранность исторических документов: PASS: изменены только два разрешённых tracked-файла и добавлен этот отчёт; policy functions и исторические документы неизменны, production root отсутствует.

Технические результаты текущей задачи: D:\CodexTestTemp\statistics-acl-helper-8f19d6deb01841629966d8fa990ca37c. Здесь новые sanitized metadata/probes, pytest logs/XML и synthetic fixtures. Прежние диагностические каталоги и файлы не изменялись.

## 7. Ограничения и следующий шаг

Исправлена только возможность получить корректные ACL facts. Готовность storage/full build не заявляется.

Остались прежние OTHER_PRINCIPAL Modify/Delete на D:\ и унаследованные разрешения project/filtered_data. Требуется отдельная административная задача: согласовать допустимую область изменения общих ACL и выбранный namespace, сохранить защищённый backup, затем отдельно подготовить и проверить private root. Команды изменения прав в этой задаче не выполнялись.

Ни реальный пилот, ни индексация не запускались. Canonical, GUI, бизнес-алгоритмы, кеш, настройки Windows и резервная копия на C: не изменены. Исторические preflight/storage_plan сохранены побайтово.

Статус helper **PASS_HELPER_FIXED**. Storage **BLOCKED_SECURITY**. После этой задачи остановиться; административная подготовка и пилот требуют отдельных задач.
