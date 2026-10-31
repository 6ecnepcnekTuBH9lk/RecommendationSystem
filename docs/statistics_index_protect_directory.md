# STAT-INDEX-PROTECT-DIRECTORY-01 — подтверждённая защита нового каталога

31.10.2026. Проект D:\RecommendationSystem, ветка feature/hybrid-ranking.
HEAD abb9cd8e98bbcd06c9906409e50ed98ea1fd4166; все изменения локальные, Git staging/commit/push не выполнялись.

**PASS_PROTECT_DIRECTORY_FIXED.** Mutating-команда запускается напрямую, её exit проверяется, а затем фактический ACL подтверждается действующей private policy. Ранее падавший тест прошёл без изменения. Storage по-прежнему **BLOCKED_SECURITY**, административное размещение — **ADMIN_DECISION_REQUIRED**.

Реальные ACL, владельцы и системные настройки не менялись. Production index root, реальный пилот/индекс/статистика/экспорт не создавались.

## 1. Исходный дефект и воспроизведение

Старый protect_directory выполнял PowerShell script с icacls.exe через PATH и завершался exit $LASTEXITCODE. При невозможности запуска команды PowerShell мог создать ErrorRecord, не присвоив ожидаемый native exit. В свежем процессе значение могло оставаться неинициализированным; при наличии предыдущей native-команды оно могло быть устаревшим. Ненулевой результат самой icacls и ошибка её запуска — разные ситуации. Неудачная обработка параметров на стороне wrapper также не подтверждает запуск ожидаемой утилиты.

В предыдущей диагностике безопасная подстановка отсутствующей утилиты дала CommandNotFoundException и process exit=0. Здесь исходный test_protect_missing_tool_error_must_not_be_reported_as_success повторно запущен **до изменения**: 1 failed, DID NOT RAISE. Сам тест и старый внешний файл не редактировались.

Проверка только process returncode прежней оболочки не подтверждала ни выполнение icacls, ни приватность DACL. Builder дополнительно проверял storage перед SQLite/publication, поэтому найденный дефект сам по себе не доказывал публикацию небезопасного поколения.

## 2. Выбор реализации: A — прямой subprocess

Рассмотрены:

| Подход | Оценка |
| --- | --- |
| A: доверенная icacls.exe напрямую | Exit принадлежит самой утилите; argv без shell parsing; timeout применяется непосредственно к mutating process |
| B: строгий PowerShell wrapper | Возможен Stop/try-catch/explicit exit, но сохраняет quoting/слой оболочки; timeout оболочки не гарантирует завершения её дочерней native-команды |

Выбран **A**, без параллельной реализации/fallback B. PowerShell остался только в уже существующем read-only inspect_acls. Новые runtime зависимости не добавлены: используется стандартный ctypes/subprocess.

Публичная сигнатура protect_directory(path) и POSIX chmod-ветка сохранены. Успешная Windows-ветка возвращается только после проверки postconditions. Неудачная Windows-ветка выдаёт PRIVATE_INDEX_ACL_FAILED без raw stdout/stderr, SID и исходного исключения.

## 3. Доверенная утилита и текущая identity

Переиспользован существующий _powershell_environment из неизменённого statistics_storage_safety.py. Он получает Windows system directory через GetSystemDirectoryW, проверяет встроенный module path и не доверяет PATH/SystemRoot из окружения. Из его фиксированной системной структуры определяется абсолютный путь к icacls.exe; executable проверяется как файл, вся его цепочка проходит reject_reparse. Поиск через пользовательский PATH и fallback на другой executable отсутствуют.

Проверка manifests не избыточный обход: post-attestation всё равно требует исправного штатного read-only helper. При недоступности встроенных модулей функция отказывает до назначения прав.

SID процесса читается в локальном _windows_user_sid:

- OpenProcessToken с TOKEN_QUERY;
- GetTokenInformation(TokenUser), проверка размера и read-only чтение структуры;
- ConvertSidToStringSidW;
- LocalFree для строки и CloseHandle для token в finally, включая error paths.

Это SID того же process user, под которым запускается native utility и дочерняя ACL-inspection. Полный SID в отчёт/логи не выводился. Подтверждён read-only запуск WinAPI на текущей машине: результат совпал с WindowsIdentity.GetCurrent().User, сохранён только boolean сравнения.

Использованы pointer-sized HANDLE/LPVOID/LPWSTR и ctypes layout SID_AND_ATTRIBUTES. Системный путь определяется API; WOW64 redirection не отключается, Sysnative не подставляется вручную. Фактически проверен Python 3.12.3, **64-bit**, Windows PowerShell 5.1. Отдельный запуск 32-bit Python не выполнялся.

Контракты: [OpenProcessToken](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocesstoken), [GetTokenInformation](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-gettokeninformation), [ConvertSidToStringSidW](https://learn.microsoft.com/en-us/windows/win32/api/sddl/nf-sddl-convertsidtostringsidw), [Windows filesystem redirector](https://learn.microsoft.com/en-us/windows/win32/winprog64/file-system-redirector).

## 4. Запуск и fail-closed

subprocess.run получает список:

~~~text
absolute_system_icacls
absolute_target_directory
/inheritance:r
/grant:r
CURRENT_USER numeric SID grant with (OI)(CI)F
SYSTEM numeric SID grant with (OI)(CI)F
ADMINISTRATORS numeric SID grant with (OI)(CI)F
~~~

Приватные SID не логируются. /T, /C, /reset, смена владельца и рекурсивные операции не добавлены. Набор разрешений и inheritance operation сохранены. См. [Microsoft icacls](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/icacls).

Аргументы передаются отдельными элементами; shell=True не используется. Кириллица, пробелы, скобки, апостроф, dollar/semicolon/ampersand/backtick и имена, похожие на placeholders, остаются одним path argument. PowerShell quoting/expansion в mutating path отсутствуют.

capture_output=True, timeout=30. При timeout стандартный subprocess.run прекращает/дожидается непосредственно запускаемого процесса, после чего функция отказывает. Никакого автоматического retry. Отмена/timeout не означает отмену уже совершённых изменений ACL.

Следующие ошибки безопасно преобразуются в PRIVATE_INDEX_ACL_FAILED:

| Сценарий | Проверка |
| --- | --- |
| Missing directory / reparse | До native call, каталог не создаётся функцией |
| Missing/invalid executable / module support | Проверенный absolute system path, без PATH fallback |
| Недоступен user token/SID | Отказ до native call, освобождение ресурсов |
| FileNotFound/PermissionError/argument error/timeout/unexpected subprocess exception | Безопасная ошибка, native diagnostics не печатаются |
| Non-zero native exit | Немедленный отказ, без фиктивной post-attestation |
| Exit=0, но ACL небезопасен или inspection недоступна/некорректна | Отказ по фактическим private postconditions |
| Directory identity изменилась | Отказ; protect_directory ничего не удаляет |

Не анализируются локализованные строки Successfully processed как подтверждение успеха. stderr также не используется как универсальный критерий: решение основано на фактическом native exit и postconditions.

## 5. Проверка фактического ACL

Последовательность Windows-ветки:

1. reject_reparse лексического path до resolve; существующий directory обязателен.
2. Запомнить path_identity (device, inode/file identity).
3. Выполнить доверенную утилиту.
4. При exit=0 повторно проверить reparse и identity.
5. inspect_acls([target]) — актуальная локальная реализация с изолированным PSModulePath и строгой JSON validation.
6. check_acl_facts(fact, private=True, root=True).
7. Повторно подтвердить directory/reparse/identity после inspection.

Проверяются допустимый owner, protected DACL, CurrentUser FullControl, отсутствие forbidden OTHER_PRINCIPAL Allow (включая InheritOnly), Deny и unsupported ACE. Прежний security allowlist и masks не менялись. Ошибки inspection не превращаются в успех или пустые разрешения.

Проверка относится к **самой защищаемой директории**, не требует, чтобы parent был production index root. Поэтому применима и к UUID generation, и к новому private pilot area. Полный validate_storage_root в builder остаётся на прежних местах и проверяет всю цепочку родителей/NTFS/children.

Identity guard не заменяет доверенную parent namespace и не является обещанием абсолютной защиты против произвольного hostile rename во время path-based операции. Caller должен соблюдать контракт новых owned-каталогов и безопасных родителей. Текущий D: namespace не объявлен готовым к setup.

## 6. Частичное применение, index cleanup и публикация

Неуспешная icacls может частично изменить ACL. Функция не делает вывод «ничего не изменилось», не пытается автоматически восстановить/ослабить права и не удаляет target. Она возвращает безопасный отказ; управление остаётся у владельца staging lifecycle.

Builder сохраняет прежнюю последовательность: собственный UUID → protection → storage validation → SQLite → integrity/provenance → atomic publication. При failure protection SQLite не открывается, current.json не меняется, старые поколения сохраняются, новый UUID не публикуется.

Cleanup builder по-прежнему использует check_owned_staging с root/folder identity и reparse checks. При подмене он отказывает вместо удаления чужого staging. Тесты подтвердили также partial-attempt marker и отсутствие повторной попытки.

Если cleanup не выполнен из-за PermissionError/другого отказа, ошибка **не подавляется**, сборка не объявляется успешной, staging может остаться для отдельной проверки. Автоматическое удаление/repair/retry не добавлялись. Protection стоит до записи исходных клиентских фактов, поэтому ACL-failure в этом месте не запускает копирование/проекцию таких данных. Для более поздних ошибок уже приватное staging может содержать данные; общая гарантия обязательной очистки не выдаётся.

## 7. Pilot и отдельно одобренный identity guard

prepare_pilot вызывает protect_directory сразу после создания destination, до copy/source projection/marker. При protection failure нет готового PILOT_ONLY.json и автоматического повторного prepare/build.

В исходном pilot cleanup проверял symlink/resolve, но не file identity; обычная directory-подмена могла привести к удалению чужого target. Для обязательного инварианта пользователь отдельно одобрил точечное изменение Application/statistics_pilot.py.

Теперь фиксируются identity parent/destination, а перед cleanup выполняются reject_reparse и сверка обеих identity. При изменении/недоступности — PILOT_CLEANUP_PATH_CHANGED, без rmtree замены. В нормальном failure очищается только созданный destination; вложенная структура синтетического пилота и PilotLimits не изменены.

Тесты подтвердили:

- остановку до copy после ACL error;
- отсутствие PILOT_ONLY.json;
- сохранение постороннего sibling file;
- сохранение replacement directory и исходного retained-owned directory;
- отсутствие ready marker и подавления ошибки при отказе cleanup.

Реальный пилот не подготавливался. Все source fixtures — маленькие синтетические данные pytest.

## 8. Регрессионные тесты

**197 passed, 1 deselected in 38.07s**, exit 0.

Запущены tests/test_statistics_index.py, tests/test_statistics_storage_safety.py, tests/test_statistics_pilot.py и неизменённый прежний внешний test_protect_missing_tool_error_must_not_be_reported_as_success.

До исправления внешний regression давал DID NOT RAISE; после — проходит. Он использует отсутствующий target и теперь также защищён ранней проверкой существования. Чтобы не ограничиваться этим, добавлены отдельные случаи с **существующим target**, вызванным mock subprocess и неверным ACL при ложном exit=0.

Покрытие новых случаев:

- абсолютная system utility, отсутствие shell/PATH fallback, точный argv с необычными Windows names;
- missing executable, permission/argument/timeout/unexpected errors, SID lookup failures;
- невалидные ACL/owner/protection/CurrentUser permissions/OTHER Allow/InheritOnly/Deny/unsupported;
- unavailable/invalid inspection, reparse до запуска и подмена directory при execution/inspection;
- WinAPI token failures и освобождение HANDLE/allocated SID string;
- запрет SQLite/publication при ACL error; old generation/current.json сохранены;
- безопасный cleanup при path substitution и явный отказ cleanup;
- синтетический pilot failure/identity substitution без marker/удаления чужих файлов;
- неизменённая регрессия inspect_acls/PSModulePath/fail-closed storage policy.

Единственный deselected: tests/test_statistics_index.py::test_private_directory_protection_on_new_directory. Он действительно назначает ACL и запрещён текущей задачей. Он не выдаётся за пройденный.

Промежуточные запуски: до исправления 1 failed (исходный дефект); целевые synthetic batches 39 passed и 38 passed на последовательных вариантах реализации. Финальное решение подтверждено указанным полным целевым запуском 197 passed. Новых непочиненных падений нет.

Python: D:\RecommendationSystem\.venv312\Scripts\python.exe -B -m pytest, -q --tb=short -p no:cacheprovider. PYTHONDONTWRITEBYTECODE=1, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, QT_QPA_PLATFORM=offscreen; TEMP/TMP не менялись. basetemp/logs/XML находятся вне проекта на D:.

Артефакты этой задачи: D:\CodexTestTemp\statistics-protect-2385db48c7fb491fae8212003563094e. Старый failing-test file и прежняя диагностика не изменялись.

Ruff и git diff --check: PASS: Ruff для 6 релевантных source/test-файлов — All checks passed; git diff --check и whitespace нового отчёта чисты; scope и сохранность предыдущих изменений подтверждены.

## 9. Изменённые файлы и сохранность

| Файл текущей задачи | Назначение |
| --- | --- |
| D:\RecommendationSystem\Application\statistics_index.py | Direct native protection, token SID helper, actual ACL/identity attestation |
| D:\RecommendationSystem\Application\statistics_pilot.py | Отдельно одобренный guard parent/destination identity перед cleanup |
| D:\RecommendationSystem\tests\test_statistics_index.py | Mock-native, policy/postcondition/resource/index lifecycle regression |
| D:\RecommendationSystem\tests\test_statistics_pilot.py | Synthetic pilot protection/cleanup regression |
| D:\RecommendationSystem\docs\statistics_index_protect_directory.md | Этот отчёт |

statistics_storage_safety.py и его ранее изменённые tests сохранены побайтово; рабочий PSModulePath fix не переписывался. Четыре исторических отчёта не менялись. Расчётные алгоритмы, schema version, canonical, GUI, кеш, модели, диагностика и резервная копия проекта C: не менялись. Предшествующий локальный Git diff сохраняется; в этой задаче commit/staging/push/reset/clean/restore не выполнялись.

## 10. Что не проверено и что остаётся заблокированным

Реальный icacls /grant или /inheritance, Set-Acl и owner changes не выполнялись даже на временных каталогах. Поэтому фактическое назначение DACL на NTFS новой реализацией остаётся отдельным будущим разрешённым smoke test в специально подготовленном test namespace. Не проверены реальная частичная Windows ACL mutation и runtime 32-bit Python; ошибки моделировались.

Read-only проверка Get-Acl текущей машины прошла. D:\, D:\RecommendationSystem и D:\RecommendationSystem\filtered_data продолжают возвращать UNSAFE_STORAGE_ACL. Production root отсутствует. Список trusted principals и решение политики не ослаблены.

Дальнейший шаг: отдельное административное решение по безопасному размещению INDEXED и влиянию parent ACL, затем отдельно разрешённая подготовка тестового/private namespace и фактический smoke test назначения. Пилот/expected_gib/full build рассматриваются только после storage admission.

Программный статус: **PASS_PROTECT_DIRECTORY_FIXED**. Инфраструктурный статус: **BLOCKED_SECURITY / ADMIN_DECISION_REQUIRED**. После задачи остановиться; ACL setup и индексация не запускаются автоматически.
