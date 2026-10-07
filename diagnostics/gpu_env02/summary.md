# GPU-ENV-02: completed

В существующей `.venv312` заменён только PyTorch: runtime `2.5.1+cpu` → `2.5.1+cu124`. Python 3.12.3, NumPy 2.2.6 и остальные 38 packages сохранены. NVIDIA driver 581.57 и CUDA Toolkit не устанавливались и не обновлялись.

Python executable: `C:\Users\Ermolenko.i\PycharmProjects\РекомендательнаяСистема\.venv312\Scripts\python.exe`.

| Проверка | Результат |
|---|---|
| pip check до / после | No broken requirements found |
| torch.version.cuda | 12.4 |
| torch.cuda.is_available() | True |
| torch.backends.cuda.is_built() | True |
| CUDA device count | 1 |
| GPU | NVIDIA GeForce RTX 3050 |
| Compute capability | 8.6 |
| VRAM по ОС | 6144 MiB |
| VRAM по torch properties | 6441926656 bytes |
| Parent / spawn worker executable и torch build | Совпадают; оба видят CUDA |
| Tensor / matrix multiply / backward / synchronize / CPU / NumPy | Passed; output и gradient finite |
| Targeted tests | 38 passed, 2 warnings |
| tests/evaluation + tests/model | 672 passed, 106 warnings |
| Полный pytest | 2043 passed, 112 warnings, 136.38 seconds |
| CUDA final-state regression | 2 passed на настоящем CUDA backend |
| Изолированная симуляция CUDA unavailable | 18 passed, 3 skipped, 2 warnings |
| Ruff / git diff --check | Passed |
| Protected input/model/settings SHA256 | 23/23 unchanged после real-data benchmark |
| Активные training/experiment workers после benchmark | 0 |

Device selection, Application code, BPR mathematics и temporal protocol не менялись. Existing CPU tests сохранены; общий synthetic helper получил необязательный `device`, по умолчанию `cpu`. CUDA tests имеют conditional skip.

В новом cost runner краткая synthetic эпоха обнаружила нулевую длительность при Windows `time.monotonic()`. Для точного wall-clock измерения выбран `time.perf_counter()`; повторные targeted, evaluation/model и полный pytest прошли. Dependencies и training mathematics при исправлении таймера не менялись.

## Evidence files

- `environment_before.json`, `environment_after.json`: executable, Python, torch runtime/build, availability.
- `pip_freeze_before.txt`, `pip_freeze_after.txt`: полные inventories.
- `pip_show_torch_before.txt`, `pip_show_torch_after.txt`: installed package metadata.
- `pip_check_before.txt`, `pip_check_after.txt`, `pip_version_before.txt`, `pip_version_after.txt`.
- `torch_install.log`: точная force-reinstall/no-deps установка из official cu124 index.
- `package_changes.json`: изменён только torch; другие 38 packages совпадают.
- `check_cuda_environment.py`, `cuda_environment.json`: безопасный parent/spawn-worker probe и smoke test.
- `nvidia_smi_after.txt`: GPU, неизменившийся driver, VRAM.
- `targeted_tests_after.txt`, `evaluation_model_tests_after.txt`, `full_pytest_after.txt`, `cuda_unavailable_skip_tests.txt`.
- `tests_after.json`: prerequisite receipt для GPU-COST-01.
- `protected_files_check.json`: результат SHA256 проверки protected files.

Commit/push не выполнялись. Один real-data GPU cost run завершён; продолжение не запускается автоматически.
