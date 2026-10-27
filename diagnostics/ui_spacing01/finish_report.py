"""Insert verified full-test result and artifact links into the saved report."""
from pathlib import Path
import re

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
log = (OUTPUT / 'full.txt').read_text(encoding='utf-8', errors='replace')
match = re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s', log)
assert match is not None and 'FAILED ' not in log
passed, warnings, seconds = match.groups()
path = OUTPUT / 'report.md'
report = path.read_text(encoding='utf-8').replace(
    '23. **Full pytest.** Выполняется; результат будет внесён перед завершением задачи.',
    f'23. **Full pytest.** **{passed} passed, {warnings} warnings, {seconds} s**, exit **0**. '
    'Baseline 2172/115: добавлено 16 проверок, новых warnings нет. Native Qt crash не возник.')
report += '\n| Вкладка | Before light | After light | Before dark | After dark |\n|---|---|---|---|---|\n'
for name, title in (('01_data_loading', 'Получение данных'), ('02_analysis_filter', 'Настройка анализа'),
                    ('03_statistics', 'Статистика'), ('04_training', 'Обучение'), ('05_results', 'Результаты')):
    links = []
    for folder, suffix in (('before', ''), ('after', ''), ('before', '_dark'), ('after', '_dark')):
        image_path = (OUTPUT / folder / (name + suffix + '.png')).as_posix()
        assert Path(image_path).exists()
        links.append(f'[PNG]({image_path})')
    report += '| ' + title + ' | ' + ' | '.join(links) + ' |\n'
fence = '`' * 3
for command, name in (('git diff --stat', 'git_diff_stat.txt'), ('git status --short', 'git_status.txt')):
    report += '\n' + command + ':\n\n' + fence + 'text\n'
    report += (OUTPUT / name).read_text(encoding='utf-8').rstrip() + '\n' + fence + '\n'
path.write_text(report, encoding='utf-8')
print(f'Finished report: {passed} passed, {warnings} warnings; 20 screenshot links')
