"""Read-only hashes and exact TRAIN-UI-04 source diff against the saved UI-03."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


before = json.loads((OUTPUT / 'before_hashes.json').read_text(encoding='utf-8'))
changed = [name for name, expected in before.items() if sha(ROOT / name) != expected]
expected_changes = {'Application/tabs/train_model_tab.py', 'Application/training_charts.py',
                    'Application/statistics_charts.py', 'tests/tabs/test_training_research_ui.py'}
protected = json.loads((OUTPUT / 'protected_baseline.json').read_text(encoding='utf-8'))
checks = {name: sha(ROOT / name) == expected for name, expected in protected.items()}
source_names = {p.relative_to(ROOT).as_posix() for directory in ('Application', 'scripts', 'tests')
                for p in (ROOT / directory).rglob('*.py')}
added = sorted(source_names - before.keys())
old_tab = (OUTPUT / 'before_sources/Application/tabs/train_model_tab.py').read_text(encoding='utf-8')
new_tab = (ROOT / 'Application/tabs/train_model_tab.py').read_text(encoding='utf-8')
model_files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'model').rglob('*') if p.is_file())
result = {'baseline_files': len(before), 'changed_for_train_ui04': changed,
          'added_source_files': added, 'protected_count': len(checks),
          'protected_unchanged': all(checks.values()), 'checks': checks,
          'all_other_baseline_files_unchanged': set(changed) == expected_changes,
          'workflow_body_unchanged': old_tab[old_tab.index('def _research'):] == new_tab[new_tab.index('def _research'):],
          'production_model_files': model_files,
          'production_current_exists': (ROOT / 'model/current.json').exists()}
(OUTPUT / 'integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
diff, stat, total_added, total_removed = [], [], 0, 0
for name in changed:
    saved = OUTPUT / 'before_sources' / name
    lines = list(difflib.unified_diff(saved.read_text(encoding='utf-8').splitlines(True),
                                    (ROOT / name).read_text(encoding='utf-8').splitlines(True),
                                    fromfile='TRAIN-UI-03/' + name, tofile='TRAIN-UI-04/' + name))
    added_lines = sum(line.startswith('+') and not line.startswith('+++') for line in lines)
    removed_lines = sum(line.startswith('-') and not line.startswith('---') for line in lines)
    total_added += added_lines
    total_removed += removed_lines
    stat.append(f'{name}: +{added_lines} / -{removed_lines}')
    diff.extend(lines)
stat.append(f'{len(changed)} files: +{total_added} / -{total_removed}')
(OUTPUT / 'task.diff').write_text(''.join(diff), encoding='utf-8')
(OUTPUT / 'task_stat.txt').write_text('\n'.join(stat) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'checks'}, ensure_ascii=False, indent=2))
print('\n'.join(stat))
assert all(checks.values()) and set(changed) == expected_changes and not added
assert result['workflow_body_unchanged'] and not model_files and not result['production_current_exists']
