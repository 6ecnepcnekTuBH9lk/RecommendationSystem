"""Read-only TRAIN-UI-03 integrity and source diff against the saved UI-02 baseline."""
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
source_names = {p.relative_to(ROOT).as_posix() for directory in ('Application', 'scripts', 'tests')
                for p in (ROOT / directory).rglob('*.py')}
added = sorted(source_names - before.keys())
protected = json.loads((OUTPUT / 'protected_baseline.json').read_text(encoding='utf-8'))
checks = {name: sha(ROOT / name) == expected for name, expected in protected.items()}
old_tab = (OUTPUT / 'before_sources/Application/tabs/train_model_tab.py').read_text(encoding='utf-8')
new_tab = (ROOT / 'Application/tabs/train_model_tab.py').read_text(encoding='utf-8')
unchanged_contracts = ('Application/model/BPRMF.py',
                       'Application/tabs/training_workflow.py',
                       'Application/evaluation/experiments/gui_history.py',
                       'Application/model/mindbox_production_training.py',
                       'scripts/mindbox_production_train.py',
                       'assets/icons/start_training.png', 'assets/icons/failure.png',
                       'requirements.txt', 'requirements-dev.txt')
contracts = {name: sha(ROOT / name) == before[name] for name in unchanged_contracts}
model_files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / 'model').rglob('*') if p.is_file())
result = {'baseline_files': len(before), 'changed_for_train_ui03': changed,
          'added_for_train_ui03': added, 'protected_count': len(checks),
          'protected_unchanged': all(checks.values()), 'checks': checks,
          'contracts_unchanged': contracts,
          'legacy_helpers_unchanged': old_tab[old_tab.index('def _get_store_city_map'):] == new_tab[new_tab.index('def _get_store_city_map'):],
          'production_model_files': model_files,
          'production_current_exists': (ROOT / 'model/current.json').exists()}
(OUTPUT / 'integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
diff, stat, total_added, total_removed = [], [], 0, 0
for name in changed + added:
    saved = OUTPUT / 'before_sources' / name
    old = saved.read_text(encoding='utf-8').splitlines(True) if saved.exists() else []
    assert saved.exists() or name in added, f'Missing before source: {name}'
    new = (ROOT / name).read_text(encoding='utf-8').splitlines(True)
    lines = list(difflib.unified_diff(old, new, fromfile='TRAIN-UI-02/' + name,
                                    tofile='TRAIN-UI-03/' + name))
    added_lines = sum(line.startswith('+') and not line.startswith('+++') for line in lines)
    removed_lines = sum(line.startswith('-') and not line.startswith('---') for line in lines)
    total_added += added_lines
    total_removed += removed_lines
    stat.append(f'{name}: +{added_lines} / -{removed_lines}')
    diff.extend(lines)
stat.append(f'{len(changed) + len(added)} files: +{total_added} / -{total_removed}')
(OUTPUT / 'task.diff').write_text(''.join(diff), encoding='utf-8')
(OUTPUT / 'task_stat.txt').write_text('\n'.join(stat) + '\n', encoding='utf-8')
print(json.dumps({key: value for key, value in result.items() if key != 'checks'}, ensure_ascii=False, indent=2))
print('\n'.join(stat))
assert all(checks.values()) and all(contracts.values())
assert result['legacy_helpers_unchanged'] and not model_files and not result['production_current_exists']
assert added == ['Application/training_charts.py']
assert set(changed) == {'Application/tabs/train_model_tab.py', 'Application/statistics_charts.py',
                        'Application/evaluation/experiments/gui_run.py',
                        'tests/tabs/test_training_research_ui.py', 'tests/evaluation/test_gui_run.py', 'README.md'}
