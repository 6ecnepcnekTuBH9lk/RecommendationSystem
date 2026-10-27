"""Read-only integrity and exact TRAIN-UI-02 source diff against saved baseline."""
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
protected = json.loads((OUTPUT / 'protected_baseline.json').read_text(encoding='utf-8'))
checks = {name: sha(ROOT / name) == expected for name, expected in protected.items()}
old_tab = (OUTPUT / 'before_sources/Application/tabs/train_model_tab.py').read_text(encoding='utf-8')
new_tab = (ROOT / 'Application/tabs/train_model_tab.py').read_text(encoding='utf-8')
result = {'changed_for_train_ui02': changed, 'protected_count': len(checks),
          'protected_unchanged': all(checks.values()), 'checks': checks,
          'bpr_math_unchanged': sha(ROOT / 'Application/model/BPRMF.py') == before['Application/model/BPRMF.py'],
          'history_schema_unchanged': sha(ROOT / 'Application/evaluation/experiments/gui_history.py') == before['Application/evaluation/experiments/gui_history.py'],
          'legacy_helpers_unchanged': old_tab[old_tab.index('def _get_store_city_map'):] == new_tab[new_tab.index('def _get_store_city_map'):],
          'icon_unchanged': sha(ROOT / 'assets/icons/start_training.png') == before['assets/icons/start_training.png'],
          'production_current_exists': (ROOT / 'model/current.json').exists()}
(OUTPUT / 'integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
diff = []
for name in changed:
    saved = OUTPUT / 'before_sources' / name
    if saved.exists():
        diff.extend(difflib.unified_diff(saved.read_text(encoding='utf-8').splitlines(True),
                                       (ROOT / name).read_text(encoding='utf-8').splitlines(True),
                                       fromfile='TRAIN-UI-01/' + name, tofile='TRAIN-UI-02/' + name))
(OUTPUT / 'task.diff').write_text(''.join(diff), encoding='utf-8')
print({key: value for key, value in result.items() if key != 'checks'})
assert all(checks.values()) and result['bpr_math_unchanged'] and result['history_schema_unchanged']
assert result['legacy_helpers_unchanged'] and result['icon_unchanged']
