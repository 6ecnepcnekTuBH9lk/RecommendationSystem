"""Read-only source/data/model integrity check against recorded SHA-256 baselines."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


before = {Path(k).as_posix(): v for k, v in json.loads((OUTPUT / 'before_hashes.json').read_text(encoding='utf-8')).items()}
previous = json.loads((OUTPUT / 'integrity.json').read_text(encoding='utf-8')) if (OUTPUT / 'integrity.json').exists() else {}
changed = [name for name, expected in before.items() if sha(ROOT / name) != expected]
protected = json.loads((OUTPUT / 'protected_baseline.json').read_text(encoding='utf-8'))
checks = {name: {'expected_sha256': expected, 'actual_sha256': sha(ROOT / name)} for name, expected in protected.items()}
result = {'existing_sources_changed_for_train_ui01': changed,
          'protected_count': len(checks), 'protected_unchanged': all(v['expected_sha256'] == v['actual_sha256'] for v in checks.values()),
          'icon_unchanged': sha(ROOT / 'assets/icons/start_training.png') == before['assets/icons/start_training.png'],
          'bpr_math_unchanged': sha(ROOT / 'Application/model/BPRMF.py') == before['Application/model/BPRMF.py'],
          'checked_at': datetime.now(timezone.utc).isoformat(),
          'checks': checks}
text = (ROOT / 'Application/tabs/train_model_tab.py').read_text(encoding='utf-8')
expected_helpers = previous.get('legacy_helpers_before_sha256')
result['legacy_helpers_before_sha256'] = expected_helpers
result['legacy_helpers_unchanged'] = expected_helpers == hashlib.sha256(text[text.index('def _get_store_city_map'):].encode('utf-8')).hexdigest()
(OUTPUT / 'integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print({k: v for k, v in result.items() if k != 'checks'})
