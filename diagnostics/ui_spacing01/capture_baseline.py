"""Save source and protected-file hashes before any UI spacing edits."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


names = {p.relative_to(ROOT).as_posix() for folder in ('Application', 'scripts', 'tests')
         for p in (ROOT / folder).rglob('*.py')}
names.update(('main.py', 'README.md', 'Application/theme/custom.qss', 'requirements.txt', 'requirements-dev.txt',
              'assets/icons/start_training.png', 'assets/icons/failure.png'))
(OUTPUT / 'before_sources').mkdir(parents=True, exist_ok=True)
for name in names:
    target = OUTPUT / 'before_sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)
(OUTPUT / 'before_hashes.json').write_text(json.dumps({name: sha(ROOT / name) for name in sorted(names)}, indent=2), encoding='utf-8')
protected_names = set(json.loads((ROOT / 'diagnostics/train_ui04/protected_baseline.json').read_text(encoding='utf-8')))
protected_names.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'model').rglob('*') if p.is_file())
(OUTPUT / 'protected_baseline.json').write_text(json.dumps({name: sha(ROOT / name) for name in sorted(protected_names)}, indent=2), encoding='utf-8')
print(f'Saved {len(names)} source hashes and {len(protected_names)} protected-file hashes')
