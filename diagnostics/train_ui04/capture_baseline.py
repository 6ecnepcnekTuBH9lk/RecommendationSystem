"""Save the current source baseline and prepare a synthetic before-UI probe."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
names = {p.relative_to(ROOT).as_posix() for directory in ('Application', 'scripts', 'tests')
         for p in (ROOT / directory).rglob('*.py')}
names.update(('main.py', 'README.md', 'requirements.txt', 'requirements-dev.txt',
              'assets/icons/start_training.png', 'assets/icons/failure.png'))
hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
(OUTPUT / 'before_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
for name in ('Application/tabs/train_model_tab.py', 'Application/training_charts.py',
             'Application/statistics_charts.py', 'tests/tabs/test_training_research_ui.py'):
    target = OUTPUT / 'before_sources' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)
shutil.copyfile(ROOT / 'diagnostics/train_ui03/protected_baseline.json', OUTPUT / 'protected_baseline.json')
(OUTPUT / 'before').mkdir(exist_ok=True)
source = (ROOT / 'diagnostics/train_ui03/preview.py').read_text(encoding='utf-8')
probe = source.replace("'diagnostics/train_ui03'", "'diagnostics/train_ui04/before'")
probe = probe.replace('    app.processEvents()\n', '''    app.processEvents()
    from PyQt6.QtWidgets import QStyle, QStyleOptionButton
    def rectangle(rect):
        return [rect.x(), rect.y(), rect.width(), rect.height()]
    buttons = {}
    for key, button in (('start', window.start_train), ('cancel', window.cancel_train)):
        option = QStyleOptionButton()
        button.initStyleOption(option)
        buttons[key] = {'geometry': rectangle(button.geometry()),
                        'size_hint': [button.sizeHint().width(), button.sizeHint().height()],
                        'margins': [button.contentsMargins().top(), button.contentsMargins().bottom()],
                        'contents': rectangle(button.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, button)),
                        'enabled': button.isEnabled()}
    charts = {}
    for key, chart in (('loss', window.training_loss_chart), ('metrics', window.training_metric_chart)):
        charts[key] = {'plot': rectangle(chart.chart().plotArea()),
                       'placeholder': rectangle(chart.placeholder.geometry()),
                       'viewport': rectangle(chart.viewport().rect()),
                       'tick_interval': chart.x_axis.tickInterval()}
    (ROOT / 'diagnostics/train_ui04/before' / (name + '.json')).write_text(
        json.dumps({'buttons': buttons, 'charts': charts}, indent=2), encoding='utf-8')
''')
(OUTPUT / 'before_preview.py').write_text(probe, encoding='utf-8')
print(f'Saved {len(hashes)} source/contract hashes and the synthetic probe')
