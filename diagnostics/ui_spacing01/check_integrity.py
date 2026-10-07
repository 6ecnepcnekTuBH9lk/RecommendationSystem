"""Read-only scope, protected data and unchanged non-presentation method checks."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def functions(source):
    result = {}
    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                if not isinstance(node, ast.ClassDef):
                    result[name] = node
                visit(node.body, name + '.')
    visit(ast.parse(source).body)
    return result


before = json.loads((OUTPUT / 'before_hashes.json').read_text(encoding='utf-8'))
changed = [name for name, expected in before.items() if sha(ROOT / name) != expected]
added = sorted(p.relative_to(ROOT).as_posix() for folder in ('Application', 'scripts', 'tests')
               for p in (ROOT / folder).rglob('*.py') if p.relative_to(ROOT).as_posix() not in before)
protected = json.loads((OUTPUT / 'protected_baseline.json').read_text(encoding='utf-8'))
data_checks = {name: sha(ROOT / name) == expected for name, expected in protected.items()}
excluded = {
    'Application/tabs/data_loading_tab.py': ('create_data_loading_widgets_tab',),
    'Application/tabs/reference_loading_section.py': ('create_csv_loading_section', 'update_file_status'),
    'Application/tabs/analysis_filter_tab.py': ('MultiSelect.__init__', 'AnalysisFilterTab.__init__', 'AnalysisFilterTab._size_inputs'),
    'Application/tabs/dataset_statistics_tab.py': ('_section_heading', '_cards', '_actions_page', '_products_page', '_orders_page', 'DatasetStatisticsTab.__init__', 'DatasetStatisticsTab._pages', 'DatasetStatisticsTab.render'),
    'Application/tabs/train_model_tab.py': ('create_train_model_widgets_tab',),
    'Application/tabs/create_results_tab.py': ('create_result_widgets_tab',),
    'main.py': ('MainWindow.__init__', 'MainWindow.apply_static_widget_styles'),
}
method_checks = {}
for name, omit in excluded.items():
    old = functions((OUTPUT / 'before_sources' / name).read_text(encoding='utf-8'))
    new = functions((ROOT / name).read_text(encoding='utf-8'))
    for key, node in old.items():
        if any(key == prefix or key.startswith(prefix + '.') for prefix in omit):
            continue
        method_checks[name + ':' + key] = key in new and ast.dump(node) == ast.dump(new[key])
old_analysis = functions((OUTPUT / 'before_sources/Application/tabs/analysis_filter_tab.py').read_text(encoding='utf-8'))
old_sizing = old_analysis['AnalysisFilterTab._size_inputs']
for node in ast.walk(old_sizing):
    if isinstance(node, ast.Name) and node.id == 'BLOCK_SPACING':
        node.id = 'SECTION_SPACING'
new_sizing = functions((ROOT / 'Application/tabs/analysis_filter_tab.py').read_text(encoding='utf-8'))['AnalysisFilterTab._size_inputs']
method_checks['analysis_adaptive_math_preserved'] = ast.dump(old_sizing) == ast.dump(new_sizing)
old_qss = (OUTPUT / 'before_sources/Application/theme/custom.qss').read_text(encoding='utf-8')
new_qss = (ROOT / 'Application/theme/custom.qss').read_text(encoding='utf-8')
button = r'QPushButton \{\{.*?\}\}'
qss_heights_preserved = re.search(button, old_qss, re.S).group() == re.search(button, new_qss, re.S).group()
diff, stat, plus, minus = [], [], 0, 0
for name in changed + added:
    saved = OUTPUT / 'before_sources' / name
    old = saved.read_text(encoding='utf-8').splitlines(True) if saved.exists() else []
    lines = list(difflib.unified_diff(old, (ROOT / name).read_text(encoding='utf-8').splitlines(True),
                                    fromfile='before/' + name, tofile='after/' + name))
    inserted = sum(line.startswith('+') and not line.startswith('+++') for line in lines)
    deleted = sum(line.startswith('-') and not line.startswith('---') for line in lines)
    plus += inserted
    minus += deleted
    stat.append(f'{name}: +{inserted} / -{deleted}')
    diff.extend(lines)
stat.append(f'{len(changed) + len(added)} files: +{plus} / -{minus}')
(OUTPUT / 'task.diff').write_text(''.join(diff), encoding='utf-8')
(OUTPUT / 'task_stat.txt').write_text('\n'.join(stat) + '\n', encoding='utf-8')
result = {'baseline_count': len(before), 'changed': changed, 'added': added,
          'protected_count': len(data_checks), 'protected_unchanged': all(data_checks.values()),
          'non_presentation_method_count': len(method_checks), 'methods_unchanged': all(method_checks.values()),
          'button_qss_unchanged': qss_heights_preserved,
          'training_chart_queue_fix_unchanged': sha(ROOT / 'Application/training_charts.py') == before['Application/training_charts.py'],
          'data_checks': data_checks, 'method_checks': method_checks}
(OUTPUT / 'integrity.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('data_checks', 'method_checks')}, ensure_ascii=False, indent=2))
print('\n'.join(stat))
assert all(data_checks.values()) and all(method_checks.values()) and qss_heights_preserved
assert result['training_chart_queue_fix_unchanged']
assert added == ['Application/theme/layout_metrics.py', 'tests/tabs/test_ui_spacing.py']
