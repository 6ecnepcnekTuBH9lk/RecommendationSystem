"""Direct-category numbers and optional saved paths; synthetic inputs only."""

from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from Application import dataset_statistics as statistics
from Application import statistics_cache as cache
from Application.product_resolution import CatalogError
from Application.statistics_presentation import product_category_rows, LEGACY_CATEGORY_PATH, UNAVAILABLE_CATEGORY_PATH
from test_dataset_statistics import dataset, calculate, action, order, write_actions, write_orders, DEFAULT_SELECTION

__all__ = ['dataset']


def reference(dataset):
    root, catalog, _ = dataset
    catalog.write_text('КодНоменклатуры|НазваниеНаСайте|КатегорияНаСайте\n'
                       '000001|A|000123\n000002|B|000123\n000003|C|unknown\n000004|D|\n'
                       '000001|A last|000124\n', encoding='utf-8-sig')
    (root / 'site_categories.csv').write_text(
        'КодКатегории|НазваниеКатегории|КодРодительскойКатегории\n'
        '000123|Рубашки|r1\n000124|Рубашки|r2\nr1|Первый раздел|\nr2|Второй раздел|\n', encoding='utf-8-sig')
    write_actions(dataset, [action(DEFAULT_SELECTION.view_action_system_names[0], [f'{i:06}-a']) for i in range(1, 5)])
    raw = order(['CP', 'CP'])
    raw['lines'][1]['product']['ids']['offline1C'] = '000002-a'
    write_orders(dataset, [raw])


def test_same_named_categories_last_product_row_and_unchanged_additive_totals(dataset, tmp_path):
    reference(dataset)
    result = asdict(calculate(dataset))
    cache.validate_result(result)
    categories = {code: (name, *values) for code, name, *values in result['product_category_statistics']}
    assert categories == {'000123': ('Рубашки', 1, 1, 0, 1, '2'),
                          '000124': ('Рубашки', 1, 1, 0, 1, '2'),
                          'unknown': ('unknown', 1, 1, 0, 0, '0'),
                          None: ('Не указано', 1, 1, 0, 0, '0')}
    assert sum(row[2] for row in result['product_category_statistics']) == result['unique_resolved_items'] == 4
    paths = dict(result['product_category_paths'])
    assert paths == {'000123': ('Первый раздел', 'Рубашки'), '000124': ('Второй раздел', 'Рубашки'), 'unknown': None}
    assert not {'r1', 'r2'} & categories.keys()  # Parents do not receive any child counts.
    rows = product_category_rows(result)
    assert rows[0][:3] == ('000123', 'Рубашки', 'Первый раздел / Рубашки')
    assert all(tuple(row[3:]) == tuple(original[2:]) for row, original in zip(rows, result['product_category_statistics']))
    original = deepcopy(result)
    for name, payload in [('new', result), ('legacy', {k: v for k, v in result.items() if k != 'product_category_paths'})]:
        path = tmp_path / (name + '.json')
        cache.save_result(path, payload)
        assert json.loads(path.read_text(encoding='utf-8'))['schema_version'] == 10
        loaded = cache.load_result(path)
        assert loaded == json.loads(json.dumps(payload))
        assert loaded['product_category_statistics'] == json.loads(json.dumps(result['product_category_statistics']))
        if name == 'legacy':
            assert 'product_category_paths' not in loaded
            assert all(row[2] == LEGACY_CATEGORY_PATH for row in product_category_rows(loaded) if row[0] != 'Не указан')
    assert result == original


def test_paths_are_ordered_by_codes_and_invalid_ancestry_never_looks_complete(tmp_path):
    path = tmp_path / 'categories.csv'
    header = 'КодКатегории|НазваниеКатегории|КодРодительскойКатегории\n'
    rows = ['root|Одинаковое|', 'middle|Одинаковое|root', '000001|Лист|middle',
            'orphan|Лист|absent', 'child|Потомок|orphan', 'cycle1|A|cycle2', 'cycle2|B|cycle1',
            'self|S|self', 'ambiguous|Лист|root', 'ambiguous|Лист|middle', 'below|Потомок|ambiguous',
            'unnamed||root', 'under-unnamed|Лист|unnamed', '000001|Лист|middle']
    expected = None
    for values in (rows, list(reversed(rows))):
        path.write_text(header + '\n'.join(values) + '\n', encoding='utf-8-sig')
        names, paths = statistics._category_reference(path)
        assert paths['000001'] == ('Одинаковое', 'Одинаковое', 'Лист')
        assert paths['root'] == ('Одинаковое',)
        assert all(paths[code] is None for code in ('orphan', 'child', 'cycle1', 'cycle2', 'self',
                                                    'ambiguous', 'below', 'unnamed', 'under-unnamed'))
        assert names['unnamed'] == 'unnamed'  # Existing name fallback remains intact.
        if expected is not None:
            assert (names, paths) == expected
        expected = names, paths
    path.write_text(header + 'a|First|\na|Second|\n', encoding='utf-8')
    with pytest.raises(CatalogError, match='Конфликт названий'):
        statistics._category_reference(path)


def test_missing_parent_column_is_unknown_not_an_invented_root_and_deep_paths_do_not_recurse(tmp_path):
    path = tmp_path / 'categories.csv'
    path.write_text('КодКатегории|НазваниеКатегории\n001|A\n', encoding='utf-8')
    assert dict(statistics._category_reference(path)[1]) == {'001': None}
    path.write_text('КодКатегории|НазваниеКатегории|КодРодительскойКатегории\n' +
                    '\n'.join(f'{i:04}|N{i}|{i-1:04}' if i else '0000|N0|' for i in range(1100)), encoding='utf-8')
    assert statistics._category_reference(path)[1]['1099'] == tuple(f'N{i}' for i in range(1100))


def test_saved_paths_do_not_follow_later_reference_changes(dataset):
    reference(dataset)
    first = asdict(calculate(dataset))
    original = deepcopy(first)
    path = dataset[0] / 'site_categories.csv'
    path.write_text(path.read_text(encoding='utf-8-sig').replace('Первый раздел', 'Новое название'), encoding='utf-8-sig')
    second = asdict(calculate(dataset))
    assert first['product_category_statistics'] == second['product_category_statistics']
    assert dict(first['product_category_paths'])['000123'] == ('Первый раздел', 'Рубашки')
    assert dict(second['product_category_paths'])['000123'] == ('Новое название', 'Рубашки')
    assert first == original


@pytest.mark.parametrize('damage', ['dict', 'missing', 'extra', 'duplicate', 'wrong_leaf', 'empty', 'blank', 'scalar', 'bad_code'])
def test_optional_paths_are_validated_without_weakening_existing_contract(dataset, damage):
    reference(dataset)
    result = json.loads(json.dumps(asdict(calculate(dataset))))
    paths = result['product_category_paths']
    if damage == 'dict':
        result['product_category_paths'] = dict(paths)
    elif damage == 'missing':
        paths.pop()
    elif damage == 'extra':
        paths.append(['extra', None])
    elif damage == 'duplicate':
        paths.append(paths[0])
    elif damage == 'bad_code':
        paths[0][0] = None
    else:
        paths[0][1] = {'wrong_leaf': ['Wrong'], 'empty': [], 'blank': [' ', 'Рубашки'], 'scalar': 'Root → Рубашки'}[damage]
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_unknown_and_missing_category_never_invent_a_path():
    result = {'product_category_statistics': [(None, 'Не указано', 1, 1, 0, 0, '0'), ('unknown', 'unknown', 1, 1, 0, 0, '0')],
              'product_category_paths': [('unknown', None)]}
    rows = product_category_rows(result)
    assert rows[0][:3] == ('Не указан', 'Не указано', 'Категория не указана')
    assert rows[1][:3] == ('unknown', 'unknown', UNAVAILABLE_CATEGORY_PATH)
