"""Safe streaming location metadata, with no source values in diagnostics."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from Application.loading_errors import error_record, format_error
from Application.mindbox import canonical_storage as store, manual_diagnostics as diagnostics
from Application.mindbox.adapters._common import AdapterError
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.manual_sources import ManualImportError
from Application.mindbox.selection import DEFAULT_SELECTION


def order():
    return {'ids': {'mindboxId': 'PRIVATE_ORDER'}, 'customer': {'ids': {'mindboxId': 'PRIVATE_CUSTOMER'}},
            'firstAction': {'dateTimeUtc': '2022-01-01T12:00:00Z',
                            'channel': {'ids': {'externalId': 'PRIVATE_STORE'}, 'name': 'PRIVATE_NAME'}},
            'lines': [{'number': 1, 'quantity': 2, 'priceOfLine': 50, 'status': {'ids': {'externalId': 'CP'}},
                       'product': {'ids': {'offline1C': 'PRIVATE_PRODUCT'}}}]}


@pytest.mark.parametrize('file_index', [1, 2])
@pytest.mark.parametrize('field,value,category', [('quantity', None, 'missing'), ('quantity', 'PRIVATE_EMAIL', 'number'),
                                                ('product', {'ids': {}}, 'namespace')])
def test_order_failure_identifies_selected_file_record_and_raw_line_position(tmp_path, file_index, field, value, category):
    raw = order()
    raw['lines'].append({**raw['lines'][0], field: value, 'number': 99})
    for number in range(1, 3):
        (tmp_path / f'orders_part_{number:03}.json').write_text(json.dumps({'orders': [raw] if number == file_index else []}))
    with pytest.raises(ManualImportError) as error:
        store.validate_interactions('orders', tmp_path, CustomerIdResolver(), DEFAULT_SELECTION,
                                    diagnostics=diagnostics.ManualImportContext('orders', 2))
    detail = error.value.diagnostic
    assert (detail.file_index, detail.file_total, detail.record_index, detail.line_index) == (file_index, 2, 1, 2)
    assert detail.field_path == 'lines[1].' + ('product.ids' if field == 'product' else field)
    assert detail.error_category == category
    text = format_error(error_record(error.value), 'manual_interactions')
    assert f'файла {file_index} из 2, запись 1' in text and 'Позиция заказа: 2' in text
    assert not any(value in text for value in ('PRIVATE', str(tmp_path), 'AdapterError', 'Traceback'))


def test_action_adapter_error_has_safe_field_without_payload(tmp_path):
    raw = {'actionTemplate': {'ids': {'systemName': DEFAULT_SELECTION.view_action_system_names[0]}}}
    (tmp_path / 'actions_part_001.json').write_text(json.dumps({'customerActions': [raw]}))
    with pytest.raises(ManualImportError) as error:
        store.validate_interactions('actions', tmp_path, CustomerIdResolver(), DEFAULT_SELECTION,
                                    diagnostics=diagnostics.ManualImportContext('actions', 1))
    assert error.value.diagnostic.field_path == 'customer.ids.mindboxId'
    assert error.value.diagnostic.record_index == 1
    assert 'Действия → ошибка проверки' in str(error.value)


@pytest.mark.parametrize('text,category', [('{"wrong": []}', 'root_key'), ('[]', 'root_object'),
    ('{"orders": {}}', 'root_array'), ('{"orders": [{}]', 'incomplete_json'),
    ('{"orders": []} {"PRIVATE_EMAIL":"private@example.test"}', 'trailing_json'),
    ('{"orders": [null]}', 'record_object'), ('{"orders": [{"id":1,"id":2}]}', 'duplicate_key')])
def test_strict_json_categories_do_not_expose_parser_excerpts(tmp_path, text, category):
    path = tmp_path / 'source.json'
    path.write_text(text)
    with pytest.raises(ManualImportError) as error:
        list(diagnostics.ManualImportContext('orders', 1).records((path,)))
    assert error.value.diagnostic.phase == 'read' and error.value.diagnostic.error_category == category
    assert 'PRIVATE' not in str(error.value) and 'private@example.test' not in str(error.value)


def test_corrupt_tail_has_count_in_that_file_not_global_count(tmp_path):
    first, second = tmp_path / '1.json', tmp_path / '2.json'
    first.write_text('{"orders": [{}, {}, {}]}')
    second.write_text('{"orders": [{}, {}]} PRIVATE_EMAIL')
    with pytest.raises(ManualImportError) as error:
        list(diagnostics.ManualImportContext('orders', 2).records((first, second)))
    detail = error.value.diagnostic
    assert (detail.file_index, detail.file_total, detail.record_index) == (2, 2, 2)
    assert 'после записи 2' in str(error.value) and 'PRIVATE_EMAIL' not in str(error.value)


def test_invalid_utf8_uses_encoding_category_without_parser_fragment(tmp_path):
    path = tmp_path / 'source.json'
    path.write_bytes(b'{"orders":[{"field":"' + bytes([255]) + b'"}]}')
    with pytest.raises(ManualImportError) as error:
        list(diagnostics.ManualImportContext('orders', 1).records((path,)))
    assert error.value.diagnostic.error_category == 'encoding'
    assert 'field' not in str(error.value) and 'UTF-8' in str(error.value)


def test_utf16_is_rejected_without_auto_conversion(tmp_path):
    path = tmp_path / 'source.json'
    path.write_bytes('{"orders": []}'.encode('utf-16'))
    with pytest.raises(ManualImportError) as error:
        list(diagnostics.ManualImportContext('orders', 1).records((path,)))
    assert error.value.diagnostic.error_category == 'encoding'


@pytest.mark.parametrize('suffix', ['.tmp', '.download'])
def test_selected_unfinished_exports_remain_blocked(tmp_path, suffix):
    from Application.mindbox import manual_import
    path = tmp_path / ('source' + suffix)
    path.write_text('{"orders": []}')
    with pytest.raises(ManualImportError) as error:
        manual_import._copy_validate(path, tmp_path, 'orders')
    assert error.value.diagnostic.error_category == 'incomplete_export'
    assert error.value.diagnostic.phase == 'copy'


@pytest.mark.parametrize('exception,category', [(PermissionError('PRIVATE_PATH'), 'file_access'),
                                              (OSError('PRIVATE_PATH'), 'file_read'),
                                              (UnicodeError('PRIVATE_EMAIL'), 'encoding')])
def test_file_read_failures_have_safe_categories(tmp_path, monkeypatch, exception, category):
    path = tmp_path / 'source.json'
    monkeypatch.setattr(Path, 'open', lambda *a, **k: (_ for _ in ()).throw(exception))
    with pytest.raises(ManualImportError) as error:
        list(diagnostics.ManualImportContext('customers', 1).records((path,)))
    assert error.value.diagnostic.error_category == category
    assert 'PRIVATE' not in str(error.value)


def test_missing_selected_file_has_natural_order_ordinal_before_copying(tmp_path):
    first, missing = tmp_path / 'orders1.json', tmp_path / 'orders2.json'
    first.write_text('{"orders": []}')
    with pytest.raises(ManualImportError) as error:
        diagnostics.normalize_manual_sources((missing, first), 'orders')
    detail = error.value.diagnostic
    assert (detail.source, detail.phase, detail.file_index, detail.file_total, detail.record_index) == ('orders', 'selection', 2, 2, None)
    assert detail.error_category == 'file_missing' and str(tmp_path) not in str(error.value)


def test_error_after_150000_records_uses_only_counters():
    context = diagnostics.ManualImportContext('orders', 4)
    context.file_index, context.record_index = 2, 150001
    with pytest.raises(ManualImportError) as error:
        context.fail(AdapterError('quantity: отсутствует обязательное поле'))
    assert 'запись 150 001' in str(error.value) and error.value.diagnostic.file_index == 2


@pytest.mark.parametrize('exception', [RuntimeError('PRIVATE_EMAIL'), AdapterError('PRIVATE_PHONE: PRIVATE_DATA')])
def test_unknown_exception_or_unrecognized_adapter_message_is_not_forwarded(exception):
    context = diagnostics.ManualImportContext('customers', 1)
    context.record_index = 1
    with pytest.raises(ManualImportError) as error:
        context.fail(exception)
    assert error.value.diagnostic.error_category == 'unknown'
    assert 'PRIVATE' not in json.dumps(error_record(error.value))


def test_manual_cli_unknown_exception_never_serializes_its_message(monkeypatch, capsys):
    from scripts.mindbox_manual_import import main
    from Application.mindbox import manual_import
    monkeypatch.setattr(manual_import, 'import_interactions', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('PRIVATE_EMAIL_API_KEY')))
    assert main(['interactions', '--actions', 'unused.json', '--orders', 'unused.json',
                 '--since', '2022-01-01', '--until', '2023-01-01']) == 1
    output = capsys.readouterr().out
    assert 'PRIVATE' not in output and 'Внутренняя ошибка ручного импорта' not in output  # JSON is ASCII escaped.
    payload = json.loads(output.split('Error: ', 1)[1])
    assert 'Внутренняя ошибка ручного импорта' in payload['message']


def test_untrusted_structured_diagnostic_rebuilds_text_from_closed_categories():
    detail = diagnostics.ManualImportDiagnostic('customers', 'validation', 2, 4, 10, field_path='email', error_category='string')
    payload = {**detail.to_dict(), 'safe_message': 'PRIVATE_EMAIL'}
    assert 'PRIVATE' not in format_error({'manual_import': payload, 'message': 'PRIVATE_PHONE'})
    payload['field_path'] = 'PRIVATE_CUSTOMER'
    assert format_error({'manual_import': payload}) == 'Ошибка ручного импорта → некорректное описание причины от процесса.'
    with pytest.raises(ValueError):
        replace(detail, file_index=True)


def test_read_only_orders_command_reports_missing_prices_without_import_or_raw_values(tmp_path, monkeypatch):
    from scripts.check_historical_orders import main
    from Application.mindbox import manual_import
    monkeypatch.setattr(manual_import, 'import_interactions', lambda *a, **k: pytest.fail('No import in schema audit'))
    source = tmp_path / 'orders.json'
    raw = order()
    source.write_text(json.dumps({'orders': [raw, raw]}))
    before = source.read_bytes()
    output = tmp_path / 'audit.json'
    assert main([str(source), '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert not report['canonical_merges_checked'] and not report['import_performed']
    counts = report['files'][0]['counts']
    assert counts['orders'] == counts['valid_orders'] == counts['missing_base_orders'] == 2
    assert counts['raw_lines'] == counts['adapted_lines'] == counts['purchases_without_base_price'] == 2
    assert 'PRIVATE' not in output.read_text()
    assert source.read_bytes() == before and not (tmp_path / 'canonical').exists()


def test_read_only_orders_audit_does_not_report_corrupt_tail_as_complete(tmp_path):
    from scripts.check_historical_orders import main
    source = tmp_path / 'orders.json'
    source.write_text('{"orders": []} PRIVATE_EMAIL')
    output = tmp_path / 'audit.json'
    assert main([str(source), '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report['files'][0]['complete_json'] and report['files'][0]['read_error']
    assert 'PRIVATE' not in output.read_text()


def test_read_only_command_unknown_failure_does_not_print_traceback_or_source_path(tmp_path, monkeypatch, capsys):
    from scripts import check_historical_orders as checker
    monkeypatch.setattr(checker, 'check_sources', lambda *a: (_ for _ in ()).throw(RuntimeError('PRIVATE_PATH')))
    assert checker.main(['unused.json', '--output', str(tmp_path / 'report.json')]) == 1
    text = capsys.readouterr().out
    assert 'PRIVATE' not in text and 'Traceback' not in text
    assert not (tmp_path / 'report.json').exists()
