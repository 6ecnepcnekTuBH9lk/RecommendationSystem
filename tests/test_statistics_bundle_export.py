"""Atomic two-file bundle without touching canonical data or existing destinations."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook
import pytest

from Application import statistics_bundle_export as bundle
from tabs.test_dataset_statistics_tab import sample_result
from test_statistics_export import assert_minimal_tables, kpi

__all__ = ['sample_result']


def test_bundle_fixed_names_crc_same_snapshot_and_excel_design(sample_result, tmp_path):
    result = deepcopy(sample_result)
    path = tmp_path / 'user-chosen.zip'
    bundle.export_statistics_bundle(result, path)
    assert result == sample_result
    assert list(tmp_path.iterdir()) == [path]
    assert bundle.suggested_filename(result) == 'Статистика_2026-09-24_09-47-28.zip'
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ['Статистика_2026-09-24_09-47-28.xlsx', 'Статистика_2026-09-24_09-47-28.html']
        assert all(not Path(name).is_absolute() and '/' not in name and '\\' not in name for name in archive.namelist())
        book = load_workbook(BytesIO(archive.read(archive.namelist()[0])))
        try:
            assert assert_minimal_tables(book)
            assert kpi(book['Сводка'], 'Количество взаимодействий').value == result['total_interactions']
        finally:
            book.close()
        html = archive.read(archive.namelist()[1]).decode('utf-8')
        assert '<!doctype html>' in html and '24.09.2026' in html
        assert 'plotly-library' in html


@pytest.mark.parametrize('phase', ['xlsx', 'html', 'zip', 'crc', 'fsync', 'replace'])
def test_failure_keeps_old_bundle_and_removes_only_owned_temporary_files(sample_result, tmp_path, monkeypatch, phase):
    target = tmp_path / 'existing.zip'
    target.write_bytes(b'original')
    other = tmp_path / 'unrelated.txt'
    other.write_text('preserve')

    def fail(*args, **kwargs):
        raise PermissionError('synthetic')
    if phase == 'xlsx':
        monkeypatch.setattr(bundle, 'export_statistics', fail)
    elif phase == 'html':
        monkeypatch.setattr(bundle, 'export_statistics_html', fail)
    elif phase == 'zip':
        write = bundle.ZipFile.write

        def fail_archive_write(archive, *args, **kwargs):
            if str(archive.filename).endswith('bundle.zip'):
                fail()
            return write(archive, *args, **kwargs)
        monkeypatch.setattr(bundle.ZipFile, 'write', fail_archive_write)
    elif phase == 'crc':
        monkeypatch.setattr(bundle.ZipFile, 'testzip', lambda self: 'bad.xlsx')
    elif phase == 'fsync':
        sync = bundle.os.fsync
        calls = []

        def fail_archive_sync(fd):
            calls.append(fd)
            if len(calls) == 2:
                fail()
            sync(fd)
        monkeypatch.setattr(bundle.os, 'fsync', fail_archive_sync)
    else:
        replace = bundle.os.replace

        def fail_archive_replace(source, destination):
            if Path(destination) == target:
                fail()
            replace(source, destination)
        monkeypatch.setattr(bundle.os, 'replace', fail_archive_replace)
    with pytest.raises(OSError):
        bundle.export_statistics_bundle(sample_result, target)
    assert target.read_bytes() == b'original' and other.read_text() == 'preserve'
    assert set(tmp_path.iterdir()) == {target, other}
