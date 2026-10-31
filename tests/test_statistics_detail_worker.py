"""Safe worker/IPC tests: synthetic paths only, no production source access."""
from contextlib import contextmanager
from datetime import date
import io
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from scripts import statistics_detail_export as worker
from Application.statistics_detail_job import (
    MAX_MESSAGE, SourceInfo, decode_message, encode_message, inspect_sources,
)
from Application.statistics_detail_storage import DetailExportError
from tests.test_statistics_detail_backend import job as job_fixture

job = job_fixture


class PendingControl:
    def __init__(self):
        self.done = Event()

    def readline(self, size):
        self.done.wait(5)
        return b''


@contextmanager
def pending():
    control = PendingControl()
    try:
        yield control
    finally:
        control.done.set()


def options(tmp_path):
    target = tmp_path/'result.zip'
    private = tmp_path/'private'
    private.mkdir(exist_ok=True)
    return ['--raw-root', str(tmp_path/'source'), '--destination', str(target),
            '--temporary-root', str(private), '--as-of', '2026-08-31',
            '--start', '2026-01-01', '--end', '2026-08-31']


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.setattr(worker, 'inspect_sources', lambda *a: SourceInfo(date(2020,1,1),date(2026,8,31)))
    monkeypatch.setattr(worker, '_prepare_root', lambda *a: None)


def messages(output):
    raw = output.getvalue()
    assert all(len(line.encode()) < MAX_MESSAGE for line in raw.splitlines())
    return [decode_message(line) for line in raw.splitlines()]


def test_parameters_safe_json_progress_throttle_and_output_redirection(tmp_path, monkeypatch, mocked):
    captured = {}
    def export(**kwargs):
        captured.update(kwargs)
        print('SYNTH_PRIVATE_CONTACT@example.invalid')
        for _ in range(100):
            kwargs['progress'](dict(stage='orders', processed=100))
        return SimpleNamespace(manifest=dict(rows=dict(clients=1,orders=2,order_lines=3)))
    monkeypatch.setattr(worker, 'export_detail_statistics', export)
    output = io.StringIO()
    with pending() as control:
        code = worker.main(options(tmp_path), control=control, output=output)
    events = messages(output)
    assert code == 0 and events[-1] == dict(event='finished',rows=dict(clients=1,orders=2,order_lines=3))
    assert len(events) <= 3 and captured['as_of_date'] == date(2026,8,31)
    assert captured['event_period'].start_date == date(2026,1,1)
    assert captured['order_mode'].value == 'ALL_SELECTED'
    assert 'SYNTH_PRIVATE' not in output.getvalue()


@pytest.mark.parametrize('error',[
    DetailExportError('INSUFFICIENT_SPACE'), DetailExportError('ACCESS_DENIED'),
    DetailExportError('DETAIL_EXPORT_FAILED'), RuntimeError('SYNTH_SECRET@example.invalid'),
])
def test_backend_error_has_safe_code_and_preserves_old_target(tmp_path, monkeypatch, mocked, error):
    args = options(tmp_path)
    target = tmp_path/'result.zip'
    target.write_bytes(b'previous')
    def fail(**kwargs):
        raise error
    monkeypatch.setattr(worker, 'export_detail_statistics', fail)
    output = io.StringIO()
    with pending() as control:
        code = worker.main(args, control=control, output=output)
    assert code == 1 and messages(output)[-1]['event'] == 'error'
    assert 'SYNTH_SECRET' not in output.getvalue() and target.read_bytes() == b'previous'


@pytest.mark.parametrize('control,expected',[
    (b'{"command":"cancel"}\n','cancelled'), (b'', 'cancelled'),
    (b'{"customer_id":"SYNTH_SECRET"}\n','error'), (b'x'*(MAX_MESSAGE+1),'error'),
])
def test_cancel_eof_and_invalid_control_are_cooperative(tmp_path, monkeypatch, mocked, control, expected):
    def export(**kwargs):
        for _ in range(10000):
            if kwargs['cancelled']():
                raise InterruptedError
        pytest.fail('Cancellation not delivered')
    monkeypatch.setattr(worker, 'export_detail_statistics', export)
    output = io.StringIO()
    code = worker.main(options(tmp_path), control=io.BytesIO(control), output=output)
    assert code == (2 if expected == 'cancelled' else 1)
    assert messages(output)[-1]['event'] == expected and 'SYNTH_SECRET' not in output.getvalue()


@pytest.mark.parametrize('change',[
    ['--as-of','invalid'], ['--end','2026-09-01'], ['--order-mode','bad'],
    ['--customer-email','SYNTH_SECRET@example.invalid'],
])
def test_invalid_arguments_never_echo_values(tmp_path, mocked, change):
    output = io.StringIO()
    code = worker.main(options(tmp_path)+change, output=output, control=io.BytesIO(b''))
    assert code == 1 and messages(output)[-1] == dict(event='error',code='INVALID_OPTIONS')
    assert 'SYNTH_SECRET' not in output.getvalue()


@pytest.mark.parametrize('message',[
    {}, {'event':'unknown'}, {'event':'progress','stage':'orders','processed':True},
    {'event':'progress','stage':'orders','processed':1,'email':'SYNTH_SECRET'},
    {'event':'progress','stage':[],'processed':1},
    {'event':'finished','rows':{'clients':-1,'orders':0,'order_lines':0}},
    {'event':'error','code':'SYNTH_SECRET'},
    {'event':'cancelled','value':'SYNTH_SECRET'},
])
def test_strict_protocol_rejects_unknown_fields_and_values(message):
    with pytest.raises((ValueError,TypeError)):
        encode_message(message)


def test_size_duplicate_keys_and_utf8_are_rejected():
    for line in (b'x'*(MAX_MESSAGE+1), b'{"event":"cancelled","event":"cancelled"}', b'\xff'):
        with pytest.raises((ValueError,UnicodeError)):
            decode_message(line)


def test_lightweight_source_metadata_does_not_read_orders_or_customers(job, monkeypatch):
    from Application.mindbox import raw_reader
    monkeypatch.setattr(raw_reader, 'iter_export', lambda *a,**kw: pytest.fail('Rows read during metadata preflight'))
    value = inspect_sources(job.root, today=date(2026,10,31))
    assert value.first_date == date(2020,1,1) and value.last_date == date(2026,8,31)


def test_worker_with_real_backend_uses_only_small_synthetic_fixture(job):
    output = io.StringIO()
    args = ['--raw-root',str(job.root),'--destination',str(job.target),
            '--temporary-root',str(job.private),'--as-of','2026-08-31']
    with pending() as control:
        code = worker.main(args, output=output, control=control)
    assert code == 0 and messages(output)[-1]['rows'] == dict(clients=1,orders=1,order_lines=1)
    assert job.target.is_file() and not list(job.private.iterdir())
    assert 'SYNTH_CONTACT' not in output.getvalue()


def test_real_child_stdin_cancellation_has_correct_exit_and_no_traceback(tmp_path):
    script = (
        "import sys; from scripts import statistics_detail_export as w; "
        "w.inspect_sources=lambda *a:None; w._prepare_root=lambda *a:None; "
        "raise SystemExit(w.main(sys.argv[1:]))"
    )
    # Cancel is consumed before backend; no real source access or archive creation.
    result = subprocess.run([sys.executable,'-B','-u','-c',script,*options(tmp_path)],
        input=b'{"command":"cancel"}\n',capture_output=True,cwd=Path(__file__).resolve().parents[1],timeout=10)
    assert result.returncode == 2 and result.stderr == b''
    assert decode_message(result.stdout.splitlines()[-1]) == dict(event='cancelled')


def test_missing_sources_returns_error_without_export(tmp_path, monkeypatch):
    monkeypatch.setattr(worker,'export_detail_statistics',lambda **kw:pytest.fail('Unexpected export'))
    output=io.StringIO()
    with pending() as control:
        code=worker.main(options(tmp_path),control=control,output=output)
    assert code==1 and messages(output)[-1]['code']=='SOURCE_NOT_READY'


def test_real_backend_write_failure_keeps_previous_zip_and_cleans(job,monkeypatch):
    import zipfile
    job.target.write_bytes(b'previous')
    original=zipfile.ZipFile.open
    def fail(archive,name,mode='r',*args,**kwargs):
        if mode=='w' and getattr(name,'filename',name)=='orders.csv':
            raise PermissionError('SYNTH_SECRET@example.invalid')
        return original(archive,name,mode,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'open',fail)
    output=io.StringIO()
    args=['--raw-root',str(job.root),'--destination',str(job.target),
          '--temporary-root',str(job.private),'--as-of','2026-08-31']
    with pending() as control:
        code=worker.main(args,control=control,output=output)
    assert code==1 and messages(output)[-1]['code']=='ACCESS_DENIED'
    assert job.target.read_bytes()==b'previous' and not list(job.private.iterdir())
    assert 'SYNTH_SECRET' not in output.getvalue()


def test_new_private_root_setup_does_not_repair_existing_root(tmp_path,monkeypatch):
    created=[]
    validated=[]
    monkeypatch.setattr(worker.storage,'admit_destination',lambda p:None)
    monkeypatch.setattr(worker.storage,'protect_directory',lambda p:created.append(p))
    monkeypatch.setattr(worker.storage,'validate_storage_root',lambda p:validated.append(p))
    root=tmp_path/'owned-parent'/'owned-temp'
    worker._prepare_root(root)
    assert created==[root.parent,root] and validated==created
    worker._prepare_root(root)
    assert created==[root.parent,root] and validated[-1]==root


def test_final_zip_retains_private_file_acl_after_atomic_replace(tmp_path):
    from Application import statistics_detail_storage as storage
    import os
    partial=tmp_path/'owned-empty.partial'
    target=tmp_path/'owned-empty.zip'
    partial.write_bytes(b'')
    storage.protect_file(partial)
    os.replace(partial,target)
    if os.name=='nt':
        storage.check_acl_facts(storage.inspect_acls((target,))[0],private=True)
    else:
        assert target.stat().st_mode & 0o077==0


def test_oversized_unicode_cleanup_path_keeps_a_valid_safe_error(tmp_path):
    temporary=tmp_path/('Ж'*700)
    path=temporary/'detail-abcdefgh'
    error=DetailExportError('SENSITIVE_STATE_CLEANUP_FAILED',technical_path=path)
    message=worker._safe_error(error,temporary,tmp_path/'result.zip')
    assert message==dict(event='error',code='SENSITIVE_STATE_CLEANUP_FAILED')
    assert len(encode_message(message))<MAX_MESSAGE


def test_heartbeat_and_terminal_keep_backend_stdout_stderr_private(tmp_path, monkeypatch, mocked, capsys):
    import sys

    class HeartbeatOutput(io.StringIO):
        def __init__(self):
            super().__init__()
            self.progress_count = 0
            self.heartbeat = Event()

        def write(self, text):
            count = super().write(text)
            if decode_message(text)['event'] == 'progress':
                self.progress_count += 1
                if self.progress_count >= 2:
                    self.heartbeat.set()
            return count

    output = HeartbeatOutput()

    def export(**kwargs):
        kwargs['progress'](dict(stage='orders', processed=7))
        print('SYNTH_PRIVATE@example.invalid')
        print('SYNTH_PRIVATE@example.invalid', file=sys.stderr)
        assert output.heartbeat.wait(5), 'Worker heartbeat was not delivered'
        return SimpleNamespace(manifest=dict(rows=dict(clients=1, orders=1, order_lines=1)))

    monkeypatch.setattr(worker, 'export_detail_statistics', export)
    with pending() as control:
        code = worker.main(options(tmp_path), control=control, output=output)
    events = messages(output)
    assert code == 0 and output.progress_count >= 2
    assert all(event == dict(event='progress', stage='orders', processed=7) for event in events[:-1])
    assert events[-1]['event'] == 'finished'
    captured = capsys.readouterr()
    assert captured.out == captured.err == ''
    assert 'SYNTH_PRIVATE' not in output.getvalue()
