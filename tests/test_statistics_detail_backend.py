"""Synthetic canonical detail ZIPs in pytest D: directories, outside the repo."""

from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import date
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tracemalloc
from types import SimpleNamespace
import zipfile

import pytest

from Application import statistics_detail_backend as backend, statistics_detail_storage as storage
from Application.statistics_detail_contract import (
    CLIENT_COLUMNS, ORDER_COLUMNS, ORDER_LINE_COLUMNS, DetailPeriod, OrderMode,
    decode_csv_text, NULL_TOKEN,
)
from Application.statistics_storage_safety import path_identity
from tests.test_dataset_statistics import dataset as source_fixture, save, action, DEFAULT_SELECTION
from tests.test_statistics_rfm_integration import raw_order
from tests.test_analysis_filter import sources


AS_OF=date(2026,8,31)
PERIOD=DetailPeriod(date(2026,1,1),AS_OF)
SYNTH_EMAIL='SYNTH_CONTACT@example.invalid'
SYNTH_PHONE='+70000000000'
_real_protect_file=storage.protect_file


@pytest.fixture
def job(tmp_path,monkeypatch):
    source=tmp_path/'source'
    data=source_fixture.__wrapped__(source)
    (source/'canonical/.writer.lock').write_bytes(b'0')
    private=tmp_path/'private'
    private.mkdir()
    # Functional fixtures isolate Windows ACL utilities; storage admission is
    # exercised separately below, and production never skips these checks.
    monkeypatch.setattr(storage,'validate_storage_root',lambda p,**kw:path_identity(p))
    monkeypatch.setattr(storage,'protect_directory',lambda p:p.chmod(0o700))
    monkeypatch.setattr(storage,'protect_file',lambda p:p.chmod(0o600))
    monkeypatch.setattr(storage,'admit_destination',lambda p:None)
    target=tmp_path/'detail.zip'
    def configure(records):
        if data[2].get('manual_interactions'):
            entry=data[2]['manual_interactions']['orders']
            path=source/entry['directory']/'orders_part_001.json'
            path.write_text(json.dumps({'orders':records}),encoding='utf-8')
        else:
            sources(data,[action(DEFAULT_SELECTION.view_action_system_names[0],['000001-size'])],records,
                    since='2020-01-01T00:00:00Z',until='2026-09-01T00:00:00Z')
    configure([raw_order('000000000000000000000000001','2026-01-01','old')])
    with closing(sqlite3.connect(source/'canonical/customers.sqlite')) as db, db:
        record={'ids':{'mindboxId':'new'},'firstName':'SYNTH_FIRST','lastName':'SYNTH_LAST','email':SYNTH_EMAIL,
                'mobilePhone':SYNTH_PHONE,'changeDateTimeUtc':'2026-01-01T00:00:00Z',
                'subscriptions':[{'channel':'email','isSubscribed':True}]}
        db.execute('UPDATE profiles SET changed=?,raw=? WHERE id=?',('2026-01-01T00:00:00+00:00',json.dumps(record),'new'))
    return SimpleNamespace(data=data,root=source,private=private,target=target,configure=configure)


def run(job,**kwargs):
    options=dict(raw_root=job.root,destination=job.target,temporary_root=job.private,as_of_date=AS_OF,event_period=PERIOD)
    options.update(kwargs)
    return backend.export_detail_statistics(**options)


def read(job):
    with zipfile.ZipFile(job.target) as archive:
        assert tuple(archive.namelist())==backend.FILES
        manifest=json.loads(archive.read('manifest.json'))
        tables={}
        for name in backend.FILES[:3]:
            with archive.open(name) as stream:
                text=io.TextIOWrapper(stream,encoding='utf-8-sig',newline='')
                reader=csv.reader(text,delimiter=';')
                headers=next(reader)
                tables[name]=(headers,[dict(zip(headers,row)) for row in reader])
    return manifest,tables


def clean(job):
    assert list(job.private.iterdir())==[]
    assert not list(job.target.parent.glob('.statistics-detail-*.partial'))


def snapshot(job):
    return {str(p.relative_to(job.root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in job.root.rglob('*') if p.is_file()}


def test_one_customer_real_ids_contacts_permissions_schema_and_read_only_sources(job):
    before=snapshot(job)
    progress=[]
    result=run(job,progress=progress.append)
    manifest,tables=read(job)
    assert result.manifest==manifest and snapshot(job)==before
    assert manifest['rows']==dict(clients=1,orders=1,order_lines=1)
    assert [tuple(tables[name][0]) for name in backend.FILES[:3]]==[CLIENT_COLUMNS,ORDER_COLUMNS,ORDER_LINE_COLUMNS]
    c=tables['clients.csv'][1][0]
    o=tables['orders.csv'][1][0]
    line=tables['order_lines.csv'][1][0]
    assert decode_csv_text(c['customer_id'])=='new'
    assert decode_csv_text(c['email'])==SYNTH_EMAIL and decode_csv_text(c['mobile_phone'])==SYNTH_PHONE
    assert decode_csv_text(c['email_permission'])==decode_csv_text(c['sms_permission'])=='UNKNOWN'
    assert decode_csv_text(o['order_id'])=='000000000000000000000000001'
    assert c['customer_id']==o['customer_id']==line['customer_id'] and o['order_id']==line['order_id']
    assert c['strict_rfm_12m_rub_status']!="'NOT_COMPUTED"
    assert c['strict_rfm_12m_kzt_status']=="'NOT_IN_CURRENCY_BASE"
    assert all(secret not in json.dumps(manifest) for secret in ('SYNTH_FIRST',SYNTH_EMAIL,SYNTH_PHONE,'000000000000000000000000001'))
    assert all(secret not in json.dumps(progress) for secret in ('SYNTH_FIRST',SYNTH_EMAIL,SYNTH_PHONE))
    assert set(m['stage'] for m in progress)=={'source_validation','history','personal_metrics','rfm','clients','orders','order_lines','zip_verification','atomic_save'}
    clean(job)


@pytest.mark.parametrize('mode',[OrderMode.ALL_SELECTED,OrderMode.PURCHASE_EITHER])
def test_empty_and_nonpurchase_owner_scope(job,mode):
    job.configure([raw_order('returned','2026-01-01','without-profile',('Return',))])
    run(job,order_mode=mode)
    m,t=read(job)
    assert m['rows']['orders']==int(mode is OrderMode.ALL_SELECTED)
    if mode is OrderMode.ALL_SELECTED:
        c=t['clients.csv'][1][0]
        assert decode_csv_text(c['profile_status'])=='MISSING' and c['email']==NULL_TOKEN
        assert c['strict_orders']=='0' and c['strict_rfm_12m_rub_status']=="'OUTSIDE_BASE"
    clean(job)
    job.configure([])
    run(job)
    assert read(job)[0]['rows']==dict(clients=0,orders=0,order_lines=0)
    clean(job)


def test_multiple_customers_aliases_duplicates_old_history_and_future_cutoff(job,monkeypatch):
    historical=raw_order('historic','2020-01-01','old')
    current=raw_order('current','2026-01-01','new',('F','CP'))
    other=raw_order('other','2026-02-01','other',('CP',))
    job.configure([historical,deepcopy(historical),current,deepcopy(current),other,
                   raw_order('future','2027-01-01','new')])
    calls,adapted,built=[],[],[]
    source,adapt,factory=backend.iter_export,backend.adapt_order,backend.build_purchase_order_fact
    def observed(name,**kwargs):
        calls.append(name)
        yield from source(name,**kwargs)
    def observed_adapt(raw,*args,**kwargs):
        adapted.append(raw['ids']['mindboxId'])
        return adapt(raw,*args,**kwargs)
    def observed_fact(*args,**kwargs):
        built.append(kwargs['order_id'])
        return factory(*args,**kwargs)
    monkeypatch.setattr(backend,'iter_export',observed)
    monkeypatch.setattr(backend,'adapt_order',observed_adapt)
    monkeypatch.setattr(backend,'build_purchase_order_fact',observed_fact)
    run(job)
    m,t=read(job)
    assert Counter(calls)['orders']==Counter(calls)['customer_merges']==1
    assert adapted==built==['historic','current','other']
    assert m['history']['history_orders']==3 and m['history']['identical_snapshots']==2
    assert m['rows']==dict(clients=2,orders=2,order_lines=3)
    c=next(c for c in t['clients.csv'][1] if decode_csv_text(c['customer_id'])=='new')
    assert c['strict_orders']=='2' and decode_csv_text(c['strict_first_observed_cohort'])=='2020-01'
    assert c['extended_orders']=='2' and c['strict_orders_48m']=='1'
    assert c['strict_rfm_12m_rub_status']=="'AVAILABLE"
    clean(job)


def test_unicode_formula_text_and_long_product_identifiers_roundtrip(job):
    raw=raw_order('000000000000000000000000000000000003','2026-01-01','new')
    raw['lines'][0]['product']={'ids':{'offline1C':'0000000000000000000000000000000004-size'},'name':'=СУММ(1;2) "Юникод"\nСтрока'}
    job.configure([raw])
    run(job)
    _,t=read(job)
    row=t['order_lines.csv'][1][0]
    assert row['product_name'].startswith("'")
    assert decode_csv_text(row['product_name'])==raw['lines'][0]['product']['name']
    assert decode_csv_text(row['product_id'])=='0000000000000000000000000000000004-size'
    clean(job)


@pytest.mark.parametrize('scenario',['mixed','negative','unavailable_profiles','conflicting_profiles'])
def test_quality_money_and_profile_contract(job,scenario):
    raw=raw_order('quality','2026-01-01','old',('F','CP','Return','CD'))
    if scenario=='mixed':raw['lines'][1]['product']['ids']={'kanzlerKz':'ORIGINAL-KZT'}
    if scenario=='negative':raw['lines'][0]['priceOfLine']=-3
    job.configure([raw])
    path=job.root/'canonical/customers.sqlite'
    if scenario=='unavailable_profiles':
        # Synthetic fixture only: move aside to represent a missing optional source.
        path.rename(job.root/'synthetic-unused-profiles.sqlite')
    elif scenario=='conflicting_profiles':
        with closing(sqlite3.connect(path)) as db, db:
            other={'ids':{'mindboxId':'old'},'firstName':'SYNTH_CONFLICT','email':'OTHER_SYNTH@example.invalid',
                   'changeDateTimeUtc':'2026-01-01T00:00:00Z'}
            db.execute('INSERT INTO profiles VALUES(?,?,?)',('old','2026-01-01T00:00:00+00:00',json.dumps(other)))
    run(job)
    m,t=read(job)
    c=t['clients.csv'][1][0];o=t['orders.csv'][1][0]
    if scenario=='mixed':
        assert o['strict_currency']=="'RUB" and o['extended_currency']=="'mixed"
        assert o['extended_amount']==NULL_TOKEN and c['extended_rfm_12m_rub_m_score']==NULL_TOKEN
    elif scenario=='negative':
        assert o['strict_amount']==NULL_TOKEN and c['strict_unvalued_orders']=='1'
        assert c['strict_rfm_12m_rub_status']=="'UNSCORABLE"
    else:
        assert c['email']==NULL_TOKEN and c['mobile_phone']==NULL_TOKEN
        assert decode_csv_text(c['profile_status'])==('SOURCE_UNAVAILABLE' if scenario=='unavailable_profiles' else 'CONFLICT')
        assert c['strict_orders']=='1'
    assert o['return_lines']==o['cancelled_lines']=='1'
    clean(job)


@pytest.mark.parametrize('stage',['history','personal_metrics','rfm','clients','orders','order_lines','zip_verification','atomic_save'])
def test_cancellation_preserves_existing_destination_and_cleans_owned_files(job,stage):
    job.target.write_bytes(b'previous successful destination')
    stopped=False
    def progress(event):
        nonlocal stopped
        stopped |= event['stage']==stage
    with pytest.raises(InterruptedError) as error:
        run(job,progress=progress,cancelled=lambda:stopped)
    assert str(error.value)=='Detail export cancelled'
    assert job.target.read_bytes()==b'previous successful destination'
    clean(job)


@pytest.mark.parametrize('kind',['zip','replace','space','source_changed','bad_price','snapshot_conflict'])
def test_failures_are_private_atomic_and_cleanup_is_complete(job,monkeypatch,kind):
    job.target.write_bytes(b'previous')
    if kind=='zip':
        original=zipfile.ZipFile.open
        def fail(self,name,mode='r',*args,**kwargs):
            if mode=='w' and getattr(name,'filename',name)=='orders.csv':raise OSError(SYNTH_EMAIL)
            return original(self,name,mode,*args,**kwargs)
        monkeypatch.setattr(zipfile.ZipFile,'open',fail)
    elif kind=='replace':
        monkeypatch.setattr(backend.os,'replace',lambda *a:(_ for _ in ()).throw(OSError(SYNTH_PHONE)))
    elif kind=='space':
        monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    elif kind=='source_changed':
        original=backend._source_version
        calls=0
        def changed(*a,**kw):
            nonlocal calls
            calls+=1
            return original(*a,**kw) if calls==1 else 'changed'
        monkeypatch.setattr(backend,'_source_version',changed)
    elif kind=='bad_price':
        raw=raw_order('bad','2026-01-01');raw['lines'][0]['priceOfLine']='SYNTH_SECRET'
        job.configure([raw])
    else:
        job.configure([raw_order('same','2026-01-01',statuses=('Return',)),raw_order('same','2026-01-01')])
    with pytest.raises(storage.DetailExportError) as error:
        run(job)
    assert all(secret not in str(error.value) for secret in (SYNTH_EMAIL,SYNTH_PHONE,'SYNTH_SECRET'))
    assert job.target.read_bytes()==b'previous'
    clean(job)


def test_cleanup_failure_reports_owned_technical_path_without_publishing(job,monkeypatch):
    job.target.write_bytes(b'previous')
    original=Path.unlink
    with monkeypatch.context() as patch:
        def fail(p,*a,**kw):
            if p.name=='state.sqlite':raise PermissionError(SYNTH_EMAIL)
            return original(p,*a,**kw)
        patch.setattr(Path,'unlink',fail)
        with pytest.raises(storage.DetailExportError,match='SENSITIVE_STATE_CLEANUP_FAILED') as error:
            run(job)
    leftover=Path(error.value.technical_path)
    assert leftover.parent==job.private and leftover.name.startswith('detail-')
    assert SYNTH_EMAIL not in str(error.value) and job.target.read_bytes()==b'previous'
    # Remove only this test's known owned leftover after restoring deletion.
    for p in leftover.iterdir():
        assert p.name in {'state.sqlite','state.sqlite-journal'}
        p.unlink()
    leftover.rmdir()
    clean(job)


def test_admission_failure_happens_before_source_reads(job,monkeypatch):
    monkeypatch.setattr(storage,'validate_storage_root',lambda *a,**kw:(_ for _ in ()).throw(ValueError('unsafe')))
    monkeypatch.setattr(backend,'iter_export',lambda *a,**kw:pytest.fail('Source read before admission'))
    with pytest.raises(storage.DetailExportError):run(job)
    assert not job.target.exists()
    clean(job)


def test_results_are_deterministic_except_generated_time(job):
    run(job);m1,t1=read(job)
    run(job);m2,t2=read(job)
    m1.pop('generated_at');m2.pop('generated_at')
    assert m1==m2 and t1==t2
    clean(job)


def test_thousands_of_synthetic_rows_use_bounded_python_memory(job):
    data=job.data[2]
    entry=data['manual_interactions']['orders']
    path=job.root/entry['directory']/'orders_part_001.json'
    n=1500
    with path.open('w',encoding='utf-8') as stream:
        stream.write('{"orders":[')
        for i in range(n):
            if i:stream.write(',')
            json.dump(raw_order('synthetic-order-'+str(i),'2026-01-01','synthetic-owner-'+str(i%35),('F','CP')),stream)
        stream.write(']}')
    tracemalloc.start()
    result=run(job)
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert result.manifest['rows']==dict(clients=35,orders=n,order_lines=2*n)
    assert peak<24*1024**2
    clean(job)



def test_event_period_does_not_change_history_rfm_or_global_population(job):
    records=[raw_order('selected','2026-01-01','new')]
    for i in range(9):
        raw=raw_order('other-order-'+str(i),'2025-10-01','other-buyer-'+str(i))
        raw['lines'][0]['priceOfLine']=30*(i+1)
        records.append(raw)
    job.configure(records)
    run(job)
    narrow,nt=read(job)
    run(job,event_period=DetailPeriod())
    wide,wt=read(job)
    selected=next(c for c in wt['clients.csv'][1] if decode_csv_text(c['customer_id'])=='new')
    assert nt['clients.csv'][1]==[selected]
    assert narrow['rfm_population_token']==wide['rfm_population_token']
    assert narrow['rows']['clients']==1 and wide['rows']['clients']==10
    clean(job)


def test_partial_history_preserves_observations_without_scores(job):
    entry=job.data[2]['manual_interactions']
    entry['since']='2026-01-01T00:00:00Z'
    for kind in ('orders','actions'):entry[kind]['since']=entry['since']
    save(job.root,job.data[2])
    run(job)
    manifest,tables=read(job)
    row=tables['clients.csv'][1][0]
    assert row['strict_rfm_12m_rub_status']=="'UNSCORABLE"
    assert row['strict_rfm_12m_rub_f_orders']=='1'
    assert row['strict_rfm_12m_rub_monetary']==NULL_TOKEN
    assert row['strict_known_amount_rub']=='20'
    assert row['strict_rfm_12m_rub_f_score']==row['strict_rfm_12m_rub_m_score']==NULL_TOKEN
    assert manifest['quality']['INCOMPLETE_WINDOW']>0
    clean(job)


def test_empty_order_keeps_header_and_nonpositive_lines_are_not_purchases(job):
    empty=raw_order('no-lines','2026-01-01','new',statuses=())
    fractional=raw_order('fractional','2026-01-01','new')
    fractional['lines'][0]['quantity']=0.5
    job.configure([empty,fractional])
    run(job)
    manifest,tables=read(job)
    assert manifest['rows']==dict(clients=1,orders=2,order_lines=1)
    assert tables['clients.csv'][1][0]['strict_orders']=='0'
    assert tables['order_lines.csv'][1][0]['quantity']=='0.5'
    clean(job)


def test_contacts_never_enter_temporary_database_and_same_email_is_not_identity(job):
    job.configure([raw_order('a','2026-01-01','new'),raw_order('b','2026-01-01','other')])
    record={'ids':{'mindboxId':'other'},'email':SYNTH_EMAIL}
    with closing(sqlite3.connect(job.root/'canonical/customers.sqlite')) as db, db:
        db.execute('UPDATE profiles SET raw=? WHERE id=?',(json.dumps(record),'other'))
    def progress(event):
        if event['stage']=='orders':
            folders=list(job.private.iterdir())
            assert len(folders)==1
            for path in folders[0].iterdir():
                data=path.read_bytes()
                assert all(secret.encode() not in data for secret in (SYNTH_EMAIL,SYNTH_PHONE,'SYNTH_FIRST'))
    run(job,progress=progress)
    _,tables=read(job)
    clients=tables['clients.csv'][1]
    assert len(clients)==2 and len({c['customer_id'] for c in clients})==2
    assert all(decode_csv_text(c['email'])==SYNTH_EMAIL for c in clients)
    clean(job)


@pytest.mark.parametrize('kind',['runtime_space','budget','callback','mid_loop_cancel','destination_changed','partial_changed'])
def test_runtime_guards_preserve_destination(job,monkeypatch,kind):
    job.target.write_bytes(b'previous')
    cancel_calls=0
    changed=False
    def cancel():
        nonlocal cancel_calls
        cancel_calls+=1
        return kind=='mid_loop_cancel' and cancel_calls>90
    def progress(event):
        nonlocal changed
        if event['stage']=='clients':
            if kind=='runtime_space':
                monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
            elif kind=='budget':
                folder=next(job.private.iterdir())
                with (folder/'state.sqlite-journal').open('wb') as stream:stream.truncate(17*1024**2)
            elif kind=='callback':raise RuntimeError(SYNTH_EMAIL)
        if event['stage']=='atomic_save':
            if kind=='destination_changed':job.target.write_bytes(b'changed externally')
            if kind=='partial_changed':
                monkeypatch.setattr(backend.DetailWorkspace,'validate_partial',
                    lambda self:(_ for _ in ()).throw(storage.DetailExportError('PUBLICATION_PATH_CHANGED')))
                changed=True
    error_type=InterruptedError if kind=='mid_loop_cancel' else storage.DetailExportError
    with pytest.raises(error_type) as error:
        run(job,progress=progress,cancelled=cancel,max_temporary_bytes=16*1024**2)
    assert SYNTH_EMAIL not in str(error.value)
    assert job.target.read_bytes()==(b'changed externally' if kind=='destination_changed' else b'previous')
    if changed:
        # Cleanup correctly refused the mocked unowned path; remove test-owned file.
        assert getattr(error.value,'technical_path',None)
        for path in job.target.parent.glob('.statistics-detail-*.partial'):path.unlink()
    clean(job)


def test_temporary_root_and_destination_cannot_be_inside_repository(job):
    for options in (dict(temporary_root=Path.cwd()),dict(destination=Path.cwd()/'forbidden-detail.zip')):
        with pytest.raises(storage.DetailExportError):run(job,**options)
    clean(job)


def test_private_file_permissions_are_applied_to_new_owned_file(tmp_path):
    # No real inputs; exercise the actual Windows ACL helper, separately from
    # the fast functional fixtures which mock platform-specific subprocesses.
    path=tmp_path/'owned-empty-file'
    path.write_bytes(b'')
    _real_protect_file(path)
    if storage.os.name=='nt':
        storage.check_acl_facts(storage.inspect_acls((path,))[0],private=True)
    else:
        assert path.stat().st_mode & 0o077==0



@pytest.mark.parametrize('entry',['clients.csv','orders.csv','order_lines.csv'])
def test_failure_inside_stream_closes_generators_and_keeps_previous_zip(job,monkeypatch,entry):
    job.target.write_bytes(b'previous')
    original=zipfile._ZipWriteFile.write
    def fail(stream,data):
        if stream._zinfo.filename==entry and not bytes(data).startswith(b'\xef\xbb\xbf'):
            raise OSError(SYNTH_EMAIL)
        return original(stream,data)
    monkeypatch.setattr(zipfile._ZipWriteFile,'write',fail)
    with pytest.raises(storage.DetailExportError) as error:run(job)
    assert SYNTH_EMAIL not in str(error.value) and job.target.read_bytes()==b'previous'
    clean(job)


def test_fresh_backend_import_does_not_open_sources_write_files_or_import_qt():
    import os
    import subprocess
    import sys
    script="""
import os,sys
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if (isinstance(mode,str) and any(c in mode for c in 'wax+') or
                isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            raise AssertionError('Unexpected write')
        if isinstance(path,str) and (path.endswith('customers.sqlite') or 'input_data' in path):
            raise AssertionError('Unexpected data read')
    if event in ('os.mkdir','sqlite3.connect','socket.connect','subprocess.Popen'):
        raise AssertionError('Unexpected I/O')
sys.addaudithook(guard)
from Application.statistics_detail_backend import export_detail_statistics
assert callable(export_detail_statistics)
assert not any(n.startswith(('PyQt','qt_material','torch')) for n in sys.modules)
assert 'Application.statistics_index' not in sys.modules
"""
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=Path(__file__).resolve().parents[1],
                          env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert result.returncode==0,result.stderr
