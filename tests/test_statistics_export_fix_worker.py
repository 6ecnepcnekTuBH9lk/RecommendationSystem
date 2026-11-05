"""Pipe finalization, safe diagnostics and unchanged private ACL admission."""
import json
import os
import pytest
from scripts.statistics_detail_export import _ControlPipe
from Application import statistics_detail_storage as storage
from Application.statistics_detail_job import exception_diagnostic,decode_diagnostic
from Application.statistics_storage_safety import check_acl_facts,StatisticsIndexError

def test_raw_control_pipe_keeps_prefetched_report_and_cancel_and_limit():
    read,write=os.pipe()
    try:
        os.write(write,b'{"command":"report"}\n{"command":"cancel"}\n')
        os.close(write);write=None
        channel=_ControlPipe(read)
        assert channel.readline(100)==b'{"command":"report"}\n'
        assert channel.readline(100)==b'{"command":"cancel"}\n'
        assert channel.readline(100)==b""
    finally:
        os.close(read)
        if write is not None:os.close(write)

def test_diagnostic_keeps_type_and_frames_without_exception_values():
    try:
        raise KeyError("SYNTH_CUSTOMER_ID +70000000000 PRIVATE@example.invalid")
    except KeyError as error:
        record=exception_diagnostic(error,"statistics_xlsx")
    text=json.dumps(record)
    assert all(v not in text for v in ("SYNTH_CUSTOMER_ID","+70000000000","PRIVATE@example.invalid"))
    assert record["exception_type"]=="KeyError" and record["traceback"]
    assert decode_diagnostic(text)==record
    record["message"]="PRIVATE@example.invalid"
    with pytest.raises(ValueError):decode_diagnostic(json.dumps(record))

@pytest.mark.skipif(os.name!="nt",reason="Windows destination ACL policy")
@pytest.mark.parametrize("mask,allowed",[(0x1301BF,True),(0x1F01FF,False),(0x40000,False)])
def test_destination_uses_existing_local_parent_policy_but_private_remains_strict(monkeypatch,tmp_path,mask,allowed):
    fact=dict(owner_trusted=True,protected=True,unsupported=False,rules=[
        dict(trusted=True,current=True,allow=True,inherit_only=False,mask=0x1F01FF),
        dict(trusted=False,current=False,allow=True,inherit_only=False,mask=mask)])
    monkeypatch.setattr(storage,"local_volume",lambda p:None)
    monkeypatch.setattr(storage,"inspect_acls",lambda paths:[fact]*len(paths))
    if allowed:
        storage.admit_destination(tmp_path)
    else:
        with pytest.raises(StatisticsIndexError,match="UNSAFE_STORAGE_ACL"):
            storage.admit_destination(tmp_path)
    with pytest.raises(StatisticsIndexError,match="UNSAFE_STORAGE_ACL"):
        check_acl_facts(fact,private=True,root=True,local_parent=True)



from tests.tabs.test_dataset_statistics_tab import app
__all__=["app"]

def test_parent_diagnostic_ignores_private_or_deep_stderr_and_keeps_safe_record(app):
    from Application.statistics_detail_process import DetailExportProcess
    worker=DetailExportProcess()
    worker.diagnostic=dict(stderr_bytes=0,native_error=None)
    worker._collect_stderr(b"PRIVATE@example.invalid +70000000000 SYNTH_CUSTOMER_ID\n")
    worker._collect_stderr(b"["*2000+b"0"+b"]"*2000+b"\n")
    assert "PRIVATE" not in json.dumps(worker.diagnostic)
    try:
        raise KeyError("PRIVATE@example.invalid")
    except KeyError as error:
        record=exception_diagnostic(error,"statistics_html")
    encoded=json.dumps(record).encode()+b"\n"
    worker._collect_stderr(encoded[:30]);worker._collect_stderr(encoded[30:])
    assert worker.diagnostic["exception_type"]=="KeyError"
    assert worker.diagnostic["stage"]=="statistics_html"
    assert "PRIVATE" not in json.dumps(worker.diagnostic)
    worker.deleteLater()

