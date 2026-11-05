"""The public bundle API publishes one safe, complete archive atomically."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from openpyxl import load_workbook
import pytest

from Application import statistics_bundle_export as bundle
from Application import statistics_export as xlsx
from Application.statistics_detail_storage import DetailExportError
from tests.test_statistics_unified_archive import job, aggregate, clean

__all__=["job"]


def test_bundle_fixed_names_crc_same_snapshot_and_excel_design(job):
    result=aggregate(job);before=deepcopy(result)
    output=bundle.export_statistics_bundle(result,job.target)
    assert result==before and output.destination==job.target
    with ZipFile(job.target) as archive:
        assert archive.testzip() is None
        assert archive.namelist()==["Статистика.xlsx","Статистика.html"]
        assert all(not Path(name).is_absolute() and ".." not in name.split("/") for name in archive.namelist())
        book=load_workbook(BytesIO(archive.read("Статистика.xlsx")))
        assert book.sheetnames==list(xlsx.SHEET_NAMES);book.close()
        document=archive.read("Статистика.html").decode()
        assert "<!doctype html>" in document and "plotly-library" in document
        assert 'role="tablist"' in document
    clean(job)


@pytest.mark.parametrize("phase",["xlsx","html","zip","crc","fsync","replace"])
def test_failure_keeps_old_bundle_and_removes_only_owned_temporary_files(job,monkeypatch,phase):
    result=aggregate(job)
    job.target.write_bytes(b"original")
    other=job.target.parent/"unrelated.txt";other.write_text("preserve")
    def fail(*args,**kwargs):
        raise PermissionError("SYNTH_PRIVATE_VALUE")
    if phase=="xlsx":
        monkeypatch.setattr(bundle,"write_statistics_xlsx",fail)
    elif phase=="html":
        monkeypatch.setattr(bundle,"build_statistics_html",fail)
    elif phase=="zip":
        monkeypatch.setattr(bundle._ReportWorkspace,"create_archive",fail)
    elif phase=="crc":
        monkeypatch.setattr(bundle,"_verify_archive",fail)
    elif phase=="fsync":
        monkeypatch.setattr(bundle.os,"fsync",fail)
    else:
        original=bundle.os.replace
        def replace(source,destination):
            if Path(destination)==job.target:
                fail()
            return original(source,destination)
        monkeypatch.setattr(bundle.os,"replace",replace)
    with pytest.raises(DetailExportError) as caught:
        bundle.export_statistics_bundle(result,job.target)
    assert "SYNTH_PRIVATE_VALUE" not in str(caught.value)
    assert job.target.read_bytes()==b"original" and other.read_text()=="preserve"
    clean(job)
