import gzip
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest
import requests

from Application.mindbox import MindboxDownloadError, RawExportStorage
from Application.mindbox import storage as storage_module


URLS = ["https://files.example.test/one", "https://files.example.test/two"]
RAW = b'  {"synthetic": [1, 2], "text": "unchanged"}\r\n'


@pytest.mark.parametrize("compression", ["none", "gzip-file", "content-encoding", "both", "decoded"])
def test_gzip_or_raw_bytes_are_preserved(tmp_path, transport, response, compression):
    payload = gzip.compress(RAW) if compression in ("gzip-file", "content-encoding", "both") else RAW
    headers = {}
    if compression in ("content-encoding", "both"):
        headers["Content-Encoding"] = "gzip"
    if compression == "both":
        payload = gzip.compress(payload)
    item = response(body=payload, headers=headers)
    if compression == "decoded":
        # Как у Response с уже полученным/декодированным .content.
        assert item.content == RAW
        item.headers["Content-Encoding"] = "gzip"
    transport.files.responses.append(item)
    paths = transport.client.download_export("actions", URLS[:1], storage=RawExportStorage(tmp_path))
    assert paths[0].read_bytes() == RAW
    assert re.fullmatch(r"\d{8}_\d{6}", paths[0].parent.name)
    assert list(paths[0].parent.iterdir()) == paths
    assert item.raw.closed


def test_atomic_file_and_directory_publication(tmp_path, monkeypatch):
    original_replace, original_rename = storage_module.os.replace, storage_module.os.rename
    events = []

    def download(url, path):
        assert not list(tmp_path.glob("actions/[0-9]*"))
        assert not list(tmp_path.glob("actions/[0-9]*/*.json"))
        path.write_bytes(RAW)

    def replace(source, destination):
        assert source.suffix == ".tmp"
        assert source.read_bytes() == RAW
        assert not destination.exists()
        events.append("part")
        original_replace(source, destination)

    def rename(source, destination):
        assert events == ["part", "part"]
        assert len(list(source.glob("*.json"))) == 2
        assert not list(source.glob("*.tmp"))
        assert not list(source.glob("*.download"))
        assert not destination.exists()
        events.append("directory")
        original_rename(source, destination)

    monkeypatch.setattr(storage_module.os, "replace", replace)
    monkeypatch.setattr(storage_module.os, "rename", rename)
    paths = RawExportStorage(tmp_path).save_export("actions", URLS, download)
    assert events == ["part", "part", "directory"]
    assert [path.read_bytes() for path in paths] == [RAW, RAW]


def test_second_part_failure_does_not_publish_or_damage_previous_export(tmp_path, transport, response):
    storage = RawExportStorage(tmp_path)
    previous = storage.save_export("orders", URLS[:1], lambda url, path: path.write_bytes(RAW))
    transport.files.responses.extend([response(body=RAW), response(status=403)])
    with pytest.raises(MindboxDownloadError, match="403"):
        transport.client.download_export("orders", URLS, storage=storage)
    assert previous[0].read_bytes() == RAW
    assert list((tmp_path / "orders").iterdir()) == [previous[0].parent]
    assert not list(tmp_path.rglob("*.tmp"))


def test_interrupted_stream_retries_whole_part_without_duplicate_bytes(tmp_path, transport, response):
    item = response(body=RAW)

    def interrupted(chunk_size):
        yield b'{"partial":'
        raise requests.ConnectionError("synthetic interrupted download")

    item.iter_content = interrupted
    transport.files.responses.extend([item, response(body=RAW)])
    paths = transport.client.download_export("actions", URLS[:1], storage=RawExportStorage(tmp_path))
    assert paths[0].read_bytes() == RAW
    assert len(transport.files.calls) == 2
    assert item.raw.closed


@pytest.mark.parametrize("failure", [MindboxDownloadError("synthetic interrupted"), KeyboardInterrupt()])
def test_interrupted_download_cleans_staging(tmp_path, failure):
    def download(url, path):
        path.write_bytes(b"partial")
        raise failure

    with pytest.raises(type(failure)):
        RawExportStorage(tmp_path).save_export("actions", URLS, download)
    assert list((tmp_path / "actions").iterdir()) == []


@pytest.mark.parametrize("raw", [b"", b"\x1f\x8bwrong", gzip.compress(RAW)[:-5]])
def test_empty_or_corrupt_gzip_is_not_published(tmp_path, raw):
    with pytest.raises(MindboxDownloadError):
        RawExportStorage(tmp_path).save_export("actions", URLS[:1], lambda url, path: path.write_bytes(raw))
    assert list((tmp_path / "actions").iterdir()) == []


@pytest.mark.parametrize("operation", ["replace", "rename", "fsync"])
def test_disk_errors_clean_staging(tmp_path, monkeypatch, operation):
    def fail(*args):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(storage_module.os, operation, fail)
    with pytest.raises(MindboxDownloadError, match="OSError"):
        RawExportStorage(tmp_path).save_export("actions", URLS[:1], lambda url, path: path.write_bytes(RAW))
    assert list((tmp_path / "actions").iterdir()) == []


def test_concurrent_exports_in_same_second_do_not_overwrite(tmp_path, monkeypatch):
    class FrozenDateTime:
        @staticmethod
        def now():
            return datetime(2026, 1, 1, 0, 0, 0)

    monkeypatch.setattr(storage_module, "datetime", FrozenDateTime)
    barrier = threading.Barrier(2)

    def save(index):
        def download(url, path):
            path.write_bytes(str(index).encode())
            barrier.wait(timeout=5)
        return RawExportStorage(tmp_path).save_export("actions", URLS[:1], download)[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(save, [1, 2]))
    assert paths[0].parent != paths[1].parent
    assert [path.read_bytes() for path in paths] == [b"1", b"2"]
    assert {path.parent.name for path in paths} == {"20260101_000000", "20260101_000000_001"}
    assert len(list((tmp_path / "actions").iterdir())) == 2


def test_download_network_error_does_not_expose_secret(tmp_path, transport, config):
    transport.files.responses.extend([requests.ConnectionError(config.secret_key) for _ in range(3)])
    with pytest.raises(MindboxDownloadError) as error:
        transport.client.download_export("actions", URLS[:1], storage=RawExportStorage(tmp_path))
    assert config.secret_key not in "".join(traceback.format_exception(error.value))
    assert list((tmp_path / "actions").iterdir()) == []


def test_stream_deadline_prevents_publication(tmp_path, transport, response, clock):
    item = response(body=RAW)

    def slow(chunk_size):
        yield RAW[:2]
        clock.now = 601
        yield RAW[2:]

    item.iter_content = slow
    transport.files.responses.append(item)
    with pytest.raises(MindboxDownloadError, match="время"):
        transport.client.download_export("actions", URLS[:1], storage=RawExportStorage(tmp_path))
    assert list((tmp_path / "actions").iterdir()) == []


def test_invalid_export_name_cannot_escape_storage(tmp_path):
    with pytest.raises(MindboxDownloadError):
        RawExportStorage(tmp_path).save_export("../outside", URLS, lambda *args: None)
    assert list(tmp_path.iterdir()) == []


def test_reservation_cleanup_failure_does_not_report_successful_export_as_failed(tmp_path, monkeypatch, caplog):
    from pathlib import Path
    original = Path.rmdir

    def fail_reservation(path):
        if path.name.endswith(".reserve"):
            raise OSError("synthetic cleanup failure")
        return original(path)

    monkeypatch.setattr(Path, "rmdir", fail_reservation)
    paths = RawExportStorage(tmp_path).save_export("actions", URLS[:1], lambda url, path: path.write_bytes(RAW))
    assert paths[0].read_bytes() == RAW
    assert ".reserve" in caplog.text
    assert not list(tmp_path.rglob(".staging-*"))
