import json
import socket
from types import SimpleNamespace

import pytest

from Application.loading_errors import error_record, format_error, safe_message


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No network"))


def record(capsys):
    lines = capsys.readouterr().out.splitlines()
    return json.loads(next(line[7:] for line in lines if line.startswith("Error: ")))


@pytest.mark.parametrize("value", ["Invalid or unreadable raw JSON part", "Недостаточно истории объединений", "database is locked"])
def test_safe_reasons_preserved(value):
    assert error_record(ValueError(value))["message"] == value
    assert value in format_error(error_record(ValueError(value)))


@pytest.mark.parametrize("value", [
    "'Foo' object has no attribute 'bar'",
    "KeyError: 'ids'",
    "Missing column 'Город'",
    "Invalid value for field 'email'",
    "TypeError: expected 'str', got 'NoneType'",
    "Первое пояснение\nВторое пояснение",
    "Missing fields: ['ids', 'email']",
])
def test_structured_exception_details_preserved(value):
    assert safe_message("  " + value + "  ") == value
    assert error_record(ValueError(value))["message"] == value
    assert (
            format_error(
                {
                    "error_type": "ValueError",
                    "message": value,
                },
                context="Ошибка",
            )
            == "Ошибка → ValueError → " + value
    )


@pytest.mark.parametrize(
    "kind,message,expected",
    [
        (
            "AttributeError",
            "'Foo' object has no attribute 'bar'",
            "AttributeError → 'Foo' object has no attribute 'bar'",
        ),
        (
            "KeyError",
            "'ids'",
            "KeyError → 'ids'",
        ),
        (
            "ValueError",
            "  ",
            "ValueError",
        ),
        (
            None,
            "Missing column 'Город'",
            "Missing column 'Город'",
        ),
        (
            None,
            None,
            "Неизвестная ошибка.",
        ),
        (
            "",
            "",
            "Неизвестная ошибка.",
        ),
    ],
)
def test_error_type_and_message_fallbacks(
    kind,
    message,
    expected,
):
    assert (
        format_error(
            {
                "error_type": kind,
                "message": message,
            },
            context="Ошибка",
        )
        == "Ошибка → " + expected
    )


def test_empty_and_invalid_error_records():
    assert error_record(RuntimeError("  "))["message"] == ""
    for value in ([], None, 3, "private", {"source": [], "message": {}, "since": {}}):
        assert "Ошибка" in format_error(value)
    assert "ValueError" in format_error({"message": "", "error_type": "ValueError"})


@pytest.mark.parametrize("command", ["customers", "interactions"])
@pytest.mark.parametrize("expected", ["manual", "unexpected"])
def test_manual_cli_reports_real_error(monkeypatch, capsys, command, expected):
    from scripts.mindbox_manual_import import main
    from Application.mindbox import canonical_customers, manual_import
    def fail(*args, **kwargs):
        cls = manual_import.ManualImportError if expected == "manual" else RuntimeError
        raise cls("Invalid or unreadable raw JSON part")
    if command == "customers":
        monkeypatch.setattr(canonical_customers, "import_full", fail)
        args = [command, "--customers", "unused"]
    else:
        monkeypatch.setattr(manual_import, "import_interactions", fail)
        args = [command, "--actions", "unused", "--orders", "unused", "--since", "2026-01-01", "--until", "2026-07-01"]
    assert main(args) == 1
    error = record(capsys)
    assert error["source"] == ("customers" if command == "customers" else "actions/orders")
    assert error["message"] == "Invalid or unreadable raw JSON part"
    assert error["error_type"] == ("ManualImportError" if expected == "manual" else "RuntimeError")


def client_stub(monkeypatch):
    from Application import mindbox
    class Client:
        def __init__(self, config):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(mindbox.MindboxConfig, "from_env", lambda *args: SimpleNamespace())
    monkeypatch.setattr(mindbox, "MindboxClient", Client)


@pytest.mark.parametrize("source", ["customers", "actions", "orders"])
@pytest.mark.parametrize("timeout", [False, True])
def test_canonical_cli_keeps_context_type_and_message(monkeypatch, capsys, source, timeout):
    from scripts.mindbox_canonical import main
    from Application.mindbox import canonical_jobs
    from Application.mindbox.exceptions import MindboxExportTimeoutError
    client_stub(monkeypatch)
    def fail(*args, progress, **kwargs):
        progress(source, -1, "2026-09-01")
        raise (MindboxExportTimeoutError if timeout else ValueError)("Экспорт не готов")
    monkeypatch.setattr(canonical_jobs, "create_job", fail)
    args = ["customers" if source == "customers" else "export-daily", "--since", "2026-09-01", "--until", "2026-10-01", "--merge-since", "2025-01-01"]
    assert main(args) == 1
    error = record(capsys)
    assert error["message"] == "Экспорт не готов"
    assert error["source"] == source and error["since"] == "2026-09-01"
    assert error["category"] == ("timeout" if timeout else "failed")


def test_reference_cli_reports_safe_reason(monkeypatch, capsys):
    from scripts.import_reference_csv import main
    from Application.files import reference_import
    def fail(*args, **kwargs):
        raise ValueError("Широта вне допустимого диапазона")
    monkeypatch.setattr(reference_import, "import_reference", fail)
    assert main(["--file", "unused", "--kind", "Координаты городов и погода"]) == 1
    error = record(capsys)
    assert error["message"] == "Широта вне допустимого диапазона" and error["source"] == "reference_csv"


def test_legacy_resume_reports_underlying_error(monkeypatch, capsys, tmp_path):
    from scripts.mindbox_daily_batch import main
    from Application.mindbox import daily_training_batch as daily
    client_stub(monkeypatch)
    state = tmp_path / "state.json"
    def fail(*args, **kwargs):
        try:
            raise TimeoutError("Истекло время ожидания")
        except TimeoutError:
            raise daily.ChunkedBatchError(state) from None
    monkeypatch.setattr(daily, "resume_chunked_training_batch", fail)
    assert main(["resume", "--state", str(state), "--raw-root", str(tmp_path)]) == 1
    error = record(capsys)
    assert error["category"] == "timeout"
    assert error["message"] == "Истекло время ожидания" and error["error_type"] == "TimeoutError"
