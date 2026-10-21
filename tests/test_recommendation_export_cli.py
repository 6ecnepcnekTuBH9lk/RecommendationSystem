"""Regression for native runtime initialization before a blocking stdin reader."""

import builtins
import io
import json

import pytest

from scripts import export_recommendations as cli
from Application import recommendation_export_batch as batch
from Application.recommendation_export_records import ExportCancelled


@pytest.mark.parametrize("mode", [[], ["--full"]])
def test_control_reader_starts_after_native_imports_and_queued_cancel_is_retained(
    monkeypatch, tmp_path, capsys, mode
):
    events = []
    original = builtins.__import__

    def imported(name, *args, **kwargs):
        if name in (
            "Application.recommendation_export_batch",
            "Application.recommendation_export",
        ):
            events.append(name)
        return original(name, *args, **kwargs)

    class Thread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            events.append("reader")
            self.target()

    class Guard:
        def __init__(self, *a, **kw):
            pass

        def poll(self, **kw):
            pass

    class Session:
        def __init__(self, **kwargs):
            assert kwargs["cancelled"]()
            assert kwargs["limit"] == (None if mode else 50)

        def __enter__(self):
            raise ExportCancelled()

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(builtins, "__import__", imported)
    monkeypatch.setattr(cli, "Thread", Thread)
    monkeypatch.setattr(cli, "ResourceGuard", Guard)
    monkeypatch.setattr(batch, "RecommendationBatchSession", Session)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("cancel\n"))
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["export", "--destination", str(tmp_path / "Recommendations.zip"), *mode],
    )
    assert cli.main() == 2
    assert events == [
        "Application.recommendation_export_batch",
        "Application.recommendation_export",
        "reader",
    ]
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {"type": "cancelled"}
    assert not (tmp_path / "Recommendations.zip").exists()
