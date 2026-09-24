"""Reference-only dispatch: GUI never parses CSV or invokes legacy processors."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Application.tabs import data_processing_tab as ui
from Application.files.reference_import import REFERENCE_TYPES


@pytest.mark.parametrize("kind", list(REFERENCE_TYPES))
def test_reference_dispatches_to_background_controller(tmp_path, monkeypatch, kind):
    source = tmp_path / "reference.csv"
    source.write_text("synthetic", encoding="utf-8")
    controller = SimpleNamespace(start_references=Mock())
    window = SimpleNamespace(reference_paths={kind: source}, mb_controller=controller)
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *args: (str(source), ""))
    monkeypatch.setattr(ui.pd, "read_csv", lambda *a, **kw: pytest.fail("CSV parsing in GUI"))
    for name in ("analyze_orders_full_dataset", "analyze_views_full_dataset", "analyze_favorites_full_dataset"):
        monkeypatch.setattr(ui, name, lambda *a: pytest.fail("Unrelated statistics refresh"))
    ui.load_csv_file(window)
    controller.start_references.assert_called_once_with()
