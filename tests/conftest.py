import pytest

from Application.files import files_processing


@pytest.fixture(autouse=True)
def disable_file_processing_ui(monkeypatch):
    """Keep DataFrame characterization tests independent from Qt widgets."""
    for name in (
        "schedule_status_reset",
        "set_status_error",
        "set_status_ok",
        "set_status_processing",
        "show_custom_message",
    ):
        monkeypatch.setattr(files_processing, name, lambda *args, **kwargs: None)
