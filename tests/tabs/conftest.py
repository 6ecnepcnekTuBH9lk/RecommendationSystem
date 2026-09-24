import pytest
from PyQt6.QtCore import QSettings


@pytest.fixture(autouse=True)
def window_settings(monkeypatch, tmp_path):
    """Keep GUI tests away from the user's persistent window placement."""
    import main

    settings = QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat)
    # Also isolate legacy settings if window persistence is accidentally restored.
    monkeypatch.setattr(main, "QSettings", lambda *args: settings, raising=False)
    return settings
