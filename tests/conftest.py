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


@pytest.fixture
def window_settings(monkeypatch, tmp_path):
    """Keep GUI tests away from the user's persistent window placement."""
    from PyQt6.QtCore import QSettings
    import main
    from Application.tabs import dataset_statistics_tab, analysis_filter_tab
    from Application.analysis_filter import AnalysisFilter

    monkeypatch.setattr(analysis_filter_tab, "SETTINGS_PATH", tmp_path / "analysis_filter.json")
    monkeypatch.setattr(analysis_filter_tab.mapping, "SETTINGS_PATH", tmp_path / "store_city_mapping.json")
    monkeypatch.setattr(analysis_filter_tab.mapping, "load_sources", lambda: analysis_filter_tab.mapping.MappingSources())
    monkeypatch.setattr(dataset_statistics_tab, "load_filter", AnalysisFilter)
    def no_sources():
        raise ValueError("Synthetic sources not configured")
    monkeypatch.setattr(analysis_filter_tab, "load_analysis_options", no_sources)

    monkeypatch.setattr(dataset_statistics_tab, "CACHE_PATH", tmp_path / "dataset_statistics.json")

    settings = QSettings(str(tmp_path / "window.ini"), QSettings.Format.IniFormat)
    # Also isolate legacy settings if window persistence is accidentally restored.
    monkeypatch.setattr(main, "QSettings", lambda *args: settings, raising=False)
    return settings
