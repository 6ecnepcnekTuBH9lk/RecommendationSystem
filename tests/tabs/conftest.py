import pytest


@pytest.fixture(autouse=True)
def isolate_window_settings(window_settings):
    """Request the shared isolation fixture for every GUI test."""
    return window_settings
