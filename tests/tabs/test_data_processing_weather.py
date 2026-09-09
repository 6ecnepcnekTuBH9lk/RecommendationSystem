from types import SimpleNamespace

import pandas as pd
import pytest

from Application.tabs import data_processing_tab


class _TextField:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def _window():
    return SimpleNamespace(
        filter_date_from=_TextField("01.01.2025"),
        filter_date_to=_TextField("02.01.2025"),
    )


@pytest.mark.parametrize(
    ("successful", "failed", "empty", "expected_fragment"),
    [
        (2, 0, 0, "успешно"),
        (1, 1, 0, "частично"),
        (0, 0, 2, "не получены"),
        (0, 2, 0, "не получены"),
    ],
    ids=["success", "partial", "empty", "failed"],
)
def test_manual_weather_update_reports_actual_api_outcome(
    tmp_path,
    monkeypatch,
    successful,
    failed,
    empty,
    expected_fragment,
):
    monkeypatch.chdir(tmp_path)
    input_dir = tmp_path / "ВходныеДанные"
    input_dir.mkdir()
    (input_dir / "КоординатыГородов.csv").write_text("existing", encoding="utf-8")
    result = pd.DataFrame()
    result.attrs.update(
        weather_total_cities=2,
        weather_successful_cities=successful,
        weather_failed_cities=failed,
        weather_empty_cities=empty,
    )
    monkeypatch.setattr(
        data_processing_tab,
        "generate_weather_for_saved_coordinates",
        lambda *args, **kwargs: result,
    )
    ok_messages = []
    degraded_messages = []
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_ok",
        lambda window, message: ok_messages.append(message),
    )
    monkeypatch.setattr(
        data_processing_tab,
        "set_status_error",
        lambda window, message: degraded_messages.append(message),
    )
    monkeypatch.setattr(
        data_processing_tab.QApplication,
        "processEvents",
        lambda: None,
    )

    data_processing_tab._maybe_update_weather(_window())

    messages = ok_messages + degraded_messages
    assert len(messages) == 1
    assert expected_fragment in messages[0].lower()
    if failed or empty:
        assert ok_messages == []
    else:
        assert degraded_messages == []
