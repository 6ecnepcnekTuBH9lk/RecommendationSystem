from types import SimpleNamespace

import pytest

from Application.tabs import train_model_tab


class _Button:
    def __init__(self):
        self.enabled = False

    def setEnabled(self, enabled):
        self.enabled = enabled


class _Log:
    def __init__(self):
        self.messages = []

    def append(self, text):
        self.messages.append(text)


@pytest.mark.parametrize(
    ("exit_code", "exit_status"),
    [
        (1, train_model_tab.QProcess.ExitStatus.NormalExit),
        (2, train_model_tab.QProcess.ExitStatus.NormalExit),
        (0, train_model_tab.QProcess.ExitStatus.CrashExit),
    ],
)
def test_train_finished_never_reports_success_for_failed_process(
    exit_code, exit_status, monkeypatch, tmp_path
):
    window = SimpleNamespace(start_train=_Button(), train_log=_Log(), _training_active=True,
                             train_proc=SimpleNamespace(readAllStandardOutput=lambda: b''),
                             w_purchase=SimpleNamespace(value=lambda: 10.),
                             w_favorite=SimpleNamespace(value=lambda: 2.),
                             w_view_item=SimpleNamespace(value=lambda: .5),
                             epochs_input=SimpleNamespace(value=lambda: 50))
    monkeypatch.setattr(train_model_tab, 'metadata_readiness', lambda *args: (True, 'Synthetic ready', 'a' * 32))
    monkeypatch.setattr(train_model_tab, 'USER_SETTINGS_DIR', tmp_path)
    ok_statuses = []
    error_statuses = []

    monkeypatch.setattr(
        train_model_tab,
        "set_status_ok",
        lambda aboba, text: ok_statuses.append(text),
    )
    monkeypatch.setattr(
        train_model_tab,
        "set_status_error",
        lambda aboba, text: error_statuses.append(text),
    )
    monkeypatch.setattr(
        train_model_tab,
        "schedule_status_reset",
        lambda *args, **kwargs: None,
    )

    train_model_tab._on_train_finished(window, exit_code, exit_status)

    assert window.start_train.enabled is True
    assert ok_statuses == []
    assert error_statuses == [f"Обучение завершилось с ошибкой (код {exit_code})"]
    assert "Обучение успешно завершено." not in window.train_log.messages
