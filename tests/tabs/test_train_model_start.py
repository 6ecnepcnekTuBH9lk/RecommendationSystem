import json
from types import SimpleNamespace

from Application.tabs import train_model_tab


class _Button:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled):
        self.enabled = enabled


class _Repaintable:
    def repaint(self):
        pass


class _Log:
    def __init__(self):
        self.messages = []

    def clear(self):
        self.messages.clear()

    def append(self, text):
        self.messages.append(text)


class _ValueWidget:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value


class _TextWidget:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text

    def currentText(self):
        return self._text


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class _FakeProcess:
    class ProcessChannelMode:
        MergedChannels = object()

    instances = []

    def __init__(self, parent):
        self.parent = parent
        self.program = None
        self.arguments = None
        self.working_directory = None
        self.started = False
        self.readyReadStandardOutput = _Signal()
        self.finished = _Signal()
        self.__class__.instances.append(self)

    def setProgram(self, program):
        self.program = program

    def setArguments(self, arguments):
        self.arguments = arguments

    def setProcessChannelMode(self, mode):
        self.channel_mode = mode

    def setWorkingDirectory(self, directory):
        self.working_directory = directory

    def start(self):
        self.started = True

    def waitForStarted(self, timeout):
        return self.started


def _window_with_training_values():
    return SimpleNamespace(
        start_train=_Button(),
        status_label=_Repaintable(),
        status_icon=_Repaintable(),
        train_log=_Log(),
        w_view_item=_ValueWidget(0.1),
        w_favorite=_ValueWidget(2.0),
        w_purchase=_ValueWidget(10.0),
        embedding_dim_input=_TextWidget("128"),
        epochs_input=_ValueWidget(5),
        batch_size_input=_TextWidget("256"),
        lr_input=_ValueWidget(0.0003),
        n_neg=_ValueWidget(10),
        weight_decay_input=_ValueWidget(0.0),
        bpr_reg_input=_ValueWidget(0.0005),
        seed_input=_ValueWidget(42),
        top_rec=_ValueWidget(10),
        min_user_interactions_for_eval=_ValueWidget(10),
        early_stop_metric=_TextWidget("NDCG"),
        early_stop_patience=_ValueWidget(8),
        early_stop_min_delta=_ValueWidget(0.0005),
        early_stop_min_epochs=_ValueWidget(30),
        max_item_features_input=_ValueWidget(32),
        feature_dropout_input=_ValueWidget(0.1),
        feature_scale_input=_ValueWidget(0.2),
        feature_norm_input=_TextWidget("MEAN"),
        feat_reg_mult_input=_ValueWidget(1.0),
    )


def _patch_training_ui(monkeypatch):
    errors = []
    _FakeProcess.instances.clear()
    monkeypatch.setattr(train_model_tab, "QProcess", _FakeProcess)
    monkeypatch.setattr(
        train_model_tab,
        "QApplication",
        SimpleNamespace(processEvents=lambda: None),
    )
    monkeypatch.setattr(train_model_tab, "set_status_processing", lambda *args: None)
    monkeypatch.setattr(train_model_tab, "schedule_status_reset", lambda *args: None)
    monkeypatch.setattr(
        train_model_tab,
        "set_status_error",
        lambda window, message: errors.append(message),
    )
    monkeypatch.setattr(train_model_tab, "update_filter_summary", lambda window: None)
    return errors


def test_training_data_preparation_error_does_not_start_process(monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)

    def fail_preparation(window):
        raise RuntimeError("synthetic data preparation error")

    monkeypatch.setattr(
        train_model_tab,
        "_prepare_training_data_dir",
        fail_preparation,
    )

    train_model_tab.start_training_process(window)

    assert window.start_train.enabled is True
    assert errors
    assert any("synthetic data preparation error" in msg for msg in window.train_log.messages)
    assert len(_FakeProcess.instances) == 1
    assert _FakeProcess.instances[0].started is False
    assert _FakeProcess.instances[0].arguments is None


def test_config_write_error_does_not_start_process(tmp_path, monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        train_model_tab,
        "_prepare_training_data_dir",
        lambda window: "synthetic-data",
    )

    def fail_open(*args, **kwargs):
        raise OSError("synthetic config write error")

    monkeypatch.setattr(train_model_tab, "open", fail_open, raising=False)

    train_model_tab.start_training_process(window)

    assert window.start_train.enabled is True
    assert errors
    assert any("synthetic config write error" in msg for msg in window.train_log.messages)
    assert len(_FakeProcess.instances) == 1
    assert _FakeProcess.instances[0].started is False
    assert _FakeProcess.instances[0].arguments is None


def test_successful_config_preparation_always_starts_with_config(tmp_path, monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        train_model_tab,
        "_prepare_training_data_dir",
        lambda window: "synthetic-data",
    )

    train_model_tab.start_training_process(window)

    process = _FakeProcess.instances[0]
    assert errors == []
    assert process.started is True
    assert process.arguments[2] == "--train"
    assert process.arguments.count("--config") == 1
    assert len(process.arguments) == 5
    config_path = process.arguments[process.arguments.index("--config") + 1]
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    assert config["data_dir"] == "synthetic-data"
