import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from Application.tabs import train_model_tab


@pytest.fixture(autouse=True)
def training_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(train_model_tab, "INPUT_DATA_DIR", tmp_path / "input_data")
    monkeypatch.setattr(train_model_tab, "USER_SETTINGS_DIR", tmp_path / "user_settings")
    monkeypatch.setattr(train_model_tab, "metadata_readiness", lambda *args: (True, "Synthetic metadata ready", "a" * 32))
    monkeypatch.setattr(train_model_tab, "_confirm_production", lambda *args: True)


class _Button:
    def __init__(self):
        self.enabled = True
        self.enabled_calls = []

    def setEnabled(self, enabled):
        self.enabled_calls.append(enabled)
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


class _Signal:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self, *args):
        for callback in self.callbacks:
            callback(*args)


class _FakeProcess:
    ProcessState = train_model_tab.QProcess.ProcessState
    ProcessError = train_model_tab.QProcess.ProcessError
    ExitStatus = train_model_tab.QProcess.ExitStatus
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
        self.errorOccurred = _Signal()
        self.output = b""
        self.written = []
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

    def state(self):
        return self.ProcessState.NotRunning

    def readAllStandardOutput(self):
        data, self.output = self.output, b""
        return data

    def write(self, data):
        self.written.append(data)

    def deleteLater(self):
        self.deleted = True


def _window_with_training_values():
    return SimpleNamespace(
        start_train=_Button(),
        status_label=_Repaintable(),
        status_icon=_Repaintable(),
        train_log=_Log(),
        w_view_item=_ValueWidget(0.1),
        w_favorite=_ValueWidget(2.0),
        w_purchase=_ValueWidget(10.0),
        epochs_input=_ValueWidget(5),
    )


def _patch_training_ui(monkeypatch):
    errors = []
    _FakeProcess.instances.clear()
    monkeypatch.setattr(train_model_tab, "QProcess", _FakeProcess)
    monkeypatch.setattr(train_model_tab, "set_status_processing", lambda *args: None)
    monkeypatch.setattr(train_model_tab, "schedule_status_reset", lambda *args: None)
    monkeypatch.setattr(
        train_model_tab,
        "set_status_error",
        lambda window, message: errors.append(message),
    )
    return errors


def test_invalid_form_parameters_do_not_start_process(monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)

    window.w_purchase = _ValueWidget(0)

    train_model_tab.start_training_process(window)

    assert window.start_train.enabled is False
    assert errors
    assert window._training_active is False
    assert _FakeProcess.instances == []


def test_store_city_map_uses_existing_in_memory_map(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings_dir = tmp_path / "user_settings"
    settings_dir.mkdir()
    (settings_dir / "filter_settings.json").write_text(
        "{not valid json",
        encoding="utf-8",
    )
    expected = {
        "Магазин 1": "Москва",
        "Магазин 2": "Санкт-Петербург",
    }
    window = SimpleNamespace(_store_city_map=expected)
    open_mock = Mock(side_effect=AssertionError("settings file must not be read"))
    monkeypatch.setattr(train_model_tab, "open", open_mock, raising=False)

    result = train_model_tab._get_store_city_map(window)

    assert result is expected
    open_mock.assert_not_called()


def test_store_city_map_missing_settings_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    window = SimpleNamespace(_store_city_map={})

    result = train_model_tab._get_store_city_map(window)

    assert result == {}


def test_store_city_map_reads_valid_settings_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings_dir = tmp_path / "user_settings"
    settings_dir.mkdir()
    settings_path = settings_dir / "filter_settings.json"
    settings_path.write_text(
        json.dumps(
            {"store_city_map": {"Магазин 1": "Москва", "Магазин 2": "Омск"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    window = SimpleNamespace(_store_city_map={})

    result = train_model_tab._get_store_city_map(window)

    assert result == {"Магазин 1": "Москва", "Магазин 2": "Омск"}


def test_store_city_map_existing_unreadable_settings_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings_dir = tmp_path / "user_settings"
    settings_dir.mkdir()
    settings_path = settings_dir / "filter_settings.json"
    settings_path.write_text("synthetic existing settings", encoding="utf-8")
    window = SimpleNamespace(_store_city_map={})

    def fail_open(*args, **kwargs):
        raise PermissionError("synthetic unreadable settings")

    monkeypatch.setattr(train_model_tab, "open", fail_open, raising=False)

    with pytest.raises(PermissionError, match="synthetic unreadable settings"):
        train_model_tab._get_store_city_map(window)


def test_store_city_map_malformed_json_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings_dir = tmp_path / "user_settings"
    settings_dir.mkdir()
    settings_path = settings_dir / "filter_settings.json"
    settings_path.write_text("{not valid json", encoding="utf-8")
    window = SimpleNamespace(_store_city_map={})

    with pytest.raises(json.JSONDecodeError):
        train_model_tab._get_store_city_map(window)


def test_store_city_map_unexpected_json_structure_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings_dir = tmp_path / "user_settings"
    settings_dir.mkdir()
    settings_path = settings_dir / "filter_settings.json"
    settings_path.write_text("[]", encoding="utf-8")
    window = SimpleNamespace(_store_city_map={})

    with pytest.raises(AttributeError):
        train_model_tab._get_store_city_map(window)


def test_config_write_error_does_not_start_process(tmp_path, monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    monkeypatch.setattr(train_model_tab.json, "dump", Mock(side_effect=OSError("config write error")))
    train_model_tab.start_training_process(window)
    assert window.start_train.enabled and not window._training_active
    assert errors and _FakeProcess.instances == []
    assert not list((tmp_path / "user_settings").glob("train-run-*"))


def test_canonical_start_without_interaction_csv_preserves_form_config(tmp_path, monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    forbidden = Mock(side_effect=AssertionError("CSV pipeline must not run"))
    monkeypatch.setattr(train_model_tab, "_prepare_training_data_dir", forbidden)
    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden)
    # Analysis filters must not become training filters.
    window.analysis_filter = SimpleNamespace(date_from="2020-01-01", stores=("analysis-only",))
    settings = tmp_path / "user_settings"
    settings.mkdir()
    original = b'{"embedding_dim":128,"batch_size":256,"lr":0.0003,"n_neg":10,"weight_decay":0.0,"bpr_reg":0.0005,"seed":42,"topk":10,"min_user_interactions_for_eval":10,"early_stop_metric":"ndcg","early_stop_patience":8,"early_stop_min_delta":0.0005,"early_stop_min_epochs":30,"max_item_features":32,"feature_dropout":0.1,"feature_scale":0.2,"feature_norm":"mean","feat_reg_mult":1.0}'
    (settings / "train_config.json").write_bytes(original)
    monkeypatch.chdir(tmp_path)
    train_model_tab.start_training_process(window)
    process = _FakeProcess.instances[0]
    assert errors == [] and process.started
    assert "gui" in process.arguments and "--train" not in process.arguments
    args = process.arguments
    assert args[args.index("--manifest") + 1] == str(tmp_path / "input_data/MindboxRaw/canonical/training.json")
    assert args[args.index("--raw-root") + 1] == str(tmp_path / "input_data/MindboxRaw")
    assert args[args.index("--catalog") + 1] == str(tmp_path / "input_data/nomenclature.csv")
    assert args[args.index("--device") + 1] == "auto"
    config = json.loads(Path(window._train_config_path).read_text(encoding="utf-8"))
    assert config == {
        "data_dir": str(tmp_path / "input_data"), "w_view_item": 0.1, "w_favorite": 2.0, "w_purchase": 10.0,
        "embedding_dim": 128, "epochs": 5, "batch_size": 256, "lr": 0.0003, "n_neg": 10,
        "weight_decay": 0.0, "bpr_reg": 0.0005, "seed": 42, "topk": 10,
        "min_user_interactions_for_eval": 10, "early_stop_metric": "ndcg", "early_stop_patience": 8,
        "early_stop_min_delta": 0.0005, "early_stop_min_epochs": 30, "max_item_features": 32,
        "feature_dropout": 0.1, "feature_scale": 0.2, "feature_norm": "mean", "feat_reg_mult": 1.0,
    }
    forbidden.assert_not_called()
    assert not list(tmp_path.rglob("*.csv"))
    assert not (tmp_path / "filtered_data").exists()
    assert (settings / "train_config.json").read_bytes() == original


def _preflight_event(level):
    return {
        "stage": "preflight", "batch_id": "a" * 32, "error_code": "QUALITY_BLOCK" if level == "BLOCK" else None,
        "dataset": {"users": 2, "items": 3}, "interaction_window": {"since": "2026-01-01", "until": "2026-01-02"},
        "quality": {"level": level, "training_allowed": level != "BLOCK", "metrics": {"malformed_rate": 0.2},
                    "issues": [{"message": "Исключены действия без товара", "code": "MAPPED_ACTION_WITHOUT_PRODUCT",
                                "count": 1, "rate": 0.2, "breakdown": {"ProsmotrProdukta": 1}}]},
    }


def _send_event(window, event):
    from scripts.mindbox_production_train import EVENT_PREFIX
    window.train_proc.output = (EVENT_PREFIX + json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
    window.train_proc.readyReadStandardOutput.emit()


@pytest.mark.parametrize("level", ["PASS", "BLOCK"])
def test_preflight_pass_block_diagnostics_without_dialog(monkeypatch, level):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    dialog = Mock(side_effect=AssertionError("Only WARN asks for confirmation"))
    monkeypatch.setattr(train_model_tab.QMessageBox, "question", dialog)
    train_model_tab.start_training_process(window)
    event = _preflight_event(level)
    _send_event(window, event)
    assert window._train_preflight_result == event
    assert window.train_proc.written == []
    assert bool(errors) == (level == "BLOCK")
    assert any("malformed_rate" in message and "ProsmotrProdukta" in message for message in window.train_log.messages)
    dialog.assert_not_called()


@pytest.mark.parametrize("accepted", [True, False])
def test_warn_requires_explicit_confirmation(monkeypatch, accepted):
    window = _window_with_training_values()
    _patch_training_ui(monkeypatch)
    yes, no = train_model_tab.QMessageBox.StandardButton.Yes, train_model_tab.QMessageBox.StandardButton.No
    dialog = Mock(return_value=yes if accepted else no)
    monkeypatch.setattr(train_model_tab.QMessageBox, "question", dialog)
    train_model_tab.start_training_process(window)
    _send_event(window, _preflight_event("WARN"))
    assert window.train_proc.written == [b"YES\n" if accepted else b"NO\n"]
    assert dialog.call_args.args[-1] == no
    assert "MAPPED_ACTION_WITHOUT_PRODUCT" in dialog.call_args.args[2]
    assert window._training_active


def test_output_handles_split_json_and_utf8(monkeypatch):
    window = _window_with_training_values()
    _patch_training_ui(monkeypatch)
    monkeypatch.setattr(train_model_tab.QMessageBox, "question", lambda *args: train_model_tab.QMessageBox.StandardButton.No)
    train_model_tab.start_training_process(window)
    from scripts.mindbox_production_train import EVENT_PREFIX
    event = _preflight_event("WARN")
    data = (EVENT_PREFIX + json.dumps(event, ensure_ascii=False) + "\r\n").encode("utf-8")
    # Feed bytes separately, including inside multibyte Cyrillic characters.
    for byte in data:
        window.train_proc.output = bytes([byte])
        window.train_proc.readyReadStandardOutput.emit()
    assert window._train_preflight_result == event
    assert window.train_proc.written == [b"NO\n"]
    assert window._train_output_buffer == b""


def test_failed_start_restores_ui_and_cleans_config(tmp_path, monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    train_model_tab.start_training_process(window)
    window.train_proc.errorOccurred.emit(_FakeProcess.ProcessError.FailedToStart)
    assert window.start_train.enabled and not window._training_active
    assert errors == ["Не удалось запустить обучение"]
    assert not list((tmp_path / "user_settings").glob("train-run-*"))


def test_repeated_start_is_blocked_while_preflight_or_warn_pending(monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    train_model_tab.start_training_process(window)
    train_model_tab.start_training_process(window)
    assert len(_FakeProcess.instances) == 1
    assert errors == ["Обучение уже запущено"]


@pytest.mark.parametrize("published", [True, False])
def test_zero_exit_requires_verified_publication(tmp_path, monkeypatch, published):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    successes = []
    monkeypatch.setattr(train_model_tab, "set_status_ok", lambda window, message: successes.append(message))
    train_model_tab.start_training_process(window)
    if published:
        # Completion handler must drain output even if readyRead has not arrived.
        from scripts.mindbox_production_train import EVENT_PREFIX
        event = {"stage": "finished", "published": True, "postpublish_validation": True,
                 "published_generation": "b" * 32, "error_code": None}
        window.train_proc.output = (EVENT_PREFIX + json.dumps(event) + "\n").encode()
    window.train_proc.finished.emit(0, _FakeProcess.ExitStatus.NormalExit)
    assert bool(successes) == published and bool(errors) != published
    assert window.start_train.enabled and not window._training_active
    assert not list((tmp_path / "user_settings").glob("train-run-*"))
    old_process = window.train_proc
    train_model_tab.start_training_process(window)
    assert old_process.deleted and len(_FakeProcess.instances) == 2


def test_warn_declined_reports_cancellation(monkeypatch):
    window = _window_with_training_values()
    errors = _patch_training_ui(monkeypatch)
    statuses = []
    monkeypatch.setattr(train_model_tab, "set_status_ok", lambda window, message: statuses.append(message))
    train_model_tab.start_training_process(window)
    window.train_proc.finished.emit(130, _FakeProcess.ExitStatus.NormalExit)
    assert statuses == ["Обучение отменено"] and not errors
    assert window.start_train.enabled and not window._training_active


def test_process_exit_while_warn_dialog_open_never_sends_confirmation(monkeypatch):
    window = _window_with_training_values()
    _patch_training_ui(monkeypatch)
    def process_died(*args):
        window.train_proc.finished.emit(1, _FakeProcess.ExitStatus.CrashExit)
        return train_model_tab.QMessageBox.StandardButton.Yes
    monkeypatch.setattr(train_model_tab.QMessageBox, "question", process_died)
    train_model_tab.start_training_process(window)
    _send_event(window, _preflight_event("WARN"))
    assert window.train_proc.written == []
    assert window.start_train.enabled and not window._training_active


@pytest.mark.parametrize("level,accept", [("PASS", True), ("WARN", True), ("WARN", False), ("BLOCK", True)])
def test_real_qprocess_canonical_training(tmp_path, monkeypatch, level, accept):
    """Real Qt signals/stdin and real training, with all writes in the test directory."""
    from datetime import datetime, timedelta, timezone
    from PyQt6.QtCore import QEventLoop, QTimer
    from PyQt6.QtWidgets import QApplication, QWidget
    from Application.mindbox import canonical_storage as store
    from Application.model import BPRMF as core
    import torch

    app = QApplication.instance() or QApplication([])
    raw = tmp_path / "input_data/MindboxRaw"
    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    until = since + timedelta(days=1)
    actions = [
        {"ids": {"mindboxId": str(index)}, "customer": {"ids": {"mindboxId": user}},
         "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}},
         "dateTimeUtc": since.isoformat(), "creationDateTimeUtc": since.isoformat(),
         "products": [{"ids": {"offline1C": item}}]}
        for index, (user, item) in enumerate([("A", "123456"), ("B", "123457"), ("A", "123458")])
    ]
    if level != "PASS":
        actions.append({**actions[0], "products": [] if level == "WARN" else [{"ids": {"offline1C": "999999"}}]})
    with store.storage_lock(raw):
        for name, key, records in (("customer_merges", "customerMerges", []),
                                   ("actions", "customerActions", actions), ("orders", "orders", [])):
            directory = raw / "staged" / name
            directory.mkdir(parents=True)
            (directory / f"{name}_part_001.json").write_text(json.dumps({key: records}), encoding="utf-8")
            store.publish(raw, name, since, until, directory)
    (tmp_path / "input_data/nomenclature.csv").write_text(
        "КодНоменклатуры|Коллекция|СезонНоски\n123456|collection-a|winter\n"
        "123457|collection-b|summer\n123458|collection-a|summer\n", encoding="utf-8-sig")
    model_root = tmp_path / "model"
    run = model_root / "runs" / ("1" * 32)
    run.mkdir(parents=True)
    (run / "bprmf.pt").write_bytes(b"previous checkpoint")
    (model_root / "current.json").write_text(json.dumps({"generation": "1" * 32}), encoding="utf-8")
    old_current = (model_root / "current.json").read_bytes()
    # Test-only CLI wrapper redirects the production output root, never the GUI command's public options.
    script = tmp_path / "scripts/mindbox_production_train.py"
    script.parent.mkdir()
    script.write_text(
        f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(train_model_tab.PROJECT_ROOT)!r})\n"
        "from Application.model import mindbox_production_training as production\n"
        "from Application.model import BPRMF as core\n"
        "def forbidden(*args, **kwargs):\n    raise AssertionError('GUI must not use CSV preparation')\n"
        "core.prepare_training_data_from_csv = forbidden\n"
        f"production.MODEL_ROOT = Path({str(model_root)!r})\n"
        f"production.REPORT_ROOT = Path({str(tmp_path / 'reports')!r})\n"
        "from scripts.mindbox_production_train import main\nraise SystemExit(main())\n", encoding="utf-8")
    monkeypatch.setattr(train_model_tab, "PROJECT_ROOT", tmp_path)
    errors, successes = [], []
    monkeypatch.setattr(train_model_tab, "set_status_processing", lambda *args: None)
    monkeypatch.setattr(train_model_tab, "schedule_status_reset", lambda *args: None)
    monkeypatch.setattr(train_model_tab, "set_status_error", lambda w, message: errors.append(message))
    monkeypatch.setattr(train_model_tab, "set_status_ok", lambda w, message: successes.append(message))
    monkeypatch.setattr(train_model_tab.QMessageBox, "question", lambda *args:
                        train_model_tab.QMessageBox.StandardButton.Yes if accept else train_model_tab.QMessageBox.StandardButton.No)
    window = QWidget()
    for name, value in vars(_window_with_training_values()).items():
        setattr(window, name, value)
    window.epochs_input = _ValueWidget(1)
    # Hidden production configuration now belongs to the existing config layer,
    # rather than widgets removed by TRAIN-UI-01.
    settings = tmp_path / 'user_settings'
    settings.mkdir(exist_ok=True)
    (settings / 'train_config.json').write_text(json.dumps({
        'embedding_dim': 4, 'batch_size': 2, 'n_neg': 1, 'topk': 2}), encoding='utf-8')
    loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    pulse = QTimer()
    ticks = []
    pulse.timeout.connect(lambda: ticks.append(True))
    pulse.start(10)
    try:
        train_model_tab.start_training_process(window)
        window.train_proc.finished.connect(loop.quit)
        window.train_proc.errorOccurred.connect(loop.quit)
        timeout.start(30000)
        loop.exec()
        assert not window._training_active, window.train_log.messages
        assert ticks and window.start_train.enabled == (level != 'BLOCK')
        assert window._train_preflight_result["quality"]["level"] == level
        if level == "BLOCK" or not accept:
            assert (model_root / "current.json").read_bytes() == old_current
            assert window._train_publication_result is None
            assert bool(errors) == (level == "BLOCK")
            if not accept:
                assert successes == ["Обучение отменено"]
        else:
            assert successes == ["Обучение завершено"] and not errors
            maps, checkpoint = core._load_artifacts(str(model_root))
            core._build_model_from_ckpt(checkpoint, torch.device("cpu"))
            assert (len(maps["idx2user"]), len(maps["idx2item"])) == (2, 3)
            assert window._train_publication_result["training_metrics"]["epochs_completed"] == 1
            features = checkpoint["feat2idx"]
            assert any("Коллекция" in key for key in features) and any("СезонНоски" in key for key in features)
        assert (run / "bprmf.pt").read_bytes() == b"previous checkpoint"
        assert not list((tmp_path / "user_settings").glob("train-run-*"))
        assert not any(p.name in ("orders.csv", "views.csv", "favorites.csv") for p in tmp_path.rglob("*.csv"))
    finally:
        pulse.stop()
        timeout.stop()
        if window.train_proc.state() != train_model_tab.QProcess.ProcessState.NotRunning:
            window.train_proc.kill()
            window.train_proc.waitForFinished(5000)
        window.close()
        window.deleteLater()
        app.processEvents()
