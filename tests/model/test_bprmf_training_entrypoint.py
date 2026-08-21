import pandas as pd
import pytest

from Application.model import BPRMF


def _write_training_csvs(data_dir, *, with_interaction):
    data_dir.mkdir()

    if with_interaction:
        orders = pd.DataFrame(
            [
                {
                    "MindboxID": "synthetic-user",
                    "КодНоменклатуры": "synthetic-item",
                    "Количество": "1",
                }
            ]
        )
    else:
        orders = pd.DataFrame(
            columns=["MindboxID", "КодНоменклатуры", "Количество"]
        )

    views = pd.DataFrame(columns=["MindboxID", "КодНоменклатуры", "ТипТовара"])
    favorites = pd.DataFrame(columns=["MindboxID", "КодНоменклатуры"])

    orders.to_csv(data_dir / "Заказы.csv", sep="|", index=False, encoding="utf-8-sig")
    views.to_csv(data_dir / "Просмотры.csv", sep="|", index=False, encoding="utf-8-sig")
    favorites.to_csv(
        data_dir / "Избранное.csv", sep="|", index=False, encoding="utf-8-sig"
    )


def _fail_if_called(name):
    def fail(*args, **kwargs):
        pytest.fail(f"{name} must not be called")

    return fail


def test_training_reports_controlled_failure_when_required_csv_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF, "train_bprmf", _fail_if_called("train_bprmf"))
    monkeypatch.setattr(BPRMF, "_save_artifacts", _fail_if_called("_save_artifacts"))

    data_dir = tmp_path / "training_data"
    data_dir.mkdir()
    (data_dir / "Заказы.csv").write_text("synthetic", encoding="utf-8")
    (data_dir / "Просмотры.csv").write_text("synthetic", encoding="utf-8")

    result = BPRMF._train_in_this_process(BPRMF.TrainConfig(data_dir=str(data_dir)))

    assert result is False
    assert not (tmp_path / "Модель").exists()


def test_training_reports_controlled_failure_when_interactions_are_empty(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF, "train_bprmf", _fail_if_called("train_bprmf"))
    monkeypatch.setattr(BPRMF, "_save_artifacts", _fail_if_called("_save_artifacts"))

    data_dir = tmp_path / "training_data"
    _write_training_csvs(data_dir, with_interaction=False)

    result = BPRMF._train_in_this_process(BPRMF.TrainConfig(data_dir=str(data_dir)))

    assert result is False
    assert not (tmp_path / "Модель").exists()


def test_training_reports_success_only_after_artifacts_are_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF.torch.cuda, "is_available", lambda: False)

    data_dir = tmp_path / "training_data"
    _write_training_csvs(data_dir, with_interaction=True)
    calls = []
    synthetic_model = object()

    def train(maps, events, cfg, device):
        calls.append("train")
        return synthetic_model, object()

    def save(cfg, maps, model):
        assert model is synthetic_model
        calls.append("save")

    monkeypatch.setattr(BPRMF, "train_bprmf", train)
    monkeypatch.setattr(BPRMF, "_save_artifacts", save)

    result = BPRMF._train_in_this_process(BPRMF.TrainConfig(data_dir=str(data_dir)))

    assert result is True
    assert calls == ["train", "save"]


def test_training_propagates_train_bprmf_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF.torch.cuda, "is_available", lambda: False)

    data_dir = tmp_path / "training_data"
    _write_training_csvs(data_dir, with_interaction=True)

    def fail_training(*args, **kwargs):
        raise RuntimeError("synthetic training error")

    monkeypatch.setattr(BPRMF, "train_bprmf", fail_training)
    monkeypatch.setattr(BPRMF, "_save_artifacts", _fail_if_called("_save_artifacts"))

    with pytest.raises(RuntimeError, match="synthetic training error"):
        BPRMF._train_in_this_process(BPRMF.TrainConfig(data_dir=str(data_dir)))


def test_training_propagates_artifact_save_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF.torch.cuda, "is_available", lambda: False)

    data_dir = tmp_path / "training_data"
    _write_training_csvs(data_dir, with_interaction=True)

    monkeypatch.setattr(BPRMF, "train_bprmf", lambda *args: (object(), object()))

    def fail_save(*args, **kwargs):
        raise OSError("synthetic artifact save error")

    monkeypatch.setattr(BPRMF, "_save_artifacts", fail_save)

    with pytest.raises(OSError, match="synthetic artifact save error"):
        BPRMF._train_in_this_process(BPRMF.TrainConfig(data_dir=str(data_dir)))


def test_main_returns_success_code_after_successful_training(monkeypatch):
    monkeypatch.setattr(BPRMF, "_train_in_this_process", lambda cfg: True)

    assert BPRMF.main(["--train"]) == 0


def test_main_returns_controlled_failure_code_when_training_is_not_performed(
    monkeypatch,
):
    monkeypatch.setattr(BPRMF, "_train_in_this_process", lambda cfg: False)

    assert BPRMF.main(["--train"]) == 2


def test_main_does_not_convert_unexpected_exception_to_success(monkeypatch):
    def fail_training(cfg):
        raise RuntimeError("synthetic unexpected error")

    monkeypatch.setattr(BPRMF, "_train_in_this_process", fail_training)

    with pytest.raises(RuntimeError, match="synthetic unexpected error"):
        BPRMF.main(["--train"])


@pytest.mark.parametrize("argv", [[], ["--synthetic-unknown-option"]])
def test_main_preserves_successful_noop_for_non_training_arguments(argv, monkeypatch):
    monkeypatch.setattr(BPRMF, "_train_in_this_process", _fail_if_called("training"))
    monkeypatch.setattr(
        BPRMF,
        "print_recommendations",
        _fail_if_called("print_recommendations"),
    )

    assert BPRMF.main(argv) == 0


def test_main_preserves_recommendation_cli_behavior(monkeypatch):
    calls = []
    monkeypatch.setattr(BPRMF, "_train_in_this_process", _fail_if_called("training"))
    monkeypatch.setattr(
        BPRMF,
        "print_recommendations",
        lambda mindbox_id, k: calls.append((mindbox_id, k)),
    )

    assert BPRMF.main(["--recommend", "synthetic-user", "--k", "7"]) == 0
    assert calls == [("synthetic-user", 7)]
