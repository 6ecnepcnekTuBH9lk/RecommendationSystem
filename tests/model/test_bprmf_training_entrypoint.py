import pandas as pd
import pytest

from Application.model import BPRMF


_INVALID_INTERACTION_SCHEMA_CASES = [
    ("Заказы.csv", "MindboxID"),
    ("Заказы.csv", "КодНоменклатуры"),
    ("Просмотры.csv", "MindboxID"),
    ("Просмотры.csv", "КодНоменклатуры"),
    ("Просмотры.csv", "ТипТовара"),
    ("Избранное.csv", "MindboxID"),
    ("Избранное.csv", "КодНоменклатуры"),
]


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


def _training_frames():
    return {
        "Заказы.csv": pd.DataFrame(
            [
                {
                    "MindboxID": "synthetic-user",
                    "КодНоменклатуры": "order-item",
                    "Количество": "1",
                }
            ]
        ),
        "Просмотры.csv": pd.DataFrame(
            [
                {
                    "MindboxID": "synthetic-user",
                    "КодНоменклатуры": "view-item",
                    "ТипТовара": "Номенклатура",
                }
            ]
        ),
        "Избранное.csv": pd.DataFrame(
            [
                {
                    "MindboxID": "synthetic-user",
                    "КодНоменклатуры": "favorite-item",
                }
            ]
        ),
    }


def _write_training_frames(data_dir, frames):
    data_dir.mkdir()
    for filename, frame in frames.items():
        frame.to_csv(
            data_dir / filename,
            sep="|",
            index=False,
            encoding="utf-8-sig",
        )


def _fail_if_called(name):
    def fail(*args, **kwargs):
        pytest.fail(f"{name} must not be called")

    return fail


@pytest.mark.parametrize(
    ("source_name", "expected_weight"),
    [
        ("orders", 30.0),
        ("views", 0.1),
        ("favorites", 2.0),
    ],
)
@pytest.mark.parametrize(
    ("date_case", "date_value"),
    [
        ("malformed", "not-a-date"),
        ("empty", ""),
        ("whitespace", "   "),
        ("nan", pd.NA),
        ("missing_column", None),
    ],
)
def test_training_keeps_interaction_without_valid_optional_date(
    source_name,
    expected_weight,
    date_case,
    date_value,
):
    orders = pd.DataFrame(
        columns=["MindboxID", "КодНоменклатуры", "Количество", "Дата"]
    )
    views = pd.DataFrame(
        columns=["MindboxID", "КодНоменклатуры", "ТипТовара", "Дата"]
    )
    favorites = pd.DataFrame(
        columns=["MindboxID", "КодНоменклатуры", "Дата"]
    )
    source_rows = {
        "orders": {
            "MindboxID": "user",
            "КодНоменклатуры": "item",
            "Количество": "3",
        },
        "views": {
            "MindboxID": "user",
            "КодНоменклатуры": "item",
            "ТипТовара": "Номенклатура",
        },
        "favorites": {
            "MindboxID": "user",
            "КодНоменклатуры": "item",
        },
    }
    if date_case != "missing_column":
        source_rows[source_name]["Дата"] = date_value

    source_frame = pd.DataFrame([source_rows[source_name]])
    if source_name == "orders":
        orders = source_frame
    elif source_name == "views":
        views = source_frame
    else:
        favorites = source_frame

    mappings = BPRMF._build_mappings(orders, views, favorites)
    events = BPRMF._collect_user_item_events(
        orders,
        views,
        favorites,
        mappings,
        BPRMF.TrainConfig(),
    )

    assert len(events) == 1
    assert events.iloc[0]["u_idx"] == mappings.user2idx["user"]
    assert events.iloc[0]["i_idx"] == mappings.item2idx["item"]
    assert events.iloc[0]["w"] == pytest.approx(expected_weight)
    assert pd.isna(events.iloc[0]["ts"])


def test_mixed_dated_and_undated_events_use_latest_dated_event_for_evaluation():
    events = pd.DataFrame(
        {
            "u_idx": [0, 0, 0],
            "i_idx": [0, 1, 2],
            "ts": [
                pd.Timestamp("2026-01-01"),
                pd.NaT,
                pd.Timestamp("2026-02-01"),
            ],
            "w": [1.0, 1.0, 1.0],
        }
    )
    cfg = BPRMF.TrainConfig(min_user_interactions_for_eval=3)

    splits = BPRMF._train_test_split_last_per_user(events, cfg, num_users=1)

    assert splits.eval_users.tolist() == [0]
    assert splits.eval_items.tolist() == [2]
    assert set(map(tuple, splits.train_pairs.tolist())) == {(0, 0), (0, 1)}


def test_all_undated_user_events_are_not_used_for_last_event_evaluation():
    events = pd.DataFrame(
        {
            "u_idx": [0] * 10,
            "i_idx": list(range(10)),
            "ts": [pd.NaT] * 10,
            "w": [1.0] * 10,
        }
    )
    cfg = BPRMF.TrainConfig(min_user_interactions_for_eval=10)

    splits = BPRMF._train_test_split_last_per_user(events, cfg, num_users=1)

    assert splits.eval_users.size == 0
    assert splits.eval_items.size == 0
    assert set(map(tuple, splits.train_pairs.tolist())) == {
        (0, item_index) for item_index in range(10)
    }


def test_all_valid_event_split_keeps_existing_latest_event_behavior():
    events = pd.DataFrame(
        {
            "u_idx": [0, 0, 0],
            "i_idx": [0, 1, 2],
            "ts": [
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-02-01"),
                pd.Timestamp("2026-02-01"),
            ],
            "w": [1.0, 1.0, 1.0],
        }
    )
    cfg = BPRMF.TrainConfig(min_user_interactions_for_eval=3)

    splits = BPRMF._train_test_split_last_per_user(events, cfg, num_users=1)

    assert splits.eval_users.tolist() == [0]
    assert splits.eval_items.tolist() == [2]
    assert set(map(tuple, splits.train_pairs.tolist())) == {(0, 0), (0, 1)}


@pytest.mark.parametrize(
    ("filename", "missing_column"),
    _INVALID_INTERACTION_SCHEMA_CASES,
)
def test_training_rejects_invalid_interaction_schema(
    tmp_path,
    monkeypatch,
    capsys,
    filename,
    missing_column,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF, "train_bprmf", _fail_if_called("train_bprmf"))
    monkeypatch.setattr(BPRMF, "_save_artifacts", _fail_if_called("_save_artifacts"))

    frames = _training_frames()
    frames[filename] = frames[filename].drop(columns=[missing_column])
    data_dir = tmp_path / "training_data"
    _write_training_frames(data_dir, frames)

    result = BPRMF._train_in_this_process(
        BPRMF.TrainConfig(data_dir=str(data_dir))
    )
    diagnostic = capsys.readouterr().out

    assert result is False
    assert filename in diagnostic
    assert missing_column in diagnostic
    assert not (tmp_path / "Модель").exists()


@pytest.mark.parametrize(
    "empty_filename",
    ["Заказы.csv", "Просмотры.csv", "Избранное.csv"],
)
def test_training_accepts_schema_valid_empty_individual_source(
    tmp_path,
    monkeypatch,
    empty_filename,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF.torch.cuda, "is_available", lambda: False)

    frames = _training_frames()
    frames[empty_filename] = frames[empty_filename].iloc[0:0]
    data_dir = tmp_path / "training_data"
    _write_training_frames(data_dir, frames)
    calls = []

    def train(maps, events, cfg, device):
        calls.append(("train", len(events)))
        return object(), object()

    monkeypatch.setattr(BPRMF, "train_bprmf", train)
    monkeypatch.setattr(
        BPRMF,
        "_save_artifacts",
        lambda cfg, maps, model: calls.append(("save", len(maps.idx2item))),
    )

    result = BPRMF._train_in_this_process(
        BPRMF.TrainConfig(data_dir=str(data_dir))
    )

    assert result is True
    assert calls[0][0] == "train"
    assert calls[0][1] > 0
    assert calls[1][0] == "save"


def test_training_orders_without_quantity_use_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF.torch.cuda, "is_available", lambda: False)

    frames = _training_frames()
    frames["Заказы.csv"] = frames["Заказы.csv"].drop(columns=["Количество"])
    frames["Просмотры.csv"] = frames["Просмотры.csv"].iloc[0:0]
    frames["Избранное.csv"] = frames["Избранное.csv"].iloc[0:0]
    data_dir = tmp_path / "training_data"
    _write_training_frames(data_dir, frames)
    captured_events = []
    cfg = BPRMF.TrainConfig(data_dir=str(data_dir), w_purchase=7.5)

    def train(maps, events, train_cfg, device):
        captured_events.append(events.copy())
        return object(), object()

    monkeypatch.setattr(BPRMF, "train_bprmf", train)
    monkeypatch.setattr(BPRMF, "_save_artifacts", lambda *args: None)

    assert BPRMF._train_in_this_process(cfg) is True
    assert len(captured_events) == 1
    assert captured_events[0]["w"].tolist() == [pytest.approx(cfg.w_purchase)]


@pytest.mark.parametrize(
    "read_error",
    [
        PermissionError("synthetic interaction permission error"),
        pd.errors.ParserError("synthetic interaction parser error"),
    ],
    ids=["permission", "parser"],
)
def test_training_interaction_schema_read_error_remains_technical(
    tmp_path,
    monkeypatch,
    read_error,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF, "train_bprmf", _fail_if_called("train_bprmf"))
    monkeypatch.setattr(BPRMF, "_save_artifacts", _fail_if_called("_save_artifacts"))

    data_dir = tmp_path / "training_data"
    _write_training_frames(data_dir, _training_frames())

    def fail_read(*args, **kwargs):
        raise read_error

    monkeypatch.setattr(BPRMF.pd, "read_csv", fail_read)

    with pytest.raises(type(read_error), match=str(read_error)):
        BPRMF._train_in_this_process(BPRMF.TrainConfig(data_dir=str(data_dir)))


def test_training_rejects_zero_byte_interaction_source(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(BPRMF, "_set_seed", lambda seed: None)
    monkeypatch.setattr(BPRMF, "train_bprmf", _fail_if_called("train_bprmf"))
    monkeypatch.setattr(BPRMF, "_save_artifacts", _fail_if_called("_save_artifacts"))

    data_dir = tmp_path / "training_data"
    _write_training_frames(data_dir, _training_frames())
    (data_dir / "Просмотры.csv").write_bytes(b"")

    result = BPRMF._train_in_this_process(
        BPRMF.TrainConfig(data_dir=str(data_dir))
    )
    diagnostic = capsys.readouterr().out

    assert result is False
    assert "Просмотры.csv" in diagnostic
    assert "MindboxID" in diagnostic


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
