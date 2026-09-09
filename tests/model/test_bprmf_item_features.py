import numpy as np
import pandas as pd
import pytest

from Application.model import BPRMF


def _synthetic_mappings():
    return BPRMF.Mappings(
        user2idx={"user-1": 0},
        idx2user=["user-1"],
        item2idx={"item-1": 0, "item-2": 1},
        idx2item=["item-1", "item-2"],
    )


def _feature_config(*, enabled=True):
    return BPRMF.TrainConfig(
        use_item_features=enabled,
        item_feature_cols=["ВидНоменклатуры", "Марка"],
        max_item_features=2,
    )


@pytest.mark.parametrize(
    "read_error",
    [
        PermissionError("synthetic unreadable nomenclature"),
        pd.errors.ParserError("synthetic malformed nomenclature"),
    ],
)
def test_existing_unreadable_nomenclature_propagates_error(
    tmp_path, monkeypatch, read_error
):
    (tmp_path / "Номенклатура.csv").write_text(
        "synthetic source exists", encoding="utf-8"
    )

    def fail_read(path):
        raise read_error

    monkeypatch.setattr(BPRMF, "_read_csv_pipe", fail_read)

    with pytest.raises(type(read_error), match=str(read_error)):
        BPRMF._build_item_feature_matrix(
            str(tmp_path),
            _synthetic_mappings(),
            _feature_config(),
        )


def test_missing_nomenclature_keeps_empty_feature_fallback(tmp_path):
    feat2idx, item_features = BPRMF._build_item_feature_matrix(
        str(tmp_path),
        _synthetic_mappings(),
        _feature_config(),
    )

    assert feat2idx == {}
    assert item_features.shape == (2, 2)
    assert np.all(item_features == -1)


def test_disabled_item_features_do_not_read_nomenclature(tmp_path, monkeypatch):
    (tmp_path / "Номенклатура.csv").write_text(
        "synthetic source exists", encoding="utf-8"
    )

    def unexpected_read(path):
        raise AssertionError(f"unexpected read: {path}")

    monkeypatch.setattr(BPRMF, "_read_csv_pipe", unexpected_read)

    feat2idx, item_features = BPRMF._build_item_feature_matrix(
        str(tmp_path),
        _synthetic_mappings(),
        _feature_config(enabled=False),
    )

    assert feat2idx == {}
    assert item_features.shape == (2, 2)
    assert np.all(item_features == -1)


def test_valid_nomenclature_keeps_existing_feature_mapping(tmp_path):
    pd.DataFrame(
        [
            {
                "КодНоменклатуры": "item-1",
                "ВидНоменклатуры": "Рубашка",
                "Марка": "Synthetic A",
            },
            {
                "КодНоменклатуры": "item-2",
                "ВидНоменклатуры": "Брюки",
                "Марка": "Synthetic B",
            },
        ]
    ).to_csv(
        tmp_path / "Номенклатура.csv",
        sep="|",
        index=False,
        encoding="utf-8-sig",
    )

    feat2idx, item_features = BPRMF._build_item_feature_matrix(
        str(tmp_path),
        _synthetic_mappings(),
        _feature_config(),
    )

    assert feat2idx == {
        "ВидНоменклатуры=Рубашка": 0,
        "Марка=Synthetic A": 1,
        "ВидНоменклатуры=Брюки": 2,
        "Марка=Synthetic B": 3,
    }
    assert item_features.tolist() == [[0, 1], [2, 3]]
