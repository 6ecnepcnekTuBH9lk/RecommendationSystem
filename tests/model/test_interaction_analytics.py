from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from Application.model import BPRMF as core
from Application.model.training_data import Mappings
from Application.model.interaction_analytics import AnalyticsCollector, AnalyticsError, analytics_from_checkpoint, load_catalog_kinds


def mappings(users=("secret-user",), items=("secret-item",)):
    return Mappings(dict(zip(users, range(len(users)))), list(users), dict(zip(items, range(len(items)))), list(items))


@pytest.mark.parametrize("days,converted", [(-1, 0), (0, 1), (29, 1), (30, 1), (31, 0)])
@pytest.mark.parametrize("reversed_order", [False, True])
def test_date_window_and_order(days, converted, reversed_order):
    c = AnalyticsCollector()
    view = datetime(2026, 1, 1, 23, 59)
    purchase = datetime(2026, 1, 1) + timedelta(days=days)
    rows = [("VIEW", view), ("PURCHASE", purchase)]
    for kind, date in rows[::-1] if reversed_order else rows:
        c.add("secret-user", "secret-item", kind, date)
    a = c.finalize(mappings(), {})
    assert a.viewers.tolist() == [1]
    assert a.converted.tolist() == [converted]


def test_first_view_repeated_purchases_and_compact_state():
    c = AnalyticsCollector()
    start = datetime(2026, 1, 1)
    for _ in range(100):
        c.add("secret-user", "secret-item", "PURCHASE", start + timedelta(days=40), 2)
    c.add("secret-user", "secret-item", "VIEW", start + timedelta(days=39))
    c.add("secret-user", "secret-item", "VIEW", start)
    assert len(c.purchase_windows[("secret-user", "secret-item")]) == 1
    assert c.finalize(mappings(), {}).converted.tolist() == [0]
    c.add("secret-user", "secret-item", "PURCHASE", start + timedelta(days=20))
    assert c.finalize(mappings(), {}).converted.tolist() == [1]


def test_catalog_last_duplicate_and_no_history_kind_fallback(tmp_path):
    path = tmp_path / "Номенклатура.csv"
    path.write_text("КодНоменклатуры|ВидНоменклатуры\na|old\na|shirt\nb|shirt\n", encoding="utf-8-sig")
    kinds = load_catalog_kinds(path)
    assert kinds["a"] == "shirt"
    c = AnalyticsCollector()
    c.add("u", "a", "VIEW", datetime(2026, 1, 1))
    c.add("u", "a", "PURCHASE", datetime(2026, 1, 2))
    c.add("u", "b", "FAVORITE", None)
    a = c.finalize(mappings(("u",), ("a", "b")), kinds)
    assert a.conversion(["a", "b"], {}) == ({"a": 100., "b": 100.}, 100.)
    assert a.favorite_activity.tolist() == [1]


def test_conversion_csv_parity_activity_and_fallback(tmp_path):
    rows = [("u1", "a", "VIEW", 0, None), ("u1", "a", "VIEW", 2, None),
            ("u1", "a", "FAVORITE", 3, None), ("u1", "a", "PURCHASE", 30, 20),
            ("u1", "a", "PURCHASE", 30, .5), ("u2", "a", "VIEW", 0, None),
            ("u2", "b", "VIEW", 0, None), ("u2", "b", "PURCHASE", -1, None),
            ("u3", "c", "VIEW", 0, None), ("u3", "c", "PURCHASE", 1, 2.5)]
    c = AnalyticsCollector()
    frames = {k: [] for k in ("VIEW", "FAVORITE", "PURCHASE")}
    for user, item, kind, days, qty in rows:
        date = datetime(2026, 1, 2) + timedelta(days=days)
        c.add(user, item, kind, date, qty)
        frames[kind].append({"MindboxID": user, "КодНоменклатуры": item, "Дата": date.strftime("%Y-%m-%d"),
                             "Количество": qty, "ТипТовара": "Номенклатура"})
    for kind, filename in (("VIEW", "Просмотры"), ("FAVORITE", "Избранное"), ("PURCHASE", "Заказы")):
        pd.DataFrame(frames[kind]).to_csv(tmp_path / f"{filename}.csv", sep="|", encoding="utf-8-sig", index=False)
    kinds = {"a": "shirt", "b": "shirt", "c": "", "new": "shirt", "unknown": "other"}
    pd.DataFrame({"КодНоменклатуры": list(kinds), "ВидНоменклатуры": list(kinds.values())}).to_csv(
        tmp_path / "Номенклатура.csv", sep="|", encoding="utf-8-sig", index=False)
    a = c.finalize(mappings(("u1", "u2", "u3"), ("a", "b", "c")), kinds)
    assert a.viewers.tolist() == [2, 1, 1]
    assert a.converted.tolist() == [1, 0, 1]
    assert a.purchase_activity.tolist() == [11, 1, 2.5]
    assert a.favorite_activity.tolist() == [1, 0, 0]
    assert a.view_activity.tolist() == [2, 2, 1]
    expected = core._load_historical_item_conversion(str(tmp_path), kinds)
    actual = a.conversion(["a", "b", "c"], kinds)
    assert actual == expected
    assert actual[0]["new"] == 33.3333 and actual[0]["unknown"] == 50
    cfg = core.TrainConfig(data_dir=str(tmp_path))
    prepared = core.prepare_training_data_from_csv(cfg)
    assert prepared.analytics.conversion(prepared.mappings.idx2item, kinds) == expected
    assert a.rank_users(cfg) == [0, 2, 1]
    # Ties retain mapping order; phone eligibility is supplied independently.
    tied = replace(a, purchase_activity=np.ones(3), favorite_activity=np.zeros(3), view_activity=np.zeros(3))
    assert tied.rank_users(cfg) == [0, 1, 2]
    assert tied.rank_users(cfg, [False, True, True]) == [1, 2]


def test_immutable_roundtrip_safe_contract(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    maps = mappings()
    c = AnalyticsCollector()
    c.add("secret-user", "secret-item", "VIEW", datetime(2026, 1, 1))
    a = c.finalize(maps, {"secret-item": "shirt"})
    core._save_artifacts(core.TrainConfig(use_item_features=False), maps, core.BPRMF(1, 1, 128), analytics=a)
    _, checkpoint = core._load_artifacts()
    loaded = analytics_from_checkpoint(checkpoint)
    np.testing.assert_array_equal(loaded.viewers, a.viewers)
    assert "secret" not in repr(loaded)
    serialized = json.dumps(a.to_checkpoint(), default=lambda x: x.tolist())
    assert "secret" not in serialized
    with pytest.raises(FrozenInstanceError):
        loaded.num_users = 2
    with pytest.raises(ValueError):
        loaded.viewers.setflags(write=True)
    del checkpoint["interaction_analytics"]
    assert analytics_from_checkpoint(checkpoint) is None


@pytest.mark.parametrize("problem", ["partial", "version", "shape", "negative", "nan", "converted", "dimensions", "kind"])
def test_corrupt_analytics_rejected(problem):
    c = AnalyticsCollector()
    a = c.finalize(mappings(), {})
    block = a.to_checkpoint()
    if problem == "partial":
        del block["viewers"]
    elif problem == "version":
        block["version"] = 2
    elif problem == "shape":
        block["viewers"] = np.zeros(2, dtype=np.int64)
    elif problem == "negative":
        block["viewers"] = np.array([-1], dtype=np.int64)
    elif problem == "nan":
        block["purchase_activity"] = np.array([np.nan])
    elif problem == "converted":
        block["converted"] = np.ones(1, dtype=np.int64)
    elif problem == "dimensions":
        block["num_users"] = 2
    else:
        block["item_kinds"] = (123,)
    with pytest.raises(AnalyticsError):
        analytics_from_checkpoint({"num_users": 1, "num_items": 1, "interaction_analytics": block})
