import copy
from dataclasses import replace
import json
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from Application.model import BPRMF as legacy
from Application.model import mindbox_training_preparation as pipeline
from Application.model.training_data import PreparedDataError
from Application.mindbox.raw_reader import EXPORT_ROOTS, RawExportError
from Application.mindbox.identity import CustomerIdentityError
from Application.mindbox.adapters import AdapterError
from Application.mindbox.records import ProductKey
from Application.interactions import InteractionBuildError
from Application.product_resolution import CatalogError, ProductResolutionError
from scripts.mindbox_training_preparation_smoke import main


STAMP = "2026-01-01T12:00:00Z"


def write_export(directory, name, records):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_part_001.json").write_text(json.dumps({EXPORT_ROOTS[name]: records}), encoding="utf-8")


@pytest.fixture
def data(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No network or training allowed in preparation")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(legacy, "_save_artifacts", forbidden)
    actions = [{"ids": {"mindboxId": f"SECRET_ACTION_{index}"},
                "actionTemplate": {"ids": {"systemName": "ProsmotrProdukta"}},
                "dateTimeUtc": STAMP, "creationDateTimeUtc": STAMP,
                "customer": {"ids": {"mindboxId": "SECRET_A"}},
                "products": [{"ids": {"offline1C": "001234_FULL"}}]} for index in range(9)]
    actions.append(copy.deepcopy(actions[0]))
    actions[-1]["actionTemplate"]["ids"]["systemName"] = "DobavlenieProduktaVSpisokVOperaciiDobavlenie"
    actions[-1]["products"][0]["ids"] = {"offline1C": "001235_FULL"}
    orders = [{"ids": {"mindboxId": "SECRET_ORDER"}, "customer": {"ids": {"mindboxId": user}},
               "firstAction": {"dateTimeUtc": STAMP, "channel": {"ids": {"externalId": "SECRET_CHANNEL"}, "name": "Secret"}},
               "lines": [{"id": "SECRET_LINE", "number": 1, "quantity": 2.5,
                          "basePricePerItem": 10, "priceOfLine": 25,
                          "product": {"ids": {"offline1C": item}, "name": "Secret"},
                          "status": {"ids": {"externalId": "CP"}}}]}
              for user, item in [("SECRET_A", "001236_FULL"), ("SECRET_C", "001235_FULL")]]
    raw = {"actions": actions, "orders": orders, "customer_merges": []}
    dirs = {name: tmp_path / name / "20260101_000000" for name in raw}
    catalog = tmp_path / "Номенклатура.csv"
    catalog.write_text("КодНоменклатуры\n001234\n001235\n001236\n", encoding="utf-8-sig")
    return raw, dirs, catalog


def run(data, cfg=None, diagnose=False):
    raw, dirs, catalog = data
    for name in raw:
        write_export(dirs[name], name, raw[name])
    return pipeline.prepare_training_data_from_mindbox(
        actions_export_dir=dirs["actions"], orders_export_dir=dirs["orders"],
        customer_merges_export_dir=dirs["customer_merges"], catalog_path=catalog,
        train_config=cfg or legacy.TrainConfig(), diagnose=diagnose)


@pytest.mark.parametrize("custom", [False, True])
def test_end_to_end_raw_vs_legacy_equal_date_order_and_custom_weights(data, tmp_path, custom):
    cfg = legacy.TrainConfig(data_dir=str(tmp_path), **(
        dict(w_view_item=0.7, w_favorite=3.5, w_purchase=6, min_user_interactions_for_eval=4) if custom else {}))
    result = run(data, cfg)
    orders = pd.DataFrame({"MindboxID": ["SECRET_A", "SECRET_C"], "КодНоменклатуры": ["001236", "001235"],
                           "Количество": [2.5, 2.5], "Дата": ["2026-01-01"] * 2})
    views = pd.DataFrame({"MindboxID": ["SECRET_A"] * 9, "КодНоменклатуры": ["001234"] * 9,
                          "ТипТовара": ["Номенклатура"] * 9, "Дата": ["2026-01-01"] * 9})
    fav = pd.DataFrame({"MindboxID": ["SECRET_A"], "КодНоменклатуры": ["001235"], "Дата": ["2026-01-01"]})
    for name, frame in (("Заказы", orders), ("Просмотры", views), ("Избранное", fav)):
        frame.to_csv(tmp_path / f"{name}.csv", sep="|", encoding="utf-8-sig", index=False)
    old = legacy.prepare_training_data_from_csv(cfg)
    new = result.prepared_data
    assert result.complete
    assert new.mappings == old.mappings
    for field in ("train_pairs", "train_weights", "eval_users", "eval_items"):
        np.testing.assert_array_equal(getattr(new.splits, field), getattr(old.splits, field))
    assert new.splits.user_pos_train == old.splits.user_pos_train
    assert new.mappings.idx2item[new.splits.eval_items[0]] == "001234"
    assert result.diagnostics.bpr.total_train_weight == pytest.approx(8 * cfg.w_view_item + cfg.w_favorite + 5 * cfg.w_purchase)


def merge(source, target, index=1):
    return {"id": index, "dateTimeUtc": STAMP, "resultingCustomer": {"ids": {"mindboxId": target}},
            "mergedCustomers": [{"ids": {"mindboxId": source}}]}


def test_merges_applied_before_mappings(data):
    data[0]["customer_merges"] = [merge("SECRET_A", "SECRET_B")]
    result = run(data)
    assert "SECRET_A" not in result.prepared_data.mappings.user2idx
    assert result.prepared_data.mappings.idx2user == ["SECRET_B", "SECRET_C"]
    assert result.diagnostics.customer_merges == 1


def test_custom_eval_threshold_is_used(data):
    data[0]["actions"] = data[0]["actions"][:3]
    result = run(data, legacy.TrainConfig(min_user_interactions_for_eval=4))
    assert result.diagnostics.bpr.eval_events == 1
    assert run(data).diagnostics.bpr.eval_events == 0


@pytest.mark.parametrize("problem,error", [("malformed", InteractionBuildError), ("unresolved", ProductResolutionError)])
def test_recoverable_cases_strict_vs_diagnostic(data, problem, error):
    action = data[0]["actions"][0]
    action["products"] = [] if problem == "malformed" else [{"ids": {"offline1C": "999999_FULL"}}]
    with pytest.raises(error):
        run(data)
    result = run(data, diagnose=True)
    assert not result.complete
    assert result.diagnostics.bpr.events_total == 11
    assert result.diagnostics.malformed_actions == int(problem == "malformed")
    assert result.diagnostics.resolution.total.unresolved == int(problem == "unresolved")
    assert "SECRET" not in repr(result) + repr(result.diagnostics)


def test_unsupported_typed_product_recoverable(data, monkeypatch):
    original = pipeline.adapt_action

    def adapt(raw, resolver):
        event = original(raw, resolver)
        return replace(event, products=(ProductKey("SECRET_NAMESPACE", "001234_FULL"),))

    monkeypatch.setattr(pipeline, "adapt_action", adapt)
    with pytest.raises(ProductResolutionError):
        run(data)
    result = run(data, diagnose=True)
    assert not result.complete
    assert result.diagnostics.resolution.total.unsupported_namespace == 10


@pytest.mark.parametrize("cycle", [False, True])
def test_merge_structural_errors_are_fatal(data, cycle):
    data[0]["customer_merges"] = [merge("A", "B"), merge("B" if cycle else "A", "A" if cycle else "C", 2)]
    with pytest.raises(CustomerIdentityError):
        run(data, diagnose=True)


@pytest.mark.parametrize("export", ["actions", "orders", "customer_merges"])
@pytest.mark.parametrize("problem", ["missing", "wrong_root", "malformed"])
def test_raw_errors_fatal(data, export, problem):
    run(data)
    _, dirs, catalog = data
    path = dirs[export] / f"{export}_part_001.json"
    if problem == "missing":
        dirs = {**dirs, export: path.parent / "absent"}
    else:
        path.write_text('{"wrong": []}' if problem == "wrong_root" else '{', encoding="utf-8")
    with pytest.raises(RawExportError):
        pipeline.prepare_training_data_from_mindbox(
            actions_export_dir=dirs["actions"], orders_export_dir=dirs["orders"],
            customer_merges_export_dir=dirs["customer_merges"], catalog_path=catalog,
            train_config=legacy.TrainConfig(), diagnose=True)


def test_missing_catalog_and_empty_invalid_prepared(data):
    raw, dirs, catalog = data
    with pytest.raises(CatalogError):
        run((raw, dirs, catalog.parent / "absent.csv"))
    raw["actions"], raw["orders"] = [], []
    with pytest.raises(PreparedDataError):
        run(data, diagnose=True)


def test_unknown_raw_namespace_is_adapter_structural_error(data):
    data[0]["actions"][0]["products"] = [{"ids": {"unknown": "SECRET"}}]
    with pytest.raises(AdapterError):
        run(data, diagnose=True)


def test_explicit_none_never_selects_latest(data):
    with pytest.raises(ValueError, match="Explicit"):
        pipeline.prepare_training_data_from_mindbox(actions_export_dir=None, orders_export_dir=data[1]["orders"],
            customer_merges_export_dir=data[1]["customer_merges"], catalog_path=data[2], train_config=legacy.TrainConfig())


def test_training_core_integration(data, tmp_path, monkeypatch):
    result = run(data)
    cfg = legacy.TrainConfig(data_dir=str(tmp_path / "no_csv"), use_item_features=False,
                             epochs=1, embedding_dim=4, n_neg=1, batch_size=4, topk=2)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    legacy._set_seed(cfg.seed)
    model, splits = legacy.train_prepared_data(cfg, result.prepared_data, torch.device("cpu"))
    assert splits is result.prepared_data.splits
    assert model is not None
    assert not (tmp_path / "Модель").exists()


@pytest.mark.parametrize("malformed", [False, True])
def test_cli_safe_aggregates_and_complete(data, tmp_path, capsys, malformed):
    if malformed:
        data[0]["actions"][0]["products"] = []
    run(data, diagnose=True)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert main(["--raw-root", str(tmp_path), "--catalog", str(data[2]), "--diagnose"]) == int(malformed)
    output = capsys.readouterr()
    assert f"Complete: {not malformed}" in output.out
    assert "Prepared input validation: OK" in output.out
    for secret in ("SECRET", "001234", "001235", "001236", "_FULL"):
        assert secret not in output.out + output.err
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
