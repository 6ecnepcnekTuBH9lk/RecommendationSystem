from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from Application.analysis_filter import AnalysisFilter, AnalysisOptions
from Application import analysis_filter_settings as settings
from Application import statistics_cache as cache
from Application.dataset_statistics import load_analysis_options
import test_dataset_statistics as baseline
from test_dataset_statistics import calculate, action, order, put_entry, save, DEFAULT_SELECTION

dataset = baseline.dataset


@pytest.mark.parametrize("selection", [AnalysisFilter(), AnalysisFilter("2025-03-01", "2025-05-31"),
                                      AnalysisFilter(nomenclature_types=(" B ", "A", "A", None), collections=("Лето",))])
def test_settings_roundtrip_and_atomic(selection, tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    calls = []
    replace = settings.os.replace
    def replacement(source, target):
        calls.append("replace")
        assert json.loads(Path(source).read_text(encoding="utf-8"))
        replace(source, target)
    fsync = settings.os.fsync
    def sync(fd):
        calls.append("fsync")
        fsync(fd)
    monkeypatch.setattr(settings.os, "replace", replacement)
    monkeypatch.setattr(settings.os, "fsync", sync)
    settings.save_filter(selection, path)
    assert settings.load_filter(path) == selection
    assert calls == ["fsync", "replace"]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("payload", ["bad", "[]", '{"schema_version": true}',
                                     '{"schema_version":2}', '{"schema_version":1,"analysis_filter":{}}'])
def test_bad_settings_do_not_overwrite(tmp_path, payload, caplog):
    path = tmp_path / "settings.json"
    path.write_text(payload, encoding="utf-8")
    assert settings.load_filter(path) == AnalysisFilter()
    assert path.read_text(encoding="utf-8") == payload
    assert caplog.records


def test_canonicalization_reconciliation_and_zero_selection():
    options = AnalysisOptions("2025-03-01", "2025-05-31", ("A", "B", None), ("Лето", "Зима"))
    assert options.normalize(AnalysisFilter("2025-03-01", "2025-05-31", (None, " B ", "A", "A"))) == AnalysisFilter()
    selection = options.normalize(AnalysisFilter("2024-01-01", "2026-01-01", ("gone",), ("Лето", "gone")), reconcile=True)
    assert selection == AnalysisFilter(collections=("Лето",))
    assert options.normalize(AnalysisFilter("2024-01-01", "2024-02-01"), reconcile=True).end_date == "2025-03-01"
    with pytest.raises(ValueError):
        AnalysisFilter(collections=())
    with pytest.raises(ValueError):
        AnalysisFilter("2025-05-01", "2025-03-01")
    assert AnalysisFilter(nomenclature_types=("",)).nomenclature_types == (None,)


def test_summary_and_utc_boundaries():
    assert AnalysisFilter().summary() == "Отбор не установлен"
    selection = AnalysisFilter("2025-03-01", "2025-05-31", ("A", "B", "C", "D"), (None,))
    assert "4 значений" in selection.summary() and "сезон: Не указано" in selection.summary()
    for stamp, expected in (("2025-02-28T23:59:59", False), ("2025-03-01T00:00:00", True),
                             ("2025-05-31T23:59:59", True), ("2025-06-01T00:00:00", False)):
        assert selection.contains(datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)) is expected


def reference(dataset):
    _, path, _ = dataset
    path.write_text("КодНоменклатуры|Номенклатура|ВидНоменклатуры|Коллекция|СезонНоски\n"
                    "000001|A|Рубашки|Весна-лето|Другое значение\n"
                    "000002|B|Брюки|Весна-лето|Другое значение\n"
                    "000003|C|Рубашки|Осень-зима|Другое значение\n"
                    "000004|D|||\n", encoding="utf-8-sig")


def sources(dataset, actions, orders, since="2025-01-01T00:00:00+00:00", until="2027-01-01T00:00:00+00:00"):
    root, _, data = dataset
    entries = {name: put_entry(root, name, records, 91, since, until, kind="MANUAL")
               for name, records in (("actions", actions), ("orders", orders))}
    data["actions"], data["orders"] = {}, {}
    data["manual_interactions"] = {"since": since, "until": until, "updated": entries["actions"]["updated"], **entries}
    save(root, data)


def test_actions_or_and_source_ambiguity_and_cohort(dataset):
    reference(dataset)
    view = DEFAULT_SELECTION.view_action_system_names[0]
    sources(dataset, [action(view, ["000001-a", "000003-a"], price=100, available=True),
                      action(view, ["000002-a"], "other"), action(view, ["000004-a"], "inactive")], [])
    selection = AnalysisFilter(nomenclature_types=("Рубашки", "Брюки"), collections=("Весна-лето",))
    result = calculate(dataset, analysis_filter=selection)
    assert result.view_interactions == 2 and result.customers == 2 and result.interaction_users == 2
    assert result.view_parameter_actions == 1
    assert dict(result.diagnostics)["ambiguous_product_view_actions"] == 1
    assert {r[0] for r in result.top_viewed_products} == {"000001", "000002"}
    assert result.product_season_statistics == (("Весна-лето", 2, 2, 0, 0, "0"),)
    assert result.analysis_filter == selection
    cache.validate_result(asdict(result))


@pytest.mark.parametrize("selected, expected", [((None,), 1), (("Рубашки",), 0)])
def test_missing_and_unresolved(dataset, selected, expected):
    reference(dataset)
    view = DEFAULT_SELECTION.view_action_system_names[0]
    sources(dataset, [action(view, ["000004-a", "999999-a"])], [])
    baseline = calculate(dataset)
    assert baseline.view_interactions == 2 and baseline.unresolved_interactions == 1
    result = calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=selected))
    assert result.view_interactions == expected and result.unresolved_interactions == 0
    cache.validate_result(asdict(result))


@pytest.mark.parametrize("status, purchases, nonpurchases, money", [("CP", 1, 0, "5000"), ("Return", 0, 1, "0")])
def test_line_filter_financials_nonpurchase_dedup(dataset, status, purchases, nonpurchases, money):
    reference(dataset)
    raw = order([status, "CP"])
    raw["lines"][0]["priceOfLine"] = 5000
    raw["lines"][1]["priceOfLine"] = 8000
    raw["lines"][1]["product"]["ids"]["offline1C"] = "000002-a"
    sources(dataset, [], [raw, raw])
    result = calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=("Рубашки",)))
    assert (result.orders, result.order_lines, result.purchase_orders, result.orders_without_purchase) == (1, 1, purchases, nonpurchases)
    assert result.purchase_interactions == purchases
    assert result.order_financials[0][4] == money
    assert dict(result.diagnostics)["orders_duplicate_identical"] == 1
    assert result.customers == purchases
    cache.validate_result(asdict(result))


def test_zero_result_no_matching_orders_and_collection_source(dataset):
    reference(dataset)
    sources(dataset, [action(DEFAULT_SELECTION.view_action_system_names[0], ["000001-a"])], [order(["CP"])])
    for selection in (AnalysisFilter(collections=("Осень-зима",)), AnalysisFilter(collections=("Другое значение",))):
        result = calculate(dataset, analysis_filter=selection)
        assert (result.orders, result.order_lines, result.customers, result.interaction_users, result.purchase_orders) == (0, 0, 0, 0, 0)
        assert result.top_viewed_products == result.store_statistics == ()
        cache.validate_result(asdict(result))
    result = calculate(dataset, analysis_filter=AnalysisFilter(collections=("Весна-лето",)))
    assert result.product_season_statistics[0][0] == "Весна-лето"


def test_period_integration_actions_orders_and_coverage(dataset):
    reference(dataset)
    stamps = ["2025-02-28T23:59:59+00:00", "2025-03-01T00:00:00+00:00", "2025-05-31T16:00:00+00:00", "2025-06-01T00:00:00+00:00"]
    orders = []
    for i, stamp in enumerate(stamps):
        raw = order(["CP"], str(i))
        raw["firstAction"]["dateTimeUtc"] = stamp
        orders.append(raw)
    sources(dataset, [action(DEFAULT_SELECTION.view_action_system_names[0], ["000001-a"], stamp=stamp) for stamp in stamps], orders)
    result = calculate(dataset, analysis_filter=AnalysisFilter("2025-03-01", "2025-05-31"))
    assert (result.view_interactions, result.purchase_interactions, result.orders, result.repeat_buyers) == (2, 2, 2, 1)
    assert result.coverage[0].intervals[0][0].startswith("2025-01-01")
    cache.validate_result(asdict(result))


def test_repeat_buyers_are_filtered_distinct_orders(dataset):
    reference(dataset)
    orders = []
    for customer, codes in (("new", ["000001", "000001", "000002"]), ("other", ["000001", "000002"])):
        for i, code in enumerate(codes):
            raw = order(["CP"], f"{customer}{i}")
            raw["customer"]["ids"]["mindboxId"] = customer
            raw["lines"][0]["product"]["ids"]["offline1C"] = code + "-a"
            orders.append(raw)
    sources(dataset, [], orders)
    result = calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=("Рубашки",)))
    assert (result.purchase_users, result.repeat_buyers, result.purchase_orders, result.customers) == (2, 1, 3, 2)
    cache.validate_result(asdict(result))


def test_options_reference_only_and_refresh(dataset, monkeypatch):
    from Application import dataset_statistics as module
    reference(dataset)
    monkeypatch.setattr(module, "iter_export", lambda *a, **kw: pytest.fail("No event scan for options"))
    root, path, _ = dataset
    options = load_analysis_options(raw_root=root, catalog_path=path)
    assert options.nomenclature_types == ("Брюки", "Рубашки", None)
    assert options.collections == ("Весна-лето", "Осень-зима", None)
    path.write_text(path.read_text(encoding="utf-8-sig") + "000005|E|Новое|Новая|\n", encoding="utf-8-sig")
    assert "Новая" in load_analysis_options(raw_root=root, catalog_path=path).collections


def test_v9_roundtrip_v8_ignored_and_invalid_filter(dataset, tmp_path):
    reference(dataset)
    result = asdict(calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=("Рубашки",))))
    path = tmp_path / "cache.json"
    cache.save_result(path, result)
    assert cache.load_result(path) == json.loads(json.dumps(result))
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 9
    payload = json.dumps({"schema_version": 8, "result": result})
    path.write_text(payload, encoding="utf-8")
    assert cache.load_result(path) is None and path.read_text(encoding="utf-8") == payload
    result["analysis_filter"]["collections"] = []
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_default_matches_implicit(dataset):
    assert calculate(dataset).analysis_filter == AnalysisFilter()
    first, second = asdict(calculate(dataset)), asdict(calculate(dataset, analysis_filter=AnalysisFilter()))
    first.pop("calculated_at")
    second.pop("calculated_at")
    assert first == second


def test_single_pass_and_source_diagnostics(dataset, monkeypatch):
    from Application import dataset_statistics as module
    reference(dataset)
    calls = []
    original = module.iter_export
    def scan(name, **kwargs):
        calls.append(name)
        return original(name, **kwargs)
    monkeypatch.setattr(module, "iter_export", scan)
    result = calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=("Рубашки",)))
    assert calls == ["customer_merges", "actions", "orders"]
    diagnostics = dict(result.diagnostics)
    # Source action classification is deliberately not product-filtered.
    assert diagnostics["mapped_view_actions"] == 4
    assert diagnostics["mapped_without_product"] == diagnostics["unmapped_actions"] == 1
    assert result.view_interactions == 2


def test_atomic_failure_preserves_settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    settings.save_filter(AnalysisFilter(), path)
    before = path.read_bytes()
    def fail(*args):
        raise OSError
    monkeypatch.setattr(settings.os, "replace", fail)
    with pytest.raises(OSError):
        settings.save_filter(AnalysisFilter(collections=("A",)), path)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("mutation", ["unknown", "empty", "duplicate", "wrong_date", "half_period", "unsorted", "missing"])
def test_filter_cache_contract_rejects_malformed(dataset, mutation):
    result = asdict(calculate(dataset))
    value = result["analysis_filter"]
    if mutation == "unknown":
        value["product_ids"] = ["private"]
    elif mutation == "empty":
        value["collections"] = []
    elif mutation == "duplicate":
        value["collections"] = ["A", "A"]
    elif mutation == "wrong_date":
        value["start_date"], value["end_date"] = "bad", "bad"
    elif mutation == "half_period":
        value["start_date"] = "2025-01-01"
    elif mutation == "unsorted":
        value["collections"] = ["B", "A"]
    else:
        del result["analysis_filter"]
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_filtered_cache_rejects_inconsistent_counts(dataset):
    reference(dataset)
    result = asdict(calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=("Рубашки",))))
    result["orders"] += 1
    with pytest.raises(cache.StatisticsCacheError):
        cache.validate_result(result)


def test_source_order_period_remains_additional_constraint(dataset):
    reference(dataset)
    raw = order(["CP"])
    raw["firstAction"]["dateTimeUtc"] = "2025-01-01T12:00:00+00:00"
    root, _, data = dataset
    data["orders"] = {"2026-01-01": put_entry(root, "orders", [raw], 94)}
    save(root, data)
    result = calculate(dataset, analysis_filter=AnalysisFilter("2025-01-01", "2026-01-01"))
    assert result.purchase_orders == 0
    assert dict(result.diagnostics)["orders_outside_statistics_period"] == 1


def test_nonpurchase_resolution_does_not_inflate_interaction_diagnostics(dataset):
    reference(dataset)
    sources(dataset, [], [order(["Return"])])
    result = calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=("Рубашки",)))
    assert result.orders_without_purchase == 1
    assert result.resolved_interactions == result.unique_source_products == result.interaction_users == 0
    assert result.line_statuses == (("Return", 1, False),)


def test_fractional_lines_do_not_establish_product_filtered_order(dataset):
    reference(dataset)
    raw = order(["CP"])
    raw["lines"][0]["quantity"] = 1.5
    sources(dataset, [], [raw])
    result = calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=("Рубашки",)))
    assert result.orders == result.orders_without_purchase == 0
    assert dict(result.diagnostics)["fractional_quantity_order_lines"] == 1
    cache.validate_result(asdict(result))
