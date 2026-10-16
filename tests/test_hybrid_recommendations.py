"""Small synthetic references and SQLite only; never builds an index or trains."""

from contextlib import closing, contextmanager, nullcontext
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Application import customer_history_index as index
from Application.customer_card_data import CustomerCardService, ViewState
from Application.hybrid_recommendations import HybridRecommendationService, compose, similarity
from Application.model.raw_bpr_inference import RawBprRecommendation, RawBprResult, RawBprStatus
from Application.product_resolution import CatalogDiagnostics, ProductCatalog, ProductResolver
from Application.purchased_products import PurchaseEvidence, PurchasedProductsService, assess_order, classify
from Application.recommendation_catalog import FIELDS, RecommendationCatalog, item_from_metadata, stock
from Application.recommendation_rules import RecommendationRules, load_rules, save_rules


BASE = {"Номенклатура": "Брюки хлопок", "НазваниеНаСайте": "Брюки хлопок", "ВидНоменклатуры": "Брюки",
        "Коллекция": "Весна-Лето 2025", "ПолНоменклатуры": "Мужской", "СтилеваяГруппа": "Casual",
        "КатегорияНаСайте": "trousers", "Марка": "K", "ГруппаСоставов": "Хлопок",
        "ВидАссортимента": "Одежда", "СезонНоски": "Лето", "Остаток": "12"}


def write_catalog(tmp_path, rows):
    path = tmp_path / "nomenclature.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("КодНоменклатуры", *FIELDS, "Остаток"), delimiter="|")
        writer.writeheader()
        for code, changes in rows:
            writer.writerow({"КодНоменклатуры": code, **BASE, **changes})
    categories = tmp_path / "site_categories.csv"
    categories.write_text("КодКатегории|НазваниеКатегории|КодРодительскойКатегории\n"
                          "clothes|Одежда|\ntrousers|Брюки|clothes\njeans|Джинсы|clothes\n", encoding="utf-8-sig")
    return RecommendationCatalog(path, categories).load()


def raw_result(count=1):
    return RawBprResult("g", "model-user", RawBprStatus.OK, tuple(
        RawBprRecommendation(n + 1, f"source-{n}", float(count - n) - .123456789) for n in range(count)))


@pytest.fixture
def catalog(tmp_path):
    return write_catalog(tmp_path, [(f"{n:06d}", {"Коллекция": season}) for n, season in enumerate(
        ("Весна-Лето 2026", "Осень-Зима 2026", "NOS", "Всесезонный", "Осень-Зима 2026 переходящий остаток"), 1)])


def rules(catalog):
    return RecommendationRules(("Брюки",), tuple(sorted(catalog.collections)), 1)


@pytest.mark.parametrize("value,expected", [("1 234,50", Decimal("1234.50")), ("1\xa0200", Decimal("1200")),
                                          ("1\u202f200", Decimal("1200")), ("0", Decimal(0)),
                                          ("-1", None), ("?", None), ("NaN", None), ("Infinity", None)])
def test_decimal_stock_without_float(value, expected):
    assert stock(value) == expected


def test_rules_atomic_independent_roundtrip_and_empty_explicit(tmp_path, catalog, monkeypatch):
    path, analysis = tmp_path / "rules.json", tmp_path / "filter_settings.json"
    analysis.write_text("manual statistics settings", encoding="utf-8")
    setting = rules(catalog)
    save_rules(setting, path)
    assert load_rules(path) == setting
    assert analysis.read_text(encoding="utf-8") == "manual statistics settings"
    previous = path.read_bytes()
    monkeypatch.setattr("Application.recommendation_rules.os.replace", Mock(side_effect=OSError()))
    with pytest.raises(OSError):
        save_rules(setting, path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError):
        RecommendationRules((), setting.collections, 0)
    with pytest.raises(ValueError):
        RecommendationRules(setting.kinds, (), 0)
    with pytest.raises(ValueError):
        RecommendationRules(setting.kinds, setting.collections, True)
    with pytest.raises(ValueError):
        replace(setting, minimum_stock=-1)
    with pytest.raises(ValueError):
        replace(setting, collections=("Удалённая коллекция",)).validate_catalog(catalog)


@pytest.mark.parametrize("value", [{"version": 99}, [], {"version": 1, "kinds": "Брюки"}])
def test_invalid_settings_schema(tmp_path, value):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rules(path)


def test_catalog_conflicts_not_last_row_and_changes_invalidate(tmp_path):
    first = write_catalog(tmp_path, [("000001", {}), ("000001", {}), ("000002", {}),
                                     ("000002", {"Марка": "Different"}), ("000003", {"Остаток": "-3"})])
    assert set(first.items) == {"000001", "000003"}
    assert first.all_codes == {"000001", "000002", "000003"}
    assert first.diagnostics["ambiguous_codes"] == 1
    assert first.diagnostics["duplicate_rows"] == 2
    before = first.items
    assert first.load().items is before
    first.path.write_text(first.path.read_text(encoding="utf-8-sig").replace("Different", "K"), encoding="utf-8-sig")
    assert not first.current()
    assert "000002" in first.load().items


def test_category_ambiguity_cycle_and_hierarchy(tmp_path, catalog):
    source = item_from_metadata("source", BASE)
    exact = catalog.items["000001"]
    sibling = item_from_metadata("sibling", {**BASE, "КатегорияНаСайте": "jeans"})
    assert similarity(source, exact, catalog) > similarity(source, sibling, catalog)
    with catalog.categories_path.open("a", encoding="utf-8") as stream:
        stream.write("other|Брюки|clothes\ncycle|Цикл|cycle\n")
    catalog.load()
    assert "Брюки" not in catalog.category_paths and "cycle" not in catalog.category_paths


@pytest.mark.parametrize("season", ["Весна-Лето 2025", "Осень-Зима 2025"])
def test_cross_season_many_targets_and_raw_scores_preserved(catalog, season):
    raw = raw_result(2)
    metadata = {r.model_item_code: {**BASE, "Коллекция": season} for r in raw.recommendations}
    result = compose(raw, metadata, catalog, rules(catalog), PurchaseEvidence("READY"))
    assert len(result) == 5
    assert {i.target_collection for i in result} == catalog.collections
    assert all(len(item.sources) == 2 for item in result)
    assert [i.final_rank for i in result] == [1, 2, 3, 4, 5]
    for item in result:
        assert item.final_score == max(s.evidence_score for s in item.sources)
        for source in item.sources:
            original = next(r for r in raw.recommendations if r.model_item_code == source.source_model_item_code)
            assert (source.source_raw_score, source.source_raw_rank) == (original.raw_score, original.rank)
    assert compose(raw, metadata, catalog, rules(catalog), PurchaseEvidence("READY")) == result


@pytest.mark.parametrize("changes", [{"ВидНоменклатуры": "Пиджаки"}, {"ПолНоменклатуры": "Женский"},
                                     {key: "" for key in BASE if key not in ("ВидНоменклатуры", "Коллекция")}])
def test_incompatible_or_insufficient_evidence_never_random(catalog, changes):
    assert not compose(raw_result(), {"source-0": {**BASE, **changes}}, catalog, rules(catalog), PurchaseEvidence("READY"))


def test_missing_optional_feature_is_zero_not_veto_and_no_current_metadata_substitution(catalog):
    raw = raw_result()
    assert compose(raw, {"source-0": {**BASE, "СезонНоски": "", "СтилеваяГруппа": ""}}, catalog, rules(catalog), PurchaseEvidence("READY"))
    assert compose(raw, {}, catalog, rules(catalog), PurchaseEvidence("READY")) == ()
    assert similarity(item_from_metadata("source", {}), item_from_metadata("target", {}), catalog) == 0


def test_collection_type_is_soft_and_does_not_replace_wearing_season_or_merge_values(catalog):
    source = item_from_metadata("source", {**BASE, "СезонНоски": ""})
    summer = similarity(source, catalog.items["000001"], catalog)
    winter = similarity(source, catalog.items["000002"], catalog)
    assert summer - winter == pytest.approx(.05)
    assert winter >= .55
    assert source.get("СезонНоски") == ""
    assert "Осень-Зима 2026" in catalog.collections
    assert "Осень-Зима 2026 переходящий остаток" in catalog.collections


@pytest.mark.parametrize("change", [{"ВидНоменклатуры": "Носки"}, {"Коллекция": "2025"},
                                    {"Остаток": "0"}, {"Остаток": "-1"}, {"Остаток": "bad"}])
def test_hard_rules(tmp_path, change):
    catalog = write_catalog(tmp_path, [("000001", {"Коллекция": "NOS", **change}), ("000002", {"Коллекция": "NOS"})])
    setting = RecommendationRules(("Брюки",), ("NOS",), 1)
    result = compose(raw_result(), {"source-0": BASE}, catalog, setting, PurchaseEvidence("READY"))
    assert [item.target_item_code for item in result] == ["000002"]


def test_purchased_uncertain_codes_excluded_and_primary_collection_priority(catalog):
    raw = raw_result()
    evidence = PurchaseEvidence("READY", frozenset(("000001",)), frozenset(("000002",)))
    setting = replace(rules(catalog), primary_collection="Всесезонный")
    result = compose(raw, {"source-0": BASE}, catalog, setting, evidence)
    assert [i.target_item_code for i in result] == ["000004", "000003", "000005"]
    assert not compose(raw, {"source-0": BASE}, catalog, setting, replace(evidence, status="INCOMPLETE"))


def test_broad_pool_uses_next_sources_after_raw_top_ten_and_duplicate_sources_do_not_boost(tmp_path):
    catalog = write_catalog(tmp_path, [(f"{n:06d}", {"Коллекция": "NOS", "Марка": f"M{n}"}) for n in range(12)])
    raw = raw_result(12)
    metadata = {r.model_item_code: {**BASE, "Марка": f"M{r.rank - 1}"} for r in raw.recommendations}
    bought = frozenset(f"{n:06d}" for n in range(10))
    result = compose(raw, metadata, catalog, rules(catalog), PurchaseEvidence("READY", bought))
    assert {i.target_item_code for i in result} == {"000010", "000011"}
    assert {s.source_raw_rank for i in result for s in i.sources} >= {11, 12}
    repeated = replace(raw, recommendations=raw.recommendations + raw.recommendations)
    again = compose(repeated, metadata, catalog, rules(catalog), PurchaseEvidence("READY", bought))
    assert again == result
    assert all(len(i.sources) == 12 for i in again)
    assert all(i.final_score == max(s.evidence_score for s in i.sources) for i in again)


def line(status="F", *, code="000001-size", identity="line-1", ordinal=0, namespace="offline1C", quantity="1"):
    return {"raw_status": status, "source_line_key": identity, "ordinal": ordinal,
            "selected_namespace": namespace, "selected_value": code, "quantity": quantity}


@pytest.fixture
def resolver():
    return ProductResolver(ProductCatalog(frozenset(("000001", "000002")), CatalogDiagnostics()))


@pytest.mark.parametrize("status,state", [("CP", "PURCHASE"), ("delivering", "PURCHASE"), ("F", "PURCHASE"),
                                        ("Mindbox-Tech-Cancellation", "CANCELLED"), ("otmenen", "CANCELLED"),
                                        ("out-of-stock", "CANCELLED"), ("Return", "RETURN"), ("klientskii-vozvrat", "RETURN"),
                                        ("CD", "UNKNOWN"), ("N", "UNKNOWN"), ("new-status", "UNKNOWN")])
def test_agreed_operational_statuses(status, state):
    assert classify(status) == state


@pytest.mark.parametrize("states,bought,doubtful,cancel_count", [
    (("F", "CP"), True, False, 0), (("F", "Return"), True, False, 0),
    (("F", "otmenen"), False, True, 0), (("N", "CD"), False, True, 0),
    (("otmenen", "out-of-stock"), False, False, 2), (("Return",), False, True, 0),
])
def test_snapshots_consensus_not_first_last_and_return_does_not_erase(resolver, states, bought, doubtful, cancel_count):
    p, u, counts, c, _, missing = assess_order([[line(s)] for s in states], resolver)
    assert ("000001" in p, "000001" in u, c, missing) == (bought, doubtful, cancel_count, False)
    assert sum(counts.values()) == len(states)


def test_missing_duplicate_or_changing_line_identity_and_owner_conflict(resolver):
    for snapshots in ([[line(identity=None)], [line(identity=None)]],
                      [[line()], [line(code="000002-size")]],
                      [[line()], []], [[line(), line(ordinal=1)], [line()]]):
        p, u, *_ = assess_order(snapshots, resolver)
        assert not p and "000001" in u
    p, u, *_ = assess_order([[line()]], resolver, owner_conflict=True)
    assert not p and u == {"000001"}
    assert assess_order([[line(identity=None)]], resolver)[0] == {"000001"}


def test_unresolved_namespace_never_blindly_slices_and_nonpositive_purchase_uncertain(resolver):
    assert assess_order([[line(namespace="website")]], resolver)[-1]
    assert not assess_order([[line(code="999999-size")]], resolver)[-1]
    assert assess_order([[line(code="bad")]], resolver)[-1]
    assert not assess_order([[line("otmenen", namespace="website")]], resolver)[-1]
    assert assess_order([[line(quantity="0")]], resolver)[1] == {"000001"}


@pytest.mark.parametrize("minimum,value,allowed", [(0, "0", False), (0, "1", True), (3, "2", False),
                                                  (3, "3", True), (3, "15", True), (0, "?", False),
                                                  (0, "-1", False), (3, "1\xa0000", True), (3, "1 000", True)])
def test_availability_is_strictly_positive_even_without_extra_threshold(tmp_path, minimum, value, allowed):
    catalog = write_catalog(tmp_path, [("000001", {"Коллекция": "NOS", "Остаток": value})])
    result = compose(raw_result(), {"source-0": BASE}, catalog,
                     RecommendationRules(("Брюки",), ("NOS",), minimum), PurchaseEvidence("READY"))
    assert bool(result) is allowed


def test_legacy_rules_without_priority_are_still_version_one(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"version": 1, "kinds": ["Брюки"], "collections": ["NOS"], "minimum_stock": 0}), encoding="utf-8")
    assert load_rules(path).primary_collection is None
    save_rules(load_rules(path), path)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_primary_bonus_exact_formula_preserves_other_seasons_and_raw_scores(catalog):
    raw = raw_result()
    plain = compose(raw, {"source-0": BASE}, catalog, rules(catalog), PurchaseEvidence("READY"))
    priority = compose(raw, {"source-0": BASE}, catalog,
                       replace(rules(catalog), primary_collection="Всесезонный"), PurchaseEvidence("READY"))
    assert {i.target_item_code for i in priority} == {i.target_item_code for i in plain}
    assert priority[0].target_collection == "Всесезонный"
    for item in priority:
        original = next(i for i in plain if i.target_item_code == item.target_item_code)
        assert item.final_score == pytest.approx((original.final_score + .05 * (item.target_collection == "Всесезонный")) / 1.05)
        assert item.sources[0].source_raw_score == raw.recommendations[0].raw_score


@pytest.fixture
def sqlite_history(tmp_path, monkeypatch, catalog):
    root = tmp_path / "raw"
    service = PurchasedProductsService(cache_dir=tmp_path / "cache")
    path = service._index.index_path(raw_root=root)
    path.parent.mkdir(parents=True)
    start, end = datetime(2022, 1, 1, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc)
    sources = SimpleNamespace(version="source-v", components=(SimpleNamespace(name="orders", since=start, until=end, source_kind="API"),))
    monkeypatch.setattr(index, "_reader_lock", lambda _: nullcontext())
    monkeypatch.setattr(index, "_capture_sources", lambda _: sources)
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(index._SCHEMA)
        db.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("schema_version", index.INDEX_SCHEMA_VERSION), ("normalizer_version", index.records.HISTORY_SCHEMA_VERSION),
            ("schema_definition", index._SCHEMA_FINGERPRINT), ("schema_signature", index._schema_signature(db)),
            ("source_version", sources.version), ("generation", "a" * 32),
            ("counts", json.dumps(asdict(index.HistoryIndexCounts())))])
        for n in range(65):
            event, variant = f"event-{n}", f"variant-{n}"
            code = "000001-size" if n < 64 else "000002-size"
            db.execute("INSERT INTO events(event_key,source,source_event_id,canonical_customer_id,kind,event_us) VALUES (?,?,?,?,?,?)",
                       (event, "ORDER", f"order-{n}", "canonical", "ORDER", n))
            db.execute("INSERT INTO event_customers VALUES (?,?,?)", (event, "canonical", n))
            db.execute("INSERT INTO variants VALUES (?,?,?,?,?,?,?,?)", (variant, event, bytes(32), "canonical", "old", "ORDER", n, "[]"))
            db.execute("INSERT INTO order_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)", (variant, variant, None, n, None, None, None, None, None, None))
            db.execute("INSERT INTO order_lines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (f"line-{n}", variant, f"stable-{n}", f"id-{n}", f"id-{n}", 0, 0, 1, "Товар", "1", "F", "1", None, "[]"))
            db.execute("INSERT INTO product_occurrences VALUES (?,?,?,?,?,?)", (variant, "line", 0, "offline1C", code, "[]"))
    return root, service, path, sources


def test_sqlite_entire_customer_not_ui_first_page_and_indexed_plan(sqlite_history, catalog):
    root, service, path, _ = sqlite_history
    result = service.get("canonical", catalog, raw_root=root)
    assert result.status == "READY"
    assert result.purchased == {"000001", "000002"}
    assert result.diagnostics["rows_by_status"] == {"F": 65}
    assert service.get("other", catalog, raw_root=root).purchased == frozenset()
    assert service.is_current(result, raw_root=root)
    with index._read_index(path) as db:
        plans = [row[3] for row in db.execute("EXPLAIN QUERY PLAN SELECT e.event_key FROM event_customers ec "
                 "JOIN events e USING(event_key) WHERE ec.canonical_customer_id=? AND e.source='ORDER'", ("canonical",))]
        assert any("event_customers_customer" in p for p in plans)
        assert not any("SCAN e" in p for p in plans)


def test_sqlite_cancellation_missing_index_and_generation_change(sqlite_history, catalog):
    root, service, path, sources = sqlite_history
    assert service.get("canonical", catalog, raw_root=root, cancelled=lambda: True).status == "CANCELLED"
    result = service.get("canonical", catalog, raw_root=root)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE metadata SET value=? WHERE key='generation'", ("b" * 32,))
    assert not service.is_current(result, raw_root=root)
    sources.version = "changed"
    assert service.get("canonical", catalog, raw_root=root).status == "INDEX_NOT_READY"
    path.unlink()
    assert service.get("canonical", catalog, raw_root=root).status == "INDEX_NOT_READY"


def test_partial_coverage_fail_closed(sqlite_history, catalog):
    root, service, _, sources = sqlite_history
    sources.components = ()
    result = service.get("canonical", catalog, raw_root=root)
    assert result.status == "INCOMPLETE_COVERAGE" and result.diagnostics["coverage_complete"] is False


@pytest.mark.parametrize("namespace", ["offline1C", "kanzlerKz"])
def test_old_purchase_outside_catalog_does_not_block_current_purchase_exclusion(sqlite_history, catalog, namespace):
    root, service, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='999999-historical',selected_namespace=? WHERE variant_key='variant-64'", (namespace,))
    result = service.get("canonical", catalog, raw_root=root)
    assert result.status == "READY" and result.purchased == {"000001"} and not result.uncertain
    assert result.diagnostics["historical_out_of_scope_codes"] == 1
    assert result.diagnostics["historical_out_of_scope_lines"] == 1
    assert result.diagnostics["unsafe_unresolved_identifier_lines"] == 0
    assert result.diagnostics["rows_by_status"] == {"F": 65}
    targets = compose(raw_result(), {"source-0": BASE}, catalog, rules(catalog), result)
    assert "000001" not in {i.target_item_code for i in targets}
    assert "000002" in {i.target_item_code for i in targets}


@pytest.mark.parametrize("namespace,value", [("offline1C", ""), ("offline1C", " 999999"),
                                             ("offline1C", None), ("offline1C", "short"),
                                             ("offline1C", "AB CDE-size"), ("offline1C", "999999\x00"),
                                             ("website", "999999"), (None, None)])
def test_unsafe_or_ambiguous_reference_still_blocks_whole_output(sqlite_history, catalog, namespace, value):
    root, service, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_namespace=?,selected_value=? WHERE variant_key='variant-64'", (namespace, value))
    result = service.get("canonical", catalog, raw_root=root)
    assert result.status == "UNRESOLVED_PRODUCTS"
    assert result.purchased == {"000001"}
    assert result.diagnostics["unsafe_unresolved_identifier_lines"] == 1
    assert not compose(raw_result(), {"source-0": BASE}, catalog, rules(catalog), result)


def test_outside_identity_still_participates_in_conflicting_snapshot_comparison(sqlite_history, catalog):
    root, service, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("INSERT INTO variants SELECT 'changed',event_key,?,canonical_customer_id,source_customer_id,kind,event_us,quality FROM variants WHERE variant_key='variant-64'", (bytes([1]) * 32,))
        db.execute("INSERT INTO order_snapshots SELECT 'changed','changed',retail_order_id,order_us,channel_system_name,channel_external_id,channel_mindbox_id,channel_name,total_price,delivery_cost FROM order_snapshots WHERE variant_key='variant-64'")
        db.execute("INSERT INTO order_lines SELECT 'changed-line','changed',source_line_key,line_id,source_line_id,is_synthetic,ordinal,line_number,product_name,quantity,raw_status,price_of_line,base_price_per_item,quality FROM order_lines WHERE variant_key='variant-64'")
        db.execute("INSERT INTO product_occurrences VALUES ('changed','line',0,'offline1C','999999-old','[]')")
    result = service.get("canonical", catalog, raw_root=root)
    assert result.status == "READY" and result.purchased == {"000001"}
    assert result.uncertain == {"000002"} and result.diagnostics["conflicting_orders"] == 1
    assert result.diagnostics["historical_out_of_scope_codes"] == 1


def test_catalog_code_with_possible_prefix_alias_is_not_proven_disjoint(sqlite_history, catalog):
    root, service, path, _ = sqlite_history
    catalog.all_codes = catalog.all_codes | {"999999-catalog-sku"}
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE product_occurrences SET selected_value='999999-old-size' WHERE variant_key='variant-64'")
    result = service.get("canonical", catalog, raw_root=root)
    assert result.status == "UNRESOLVED_PRODUCTS"
    assert result.diagnostics["unsafe_unresolved_identifier_lines"] == 1
    assert result.diagnostics["historical_out_of_scope_codes"] == 0


def test_sqlite_resources_closed_on_success_cancellation_and_query_error(sqlite_history, catalog, monkeypatch):
    root, service, path, _ = sqlite_history
    reader, connections = index._read_index, []
    @contextmanager
    def tracked(path):
        with reader(path) as db:
            connections.append(db)
            yield db
    monkeypatch.setattr(index, "_read_index", tracked)
    assert service.get("canonical", catalog, raw_root=root).status == "READY"
    assert service.get("canonical", catalog, raw_root=root, cancelled=lambda: True).status == "CANCELLED"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DROP TABLE order_lines")
    assert service.get("canonical", catalog, raw_root=root).status == "INDEX_NOT_READY"
    for db in connections:
        with pytest.raises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")


def test_bought_and_cancelled_lines_share_order_without_order_level_mask(sqlite_history, catalog):
    root, service, path, _ = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE order_lines SET raw_status='Mindbox-Tech-Cancellation' WHERE line_key='line-64'")
    result = service.get("canonical", catalog, raw_root=root)
    assert result.status == "READY" and result.purchased == {"000001"}
    assert "000002" not in result.uncertain
    assert result.diagnostics["cancelled_lines_not_purchases"] == 1
    assert result.diagnostics["rows_by_status"] == {"F": 64, "Mindbox-Tech-Cancellation": 1}


def test_hybrid_service_unconfigured_unavailable_and_current_settings(tmp_path, catalog):
    inference = Mock()
    raw = raw_result()
    inference.get_candidate_sources.return_value = (raw, {"source-0": BASE})
    identity = SimpleNamespace(model_user_id="model-user", canonical_customer_id="canonical", model_generation="g")
    evidence = Mock()
    evidence.get.return_value = PurchaseEvidence("READY")
    evidence.is_current.return_value = True
    path = tmp_path / "rules.json"
    service = HybridRecommendationService(catalog_path=catalog.path, categories_path=catalog.categories_path,
                                          rules_path=path, purchases=evidence)
    result = service.get(identity, inference, model_dir=tmp_path, raw_root=tmp_path)
    assert result.status == "RULES_NOT_READY" and result.raw_bpr == raw
    evidence.get.assert_called_once()  # Raw presentation now requires purchase verification even without rules.
    save_rules(rules(catalog), path)
    result = service.get(identity, inference, model_dir=tmp_path, raw_root=tmp_path)
    assert result.status == "READY" and len(result.items) == 5
    assert result.raw_bpr == raw
    save_rules(replace(rules(catalog), primary_collection="Всесезонный"), path)
    priority_result = service.get(identity, inference, model_dir=tmp_path, raw_root=tmp_path)
    assert priority_result.status == "READY" and priority_result.items[0].target_collection == "Всесезонный"
    assert priority_result.raw_bpr == raw
    save_rules(replace(rules(catalog), minimum_stock=50), path)
    assert not service.is_current(result, raw_root=tmp_path)
    assert service.get(identity, inference, model_dir=tmp_path, raw_root=tmp_path).status == "EMPTY"
    evidence.get.return_value = PurchaseEvidence("UNRESOLVED_PRODUCTS")
    result = service.get(identity, inference, model_dir=tmp_path, raw_root=tmp_path)
    assert result.status == "PURCHASES_NOT_READY" and not result.items


def test_card_hybrid_does_not_change_profile_or_raw_source_contract(tmp_path, catalog):
    from tests.test_customer_card_data import identity
    bridge = Mock()
    bridge.resolve.return_value = replace(identity(), model_user_id="model-user")
    bridge.is_current.return_value = True
    inference = Mock()
    inference.get_candidate_sources.return_value = (raw_result(), {"source-0": BASE})
    purchases = Mock()
    purchases.get.return_value = PurchaseEvidence("READY")
    purchases.get.return_value = replace(purchases.get.return_value, historical=PurchaseEvidence(
        "READY", checked_codes=frozenset(r.model_item_code for r in raw_result().recommendations)))
    purchases.is_current.return_value = True
    path = tmp_path / "rules.json"
    save_rules(rules(catalog), path)
    hybrid = HybridRecommendationService(catalog_path=catalog.path, categories_path=catalog.categories_path,
                                         rules_path=path, purchases=purchases)
    service = CustomerCardService(bridge=bridge, inference=inference, hybrid=hybrid)
    result = service.load("canonical")
    assert result.state is ViewState.READY
    assert result.profile is not None and result.hybrid.raw_bpr == raw_result()
    assert result.products[0].recommendation.raw_score == raw_result().recommendations[0].raw_score
    assert result.products[0].hybrid_item.target_item_code == "000001"
    purchases.is_current.return_value = False
    stale = service.validate_result(result)
    assert stale.state is ViewState.STALE and not stale.products and stale.profile == result.profile
