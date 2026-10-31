"""Synthetic canonical sources only. Never build or scan the real history."""

from copy import deepcopy
from dataclasses import asdict, fields
import json
import sqlite3

import pytest

from Application import dataset_statistics as reference
from Application import statistics_index as index
from Application import statistics_slices as cache
from Application.analysis_filter import AnalysisFilter
from Application.mindbox.adapters._common import AdapterError
from Application.statistics_query import calculate_indexed_statistics
from tests.test_dataset_statistics import dataset as dataset_fixture, action, order, save
from tests.test_analysis_filter import reference as catalog, sources


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    # Ordinary fixture builds exercise everything except repeatedly launching the
    # Windows ACL utility. One dedicated test below uses the real protection.
    monkeypatch.setattr(index, "protect_directory", lambda p: p.chmod(0o700))
    monkeypatch.setattr(index, "validate_storage_root", lambda p, **kw: index.path_identity(p))
    data = dataset_fixture.__wrapped__(tmp_path)
    (data[0]/"analytics").mkdir()
    return data


def build(dataset, **kwargs):
    (dataset[0]/"analytics").mkdir(exist_ok=True)
    return index.build_statistics_index(raw_root=dataset[0], catalog_path=dataset[1],
        index_dir=dataset[0]/"analytics", expected_bytes=1024**2, **kwargs)


def query(dataset, selection=AnalysisFilter(), **kwargs):
    return calculate_indexed_statistics(raw_root=dataset[0], catalog_path=dataset[1],
        index_dir=dataset[0]/"analytics", analysis_filter=selection, **kwargs)


def normalized(value):
    result = json.loads(json.dumps(asdict(value)))
    result.pop("calculated_at")
    return result


def golden(dataset, selection):
    expected = reference.calculate_statistics(raw_root=dataset[0], catalog_path=dataset[1], analysis_filter=selection)
    actual = query(dataset, selection)
    assert len(fields(expected)) == len(fields(actual)) == 93
    assert normalized(actual) == normalized(expected)
    return actual


def test_full_field_matrix_preserves_first_snapshot_scopes_decimal_and_product_occurrences(dataset):
    catalog(dataset)
    first = order(["CP", "CD", "Return", "delivering", "F"], "same")
    first["lines"][1]["product"]["ids"]["offline1C"] = "000002-size"
    later = deepcopy(first)
    later["firstAction"]["dateTimeUtc"] = "2026-03-01T12:00:00Z"
    later["customer"]["ids"]["mindboxId"] = "other"
    later["lines"][0]["status"]["ids"]["externalId"] = "Return"
    later["lines"][0]["product"]["ids"]["offline1C"] = "000002-size"
    conflict_unused = deepcopy(first)
    conflict_unused["unused"] = "different"
    fractional = order(["F"], "fractional")
    fractional["lines"][0]["quantity"] = 1.25
    mixed = order(["F", "F"], "mixed")
    mixed["lines"][1]["product"]["ids"] = {"kanzlerKz": "000002-size"}
    unknown = order(["F"], "unknown")
    unknown["lines"][0]["product"]["ids"] = {"kanzlerKz": "999999-size"}
    outside = order(["F"], "outside")
    outside["firstAction"]["dateTimeUtc"] = "2030-01-01T00:00:00Z"
    technical = action("Technical")
    technical.pop("customer")
    technical.pop("dateTimeUtc")
    actions = [action("ProsmotrProdukta", ["000001-size", "000001-size", "999999-size"], price=10, available=False),
               action("ProsmotrProdukta", ["000002-size"], stamp="2026-03-01T01:00:00Z", price=9.99, available=True),
               action("ProsmotrProdukta"), technical,
               action("DobavlenieProduktaVSpisokVOperaciiDobavlenie", ["000001-size"], customer="other")]
    sources(dataset, actions, [first, first, conflict_unused, later, fractional, mixed, unknown, outside])
    build(dataset)
    for selected in (
        AnalysisFilter(), AnalysisFilter("2026-01-01", "2026-01-01"),
        AnalysisFilter("2026-01-02", "2026-02-28"), AnalysisFilter("2026-03-01", "2026-03-01"),
        AnalysisFilter("2026-01-01", "2026-12-31"), AnalysisFilter(nomenclature_types=("Рубашки",)),
        AnalysisFilter(nomenclature_types=("Брюки",)), AnalysisFilter(nomenclature_types=(None,)),
        AnalysisFilter(collections=("Весна-лето",)), AnalysisFilter(collections=("Осень-зима",)),
        AnalysisFilter(collections=(None,)), AnalysisFilter(collections=("missing",)),
        AnalysisFilter("2026-01-01", "2026-03-01", ("Брюки",), ("Осень-зима",)),
        AnalysisFilter("2025-01-01", "2025-01-01"),
    ):
        golden(dataset, selected)


def test_basic_projection_does_not_read_any_raw_json_or_customers_rows_at_query(dataset, monkeypatch):
    expected = reference.calculate_statistics(raw_root=dataset[0], catalog_path=dataset[1])
    built = build(dataset)
    monkeypatch.setattr(reference, "iter_export", lambda *a, **k: pytest.fail("raw scan"))
    monkeypatch.setattr(reference, "_customer_snapshot", lambda *a, **k: pytest.fail("profiles scan"))
    actual = query(dataset)
    assert normalized(actual) == normalized(expected)
    folder = dataset[0]/"analytics"/built["generation"]
    with sqlite3.connect(folder/"statistics.sqlite") as db:
        assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert not db.execute("SELECT name FROM sqlite_master WHERE name IN ('customer_ids','order_ids')").fetchall()
        sql = " ".join(r[0] or "" for r in db.execute("SELECT sql FROM sqlite_master"))
        assert all(word not in sql for word in ("email", "phone", "birthDate", "raw TEXT"))
    content = (folder/"statistics.sqlite").read_bytes()
    assert b'"mindboxId"' not in content
    assert not list(folder.glob("*-journal"))


def test_indexed_and_reference_keep_same_frozen_category_paths_without_raw_reads(dataset, monkeypatch):
    dataset[1].write_text('КодНоменклатуры|КатегорияНаСайте\n000001|000123\n000002|000124\n', encoding='utf-8-sig')
    (dataset[0] / 'site_categories.csv').write_text(
        'КодКатегории|НазваниеКатегории|КодРодительскойКатегории\n'
        '000123|Рубашки|root\n000124|Рубашки|missing\nroot|Раздел|\n', encoding='utf-8-sig')
    expected = reference.calculate_statistics(raw_root=dataset[0], catalog_path=dataset[1])
    build(dataset)
    monkeypatch.setattr(reference, 'iter_export', lambda *a, **k: pytest.fail('raw read'))
    actual = query(dataset)
    assert normalized(actual) == normalized(expected)
    assert dict(actual.product_category_paths) == {'000123': ('Раздел', 'Рубашки'), '000124': None}


@pytest.mark.parametrize("source", ["revision", "actions", "merges", "customers", "catalog", "categories", "selection", "algorithm", "runtime"])
def test_every_provenance_dependency_invalidates(dataset, monkeypatch, source):
    build(dataset)
    root, path, data = dataset
    if source == "revision":
        data["revision"] = "changed"
        save(root, data)
    elif source in ("actions", "merges"):
        name = "customer_merges" if source == "merges" else source
        entry = data[name] if name == "customer_merges" else next(iter(data[name].values()))
        part = root/entry["directory"]/f"{name}_part_001.json"
        part.write_bytes(part.read_bytes()+b" ")
    elif source == "customers":
        with sqlite3.connect(root/"canonical/customers.sqlite") as db:
            db.execute("UPDATE profiles SET raw=raw||' '")
    elif source in ("catalog", "categories"):
        part = path if source == "catalog" else path.with_name("site_categories.csv")
        part.write_bytes(part.read_bytes() + ("000003|Новое|Новое\n" if source == "catalog" else "001|Новое\n").encode())
    elif source == "selection":
        data["selection"]["purchase_line_statuses"] = ["F"]
        save(root, data)
    elif source == "algorithm":
        monkeypatch.setattr(cache, "_algorithm_version", lambda: "changed")
    else:
        original = cache.capture_sources

        def changed(*a, **kw):
            value, effective = original(*a, **kw)
            value["runtime"]["decimal"]["prec"] += 1
            return value, effective

        import Application.statistics_query as query_module
        monkeypatch.setattr(query_module, "capture_sources", changed)
    with pytest.raises((index.StatisticsIndexError, ValueError)):
        query(dataset)


@pytest.mark.parametrize("stage", ["start", "projection", "sql"])
def test_cancellation_keeps_previous_generation_and_removes_staging(dataset, stage):
    before = build(dataset)
    directory = dataset[0]/"analytics"
    pointer = (directory/"current.json").read_bytes()
    stopped = stage == "start"

    def progress(message):
        nonlocal stopped
        if (stage == "projection" and "actions" in message) or (stage == "sql" and "SQL" in message):
            stopped = True

    with pytest.raises(InterruptedError):
        build(dataset, cancelled=lambda: stopped, progress=progress)
    assert (directory/"current.json").read_bytes() == pointer
    assert {p.name for p in directory.iterdir() if p.is_dir()} == {before["generation"]}
    assert index.index_state(raw_root=dataset[0], catalog_path=dataset[1], index_dir=directory) == "READY"
    # Both locks must have been released.
    with index._writer_lock(directory), reference.storage_lock(dataset[0]):
        pass


def test_source_changed_during_build_never_publishes(dataset):
    before = build(dataset)
    pointer = (dataset[0]/"analytics/current.json").read_bytes()
    once = False

    def progress(message):
        nonlocal once
        if not once:
            once = True
            dataset[2]["revision"] = "modified-during-build"
            save(dataset[0], dataset[2])

    with pytest.raises(index.StatisticsIndexError, match="STALE_SOURCE"):
        build(dataset, progress=progress)
    assert (dataset[0]/"analytics/current.json").read_bytes() == pointer
    assert (dataset[0]/"analytics"/before["generation"]).is_dir()


def test_disk_floor_failure_cleans_and_preserves_sources(dataset, monkeypatch):
    before = (dataset[0]/"canonical/catalog.json").read_bytes()
    monkeypatch.setattr(index.shutil, "disk_usage", lambda p: type("Usage", (), {"free": 20*index.GIB})())
    with pytest.raises(index.StatisticsIndexError, match="INSUFFICIENT_DISK"):
        build(dataset)
    assert not (dataset[0]/"analytics/current.json").exists()
    assert not [p for p in (dataset[0]/"analytics").iterdir() if p.is_dir()]
    assert (dataset[0]/"canonical/catalog.json").read_bytes() == before


def test_missing_index_auto_fallback_is_explicit_never_builds(dataset, monkeypatch):
    monkeypatch.setattr(index, "build_statistics_index", lambda **k: pytest.fail("automatic build"))
    progress = []
    result = cache.calculate_cached_statistics(raw_root=dataset[0], catalog_path=dataset[1], cache_dir=None,
        backend="AUTO", index_dir=dataset[0]/"missing", progress=progress.append)
    assert result["source_actions"] == 7
    assert any("REFERENCE" in text and "INDEX_MISSING" in text for text in progress)
    with pytest.raises(index.StatisticsIndexError, match="INDEX_MISSING"):
        reference.calculate_statistics(raw_root=dataset[0], catalog_path=dataset[1], backend="INDEXED", index_dir=dataset[0]/"missing")


def test_auto_source_preflight_failure_retains_reference_catalog_error(dataset):
    from Application.product_resolution import CatalogError
    dataset[1].unlink()
    with pytest.raises(CatalogError):
        cache.calculate_cached_statistics(raw_root=dataset[0], catalog_path=dataset[1], cache_dir=None,
            backend="AUTO", index_dir=dataset[0]/"missing")
    with pytest.raises(index.StatisticsIndexError, match="SOURCE_UNAVAILABLE"):
        query(dataset)


def test_indexed_result_cache_reuses_exact_valid_result(dataset, monkeypatch):
    build(dataset)
    arguments = dict(raw_root=dataset[0], catalog_path=dataset[1], cache_dir=dataset[0]/"slices",
                     backend="INDEXED", index_dir=dataset[0]/"analytics")
    first = cache.calculate_cached_statistics(**arguments)
    monkeypatch.setattr(reference, "calculate_statistics", lambda **kw: pytest.fail("cache miss"))
    assert cache.calculate_cached_statistics(**arguments) == json.loads(json.dumps(first))


def test_unmapped_invalid_date_and_mapped_adapter_errors_remain_query_dependent(dataset):
    catalog(dataset)
    bad = action("Technical", ["000001-size"])
    bad["dateTimeUtc"] = "invalid"
    sources(dataset, [bad], [])
    build(dataset)
    golden(dataset, AnalysisFilter())
    with pytest.raises(AdapterError):
        query(dataset, AnalysisFilter("2026-01-01", "2026-01-01"))


def test_duplicate_bad_adapter_ignored_after_first_but_date_can_choose_it(dataset):
    catalog(dataset)
    first = order(["F"], "same")
    later = deepcopy(first)
    later["firstAction"]["dateTimeUtc"] = "2026-02-01T12:00:00Z"
    later["lines"][0]["number"] = "invalid"
    sources(dataset, [], [first, later])
    build(dataset)
    golden(dataset, AnalysisFilter())
    golden(dataset, AnalysisFilter(nomenclature_types=("Брюки",)))
    with pytest.raises(AdapterError):
        query(dataset, AnalysisFilter("2026-02-01", "2026-02-01"))


def test_filtered_canonical_profiles_keep_first_source_order_unfiltered_keeps_all(dataset):
    with sqlite3.connect(dataset[0]/"canonical/customers.sqlite") as db:
        db.execute("INSERT INTO profiles VALUES('old','',?)", (json.dumps({"birthDate": "2000-01-01", "sex": "male"}),))
        db.execute("UPDATE profiles SET raw=? WHERE id='new'", (json.dumps({"birthDate": "1990-01-01", "sex": "female"}),))
    build(dataset)
    golden(dataset, AnalysisFilter())
    golden(dataset, AnalysisFilter(collections=("missing",)))


def test_absent_customers_partial_merges_coverage_and_empty_sources(dataset):
    (dataset[0]/"canonical/customers.sqlite").unlink()
    dataset[2]["customer_merges"]["since"] = "2026-01-01T01:00:00+00:00"
    save(dataset[0], dataset[2])
    build(dataset)
    actual = golden(dataset, AnalysisFilter())
    assert actual.customers is None
    assert actual.warnings


def test_malformed_customers_json_fails_build_even_for_empty_future_cohort(dataset):
    with sqlite3.connect(dataset[0]/"canonical/customers.sqlite") as db:
        db.execute("UPDATE profiles SET raw='invalid'")
    with pytest.raises(index.StatisticsIndexError, match="SQLITE_BUILD_ERROR"):
        build(dataset)
    assert not (dataset[0]/"analytics/current.json").exists()


def test_schema_manifest_and_pointer_corruption_fail_closed(dataset):
    built = build(dataset)
    path = dataset[0]/"analytics"/built["generation"]/"statistics.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=999")
    with pytest.raises(index.StatisticsIndexError, match="INCOMPATIBLE_SCHEMA"):
        query(dataset)
    (dataset[0]/"analytics/current.json").write_text('{"schema":1,"generation":"../../outside","manifest_sha256":"bad"}')
    with pytest.raises(index.StatisticsIndexError):
        query(dataset)


def test_changed_sources_after_query_rejected(dataset, monkeypatch):
    build(dataset)
    import Application.statistics_query as query_module
    original = query_module.capture_sources
    calls = 0

    def capture(*a, **kw):
        nonlocal calls
        calls += 1
        result, selected = original(*a, **kw)
        if calls > 1:
            result["canonical"]["revision"] = "changed"
        return result, selected

    monkeypatch.setattr(query_module, "capture_sources", capture)
    with pytest.raises(index.StatisticsIndexError, match="STALE_SOURCE"):
        query(dataset)


@pytest.mark.parametrize("payload", ["[]", "broken", '{"schema":1,"schema":1}', '"string"'])
def test_bad_pointer_documents_are_safe_index_errors(dataset, payload):
    build(dataset)
    (dataset[0]/"analytics/current.json").write_text(payload)
    with pytest.raises(index.StatisticsIndexError):
        query(dataset)


def test_bad_generation_manifest_and_missing_schema_do_not_escape_state_handling(dataset):
    built = build(dataset)
    folder = dataset[0]/"analytics"/built["generation"]
    manifest = folder/"manifest.json"
    manifest.write_text("[]")
    with pytest.raises(index.StatisticsIndexError, match="INVALID_MANIFEST"):
        query(dataset)


def test_query_cancel_closes_connection_and_does_not_change_generation(dataset):
    build(dataset)
    pointer = (dataset[0]/"analytics/current.json").read_bytes()
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls > 3

    with pytest.raises(InterruptedError):
        query(dataset, cancelled=cancel)
    assert (dataset[0]/"analytics/current.json").read_bytes() == pointer
    golden(dataset, AnalysisFilter())


def test_default_raw_reader_still_uses_untouched_reference_when_index_exists(dataset, monkeypatch):
    build(dataset)
    calls = []
    original = reference.iter_export

    def reader(*a, **kw):
        calls.append(a[0])
        yield from original(*a, **kw)

    monkeypatch.setattr(reference, "iter_export", reader)
    reference.calculate_statistics(raw_root=dataset[0], catalog_path=dataset[1])
    assert set(calls) == {"actions", "orders", "customer_merges"}


def test_private_directory_protection_on_new_directory(tmp_path):
    folder = tmp_path/"private"
    folder.mkdir()
    index.protect_directory(folder)
    assert folder.is_dir()


def test_no_unnecessary_source_ids_or_contacts_in_published_file(dataset):
    customer = "SYNTHETIC-CUSTOMER-SECRET-8cdaf78a"
    order_id = "SYNTHETIC-ORDER-SECRET-32b6117b"
    extra = "SYNTHETIC-UNUSED-CONTACT-7b87d2ce@example.invalid"
    a = action("ProsmotrProdukta", ["000001-size"], customer=customer)
    o = order(["F"], order_id)
    o["customer"]["ids"]["mindboxId"] = customer
    o["unusedContact"] = extra
    sources(dataset, [a], [o])
    with sqlite3.connect(dataset[0]/"canonical/customers.sqlite") as db:
        db.execute("INSERT INTO profiles VALUES(?,'',?)", (customer,json.dumps({"email":extra,"sex":"male"})))
    built = build(dataset)
    path = dataset[0]/"analytics"/built["generation"]/"statistics.sqlite"
    contents = path.read_bytes()
    assert all(value.encode() not in contents for value in (customer,order_id,extra))
    golden(dataset, AnalysisFilter())


def test_competing_builder_fails_without_changing_ready_generation(dataset):
    build(dataset)
    directory = dataset[0]/"analytics"
    pointer = (directory/"current.json").read_bytes()
    from Application.mindbox.training_batch import TrainingBatchError
    with index._writer_lock(directory):
        with pytest.raises(TrainingBatchError):
            build(dataset)
    assert (directory/"current.json").read_bytes() == pointer


def test_cooperative_cli_cancel_file_keeps_old_pointer(dataset, tmp_path, capsys):
    from scripts.build_statistics_index import main
    build(dataset)
    pointer = (dataset[0]/"analytics/current.json").read_bytes()
    cancel = tmp_path/"cancel.request"
    cancel.touch()
    assert main(["--build","--expected-gib","0.001","--raw-root",str(dataset[0]),
                 "--catalog",str(dataset[1]),"--index-dir",str(dataset[0]/"analytics"),
                 "--cancel-file",str(cancel)]) == 2
    assert '"event": "cancelled"' in capsys.readouterr().out
    assert (dataset[0]/"analytics/current.json").read_bytes() == pointer


@pytest.mark.parametrize("which", ["action_adapter", "action_header", "order_header", "payment"])
def test_errors_excluded_by_date_or_business_scope_are_not_eagerly_fatal(dataset, which):
    catalog(dataset)
    a = action("ProsmotrProdukta", ["000001-size"], stamp="2026-03-01T00:00:00Z")
    o = order(["F"], "bad")
    o["firstAction"]["dateTimeUtc"] = "2026-03-01T00:00:00Z"
    if which == "action_adapter":
        a["creationDateTimeUtc"] = "invalid"
    elif which == "action_header":
        a["products"] = "invalid"
    elif which == "order_header":
        o["lines"][0]["quantity"] = "invalid"
    else:
        o["payments"] = "invalid"
    sources(dataset, [a], [o])
    build(dataset)
    golden(dataset, AnalysisFilter("2026-01-01", "2026-01-01"))
    with pytest.raises(AdapterError):
        query(dataset)
    with pytest.raises(AdapterError):
        reference.calculate_statistics(raw_root=dataset[0], catalog_path=dataset[1])


def test_semantic_fingerprint_equal_numeric_representations_but_unused_conflict(dataset):
    from Application.mindbox.canonical_customers import encode
    from decimal import Decimal
    a = order(["F"], "same")
    b = deepcopy(a)
    b["lines"][0]["priceOfLine"] = Decimal("20.000")
    c = deepcopy(a)
    c["notUsedByStatistics"] = "conflict"
    sources(dataset, [], [])
    entry = dataset[2]["manual_interactions"]["orders"]
    (dataset[0]/entry["directory"]/"orders_part_001.json").write_text(encode({"orders":[a,b,c]}))
    build(dataset)
    actual = golden(dataset, AnalysisFilter())
    assert dict(actual.diagnostics)["orders_duplicate_identical"] == 1
    assert dict(actual.diagnostics)["orders_duplicate_conflicting"] == 1


def test_existing_decimal_monthly_validation_edge_not_bypassed_or_rounded(dataset):
    from Application.statistics_cache import StatisticsCacheError, validate_result
    from Application.statistics_query import IndexedInput
    big = order(["F"], "big")
    big["lines"][0]["priceOfLine"] = 10**27
    small = []
    for n in range(10):
        event = order(["F"], f"small-{n}")
        event["firstAction"]["dateTimeUtc"] = "2026-02-01T00:00:00Z"
        event["lines"][0]["priceOfLine"] = 0.1
        small.append(event)
    sources(dataset, [], [big, *small])
    expected = reference.calculate_statistics(raw_root=dataset[0], catalog_path=dataset[1])
    with pytest.raises(StatisticsCacheError):
        validate_result(asdict(expected))
    build(dataset)
    provenance, effective = cache.capture_sources(dataset[0], dataset[1], AnalysisFilter())
    db, _ = index.open_generation(dataset[0]/"analytics", provenance)
    try:
        source = IndexedInput(db, provenance, effective, lambda: None, {"sqlite_seconds":0., "source_headers":0})
        actual = reference._calculate_statistics(dataset[0], dataset[1], None, lambda:None, effective, source)
        assert normalized(actual) == normalized(expected)
    finally:
        db.close()
    with pytest.raises(StatisticsCacheError):
        query(dataset)


def test_full_cli_build_requires_explicit_mode_and_budget(tmp_path, capsys):
    from scripts.build_statistics_index import main
    with pytest.raises(SystemExit):
        main(["--build"])
    assert "--expected-gib" in capsys.readouterr().err
    assert main(["--status", "--raw-root", str(tmp_path), "--index-dir", str(tmp_path/"missing")]) == 0
    assert "SOURCE_UNAVAILABLE" in capsys.readouterr().out


def test_explicit_cli_build_success_and_status_use_same_published_generation(dataset, capsys):
    from scripts.build_statistics_index import main
    arguments = ["--raw-root",str(dataset[0]),"--catalog",str(dataset[1]),"--index-dir",str(dataset[0]/"analytics")]
    assert main([*arguments,"--build","--expected-gib","0.001"]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "result"
    assert events[-1]["value"]["counts"]["actions"] == 7
    assert main([*arguments,"--status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"state":"READY"}


@pytest.fixture
def private_protection(tmp_path, monkeypatch):
    """Never execute a native ACL utility; inspect only synthetic ACL facts."""
    import subprocess
    from types import SimpleNamespace

    folder = tmp_path / "private"
    folder.mkdir()
    modules = tmp_path / "Windows" / "System32" / "WindowsPowerShell" / "v1.0" / "Modules"
    executables = {modules.parent / "powershell.exe", modules.parents[2] / "icacls.exe"}
    original_is_file = type(folder).is_file
    monkeypatch.setattr(type(folder), "is_file",
                        lambda p: p in executables or original_is_file(p))
    environment = {"PSModulePath": str(modules), "PATH": "UNTRUSTED_PATH"}
    monkeypatch.setattr(index, "_powershell_environment", lambda: environment.copy())
    monkeypatch.setattr(index, "_windows_user_sid", lambda: "S-1-0-0")
    facts = dict(owner_trusted=True, protected=True, unsupported=False, rules=[
        dict(trusted=True, current=True, mask=0x1F01FF, allow=True, inherit_only=False)])
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"Successfully processed", stderr=b"")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(index, "inspect_acls", lambda paths: [facts])
    return SimpleNamespace(folder=folder, modules=modules, calls=calls, facts=facts)


def test_protect_success_requires_private_acl_attestation(private_protection, monkeypatch):
    p = private_protection
    inspections = []

    def inspect(paths):
        inspections.append(paths)
        return [p.facts]

    monkeypatch.setattr(index, "inspect_acls", inspect)
    monkeypatch.setattr(index, "validate_storage_root",
                        lambda *a, **kw: pytest.fail("No production parent requirement"))
    index.protect_directory(p.folder)
    assert inspections == [[p.folder]]
    assert len(p.calls) == 1
    args, kwargs = p.calls[0]
    assert args[0] == str(p.modules.parents[2] / "icacls.exe")
    assert args[1] == str(p.folder)
    assert args[2:4] == ["/inheritance:r", "/grant:r"]
    assert kwargs["timeout"] == 30
    assert kwargs["capture_output"] is True
    assert not kwargs.get("shell", False)


@pytest.mark.parametrize("name", [
    "space folder", "Кириллица (скобки)", "apostrophe'folder",
    "dollar$semicolon;" + chr(96) + "ampersand&", "__PATHS__ __MODULES__",
])
def test_protect_path_is_one_native_argument(private_protection, name):
    p = private_protection
    folder = p.folder / name
    folder.mkdir()
    index.protect_directory(folder)
    args, kwargs = p.calls[0]
    assert args[0] == str(p.modules.parents[2] / "icacls.exe")
    assert args[1] == str(folder)
    assert args[2:4] == ["/inheritance:r", "/grant:r"]
    assert len(args) == 7
    assert not kwargs.get("shell", False)
    assert "/T" not in args


def test_protect_missing_trusted_executable_has_no_path_fallback(private_protection, monkeypatch):
    p = private_protection
    original = type(p.folder).is_file
    monkeypatch.setattr(type(p.folder), "is_file",
                        lambda path: False if path.name == "icacls.exe" else original(path))
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(p.folder)
    assert not p.calls


@pytest.mark.parametrize("failure", ["missing", "denied", "invalid_arguments", "timeout", "unexpected"])
def test_protect_subprocess_exceptions_are_safe(private_protection, monkeypatch, capsys, failure):
    import subprocess
    p = private_protection

    def run(*a, **kw):
        if failure == "missing":
            raise FileNotFoundError("SENSITIVE_NATIVE_DETAIL")
        if failure == "denied":
            raise PermissionError("SENSITIVE_NATIVE_DETAIL")
        if failure == "invalid_arguments":
            raise ValueError("SENSITIVE_NATIVE_DETAIL")
        if failure == "timeout":
            raise subprocess.TimeoutExpired("SENSITIVE_COMMAND", 30, stderr=b"SENSITIVE_STDERR")
        raise RuntimeError("SENSITIVE_NATIVE_DETAIL")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(index.StatisticsIndexError) as caught:
        index.protect_directory(p.folder)
    assert str(caught.value) == "PRIVATE_INDEX_ACL_FAILED"
    assert caught.value.__suppress_context__
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("code", [1, 2, 5])
def test_protect_nonzero_native_exit_never_attests(private_protection, monkeypatch, capsys, code):
    import subprocess
    from types import SimpleNamespace
    p = private_protection
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=code, stdout=b"SENSITIVE_STDOUT", stderr=b"SENSITIVE_STDERR"))
    monkeypatch.setattr(index, "inspect_acls", lambda *a: pytest.fail("Failed assignment must stop"))
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(p.folder)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("damage", [
    "owner", "unsupported", "not_protected", "owner_readonly",
    "foreign_allow", "foreign_inherit_only", "deny", "invalid_shape",
])
def test_protect_zero_exit_with_bad_acl_is_not_success(private_protection, damage):
    p = private_protection
    if damage == "owner":
        p.facts["owner_trusted"] = False
    elif damage == "unsupported":
        p.facts["unsupported"] = True
    elif damage == "not_protected":
        p.facts["protected"] = False
    elif damage == "owner_readonly":
        p.facts["rules"][0]["mask"] = 0x120089
    elif damage in ("foreign_allow", "foreign_inherit_only"):
        p.facts["rules"].append(dict(trusted=False, current=False, mask=0x1301BF,
                                    allow=True, inherit_only=damage.endswith("only")))
    elif damage == "deny":
        p.facts["rules"][0]["allow"] = False
    else:
        p.facts.clear()
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(p.folder)
    assert len(p.calls) == 1


@pytest.mark.parametrize("reply", [None, [], [None], [{}, {}]])
def test_protect_invalid_acl_reply_is_safe(private_protection, monkeypatch, reply):
    monkeypatch.setattr(index, "inspect_acls", lambda paths: reply)
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(private_protection.folder)


def test_protect_unavailable_inspection_is_safe(private_protection, monkeypatch):
    def unavailable(paths):
        raise index.StatisticsIndexError("ACL_INSPECTION_FAILED")
    monkeypatch.setattr(index, "inspect_acls", unavailable)
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(private_protection.folder)


def test_protect_reparse_is_rejected_before_native_call(private_protection, monkeypatch):
    p = private_protection
    original = type(p.folder).lstat

    def reparse(path):
        if path == p.folder:
            return type("Info", (), dict(st_mode=0o40700, st_file_attributes=0x400))()
        return original(path)

    monkeypatch.setattr(type(p.folder), "lstat", reparse)
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(p.folder)
    assert not p.calls


@pytest.mark.parametrize("stage", ["native", "inspection"])
def test_protect_replacement_is_detected_and_not_removed(private_protection, monkeypatch, stage):
    import subprocess
    from types import SimpleNamespace
    p = private_protection

    def replace():
        p.folder.rename(p.folder.with_name("retained-original"))
        p.folder.mkdir()
        (p.folder / "foreign.txt").write_text("preserve")

    if stage == "native":
        def run(*a, **kw):
            replace()
            return SimpleNamespace(returncode=0)
        monkeypatch.setattr(subprocess, "run", run)
    else:
        def inspect(paths):
            replace()
            return [p.facts]
        monkeypatch.setattr(index, "inspect_acls", inspect)
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(p.folder)
    assert (p.folder / "foreign.txt").read_text() == "preserve"


def test_protect_missing_directory_is_not_created(private_protection):
    p = private_protection
    missing = p.folder / "missing"
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(missing)
    assert not missing.exists() and not p.calls


def test_protect_failure_never_opens_sqlite_or_changes_old_generation(dataset, monkeypatch):
    old = build(dataset)
    directory = dataset[0] / "analytics"
    pointer = (directory / "current.json").read_bytes()
    calls = []

    def failure(path):
        calls.append(path)
        (path / "partial-attempt.txt").write_text("synthetic partial failure")
        raise index.StatisticsIndexError("PRIVATE_INDEX_ACL_FAILED")

    monkeypatch.setattr(index, "protect_directory", failure)
    monkeypatch.setattr(index.sqlite3, "connect", lambda *a, **kw: pytest.fail("No SQLite after ACL failure"))
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        build(dataset)
    assert len(calls) == 1
    assert (directory / "current.json").read_bytes() == pointer
    assert {p.name for p in directory.iterdir() if p.is_dir()} == {old["generation"]}


def test_protect_failure_cleanup_does_not_delete_replaced_staging(dataset, monkeypatch):
    old = build(dataset)
    directory = dataset[0] / "analytics"
    pointer = (directory / "current.json").read_bytes()
    replacements = []

    def failure(path):
        path.rename(directory / "retained-owned-staging")
        path.mkdir()
        marker = path / "foreign.txt"
        marker.write_text("preserve")
        replacements.append(marker)
        raise index.StatisticsIndexError("PRIVATE_INDEX_ACL_FAILED")

    monkeypatch.setattr(index, "protect_directory", failure)
    with pytest.raises(index.StatisticsIndexError, match="^STAGING_PATH_CHANGED$"):
        build(dataset)
    assert replacements[0].read_text() == "preserve"
    assert (directory / "current.json").read_bytes() == pointer
    assert (directory / old["generation"]).is_dir()


def test_protect_failure_cleanup_error_remains_a_failure(dataset, monkeypatch):
    old = build(dataset)
    directory = dataset[0] / "analytics"
    pointer = (directory / "current.json").read_bytes()
    staging = []

    def failure(path):
        staging.append(path)
        raise index.StatisticsIndexError("PRIVATE_INDEX_ACL_FAILED")

    def cleanup(path):
        raise PermissionError("Synthetic cleanup failure")

    monkeypatch.setattr(index, "protect_directory", failure)
    monkeypatch.setattr(index.shutil, "rmtree", cleanup)
    with pytest.raises(PermissionError):
        build(dataset)
    assert staging[0].is_dir()
    assert (directory / old["generation"]).is_dir()
    assert (directory / "current.json").read_bytes() == pointer


@pytest.mark.parametrize("failure", [
    None, "open", "size", "query", "convert", "convert_partial", "empty",
])
def test_windows_user_sid_reads_token_and_releases_resources(monkeypatch, failure):
    import ctypes
    from ctypes import wintypes
    from types import SimpleNamespace

    closed, freed = [], []
    sid_buffer = ctypes.create_unicode_buffer("" if failure == "empty" else "S-1-0-0")

    def open_token(process, access, output):
        assert access == 0x0008
        if failure == "open":
            return False
        ctypes.cast(output, ctypes.POINTER(wintypes.HANDLE))[0] = wintypes.HANDLE(123)
        return True

    def token_information(token, kind, buffer, length, needed):
        assert kind == 1
        if buffer is None:
            ctypes.cast(needed, ctypes.POINTER(wintypes.DWORD))[0] = wintypes.DWORD(32)
            ctypes.set_last_error(87 if failure == "size" else 122)
            return False
        if failure == "query":
            return False
        ctypes.cast(buffer, ctypes.POINTER(wintypes.LPVOID))[0] = wintypes.LPVOID(456)
        return True

    def convert_sid(sid, output):
        assert sid == 456
        if failure == "convert":
            return False
        ctypes.cast(output, ctypes.POINTER(wintypes.LPWSTR))[0] = ctypes.cast(sid_buffer, wintypes.LPWSTR)
        return failure != "convert_partial"

    def close(token):
        closed.append(token.value)
        return True

    def free(pointer):
        freed.append(pointer.value)
        return None

    kernel = SimpleNamespace(GetCurrentProcess=lambda: -1, CloseHandle=close, LocalFree=free)
    security = SimpleNamespace(OpenProcessToken=open_token, GetTokenInformation=token_information,
                               ConvertSidToStringSidW=convert_sid)
    monkeypatch.setattr(ctypes, "WinDLL",
                        lambda name, **kw: kernel if name == "kernel32" else security, raising=False)
    if failure is None:
        assert index._windows_user_sid() == "S-1-0-0"
    else:
        with pytest.raises(ValueError):
            index._windows_user_sid()
    assert closed == ([] if failure == "open" else [123])
    assert len(freed) == (1 if failure in (None, "convert_partial", "empty") else 0)


def test_protect_sid_lookup_failure_is_safe(private_protection, monkeypatch):
    def unavailable():
        raise RuntimeError("SENSITIVE_TOKEN_DETAIL")
    monkeypatch.setattr(index, "_windows_user_sid", unavailable)
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(private_protection.folder)
    assert not private_protection.calls


def test_protect_replayed_false_zero_with_existing_directory_is_rejected(private_protection, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    p = private_protection
    p.facts["protected"] = False
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=b"Successfully processed", stderr=b"CommandNotFoundException"))
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        index.protect_directory(p.folder)
