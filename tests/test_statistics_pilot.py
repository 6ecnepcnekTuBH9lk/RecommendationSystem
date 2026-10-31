"""Synthetic source isolation and bounded real-pilot infrastructure tests."""

from dataclasses import asdict, replace
import json
import sqlite3

import pytest

from Application import statistics_pilot as pilot
from Application import statistics_index as index
from Application.dataset_statistics import calculate_statistics
from Application.statistics_query import calculate_indexed_statistics
from Application.analysis_filter import AnalysisFilter
from tests.test_dataset_statistics import dataset as fixture, save, put_entry


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "protect_directory", lambda p: p.chmod(0o700))
    monkeypatch.setattr(index, "protect_directory", lambda p: p.chmod(0o700))
    monkeypatch.setattr(index, "validate_storage_root", lambda p, **kw: index.path_identity(p))
    return fixture.__wrapped__(tmp_path/"real-like")


def plan(source, **kw):
    return pilot.plan_pilot(raw_root=source[0], catalog_path=source[1], days=["2026-01-01"], **kw)


def prepare(source, destination, **kw):
    result = pilot.prepare_pilot(plan=plan(source), raw_root=source[0], catalog_path=source[1], destination=destination, **kw)
    (destination/"index").mkdir()
    return result


def normalize(value):
    value = json.loads(json.dumps(asdict(value)))
    value.pop("calculated_at")
    return value


def test_complete_api_sources_same_88_fields_and_production_untouched(source, tmp_path):
    before = {p:p.read_bytes() for p in (source[0]/"canonical").rglob("*") if p.is_file()}
    directory = tmp_path/"private"
    marker = prepare(source, directory)
    assert marker["scope"] == "TEST_ONLY"
    assert marker["counts"]["profiles"] == 2  # Not all 3 source profiles.
    with sqlite3.connect(directory/"sources/canonical/customers.sqlite") as db:
        assert all(set(json.loads(raw)) == {"birthDate", "sex"} for raw, in db.execute("SELECT raw FROM profiles"))
    built = pilot.build_pilot(directory)
    assert built["counts"]["actions"] == 7
    args = pilot.pilot_paths(directory)
    expected = calculate_statistics(raw_root=args["raw_root"], catalog_path=args["catalog_path"])
    actual = calculate_indexed_statistics(**args)
    assert normalize(actual) == normalize(expected)
    assert {p:p.read_bytes() for p in before} == before
    assert not (source[0]/"index/current.json").exists()
    assert not list((directory/"index").rglob("*-journal"))


@pytest.mark.parametrize("days", [[], ["2026-01-01","2026-01-03"], ["2026-01-02"], ["2026-01-01"]*2])
def test_only_complete_contiguous_paired_days_allowed(source, days):
    with pytest.raises(index.StatisticsIndexError):
        pilot.plan_pilot(raw_root=source[0], catalog_path=source[1], days=days)


def test_manual_only_not_cut_into_arbitrary_row_or_part_sample(source):
    source[2]["actions"] = {}
    save(source[0],source[2])
    with pytest.raises(index.StatisticsIndexError, match="PAIRED"):
        plan(source)


@pytest.mark.parametrize("limit", ["interaction_bytes", "merges_bytes"])
def test_file_budget_preflight_before_reading_rows(source, monkeypatch, limit):
    monkeypatch.setattr(pilot,"iter_export",lambda *a, **kw: pytest.fail("row read"))
    with pytest.raises(index.StatisticsIndexError, match="BUDGET"):
        plan(source, limits=replace(pilot.PilotLimits(), **{limit:1}))


@pytest.mark.parametrize("target", ["existing", "canonical", "production_index", "production_cache"])
def test_unsafe_destinations_rejected_without_writes(source, tmp_path, monkeypatch, target):
    paths = {"existing":tmp_path/"existing", "canonical":source[0]/"canonical/new",
             "production_index":tmp_path/"production/index/new", "production_cache":tmp_path/"production/cache/new"}
    (tmp_path/"existing").mkdir()
    sentinel = tmp_path/"existing/user.txt"
    sentinel.write_text("preserve")
    monkeypatch.setattr(pilot,"DEFAULT_INDEX_DIR",tmp_path/"production/index")
    monkeypatch.setattr(pilot,"DEFAULT_CACHE_DIR",tmp_path/"production/cache")
    with pytest.raises(index.StatisticsIndexError, match="UNSAFE"):
        prepare(source,paths[target])
    assert sentinel.read_text() == "preserve"


def test_changed_revision_plan_aborts_before_directory_creation(source, tmp_path):
    selected = plan(source)
    source[2]["revision"] = "updated"
    save(source[0],source[2])
    with pytest.raises(index.StatisticsIndexError, match="STALE"):
        pilot.prepare_pilot(plan=selected, raw_root=source[0], catalog_path=source[1], destination=tmp_path/"private")
    assert not (tmp_path/"private").exists()


@pytest.mark.parametrize("failure", ["cancel", "disk", "record_budget"])
def test_preparation_failure_cleans_owned_directory(source, tmp_path, monkeypatch, failure):
    destination = tmp_path/"private"
    kwargs = {}
    error = index.StatisticsIndexError
    if failure == "cancel":
        kwargs["cancelled"] = lambda:True
        error = InterruptedError
    elif failure == "disk":
        monkeypatch.setattr(index.shutil,"disk_usage",lambda p:type("Usage",(),{"free":20*index.GIB})())
    else:
        kwargs["limits"] = replace(pilot.PilotLimits(),action_records=1)
    with pytest.raises(error):
        prepare(source,destination,**kwargs)
    assert not destination.exists()
    assert (source[0]/"canonical/catalog.json").exists()


def test_all_aliases_preserve_first_profile_cohort_order(source, tmp_path):
    with sqlite3.connect(source[0]/"canonical/customers.sqlite") as db:
        db.execute("INSERT INTO profiles VALUES('old','',?)", (json.dumps({"birthDate":"1990-01-01","sex":"male"}),))
    directory = tmp_path/"private"
    prepare(source,directory)
    with sqlite3.connect(directory/"sources/canonical/customers.sqlite") as db:
        assert {row[0] for row in db.execute("SELECT id FROM profiles")} == {"old","new","other"}
    pilot.build_pilot(directory)
    args = pilot.pilot_paths(directory)
    selected = AnalysisFilter(collections=("no-match",))
    expected = calculate_statistics(raw_root=args["raw_root"],catalog_path=args["catalog_path"],analysis_filter=selected)
    actual = calculate_indexed_statistics(**args,analysis_filter=selected)
    assert normalize(actual)==normalize(expected)


def test_builder_cancel_manifest_corruption_and_no_automatic_second_attempt(source, tmp_path):
    directory = tmp_path/"private"
    prepare(source,directory)
    with pytest.raises(InterruptedError):
        pilot.build_pilot(directory,cancelled=lambda:True)
    assert not (directory/"index/current.json").exists()
    assert not [p for p in (directory/"index").iterdir() if p.is_dir()]
    with pytest.raises(FileExistsError):
        pilot.build_pilot(directory)
    (directory/"PILOT_ONLY.json").write_text('{"schema":999,"scope":"TEST_ONLY"}')
    with pytest.raises(index.StatisticsIndexError,match="MANIFEST"):
        pilot.pilot_paths(directory)


def test_corrupt_pilot_sqlite_rejected_without_production_fallback(source, tmp_path):
    directory = tmp_path/"private"
    prepare(source,directory)
    built = pilot.build_pilot(directory)
    (directory/"index"/built["generation"]/"statistics.sqlite").write_bytes(b"corrupt")
    with pytest.raises(index.StatisticsIndexError):
        calculate_indexed_statistics(**pilot.pilot_paths(directory))


def test_unselected_daily_files_not_parsed_or_copied(source, tmp_path):
    entry = put_entry(source[0], "actions", [], 5, "2026-01-02T00:00:00+00:00", "2026-01-03T00:00:00+00:00")
    unselected = source[0]/entry["directory"]/"actions_part_001.json"
    unselected.write_text("must not be parsed")
    source[2]["actions"]["2026-01-02"] = entry
    save(source[0],source[2])
    destination = tmp_path/"private"
    prepare(source,destination)
    assert not (destination/"sources"/entry["directory"]).exists()
    built = pilot.build_pilot(destination)
    assert built["counts"]["actions"] == 7
    with pytest.raises(index.StatisticsIndexError,match="STALE_SOURCE"):
        # Even copying a pilot pointer cannot turn it into a production index.
        calculate_indexed_statistics(raw_root=source[0],catalog_path=source[1],index_dir=destination/"index")


def test_cancellation_mid_copy_releases_original_source_lock(source,tmp_path):
    from Application.mindbox.canonical_storage import storage_lock
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 3
    with pytest.raises(InterruptedError):
        prepare(source,tmp_path/"private",cancelled=cancelled)
    assert not (tmp_path/"private").exists()
    with storage_lock(source[0]):
        pass


def test_copy_failure_rolls_back_only_owned_fixture(source,tmp_path,monkeypatch):
    original = (source[0]/"canonical/catalog.json").read_bytes()
    def fail(*args,**kwargs):
        raise OSError("synthetic write failure")
    monkeypatch.setattr(pilot.shutil,"copyfile",fail)
    with pytest.raises(OSError):
        prepare(source,tmp_path/"private")
    assert not (tmp_path/"private").exists()
    assert (source[0]/"canonical/catalog.json").read_bytes()==original


def test_incomplete_or_mismatched_daily_window_not_presented_as_full_day(source):
    from Application.mindbox.training_batch import TrainingBatchError
    source[2]["orders"]["2026-01-01"]["until"] = "2026-01-01T18:00:00+00:00"
    save(source[0],source[2])
    # Existing canonical validation may reject the partial window even before
    # the pilot's complete-pair guard. Both preserve the fail-closed contract.
    with pytest.raises((index.StatisticsIndexError,TrainingBatchError)):
        plan(source)


def test_prepare_acl_failure_stops_before_copy_and_preserves_other_files(source, tmp_path, monkeypatch):
    destination = tmp_path / "new-private"
    foreign = tmp_path / "foreign.txt"
    foreign.write_text("preserve")
    calls = []

    def failure(path):
        calls.append(path)
        raise index.StatisticsIndexError("PRIVATE_INDEX_ACL_FAILED")

    monkeypatch.setattr(pilot, "protect_directory", failure)
    monkeypatch.setattr(pilot.shutil, "copyfile", lambda *a, **kw: pytest.fail("No copy after ACL failure"))
    with pytest.raises(index.StatisticsIndexError, match="^PRIVATE_INDEX_ACL_FAILED$"):
        pilot.prepare_pilot(plan=plan(source), raw_root=source[0],
                            catalog_path=source[1], destination=destination)
    assert len(calls) == 1
    assert not destination.exists()
    assert not (destination / "PILOT_ONLY.json").exists()
    assert foreign.read_text() == "preserve"


def test_prepare_acl_failure_never_cleans_substituted_directory(source, tmp_path, monkeypatch):
    destination = tmp_path / "new-private"
    retained = tmp_path / "retained-owned"

    def failure(path):
        path.rename(retained)
        path.mkdir()
        (path / "foreign.txt").write_text("preserve")
        raise index.StatisticsIndexError("PRIVATE_INDEX_ACL_FAILED")

    monkeypatch.setattr(pilot, "protect_directory", failure)
    with pytest.raises(index.StatisticsIndexError, match="^PILOT_CLEANUP_PATH_CHANGED$"):
        pilot.prepare_pilot(plan=plan(source), raw_root=source[0],
                            catalog_path=source[1], destination=destination)
    assert retained.is_dir()
    assert (destination / "foreign.txt").read_text() == "preserve"
    assert not (destination / "PILOT_ONLY.json").exists()


def test_prepare_partial_acl_failure_reports_cleanup_failure(source, tmp_path, monkeypatch):
    destination = tmp_path / "new-private"

    def failure(path):
        (path / "partial-attempt.txt").write_text("synthetic partial failure")
        raise index.StatisticsIndexError("PRIVATE_INDEX_ACL_FAILED")

    def cleanup(path):
        raise PermissionError("Synthetic cleanup failure")

    monkeypatch.setattr(pilot, "protect_directory", failure)
    monkeypatch.setattr(pilot.shutil, "rmtree", cleanup)
    with pytest.raises(PermissionError):
        pilot.prepare_pilot(plan=plan(source), raw_root=source[0],
                            catalog_path=source[1], destination=destination)
    assert (destination / "partial-attempt.txt").is_file()
    assert not (destination / "PILOT_ONLY.json").exists()
