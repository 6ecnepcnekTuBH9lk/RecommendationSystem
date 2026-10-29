"""Controlled local-calendar dates; synthetic sources only, all fields exact."""

from dataclasses import asdict
from datetime import datetime
import json
import sqlite3

import pytest

from Application import dataset_statistics as reference
from Application import statistics_index as index
from Application import statistics_slices as slices
from Application.analysis_filter import AnalysisFilter
from tests.test_analysis_filter import reference as catalog
from tests.test_statistics_index import build, dataset as dataset_fixture, query


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    return dataset_fixture.__wrapped__(tmp_path, monkeypatch)


def instant(day):
    # Noon UTC safely lands on this date in the application's Europe/Moscow zone.
    return datetime.fromisoformat(day + "T12:00:00+00:00")


def profiles(data):
    values = [("new", None, "female"), ("old", "2000-02-29", "male"),
              ("other", "2008-10-24", "male"), ("inactive", "1990-01-01", None),
              ("future", "2027-01-01", "female"), ("bad", "2026-02-30", None),
              ("unknown", None, None), ("band", "2000-10-24", None)]
    with sqlite3.connect(data[0]/"canonical/customers.sqlite") as db:
        db.execute("DELETE FROM profiles")
        db.executemany("INSERT INTO profiles VALUES(?,'',?)", (
            (sid, json.dumps({"birthDate": born, "sex": sex})) for sid, born, sex in values))


def exact(data, day, selection=AnalysisFilter()):
    at = instant(day)
    expected = reference.calculate_statistics(raw_root=data[0], catalog_path=data[1],
                                              analysis_filter=selection, calculated_at=at)
    actual = query(data, selection, calculated_at=at)
    assert len(asdict(actual)) == 90
    assert asdict(actual) == asdict(expected)
    return actual


def test_same_generation_all_fields_over_birthdays_year_leap_future_and_filters(dataset):
    profiles(dataset)
    catalog(dataset)
    built = build(dataset)
    pointer = (dataset[0]/"analytics/current.json").read_bytes()
    folder = dataset[0]/"analytics"/built["generation"]
    before = {p.name: index.component_digest(p) for p in folder.iterdir() if p.is_file()}
    for day in ("2024-02-28", "2024-02-29", "2025-02-28", "2025-03-01",
                "2026-10-23", "2026-10-24", "2026-12-31", "2027-01-01", "2027-01-02"):
        for selected in (AnalysisFilter(), AnalysisFilter(nomenclature_types=("Рубашки",)),
                         AnalysisFilter(collections=("Весна-лето",)),
                         AnalysisFilter("2026-01-01", "2026-01-01"),
                         AnalysisFilter(collections=("missing",))):
            exact(dataset, day, selected)
    assert (dataset[0]/"analytics/current.json").read_bytes() == pointer
    assert {p.name: index.component_digest(p) for p in folder.iterdir() if p.is_file()} == before
    with sqlite3.connect(folder/"statistics.sqlite") as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='profiles'").fetchall()


def test_profile_alias_first_missing_birthday_is_not_replaced_and_no_filter_counts_rows(dataset):
    profiles(dataset)
    catalog(dataset)
    build(dataset)
    all_rows = exact(dataset, "2026-10-24")
    selected = exact(dataset, "2026-10-24", AnalysisFilter(nomenclature_types=("Рубашки",)))
    assert all_rows.customers == 8
    assert selected.customers == 1  # old -> new, first source id is new with NULL birthday
    assert selected.mean_age is None


def test_same_day_cache_and_rollover_never_reuse_yesterdays_age(dataset, monkeypatch, tmp_path):
    profiles(dataset)
    build(dataset)
    at = instant("2026-10-23")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return at.astimezone(tz)

    monkeypatch.setattr(slices, "datetime", Clock)
    args = dict(raw_root=dataset[0], catalog_path=dataset[1], index_dir=dataset[0]/"analytics",
                cache_dir=tmp_path/"results", backend="INDEXED")
    first = json.loads(json.dumps(slices.calculate_cached_statistics(**args)))
    original = reference.calculate_statistics
    with monkeypatch.context() as context:
        context.setattr(reference, "calculate_statistics", lambda **kw: pytest.fail("cache miss"))
        assert slices.calculate_cached_statistics(**args) == first
    at = instant("2026-10-24")
    second = json.loads(json.dumps(slices.calculate_cached_statistics(**args)))
    assert second == json.loads(json.dumps(asdict(original(
        raw_root=dataset[0], catalog_path=dataset[1], calculated_at=at))))
    assert second["mean_age"] != first["mean_age"]
    assert slices.calculate_cached_statistics(**args) == second
    assert len(list((tmp_path/"results").glob("*.json"))) == 2
    # A damaged ready result is rejected and recalculated, not treated as demographics.
    for path in (tmp_path/"results").glob("*.json"):
        path.write_text("broken", encoding="utf-8")
    assert json.loads(json.dumps(slices.calculate_cached_statistics(**args))) == second


@pytest.mark.parametrize("damage", ["missing", "byte", "other_generation"])
def test_demographic_component_corruption_and_generation_mix_fail_closed(dataset, damage):
    first = build(dataset)
    folder = dataset[0]/"analytics"/first["generation"]
    component = folder/"demographics.sqlite"
    if damage == "missing":
        component.unlink()
    elif damage == "byte":
        with component.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"x")
    else:
        second = build(dataset)
        other = dataset[0]/"analytics"/second["generation"]/"demographics.sqlite"
        other.write_bytes(component.read_bytes())
    with pytest.raises(index.StatisticsIndexError):
        query(dataset)


def test_reference_and_indexed_pin_clock_before_work_crosses_midnight(dataset, monkeypatch):
    profiles(dataset)
    build(dataset)
    at = instant("2026-10-23")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return at.astimezone(tz)

    import Application.statistics_query as queries
    monkeypatch.setattr(reference, "datetime", Clock)
    monkeypatch.setattr(queries, "datetime", Clock)

    def progress(_):
        nonlocal at
        at = instant("2026-10-24")

    expected = reference.calculate_statistics(raw_root=dataset[0], catalog_path=dataset[1], progress=progress)
    at = instant("2026-10-23")
    actual = query(dataset, progress=progress)
    assert asdict(actual) == asdict(expected)
    assert actual.calculated_at == instant("2026-10-23").isoformat()


def test_no_customers_across_day_change(dataset):
    (dataset[0]/"canonical/customers.sqlite").unlink()
    build(dataset)
    assert exact(dataset, "2026-12-31").customers is None
    assert exact(dataset, "2027-01-01").customers is None


def test_date_and_timezone_only_change_slice_not_generation_provenance(dataset):
    first, _ = slices.capture_sources(dataset[0], dataset[1], AnalysisFilter(), calculated_at=instant("2026-10-23"))
    second, _ = slices.capture_sources(dataset[0], dataset[1], AnalysisFilter(), calculated_at=instant("2026-10-24"))
    second["timezone"] = "different local zone"
    assert index.source_version(first) == index.source_version(second)
    assert slices._digest(first) != slices._digest(second)


def test_cancel_during_demographic_digest_does_not_become_index_unavailable(dataset):
    built = build(dataset)
    provenance, _ = slices.capture_sources(dataset[0], dataset[1], AnalysisFilter())

    def stop():
        raise InterruptedError("cancel")

    with pytest.raises(InterruptedError):
        index.open_generation(dataset[0]/"analytics", provenance, check=stop)
    assert index.read_pointer(dataset[0]/"analytics")["generation"] == built["generation"]


@pytest.mark.parametrize("source", ["customers", "merges"])
def test_source_change_requires_new_bound_generation_then_all_fields_match(dataset, source):
    profiles(dataset)
    first = build(dataset)
    if source == "customers":
        with sqlite3.connect(dataset[0]/"canonical/customers.sqlite") as db:
            db.execute("UPDATE profiles SET raw=? WHERE id='new'", (json.dumps({"birthDate":"1980-01-01"}),))
    else:
        entry = dataset[2]["customer_merges"]
        part = dataset[0]/entry["directory"]/"customer_merges_part_001.json"
        payload = json.loads(part.read_text())
        next(iter(payload.values()))[0]["resultingCustomer"]["ids"]["mindboxId"] = "other"
        part.write_text(json.dumps(payload))
    with pytest.raises(index.StatisticsIndexError, match="STALE_SOURCE"):
        query(dataset, calculated_at=instant("2026-10-24"))
    second = build(dataset)
    assert first["generation"] != second["generation"]
    exact(dataset, "2026-10-24")
    exact(dataset, "2026-10-25", AnalysisFilter(collections=("missing",)))


def test_cancel_profile_iteration_drops_temporary_cohort_and_closes_reader(dataset):
    from Application.statistics_facts import profile_result
    built = build(dataset)
    provenance, _ = slices.capture_sources(dataset[0], dataset[1], AnalysisFilter())
    db, _ = index.open_generation(dataset[0]/"analytics", provenance)
    try:
        cohort = {str(r[0]) for r in db.execute("SELECT customer FROM demographics.profiles")}

        def stop():
            raise InterruptedError("cancel")

        with pytest.raises(InterruptedError):
            profile_result(db, provenance, None, instant("2026-10-24").date(), stop, None, cohort=cohort)
        assert not db.execute("SELECT name FROM sqlite_temp_master WHERE name='cohort'").fetchall()
    finally:
        db.close()
    assert index.read_pointer(dataset[0]/"analytics")["generation"] == built["generation"]
