"""External archive populated only by tiny generated CSV fixtures."""

import csv
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json

import pytest

from Application.evaluation import catalog_snapshots as api
from Application.evaluation.catalog_snapshots import (
    CatalogSnapshotError, HistoricalCatalogArchive, SnapshotProvenance, SourceOrigin,
)
from Application.evaluation.hybrid_provenance import Validity
from Application.recommendation_catalog import FIELDS


STAMP = datetime(2026, 6, 1, 9, 30, tzinfo=timezone.utc)
BASE = {"ВидНоменклатуры": "Брюки", "ПолНоменклатуры": "Мужской", "Коллекция": "NOS",
        "КатегорияНаСайте": "001", "НазваниеНаСайте": "Тестовый товар", "Остаток": "2,50"}


def provenance(stamp=STAMP, **changes):
    return replace(SnapshotProvenance(stamp.date(), stamp, stamp + timedelta(days=90),
                                      stamp + timedelta(days=90), stamp + timedelta(days=90),
                                      SourceOrigin.HISTORICAL_1C, SourceOrigin.CURRENT_1C, SourceOrigin.CURRENT_1C), **changes)


def inputs(folder, rows=None, categories=None):
    folder.mkdir(parents=True, exist_ok=True)
    nom, cat = folder / "nomenclature.csv", folder / "site_categories.csv"
    with nom.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, delimiter="|", fieldnames=("КодНоменклатуры", *FIELDS, "Остаток"))
        writer.writeheader()
        for code, changes in rows if rows is not None else [("000001", {})]:
            writer.writerow({"КодНоменклатуры": code, **BASE, **changes})
    with cat.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter="|")
        writer.writerow(("КодКатегории", "НазваниеКатегории", "КодРодительскойКатегории"))
        writer.writerows(categories if categories is not None else [("000", "Одежда", ""), ("001", "Брюки", "000")])
    return nom, cat


@pytest.fixture
def archive(tmp_path):
    return HistoricalCatalogArchive(tmp_path / "archive")


def register(archive, tmp_path, *, rows=None, categories=None, metadata=None):
    return archive.register(*inputs(tmp_path / "source", rows, categories), metadata or provenance())


def test_register_roundtrip_raw_hashes_and_leading_zeros(archive, tmp_path):
    nom, cat = inputs(tmp_path / "source")
    original = (nom.read_bytes(), cat.read_bytes())
    snapshot = archive.register(nom, cat, provenance())
    assert snapshot.known_codes == snapshot.in_stock_codes == {"000001"}
    assert str(snapshot.stocks["000001"]) == "2.50"
    assert snapshot.category_paths["001"] == {"000", "001"}
    assert "Брюки" not in snapshot.category_paths
    assert snapshot.diagnostics["nomenclature_rows"] == 1
    assert snapshot.diagnostics["known_products"] == 1
    manifest = json.loads(snapshot.manifest_json)
    for name, content in zip(("nomenclature.csv", "site_categories.csv"), original):
        assert manifest["files"][name] == {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        assert (archive.root / snapshot.info.snapshot_id / name).read_bytes() == content
    assert len(manifest["normalized_sha256"]) == 64
    assert (nom.read_bytes(), cat.read_bytes()) == original
    assert archive.load(snapshot.info.snapshot_id).manifest_json == snapshot.manifest_json


def test_multiple_dates_and_versions_never_select_silently(archive, tmp_path):
    first = register(archive, tmp_path)
    second_time = STAMP + timedelta(hours=2)
    second = register(archive, tmp_path, metadata=provenance(second_time))
    july = register(archive, tmp_path, metadata=provenance(STAMP.replace(month=7)))
    assert archive.available_dates() == (date(2026, 6, 1), date(2026, 7, 1))
    assert len(archive.list_snapshots(STAMP.date())) == 2
    with pytest.raises(CatalogSnapshotError, match="AMBIGUOUS"):
        archive.load_for_date(STAMP.date())
    assert archive.load_for_date(STAMP.date(), snapshot_id=second.info.snapshot_id).info == second.info
    assert archive.load_for_date(date(2026, 7, 1)).info == july.info
    assert archive.load(first.info.snapshot_id).info == first.info
    with pytest.raises(CatalogSnapshotError, match="DATE_MISMATCH"):
        archive.load_for_date(date(2026, 7, 1), snapshot_id=first.info.snapshot_id)
    with pytest.raises(CatalogSnapshotError, match="AMBIGUOUS"):
        archive.load_for_cutoff(STAMP)
    assert archive.load_for_cutoff(STAMP, snapshot_id=first.info.snapshot_id).info == first.info


def test_exact_time_utc_and_unknown_time(archive, tmp_path):
    local = STAMP.astimezone(timezone(timedelta(hours=3)))
    p = provenance(inventory_as_of=local)
    snapshot = register(archive, tmp_path, metadata=p)
    assert snapshot.info.provenance.inventory_as_of == STAMP
    assert archive.load_for_cutoff(local).info == snapshot.info
    with pytest.raises(CatalogSnapshotError, match="CUTOFF_MISMATCH"):
        archive.load_for_cutoff(STAMP + timedelta(seconds=1))
    unknown = register(archive, tmp_path, metadata=provenance(inventory_as_of=None))
    assert unknown.info.provenance.inventory_as_of is None
    assert unknown.assess_cutoff(STAMP).status is Validity.UNAVAILABLE
    with pytest.raises(CatalogSnapshotError, match="UNKNOWN_INVENTORY_TIME"):
        archive.load_for_cutoff(STAMP, snapshot_id=unknown.info.snapshot_id)
    with pytest.raises(CatalogSnapshotError, match="AWARE_TIMESTAMP"):
        snapshot.assess_cutoff(STAMP.replace(tzinfo=None))


def test_always_proxy_later_metadata_and_even_same_day_exports(archive, tmp_path):
    snapshot = register(archive, tmp_path)
    assessment = snapshot.assess_cutoff(STAMP)
    assert snapshot.validity is assessment.status is Validity.PROXY
    assert set(assessment.reasons) == {"POINT_IN_TIME_UNVERIFIED", "LATER_ATTRIBUTES", "LATER_CATEGORIES",
                                       "RETROSPECTIVE_EXPORT", "CATEGORY_EXPORT_TIME_UNKNOWN"}
    assert json.loads(snapshot.manifest_json)["limitations"] == list(assessment.reasons)
    p = provenance(exported_at=STAMP, attributes_observed_at=STAMP, categories_observed_at=STAMP)
    assert register(archive, tmp_path, metadata=p).assess_cutoff(STAMP).status is Validity.PROXY
    with pytest.raises((FrozenInstanceError, TypeError)):
        snapshot.validity = Validity.STRICT


def test_idempotence_and_changed_content_create_distinct_immutable_versions(archive, tmp_path):
    first = register(archive, tmp_path)
    before = (archive.root / first.info.snapshot_id / "nomenclature.csv").read_bytes()
    assert register(archive, tmp_path).info == first.info
    second = register(archive, tmp_path, rows=[("000001", {"Остаток": "3"})])
    assert second.info.snapshot_id != first.info.snapshot_id
    assert len(archive.list_snapshots()) == 2
    assert (archive.root / first.info.snapshot_id / "nomenclature.csv").read_bytes() == before


@pytest.mark.parametrize("damage", ["json", "version", "strict", "bool", "diagnostics", "extra", "provenance"])
def test_malformed_manifest_never_authorizes_snapshot(archive, tmp_path, damage):
    snapshot = register(archive, tmp_path)
    path = archive.root / snapshot.info.snapshot_id / "manifest.json"
    body = json.loads(path.read_text())
    if damage == "json":
        path.write_text("{")
    else:
        if damage == "version": body["format_version"] = 99
        elif damage == "strict": body["validity"] = "STRICT"
        elif damage == "bool": body["validity"] = True
        elif damage == "diagnostics": body["diagnostics"]["known_products"] = 999
        elif damage == "extra": body["path"] = "outside"
        else: body["provenance"]["inventory_as_of"] = "invalid"
        path.write_text(json.dumps(body))
    with pytest.raises(CatalogSnapshotError, match="MANIFEST"):
        archive.load(snapshot.info.snapshot_id)


def test_tampered_csv_and_repeat_registration_do_not_replace_saved_data(archive, tmp_path):
    snapshot = register(archive, tmp_path)
    path = archive.root / snapshot.info.snapshot_id / "nomenclature.csv"
    path.write_bytes(b"damaged")
    with pytest.raises(CatalogSnapshotError, match="CHECKSUM"):
        archive.load(snapshot.info.snapshot_id)
    with pytest.raises(CatalogSnapshotError, match="CHECKSUM"):
        register(archive, tmp_path)
    assert path.read_bytes() == b"damaged"
    assert not list(archive.root.glob(".staging-*"))


@pytest.mark.parametrize("content", [b"wrong,header\nx,y\n", b"wrong|header\nx|y\n", b"\xff\xfe\x00",
                                    'КодНоменклатуры|КодНоменклатуры\na|b\n'.encode(),
                                    'КодНоменклатуры|ВидНоменклатуры\n"unterminated'.encode()])
def test_bad_csv_is_not_published(archive, tmp_path, content):
    nom, cat = inputs(tmp_path / "source")
    nom.write_bytes(content)
    with pytest.raises(CatalogSnapshotError, match="CSV"):
        archive.register(nom, cat, provenance())
    assert list(archive.root.iterdir()) == []


@pytest.mark.parametrize("code", [" 000001", "00 001", "000001\t", "-", "null"])
def test_invalid_product_codes_are_not_repaired(archive, tmp_path, code):
    with pytest.raises(CatalogSnapshotError, match="PRODUCT_CODE"):
        register(archive, tmp_path, rows=[(code, {})])


def test_invalid_and_zero_stocks_are_diagnosed_without_imputation(archive, tmp_path):
    values = ["", "-1", "bad", "NaN", "0", "1 234,50"]
    snapshot = register(archive, tmp_path, rows=[(f"{n:06d}", {"Остаток": v}) for n, v in enumerate(values)])
    assert snapshot.diagnostics["invalid_stock_rows"] == 4
    assert snapshot.in_stock_codes == {"000005"}
    assert snapshot.stocks["000000"] is None
    assert "INVALID_STOCK" in snapshot.exclusions["000001"]
    assert "ZERO_STOCK" in snapshot.exclusions["000004"]
    assert snapshot.stocks["000005"] == 1234.5


def test_duplicates_conflicts_and_multi_category_are_not_merged(archive, tmp_path):
    rows = [("000001", {}), ("000001", {}), ("000002", {}), ("000002", {"Марка": "other"}),
            ("000003", {}), ("000003", {"КатегорияНаСайте": "000"})]
    snapshot = register(archive, tmp_path, rows=rows)
    assert snapshot.known_codes == {"000001", "000002", "000003"}
    assert set(snapshot.items) == snapshot.in_stock_codes == {"000001"}
    assert snapshot.diagnostics["duplicate_rows"] == 3
    assert snapshot.diagnostics["ambiguous_codes"] == 2
    assert snapshot.diagnostics["multi_category_codes"] == 1
    assert snapshot.exclusions["000003"] == ("AMBIGUOUS_CODE",)


def test_category_codes_unknown_parent_cycles_empty_and_missing_attributes(archive, tmp_path):
    categories = [("000", "Root", ""), ("001", "Valid", "000"), ("002", "Broken", "999"),
                  ("003", "Cycle A", "004"), ("004", "Cycle B", "003")]
    rows = [("000001", {"КатегорияНаСайте": "Valid"}), ("000002", {"КатегорияНаСайте": "002"}),
            ("000003", {"КатегорияНаСайте": "003"}), ("000004", {"КатегорияНаСайте": ""}),
            ("000005", {"ВидНоменклатуры": "", "ПолНоменклатуры": "", "Коллекция": ""})]
    snapshot = register(archive, tmp_path, rows=rows, categories=categories)
    assert snapshot.diagnostics["unknown_category_rows"] == 1  # A label is not a category code.
    assert snapshot.diagnostics["invalid_category_path_rows"] == 2
    assert snapshot.diagnostics["invalid_categories"] == 3
    assert snapshot.diagnostics["missing_parent_refs"] == 1
    assert snapshot.diagnostics["empty_category_rows"] == 1
    assert "MISSING_GENDER" in snapshot.exclusions["000005"]
    assert set(snapshot.category_paths) == {"000", "001"}
    # In-stock is availability only, not a duplicated production eligibility filter.
    assert "000005" in snapshot.in_stock_codes


@pytest.mark.parametrize("failure", ["copy", "rename", "manifest_flush"])
def test_atomic_publication_preserves_existing_versions(archive, tmp_path, monkeypatch, failure):
    first = register(archive, tmp_path)
    before = {p.name: p.read_bytes() for p in (archive.root / first.info.snapshot_id).iterdir()}
    def fail(*args, **kwargs):
        raise OSError("synthetic failure")
    if failure == "manifest_flush":
        original = api.os.fsync
        calls = 0
        def fail_manifest(fd):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("synthetic manifest flush failure")
            return original(fd)
        monkeypatch.setattr(api.os, "fsync", fail_manifest)
    else:
        monkeypatch.setattr(api.shutil if failure == "copy" else api.os, "copyfileobj" if failure == "copy" else "rename", fail)
    with pytest.raises(OSError):
        register(archive, tmp_path, rows=[("000002", {})])
    assert {p.name: p.read_bytes() for p in (archive.root / first.info.snapshot_id).iterdir()} == before
    assert [p.name for p in archive.root.iterdir()] == [first.info.snapshot_id]


def test_load_selected_id_does_not_inspect_other_versions(archive, tmp_path):
    first = register(archive, tmp_path)
    second = register(archive, tmp_path, metadata=provenance(STAMP.replace(month=7)))
    (archive.root / second.info.snapshot_id / "manifest.json").write_text("invalid")
    assert archive.load(first.info.snapshot_id).info == first.info


def test_external_root_production_sentinels_row_limits_and_immutability(tmp_path, monkeypatch):
    project = tmp_path / "project"
    nom, cat = inputs(project / "input_data")
    before = (nom.read_bytes(), cat.read_bytes())
    monkeypatch.setattr(api, "PROJECT_ROOT", project)
    with pytest.raises(CatalogSnapshotError, match="EXTERNAL_ARCHIVE"):
        HistoricalCatalogArchive(project / "input_data" / "historical")
    archive = HistoricalCatalogArchive(tmp_path / "external")
    snapshot = archive.register(nom, cat, provenance())
    assert (nom.read_bytes(), cat.read_bytes()) == before
    with pytest.raises(TypeError):
        snapshot.items["000001"] = None
    with pytest.raises(TypeError):
        snapshot.diagnostics["known_products"] = 2
    small = HistoricalCatalogArchive(tmp_path / "small", max_rows=1)
    with pytest.raises(CatalogSnapshotError, match="ROW_LIMIT"):
        small.register(nom, cat, provenance())
    assert list(small.root.iterdir()) == []


def test_empty_archive_and_invalid_paths_dates(tmp_path):
    archive = HistoricalCatalogArchive(tmp_path / "external")
    assert archive.list_snapshots() == archive.available_dates() == ()
    assert not archive.root.exists()
    for value in ("../escape", "not-an-id", None):
        with pytest.raises(CatalogSnapshotError, match="SNAPSHOT_ID"):
            archive.load(value)
    with pytest.raises(CatalogSnapshotError, match="CALENDAR_DATE"):
        archive.load_for_date(STAMP)
    with pytest.raises(CatalogSnapshotError, match="AWARE_TIMESTAMP"):
        provenance(inventory_as_of=STAMP.replace(tzinfo=None))


def test_duplicate_manifest_keys_and_git_storage_are_rejected(archive, tmp_path):
    snapshot = register(archive, tmp_path)
    path = archive.root / snapshot.info.snapshot_id / "manifest.json"
    content = path.read_text()
    path.write_text('{"format_version": 1,' + content[1:])
    with pytest.raises(CatalogSnapshotError, match="MANIFEST"):
        archive.load(snapshot.info.snapshot_id)
    git = tmp_path / "other-repo"
    (git / ".git").mkdir(parents=True)
    with pytest.raises(CatalogSnapshotError, match="GIT_WORKTREE"):
        HistoricalCatalogArchive(git / "data")


def test_semantic_checksum_is_separate_from_original_file_checksum(archive, tmp_path):
    nom, cat = inputs(tmp_path / "source")
    first = archive.register(nom, cat, provenance())
    nom.write_bytes(nom.read_bytes().replace(b"\r\n", b"\n"))
    second = archive.register(nom, cat, provenance())
    assert first.info.snapshot_id != second.info.snapshot_id
    assert json.loads(first.manifest_json)["normalized_sha256"] == json.loads(second.manifest_json)["normalized_sha256"]


def test_declared_dates_origins_and_missing_provenance_are_not_guessed():
    with pytest.raises(CatalogSnapshotError, match="DATE_MISMATCH"):
        provenance(inventory_date=date(2026, 7, 1))
    with pytest.raises(CatalogSnapshotError, match="SOURCE_ORIGIN"):
        provenance(inventory_origin="1C")
    with pytest.raises(CatalogSnapshotError, match="PROVENANCE"):
        SnapshotProvenance.from_dict({"inventory_date": "2026-06-01", "strict": True})


def test_category_duplicates_and_blank_product_rows_are_explicit(archive, tmp_path):
    snapshot = register(archive, tmp_path, rows=[("", {}), ("000001", {})],
                        categories=[("001", "A", ""), ("001", "A", ""), ("001", "B", "")])
    assert snapshot.diagnostics["blank_code_rows"] == 1
    assert snapshot.diagnostics["duplicate_category_rows"] == 2
    assert snapshot.diagnostics["ambiguous_category_codes"] == 1
    assert "001" not in snapshot.category_paths
    assert "INVALID_CATEGORY_PATH" in snapshot.exclusions["000001"]


def test_changed_source_during_copy_is_not_published(archive, tmp_path, monkeypatch):
    nom, cat = inputs(tmp_path / "source")
    original = api.shutil.copyfileobj
    def copy_and_change(source, target, length):
        original(source, target, length)
        with nom.open("ab") as stream:
            stream.write(b"\n")
    monkeypatch.setattr(api.shutil, "copyfileobj", copy_and_change)
    with pytest.raises(CatalogSnapshotError, match="SOURCE_CHANGED"):
        archive.register(nom, cat, provenance())
    assert list(archive.root.iterdir()) == []
