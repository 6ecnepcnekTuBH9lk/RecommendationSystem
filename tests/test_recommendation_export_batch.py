"""Frozen purchase preparation vs existing consensus on synthetic SQLite only."""

from contextlib import closing
from dataclasses import replace
import sqlite3
from types import SimpleNamespace

import pytest

from Application import recommendation_export_batch as batch
from Application.customer_propensity_service import PropensityStatus
from Application.recommendation_export_records import ExportError, ExportCancelled
from tests.test_hybrid_recommendations import (
    catalog as catalog_fixture,
    sqlite_history as history_fixture,
)


@pytest.fixture
def catalog(tmp_path):
    return catalog_fixture.__wrapped__(tmp_path)


@pytest.fixture
def sqlite_history(tmp_path, monkeypatch, catalog):
    return history_fixture.__wrapped__(tmp_path, monkeypatch, catalog)


@pytest.mark.parametrize(
    "status",
    [
        "CP",
        "delivering",
        "F",
        "Mindbox-Tech-Cancellation",
        "otmenen",
        "Return",
        "CD",
        "unknown",
    ],
)
def test_batch_purchase_evidence_is_identical_and_does_not_leak_between_customers(
    sqlite_history, catalog, status
):
    root, service, path, sources = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE order_lines SET raw_status=?", (status,))
    codes = frozenset({"000001", "000002", "old-long-unresolvable"})
    expected = service.get("canonical", catalog, raw_root=root, historical_codes=codes)
    with batch.history._read_index(path) as db:
        ready = batch.history._inspect_connection(db, sources)
        db.row_factory = sqlite3.Row
        reader = batch._SnapshotPurchases(db, ready, catalog, codes, lambda: False)
        assert (
            reader.get("canonical", catalog, raw_root=root, historical_codes=codes)
            == expected
        )
        assert (
            reader.get(
                "other", catalog, raw_root=root, historical_codes=codes
            ).purchased
            == frozenset()
        )
        assert (
            reader.get("canonical", catalog, raw_root=root, historical_codes=codes)
            == expected
        )
        assert reader.current.purchased == set() and reader.current.statuses == {}


def test_batch_conflicting_snapshots_match_card_and_manual_cancel(
    sqlite_history, catalog
):
    root, service, path, sources = sqlite_history
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "INSERT INTO variants SELECT 'changed',event_key,?,canonical_customer_id,source_customer_id,kind,event_us,quality FROM variants WHERE variant_key='variant-64'",
            (bytes([1]) * 32,),
        )
        db.execute(
            "INSERT INTO order_lines SELECT 'changed-line','changed',source_line_key,line_id,source_line_id,is_synthetic,ordinal,line_number,product_name,quantity,'otmenen',price_of_line,base_price_per_item,quality FROM order_lines WHERE variant_key='variant-64'"
        )
        db.execute(
            "INSERT INTO product_occurrences SELECT 'changed',role,ordinal,selected_namespace,selected_value,quality FROM product_occurrences WHERE variant_key='variant-64'"
        )
    codes = frozenset({"000001", "000002"})
    expected = service.get("canonical", catalog, raw_root=root, historical_codes=codes)
    assert expected.uncertain == {"000002"}
    with batch.history._read_index(path) as db:
        ready = batch.history._inspect_connection(db, sources)
        db.row_factory = sqlite3.Row
        reader = batch._SnapshotPurchases(db, ready, catalog, codes, lambda: False)
        assert (
            reader.get("canonical", catalog, raw_root=root, historical_codes=codes)
            == expected
        )
        with pytest.raises(ExportCancelled):
            reader.get(
                "canonical",
                catalog,
                raw_root=root,
                historical_codes=codes,
                cancelled=lambda: True,
            )
        with pytest.raises(ExportError, match="INCOMPATIBLE_PURCHASE_SCOPE"):
            reader.get(
                "canonical", catalog, raw_root=root, historical_codes={"different"}
            )
        assert reader.is_current(expected, raw_root=root)
        assert not reader.is_current(
            replace(expected, generation="changed"), raw_root=root
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"audience": "profiles"},
        {"limit": 0},
        {"limit": True},
        {"batch_size": 0},
        {"batch_size": 1001},
    ],
)
def test_invalid_audience_and_limits_fail_before_sources(kwargs):
    with pytest.raises(ExportError, match="INVALID_INPUT"):
        batch.RecommendationBatchSession(**kwargs)


def test_missing_propensity_never_fabricates_probability_and_only_bpr_ids_are_used():
    session = batch.RecommendationBatchSession()
    session.propensity_status = PropensityStatus.KEY_UNAVAILABLE
    result = session._prediction("synthetic")
    assert (
        result.probability is None and result.status is PropensityStatus.KEY_UNAVAILABLE
    )
    session.index = SimpleNamespace(
        _groups={"C2": ("M2",), "C1": ("M1",), "ambiguous": ("M3", "M4")}
    )
    assert list(session._ids()) == ["C1", "C2"]


@pytest.mark.parametrize(
    "changed",
    [
        "identity",
        "profile_identity",
        "profile_db",
        "history_source",
        "history_db",
        "catalog",
        "rules",
    ],
)
def test_batch_guard_rejects_every_source_change(monkeypatch, changed):
    from Application.model import customer_identity_bridge

    session = batch.RecommendationBatchSession()
    versions = dict(
        identity="i",
        profile_identity="p",
        profile_db="pd",
        history_source="h",
        history_db="hd",
        catalog=(1, 2, 3),
        categories=(4, 5, 6),
        rules=(7, 8, 9),
    )
    session.root = "root"
    session.card = SimpleNamespace(model_dir="model")
    session.identity_sources = SimpleNamespace(key=versions["identity"])
    session.profile_key, session.profile_version = (
        versions["profile_identity"],
        versions["profile_db"],
    )
    session.profile_path = "profiles"
    session.history_path = "history"
    session.history_sources = SimpleNamespace(version=versions["history_source"])
    session.history_version = versions["history_db"]
    session.catalog = SimpleNamespace(path="catalog", categories_path="categories")
    session.hybrid = SimpleNamespace(rules_path="rules")
    session.catalog_version = (versions["catalog"], versions["categories"])
    session.rules_version = versions["rules"]
    monkeypatch.setattr(
        customer_identity_bridge,
        "_capture_sources",
        lambda *a: SimpleNamespace(key=versions["identity"]),
    )
    monkeypatch.setattr(
        batch.profiles,
        "_identity_source",
        lambda *a: (versions["profile_identity"], None, None),
    )
    monkeypatch.setattr(
        batch.profiles, "_database_version", lambda *a: versions["profile_db"]
    )
    monkeypatch.setattr(
        batch.history,
        "_capture_sources",
        lambda *a: SimpleNamespace(version=versions["history_source"]),
    )
    monkeypatch.setattr(
        batch.history, "_database_version", lambda *a: versions["history_db"]
    )
    monkeypatch.setattr(batch, "signature", lambda path: versions[path])
    session.guard()
    versions[changed] = "changed"
    with pytest.raises(ExportError, match="SOURCES_CHANGED"):
        session.guard()
