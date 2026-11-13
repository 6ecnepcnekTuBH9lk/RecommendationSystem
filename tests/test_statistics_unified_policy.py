import csv
from decimal import Decimal
from pathlib import Path

import pytest

from Application.statistics_unified_policy import (
    STATUS_CATALOG,
    EvidenceQuality as Quality,
    Measured,
    StatusGroup,
    classify_status_code,
    is_purchase_candidate,
    qualify_quantity,
)


def test_catalog_matches_all_inventory_metadata():
    path = Path(__file__).resolve().parents[1] / "docs" / "statistics_status_inventory.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["external_id"]: row for row in csv.DictReader(stream)}
    assert len(STATUS_CATALOG) == 43
    assert set(STATUS_CATALOG) == set(rows)
    for code, row in rows.items():
        definition = STATUS_CATALOG[code]
        assert definition.external_id == code
        assert definition.name == (row["mindbox_name"] or None)
        assert definition.mindbox_category == (row["mindbox_category"] or None)
        assert definition.group == row["current_status_group"]
        assert definition.metadata_confirmation == row["metadata_confirmation"]
        assert definition.metadata_source is None or definition.metadata_source in row["metadata_source"]
        assert (definition.metadata_source is None) == (code == "Mindbox-Tech-Cancellation")
        assert definition.purchase_candidate == (code in {"F", "sold", "CP"})
        assert definition.cancelled == (row["current_status_group"] == "CANCELLED")
        assert definition.returned == (row["current_status_group"] == "RETURNED")
        assert classify_status_code(code) == (Measured(code, Quality.KNOWN), definition.group)


@pytest.mark.parametrize("raw,quality,reason,value", [
    ("f", Quality.KNOWN, "UNKNOWN_CODE", "f"),
    ("future-status", Quality.KNOWN, "UNKNOWN_CODE", "future-status"),
    (None, Quality.UNKNOWN, "MISSING_CODE", None),
    ("", Quality.UNKNOWN, "MISSING_CODE", None),
    (" F", Quality.INVALID, "INVALID_CODE", None),
    ("F ", Quality.INVALID, "INVALID_CODE", None),
    ("F\x00X", Quality.INVALID, "INVALID_CODE", None),
    ("\u200bF", Quality.INVALID, "INVALID_CODE", None),
    (True, Quality.INVALID, "INVALID_CODE", None),
    (0, Quality.INVALID, "INVALID_CODE", None),
    ({"status": "F"}, Quality.INVALID, "INVALID_CODE", None),
])
def test_unknown_missing_and_invalid_status(raw, quality, reason, value):
    assert classify_status_code(raw) == (Measured(value, quality, (reason,)), StatusGroup.UNKNOWN)
    assert not is_purchase_candidate(raw, 1)


@pytest.mark.parametrize("raw,quality,reason,value", [
    (0, Quality.KNOWN, "ZERO_QUANTITY", Decimal(0)),
    (-1, Quality.INVALID, "NEGATIVE_QUANTITY", None),
    (Decimal("0.01"), Quality.INVALID, "FRACTIONAL_QUANTITY", None),
    (None, Quality.UNKNOWN, "MISSING_QUANTITY", None),
    ("1", Quality.INVALID, "INVALID_QUANTITY", None),
    (True, Quality.INVALID, "INVALID_QUANTITY", None),
    (Decimal("NaN"), Quality.INVALID, "INVALID_QUANTITY", None),
    (float("inf"), Quality.INVALID, "INVALID_QUANTITY", None),
])
def test_unqualified_quantity_keeps_reason(raw, quality, reason, value):
    assert qualify_quantity(raw) == Measured(value, quality, (reason,))
    assert not is_purchase_candidate("F", raw)


@pytest.mark.parametrize("value", [2, 2.0, Decimal("2.000")])
def test_only_positive_integral_quantities_create_candidates(value):
    assert qualify_quantity(value) == Measured(Decimal(2), Quality.KNOWN)
    assert all(is_purchase_candidate(code, value) for code in ("F", "sold", "CP"))
    assert not is_purchase_candidate("delivering", value)


def test_measured_zero_and_absence_are_distinct():
    assert Measured(Decimal(0), Quality.KNOWN).value == 0
    assert Measured(None, Quality.UNKNOWN, ("MISSING",)).value is None
    with pytest.raises(ValueError):
        Measured(None, Quality.KNOWN)
    with pytest.raises(ValueError):
        Measured(Decimal(0), Quality.UNKNOWN, ("MISSING",))
