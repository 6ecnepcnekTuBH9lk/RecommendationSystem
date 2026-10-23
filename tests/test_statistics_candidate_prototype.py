"""Date-candidate proof, not production index implementation."""

from dataclasses import asdict
from decimal import Decimal
import json

import pytest

from Application import dataset_statistics as stats
from Application.analysis_filter import AnalysisFilter
from Application.mindbox.canonical_customers import encode
from tests.statistics_candidate_prototype import CandidatePrototype
from tests.test_dataset_statistics import dataset as fixture, action, order
from tests.test_analysis_filter import reference, sources


@pytest.fixture
def dataset(tmp_path):
    return fixture.__wrapped__(tmp_path)


def result(dataset, selection):
    return stats.calculate_statistics(
        raw_root=dataset[0], catalog_path=dataset[1], analysis_filter=selection
    )


def normalized(value):
    payload = json.loads(json.dumps(asdict(value)))
    payload.pop("calculated_at")
    return payload


@pytest.mark.parametrize(
    "selection",
    [
        AnalysisFilter(),
        AnalysisFilter("2026-01-01", "2026-01-01"),
        AnalysisFilter("2026-02-01", "2026-02-01"),
        AnalysisFilter("2026-01-01", "2026-03-31"),
        AnalysisFilter(nomenclature_types=("Рубашки",)),
        AnalysisFilter(nomenclature_types=("Брюки", "Рубашки")),
        AnalysisFilter(collections=("Осень-зима",)),
        AnalysisFilter(collections=("NOT PRESENT",)),
        AnalysisFilter("2026-02-01", "2026-02-28", ("Брюки",)),
        AnalysisFilter("2026-01-01", "2026-12-31", collections=("Весна-лето",)),
    ],
)
def test_sql_candidates_preserve_every_field_before_dedup_and_product_selection(
    dataset, selection
):
    reference(dataset)
    actions = [
        action("ProsmotrProdukta", ["000001-size", "000001-size"]),
        action("ProsmotrProdukta", ["000002-size"], stamp="2026-02-01T00:00:00Z"),
        action("UnknownWithoutTimestamp"),
    ]
    actions[-1].pop("dateTimeUtc")
    first = order(["CP", "CD"], "same-order")
    second = order(["F", "Return"], "same-order")
    second["firstAction"]["dateTimeUtc"] = "2026-02-01T12:00:00Z"
    second["lines"][0]["product"]["ids"]["offline1C"] = "000002-size"
    fractional = order(["F"], "fractional")
    fractional["lines"][0]["quantity"] = 1.25
    outside = order(["F"], "outside")
    outside["firstAction"]["dateTimeUtc"] = "2030-01-01T00:00:00Z"
    sources(dataset, actions, [first, first, second, fractional, outside])
    root, _, data = dataset
    path = (
        root
        / data["manual_interactions"]["orders"]["directory"]
        / "orders_part_001.json"
    )
    orders = json.loads(path.read_text(), parse_float=Decimal)
    orders["orders"][0]["lines"][0]["priceOfLine"] = Decimal(
        "123.12345678901234567890123456"
    )
    # Make duplicate truly identical after the Decimal edit.
    orders["orders"][1] = orders["orders"][0]
    path.write_text(encode(orders), encoding="utf-8")
    expected = result(dataset, selection)
    prototype = CandidatePrototype(dataset, root / "prototype.sqlite")
    try:
        with prototype.reader(selection):
            actual = result(dataset, selection)
        assert normalized(expected) == normalized(actual)
        assert dict(actual.diagnostics)["orders_outside_statistics_period"] == 1
    finally:
        prototype.close()


def test_invalid_timestamp_not_eagerly_rejected_for_unmapped_action_but_query_error_preserved(
    dataset,
):
    bad = action("Unknown", ["000001-size"])
    bad["dateTimeUtc"] = "not a date"
    sources(dataset, [bad], [])
    prototype = CandidatePrototype(dataset, dataset[0] / "prototype.sqlite")
    try:
        expected = result(dataset, AnalysisFilter())
        with prototype.reader(AnalysisFilter()):
            assert normalized(result(dataset, AnalysisFilter())) == normalized(expected)
        selected = AnalysisFilter("2026-01-01", "2026-01-01")
        with pytest.raises(Exception) as error:
            result(dataset, selected)
        with prototype.reader(selected):
            with pytest.raises(type(error.value)):
                result(dataset, selected)
    finally:
        prototype.close()
