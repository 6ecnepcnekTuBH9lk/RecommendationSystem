"""Event quality uses source counters, never item counts or source reads."""

from copy import deepcopy
from dataclasses import asdict

import pytest

from Application.analysis_filter import AnalysisFilter
from Application.statistics_presentation import source_event_quality_rows
from test_dataset_statistics import calculate, dataset

__all__ = ['dataset']


def snapshot(source, views, favorites, missing, unmapped, **extra):
    return dict(source_actions=source, diagnostics=(('mapped_view_actions', views),
                ('mapped_favorite_actions', favorites), ('mapped_without_product', missing),
                ('unmapped_actions', unmapped)), **extra)


def test_customer_example_denominator_and_no_mutation():
    result = snapshot(2292725, 2173313, 73978, 1032982, 45434,
                      actions_with_product=1216544, actions_without_product=1076181)
    before = deepcopy(result)
    rows = source_event_quality_rows(result)
    assert [value for _, value in rows[:6]] == [2292725, 2173313, 73978, 2247291, 1214309, 1032982]
    assert rows[6][1] == pytest.approx(100 * 1032982 / 2247291)
    assert f'{rows[6][1]:.2f}' == '45.97'
    assert rows[7][1] == 45434
    assert result == before


def test_all_action_products_and_item_interactions_are_not_denominators():
    result = snapshot(100, 40, 20, 30, 40, actions_with_product=70,
                      actions_without_product=30, view_interactions=500, purchase_interactions=1000)
    rows = source_event_quality_rows(result)
    assert rows[4][1] == 30  # Not 70 all-Action product events.
    assert rows[6][1] == 50.  # Not 30 / 100, nor a share of item interactions.


@pytest.mark.parametrize('unmapped', [0, 10])
def test_no_recognized_events_has_no_percentage(unmapped):
    rows = source_event_quality_rows(snapshot(unmapped, 0, 0, 0, unmapped))
    assert rows[4][1] == rows[5][1] == 0
    assert rows[6][1] is None


@pytest.mark.parametrize('result', [snapshot(3, 1, 1, 3, 1), snapshot(4, 1, 1, 1, 1)])
def test_inconsistent_saved_counters_never_invent_a_negative_difference_or_rate(result):
    rows = source_event_quality_rows(result)
    assert rows[4][1] is rows[6][1] is None
    assert rows[5][1] == dict(result['diagnostics'])['mapped_without_product']


def test_multiple_products_and_product_filter_keep_source_population(dataset):
    # Seven tiny synthetic source records; one VIEW has two product occurrences.
    from test_analysis_filter import reference
    reference(dataset)
    baseline = asdict(calculate(dataset))
    filtered = asdict(calculate(dataset, analysis_filter=AnalysisFilter(nomenclature_types=('Рубашки',))))
    before = deepcopy((baseline, filtered))
    rows = source_event_quality_rows(baseline)
    assert [value for _, value in rows[:6]] == [7, 4, 2, 6, 5, 1]
    assert baseline['view_interactions'] == 4  # Includes the two-product VIEW.
    assert filtered['view_interactions'] < baseline['view_interactions']
    assert source_event_quality_rows(filtered) == rows
    assert (baseline, filtered) == before
