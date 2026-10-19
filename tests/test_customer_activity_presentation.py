"""UI02 formatting only; no SQLite, models or source exports."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from Application.customer_activity_presentation import ACTIVITY_FIELDS, activity_content
from Application.customer_activity_service import ActivityIssue, ActivityStatus, CustomerActivity
from Application.customer_card_data import ViewState
from Application.customer_history_index import HistoryCoverage


def span(component, since="2022-01-01", until="2026-09-01"):
    return HistoryCoverage(component, datetime.fromisoformat(since).replace(tzinfo=timezone.utc),
                           datetime.fromisoformat(until).replace(tzinfo=timezone.utc), "API")


@pytest.fixture
def activity():
    return CustomerActivity(
        ActivityStatus.READY, "synthetic-canonical", orders=3, units=Decimal("6.5"),
        views=12, favorites=2, days_since_purchase=10, unique_viewed_products=7,
        coverage=(span("orders"), span("actions")),
        orders_coverage_complete=True, actions_coverage_complete=True,
        purchase_as_of_utc=datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc))


def test_exact_eight_tiles_and_removed_explanations(activity):
    content = activity_content(activity)
    assert [f.title for f in content.fields] == [
        "Заказы", "Количество покупок", "Сумма покупок", "Средний чек", "Просмотры", "Избранное",
        "Дней с последней покупки", "Период анализа"]
    assert content.message == ""
    assert content.fields[1].value == "6,5" and content.fields[1].note is None
    assert content.fields[6].note == "По состоянию на 31.08.2026."
    assert content.fields[7].value == "01.01.2022 - 31.08.2026"
    assert activity.unique_viewed_products == 7 and activity.units == Decimal("6.5")


@pytest.mark.parametrize("coverage,expected", [
    ((span("orders"), span("actions")), "01.01.2022 - 31.08.2026"),
    ((span("orders"), span("actions", "2023-02-01", "2026-10-01")), "01.01.2022 - 30.09.2026"),
    ((span("orders", "2026-01-01", "2026-01-02"), span("actions", "2026-01-02", "2026-01-03")), "01.01.2026 - 02.01.2026"),
    ((span("orders"), span("actions"), span("customer_merges", "2000-01-01", "2030-01-01")), "01.01.2022 - 31.08.2026"),
    ((span("orders", "2026-01-01", "2026-02-01"), span("actions", "2026-03-01", "2026-04-01")), None),
    ((span("orders", "2026-01-01", "2026-01-02"), span("orders", "2026-01-03", "2026-01-04"), span("actions")), None),
    ((span("orders", "2026-01-01", "2026-01-01"), span("actions")), None),
    ((span("actions"),), None),  # A claimed complete Orders coverage is missing.
    ((), None),
])
def test_common_confirmed_period_and_exclusive_end(activity, coverage, expected):
    field = activity_content(replace(activity, coverage=coverage)).fields[7]
    assert field.value == expected
    assert field.state is (ViewState.READY if expected is not None else ViewState.UNAVAILABLE)
    assert field.note is None


@pytest.mark.parametrize("orders,actions,expected", [
    (True, False, "01.01.2022 - 31.08.2026"),
    (False, True, "01.02.2023 - 30.09.2026"),
    (False, False, None),
])
def test_only_confirmed_used_sources_define_period(activity, orders, actions, expected):
    activity = replace(activity, orders_coverage_complete=orders, actions_coverage_complete=actions,
                       coverage=(span("orders"), span("actions", "2023-02-01", "2026-10-01")))
    assert activity_content(activity).fields[7].value == expected


def test_adjacent_shuffled_coverage_parts_merge_without_a_gap(activity):
    activity = replace(activity, coverage=(span("orders", "2024-01-01"), span("actions"),
                                           span("orders", "2022-01-01", "2024-01-01")))
    assert activity_content(activity).fields[7].value == "01.01.2022 - 31.08.2026"


def test_naive_coverage_is_unavailable_and_dates_are_not_mutated(activity):
    bad = replace(span("orders"), since_utc=datetime(2022, 1, 1))
    result = replace(activity, coverage=(bad, span("actions")))
    assert activity_content(result).fields[7].state is ViewState.UNAVAILABLE
    local = timezone(timedelta(hours=3))
    shifted = tuple(replace(c, since_utc=c.since_utc.astimezone(local), until_utc=c.until_utc.astimezone(local))
                    for c in activity.coverage)
    result = replace(activity, coverage=shifted)
    assert activity_content(result).fields[7].value == "01.01.2022 - 31.08.2026"
    assert result.coverage == shifted and result.coverage[0].since_utc.utcoffset() == timedelta(hours=3)


def test_hidden_unique_product_issue_does_not_replace_the_analysis_period(activity):
    result = replace(activity, unique_viewed_products=None,
                     issues=(ActivityIssue("unique_viewed_products", "product_identity"),))
    content = activity_content(result)
    assert content.fields[7].value == "01.01.2022 - 31.08.2026"
    assert not any(f.title == "Уникальные просмотренные товары" for f in content.fields)


def test_unavailable_purchase_quantity_has_no_lower_caption(activity):
    result = replace(activity, units=None, issues=(ActivityIssue("units", "quantity_conflict"),))
    field = activity_content(result).fields[1]
    assert field.state is ViewState.UNAVAILABLE and field.note is None


@pytest.mark.parametrize("state", [ViewState.INITIAL, ViewState.LOADING, ViewState.CANCELLED])
def test_placeholder_states_have_no_period_header(state):
    content = activity_content(state=state)
    assert content.message == "" and len(content.fields) == 8
    assert tuple(f.title for f in content.fields) == ACTIVITY_FIELDS
    assert all(f.value is None and f.note is None for f in content.fields)


@pytest.mark.parametrize("status", list(ActivityStatus))
def test_all_backend_states_have_no_separate_period_header(activity, status):
    assert activity_content(replace(activity, status=status)).message == ""
