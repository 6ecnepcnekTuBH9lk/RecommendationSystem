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


@pytest.mark.parametrize("start,end,expected", [
    ("2022-01-01T00:00:00+00:00", "2023-01-01T00:00:00+00:00", "01.01.2022 — 31.12.2022"),
    ("2026-01-01T00:00:00Z", "2026-09-01T00:00:00Z", "01.01.2026 — 31.08.2026"),
    ("2026-08-31T00:00:00+12:00", "2026-09-01T00:00:00+12:00", "31.08.2026 — 31.08.2026"),
    ("2026-08-31T12:00:00Z", "2026-09-01T12:00:00Z",
     "31.08.2026 12:00 — 01.09.2026 12:00 (не включая)"),
    ("2026-08-31T00:00:00Z", "2026-08-31T12:00:00Z",
     "31.08.2026 00:00 — 31.08.2026 12:00 (не включая)"),
    ("2026-08-31T12:00:00.123456Z", "2026-08-31T12:00:00.234567Z",
     "31.08.2026 12:00:00.123456 — 31.08.2026 12:00:00.234567 (не включая)"),
])
def test_display_half_open_calendar_dates_without_timezone_conversion(start, end, expected):
    from Application.statistics_presentation import interval_text
    assert interval_text(start, end) == expected


def test_partial_interval_with_different_offsets_keeps_both_offsets():
    from Application.statistics_presentation import interval_text
    value = interval_text("2026-08-31T12:00:00+03:00", "2026-08-31T12:00:00+00:00")
    assert value == "31.08.2026 12:00 +0300 — 31.08.2026 12:00 +0000 (не включая)"


def test_confirmed_status_names_are_exact_and_unknowns_are_not_invented():
    from Application.statistics_presentation import STATUS_LABELS, status_label
    from Application.statistics_purchase_policy import classify_status, StatusGroup
    assert status_label("N") == "Принят (N)"
    assert status_label("CP") == "Выполнен частично (CP)"
    assert status_label("F") == "Выполнен (F)"
    assert status_label("sold") == "Продан (sold)"
    assert status_label("no-product") == "Нет в наличии (crm) (no-product)"
    assert status_label("Mindbox-Tech-Cancellation") == "Техническая отмена (Mindbox-Tech-Cancellation)"
    assert status_label("unknown-future-status") == "unknown-future-status"
    assert status_label("f") == "f"
    assert all(classify_status(code) != StatusGroup.UNKNOWN for code in STATUS_LABELS)
    assert len({status_label(code) for code in STATUS_LABELS}) == len(STATUS_LABELS)


def test_source_quality_percentage_is_a_row_without_changing_counts():
    from Application.statistics_presentation import source_event_quality_display_rows
    result = snapshot(10000, 6000, 4000, 4518, 0)
    original = deepcopy(result)
    values = source_event_quality_display_rows(result)
    assert len(values) == 8
    assert values[6] == ("Доля просмотров / избранного без товара, %", "45,18 %")
    assert [value for _, value in values[:6]] == [10000, 6000, 4000, 10000, 5482, 4518]
    assert result == original


def test_both_currency_annual_tables_show_inclusive_calendar_years_without_mutation():
    from datetime import datetime, timezone
    from decimal import Decimal
    from tests.test_statistics_buyer_aggregates import aggregate, purchase, AT
    from Application.statistics_presentation import buyer_sections
    coverage = [(datetime(2022, 1, 1, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc))]
    saved = aggregate([
        purchase("buyer", "first", "2022-01-01T12:00:00Z"),
        purchase("buyer", "last", "2026-08-31T12:00:00Z",
                 (("F", 1, "20", "kanzlerKz"),)),
    ], coverage=coverage).result(AT.date())
    original = deepcopy(saved)
    result = {"buyer_statistics": saved}
    detailed = buyer_sections(result)
    compact = buyer_sections(result, compact=True)
    assert compact == detailed[:-1]
    assert detailed[-1][0] == "Качество и ограничения покупательской аналитики"
    annual = [(title, tables[-1]) for title, _, tables, _ in compact if title.startswith("Денежные показатели подтверждённых покупок")]
    assert len(annual) == 2
    for title, (headers, rows) in annual:
        assert headers[0] == "Год"
        years = {row[0]: row for row in rows}
        assert years["2022"][1] == "Полный год: 01.01.2022 — 31.12.2022"
        assert years["2026"][1] == "Неполный год: 01.01.2026 — 31.08.2026"
        currency = "RUB" if title.endswith("RUB") else "KZT"
        for year in saved["policies"]["STRICT_COMPLETED"]["annual"]:
            assert years[year["year"]][2] == year["money"][currency]["known_orders"]
            assert years[year["year"]][4] == Decimal(year["money"][currency]["amount"])
    assert saved == original


def test_warning_protocol_keys_remain_original_while_labels_are_russian():
    from Application.statistics_presentation import WARNING_LABELS
    keys = (
        "CustomerMerges не покрывает весь период: canonical identity может быть неполной.",
        "Есть конфликтующие снимки заказов: item interactions рассчитаны по первому снимку.",
        "Часть Actions не содержит customer ID; число клиентов Actions учитывает только известные ID.",
        "Canonical Customers отсутствует: число профилей неизвестно.",
    )
    assert tuple(WARNING_LABELS) == keys
    assert WARNING_LABELS[keys[2]] == ("Часть событий не содержит идентификатора клиента; "
                                     "число клиентов учитывает только известные идентификаторы.")
    assert WARNING_LABELS[keys[3]] == "Данные клиентов отсутствуют: число профилей неизвестно."


def test_buyer_navigation_groups_are_stable_and_keep_public_export_order():
    from tests.test_statistics_buyer_aggregates import aggregate, purchase, AT
    from Application.statistics_presentation import (
        buyer_sections, grouped_buyer_sections, CUSTOMER_GROUP_ORDER)
    saved = {"buyer_statistics": aggregate([purchase()]).result(AT.date())}
    before = deepcopy(saved)
    assert CUSTOMER_GROUP_ORDER == ("overview", "purchases", "money", "rfm", "cohorts")
    compact = grouped_buyer_sections(saved, compact=True)
    assert [group for group, _ in compact] == [
        "purchases", "purchases", "money", "money", "overview", "overview", "overview",
        "money", "money", "money", "money", "money", "purchases", *["purchases"]*4]
    for is_compact in (False, True):
        tagged = grouped_buyer_sections(saved, compact=is_compact)
        assert [section for _, section in tagged] == buyer_sections(saved, compact=is_compact)
        assert all(group in CUSTOMER_GROUP_ORDER for group, _ in tagged)
    assert grouped_buyer_sections({}, compact=True)[0][0] == "purchases"
    assert saved == before
