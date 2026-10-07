from datetime import datetime, timedelta, timezone

import pytest

from Application.evaluation.temporal import TemporalConfig
from Application.interactions import InteractionRecord, InteractionSource, InteractionType
from Application.mindbox.records import ProductKey
from Application.product_resolution import ResolvedInteraction


@pytest.fixture
def event():
    def make(user, item, hour, kind=InteractionType.VIEW, quantity=None):
        stamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hour)
        return ResolvedInteraction(InteractionRecord(
            source_customer_id=user, customer_id=user, product=ProductKey("offline1C", item),
            interaction_type=kind, event_datetime_utc=stamp,
            source=InteractionSource.ORDER if kind is InteractionType.PURCHASE else InteractionSource.ACTION,
            source_event_id="synthetic", quantity=quantity,
        ), item)
    return make


@pytest.fixture
def config(event):
    return TemporalConfig(event("u", "a", 10).interaction.event_datetime_utc,
                          event("u", "a", 20).interaction.event_datetime_utc,
                          event("u", "a", 30).interaction.event_datetime_utc, min_history_events=2)


@pytest.fixture
def protocol(event, config):
    from Application.evaluation.temporal import build_temporal_protocol
    return build_temporal_protocol([
        event("u", "A", 1), event("u", "B", 2),
        event("other", "C", 1), event("other", "D", 2), event("other", "E", 3),
        event("u", "C", 12), event("u", "D", 22),
    ], config)
