"""GUI-only dates and metadata. Readiness imports no ML/runtime preparation.

The 2025 BPR-EXP-01 configuration remains owned by bpr_weights.py. Missing
benchmark metadata means a legacy GUI record, never the current protocol.
"""

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class GuiBenchmark:
    id: str
    history_start: datetime
    validation_start: datetime
    test_start: datetime
    test_end: datetime
    min_history_events: int


GUI_BENCHMARK = GuiBenchmark(
    "mindbox-validation-2026-07",
    datetime(2022, 1, 1, tzinfo=timezone.utc), datetime(2026, 7, 1, tzinfo=timezone.utc),
    datetime(2026, 8, 1, tzinfo=timezone.utc), datetime(2026, 9, 1, tzinfo=timezone.utc), 10,
)
LEGACY_BENCHMARK = {"id": "mindbox-validation-2025-11", "history_end": "2025-11-01T00:00:00Z",
                    "validation_end": "2025-12-01T00:00:00Z", "min_history_events": 10}


def gui_benchmark_config():
    # Only the worker needs TemporalConfig; keep the GUI metadata path lightweight.
    from Application.evaluation.temporal import TemporalConfig
    return TemporalConfig(GUI_BENCHMARK.validation_start, GUI_BENCHMARK.test_start,
                          GUI_BENCHMARK.test_end, GUI_BENCHMARK.min_history_events)


def gui_benchmark_metadata():
    return {"id": GUI_BENCHMARK.id, "history_start": GUI_BENCHMARK.history_start.isoformat(),
            "history_end": GUI_BENCHMARK.validation_start.isoformat(),
            "validation_end": GUI_BENCHMARK.test_start.isoformat(),
            "test_end": GUI_BENCHMARK.test_end.isoformat(), "min_history_events": GUI_BENCHMARK.min_history_events}


def _utc(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Invalid benchmark UTC boundary")
    return value.astimezone(timezone.utc)


def require_gui_coverage(since, until):
    """Check the actual selected continuous batch, not disconnected effective entries."""
    if _utc(since) > GUI_BENCHMARK.history_start or _utc(until) < GUI_BENCHMARK.test_end:
        raise ValueError("GUI benchmark requires continuous coverage from January 2022 through August 2026")


def clean_benchmark_metadata(value):
    """Preserve stored legacy fields; never relabel a legacy run as the current benchmark."""
    if value is None:
        return dict(LEGACY_BENCHMARK)
    if not isinstance(value, dict):
        raise ValueError("Invalid benchmark metadata")
    expected = gui_benchmark_metadata() if value.get("id") == GUI_BENCHMARK.id else LEGACY_BENCHMARK
    if value.get("id") != expected["id"]:
        raise ValueError("Unknown benchmark identity")
    result = {"id": value["id"]}
    for name in ("history_start", "history_end", "validation_end", "test_end"):
        if name not in value:
            if name in expected:
                raise ValueError("Incomplete benchmark boundaries")
            continue
        parsed = _utc(value[name])
        if name in expected and parsed != _utc(expected[name]):
            raise ValueError("Benchmark boundary differs from its identity")
        result[name] = value[name]  # Retain original timestamp spelling in old records.
    minimum = value.get("min_history_events")
    if type(minimum) is not int or minimum != expected["min_history_events"]:
        raise ValueError("Invalid benchmark history threshold")
    result["min_history_events"] = minimum
    if "history_start" in result and _utc(result["history_start"]) >= _utc(result["history_end"]):
        raise ValueError("Invalid benchmark history interval")
    if "test_end" in result and _utc(result["test_end"]) <= _utc(result["validation_end"]):
        raise ValueError("Invalid benchmark test interval")
    return result
