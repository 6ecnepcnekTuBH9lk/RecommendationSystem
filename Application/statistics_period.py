"""Shared coverage for statistics eligibility and its displayed period."""

from datetime import datetime, timezone


def shared_intervals(actions, orders):
    def parsed(intervals):
        result = []
        for start, end in intervals:
            pair = tuple(datetime.fromisoformat(value.replace("Z", "+00:00")) for value in (start, end))
            if any(value.tzinfo is None for value in pair) or pair[0] >= pair[1]:
                raise ValueError("Некорректный период статистики.")
            result.append(tuple(value.astimezone(timezone.utc) for value in pair))
        return result

    actions, orders = parsed(actions), parsed(orders)
    # Preserve the existing single-source fallback used by the status message.
    intervals = ([(max(a, c), min(b, d)) for a, b in actions for c, d in orders if max(a, c) < min(b, d)]
                 if actions and orders else actions or orders)
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)
