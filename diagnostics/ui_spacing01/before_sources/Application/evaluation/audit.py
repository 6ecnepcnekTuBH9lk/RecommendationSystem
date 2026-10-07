"""Read-only canonical temporal dataset audit; no model imports or training."""

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from Application.interactions import InteractionBuilder, InteractionBuildError, InteractionType, classify_action_system_name
from Application.mindbox.adapters import adapt_action, adapt_action_system_name, adapt_customer_merge, adapt_order
from Application.mindbox.canonical_storage import catalog, current_batch, effective_entries, storage_lock
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.order_dedup import OrderSnapshots
from Application.mindbox.raw_reader import iter_export, part_files
from Application.product_resolution import ProductResolver, load_catalog
from .temporal import TemporalConfig, build_temporal_protocol


class TemporalAuditError(ValueError):
    """Safe audit failure, without raw records or personal identifiers."""


def load_canonical_events(raw_root, catalog_path, progress=None):
    """Same source selection, action classification and order policy as production.

    Retain resolved domain events instead of reducing them to BPR weighted pairs.
    Diagnose malformed mapped actions/unresolved products; fail on conflicting orders.
    The existing storage lock only touches its coordination file, never source data.
    """
    root = Path(raw_root).resolve()
    events = []
    with storage_lock(root):
        metadata = catalog(root)
        batch = current_batch(root)
        source_files = [p for c in batch.components
                        for p in part_files(root / c.export.relative_directory, c.name)] + [Path(catalog_path)]
        source_state = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in source_files}
        merge_component = batch.components[0]
        merge_count = 0

        def merges():
            nonlocal merge_count
            for raw in iter_export("customer_merges", input_dir=root / merge_component.export.relative_directory):
                merge_count += 1
                yield adapt_customer_merge(raw)

        customers = CustomerIdResolver(merges())
        builder = InteractionBuilder(batch.selection.interaction_rules())
        products = ProductResolver(load_catalog(Path(catalog_path)))
        orders = OrderSnapshots()

        def resolved(interaction):
            value = products.resolve_interaction(interaction, strict=False)
            if value is not None:
                events.append(value)

        for name in ("actions", "orders"):
            raw_count = 0
            for component in batch.components:
                if component.name != name:
                    continue
                for raw in iter_export(name, input_dir=root / component.export.relative_directory):
                    raw_count += 1
                    if name == "actions":
                        system = adapt_action_system_name(raw)
                        if classify_action_system_name(system, builder.rules) is None:
                            builder.record_unmapped_action(system)
                        else:
                            action = adapt_action(raw, customers, product_namespaces=batch.selection.action_product_namespaces)
                            try:
                                interactions = builder.from_action(action)
                            except InteractionBuildError:
                                interactions = ()
                            for interaction in interactions:
                                resolved(interaction)
                    elif orders.accept(raw):
                        for line in adapt_order(raw, customers, product_namespaces=batch.selection.order_product_namespaces):
                            interaction = builder.from_order_line(line)
                            if interaction is not None:
                                resolved(interaction)
                    if progress and raw_count % 100000 == 0:
                        progress(f"{name}: {raw_count} raw records; {len(events)} resolved events")
            if progress:
                progress(f"{name}: completed {raw_count} raw records; {len(events)} resolved events")
        if (current_batch(root) != batch or any((p.stat().st_size, p.stat().st_mtime_ns) != state
                                               for p, state in source_state.items())):
            raise TemporalAuditError("Canonical input changed during audit")
        d, resolution = builder.diagnostics, products.diagnostics
        source = {
            "canonical_revision": metadata["revision"],
            "declared_start": batch.window.interaction_since.isoformat(),
            "declared_end": batch.window.interaction_until.isoformat(),
            "updated": batch.created_at_utc.isoformat(),
            "source_kinds": sorted({c.export.source_kind for c in batch.components if c.name != "customer_merges"}),
            "effective_intervals": {name: [[e["since"], e["until"]] for e in effective_entries(metadata, name)]
                                    for name in ("actions", "orders")},
            "selected_components": len(batch.components), "input_stable_during_scan": True,
            "selected_file_count": len(source_files), "selected_file_bytes": sum(size for size, _ in source_state.values()),
            "selection": asdict(batch.selection),
            "merge_start": merge_component.since.isoformat(), "merge_end": merge_component.until.isoformat(),
            "ingestion": {"customer_merges": merge_count, "actions_raw": d.actions_total,
                          "unmapped_actions": d.actions_unmapped, "malformed_mapped_actions": d.actions_malformed,
                          "order_lines": d.order_lines_total, "orders_raw": orders.raw,
                          "orders_unique": len(orders.fingerprints), "orders_duplicate_identical": orders.identical,
                          "orders_duplicate_conflicting": orders.conflicting,
                          "classified_interactions": d.total_interactions,
                          "resolved_interactions": resolution.total.resolved,
                          "unresolved_interactions": resolution.total.unresolved,
                          "unsupported_products": resolution.total.unsupported_namespace,
                          "resolution_by_type": {k.value: asdict(v) for k, v in resolution.by_type.items()}},
        }
    return events, source


def distribution(values):
    """Linear quantiles; nulls, never fabricated zeros, for an empty population."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {key: None for key in ("min", "p25", "median", "p75", "p90", "max", "mean")}
    quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 0.9, 1])
    return {**dict(zip(("min", "p25", "median", "p75", "p90", "max"), map(float, quantiles))),
            "mean": float(values.mean())}


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "percent": 100 * numerator / denominator if denominator else None}


def coverage_summary(events, source):
    days = Counter()
    types = Counter()
    weekly = defaultdict(lambda: {"events": 0, "users": set(), "items": set(), "types": Counter()})
    customers, items = set(), set()
    first = last = None
    by_type_dates = {}
    outside = Counter()
    start, end = map(datetime.fromisoformat, (source["declared_start"], source["declared_end"]))
    for event in events:
        record = event.interaction
        stamp, kind = record.event_datetime_utc, record.interaction_type.value
        first = stamp if first is None else min(first, stamp)
        last = stamp if last is None else max(last, stamp)
        days[stamp.date()] += 1
        types[kind] += 1
        old_first, old_last = by_type_dates.get(kind, (stamp, stamp))
        by_type_dates[kind] = (min(old_first, stamp), max(old_last, stamp))
        if not start <= stamp < end:
            outside[kind] += 1
        customers.add(record.customer_id)
        items.add(event.item_id)
        week = stamp.date() - timedelta(days=stamp.weekday())
        row = weekly[week]
        row["events"] += 1
        row["users"].add(record.customer_id)
        row["items"].add(event.item_id)
        row["types"][kind] += 1
    daily = [{"day": (start + timedelta(days=i)).date().isoformat(),
              "events": days[(start + timedelta(days=i)).date()]} for i in range((end - start).days)]
    return {"earliest": first.isoformat() if first else None, "latest": last.isoformat() if last else None,
            "resolved_events": len(events), "unique_customers": len(customers), "unique_items": len(items),
            "interaction_types": {kind.value: types[kind.value] for kind in InteractionType},
            "type_timestamp_ranges": {k: [a.isoformat(), b.isoformat()] for k, (a, b) in by_type_dates.items()},
            "outside_declared_range_by_type": dict(outside), "days": daily,
            "zero_event_days": [r["day"] for r in daily if r["events"] == 0],
            "weeks": [{"week_start": week.isoformat(), "events": row["events"],
                       "active_customers": len(row["users"]), "active_items": len(row["items"]),
                       "types": dict(row["types"])} for week, row in sorted(weekly.items())],
            "tail": daily[-14:], "day_event_distribution": distribution([r["events"] for r in daily])}


def summarize_snapshot(snapshot):
    counts = Counter(e.interaction.customer_id for e in snapshot.history)
    popularity = Counter(e.item_id for e in snapshot.history)
    items = set(snapshot.item_universe)
    violations = {
        "history_at_or_after_cutoff": sum(e.interaction.event_datetime_utc >= snapshot.cutoff for e in snapshot.history),
        "target_outside_window": sum(c.target_timestamp < snapshot.cutoff or (
            snapshot.future_end is not None and c.target_timestamp >= snapshot.future_end) for c in snapshot.cases),
        "target_seen": sum(c.target_item in c.seen_items for c in snapshot.cases),
        "target_not_warm": sum(c.target_item not in items for c in snapshot.cases),
        "history_threshold_violation": 0,
    }
    if any(violations.values()):
        raise TemporalAuditError("Temporal no-leakage invariant violated; audit blocked")
    diag = asdict(snapshot.diagnostics)
    futures, novel, cases = diag["future_users"], diag["eligible_warm_user_targets"], len(snapshot.cases)
    target_counts = Counter(c.target_item for c in snapshot.cases)
    delay = lambda selected: [(c.target_timestamp - snapshot.cutoff).total_seconds() / 86400 for c in selected]
    return {"cutoff": snapshot.cutoff.isoformat(), "window_end": snapshot.future_end.isoformat() if snapshot.future_end else None,
            "diagnostics": diag,
            "rates": {"cold_users_per_future_user": rate(diag["excluded_cold_users"], futures),
                      "sparse_users_per_future_user": rate(diag["excluded_sparse_users"], futures),
                      "no_novel_per_future_user": rate(diag["no_novel_target_users"], futures),
                      "no_novel_per_history_eligible_user": rate(diag["no_novel_target_users"], diag["users_passing_history_threshold"]),
                      "cold_items_per_eligible_novel_target": rate(diag["excluded_cold_item_targets"], novel),
                      "cold_item_cases_per_future_user": rate(diag["excluded_cold_item_targets"], futures),
                      "history_eligible_per_future_user": rate(diag["users_passing_history_threshold"], futures),
                      "final_cases_per_future_user": rate(cases, futures)},
            "target_type_rates": {kind.value: rate(diag[kind.value.lower() + "_targets"], cases) for kind in InteractionType},
            "target_delay_days": distribution(delay(snapshot.cases)),
            "target_delay_days_by_type": {kind.value: {"observations": diag[kind.value.lower() + "_targets"],
                "distribution": distribution(delay([c for c in snapshot.cases if c.interaction_type is kind]))} for kind in InteractionType},
            "benchmark_history_events": distribution([counts[c.customer_id] for c in snapshot.cases]),
            "benchmark_unique_history_items": distribution([len(c.seen_items) for c in snapshot.cases]),
            "candidate_count": distribution([len(items) - len(c.seen_items) for c in snapshot.cases]),
            "unique_target_items": len(target_counts),
            "target_concentration": {"top10": rate(sum(n for _, n in target_counts.most_common(10)), cases),
                                     "top50": rate(sum(n for _, n in target_counts.most_common(50)), cases)},
            "target_training_popularity_events": distribution([popularity[c.target_item] for c in snapshot.cases]),
            "invariants": violations}


def audit_scenarios(events, source, *, horizons=(14, 30, 60), thresholds=(5, 10, 20),
                    sensitivity_horizon=30, test_end=None, progress=None):
    for values in (horizons, thresholds, (sensitivity_horizon,)):
        if not values or any(type(n) is not int or n < 1 for n in values):
            raise TemporalAuditError("Horizons and history thresholds must be positive integers")
    coverage = coverage_summary(events, source)
    if not events:
        return {"schema_version": 1, "source": source, "coverage": coverage,
                "scenarios": [], "unavailable": [{"reason": "empty resolved dataset"}]}
    declared_start, declared_end = map(datetime.fromisoformat, (source["declared_start"], source["declared_end"]))
    # No source proof of complete last observed day: keep it in data, exclude from windows.
    last = datetime.fromisoformat(coverage["latest"])
    conservative_end = min(declared_end, last.replace(hour=0, minute=0, second=0, microsecond=0))
    end = conservative_end if test_end is None else test_end
    if (not isinstance(end, datetime) or end.tzinfo is None or end.utcoffset() is None):
        raise TemporalAuditError("Audit test end must be timezone-aware")
    end = end.astimezone(timezone.utc)
    if end > conservative_end:
        raise TemporalAuditError("Test end exceeds conservative observed coverage")
    scenarios, unavailable = [], []
    specifications = list(dict.fromkeys((h, 10) for h in horizons))
    specifications += [(sensitivity_horizon, threshold) for threshold in thresholds
                       if (sensitivity_horizon, threshold) not in specifications]
    for horizon, threshold in specifications:
        val_start, test_start = end - timedelta(days=2 * horizon), end - timedelta(days=horizon)
        if val_start <= declared_start or val_start <= datetime.fromisoformat(coverage["earliest"]):
            unavailable.append({"horizon_days": horizon, "min_history_events": threshold, "reason": "insufficient history/coverage"})
            continue
        if progress:
            progress(f"temporal scenario: {horizon} days, min_history_events={threshold}")
        config = TemporalConfig(val_start, test_start, end, threshold)
        protocol = build_temporal_protocol(events, config)
        validation, test = summarize_snapshot(protocol.validation), summarize_snapshot(protocol.test)
        for snapshot, summary in ((protocol.validation, validation), (protocol.test, test)):
            history_counts = Counter(e.interaction.customer_id for e in snapshot.history)
            if any(history_counts[c.customer_id] < threshold for c in snapshot.cases):
                raise TemporalAuditError("History threshold invariant violated; audit blocked")
        val_users, test_users = ({c.customer_id for c in s.cases} for s in (protocol.validation, protocol.test))
        scenarios.append({"horizon_days": horizon, "min_history_events": threshold,
                          "validation": validation, "test": test,
                          "user_overlap": {"validation_only": len(val_users - test_users),
                                           "test_only": len(test_users - val_users), "both": len(val_users & test_users)}})
        del protocol
    return {"schema_version": 1, "source": source, "coverage": coverage,
            "test_end": end.isoformat(), "tail_policy": "exclude last observed calendar day; completeness not independently proven",
            "scenarios": scenarios, "unavailable": unavailable,
            "method": {"quantiles": "numpy linear quantiles", "delay_unit": "days",
                       "popularity": "interaction occurrences in corresponding training history only",
                       "no_novel_scope": "future users passing history threshold; exclusive of cold/sparse users",
                       "model_training": False, "protocol_semantics_changed": False}}


def markdown_summary(report):
    c = report["coverage"]
    lines = ["# TRAIN-03A — canonical temporal dataset audit", "",
             f"Resolved events: {c['resolved_events']}; customers: {c['unique_customers']}; items: {c['unique_items']}.",
             f"Observed UTC range: {c['earliest']} — {c['latest']}.",
             f"Declared canonical coverage: {report['source']['declared_start']} — {report['source']['declared_end']} (exclusive end).",
             f"Interaction types: {c['interaction_types']}.",
             f"Resolved events outside declared range: {c['outside_declared_range_by_type']} (retained, never silently deleted).",
             f"Tail policy: {report.get('tail_policy', 'no available scenarios')}.", "",
             "Cold-user / sparse / no-novel rates use future users as denominator.",
             "Cold-item rate uses history-eligible first-novel targets; PURCHASE rate uses final cases.", "",
             "| Horizon | Min history | Validation window | Test window | Val future / cases | Test future / cases | Cold users V/T % | Cold items V/T % | No novel V/T % | PURCHASE V/T % | Median delay V/T days |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    def percent(summary, key):
        p = summary["rates"][key]["percent"]
        return "NA" if p is None else f"{p:.2f}"
    def number(value):
        return "NA" if value is None else f"{value:.2f}"
    for scenario in report["scenarios"]:
        v, t = scenario["validation"], scenario["test"]
        vd, td = v["diagnostics"], t["diagnostics"]
        pair = lambda key: percent(v, key) + "/" + percent(t, key)
        lines.append(f"| {scenario['horizon_days']} | {scenario['min_history_events']} | {v['cutoff'][:10]} → {v['window_end'][:10]} | "
                     f"{t['cutoff'][:10]} → {t['window_end'][:10]} | {vd['future_users']} / {vd['evaluation_targets']} | "
                     f"{td['future_users']} / {td['evaluation_targets']} | {pair('cold_users_per_future_user')} | "
                     f"{pair('cold_items_per_eligible_novel_target')} | {pair('no_novel_per_future_user')} | "
                     f"{number(v['target_type_rates']['PURCHASE']['percent'])}/{number(t['target_type_rates']['PURCHASE']['percent'])} | "
                     f"{number(v['target_delay_days']['median'])}/{number(t['target_delay_days']['median'])} |")
    for scenario in report["scenarios"]:
        lines += ["", f"## {scenario['horizon_days']} days / min history {scenario['min_history_events']}",
                  f"User overlap: {scenario['user_overlap']}."]
        for name in ("validation", "test"):
            s = scenario[name]
            lines += ["", f"### {name}", f"Cutoff: {s['cutoff']}; window end: {s['window_end']}.", "",
                      "| Diagnostic | Count |", "|---|---|"]
            lines += [f"| {key} | {value} |" for key, value in s["diagnostics"].items()
                      if key not in ("history_event_count_distribution", "mean_target_delay_seconds")]
            lines += ["", "| Rate | Numerator | Denominator | % |", "|---|---|---|---|"]
            lines += [f"| {key} | {row['numerator']} | {row['denominator']} | {number(row['percent'])} |"
                      for key, row in s["rates"].items()]
            lines += ["", "| Target type | Cases | % of final cases |", "|---|---|---|"]
            lines += [f"| {key} | {row['numerator']} | {number(row['percent'])} |"
                      for key, row in s["target_type_rates"].items()]
            lines += ["", "| Distribution | Min | P25 | Median | P75 | P90 | Max | Mean |",
                      "|---|---|---|---|---|---|---|---|"]
            for key in ("target_delay_days", "benchmark_history_events", "benchmark_unique_history_items",
                        "candidate_count", "target_training_popularity_events"):
                lines.append(f"| {key} | " + " | ".join(number(s[key][q]) for q in
                             ("min", "p25", "median", "p75", "p90", "max", "mean")) + " |")
            lines += ["", "| Target type delay (days) | N | P25 | Median | P75 | P90 |", "|---|---|---|---|---|---|"]
            for key, row in s["target_delay_days_by_type"].items():
                lines.append(f"| {key} | {row['observations']} | " + " | ".join(number(row["distribution"][q])
                             for q in ("p25", "median", "p75", "p90")) + " |")
            lines += ["", f"Unique target items: {s['unique_target_items']}."]
            for key, row in s["target_concentration"].items():
                lines.append(f"{key} concentration: {row['numerator']} / {row['denominator']} = {number(row['percent'])}%.")
            lines.append(f"No-leakage invariant violations: {s['invariants']}.")
    lines += ["", f"Unavailable scenarios: {report['unavailable']}.",
              f"Zero-event days: {c['zero_event_days']}.", "", "## Ingestion diagnostics", "",
              "| Diagnostic | Count |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in report["source"].get("ingestion", {}).items()
              if key != "resolution_by_type"]
    lines += ["", "## Last 14 declared days", "",
              "| UTC day | Resolved events |", "|---|---|"]
    lines += [f"| {row['day']} | {row['events']} |" for row in c["tail"]]
    lines += ["", "## Weekly density", "", "| Week start | Events | Active customers | Active items | VIEW / FAVORITE / PURCHASE |",
              "|---|---|---|---|---|"]
    lines += [f"| {r['week_start']} | {r['events']} | {r['active_customers']} | {r['active_items']} | "
              + " / ".join(str(r["types"].get(k.value, 0)) for k in InteractionType) + " |" for r in c["weeks"]]
    lines += ["", "Source coverage is declarative for MANUAL imports; no independent completeness proof.",
              "No model training, publication, item features or model-quality-based cutoff selection.",
              "Current mutable catalog/identity metadata do not establish historical as-of provenance.",
              "Full daily density and exact aggregates are retained in report.json."]
    return "\n".join(lines) + "\n"
