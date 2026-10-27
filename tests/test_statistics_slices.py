"""Small canonical fixtures only: exact DTO round trips and fail-closed provenance."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

import pytest

from Application import statistics_slices as cache
from Application.analysis_filter import AnalysisFilter
from Application.statistics_cache import save_result
from tests.test_dataset_statistics import (
    dataset as dataset_fixture,
    action,
    order,
    save,
)
from tests.test_analysis_filter import reference, sources


@pytest.fixture
def dataset(tmp_path):
    return dataset_fixture.__wrapped__(tmp_path)


def run(dataset, **kwargs):
    root, reference, _ = dataset
    return cache.calculate_cached_statistics(
        raw_root=root, catalog_path=reference, cache_dir=root / "slices", **kwargs
    )


def canonical(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize(
    "selection",
    [
        AnalysisFilter(),
        AnalysisFilter("2026-01-01", "2026-01-01"),
        AnalysisFilter("2026-01-01", "2026-03-31"),
        AnalysisFilter("2026-01-01", "2026-12-31"),
        AnalysisFilter(nomenclature_types=("Рубашки",)),
        AnalysisFilter(nomenclature_types=("Брюки", "Рубашки")),
        AnalysisFilter(collections=("Весна-лето",)),
        AnalysisFilter("2026-01-01", "2026-01-31", ("Рубашки",)),
        AnalysisFilter("2026-01-01", "2026-02-28", collections=("Осень-зима",)),
        AnalysisFilter(collections=("不存在",)),
        AnalysisFilter("2025-01-01", "2025-01-01"),
    ],
)
def test_all_fields_exact_with_duplicates_conflicts_mixed_lines_months_and_repeat_buyers(
    dataset, selection, monkeypatch
):
    reference(dataset)
    view = cache.statistics.MindboxSelectionConfig().view_action_system_names[0]
    actions = [
        action(view, ["000001-size", "999999-size"]),
        action(view, ["000002-size"], stamp="2026-03-01T01:00:00Z"),
        action(view),
        action("UnknownWithoutCustomer", customer=None),
    ]
    # Missing technical-action ID must be represented by absence, never a null adapter identity.
    actions[-1].pop("customer")
    a = order(["CP", "CD"])
    a["lines"][1]["product"]["ids"]["offline1C"] = "000002-size"
    conflict = json.loads(json.dumps(a))
    conflict["lines"][0]["status"]["ids"]["externalId"] = "cancelled"
    later = order(["F", "Return"], "later")
    later["firstAction"]["dateTimeUtc"] = "2026-03-01T12:00:00Z"
    sources(dataset, actions, [a, a, conflict, later])
    root, path, _ = dataset
    expected = asdict(
        cache.statistics.calculate_statistics(
            raw_root=root, catalog_path=path, analysis_filter=selection
        )
    )
    first = run(dataset, analysis_filter=selection)
    expected.pop("calculated_at")
    actual = dict(first)
    actual.pop("calculated_at")
    assert canonical(actual) == canonical(
        expected
    )  # All fields, exact counts/Decimal strings/floats; no float reorder.
    monkeypatch.setattr(
        cache.statistics,
        "calculate_statistics",
        lambda **kw: pytest.fail("Cache hit must not rescan"),
    )
    second = run(dataset, analysis_filter=selection)
    assert second == canonical(first)  # Including original calculated_at.


def test_two_filters_normalized_and_survive_new_process_state(dataset, monkeypatch):
    reference(dataset)
    a = run(
        dataset,
        analysis_filter=AnalysisFilter(
            nomenclature_types=("Рубашки", "Брюки", "Рубашки")
        ),
    )
    b = run(dataset, analysis_filter=AnalysisFilter(collections=("Осень-зима",)))
    monkeypatch.setattr(
        cache.statistics,
        "calculate_statistics",
        lambda **kw: pytest.fail("No recomputation"),
    )
    assert run(
        dataset,
        analysis_filter=AnalysisFilter(nomenclature_types=(" Брюки ", "Рубашки")),
    ) == canonical(a)
    assert run(
        dataset, analysis_filter=AnalysisFilter(collections=("Осень-зима",))
    ) == canonical(b)
    assert len(list((dataset[0] / "slices").glob("*.json"))) == 2


@pytest.mark.parametrize(
    "changed",
    [
        "revision",
        "merges",
        "actions",
        "orders",
        "customers",
        "catalog",
        "categories",
        "selection",
        "algorithm",
        "age_date",
    ],
)
def test_provenance_invalidates_every_dependency(dataset, monkeypatch, changed):
    root, path, data = dataset
    before, _ = cache.capture_sources(root, path, AnalysisFilter())
    result = run(dataset)
    if changed == "revision":
        data["revision"] = "next"
        save(root, data)
    elif changed in ("merges", "actions", "orders"):
        name = "customer_merges" if changed == "merges" else changed
        entry = (
            data[name] if name == "customer_merges" else next(iter(data[name].values()))
        )
        file = root / entry["directory"] / f"{name}_part_001.json"
        file.write_bytes(file.read_bytes() + b" ")
    elif changed == "customers":
        db = sqlite3.connect(root / "canonical/customers.sqlite")
        try:
            db.execute(
                "UPDATE profiles SET raw=replace(raw,'new','NEW') WHERE id='new'"
            )
            db.commit()
        finally:
            db.close()
    elif changed in ("catalog", "categories"):
        file = path if changed == "catalog" else path.with_name("site_categories.csv")
        old = file.stat()
        file.write_bytes(
            file.read_bytes().replace("Рубашка".encode(), "Сорочка".encode())
            if changed == "catalog"
            else b"\xef\xbb\xbf"
            + "КодКатегории|НазваниеКатегории\n001|Товар\n".encode()
        )
        os.utime(
            file, ns=(old.st_atime_ns, old.st_mtime_ns)
        )  # Content SHA, not mtime alone.
    elif changed == "selection":
        data["selection"]["purchase_line_statuses"] = ["F"]
        save(root, data)
    elif changed == "algorithm":
        monkeypatch.setattr(cache, "_algorithm_version", lambda: "other-code")
    else:

        class Tomorrow(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime(2099, 1, 1, tzinfo=timezone.utc)

        monkeypatch.setattr(cache, "datetime", Tomorrow)
    after, _ = cache.capture_sources(root, path, AnalysisFilter())
    assert before != after
    assert cache.load_slice(root / "slices", after) is None
    assert cache.load_slice(root / "slices", before) == canonical(result)


def test_missing_customers_cached_but_missing_merges_and_incomplete_sources_never_reuse_old(
    dataset,
):
    root, path, data = dataset
    run(dataset)
    (root / "canonical/customers.sqlite").unlink()
    assert run(dataset)["customers"] is None
    data["customer_merges"] = None
    save(root, data)
    with pytest.raises(ValueError, match="CustomerMerges"):
        run(dataset)
    data["actions"].clear()
    data["orders"].clear()
    save(root, data)
    assert run(dataset)["source_actions"] == 0


@pytest.mark.parametrize("damage", ["json", "checksum", "result", "filter"])
def test_corruption_is_miss_and_last_success_preserved(dataset, damage):
    result = run(dataset)
    root, reference, _ = dataset
    last = root / "last-success.json"
    save_result(last, result)
    saved = last.read_bytes()
    provenance, _ = cache.capture_sources(root, reference, AnalysisFilter())
    path = root / "slices" / (cache._digest(provenance) + ".json")
    payload = json.loads(path.read_text())
    if damage == "json":
        path.write_text("{")
    else:
        if damage == "checksum":
            payload["result_sha256"] = "0" * 64
        elif damage == "result":
            payload["result"]["orders"] = -1
            payload["result_sha256"] = cache._digest(payload["result"])
        else:
            payload["result"]["analysis_filter"]["collections"] = ["different"]
            payload["result_sha256"] = cache._digest(payload["result"])
        path.write_text(json.dumps(payload))
    assert cache.load_slice(root / "slices", provenance) is None
    assert last.read_bytes() == saved


def test_cancel_and_changed_sources_never_publish_new_cache_or_replace_last(
    dataset, monkeypatch
):
    root, _, data = dataset
    result = run(dataset)
    last = root / "last.json"
    save_result(last, result)
    previous = last.read_bytes()
    with pytest.raises(InterruptedError):
        run(dataset, cancelled=lambda: True)
    assert last.read_bytes() == previous
    before = set((root / "slices").glob("*.json"))
    original = cache.statistics.calculate_statistics

    def changed(**kw):
        value = original(**kw)
        data["revision"] = "changed-during-calculation"
        save(root, data)
        return value

    monkeypatch.setattr(cache.statistics, "calculate_statistics", changed)
    run(dataset, analysis_filter=AnalysisFilter(collections=("new",)))
    assert (
        set((root / "slices").glob("*.json")) == before
        and last.read_bytes() == previous
    )


def test_cache_write_error_does_not_change_calculated_result_or_last_success(
    dataset, monkeypatch
):
    monkeypatch.setattr(
        cache,
        "save_slice",
        lambda *a: (_ for _ in ()).throw(OSError("synthetic disk error")),
    )
    assert run(dataset)["source_actions"] == 7


def test_live_customers_sidecar_blocks_attestation(dataset):
    root, path, _ = dataset
    (root / "canonical/customers.sqlite-wal").write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="transaction"):
        cache.capture_sources(root, path, AnalysisFilter())


def test_decimal_context_and_reference_contents_participate_in_version(dataset):
    from decimal import localcontext

    root, path, _ = dataset
    first, _ = cache.capture_sources(root, path, AnalysisFilter())
    with localcontext() as context:
        context.prec = 40
        second, _ = cache.capture_sources(root, path, AnalysisFilter())
    assert first != second


def test_decimal_sticky_flags_do_not_prevent_publication_or_next_hit(
    dataset, monkeypatch
):
    from decimal import Inexact, localcontext

    orders = [order(["CP"], f"order-{n}") for n in range(3)]
    for record, price in zip(orders, (1, 1, 2)):
        record["lines"][0]["priceOfLine"] = price
    sources(dataset, [], orders)  # Mean order amount = 4/3 legitimately sets Inexact.

    original = cache.statistics.calculate_statistics
    calls = []

    def calculate(**kwargs):
        calls.append(True)
        return original(**kwargs)

    monkeypatch.setattr(cache.statistics, "calculate_statistics", calculate)
    with localcontext() as context:
        context.clear_flags()
        first = run(dataset)
        assert context.flags[Inexact]
        assert len(list((dataset[0] / "slices").glob("*.json"))) == 1
        context.clear_flags()  # A fresh CLI process has no sticky flags.
        assert run(dataset) == canonical(first)
    assert len(calls) == 1


def test_bounded_cache_does_not_remove_last_success(dataset, monkeypatch):
    monkeypatch.setattr(cache, "MAX_ENTRIES", 2)
    root, _, _ = dataset
    saved = run(dataset)
    last = root / "slices" / "dataset_statistics.json"
    save_result(last, saved)
    previous = last.read_bytes()
    run(dataset, analysis_filter=AnalysisFilter(collections=("a",)))
    run(dataset, analysis_filter=AnalysisFilter(collections=("b",)))
    assert (
        len(list((root / "slices").glob("*.json"))) == 3
        and last.read_bytes() == previous
    )


def test_restart_cli_reads_persistent_slice_without_recalculation(dataset):
    import subprocess
    import sys

    root, catalog, _ = dataset
    script = Path(__file__).resolve().parents[1] / "scripts/dataset_statistics.py"
    command = [
        sys.executable,
        "-B",
        str(script),
        "--raw-root",
        str(root),
        "--catalog",
        str(catalog),
        "--slice-cache",
        str(root / "slices"),
    ]
    messages = []
    for _ in range(2):
        process = subprocess.run(
            command, capture_output=True, encoding="utf-8", check=True
        )
        messages.append([json.loads(line) for line in process.stdout.splitlines()])
    assert messages[0][-1]["value"] == messages[1][-1]["value"]
    assert any(
        row["event"] == "progress" and row["value"].startswith("Кеш:")
        for row in messages[1]
    )
