from dataclasses import replace
from datetime import datetime, timezone
from threading import Thread

import pytest

from Application.mindbox.identity import CustomerIdResolver, CustomerIdentityError
from Application.mindbox.records import CustomerMergeRecord


def merge(source, target):
    return CustomerMergeRecord("synthetic-merge", datetime(2026, 1, 1, tzinfo=timezone.utc), target, (source,))


def test_simple_merge_unknown_and_missing_target_profile():
    resolver = CustomerIdResolver([merge("100", "200")])
    assert resolver.resolve("100") == "200"
    assert resolver.resolve("200") == "200"
    assert resolver.resolve("unknown") == "unknown"
    assert resolver.alias_count == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_transitive_merges_and_compression(reverse):
    records = [merge("100", "200"), merge("200", "300")]
    resolver = CustomerIdResolver(reversed(records) if reverse else records)
    assert [resolver.resolve(value) for value in ("100", "200", "300")] == ["300"] * 3
    assert resolver._canonical["100"] == "300"
    assert resolver.alias_count == 2
    size = len(resolver._canonical)
    for index in range(100):
        resolver.resolve(f"unknown-{index}")
    assert len(resolver._canonical) == size


def test_multiple_sources_and_duplicate_identical_merges():
    record = replace(merge("A", "C"), merged_customer_ids=("A", "B"))
    resolver = CustomerIdResolver([record, record])
    assert resolver.resolve("A") == resolver.resolve("B") == "C"
    assert resolver.alias_count == 2


@pytest.mark.parametrize("edges", [[("A", "A")], [("A", "B"), ("B", "A")],
    [("A", "B"), ("B", "C"), ("C", "B")]])
def test_cycle_is_rejected_without_ids(edges):
    with pytest.raises(CustomerIdentityError, match="Cycle"):
        CustomerIdResolver(merge(source, target) for source, target in edges)


@pytest.mark.parametrize("edges", [[("A", "B"), ("A", "C")],
    [("A", "B"), ("B", "C"), ("A", "C")]])
def test_direct_conflict_is_rejected_even_if_targets_converge(edges):
    with pytest.raises(CustomerIdentityError, match="Conflicting"):
        CustomerIdResolver(merge(source, target) for source, target in edges)


@pytest.mark.parametrize("record", [None, replace(merge("A", "B"), merged_customer_ids=()),
    replace(merge("A", "B"), merged_customer_ids="PRIVATE_ID"),
    merge("", "B"), merge("A", ""), merge(None, "B"), merge("A", 1)])
def test_malformed_merge(record):
    with pytest.raises(CustomerIdentityError):
        CustomerIdResolver([record])


def test_long_chain_is_iterative():
    resolver = CustomerIdResolver(merge(str(index), str(index + 1)) for index in range(3000))
    assert resolver.resolve("0") == "3000"
    assert resolver.alias_count == 3000


def test_worker_yields_without_changing_canonical_results(monkeypatch):
    from Application.mindbox import identity
    records = [merge(str(i), str(i+1)) for i in range(1025)]
    expected = CustomerIdResolver(records)
    pauses, results = [], []
    monkeypatch.setattr(identity, "sleep", lambda seconds: pauses.append(seconds), raising=False)
    def build():
        resolver = CustomerIdResolver(records)
        results.append((resolver._parents, resolver._canonical, resolver.alias_count, resolver.resolve("unknown")))
    thread = Thread(target=build)
    thread.start()
    thread.join(3)
    assert not thread.is_alive()
    assert results == [(expected._parents, expected._canonical, expected.alias_count, "unknown")]
    assert pauses and all(0 < seconds <= .001 for seconds in pauses)


def test_main_thread_never_sleeps_during_identity_build(monkeypatch):
    from Application.mindbox import identity
    monkeypatch.setattr(identity, "sleep", lambda *a: pytest.fail("No sleep on GUI/main thread"), raising=False)
    resolver = CustomerIdResolver(merge(str(i), str(i+1)) for i in range(1025))
    assert resolver.resolve("0") == "1025"


@pytest.mark.parametrize("tail", [[merge("0", "different")], [merge("1025", "0")]])
def test_worker_still_rejects_conflicts_and_cycles_after_yield(monkeypatch, tail):
    from Application.mindbox import identity
    pauses, errors = [], []
    records = [merge(str(i), str(i+1)) for i in range(1025)] + tail
    monkeypatch.setattr(identity, "sleep", lambda seconds: pauses.append(seconds), raising=False)
    def build():
        try:
            CustomerIdResolver(records)
        except CustomerIdentityError as exc:
            errors.append(type(exc))
    thread = Thread(target=build)
    thread.start()
    thread.join(3)
    assert not thread.is_alive()
    assert errors == [CustomerIdentityError] and pauses
