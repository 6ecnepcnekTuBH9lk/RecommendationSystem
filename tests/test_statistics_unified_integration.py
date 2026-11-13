"""Synthetic single-source report, version dispatch and protected cache tests."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
import json

import pytest

from Application import dataset_statistics as legacy
from Application import statistics_cache as legacy_cache
from Application import statistics_unified_cache as cache
from Application import statistics_unified_dto as dto
from Application import statistics_unified_index as index
from Application import statistics_unified_service as service
from Application.mindbox.canonical_customers import encode
from Application.mindbox.selection import DEFAULT_SELECTION
from Application.statistics_storage_safety import StatisticsIndexError, path_identity
from tests import test_statistics_unified_cohorts as cohort_tests
from tests import test_statistics_unified_rfm as rfm_tests
from tests.test_dataset_statistics import action

_AMOUNT = Decimal("1000.00000000000000000000000001")
_FIRST = "2025-10-02T10:00:00Z"
_SECOND = "2025-11-01T10:00:00Z"


@pytest.fixture
def source(tmp_path, monkeypatch):
    result = cohort_tests.source.__wrapped__(tmp_path, monkeypatch)
    root = result[0]
    folder = root / "cache"
    folder.mkdir(mode=0o700)
    for module in (cache, service):
        monkeypatch.setattr(module, "validate_storage_root", lambda path, **kwargs: path_identity(path))
    return result


def _options(source, **overrides):
    values = rfm_tests._options(source)
    del values["customer_key"]
    return values | overrides


def _report(source, **kwargs):
    return service.calculate_unified_statistics(**_options(source, **kwargs))


def _write(source):
    order = cohort_tests._order
    first = order("purchase", _FIRST, amount=_AMOUNT)
    cp_line = deepcopy(order("cp", _FIRST, "CP", amount=500)["lines"][0])
    cp_line.update(id="cp-line", number=2)
    first["lines"].append(cp_line)
    second = order("repeat", _SECOND, product="000002", amount=35)
    returned = deepcopy(second)
    returned["lines"][0]["status"]["ids"]["externalId"] = "Return"
    kzt = order("sold", _SECOND, "sold", owner="other", amount=0, namespace="kanzlerKz")
    invalid = order("invalid", _SECOND, owner="other", amount="invalid")
    unknown = order("unknown", _SECOND, "custom-new-status", owner="inactive")
    cancelled = order("cancelled", _SECOND, "CD", owner="inactive")
    returned_only = order("return-only", _SECOND, "Return", owner="inactive")
    delivering = order("delivering", _SECOND, "delivering", owner="inactive")
    rows = [first, first, second, returned, kzt, invalid, unknown, cancelled, returned_only, delivering]
    rfm_tests._write_orders(source, rows)
    root, _, data, _, _ = source
    directory = root / data["manual_interactions"]["actions"]["directory"]
    events = [action(DEFAULT_SELECTION.view_action_system_names[0], ["000001-a"], "new", stamp=_FIRST),
              action(DEFAULT_SELECTION.favorite_action_system_names[0], ["000001-a"], "old", stamp=_FIRST)]
    (directory / "actions_part_001.json").write_text(encode({"customerActions": events}), encoding="utf-8")
    return rows


def _business(result):
    value = deepcopy(result.envelope()["result"])
    del value["source"]["reader"], value["source"]["projection_schema"]
    del value["metadata"]["detail"]["available"], value["metadata"]["detail"]["context_ids"]
    for field in ("source_binding", "fingerprint", "content_digest"):
        del value["metadata"][field]
    return value


def test_full_reference_report_with_independent_expected_values(source, monkeypatch):
    _write(source)
    calls = []
    original = service._orders

    def counted(*args, **kwargs):
        for entry in original(*args, **kwargs):
            calls.append(entry.whole.order_sort_rank)
            yield entry

    monkeypatch.setattr(service, "_orders", counted)
    result = _report(source)
    assert len(calls) == len(set(calls)) == 8
    assert result.envelope()["schema_version"] == 11
    assert result.policy["policy_id"] == "UNIFIED_PURCHASE_V1"
    assert result.policy["time_basis_id"] == "ORDER_CREATED_PURCHASE_PROXY_V1"
    assert result.source["reader"] == "REFERENCE" and result.source["projection_schema"] is None
    assert set(result.contexts) == set(result.coverage) == set(dto.CONTEXTS)
    assert result.orders["purchase_orders"]["known"] == 4
    assert result.orders["purchase_positions"]["known"] == 5
    assert result.orders["unique_orders"]["known"] == 8
    assert result.orders["cancellations"]["orders"]["known"] == 1
    assert result.orders["return_states"]["orders"]["known"] == 1
    assert result.orders["return_states"]["returned_units"]["value"] is None
    assert result.actions["data"]["VIEW_FAVORITE_PURCHASE"]["value"] == 1
    rub = result.money["currencies"]["RUB"]["amounts"]
    assert Decimal(rub["recognized_purchase"]["known"]) == Decimal("1035.00000000000000000000000001")
    assert rub["recognized_purchase"]["value"] is None
    assert {"CP_ALLOCATION_UNKNOWN", "PRICE_INVALID", "STATUS_CONFLICT"} <= set(rub["recognized_purchase"]["reasons"])
    assert rub["confirmed_refund"]["reasons"] == ["LEDGER_UNAVAILABLE"]
    assert rub["confirmed_refund"]["value"] is rub["net_after_refunds"]["value"] is None
    assert result.money["currencies"]["KZT"]["amounts"]["recognized_purchase"]["value"] == "0"
    statuses = result.status_catalog["rows"]
    assert len([row for row in statuses if row["bucket"] == "CATALOG"]) == 43
    assert any(row["external_id"] == "custom-new-status" and row["bucket"] == "UNKNOWN_CODE" for row in statuses)
    assert result.diagnostics["counters"]["exact_duplicates"] == 1
    assert result.diagnostics["counters"]["snapshot_conflicts"] == 1
    last = result.rfm["windows"]["LAST_12M"]["currencies"]["RUB"]
    assert dict((row[0], row[1]) for row in last["segments"])["UNSCORABLE"] == 2
    october = result.cohorts["months"]["2025-10"]
    assert october["cohort_size"] == 1 and october["horizons"]["30"]["repeated"] == 1
    assert october["retention"][1]["active"] == 1
    assert result.metadata["detail"]["available"] is False
    assert dto.read_envelope(json.loads(json.dumps(result.envelope()))) == result


def test_serialized_schema4_parity_and_no_raw_fallback(source, monkeypatch):
    _write(source)
    root, catalog, _, _, target = source
    index.build_unified_index(raw_root=root, catalog_path=catalog, index_dir=target, expected_bytes=1024**2)
    expected = _report(source, selected_items=["000002"])
    pointer = target / index.POINTER_NAME
    saved = pointer.read_bytes()

    def no_raw(*args, **kwargs):
        pytest.fail("INDEXED cannot reread raw Orders or Actions")

    with monkeypatch.context() as patch:
        patch.setattr(index, "_records", no_raw)
        patch.setattr(service, "_reference_actions", no_raw)
        actual = _report(source, reader="INDEXED", index_dir=target, selected_items=["000002"])
    assert actual.source["projection_schema"] == 4
    assert _business(actual) == _business(expected)
    assert actual.contexts["order_product_slice"]["product_filter_key"]
    assert actual.contexts["rfm_48m"]["product_filter_key"] is None
    assert actual.orders["purchase_orders"]["known"] == 1
    assert actual.rfm["base_buyers"] == 2 and actual.cohorts["quality"]["base_buyers"] == 2
    assert pointer.read_bytes() == saved


@pytest.mark.parametrize("mode", ["latest", "slice"])
def test_cache_roundtrip_uses_pinned_fingerprint_and_skips_raw_spool(source, monkeypatch, mode):
    _write(source)
    directory = source[0] / "cache"
    first = _report(source, cache_dir=directory, cache_mode=mode)
    stored, = directory.iterdir()
    payload = json.loads(stored.read_text(encoding="utf-8"))
    assert payload["schema_version" if mode == "latest" else "schema"] == (11 if mode == "latest" else 2)

    def no_spool(*args, **kwargs):
        pytest.fail("A valid REFERENCE cache hit must precede raw projection")

    monkeypatch.setattr(service, "reference_order_reader", no_spool)
    assert _report(source, cache_dir=directory, cache_mode=mode) == first
    assert cache.UnifiedCache(directory, mode=mode).load("a"*64) is None
    assert dto.read_envelope(first.envelope()) == first


def test_legacy_schema10_reader_preserves_values_and_does_not_rewrite(source):
    root, catalog, _, _, _ = source
    report = legacy.calculate_statistics(raw_root=root, catalog_path=catalog, calculated_at=rfm_tests._NOW)
    path = root / "legacy-schema10.json"
    legacy_cache.save_result(path, asdict(report))
    old = path.read_bytes()
    decoded = dto.read_statistics_result(path)
    assert decoded == legacy_cache.load_result(path)
    assert path.read_bytes() == old
    directory = root / "cache"
    (directory / "unified-v11-latest.json").write_bytes(old)
    assert cache.UnifiedCache(directory).load("a"*64) is None
    assert path.read_bytes() == old
    with pytest.raises(dto.StatisticsResultError, match="INCOMPATIBLE_RESULT_SCHEMA"):
        dto.read_envelope({"schema_version": 12, "result": decoded})


@pytest.mark.parametrize(("field", "error"), [
    ("schema", "INCOMPATIBLE_RESULT_SCHEMA"),
    ("policy", "INCOMPATIBLE_POLICY"),
    ("contexts", "CONTEXT_BINDING_MISMATCH"),
    ("money", "INVALID_MEASUREMENT"),
    ("refund", "INVALID_REFUND_VALUE"),
    ("orders", "INVALID_STATE_DISTRIBUTION"),
    ("cohorts", "INVALID_COHORT_DENOMINATOR"),
    ("rfm", "INVALID_RFM_SEGMENTS"),
])
def test_validator_rejects_semantic_corruption_even_with_recomputed_content_digest(source, field, error):
    _write(source)
    envelope = deepcopy(_report(source).envelope())
    raw = envelope["result"]
    if field == "schema":
        envelope["schema_version"] = 99
    elif field == "policy":
        raw["policy"]["policy_id"] = "STRICT_COMPLETED"
    elif field == "contexts":
        raw["contexts"]["rfm_48m"]["identity_space"] = "wrong"
    elif field == "money":
        raw["money"]["currencies"]["RUB"]["amounts"]["recognized_purchase"]["value"] = "0"
    elif field == "refund":
        raw["money"]["currencies"]["RUB"]["amounts"]["confirmed_refund"] = dict(
            known="0", value="0", quality="KNOWN", reasons=[])
    elif field == "orders":
        cell = raw["orders"]["state_distribution"]["COMPLETED_ONLY"]
        cell["known"] += 100
        if cell["value"] is not None:
            cell["value"] = cell["known"]
    elif field == "cohorts":
        raw["cohorts"]["months"]["2025-10"]["horizons"]["30"]["confirmed_without_event"] = 1
    elif field == "rfm":
        raw["rfm"]["windows"]["LAST_12M"]["currencies"]["RUB"]["segments"][-1][1] = 0
    raw["metadata"]["content_digest"] = dto.content_digest(raw)
    with pytest.raises(dto.StatisticsResultError, match=error):
        dto.read_envelope(envelope)


def test_cache_invalidates_source_code_filter_and_rejects_changed_policy(source, monkeypatch):
    rows = _write(source)
    directory = source[0] / "cache"
    initial = _report(source, cache_dir=directory)
    path = directory / "unified-v11-latest.json"
    altered = json.loads(path.read_text(encoding="utf-8"))
    altered["result"]["policy"]["time_basis_id"] = "another-basis"
    path.write_text(json.dumps(altered), encoding="utf-8")
    assert cache.UnifiedCache(directory).load(initial.metadata["fingerprint"]) is None
    fresh = _report(source, cache_dir=directory)
    assert fresh == initial
    rows[-1]["lines"][0]["status"]["ids"]["externalId"] = "F"
    rfm_tests._write_orders(source, rows)
    changed = _report(source, cache_dir=directory)
    assert changed.orders["purchase_orders"]["known"] == 5
    assert changed.metadata["fingerprint"] != initial.metadata["fingerprint"]
    version = "7"*64
    monkeypatch.setattr(service, "algorithm_version", lambda: version)
    recalculated = _report(source, cache_dir=directory)
    assert recalculated.metadata["calculation_algorithm"] == version
    assert recalculated.metadata["fingerprint"] != changed.metadata["fingerprint"]
    restricted = _report(source, cache_dir=directory, selected_items=["000002"])
    assert restricted.metadata["fingerprint"] != recalculated.metadata["fingerprint"]
    assert restricted.rfm == recalculated.rfm and restricted.cohorts == recalculated.cohorts


@pytest.mark.parametrize(("legacy_only", "error"), [(True, "INDEX_MISSING"), (False, "INCOMPATIBLE_SCHEMA")])
def test_missing_or_incompatible_index_is_fail_closed(source, legacy_only, error):
    _write(source)
    target = source[-1]
    sentinel = b'{"schema":3}'
    (target / "current.json").write_bytes(sentinel)
    if not legacy_only:
        (target / index.POINTER_NAME).write_bytes(sentinel)
    directory = source[0] / "cache"
    with pytest.raises(StatisticsIndexError, match=error):
        _report(source, reader="INDEXED", index_dir=target, cache_dir=directory)
    assert (target / "current.json").read_bytes() == sentinel
    assert not list(directory.iterdir())


@pytest.mark.parametrize("failure", ["calculation_cancel", "write_cancel", "replace_error"])
def test_error_or_cancel_preserves_previous_cache_and_removes_own_temporary_file(source, monkeypatch, failure):
    _write(source)
    root, _, _, scratch, _ = source
    directory = root / "cache"
    previous = _report(source, cache_dir=directory)
    path = directory / "unified-v11-latest.json"
    before = path.read_bytes()
    options = dict(cache_dir=directory, selected_items=["000002"])
    if failure == "calculation_cancel":
        active = True
        original = service.cohorts._spool_order

        def stop(*args, **kwargs):
            nonlocal active
            active = False
            return original(*args, **kwargs)

        monkeypatch.setattr(service.cohorts, "_spool_order", stop)
        options["cancelled"] = lambda: not active
    elif failure == "write_cancel":
        options["cancelled"] = lambda: any(directory.glob(".unified-v11-*.tmp"))
    else:
        original = cache.os.replace

        def reject(temp, target):
            if target == path:
                raise PermissionError("synthetic publication failure")
            return original(temp, target)

        monkeypatch.setattr(cache.os, "replace", reject)
    expected = PermissionError if failure == "replace_error" else InterruptedError
    with pytest.raises(expected):
        _report(source, **options)
    assert path.read_bytes() == before
    assert cache.UnifiedCache(directory).load(previous.metadata["fingerprint"]) == previous
    assert list(directory.iterdir()) == [path] and not list(scratch.iterdir())


def test_unavailable_orders_still_return_a_complete_report(source):
    root, _, data, _, _ = source
    data["manual_interactions"] = None
    data["orders"].clear()
    cohort_tests.save(root, data)
    result = _report(source, calculated_at=datetime(2026, 1, 3, tzinfo=timezone.utc), selected_end=None)
    assert result.orders["purchase_orders"]["value"] is None
    assert result.rfm["quality"]["availability"] == "NO_ORDERS_COVERAGE"
    assert result.cohorts["quality"]["availability"] == "NO_ORDERS_COVERAGE"
    assert len(result.status_catalog["rows"]) >= 43


def test_source_change_after_calculation_leaves_previous_cache(source, monkeypatch):
    rows = _write(source)
    directory = source[0] / "cache"
    old = _report(source, cache_dir=directory)
    path = directory / "unified-v11-latest.json"
    before = path.read_bytes()
    assemble = service.assemble_result

    def change_source(*args, **kwargs):
        result = assemble(*args, **kwargs)
        rfm_tests._write_orders(source, rows[:-1])
        return result

    monkeypatch.setattr(service, "assemble_result", change_source)
    with pytest.raises(StatisticsIndexError, match="STALE_SOURCE"):
        _report(source, cache_dir=directory, selected_items=["000002"])
    assert path.read_bytes() == before
    assert cache.UnifiedCache(directory).load(old.metadata["fingerprint"]) == old


def test_cache_rejects_hard_link_before_read_or_replace(source):
    _write(source)
    root = source[0]
    directory = root / "cache"
    original = root / "external-report.json"
    original.write_bytes(b"outside cache")
    link = directory / "unified-v11-latest.json"
    try:
        link.hardlink_to(original)
    except OSError:
        pytest.skip("Filesystem does not support test hard links")
    target = cache.UnifiedCache(directory)
    with pytest.raises(StatisticsIndexError, match="UNSAFE_CACHE_FILE"):
        target.load("a"*64)
    with pytest.raises(StatisticsIndexError, match="UNSAFE_CACHE_FILE"):
        target.save(_report(source))
    assert original.read_bytes() == b"outside cache"
