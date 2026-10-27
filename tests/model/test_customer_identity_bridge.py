"""Synthetic identity/source tests; no real customers or history reads."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone
import json
from threading import Event

import pytest
import torch

from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.records import CustomerMergeRecord
from Application.mindbox.selection import DEFAULT_SELECTION
from Application.model import customer_identity_bridge as bridge
from Application.model.raw_bpr_inference import RawBprInferenceService, RawBprStatus


S = bridge.CustomerModelIdentityStatus


def _resolver(edges):
    return CustomerIdResolver(CustomerMergeRecord("synthetic", datetime(2026, 1, 1, tzinfo=timezone.utc),
                                                 target, (source,)) for source, target in edges)


@pytest.mark.parametrize("edges,users,query,canonical,model,status,count", [
    ([], ["A"], "A", "A", "A", S.MATCHED_EXACT, 1),
    ([("A", "B")], ["A"], "B", "B", "A", S.MATCHED_VIA_MERGE, 1),
    ([("A", "B"), ("B", "C")], ["A"], "A", "C", "A", S.MATCHED_VIA_MERGE, 1),
    ([("B", "C"), ("A", "B")], ["A"], "C", "C", "A", S.MATCHED_VIA_MERGE, 1),
    ([("A", "C"), ("B", "C")], ["A", "B"], "C", "C", None, S.AMBIGUOUS, 2),
    ([("A", "C"), ("B", "C")], ["A", "B", "C"], "C", "C", None, S.AMBIGUOUS, 3),
    ([("A", "C"), ("B", "C")], ["A", "B", "C"], "A", "C", None, S.AMBIGUOUS, 3),
    ([("A", "C"), ("B", "C")], ["A"], "B", "C", "A", S.MATCHED_VIA_MERGE, 1),
    ([("A", "C")], ["C"], "A", "C", "C", S.MATCHED_EXACT, 1),
    ([("A", "C")], ["D"], "A", "C", None, S.NOT_IN_MODEL, 0),
    ([], ["A"], "unknown", "unknown", None, S.NOT_IN_MODEL, 0),
    ([], ["00123"], "123", "123", None, S.NOT_IN_MODEL, 0),
])
def test_identity_policy(edges, users, query, canonical, model, status, count):
    result = bridge.CustomerIdentityIndex(_resolver(edges), users).resolve(query)
    assert result.canonical_customer_id == canonical
    assert result.model_user_id == model
    assert result.status is status
    assert result.candidate_count == count
    assert result.is_matched == (count == 1)


def test_index_preserves_resolver_mappings_and_group_uniqueness():
    resolver = _resolver([("A", "C"), ("B", "C"), ("D", "E")])
    before = dict(resolver._parents), dict(resolver._canonical)
    users = ["A", "B", "D", "F"]
    index = bridge.CustomerIdentityIndex(resolver, users)
    assert index._groups == {"C": ("A", "B"), "E": ("D",), "F": ("F",)}
    assert index.diagnostics == bridge.CustomerIdentityDiagnostics(4, 1, 3, 2, 1, 2)
    for query in ("A", "C", "unknown", "F"):
        index.resolve(query)
    assert before == (resolver._parents, resolver._canonical)
    assert users == ["A", "B", "D", "F"]
    members = [user for group in index._groups.values() for user in group]
    assert sorted(members) == sorted(users) and len(members) == len(set(members))
    users.append("NEW")
    assert index.resolve("NEW").status is S.NOT_IN_MODEL
    with pytest.raises(TypeError):
        index._groups["C"] = ("A",)


@pytest.mark.parametrize("value", [None, 12, True, [], "", " ", " A", "A ", "\tA"])
def test_invalid_id_never_loads_sources(value, monkeypatch):
    monkeypatch.setattr(bridge, "_capture_sources", lambda *a: pytest.fail("No source reads for INVALID_ID"))
    result = bridge.CustomerIdentityBridge().resolve(value)
    assert result.status is S.INVALID_ID
    assert result.canonical_customer_id is result.model_user_id is None
    assert result.candidate_count == 0


def test_results_are_immutable_private_and_validate_invariants():
    index = bridge.CustomerIdentityIndex(_resolver([]), ["PRIVATE_CUSTOMER_ID"])
    result = index.resolve("PRIVATE_CUSTOMER_ID")
    assert "PRIVATE_CUSTOMER_ID" not in repr(result)
    assert "PRIVATE_CUSTOMER_ID" not in repr(index)
    assert "PRIVATE_CUSTOMER_ID" not in repr(index.diagnostics)
    with pytest.raises(FrozenInstanceError):
        result.model_user_id = "B"
    with pytest.raises(FrozenInstanceError):
        index.diagnostics.model_users = 99
    for canonical, user, status, count in [
        ("C", "A", S.AMBIGUOUS, 2), ("C", "A", S.MATCHED_VIA_MERGE, 2),
        ("C", None, S.AMBIGUOUS, 1), ("C", None, S.NOT_IN_MODEL, 1),
        ("C", "A", S.MATCHED_EXACT, 1), ("C", "C", S.MATCHED_VIA_MERGE, 1),
    ]:
        with pytest.raises(ValueError):
            bridge.CustomerModelIdentityResult(canonical, user, status, count)
    with pytest.raises(TypeError):
        bridge.CustomerModelIdentityResult(None, None, "NOT_IN_MODEL", 0)


def _model(root, generation, users):
    directory = root if generation is None else root / "runs" / generation
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mappings.json").write_text(json.dumps({"idx2user": users, "idx2item": ["item"]}), encoding="utf-8")
    # Identity lookup must not deserialize this deliberately invalid tensor payload.
    (directory / "bprmf.pt").write_bytes(b"identity needs mappings only")
    if generation is not None:
        (root / "current.json").write_text(json.dumps({"generation": generation}), encoding="utf-8")
    return directory


def _merges(root, edges, *, generation="1" * 32, revision="revision-a"):
    directory = root / "canonical/objects" / generation / "customer_merges"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"synthetic-{i}", "dateTimeUtc": "2026-01-01T00:00:00Z",
             "resultingCustomer": {"ids": {"mindboxId": target}},
             "mergedCustomers": [{"ids": {"mindboxId": source}}]} for i, (source, target) in enumerate(edges)]
    (directory / "customer_merges_part_001.json").write_text(json.dumps({"customerMerges": rows}), encoding="utf-8")
    entry = {"directory": directory.relative_to(root).as_posix(), "parts": 1,
             "since": "2022-01-01T00:00:00+00:00", "until": "2026-09-01T00:00:00+00:00",
             "updated": "2026-10-12T00:00:00+00:00", "source_kind": "API"}
    data = {"schema_version": 1, "revision": revision, "actions": {}, "orders": {},
            "customer_merges": entry, "selection": asdict(DEFAULT_SELECTION)}
    (root / "canonical/catalog.json").write_text(json.dumps(data), encoding="utf-8")
    return directory


@pytest.fixture
def sources(tmp_path):
    model_root, raw_root = tmp_path / "model", tmp_path / "raw"
    model_directory = _model(model_root, "generation-a", ["A", "D"])
    merge_directory = _merges(raw_root, [("A", "C"), ("B", "C")])
    return model_root, raw_root, model_directory, merge_directory


def _resolve(service, sources, customer="C"):
    return service.resolve(customer, model_dir=sources[0], raw_root=sources[1])


def _diagnostics(service, sources):
    return service.get_diagnostics(model_dir=sources[0], raw_root=sources[1])


def test_public_api_reads_only_current_mappings_and_merges(sources, monkeypatch):
    service = bridge.CustomerIdentityBridge()
    monkeypatch.setattr(bridge, "_DEFAULT_BRIDGE", service)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("No checkpoint loading"))
    real_export = bridge.iter_export
    calls = []
    def export(name, **kwargs):
        assert name == "customer_merges"
        calls.append(name)
        return real_export(name, **kwargs)
    monkeypatch.setattr(bridge, "iter_export", export)
    result = bridge.resolve_customer_model_identity("C", model_dir=sources[0], raw_root=sources[1])
    assert result.status is S.MATCHED_VIA_MERGE and result.model_user_id == "A"
    assert result.model_generation == "generation-a" and result.canonical_revision == "revision-a"
    assert result.source_version is not None
    assert _resolve(service, sources, "D").status is S.MATCHED_EXACT
    assert _diagnostics(service, sources) == bridge.CustomerIdentityDiagnostics(2, 1, 1, 2, 0, 0)
    assert calls == ["customer_merges"]
    assert not (sources[1] / "canonical/customers.sqlite").exists()


def test_generation_switch_has_no_exact_match_preference_or_stale_cache(sources):
    service = bridge.CustomerIdentityBridge()
    old = _resolve(service, sources)
    _model(sources[0], "generation-b", ["A", "B", "C"])
    result = _resolve(service, sources)
    assert result.status is S.AMBIGUOUS and result.candidate_count == 3
    assert result.model_user_id is None and result.model_generation == "generation-b"
    assert old.source_version != result.source_version
    assert not service.is_current(old, model_dir=sources[0], raw_root=sources[1])


def test_merge_publication_invalidates_index(sources):
    service = bridge.CustomerIdentityBridge()
    old = _resolve(service, sources)
    _merges(sources[1], [("A", "D")], generation="2" * 32, revision="revision-b")
    missing = _resolve(service, sources)
    ambiguous = _resolve(service, sources, "D")
    assert missing.status is S.NOT_IN_MODEL
    assert ambiguous.status is S.AMBIGUOUS and ambiguous.candidate_count == 2
    assert missing.canonical_revision == "revision-b"
    assert old.source_version != missing.source_version


@pytest.mark.parametrize("source", ["mappings", "merges"])
def test_in_place_source_change_is_detected(sources, source):
    service = bridge.CustomerIdentityBridge()
    _resolve(service, sources)
    if source == "mappings":
        (sources[2] / "mappings.json").write_text(json.dumps({"idx2user": ["D"], "idx2item": ["item"]}), encoding="utf-8")
    else:
        (sources[3] / "customer_merges_part_001.json").write_text('{"customerMerges":[]}', encoding="utf-8")
    assert _resolve(service, sources).status is S.NOT_IN_MODEL


def test_customers_changes_are_irrelevant_to_identity(sources, monkeypatch):
    service = bridge.CustomerIdentityBridge()
    old = _resolve(service, sources)
    monkeypatch.setattr(bridge, "_build_index", lambda *a: pytest.fail("No rebuild for Customers update"))
    (sources[1] / "canonical/customers.sqlite").write_bytes(b"must not be opened")
    assert _resolve(service, sources) == old


@pytest.mark.parametrize("payload", ["broken", "[]", "{}", '{"generation":"../escape"}', '{"generation":"absent"}',
                                    '{"generation":"generation-b","generation":"generation-a"}'])
def test_bad_new_model_never_uses_old_cache(sources, payload):
    service = bridge.CustomerIdentityBridge()
    _resolve(service, sources)
    (sources[0] / "current.json").write_text(payload, encoding="utf-8")
    result = _resolve(service, sources)
    expected = S.SOURCE_UNAVAILABLE if "absent" in payload else S.SOURCE_INCONSISTENT
    assert result.status is expected and result.model_user_id is None
    assert service._cached is None


@pytest.mark.parametrize("payload", ["broken", "[]", "{}", '{"idx2user":["A","A"],"idx2item":["item"]}',
    '{"idx2user":[12],"idx2item":["item"]}', '{"idx2user":[],"idx2item":["item"]}',
    '{"idx2user":[" A"],"idx2item":["item"]}', '{"idx2user":["A"],"idx2item":[]}',
    '{"idx2user":["A"],"idx2user":["D"],"idx2item":["item"]}'])
def test_invalid_mappings_are_source_error_not_not_in_model(sources, payload):
    service = bridge.CustomerIdentityBridge()
    _resolve(service, sources)
    (sources[2] / "mappings.json").write_text(payload, encoding="utf-8")
    result = _resolve(service, sources)
    assert result.status is S.SOURCE_INCONSISTENT
    assert result.canonical_customer_id is result.model_user_id is None
    assert service._cached is None


@pytest.mark.parametrize("edges", [[("A", "C"), ("A", "D")], [("A", "B"), ("B", "A")], [("A", "A")]])
def test_conflict_and_cycle_never_return_a_match(sources, edges):
    service = bridge.CustomerIdentityBridge()
    _resolve(service, sources)
    _merges(sources[1], edges, generation="2" * 32, revision="revision-b")
    result = _resolve(service, sources)
    assert result.status is S.SOURCE_INCONSISTENT and result.model_user_id is None
    assert service._cached is None


@pytest.mark.parametrize("payload", ["broken PRIVATE_CUSTOMER_ID", '{"customerMerges":[{}]}',
                                    '{"customerMerges":[],"extra":[]}', '{"customerMerges":[]'])
def test_invalid_merge_json_is_rejected_without_payload_leaks(sources, payload, capsys, caplog):
    (sources[3] / "customer_merges_part_001.json").write_text(payload, encoding="utf-8")
    result = _resolve(bridge.CustomerIdentityBridge(), sources, "PRIVATE_CUSTOMER_ID")
    assert result.status is S.SOURCE_INCONSISTENT
    assert "PRIVATE_CUSTOMER_ID" not in repr(result) + caplog.text + str(capsys.readouterr())


@pytest.mark.parametrize("component", ["catalog", "merges", "mappings", "checkpoint"])
def test_temporary_unavailability_clears_cache_and_can_recover(sources, component):
    service = bridge.CustomerIdentityBridge()
    old = _resolve(service, sources)
    path = {"catalog": sources[1] / "canonical/catalog.json",
            "merges": sources[3] / "customer_merges_part_001.json",
            "mappings": sources[2] / "mappings.json", "checkpoint": sources[2] / "bprmf.pt"}[component]
    contents = path.read_bytes()
    path.unlink()
    result = _resolve(service, sources)
    assert result.status is S.SOURCE_UNAVAILABLE and result.model_user_id is None
    assert service._cached is None
    assert _diagnostics(service, sources) is None
    assert not service.is_current(old, model_dir=sources[0], raw_root=sources[1])
    path.write_bytes(contents)
    assert _resolve(service, sources).status is S.MATCHED_VIA_MERGE


@pytest.mark.parametrize("mutation", ["no_merges", "parts", "revision", "directory", "coverage", "incomplete", "schema"])
def test_invalid_catalog_contract_and_missing_graph(sources, mutation):
    path = sources[1] / "canonical/catalog.json"
    data = json.loads(path.read_text())
    if mutation == "no_merges":
        data["customer_merges"] = None
    elif mutation == "parts":
        data["customer_merges"]["parts"] = 2
    elif mutation == "revision":
        data["revision"] = 12
    elif mutation == "directory":
        data["customer_merges"]["directory"] = "../unsafe"
    elif mutation == "coverage":
        data["customer_merges"]["until"] = data["customer_merges"]["since"]
    elif mutation == "incomplete":
        (sources[3] / "unfinished.download").write_bytes(b"incomplete")
    elif mutation == "schema":
        data["schema_version"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    result = _resolve(bridge.CustomerIdentityBridge(), sources)
    assert result.status is (S.SOURCE_UNAVAILABLE if mutation == "no_merges" else S.SOURCE_INCONSISTENT)


def test_failed_new_index_never_falls_back_or_logs_exception(sources, monkeypatch, caplog, capsys):
    service = bridge.CustomerIdentityBridge()
    _resolve(service, sources)
    _merges(sources[1], [], generation="2" * 32, revision="revision-b")
    def fail(*args):
        raise RuntimeError("PRIVATE_CUSTOMER_ID")
    monkeypatch.setattr(bridge, "_build_index", fail)
    result = _resolve(service, sources)
    assert result.status is S.SOURCE_INCONSISTENT and service._cached is None
    assert "PRIVATE_CUSTOMER_ID" not in repr(result) + caplog.text + str(capsys.readouterr())


def test_publication_during_build_retries_with_consistent_generation(sources, monkeypatch):
    real_build = bridge._build_index
    calls = []
    def build(snapshot):
        calls.append(snapshot.generation)
        result = real_build(snapshot)
        if len(calls) == 1:
            _model(sources[0], "generation-b", ["C"])
        return result
    monkeypatch.setattr(bridge, "_build_index", build)
    result = _resolve(bridge.CustomerIdentityBridge(), sources)
    assert calls == ["generation-a", "generation-b"]
    assert result.status is S.MATCHED_EXACT and result.model_user_id == "C"
    assert result.model_generation == "generation-b"


def test_merges_publication_during_build_retries(sources, monkeypatch):
    real_build = bridge._build_index
    calls = []
    def build(snapshot):
        calls.append(snapshot.revision)
        result = real_build(snapshot)
        if len(calls) == 1:
            _merges(sources[1], [("A", "D")], generation="2" * 32, revision="revision-b")
        return result
    monkeypatch.setattr(bridge, "_build_index", build)
    result = _resolve(bridge.CustomerIdentityBridge(), sources, "D")
    assert calls == ["revision-a", "revision-b"]
    assert result.status is S.AMBIGUOUS and result.candidate_count == 2
    assert result.canonical_revision == "revision-b"


def test_unstable_sources_stop_after_one_retry(sources, monkeypatch):
    real_build = bridge._build_index
    calls = []
    def build(snapshot):
        calls.append(snapshot.revision)
        result = real_build(snapshot)
        _merges(sources[1], [], generation=str(len(calls) + 1) * 32, revision=f"revision-{len(calls)}")
        return result
    monkeypatch.setattr(bridge, "_build_index", build)
    service = bridge.CustomerIdentityBridge()
    result = _resolve(service, sources)
    assert result.status is S.SOURCE_INCONSISTENT and len(calls) == 2
    assert service._cached is None


def test_parallel_queries_publish_one_complete_index(sources, monkeypatch):
    service = bridge.CustomerIdentityBridge()
    entered, release = Event(), Event()
    real_build = bridge._build_index
    calls = []
    def build(snapshot):
        calls.append(snapshot)
        entered.set()
        assert release.wait(5)
        return real_build(snapshot)
    monkeypatch.setattr(bridge, "_build_index", build)
    with ThreadPoolExecutor(max_workers=4) as pool:
        first = pool.submit(_resolve, service, sources)
        assert entered.wait(5)
        assert service._cached is None
        others = [pool.submit(_resolve, service, sources, user) for user in ("A", "C", "D")]
        release.set()
        results = [first.result(), *[f.result() for f in others]]
    assert len(calls) == 1
    assert all(r.is_matched for r in results)
    assert len({r.source_version for r in results}) == 1


def test_separate_roots_and_legacy_model_do_not_mix(sources, tmp_path):
    service = bridge.CustomerIdentityBridge()
    old = _resolve(service, sources)
    other_model, other_raw = tmp_path / "other-model", tmp_path / "other-raw"
    _model(other_model, None, ["C"])
    _merges(other_raw, [])
    result = service.resolve("C", model_dir=other_model, raw_root=other_raw)
    assert result.status is S.MATCHED_EXACT and result.model_generation is None
    assert result.source_version != old.source_version


def test_guarded_raw_bpr_handoff_uses_only_confirmed_id_and_same_snapshot(sources):
    from Application.model import BPRMF as core
    model = core.BPRMF(2, 1, 2)
    with torch.no_grad():
        model.user_emb.weight.copy_(torch.tensor([[2., 0.], [-2., 0.]]))
        model.item_emb.weight.copy_(torch.tensor([[1., 0.]]))
    torch.save({"model_type": "bprmf", "num_users": 2, "num_items": 1,
                "config": {"embedding_dim": 2}, "state_dict": model.state_dict()}, sources[2] / "bprmf.pt")
    service = bridge.CustomerIdentityBridge()
    identity = _resolve(service, sources)
    assert identity.is_matched and identity.model_user_id == "A"
    inference = RawBprInferenceService().get_recommendations(identity.model_user_id, model_dir=sources[0], k=1)
    assert inference.status is RawBprStatus.OK and inference.recommendations[0].raw_score == 2
    assert inference.model_generation == identity.model_generation
    assert service.is_current(identity, model_dir=sources[0], raw_root=sources[1])
    # A changed generation cannot be presented under the previous identity.
    _model(sources[0], "generation-b", ["D"])
    assert not service.is_current(identity, model_dir=sources[0], raw_root=sources[1])


def test_ambiguity_and_all_unmatched_statuses_cannot_call_raw_bpr(sources, monkeypatch):
    from Application.model import raw_bpr_inference
    monkeypatch.setattr(raw_bpr_inference, "get_raw_bpr_recommendations",
                        lambda *a, **k: pytest.fail("Unmatched identity cannot call Raw BPR"))
    _model(sources[0], "generation-b", ["A", "B", "C"])
    service = bridge.CustomerIdentityBridge()
    results = [_resolve(service, sources, customer) for customer in ("C", "unknown", " ")]
    (sources[1] / "canonical/catalog.json").unlink()
    results.append(_resolve(service, sources))
    assert {r.status for r in results} == {S.AMBIGUOUS, S.NOT_IN_MODEL, S.INVALID_ID, S.SOURCE_UNAVAILABLE}
    for result in results:
        assert not result.is_matched and result.model_user_id is None
        if result.is_matched:
            raw_bpr_inference.get_raw_bpr_recommendations(result.model_user_id)
