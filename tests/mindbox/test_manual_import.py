from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket

import pytest

from Application.mindbox import daily_training_batch as daily, manual_import as manual
from Application.mindbox.customer_profile_snapshot import load_customer_profile_snapshot, load_customer_contact_index
from Application.mindbox.raw_reader import iter_export, EXPORT_ROOTS
from Application.mindbox.selection import DEFAULT_SELECTION
from Application.mindbox.training_batch import TrainingBatchExport, TrainingBatchWindow


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("No network"))
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    window = TrainingBatchWindow(start, start + timedelta(days=1), start)
    components = []
    for name in ("customer_merges", "actions", "orders"):
        directory = tmp_path / name / "20260802_000000"
        directory.mkdir(parents=True)
        records = []
        if name == "customer_merges":
            records = [{"id": 1, "dateTimeUtc": "2026-08-01T12:00:00Z",
                        "resultingCustomer": {"ids": {"mindboxId": 2}},
                        "mergedCustomers": [{"ids": {"mindboxId": 1}}]}]
        (directory / f"{name}_part_001.json").write_text(json.dumps({EXPORT_ROOTS[name]: records}), encoding="utf-8")
        entry = TrainingBatchExport(name, "api-export", name, directory.relative_to(tmp_path).as_posix(), 1)
        components.append(daily.BatchComponent(name, start, window.interaction_until, name, "READY", entry))
    batch = daily.ChunkedTrainingBatch("a" * 32, start, window, "0" * 64, tuple(components), True)
    path = tmp_path / "training_batches" / batch.batch_id / "manifest.json"
    path.parent.mkdir(parents=True)
    daily._atomic_write(path, batch)
    return tmp_path, batch, path


def write(root, name, records=None, text=None):
    path = root / (name + ".json")
    path.write_text(text if text is not None else json.dumps({EXPORT_ROOTS[name]: records or []}), encoding="utf-8")
    return path


def test_manual_batch_atomic_metadata_selection_and_common_readers(saved, monkeypatch):
    root, source, _ = saved
    actions = write(root, "actions", [{"arbitrary": 1}])
    orders = write(root, "orders")
    before = actions.read_bytes()
    selection = replace(DEFAULT_SELECTION, view_action_system_names=("CustomView",))
    progress = []
    original = manual.os.replace
    def publish(src, dst):
        if Path(src).is_dir():
            assert not Path(dst).exists()
            assert (Path(src) / "manifest.json").is_file()
            assert (Path(src) / "actions/actions_part_001.json").read_bytes() == before
        return original(src, dst)
    monkeypatch.setattr(manual.os, "replace", publish)
    batch = manual.import_interactions(actions, orders, raw_root=root, window=source.window,
                                        selection=selection, progress=progress.append)
    manifest = root / "training_batches" / batch.batch_id / "manifest.json"
    loaded = daily.load_chunked_training_batch(manifest, raw_root=root, require_complete=True)
    assert loaded == batch and loaded.selection == selection
    assert batch.source_kind == "MANUAL"
    assert batch.merge_source_training_batch_id == source.batch_id
    assert batch.components[0] == source.components[0]
    for component in batch.components[1:]:
        assert component.export.source_kind == "MANUAL" and component.export.export_id is None
    assert list(iter_export("actions", input_dir=root / batch.components[1].export.relative_directory)) == [{"arbitrary": 1}]
    assert actions.read_bytes() == before
    assert str(actions) not in manifest.read_text(encoding="utf-8")
    assert any("API snapshot" in line for line in progress)
    assert manual.select_merges_source(root).batch_id == source.batch_id  # Manual batch cannot become merges source.


@pytest.mark.parametrize("name", ["actions", "orders", "customers"])
@pytest.mark.parametrize("text", ['{"wrong": []}', '{"orders": [], "extra": 1}', '{',
                                  '{"ROOT":[{"ids":{"mindboxId":1,"mindboxId":2}}]}',
                                  '{"ROOT":[null]}', '{"ROOT":[]} trailing'])
def test_strict_json_rejection_no_publication(saved, name, text):
    root, batch, _ = saved
    bad = write(root, name, text=text.replace("ROOT", EXPORT_ROOTS[name]))
    before = bad.read_bytes()
    with pytest.raises(manual.ManualImportError):
        if name == "customers":
            manual.import_customers(bad, raw_root=root)
        else:
            paths = {name: bad, ("orders" if name == "actions" else "actions"): write(root, "orders" if name == "actions" else "actions")}
            manual.import_interactions(**paths, raw_root=root, window=batch.window)
    assert bad.read_bytes() == before
    assert len(list((root / "training_batches").glob("*/manifest.json"))) == 1
    assert not list((root / "customer_profile_snapshots").glob("*/manifest.json"))
    assert not list(root.glob("*/.manual-*"))


@pytest.mark.parametrize("operation", ["interactions", "customers"])
def test_cancel_and_failed_commit_do_not_publish(saved, monkeypatch, operation):
    root, batch, _ = saved
    paths = {name: write(root, name) for name in ("actions", "orders", "customers")}
    def run(**kwargs):
        if operation == "customers":
            return manual.import_customers(paths["customers"], raw_root=root, **kwargs)
        return manual.import_interactions(paths["actions"], paths["orders"], raw_root=root, window=batch.window, **kwargs)
    with pytest.raises(InterruptedError):
        run(cancelled=lambda: True)
    original = manual.os.replace
    def fail_commit(src, dst):
        if Path(src).is_dir():
            raise OSError("synthetic")
        return original(src, dst)
    monkeypatch.setattr(manual.os, "replace", fail_commit)
    with pytest.raises(OSError):
        run()
    assert len(list((root / "training_batches").glob("*/manifest.json"))) == 1
    assert not list((root / "customer_profile_snapshots").glob("*/manifest.json"))


def test_missing_incompatible_or_broken_merges_blocks(saved):
    root, batch, manifest = saved
    with pytest.raises(manual.ManualImportError):
        manual.select_merges_source(root / "missing")
    with pytest.raises(manual.ManualImportError):
        manual.select_merges_source(root, replace(batch.window, interaction_until=batch.window.interaction_until + timedelta(days=1)))
    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(manual.ManualImportError, match="повреждён"):
        manual.select_merges_source(root)


def test_streamed_manual_customers_numeric_ids_contacts_and_old_snapshot(saved, monkeypatch):
    root, batch, _ = saved
    source = write(root, "customers", [{"ids": {"mindboxId": 1}, "mobilePhone": 79999999999,
                                       "changeDateTimeUtc": "2026-08-03T00:00:00Z"}])
    original_load = json.load
    def metadata_only(stream, *args, **kwargs):
        assert Path(stream.name).name == "manifest.json"
        return original_load(stream, *args, **kwargs)
    monkeypatch.setattr(json, "load", metadata_only)
    snapshot = manual.import_customers(source, raw_root=root)
    manifest = root / "customer_profile_snapshots" / snapshot.snapshot_id / "manifest.json"
    assert load_customer_profile_snapshot(manifest, raw_root=root) == snapshot
    assert snapshot.originating_training_batch_id is None
    assert snapshot.merge_source_training_batch_id == batch.batch_id
    assert snapshot.source_kind == "MANUAL"
    index = load_customer_contact_index(manifest, {"idx2user": ["2"]}, raw_root=root)
    assert index.diagnostics.matched_profiles == 1
    assert index.contacts[0].mobile_phone == "79999999999"


@pytest.mark.parametrize("version", [2, 3])
def test_old_api_versions_keep_defaults_and_selection(saved, version):
    root, batch, path = saved
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = version
    del data["source_kind"], data["merge_source_training_batch_id"]
    for component in data["components"]:
        del component["export"]["source_kind"]
    if version == 2:
        del data["selection"]
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = daily.load_chunked_training_batch(path, raw_root=root, require_complete=True)
    assert loaded.selection == DEFAULT_SELECTION and loaded.source_kind == "API"
    assert manual.select_merges_source(root).batch_id == batch.batch_id



def test_manual_transport_uses_normal_preparation_and_dedup(saved):
    from Application.model.BPRMF import TrainConfig
    root, source, _ = saved
    action = {"ids": {"mindboxId": 10}, "customer": {"ids": {"mindboxId": 1}},
              "dateTimeUtc": "2026-08-01T12:00:00Z", "creationDateTimeUtc": "2026-08-01T12:00:00Z",
              "actionTemplate": {"ids": {"systemName": "CustomView"}},
              "products": [{"ids": {"offline1C": "123456_variant"}}]}
    actions = write(root, "actions", [action] * 9 + [{"actionTemplate": {"ids": {"systemName": "UstanovkaSpiskaProduktovV"}},
                                                     "products": [{"ids": {"website": "ignored"}}]}])
    order = {"ids": {"mindboxId": 20}, "customer": {"ids": {"mindboxId": 1}},
             "firstAction": {"dateTimeUtc": "2026-01-01T12:00:00Z", "channel": {"ids": {"externalId": "channel"}, "name": "test"}},
             "lines": [{"id": 1, "number": 1, "quantity": 2, "basePricePerItem": 1, "priceOfLine": 2,
                        "status": {"ids": {"externalId": "CP"}}, "product": {"ids": {"offline1C": "654321_variant"}}}]}
    orders = write(root, "orders", [order, order])
    selection = replace(DEFAULT_SELECTION, view_action_system_names=("CustomView",))
    batch = manual.import_interactions(actions, orders, raw_root=root, window=source.window, selection=selection)
    manifest = root / "training_batches" / batch.batch_id / "manifest.json"
    loaded = daily.load_chunked_training_batch(manifest, raw_root=root, require_complete=True)
    catalog = root / "Номенклатура.csv"
    catalog.write_text("КодНоменклатуры\n123456\n654321\n", encoding="utf-8-sig")
    result = daily.prepare_training_data_from_chunked_batch(loaded, raw_root=root, catalog_path=catalog, train_config=TrainConfig())
    assert result.complete
    assert result.diagnostics.view_interactions == 9
    assert result.diagnostics.purchase_interactions == 1
    assert result.diagnostics.orders_duplicate_identical == 1
    assert result.diagnostics.unmapped_actions == 1
    assert result.prepared_data.mappings.idx2user == ["2"]
