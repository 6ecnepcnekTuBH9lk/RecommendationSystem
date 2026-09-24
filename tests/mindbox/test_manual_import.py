from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import shutil

import pytest

from Application.mindbox import daily_training_batch as daily, manual_import as manual, canonical_storage as store
from Application.mindbox.customer_profile_snapshot import load_customer_profile_snapshot, load_customer_contact_index
from Application.mindbox.raw_reader import iter_export, EXPORT_ROOTS
from Application.mindbox.selection import DEFAULT_SELECTION
from Application.mindbox.training_batch import TrainingBatchExport, TrainingBatchWindow


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("No real network"))


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
    staged = tmp_path / "staged_merges"
    shutil.copytree(tmp_path / components[0].export.relative_directory, staged)
    with store.storage_lock(tmp_path):
        store.publish(tmp_path, "customer_merges", start, window.interaction_until, staged,
                      source_kind="API", export_id="synthetic", operation="merges")
    return tmp_path, batch, path


def write(root, name, records=None, text=None):
    path = root / (name + ".json")
    path.write_text(text if text is not None else json.dumps({EXPORT_ROOTS[name]: records or []}), encoding="utf-8")
    return path


def test_manual_batch_atomic_metadata_selection_and_common_readers(saved, monkeypatch):
    root, source, legacy = saved
    records = [{"actionTemplate": {"ids": {"systemName": "Unmapped"}}}]
    actions = write(root, "actions", records)
    orders = write(root, "orders")
    before = actions.read_bytes(), orders.read_bytes()
    selection = replace(DEFAULT_SELECTION, view_action_system_names=("CustomView",))
    original = manual.os.replace
    def publish(src, dst):
        if Path(src).is_dir():
            assert not Path(dst).exists()
            assert (Path(src) / "actions/actions_part_001.json").read_bytes() == before[0]
            assert (Path(src) / "orders/orders_part_001.json").read_bytes() == before[1]
        return original(src, dst)
    monkeypatch.setattr(manual.os, "replace", publish)
    manifest = manual.import_interactions(actions, orders, raw_root=root, window=source.window, selection=selection)
    batch = daily.load_chunked_training_batch(manifest, raw_root=root, require_complete=True)
    assert manifest == root / "canonical/training.json"
    assert batch.selection == selection and batch.source_kind == "CANONICAL"
    pair = store.catalog(root)["manual_interactions"]
    assert pair["actions"]["directory"].rsplit("/", 1)[0] == pair["orders"]["directory"].rsplit("/", 1)[0]
    for component in batch.components[1:]:
        assert component.export.source_kind == "MANUAL" and component.export.export_id is None
        assert component.operation == "MANUAL"
    assert list(iter_export("actions", input_dir=root / batch.components[1].export.relative_directory)) == records
    assert (actions.read_bytes(), orders.read_bytes()) == before
    assert list((root / "training_batches").glob("*/manifest.json")) == [legacy]
    assert manual.select_merges_source(root).batch_id == source.batch_id


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
    manifest = manual.import_interactions(actions, orders, raw_root=root, window=source.window, selection=selection)
    loaded = daily.load_chunked_training_batch(manifest, raw_root=root, require_complete=True)
    catalog = root / "nomenclature.csv"
    catalog.write_text("КодНоменклатуры\n123456\n654321\n", encoding="utf-8-sig")
    result = daily.prepare_training_data_from_chunked_batch(loaded, raw_root=root, catalog_path=catalog, train_config=TrainConfig())
    assert result.complete
    assert result.diagnostics.view_interactions == 9
    assert result.diagnostics.purchase_interactions == 1
    assert result.diagnostics.orders_duplicate_identical == 1
    assert result.diagnostics.unmapped_actions == 1
    assert result.prepared_data.mappings.idx2user == ["2"]


def utc(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def publish_merges(root, since="2025-01-01", until="2027-01-01"):
    directory = root / "staged"
    directory.mkdir()
    (directory / "customer_merges_part_001.json").write_text('{"customerMerges": []}')
    with store.storage_lock(root):
        store.publish(root, "customer_merges", utc(since), utc(until), directory,
                      source_kind="API", export_id="test", operation="merges")


def import_pair(root, since="2026-01-01", until="2026-07-01", **kwargs):
    return manual.import_interactions(write(root, "actions"), write(root, "orders"), raw_root=root,
        window=TrainingBatchWindow(utc(since), utc(until), utc(since)), **kwargs)


@pytest.mark.parametrize("merges", [None, ("2026-02-01", "2027-01-01"), ("2025-01-01", "2026-06-01")])
def test_canonical_merges_required_for_entire_manual_interval(tmp_path, merges):
    if merges:
        publish_merges(tmp_path, *merges)
    with pytest.raises(manual.ManualImportError, match="API Mindbox") as error:
        import_pair(tmp_path)
    assert str(tmp_path) not in str(error.value)
    assert not store.catalog(tmp_path)["manual_interactions"]


def test_legacy_merges_alone_cannot_authorize_new_manual_import(saved):
    root, batch, _ = saved
    (root / "canonical/catalog.json").unlink()
    with pytest.raises(manual.ManualImportError, match="API Mindbox"):
        manual.import_interactions(write(root, "actions"), write(root, "orders"), raw_root=root, window=batch.window)


def test_long_range_replacement_and_source_safety(tmp_path):
    publish_merges(tmp_path)
    manifest = import_pair(tmp_path)
    old = store.catalog(tmp_path)["manual_interactions"]
    old_object = (tmp_path / old["actions"]["directory"]).parent
    assert old_object.exists()
    batch = daily.load_chunked_training_batch(manifest, raw_root=tmp_path)
    assert len(batch.components) == 3
    assert batch.diagnostics["days_total"] == batch.diagnostics["days_ready"] == 181
    assert batch.window.interaction_since == utc("2026-01-01")
    assert not (tmp_path / "training_batches").exists()
    original = {name: (tmp_path / (name + ".json")).read_bytes() for name in ("actions", "orders")}
    import_pair(tmp_path, "2026-02-01", "2026-08-01")
    assert not old_object.exists()
    assert len(list((tmp_path / "canonical/objects").iterdir())) == 2
    assert all((tmp_path / (name + ".json")).read_bytes() == value for name, value in original.items())
    assert store.summary(tmp_path)["orders"]["since"] == utc("2026-02-01").isoformat()


@pytest.mark.parametrize("failure", ["orders_json", "orders_record", "cancel_copy", "cancel_validation", "cancel_commit", "catalog_commit"])
def test_failed_replacement_keeps_entire_old_pair(tmp_path, monkeypatch, failure):
    publish_merges(tmp_path)
    import_pair(tmp_path)
    before = (tmp_path / "canonical/catalog.json").read_bytes()
    data = store.catalog(tmp_path)
    objects = set((tmp_path / "canonical/objects").iterdir())
    paths = {name: write(tmp_path, name) for name in ("actions", "orders")}
    if failure == "orders_json":
        paths["orders"].write_text('{"orders": [')
    if failure == "orders_record":
        paths["orders"].write_text('{"orders": [{}]}')
    source_bytes = {name: path.read_bytes() for name, path in paths.items()}
    cancelled = False
    def progress(message):
        nonlocal cancelled
        if ((failure == "cancel_copy" and message.startswith("Копирование orders"))
                or (failure == "cancel_validation" and message.startswith("Проверка orders завершена"))):
            cancelled = True
    original = store.atomic_json
    def commit(path, value):
        nonlocal cancelled
        if failure == "catalog_commit" and Path(path).name == "catalog.json":
            raise OSError("synthetic")
        original(path, value)
        if failure == "cancel_commit" and Path(path).name == "training.json":
            cancelled = True
    monkeypatch.setattr(store, "atomic_json", commit)
    with pytest.raises((manual.ManualImportError, InterruptedError, OSError)):
        manual.import_interactions(**paths, raw_root=tmp_path,
            window=TrainingBatchWindow(utc("2026-01-01"), utc("2026-08-01"), utc("2026-01-01")),
            cancelled=lambda: cancelled, progress=progress)
    assert (tmp_path / "canonical/catalog.json").read_bytes() == before
    assert store.catalog(tmp_path) == data
    assert set((tmp_path / "canonical/objects").iterdir()) == objects
    assert not list((tmp_path / "canonical").glob(".manual-interactions-*"))
    assert all(path.read_bytes() == source_bytes[name] for name, path in paths.items())


def test_manual_cleanup_confined_and_keeps_live_pair(tmp_path):
    publish_merges(tmp_path)
    import_pair(tmp_path)
    abandoned = tmp_path / "canonical/.manual-interactions-interrupted"
    abandoned.mkdir()
    (abandoned / "partial.json").write_text("{")
    unrelated = tmp_path / ".manual-interactions-user"
    unrelated.mkdir()
    legacy = tmp_path / "training_batches/legacy"
    legacy.mkdir(parents=True)
    with store.storage_lock(tmp_path):
        store.collect_unreferenced(tmp_path)
    assert not abandoned.exists()
    assert unrelated.exists() and legacy.exists()
    assert len(store.current_batch(tmp_path).components) == 3


def test_canonical_v1_read_does_not_rewrite_until_mutation(tmp_path):
    publish_merges(tmp_path)
    data = store.catalog(tmp_path)
    data.pop("manual_interactions")
    data["schema_version"] = 1
    path = tmp_path / "canonical/catalog.json"
    store.atomic_json(path, data)
    before = path.read_bytes()
    assert store.catalog(tmp_path)["manual_interactions"] is None
    assert path.read_bytes() == before
    import_pair(tmp_path)
    assert store.catalog(tmp_path)["schema_version"] == 2


def test_legacy_manual_v4_remains_readable_and_preparable(saved, monkeypatch):
    root, source, _ = saved
    batch_id = "b" * 32
    directory = root / "training_batches" / batch_id
    components = [source.components[0]]
    for name in ("actions", "orders"):
        part_dir = directory / name
        part_dir.mkdir(parents=True)
        (part_dir / f"{name}_part_001.json").write_text(json.dumps({EXPORT_ROOTS[name]: []}))
        export = TrainingBatchExport(name, None, "MANUAL", part_dir.relative_to(root).as_posix(), 1, "MANUAL")
        components.append(daily.BatchComponent(name, source.window.interaction_since, source.window.interaction_until,
                                               "MANUAL", "READY", export))
    batch = replace(source, batch_id=batch_id, components=tuple(components), source_kind="MANUAL",
                    merge_source_training_batch_id=source.batch_id)
    daily._atomic_write(directory / "manifest.json", batch)
    loaded = daily.load_chunked_training_batch(directory / "manifest.json", raw_root=root, require_complete=True)
    assert loaded == batch
    from Application.model import mindbox_training_preparation as preparation
    received = {}
    monkeypatch.setattr(preparation, "_prepare_training_data_from_mindbox_sources", lambda **kwargs: received.update(kwargs))
    daily.prepare_training_data_from_chunked_batch(loaded, raw_root=root, catalog_path="unused", train_config=None)
    assert received["actions_export_dirs"] == (directory / "actions",)


def test_manual_cli_uses_canonical_entry_without_merge_since(tmp_path, capsys):
    from scripts.mindbox_manual_import import main
    publish_merges(tmp_path)
    assert main(["interactions", "--raw-root", str(tmp_path), "--actions", str(write(tmp_path, "actions")),
                 "--orders", str(write(tmp_path, "orders")), "--since", "2026-01-01", "--until", "2026-07-01"]) == 0
    assert f"Manifest: {tmp_path / 'canonical/training.json'}" in capsys.readouterr().out


@pytest.mark.parametrize("failure", [KeyboardInterrupt, OSError])
def test_interrupt_immediately_after_catalog_switch_keeps_published_raw(tmp_path, monkeypatch, failure):
    publish_merges(tmp_path)
    import_pair(tmp_path)
    previous = store.catalog(tmp_path)["manual_interactions"]
    original = store.atomic_json
    def interrupt(path, data):
        original(path, data)
        if Path(path).name == "catalog.json":
            raise failure()
    monkeypatch.setattr(store, "atomic_json", interrupt)
    with pytest.raises(failure):
        import_pair(tmp_path, until="2026-08-01")
    current = store.catalog(tmp_path)["manual_interactions"]
    assert current != previous
    assert store.current_batch(tmp_path).window.interaction_until == utc("2026-08-01")
    with store.storage_lock(tmp_path):
        store.collect_unreferenced(tmp_path)
    assert len(list((tmp_path / "canonical/objects").iterdir())) == 2


def test_manual_validation_consumes_records_once_incrementally(tmp_path, monkeypatch):
    from Application.mindbox.adapters import actions as action_adapter
    publish_merges(tmp_path)
    processed = 0
    calls = []
    original = action_adapter.adapt_action_system_name
    def adapt(raw):
        nonlocal processed
        processed += 1
        return original(raw)
    def records(name, **kwargs):
        calls.append(name)
        if name == "actions":
            for index in range(1000):
                assert processed == index
                yield {"actionTemplate": {"ids": {"systemName": "Unmapped"}}}
    monkeypatch.setattr(action_adapter, "adapt_action_system_name", adapt)
    monkeypatch.setattr(store, "iter_export", records)
    import_pair(tmp_path)
    assert processed == 1000 and calls == ["actions", "orders"]


def test_manual_import_holds_training_storage_lock(tmp_path):
    publish_merges(tmp_path)
    import_pair(tmp_path)
    before = (tmp_path / "canonical/catalog.json").read_bytes()
    with store.storage_lock(tmp_path), pytest.raises(ValueError):
        import_pair(tmp_path, until="2026-08-01")
    assert (tmp_path / "canonical/catalog.json").read_bytes() == before


def test_incomplete_canonical_merges_blocks_manual_publication(tmp_path):
    publish_merges(tmp_path)
    import_pair(tmp_path)
    data = store.catalog(tmp_path)
    data["customer_merges"]["parts"] = 2  # A missing final part must not look like complete history.
    store.atomic_json(tmp_path / "canonical/catalog.json", data)
    before = (tmp_path / "canonical/catalog.json").read_bytes()
    with pytest.raises(manual.ManualImportError, match="Обновите их через API Mindbox"):
        import_pair(tmp_path, until="2026-08-01")
    assert (tmp_path / "canonical/catalog.json").read_bytes() == before


def multipart_sources(root, name, numbers):
    paths = []
    for number in numbers:
        if name == "actions":
            record = {"actionTemplate": {"ids": {"systemName": "Unmapped"}}, "part": number}
        else:
            record = {"ids": {"mindboxId": number}, "customer": {"ids": {"mindboxId": 1}},
                      "firstAction": {"dateTimeUtc": "2026-01-01T12:00:00Z",
                                      "channel": {"ids": {"externalId": "test"}, "name": "test"}},
                      "lines": []}
        path = root / f"{name}_part{number}.json"
        path.write_text(json.dumps({EXPORT_ROOTS[name]: [record]}), encoding="utf-8")
        paths.append(path)
    return paths


def test_multipart_unequal_counts_natural_order_and_readers(saved):
    root, batch, _ = saved
    paths = {"actions": multipart_sources(root, "actions", [10, 2, 1, 3]),
             "orders": multipart_sources(root, "orders", [10, 2])}
    before = {path: path.read_bytes() for sources in paths.values() for path in sources}
    messages = []
    manifest = manual.import_interactions(**paths, raw_root=root, window=batch.window, progress=messages.append)
    pair = store.catalog(root)["manual_interactions"]
    loaded = daily.load_chunked_training_batch(manifest, raw_root=root, require_complete=True)
    assert [component.export.parts_count for component in loaded.components[1:]] == [4, 2]
    for name, numbers in (("actions", [1, 2, 3, 10]), ("orders", [2, 10])):
        assert pair[name]["parts"] == len(numbers)
        directory = root / pair[name]["directory"]
        assert sorted(path.name for path in directory.iterdir()) == [
            f"{name}_part_{number:03d}.json" for number in range(1, len(numbers) + 1)]
        records = list(iter_export(name, input_dir=directory))
        assert [raw["part"] if name == "actions" else raw["ids"]["mindboxId"] for raw in records] == numbers
    assert "Копирование actions: файл 2 из 4" in messages
    assert "Проверка orders ..." in messages
    assert all(str(root) not in text for text in messages)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("source", ["actions", "orders"])
@pytest.mark.parametrize("failure", ["invalid", "cancel"])
def test_multipart_later_failure_preserves_pair_and_cleans_staging(saved, source, failure):
    root, batch, _ = saved
    manual.import_interactions(write(root, "actions"), write(root, "orders"), raw_root=root, window=batch.window)
    before = (root / "canonical/catalog.json").read_bytes()
    objects = set((root / "canonical/objects").iterdir())
    paths = {name: multipart_sources(root, name, [1, 2, 3]) for name in ("actions", "orders")}
    if failure == "invalid":
        paths[source][1].write_text('{"broken": [', encoding="utf-8")
    cancelled = False
    def progress(message):
        nonlocal cancelled
        if failure == "cancel" and message == f"Копирование {source}: файл 2 из 3":
            cancelled = True
    with pytest.raises(manual.ManualImportError if failure == "invalid" else InterruptedError):
        manual.import_interactions(**paths, raw_root=root, window=batch.window,
                                   cancelled=lambda: cancelled, progress=progress)
    assert (root / "canonical/catalog.json").read_bytes() == before
    assert set((root / "canonical/objects").iterdir()) == objects
    assert not list((root / "canonical").glob(".manual-interactions-*"))
    assert len(store.current_batch(root).components) == 3


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "missing", "directory"])
def test_manual_source_validation(saved, invalid):
    root, batch, _ = saved
    actions, orders = write(root, "actions"), write(root, "orders")
    values = {"empty": [], "duplicate": [actions, str(actions.parent / "." / actions.name)],
              "missing": [root / "missing.json"], "directory": [root]}
    with pytest.raises(manual.ManualImportError) as error:
        manual.import_interactions(values[invalid], orders, raw_root=root, window=batch.window)
    assert str(root) not in str(error.value)
    if invalid == "duplicate":
        assert "несколько раз" in str(error.value)
    assert store.catalog(root)["manual_interactions"] is None


def test_manual_cli_repeated_source_options(saved, capsys):
    from scripts.mindbox_manual_import import main
    root, _, _ = saved
    paths = {"actions": multipart_sources(root, "actions", [10, 2, 1, 3]),
             "orders": multipart_sources(root, "orders", [2, 1])}
    args = ["interactions", "--raw-root", str(root), "--since", "2026-08-01", "--until", "2026-08-02"]
    for name, sources in paths.items():
        for path in sources:
            args.extend(["--" + name, str(path)])
    assert main(args) == 0
    pair = store.catalog(root)["manual_interactions"]
    assert pair["actions"]["parts"] == 4 and pair["orders"]["parts"] == 2
    assert "Копирование actions: файл 4 из 4" in capsys.readouterr().out


@pytest.mark.parametrize("cancel", [False, True])
def test_multipart_bounded_copy_and_cancel_inside_later_part(saved, monkeypatch, cancel):
    root, batch, _ = saved
    orders = write(root, "orders")
    manual.import_interactions(write(root, "actions"), orders, raw_root=root, window=batch.window)
    before = (root / "canonical/catalog.json").read_bytes()
    paths = multipart_sources(root, "actions", [1, 2])
    record = {"actionTemplate": {"ids": {"systemName": "Unmapped"}}, "padding": "x" * (2 * 1024 * 1024)}
    paths[1].write_text(json.dumps({"customerActions": [record]}), encoding="utf-8")
    original = Path.open
    reads = []
    cancelled = False
    class BoundedInput:
        def __enter__(self):
            self.stream = original(paths[1], "rb")
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            nonlocal cancelled
            assert 0 < size <= 1024 * 1024
            chunk = self.stream.read(size)
            reads.append(len(chunk))
            cancelled = cancel
            return chunk

    def open_file(path, *args, **kwargs):
        if path == paths[1] and args == ("rb",):
            return BoundedInput()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", open_file)
    def run_import():
        manual.import_interactions(paths, orders, raw_root=root, window=batch.window, cancelled=lambda: cancelled)
    if cancel:
        with pytest.raises(InterruptedError):
            run_import()
        assert (root / "canonical/catalog.json").read_bytes() == before
        assert reads == [1024 * 1024]
    else:
        run_import()
        assert len(reads) == 4 and reads[-1] == 0
    assert not list((root / "canonical").glob(".manual-interactions-*"))
