"""Synthetic security facts and failure injection; never mutate real ACLs."""

import sqlite3
from pathlib import Path

import pytest

from Application import statistics_index as index
from Application import statistics_storage_safety as safety
from tests.test_statistics_index import build, dataset as dataset_fixture, query


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    return dataset_fixture.__wrapped__(tmp_path, monkeypatch)


def fact():
    return dict(owner_trusted=True, protected=True, unsupported=False,
        rules=[dict(trusted=True, current=True, mask=0x1F01FF, allow=True, inherit_only=False)])


@pytest.mark.parametrize("mask", [1, 2, 4, 0x40, 0x10000, 0x40000, 0x80000, 0x10000000])
def test_private_root_rejects_every_foreign_allow_including_generic_all(mask):
    value = fact()
    value["rules"].append(dict(trusted=False, current=False, mask=mask, allow=True, inherit_only=True))
    with pytest.raises(index.StatisticsIndexError, match="UNSAFE_STORAGE_ACL"):
        safety.check_acl_facts(value, private=True, root=True)


@pytest.mark.parametrize("mask", [0x40, 0x10000, 0x40000, 0x80000, 0x110, 0x10000000, 0x40000000])
def test_mutable_ancestor_fails_even_when_child_is_private(mask):
    value = fact()
    value["rules"].append(dict(trusted=False, current=False, mask=mask, allow=True, inherit_only=False))
    with pytest.raises(index.StatisticsIndexError, match="UNSAFE_STORAGE_ACL"):
        safety.check_acl_facts(value, private=False)


def test_foreign_read_only_ancestor_allowed_but_not_private_file():
    value = fact()
    value["rules"].append(dict(trusted=False, current=False, mask=0x1200A9, allow=True, inherit_only=False))
    safety.check_acl_facts(value, private=False)
    with pytest.raises(index.StatisticsIndexError):
        safety.check_acl_facts(value, private=True)


@pytest.mark.parametrize("change", ["owner", "inherited", "unsupported", "deny", "readonly_owner"])
def test_inadequate_root_acl_is_rejected(change):
    value = fact()
    if change == "owner": value["owner_trusted"] = False
    elif change == "inherited": value["protected"] = False
    elif change == "unsupported": value["unsupported"] = True
    elif change == "deny": value["rules"][0]["allow"] = False
    else: value["rules"][0]["mask"] = 0x120089
    with pytest.raises(index.StatisticsIndexError):
        safety.check_acl_facts(value, private=True, root=True)


def test_generic_full_control_maps_to_current_user_permissions():
    value = fact()
    value["rules"][0]["mask"] = 0x10000000
    safety.check_acl_facts(value, private=True, root=True)


def test_nonexistent_root_is_not_created_or_acl_repaired(tmp_path):
    path = tmp_path/"missing"
    with pytest.raises(index.StatisticsIndexError, match="PRIVATE_INDEX_ROOT_REQUIRED"):
        safety.validate_storage_root(path)
    assert not path.exists()


def test_known_synchronized_directory_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('OneDrive',str(tmp_path))
    with pytest.raises(index.StatisticsIndexError, match='SYNCHRONIZED_ROOT_UNSUPPORTED'):
        safety.validate_storage_root(tmp_path/'index')


def test_unavailable_or_network_volume_fails_without_creating_files(dataset, monkeypatch):
    def unavailable(_): raise index.StatisticsIndexError("LOCAL_FIXED_VOLUME_REQUIRED")
    monkeypatch.setattr(index, "validate_storage_root", safety.validate_storage_root)
    monkeypatch.setattr(safety, "local_volume", unavailable)
    with pytest.raises(index.StatisticsIndexError, match="LOCAL_FIXED_VOLUME_REQUIRED"):
        build(dataset)
    assert not list((dataset[0]/"analytics").iterdir())


def test_runtime_disk_exhaustion_keeps_old_pointer_and_removes_only_staging(dataset, monkeypatch):
    old = build(dataset)
    directory = dataset[0]/"analytics"
    pointer = (directory/"current.json").read_bytes()
    original = index.shutil.disk_usage
    exhausted = False

    def progress(_):
        nonlocal exhausted
        exhausted = True

    def disk(path):
        return type("Usage", (), {"free":20*index.GIB-1})() if exhausted else original(path)

    monkeypatch.setattr(index.shutil, "disk_usage", disk)
    with pytest.raises(index.StatisticsIndexError, match="INSUFFICIENT_DISK"):
        build(dataset, progress=progress)
    assert (directory/"current.json").read_bytes() == pointer
    assert {p.name for p in directory.iterdir() if p.is_dir()} == {old["generation"]}


def test_atomic_publication_failure_keeps_old_generation_and_cleans_staging(dataset, monkeypatch):
    old = build(dataset)
    directory = dataset[0]/"analytics"
    pointer = (directory/"current.json").read_bytes()
    original = index.atomic_json

    def fail(path, value):
        if path.name == "current.json": raise PermissionError("synthetic")
        return original(path, value)

    monkeypatch.setattr(index, "atomic_json", fail)
    with pytest.raises(PermissionError): build(dataset)
    assert (directory/"current.json").read_bytes() == pointer
    assert {p.name for p in directory.iterdir() if p.is_dir()} == {old["generation"]}
    query(dataset)


def test_acl_is_rechecked_before_publication(dataset, monkeypatch):
    old = build(dataset)
    calls = 0

    def validate(path, **kw):
        nonlocal calls
        calls += 1
        if any(p.name == "manifest.json" for p in kw.get("children", ())):
            raise index.StatisticsIndexError("UNSAFE_STORAGE_ACL")
        return index.path_identity(path)

    monkeypatch.setattr(index, "validate_storage_root", validate)
    with pytest.raises(index.StatisticsIndexError, match="UNSAFE_STORAGE_ACL"): build(dataset)
    assert calls == 3
    assert index.read_pointer(dataset[0]/"analytics")["generation"] == old["generation"]
    assert len([p for p in (dataset[0]/"analytics").iterdir() if p.is_dir()]) == 1


def test_damaged_staging_fails_integrity_and_does_not_publish(dataset, monkeypatch):
    old = build(dataset)
    original = index.Projector

    class Broken(original):
        def profiles(self, *args):
            super().profiles(*args)
            self.db.execute("PRAGMA ignore_check_constraints=ON")
            self.db.execute("INSERT INTO action_products VALUES(999999,0,'offline1C','synthetic',NULL)")

    # Inject a broken FK through a separately controlled connection setting.
    original_connect = index.sqlite3.connect
    class Connection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA foreign_keys=ON": sql = "PRAGMA foreign_keys=OFF"
            return super().execute(sql, *args, **kwargs)
    def connect(*args, **kwargs):
        kwargs['factory'] = Connection
        return original_connect(*args, **kwargs)
    monkeypatch.setattr(index.sqlite3, "connect", connect)
    monkeypatch.setattr(index, "Projector", Broken)
    with pytest.raises(index.StatisticsIndexError, match="INTEGRITY_FAILED"): build(dataset)
    assert index.read_pointer(dataset[0]/"analytics")["generation"] == old["generation"]


def test_substituted_staging_is_not_recursively_deleted(tmp_path):
    root=tmp_path/"root";root.mkdir()
    staging=root/"stage";staging.mkdir()
    root_id, folder_id=safety.path_identity(root), safety.path_identity(staging)
    staging.rename(root/"retained-original")
    staging.mkdir()
    sentinel=staging/"not-owned";sentinel.write_text("synthetic")
    with pytest.raises(index.StatisticsIndexError, match="STAGING_PATH_CHANGED"):
        safety.check_owned_staging(root, staging, root_id, folder_id)
    assert sentinel.read_text()=="synthetic"


def test_reparse_attribute_rejected_before_resolve(tmp_path, monkeypatch):
    original=type(tmp_path).lstat
    def altered(path):
        if path==tmp_path:
            return type('Info',(),dict(st_mode=0o40700,st_file_attributes=0x400))()
        return original(path)
    monkeypatch.setattr(type(tmp_path),'lstat',altered)
    with pytest.raises(index.StatisticsIndexError, match='UNSAFE_REPARSE_PATH'):
        safety.reject_reparse(tmp_path/'child')


def test_dot_dot_cannot_hide_a_reparse_component_before_normalization(tmp_path):
    with pytest.raises(index.StatisticsIndexError, match='UNSAFE_RELATIVE_PATH'):
        safety.reject_reparse(tmp_path/'possibly-a-junction'/'..'/'index')


@pytest.mark.parametrize("component", ["main", "demographics"])
def test_actual_integrity_check_rejects_broken_staging_constraints(dataset, monkeypatch, component):
    old=build(dataset)
    original=index.Projector

    class Broken(original):
        def profiles(self, root, demographics, check):
            super().profiles(root, demographics, check)
            db=self.db if component=='main' else demographics
            db.execute('CREATE TABLE broken(value INTEGER CHECK(value>0))')
            db.execute('PRAGMA ignore_check_constraints=ON')
            db.execute('INSERT INTO broken VALUES(-1)')
            db.execute('PRAGMA ignore_check_constraints=OFF')

    monkeypatch.setattr(index,'Projector',Broken)
    with pytest.raises(index.StatisticsIndexError, match='INTEGRITY_FAILED'):build(dataset)
    assert index.read_pointer(dataset[0]/'analytics')['generation']==old['generation']


def test_real_atomic_json_replace_failure_removes_metadata_temp_and_preserves_old_files(dataset, monkeypatch):
    old=build(dataset)
    directory=dataset[0]/'analytics'
    folder=directory/old['generation']
    hashes={p.name:index.component_digest(p) for p in folder.iterdir()}
    original=index.os.replace

    def fail(source,target):
        if Path(target).name=='current.json':raise OSError('synthetic replace failure')
        return original(source,target)

    monkeypatch.setattr(index.os,'replace',fail)
    with pytest.raises(OSError):build(dataset)
    assert not list(directory.glob('.metadata-*'))
    assert index.read_pointer(directory)['generation']==old['generation']
    assert {p.name:index.component_digest(p) for p in folder.iterdir()}==hashes
