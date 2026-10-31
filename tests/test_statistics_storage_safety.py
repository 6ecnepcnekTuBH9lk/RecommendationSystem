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


@pytest.fixture
def helper_system(monkeypatch, tmp_path):
    """Mock the OS directory API and manifests, never install PowerShell modules."""
    import ctypes

    system = tmp_path / "Windows" / "System32"
    modules = system / "WindowsPowerShell" / "v1.0" / "Modules"

    def get_system_directory(buffer, size):
        buffer.value = str(system)
        return len(str(system))

    class Kernel:
        GetSystemDirectoryW = staticmethod(get_system_directory)

    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **kw: Kernel(), raising=False)
    manifests = {modules / name / (name + ".psd1")
                 for name in ("Microsoft.PowerShell.Security", "Microsoft.PowerShell.Utility")}
    monkeypatch.setattr(Path, "is_file", lambda path: path in manifests)
    return modules, get_system_directory


@pytest.mark.parametrize("module_keys", [
    {}, {"PSModulePath": "UNTRUSTED_MODULES"},
    {"PSMODULEPATH": "UNTRUSTED_MODULES"},
    {"psmodulepath": "UNTRUSTED_MODULES", "PSModulePath": "OTHER_UNTRUSTED_MODULES"},
])
def test_acl_helper_isolates_only_child_module_environment(helper_system, monkeypatch, module_keys):
    import base64
    import json
    from types import SimpleNamespace

    modules, _ = helper_system
    parent = {"PATH": "PRESERVED_PATH", "TEMP": "PRESERVED_TEMP", "TMP": "PRESERVED_TMP",
              "SYSTEMROOT": "UNTRUSTED_ROOT", "UNCHANGED": "value", **module_keys}
    before = parent.copy()
    monkeypatch.setattr(safety.os, "environ", parent)
    seen = {}

    def run(args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        assert kwargs["env"] is not parent
        assert kwargs["env"] == {**{k: v for k, v in before.items()
                                    if k.casefold() != "psmodulepath"}, "PSModulePath": str(modules)}
        kwargs["env"]["CHILD_ONLY"] = "does not propagate"
        return SimpleNamespace(returncode=0, stdout=json.dumps([fact()]).encode(), stderr=b"")

    monkeypatch.setattr(safety.subprocess, "run", run)
    path = Path("D:/__MODULES__/Юникод/Каталог'с пробелом")
    assert safety.inspect_acls([path]) == [fact()]
    assert parent == before
    assert seen["args"][:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand"]
    script = base64.b64decode(seen["args"][4]).decode("utf-16le")
    pin = "$env:PSModulePath='" + str(modules).replace("'", "''") + "'"
    assert pin in script and script.index(pin) < script.index("Get-Acl")
    assert str(path).replace("'", "''") in script
    assert "UNTRUSTED_MODULES" not in script
    assert "Set-Acl" not in script and "ExecutionPolicy" not in script
    assert seen["kwargs"]["capture_output"] is True
    assert seen["kwargs"]["timeout"] == 30


@pytest.mark.parametrize("module", ["Microsoft.PowerShell.Security", "Microsoft.PowerShell.Utility"])
def test_acl_helper_missing_builtin_manifest_fails_without_fallback(helper_system, monkeypatch, module):
    modules, _ = helper_system
    monkeypatch.setattr(Path, "is_file", lambda path: path.parent.name != module)
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **kw: pytest.fail("No module fallback"))
    with pytest.raises(safety.StatisticsIndexError, match="^ACL_INSPECTION_FAILED$"):
        safety.inspect_acls([modules])


@pytest.mark.parametrize("length", [0, 32768])
def test_acl_helper_invalid_os_system_directory_fails_closed(helper_system, monkeypatch, length):
    _, get_system_directory = helper_system

    def invalid(buffer, size):
        return length

    import ctypes
    from types import SimpleNamespace
    monkeypatch.setattr(ctypes, "WinDLL",
                        lambda *a, **kw: SimpleNamespace(GetSystemDirectoryW=invalid))
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **kw: pytest.fail("No environment fallback"))
    with pytest.raises(safety.StatisticsIndexError, match="^ACL_INSPECTION_FAILED$"):
        safety.inspect_acls([Path("D:/")])


@pytest.mark.parametrize("failure", ["missing_executable", "denied", "timeout", "os_api"])
def test_acl_helper_launch_errors_are_safe(helper_system, monkeypatch, capsys, failure):
    import subprocess

    def run(*args, **kwargs):
        if failure == "missing_executable":
            raise FileNotFoundError("SENSITIVE_DIAGNOSTIC")
        if failure == "denied":
            raise PermissionError("SENSITIVE_DIAGNOSTIC")
        if failure == "timeout":
            raise subprocess.TimeoutExpired("SENSITIVE_DIAGNOSTIC", 30, stderr=b"SENSITIVE_STDERR")
        raise OSError("SENSITIVE_DIAGNOSTIC")

    monkeypatch.setattr(safety.subprocess, "run", run)
    if failure == "os_api":
        import ctypes
        monkeypatch.setattr(ctypes, "WinDLL", run)
    with pytest.raises(safety.StatisticsIndexError) as caught:
        safety.inspect_acls([Path("D:/")])
    assert str(caught.value) == "ACL_INSPECTION_FAILED"
    assert caught.value.__suppress_context__
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("reason", ["Get-Acl failure", "module load failure"])
def test_acl_helper_nonzero_exit_never_admits_or_prints_stderr(helper_system, monkeypatch, capsys, reason):
    import json
    from types import SimpleNamespace

    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout=json.dumps([fact()]).encode(),
                                        stderr=(reason + " SENSITIVE_STDERR").encode()))
    with pytest.raises(safety.StatisticsIndexError, match="^ACL_INSPECTION_FAILED$"):
        safety.inspect_acls([Path("D:/")])
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("payload", [
    b"", b"SENSITIVE_NOT_JSON", b"\xff", b"{}", b"null", b"[]", b"[{},{}]",
    b"[null]", b"[{}]",
    b'[{"owner_trusted":true,"protected":true,"unsupported":false,"rules":{}}]',
    b'[{"owner_trusted":"false","protected":true,"unsupported":false,"rules":[]}]',
    b'[{"owner_trusted":true,"protected":true,"unsupported":false,"rules":[{}]}]',
])
def test_acl_helper_invalid_json_or_fact_shape_fails_closed(helper_system, monkeypatch, capsys, payload):
    from types import SimpleNamespace
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout=payload, stderr=b"SENSITIVE_STDERR"))
    with pytest.raises(safety.StatisticsIndexError, match="^ACL_INSPECTION_FAILED$"):
        safety.inspect_acls([Path("D:/")])
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("key,value", [
    ("trusted", "false"), ("current", 1), ("allow", None), ("inherit_only", 0),
    ("mask", True), ("mask", "2032127"), ("mask", 2**32), ("mask", -(2**31)-1),
])
def test_acl_helper_invalid_ace_shape_is_not_trusted(helper_system, monkeypatch, key, value):
    import json
    from types import SimpleNamespace
    invalid = fact()
    invalid["rules"][0][key] = value
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout=json.dumps([invalid]).encode(), stderr=b""))
    with pytest.raises(safety.StatisticsIndexError, match="^ACL_INSPECTION_FAILED$"):
        safety.inspect_acls([Path("D:/")])


@pytest.mark.parametrize("mask", [-2147483648, 0x80000000, 0xFFFFFFFF])
def test_acl_helper_preserves_signed_and_unsigned_masks(helper_system, monkeypatch, mask):
    import json
    from types import SimpleNamespace
    value = fact()
    value["rules"][0]["mask"] = mask
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout=json.dumps([value]).encode(), stderr=b""))
    assert safety.inspect_acls([Path("D:/")]) == [value]


@pytest.mark.parametrize("damage,private,expected", [
    ("none", True, "PASS"),
    ("modify", False, "UNSAFE_STORAGE_ACL"),
    ("foreign_read", True, "UNSAFE_STORAGE_ACL"),
    ("owner", True, "UNTRUSTED_STORAGE_OWNER_OR_ACL"),
    ("unsupported", True, "UNTRUSTED_STORAGE_OWNER_OR_ACL"),
    ("inherited_root", True, "PRIVATE_INDEX_ROOT_REQUIRED"),
    ("owner_readonly", True, "OWNER_FULL_CONTROL_REQUIRED"),
    ("deny", True, "UNSUPPORTED_PRIVATE_DENY_ACL"),
])
def test_acl_helper_environment_fix_preserves_policy_decision(
        helper_system, monkeypatch, damage, private, expected):
    import json
    from types import SimpleNamespace
    value = fact()
    if damage in ("modify", "foreign_read"):
        value["rules"].append(dict(trusted=False, current=False, mask=0x1301BF if damage == "modify"
                                   else 0x1200A9, allow=True, inherit_only=False))
    elif damage == "owner":
        value["owner_trusted"] = False
    elif damage == "unsupported":
        value["unsupported"] = True
    elif damage == "inherited_root":
        value["protected"] = False
    elif damage == "owner_readonly":
        value["rules"][0]["mask"] = 0x120089
    elif damage == "deny":
        value["rules"][0]["allow"] = False
    monkeypatch.setenv("PSModulePath", "UNTRUSTED_MODULES")
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout=json.dumps([value]).encode(), stderr=b""))

    def decision(facts):
        try:
            safety.check_acl_facts(facts, private=private, root=private)
            return "PASS"
        except safety.StatisticsIndexError as error:
            return str(error)

    assert decision(value) == decision(safety.inspect_acls([Path("D:/")])[0]) == expected


def test_acl_helper_reparse_module_location_is_a_safe_inspection_error(helper_system, monkeypatch):
    def reparse(_):
        raise safety.StatisticsIndexError("UNSAFE_REPARSE_PATH")
    monkeypatch.setattr(safety, "reject_reparse", reparse)
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **kw: pytest.fail("No module path fallback"))
    with pytest.raises(safety.StatisticsIndexError, match="^ACL_INSPECTION_FAILED$"):
        safety.inspect_acls([Path("D:/")])
