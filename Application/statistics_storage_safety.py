"""Read-only storage admission, never an ACL repair or a production path change.

Trust the current Windows user, SYSTEM, administrators and OS servicing. Other
principals must not be able to read private objects or mutate their namespace.
Concurrent hostile actions by those trusted principals are outside this model.
"""

import base64
import json
import os
from pathlib import Path
import stat
import subprocess


class StatisticsIndexError(ValueError):
    """Safe state code; no identities, ACL entries or source contents."""


def reject_reparse(path):
    """Inspect lexical ancestors BEFORE resolve(), including Windows junctions."""
    if ".." in Path(path).parts:
        raise StatisticsIndexError("UNSAFE_RELATIVE_PATH")
    path = Path(os.path.abspath(path))
    for node in (path, *path.parents):
        try:
            info = node.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise StatisticsIndexError("UNSAFE_REPARSE_PATH")
    return path


def path_identity(path):
    value = Path(path).lstat()
    return value.st_dev, value.st_ino


def check_owned_staging(root, folder, root_id, folder_id):
    """Fail closed before deletion; never follow a substituted staging path."""
    root, folder = reject_reparse(root), reject_reparse(folder)
    if (folder.parent != root or path_identity(root) != root_id
            or path_identity(folder) != folder_id):
        raise StatisticsIndexError("STAGING_PATH_CHANGED")
    for child in folder.iterdir():
        reject_reparse(child)
        if not child.is_file():
            raise StatisticsIndexError("UNEXPECTED_STAGING_CONTENT")


def local_volume(path):
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel.GetVolumeInformationW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        wintypes.LPWSTR, wintypes.DWORD]
    anchor = Path(path).anchor
    if not anchor or anchor.startswith("\\\\") or kernel.GetDriveTypeW(anchor) != 3:
        raise StatisticsIndexError("LOCAL_FIXED_VOLUME_REQUIRED")
    filesystem = ctypes.create_unicode_buffer(64)
    flags = wintypes.DWORD()
    if (not kernel.GetVolumeInformationW(anchor, None, 0, None, None, ctypes.byref(flags), filesystem, 64)
            or filesystem.value != "NTFS" or flags.value & 0x80000):
        raise StatisticsIndexError("WRITABLE_NTFS_REQUIRED")


def _powershell_environment():
    """Restrict this child's module lookup to the OS Windows PowerShell modules."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetSystemDirectoryW.argtypes = [wintypes.LPWSTR, wintypes.UINT]
    kernel.GetSystemDirectoryW.restype = wintypes.UINT
    directory = ctypes.create_unicode_buffer(32768)
    length = kernel.GetSystemDirectoryW(directory, len(directory))
    if not 0 < length < len(directory):
        raise StatisticsIndexError("ACL_INSPECTION_FAILED")
    modules = reject_reparse(Path(directory.value) / "WindowsPowerShell" / "v1.0" / "Modules")
    for name in ("Microsoft.PowerShell.Security", "Microsoft.PowerShell.Utility"):
        manifest = reject_reparse(modules / name / (name + ".psd1"))
        if not manifest.is_file():
            raise StatisticsIndexError("ACL_INSPECTION_FAILED")
    environment = {key: value for key, value in os.environ.items()
                   if key.casefold() != "psmodulepath"}
    environment["PSModulePath"] = str(modules)
    return environment


def inspect_acls(paths):
    """One Windows API/.NET batch; return masks/booleans, never SID/name values."""
    literals = ",".join("'" + str(p).replace("'", "''") + "'" for p in paths)
    command = r"""
$ErrorActionPreference='Stop'
$me=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$trusted=@($me,'S-1-5-18','S-1-5-32-544')
try {$trusted+=([System.Security.Principal.NTAccount]::new('NT SERVICE\TrustedInstaller')).Translate([System.Security.Principal.SecurityIdentifier]).Value} catch {}
$rows=foreach($p in @(__PATHS__)) {
 $a=Get-Acl -LiteralPath $p
 $raw=[System.Security.AccessControl.RawSecurityDescriptor]::new($a.GetSecurityDescriptorBinaryForm(),0)
 $unsupported=($null -eq $raw.DiscretionaryAcl)
 if($raw.DiscretionaryAcl){foreach($ace in $raw.DiscretionaryAcl){if($ace.AceType.ToString() -notin @('AccessAllowed','AccessDenied')){$unsupported=$true}}}
 $rules=@(foreach($r in $a.Access){
  $s=$r.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
  @{trusted=($s -in $trusted);current=($s -eq $me);mask=[int]$r.FileSystemRights;
    allow=($r.AccessControlType -eq 'Allow');inherit_only=([bool]($r.PropagationFlags -band [System.Security.AccessControl.PropagationFlags]::InheritOnly))}
 })
 @{owner_trusted=($a.GetOwner([System.Security.Principal.SecurityIdentifier]).Value -in $trusted);
   protected=$a.AreAccessRulesProtected;unsupported=$unsupported;rules=$rules}
}
ConvertTo-Json -InputObject @($rows) -Depth 6 -Compress
""".replace("__PATHS__", literals)
    try:
        environment = _powershell_environment()
        # Windows PowerShell startup can prepend user/all-user module paths.
        # Pin the lookup again inside the child before either module autoloads.
        command = "$env:PSModulePath='" + environment["PSModulePath"].replace("'", "''") + "'\n" + command
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
            base64.b64encode(command.encode("utf-16le")).decode("ascii")],
            env=environment, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError, StatisticsIndexError):
        raise StatisticsIndexError("ACL_INSPECTION_FAILED") from None
    if result.returncode:
        raise StatisticsIndexError("ACL_INSPECTION_FAILED")
    try:
        facts = json.loads(result.stdout.decode("utf-8-sig"))
        if not isinstance(facts, list) or len(facts) != len(paths):
            raise ValueError
        for fact in facts:
            if (not isinstance(fact, dict) or set(fact) != {"owner_trusted", "protected", "unsupported", "rules"}
                    or any(type(fact[key]) is not bool for key in ("owner_trusted", "protected", "unsupported"))
                    or not isinstance(fact["rules"], list)):
                raise ValueError
            for rule in fact["rules"]:
                if (not isinstance(rule, dict) or set(rule) != {"trusted", "current", "allow", "inherit_only", "mask"}
                        or any(type(rule[key]) is not bool for key in ("trusted", "current", "allow", "inherit_only"))
                        or type(rule["mask"]) is not int or not -(2**31) <= rule["mask"] < 2**32):
                    raise ValueError
        return facts
    except (UnicodeError, ValueError, TypeError):
        raise StatisticsIndexError("ACL_INSPECTION_FAILED") from None


def check_acl_facts(fact, *, private, root=False):
    # Conservative allowlist admission, NOT an arbitrary-token AccessCheck:
    # foreign allows are rejected even if another ACE could deny that token.
    if fact["unsupported"] or not fact["owner_trusted"]:
        raise StatisticsIndexError("UNTRUSTED_STORAGE_OWNER_OR_ACL")
    if root and not fact["protected"]:
        raise StatisticsIndexError("PRIVATE_INDEX_ROOT_REQUIRED")
    owner_access = 0
    for rule in fact["rules"]:
        mask = rule["mask"] & 0xFFFFFFFF
        for generic, mapped in ((0x10000000, 0x1F01FF), (0x40000000, 0x120116),
                                (0x80000000, 0x120089), (0x20000000, 0x1200A0)):
            if mask & generic:
                mask |= mapped
        if rule["allow"]:
            if rule["current"] and not rule["inherit_only"]:
                owner_access |= mask
            if not rule["trusted"]:
                # Private objects also reject inherit-only foreign grants: they
                # could expose newly created journals/pointer/children.
                if private or (not rule["inherit_only"] and mask & 0xD0150):
                    raise StatisticsIndexError("UNSAFE_STORAGE_ACL")
        elif private:
            raise StatisticsIndexError("UNSUPPORTED_PRIVATE_DENY_ACL")
    if private and owner_access & 0x1F01FF != 0x1F01FF:
        raise StatisticsIndexError("OWNER_FULL_CONTROL_REQUIRED")


def validate_storage_root(directory, *, children=()):
    """A root must be prepared separately; do not create it or alter its ACL."""
    root = reject_reparse(directory)
    for name in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        value = os.environ.get(name)
        if value and root.is_relative_to(Path(os.path.abspath(value))):
            raise StatisticsIndexError("SYNCHRONIZED_ROOT_UNSUPPORTED")
    if not root.is_dir():
        raise StatisticsIndexError("PRIVATE_INDEX_ROOT_REQUIRED")
    local_volume(root)
    children = tuple(dict.fromkeys((*(Path(p) for p in children),
        *(root / name for name in ("current.json", ".writer.lock") if os.path.lexists(root / name)))))
    for child in children:
        reject_reparse(child)
        if not child.is_relative_to(root) or child.stat().st_dev != root.stat().st_dev:
            raise StatisticsIndexError("INVALID_GENERATION_PATH")
    paths = (root, *root.parents, *children)
    if os.name == "nt":
        try:
            facts = inspect_acls(paths)
            for position, fact in enumerate(facts):
                check_acl_facts(fact, private=position == 0 or position > len(root.parents), root=position == 0)
        except StatisticsIndexError:
            raise
        except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError):
            raise StatisticsIndexError("ACL_INSPECTION_FAILED") from None
    else:
        # Private POSIX root and trusted namespace; sticky /tmp does not grant
        # another user permission to rename this user's existing directory.
        if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
            raise StatisticsIndexError("PRIVATE_INDEX_ROOT_REQUIRED")
        for parent in root.parents:
            info = parent.stat()
            if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
                raise StatisticsIndexError("UNSAFE_STORAGE_ACL")
    return path_identity(root)
