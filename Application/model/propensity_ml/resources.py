"""Cooperative Windows private-commit, physical RAM and volume-space monitoring."""
import ctypes
from ctypes import wintypes
import base64
import os
from pathlib import Path
import shutil
import subprocess
from time import monotonic

from Application.model.customer_propensity_resources import volume_identity
from .contracts import MLCancelled, MLDataError

GIB = 1024 ** 3


def sample_resources():
    if os.name != "nt":
        raise MLDataError("PRIVATE_COMMIT_MONITOR_REQUIRES_WINDOWS")

    class Memory(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "commit_limit", "available_commit",
                "total_virtual", "available_virtual", "extended")]

    class Process(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile_usage", "peak_pagefile_usage", "private_usage")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Memory)]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Process), wintypes.DWORD]
    mem, proc = Memory(), Process()
    mem.length, proc.cb = ctypes.sizeof(mem), ctypes.sizeof(proc)
    if (not kernel.GlobalMemoryStatusEx(ctypes.byref(mem))
            or not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(proc), proc.cb)):
        raise MLDataError("RESOURCE_MONITOR_UNAVAILABLE")
    return {name: getattr(mem, name) for name in (
        "total_physical", "available_physical", "commit_limit", "available_commit")} | {
        name: getattr(proc, name) for name in ("private_usage", "working_set", "peak_working_set", "peak_pagefile_usage")}


def protect_directory(path):
    """Protect only a newly owned run directory; never modify a parent/user directory."""
    if os.name != "nt":
        Path(path).chmod(0o700)
        return
    quoted_path = "'" + str(Path(path).resolve()).replace("'", "''") + "'"
    script = (f"$p={quoted_path}; $sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
              "& icacls.exe $p /inheritance:r /grant:r \"*$($sid):(OI)(CI)F\" "
              "'*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' | Out-Null; exit $LASTEXITCODE")
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            capture_output=True)
    if result.returncode:
        raise MLDataError("PRIVATE_CACHE_ACL_FAILED")


class ResourceGuard:
    """Periodic checks cannot guarantee a hard OS allocation limit."""
    def __init__(self, directories, cancelled=lambda: False, *, private_limit=3 * GIB,
                 physical_floor=4 * GIB, commit_floor=3 * GIB, disk_floor=20 * GIB,
                 sampler=sample_resources, observer=lambda value: None):
        if disk_floor < 20 * GIB:
            raise ValueError("Disk floor must be >=20 GiB")
        self.directories = tuple(Path(p).resolve(strict=True) for p in directories)
        self.cancelled, self.sampler, self.observer = cancelled, sampler, observer
        self.private_limit, self.physical_floor, self.commit_floor = private_limit, physical_floor, commit_floor
        self.disk_floor, self.last = disk_floor, -float("inf")
        self.started = monotonic()
        self.samples = 0
        self.peak_private = self.peak_working_set = 0
        self.minimum_free = self.minimum_physical = self.minimum_commit = None

    def poll(self, *, force=False, extra_disk_bytes=0):
        if self.cancelled():
            raise MLCancelled
        now = monotonic()
        if not force and now - self.last < 1:
            return
        s = self.sampler()
        free, seen = [], set()
        for p in self.directories:
            volume = volume_identity(p)
            if volume not in seen:
                free.append(shutil.disk_usage(p).free)
                seen.add(volume)
        s["free_disk"] = min(free)
        s["elapsed_seconds"] = now - self.started
        self.last = now
        self.samples += 1
        self.peak_private = max(self.peak_private, s["private_usage"])
        self.peak_working_set = max(self.peak_working_set, s["peak_working_set"])
        for attr, field in (("minimum_free", "free_disk"), ("minimum_physical", "available_physical"),
                            ("minimum_commit", "available_commit")):
            old = getattr(self, attr)
            setattr(self, attr, s[field] if old is None else min(old, s[field]))
        self.observer(s)
        if s["private_usage"] > self.private_limit:
            raise MLDataError("PRIVATE_COMMIT_LIMIT_EXCEEDED")
        if s["available_physical"] < self.physical_floor or s["available_commit"] < self.commit_floor:
            raise MLDataError("SYSTEM_MEMORY_RESERVE_EXCEEDED")
        if min(free) < self.disk_floor + extra_disk_bytes:
            raise MLDataError("INSUFFICIENT_DISK_RESERVE")

    def payload(self):
        return {"samples": self.samples, "sampled_peak_private_bytes": self.peak_private,
                "peak_working_set_bytes": self.peak_working_set, "minimum_free_disk_bytes": self.minimum_free,
                "minimum_available_physical_bytes": self.minimum_physical, "minimum_available_commit_bytes": self.minimum_commit,
                "private_limit_bytes": self.private_limit, "disk_floor_bytes": self.disk_floor,
                "elapsed_seconds": monotonic() - self.started,
                "enforcement": "COOPERATIVE_PERIODIC; not a hard OS allocation reservation"}
