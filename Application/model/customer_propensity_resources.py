"""Read-only SQLite temp discovery and volume-aware disk floors for propensity only.

No global SQLite configuration, environment changes or database writes. Unsupported
runtimes fail closed rather than treating TEMP/TMP as evidence of SQLite placement.
"""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import sqlite3
import _sqlite3
import struct
from contextlib import closing

from Application.model.customer_propensity_history import PropensityDataError

MIN_FREE_BYTES = 20 * 1024**3


def validate_reserve(value):
    if type(value) is not int or value < MIN_FREE_BYTES:
        raise ValueError("Disk reserve must be at least 20 GiB")


def _extension_imports_sqlite_dll():
    """Verify the stdlib extension is dynamically linked to sqlite3.dll.

    Read the PE import directory from disk, never CPython/private pointers or
    arbitrary native memory. Static/custom/unknown extensions are unsupported.
    """
    try:
        path = Path(_sqlite3.__file__)
        if path.stat().st_size > 8 * 1024**2:
            return False
        data = path.read_bytes()
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[:2] != b"MZ" or data[pe:pe + 4] != b"PE\0\0":
            return False
        sections = struct.unpack_from("<H", data, pe + 6)[0]
        optional_size = struct.unpack_from("<H", data, pe + 20)[0]
        optional = pe + 24
        magic = struct.unpack_from("<H", data, optional)[0]
        if not 1 <= sections <= 96 or magic not in (0x10B, 0x20B):
            return False
        directory = optional + (96 if magic == 0x10B else 112) + 8
        if directory + 8 > optional + optional_size:
            return False
        import_rva, import_size = struct.unpack_from("<II", data, directory)
        table = optional + optional_size

        def offset(rva):
            for n in range(sections):
                virtual, address, raw_size, raw = struct.unpack_from("<IIII", data, table + 40 * n + 8)
                if address <= rva < address + min(virtual or raw_size, raw_size):
                    result = raw + rva - address
                    if result < len(data):
                        return result
            raise ValueError

        start = offset(import_rva)
        for n in range(0, min(import_size, 65536) - 19, 20):
            descriptor = struct.unpack_from("<IIIII", data, start + n)
            if not any(descriptor):
                break
            name = offset(descriptor[3])
            if data[name:name + 256].split(b"\0", 1)[0].lower() == b"sqlite3.dll":
                return True
    except (OSError, AttributeError, ValueError, struct.error):
        return False
    return False


def _loaded_sqlite_dll():
    """Use the unique already-loaded sqlite3.dll; never search PATH/load a new copy."""
    if os.name != "nt" or not _extension_imports_sqlite_dll():
        raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                        wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    psapi.EnumProcessModules.restype = wintypes.BOOL
    kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetModuleFileNameW.restype = wintypes.DWORD
    modules = (wintypes.HMODULE * 2048)()
    needed = wintypes.DWORD()
    if (not psapi.EnumProcessModules(kernel.GetCurrentProcess(), modules, ctypes.sizeof(modules), ctypes.byref(needed))
            or needed.value > ctypes.sizeof(modules)):
        raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
    found = []
    for handle in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        buffer = ctypes.create_unicode_buffer(32768)
        n = kernel.GetModuleFileNameW(handle, buffer, len(buffer))
        if not 0 < n < len(buffer):
            raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
        if Path(buffer.value).name.casefold() == "sqlite3.dll":
            found.append((buffer.value, handle))
    if len(found) != 1:
        raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
    return ctypes.CDLL(found[0][0], handle=found[0][1])


def _bind_sqlite():
    lib = _loaded_sqlite_dll()
    lib.sqlite3_sourceid.argtypes = []
    lib.sqlite3_sourceid.restype = ctypes.c_char_p
    lib.sqlite3_libversion.argtypes = []
    lib.sqlite3_libversion.restype = ctypes.c_char_p
    lib.sqlite3_open_v2.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_char_p]
    lib.sqlite3_open_v2.restype = ctypes.c_int
    lib.sqlite3_file_control.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    lib.sqlite3_file_control.restype = ctypes.c_int
    lib.sqlite3_free.argtypes = [ctypes.c_void_p]
    lib.sqlite3_free.restype = None
    lib.sqlite3_close.argtypes = [ctypes.c_void_p]
    lib.sqlite3_close.restype = ctypes.c_int
    # The memory connection interrogates the Python-linked runtime, not its file path.
    with closing(sqlite3.connect(":memory:")) as conn:
        expected = conn.execute("SELECT sqlite_source_id()").fetchone()[0]
    if (lib.sqlite3_sourceid().decode("utf-8") != expected
            or lib.sqlite3_libversion().decode("ascii") != sqlite3.sqlite_version):
        raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
    return lib


def _vfs_temp_directory(lib, database):
    """Read-only auxiliary handle; no private CPython sqlite3 pointer/ABI access."""
    db = ctypes.c_void_p()
    try:
        # SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX; no CREATE/URI/global changes.
        if lib.sqlite3_open_v2(os.fsencode(database), ctypes.byref(db), 1 | 0x10000, None) != 0:
            raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
        values = []
        for opcode in (12, 16):  # SQLITE_FCNTL_VFSNAME, SQLITE_FCNTL_TEMPFILENAME
            pointer = ctypes.c_void_p()
            try:
                if lib.sqlite3_file_control(db, b"main", opcode, ctypes.byref(pointer)) != 0 or not pointer.value:
                    raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
                values.append(ctypes.string_at(pointer.value).decode("utf-8"))
            finally:
                if pointer.value:
                    lib.sqlite3_free(pointer)
        if values[0] not in ("win32", "win32-longpath"):
            raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
        filename = Path(values[1])
        if not filename.is_absolute():
            raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
        directory = filename.parent.resolve(strict=True)
        if not directory.is_dir():
            raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
        return directory
    finally:
        if db.value and lib.sqlite3_close(db) != 0:
            raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")


def make_sqlite_temp_locator(database):
    """Bind once per build, re-query the current VFS directory at each disk poll."""
    try:
        lib = _bind_sqlite()
    except (OSError, AttributeError, UnicodeError, sqlite3.Error):
        raise PropensityDataError("SQLITE_TEMP_UNVERIFIED") from None

    def locate():
        try:
            return _vfs_temp_directory(lib, Path(database))
        except (OSError, AttributeError, UnicodeError):
            raise PropensityDataError("SQLITE_TEMP_UNVERIFIED") from None
    return locate


def volume_identity(directory):
    """Resolve mounted volumes/junctions; drive-letter text is not an identity."""
    directory = Path(directory).resolve(strict=True)
    if os.name != "nt":
        return str(directory.stat().st_dev)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetVolumePathNameW.restype = wintypes.BOOL
    kernel.GetVolumeNameForVolumeMountPointW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetVolumeNameForVolumeMountPointW.restype = wintypes.BOOL
    mount = ctypes.create_unicode_buffer(32768)
    guid = ctypes.create_unicode_buffer(1024)
    if (not kernel.GetVolumePathNameW(str(directory), mount, len(mount))
            or not kernel.GetVolumeNameForVolumeMountPointW(mount.value, guid, len(guid))):
        raise PropensityDataError("DISK_INFO_UNAVAILABLE")
    return guid.value.casefold()


def check_disk_space(output_directory, temp_directory, reserve):
    """Check each actual volume once. Free bytes already reflect live temp/output."""
    validate_reserve(reserve)
    if temp_directory is None:
        raise PropensityDataError("SQLITE_TEMP_UNVERIFIED")
    try:
        seen = set()
        for directory in (output_directory, temp_directory):
            identity = volume_identity(directory)
            if identity not in seen:
                if shutil.disk_usage(directory).free < reserve:
                    raise PropensityDataError("INSUFFICIENT_FREE_SPACE")
                seen.add(identity)
    except (OSError, AttributeError):
        raise PropensityDataError("DISK_INFO_UNAVAILABLE") from None
