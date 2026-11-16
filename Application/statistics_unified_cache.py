"""Private versioned cache for latest and filter-keyed reports."""

from contextlib import suppress
import json
import os
from pathlib import Path
import stat
import tempfile

from Application.statistics_cache import _unique_object
from Application.statistics_storage_safety import StatisticsIndexError, path_identity, reject_reparse, validate_storage_root
from Application.statistics_unified_dto import (
    RESULT_SCHEMA, SLICE_SCHEMA, StatisticsResultError, UnifiedResult, _digest, read_envelope, validate_unified,
)


def _safe_file(path):
    info = reject_reparse(path).lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise StatisticsIndexError("UNSAFE_CACHE_FILE")
    return info


_NAMESPACE = f"unified-v{RESULT_SCHEMA}"
LATEST_RESULT_NAME = f"{_NAMESPACE}-latest.json"
LATEST_SELECTION_NAME = f"{_NAMESPACE}-last-selection.json"
_SLICE_LIMIT = 16

def _identity(info):
    return info.st_dev, info.st_ino


def _stamp(info):
    return (*_identity(info), info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class UnifiedCache:
    def __init__(self, directory, *, mode="latest"):
        if mode not in ("latest", "slice"):
            raise ValueError("Invalid unified cache mode")
        self.directory = reject_reparse(directory)
        self.root_id = validate_storage_root(self.directory)
        self.mode = mode

    def _path(self, fingerprint):
        if not (fingerprint is None and self.mode == "latest") and not _digest(fingerprint):
            raise ValueError("Invalid unified cache fingerprint")
        name = LATEST_RESULT_NAME if self.mode == "latest" else f"{_NAMESPACE}-slice-{fingerprint}.json"
        return self.directory / name

    def _verify(self, *paths):
        if validate_storage_root(self.directory, children=paths) != self.root_id:
            raise StatisticsIndexError("CACHE_ROOT_CHANGED")

    def load(self, fingerprint, *, check=lambda: None, semantic=False):
        path = self._path(fingerprint)
        self._verify()
        check()
        if not os.path.lexists(path):
            return None
        before = _safe_file(path)
        self._verify(path)
        try:
            with path.open(encoding="utf-8") as stream:
                opened = _stamp(os.fstat(stream.fileno()))
                # Windows can report change time for the handle and creation
                # time for lstat. Compare ctime only within the same API.
                if opened[:-1] != _stamp(before)[:-1]:
                    raise StatisticsIndexError("CACHE_FILE_CHANGED")
                try:
                    saved = json.load(stream, object_pairs_hook=_unique_object)
                except (ValueError, UnicodeError, RecursionError):
                    saved = None
                if _stamp(os.fstat(stream.fileno())) != opened:
                    raise StatisticsIndexError("CACHE_FILE_CHANGED")
        except FileNotFoundError:
            raise StatisticsIndexError("CACHE_FILE_CHANGED") from None
        self._verify(path)
        if _stamp(_safe_file(path)) != _stamp(before):
            raise StatisticsIndexError("CACHE_FILE_CHANGED")
        check()
        if not isinstance(saved, dict):
            return None
        if self.mode == "slice":
            if (set(saved) != {"schema", "fingerprint", "report"} or type(saved["schema"]) is not int
                    or saved["schema"] != SLICE_SCHEMA or saved["fingerprint"] != fingerprint):
                return None
            saved = saved["report"]
        if not isinstance(saved, dict) or saved.get("schema_version") != RESULT_SCHEMA:
            return None
        try:
            result = read_envelope(saved, check=check)
        except StatisticsResultError:
            return None
        return result if isinstance(result, UnifiedResult) and (semantic or fingerprint is None or
                         result.metadata["fingerprint"] == fingerprint) else None

    def latest_candidate(self, *, check=lambda: None):
        """Read a validated candidate. The service still has to verify compatibility."""
        if self.mode != "latest":
            raise ValueError("Only the latest cache has a single candidate")
        return self.load(None, check=check)

    def last_selection(self):
        path = self.directory / LATEST_SELECTION_NAME
        if not os.path.lexists(path):
            return None
        self._verify(path)
        before = _stamp(_safe_file(path))
        if before[2] > 64 * 1024:
            return {}
        try:
            with path.open(encoding="utf-8") as stream:
                value = json.load(stream, object_pairs_hook=_unique_object)
        except (OSError, ValueError):
            return {}
        if _stamp(_safe_file(path)) != before:
            raise StatisticsIndexError("CACHE_FILE_CHANGED")
        return value if isinstance(value, dict) else {}

    def publish_selection(self, *, key, selection, fingerprint, analytics_generation, check):
        from Application.mindbox.canonical_storage import atomic_json
        if self.mode != "slice" or not _digest(key) or not _digest(fingerprint):
            raise ValueError("Invalid latest report binding")
        path = self.directory / LATEST_SELECTION_NAME
        self._verify(self._path(key))
        if os.path.lexists(path):
            _safe_file(path)
            self._verify(path)
        check()
        atomic_json(path, dict(key=key, selection=selection, fingerprint=fingerprint,
                               analytics_generation=analytics_generation))
        # Keep a bounded history. Only remove files in our exact slice namespace.
        with suppress(OSError, StatisticsIndexError):
            active = self._path(key)
            candidates = []
            for item in self.directory.glob(f"{_NAMESPACE}-slice-*.json"):
                if item != active and _digest(item.stem.removeprefix(f"{_NAMESPACE}-slice-")):
                    with suppress(OSError, StatisticsIndexError):
                        candidates.append((_stamp(_safe_file(item)), item))
            candidates.sort(key=lambda entry: (entry[0][3], entry[1].name), reverse=True)
            for stamp, item in candidates[_SLICE_LIMIT - 1:]:
                with suppress(OSError, StatisticsIndexError):
                    self._verify(item)
                    if _stamp(_safe_file(item)) == stamp:
                        item.unlink()

    def save(self, result, *, check=lambda: None, key=None):
        validate_unified(result, check=check)
        fingerprint = key if key is not None else result.metadata["fingerprint"]
        path = self._path(fingerprint)
        self._verify()
        previous = _stamp(_safe_file(path)) if os.path.lexists(path) else None
        if previous is not None:
            self._verify(path)
        saved = result.envelope()
        if self.mode == "slice":
            saved = {"schema": SLICE_SCHEMA, "fingerprint": fingerprint, "report": saved}
        check()
        fd, temp = tempfile.mkstemp(prefix=f".{_NAMESPACE}-", suffix=".tmp", dir=self.directory)
        temp = Path(temp)
        identity = None
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                identity = _identity(os.fstat(stream.fileno()))
                self._verify(temp)
                if _identity(_safe_file(temp)) != identity:
                    raise StatisticsIndexError("CACHE_FILE_CHANGED")
                encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                for i, piece in enumerate(encoder.iterencode(saved)):
                    if i % 4096 == 0:
                        check()
                    stream.write(piece)
                stream.flush()
                os.fsync(stream.fileno())
            self._verify(*(p for p in (temp, path) if os.path.lexists(p)))
            current = _stamp(_safe_file(path)) if os.path.lexists(path) else None
            if current != previous or _identity(_safe_file(temp)) != identity:
                raise StatisticsIndexError("CACHE_FILE_CHANGED")
            check()
            os.replace(temp, path)
        finally:
            with suppress(OSError, StatisticsIndexError):
                if (identity is not None and path_identity(reject_reparse(self.directory)) == self.root_id
                        and _identity(_safe_file(temp)) == identity):
                    temp.unlink()
