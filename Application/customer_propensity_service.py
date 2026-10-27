"""Optional, fail-closed lookup of frozen research predictions; no Qt or scoring."""
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import base64
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
from time import perf_counter

from Application import customer_history_index as history
from Application.mindbox.adapters import adapt_customer_merge
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import iter_export
from Application.paths import PROJECT_ROOT, USER_SETTINGS_DIR

logger = logging.getLogger(__name__)
KEY_SETTINGS_PATH = USER_SETTINGS_DIR / "customer_propensity.json"

MODEL_VERSION = "33b895839de5b0a1968bdcc29e9e13aa7c27d1780cef6a56017f6d19ddd6129f"
PREDICTION_VERSION = "cf19b2bc746ab9ba9d4468605b5d63affd5abc18bc7610f40509f31984da5070"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "filtered_data/customer_propensity_predictions" / PREDICTION_VERSION


class PropensityStatus(str, Enum):
    READY = "READY"
    WARNING = "WARNING"
    NOT_FOUND = "NOT_FOUND"
    NOT_READY = "NOT_READY"
    STALE = "STALE"
    IDENTITY_NOT_CONFIRMED = "IDENTITY_NOT_CONFIRMED"
    KEY_UNAVAILABLE = "KEY_UNAVAILABLE"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True, repr=False)
class CustomerPropensity:
    status: PropensityStatus
    canonical_customer_id: str | None = None
    probability: float | None = None
    prediction_date: str | None = None
    target_until: str | None = None
    model_version: str | None = None
    warnings: tuple[str, ...] = ()
    snapshot: tuple = ()
    lookup_ms: float | None = None
    key_error: str | None = None

    def __post_init__(self):
        if not isinstance(self.status, PropensityStatus):
            raise TypeError("Invalid propensity status")
        if self.status in (PropensityStatus.READY, PropensityStatus.WARNING):
            if (not isinstance(self.canonical_customer_id, str) or not self.canonical_customer_id
                    or not isinstance(self.probability, (float, int)) or not math.isfinite(self.probability)
                    or not 0 <= self.probability <= 1 or not isinstance(self.model_version, str)
                    or len(self.model_version) != 64):
                raise ValueError("Invalid research prediction")
            start = datetime.fromisoformat(self.prediction_date.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.target_until.replace("Z", "+00:00"))
            if start.utcoffset() != timedelta(0) or end.utcoffset() != timedelta(0) or end-start != timedelta(days=30):
                raise ValueError("Invalid research prediction horizon")
        if not isinstance(self.warnings, tuple) or any(not isinstance(w, str) for w in self.warnings):
            raise TypeError("Invalid research warnings")


class _KeyUnavailable(Exception):
    def __init__(self, reason="UNAVAILABLE"):
        # Only static codes may reach logs; never paths or OS exception text.
        allowed = {"UNAVAILABLE", "CONFIG_INVALID", "FILE_NOT_FOUND", "ACCESS_DENIED", "INVALID_FORMAT",
                   "UNSAFE_ACL", "ACL_CHECK_FAILED", "ACL_CHECK_TIMEOUT", "ACL_CHECKER_UNAVAILABLE",
                   "FILE_CHANGED", "READ_FAILED"}
        self.reason = reason if reason in allowed else "UNAVAILABLE"
        super().__init__(self.reason)


class _Stale(Exception):
    pass


class _Identity(Exception):
    pass


def pseudonym(canonical_id, secret, expected_key_id):
    """Exactly propensity v2's domains and UTF-8 input, with no normalization."""
    if (not isinstance(canonical_id, str) or not canonical_id or canonical_id != canonical_id.strip()
            or not all(c.isprintable() for c in canonical_id)):
        raise _Identity
    if not isinstance(secret, bytes) or not 32 <= len(secret) <= 4096:
        raise _KeyUnavailable("INVALID_FORMAT")
    actual = hmac.new(secret, b"kanzler-propensity-key-id-v1", hashlib.sha256).hexdigest()
    if not isinstance(expected_key_id, str) or not hmac.compare_digest(actual, expected_key_id):
        raise _Stale("KEY_ID_MISMATCH")
    return hmac.new(secret, b"kanzler-propensity-customer-v1\0" + canonical_id.encode("utf-8"), hashlib.sha256).hexdigest()


def default_key_path():
    # Explicit local path identifies the SAME BUILD-01 key when packaged Windows
    # filesystem virtualization makes its logical AppData path host-dependent.
    # Invalid configuration must never fall back to another key.
    missing = object()
    try:
        with KEY_SETTINGS_PATH.open(encoding="utf-8") as stream:
            config = json.load(stream)
    except FileNotFoundError:
        config = missing
    except (OSError, ValueError):
        raise _KeyUnavailable("CONFIG_INVALID") from None
    if config is not missing:
        if (not isinstance(config, dict) or set(config) != {"key_file"}
                or not isinstance(config["key_file"], str) or not config["key_file"].strip()
                or "\0" in config["key_file"] or not Path(config["key_file"]).is_absolute()):
            raise _KeyUnavailable("CONFIG_INVALID")
        return Path(config["key_file"])
    # Legacy location for builds made outside a packaged/virtualized process.
    # No LOCALAPPDATA guessing, directory search, new key or fallback.
    return Path.home() / "AppData/Local/KANZLER-AI/propensity-secrets/propensity-v2.key"


def _stamp(path):
    s = Path(path).stat()
    return s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _restricted_key(path):
    """Read only an existing locally protected key. No creation, chmod or fallback."""
    try:
        if path is None:
            raise _KeyUnavailable("FILE_NOT_FOUND")
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or not 32 <= info.st_size <= 4096:
            raise _KeyUnavailable("INVALID_FORMAT")
    except FileNotFoundError:
        raise _KeyUnavailable("FILE_NOT_FOUND") from None
    except PermissionError:
        raise _KeyUnavailable("ACCESS_DENIED") from None
    except OSError:
        raise _KeyUnavailable("READ_FAILED") from None
    if os.name == "nt":
        # File.GetAccessControl avoids shell filesystem-provider initialization.
        # Only a static verdict is returned, never a SID, key, ID or pseudonym.
        quoted = "'" + str(path.resolve()).replace("'", "''") + "'"
        script = ("$ErrorActionPreference='Stop'; $sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value; "
                  f"$a=[System.IO.File]::GetAccessControl({quoted}); "
                  "$bad=@($a.Access | Where-Object { $_.AccessControlType -eq 'Allow' -and "
                  "$_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value -notin "
                  "@($sid,'S-1-5-18','S-1-5-32-544') }); "
                  "if($bad.Count -eq 0 -and $a.Access.Count -gt 0){Write-Output 'KEY_ACL_OK'}else{exit 2}")
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        try:
            result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                    capture_output=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode == 2:
                raise _KeyUnavailable("UNSAFE_ACL")
            if result.returncode or b"KEY_ACL_OK" not in result.stdout:
                raise _KeyUnavailable("ACL_CHECK_FAILED")
        except subprocess.TimeoutExpired:
            raise _KeyUnavailable("ACL_CHECK_TIMEOUT") from None
        except OSError:
            raise _KeyUnavailable("ACL_CHECKER_UNAVAILABLE") from None
    elif path.stat().st_mode & 0o077:
        raise _KeyUnavailable("UNSAFE_ACL")
    try:
        value = path.read_bytes()
        if not 32 <= len(value) <= 4096:
            raise _KeyUnavailable("INVALID_FORMAT")
        return value
    except PermissionError:
        raise _KeyUnavailable("ACCESS_DENIED") from None
    except FileNotFoundError:
        raise _KeyUnavailable("FILE_NOT_FOUND") from None
    except OSError:
        raise _KeyUnavailable("READ_FAILED") from None


class CustomerPropensityService:
    """Owned by the existing serial card worker. Initial artifact check is cached."""
    def __init__(self, *, prediction_dir=DEFAULT_PREDICTIONS, key_path=None, cache_dir=history.DEFAULT_CACHE_DIR):
        self.prediction_dir = Path(prediction_dir)
        # Resolve optional configuration on the worker, not in the GUI constructor.
        self.key_path = Path(key_path) if key_path is not None else None
        self._history = history.CustomerHistoryIndexService(cache_dir=cache_dir)
        self._store = self._artifact_version = self._resolver_key = self._resolver = None
        self._secret = self._key_version = None

    def _files(self):
        base = PROJECT_ROOT / "filtered_data"
        # The artifact identifies all other paths after its initial validation.
        paths = [self.prediction_dir / n for n in ("manifest.json", "predictions.sqlite")]
        if self._store is not None:
            c = self._store.manifest["contract"]
            paths += [base / "customer_propensity_models" / c["model_version"] / n for n in (
                "manifest.json", "preprocessing.json", "linear.json", "calibration.json", "metrics.json")]
            paths += [base / "customer_propensity" / c["dataset_version"] / "manifest.json",
                      base / "customer_propensity/CURRENT_INFERENCE.json"]
        return tuple(_stamp(p) for p in paths)

    def _load_store(self):
        if self._store is not None:
            if self._files() != self._artifact_version:
                raise _Stale
            return self._store
        from Application.model.propensity_ml.predictions import PredictionStore, validate_inputs
        version = self._files()
        store = PredictionStore(self.prediction_dir, verify_integrity=True)
        c = store.manifest["contract"]
        base = PROJECT_ROOT / "filtered_data"
        model, dataset = validate_inputs(base / "customer_propensity_models" / c["model_version"],
                                         base / "customer_propensity" / c["dataset_version"])
        if (model.metadata["version"] != c["model_version"] or dataset["version"] != c["dataset_version"]
                or any(dataset[k] != c[k] for k in ("source_version", "generation"))
                or dataset["identity"]["key_id"] != c["key_id"]
                or dataset["prediction_dates"] != [c["prediction_date"]]):
            raise _Stale
        if self._files() != version:
            raise _Stale
        self._store = store
        self._artifact_version = self._files()
        return store

    def _identity_snapshot(self, canonical_id, root, contract):
        # The same source lock, graph and metadata gate used by history v2.
        sources = history._capture_sources(root)
        path = self._history.index_path(raw_root=root)
        version = history._database_version(path)
        with history._read_index(path) as db:
            ready = history._inspect_connection(db, sources)  # NEVER verify_integrity here.
        if ready.source_version != contract["source_version"] or ready.generation != contract["generation"]:
            raise _Stale
        merge = sources.components[0]
        if self._resolver_key != merge.key:
            self._resolver = CustomerIdResolver(adapt_customer_merge(row) for row in
                                                iter_export("customer_merges", input_dir=merge.directory))
            self._resolver_key = merge.key
        if self._resolver.resolve(canonical_id) != canonical_id:
            raise _Identity
        if version != history._database_version(path) or history._capture_sources(root).version != sources.version:
            raise _Stale
        return sources.version, ready.generation, merge.key, version

    def _key(self):
        try:
            if self.key_path is None:
                self.key_path = default_key_path()
            before = _stamp(self.key_path)
            if self._key_version != before:
                self._secret = None
                value = _restricted_key(self.key_path)
                if _stamp(self.key_path) != before:
                    raise _KeyUnavailable("FILE_CHANGED")
                self._secret, self._key_version = value, before
            return self._secret
        except FileNotFoundError:
            raise _KeyUnavailable("FILE_NOT_FOUND") from None
        except PermissionError:
            raise _KeyUnavailable("ACCESS_DENIED") from None
        except OSError:
            raise _KeyUnavailable("READ_FAILED") from None

    def get(self, canonical_id, *, raw_root=history.DEFAULT_RAW_ROOT, cancelled=lambda: False):
        started = perf_counter()
        if cancelled():
            return CustomerPropensity(PropensityStatus.CANCELLED)
        if not isinstance(canonical_id, str) or not canonical_id or canonical_id != canonical_id.strip():
            return CustomerPropensity(PropensityStatus.IDENTITY_NOT_CONFIRMED)
        try:
            store = self._load_store()
            c = store.manifest["contract"]
            root = Path(raw_root).resolve()
            with history._reader_lock(root):
                snapshot = self._identity_snapshot(canonical_id, root, c)
                key = pseudonym(canonical_id, self._key(), c["key_id"])
                if cancelled():
                    return CustomerPropensity(PropensityStatus.CANCELLED)
                found = store.get_prediction(key, c["model_version"], c["prediction_date"])
                from Application.model.propensity_ml.predictions import PredictionStatus
                if found.status is PredictionStatus.NOT_FOUND:
                    result = CustomerPropensity(PropensityStatus.NOT_FOUND, canonical_id)
                elif found.status is not PredictionStatus.READY:
                    raise _Stale
                else:
                    p = found.prediction
                    result = CustomerPropensity(PropensityStatus.WARNING if p.warnings else PropensityStatus.READY,
                                                canonical_id, p.probability, p.prediction_date, p.target_until,
                                                p.model_version, p.warnings)
                token = snapshot, self._artifact_version, self._key_version
                if snapshot != self._identity_snapshot(canonical_id, root, c) or self._files() != self._artifact_version:
                    raise _Stale
                if _stamp(self.key_path) != self._key_version:
                    raise _KeyUnavailable("FILE_CHANGED")
            if cancelled():
                return CustomerPropensity(PropensityStatus.CANCELLED)
            from dataclasses import replace
            return replace(result, snapshot=token, lookup_ms=(perf_counter() - started)*1000)
        except _Identity:
            return CustomerPropensity(PropensityStatus.IDENTITY_NOT_CONFIRMED)
        except _KeyUnavailable as exc:
            logger.warning("customer_propensity stage=key reason=%s", exc.reason)
            return CustomerPropensity(PropensityStatus.KEY_UNAVAILABLE, key_error=exc.reason)
        except _Stale as exc:
            reason = "KEY_ID_MISMATCH" if exc.args == ("KEY_ID_MISMATCH",) else None
            if reason:
                logger.warning("customer_propensity stage=key reason=%s", reason)
            return CustomerPropensity(PropensityStatus.STALE, key_error=reason)
        except history._Stale:
            return CustomerPropensity(PropensityStatus.STALE)
        except FileNotFoundError:
            return CustomerPropensity(PropensityStatus.NOT_READY)
        except (OSError, sqlite3.OperationalError):
            return CustomerPropensity(PropensityStatus.NOT_READY)
        except Exception:
            return CustomerPropensity(PropensityStatus.ERROR)

    def validate(self, result, *, raw_root=history.DEFAULT_RAW_ROOT):
        if result.status not in (PropensityStatus.READY, PropensityStatus.WARNING, PropensityStatus.NOT_FOUND):
            return result
        try:
            store = self._load_store()
            root = Path(raw_root).resolve()
            with history._reader_lock(root):
                token = (self._identity_snapshot(result.canonical_customer_id, root, store.manifest["contract"]),
                         self._artifact_version, _stamp(self.key_path))
            if token == result.snapshot:
                return result
        except Exception:
            pass
        return CustomerPropensity(PropensityStatus.STALE)
