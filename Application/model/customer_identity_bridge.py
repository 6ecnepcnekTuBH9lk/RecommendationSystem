"""Current CustomerMerges -> exact historical BPR user IDs, never embeddings.

This is a present-day identity bridge, not an as-of graph for temporal evaluation.
Profiles and tensor payloads are not read. Published source files must be immutable;
changes are detected through their versions and optimistic snapshot checks.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from pathlib import Path
from threading import Lock
from types import MappingProxyType

from Application.customer_profiles import user_mapping_digest
from Application.mindbox.adapters.customer_merges import adapt_customer_merge
from Application.mindbox.canonical_storage import catalog, checked_directory
from Application.mindbox.identity import CustomerIdResolver
from Application.mindbox.raw_reader import DEFAULT_RAW_ROOT, RawExportError, _unique_object, iter_export, part_files


class CustomerModelIdentityStatus(str, Enum):
    MATCHED_EXACT = "MATCHED_EXACT"
    MATCHED_VIA_MERGE = "MATCHED_VIA_MERGE"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_IN_MODEL = "NOT_IN_MODEL"
    INVALID_ID = "INVALID_ID"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SOURCE_INCONSISTENT = "SOURCE_INCONSISTENT"


def _valid_id(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


@dataclass(frozen=True, slots=True)
class CustomerModelIdentityResult:
    canonical_customer_id: str | None = field(repr=False)
    model_user_id: str | None = field(repr=False)
    status: CustomerModelIdentityStatus
    candidate_count: int
    model_generation: str | None = None
    canonical_revision: str | None = None
    source_version: str | None = field(default=None, repr=False)

    def __post_init__(self):
        if not isinstance(self.status, CustomerModelIdentityStatus):
            raise TypeError("Invalid identity status")
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("Invalid identity candidate count")
        if self.is_matched:
            if (self.candidate_count != 1 or not _valid_id(self.canonical_customer_id)
                    or not _valid_id(self.model_user_id)):
                raise ValueError("A match requires exactly one valid model ID")
            exact = self.canonical_customer_id == self.model_user_id
            if exact != (self.status is CustomerModelIdentityStatus.MATCHED_EXACT):
                raise ValueError("Inconsistent identity match status")
        elif self.model_user_id is not None:
            raise ValueError("An unmatched result cannot select a model ID")
        elif self.status is CustomerModelIdentityStatus.AMBIGUOUS:
            if self.candidate_count < 2 or not _valid_id(self.canonical_customer_id):
                raise ValueError("Ambiguity requires multiple candidates")
        elif self.candidate_count != 0:
            raise ValueError("An unavailable match cannot have candidates")
        elif self.status is CustomerModelIdentityStatus.NOT_IN_MODEL:
            if not _valid_id(self.canonical_customer_id):
                raise ValueError("A resolved non-model ID must be valid")
        elif self.canonical_customer_id is not None:
            raise ValueError("An invalid or unavailable source cannot resolve an ID")
        for value in (self.model_generation, self.canonical_revision, self.source_version):
            if value is not None and not _valid_id(value):
                raise ValueError("Invalid identity source version")

    @property
    def is_matched(self) -> bool:
        return self.status in (CustomerModelIdentityStatus.MATCHED_EXACT,
                               CustomerModelIdentityStatus.MATCHED_VIA_MERGE)


@dataclass(frozen=True, slots=True)
class CustomerIdentityDiagnostics:
    model_users: int
    unchanged_model_ids: int
    redirected_model_ids: int
    single_model_groups: int
    ambiguous_groups: int
    ambiguous_model_ids: int


class CustomerIdentityIndex:
    """In-memory grouping of a validated resolver and copied model IDs.

    CustomerIdResolver validates/compresses every alias in its constructor and
    does not cache unknown IDs, so its supported resolve calls do not mutate the
    supplied graph. Group members are immutable and never exposed in repr.
    """

    def __init__(self, resolver: CustomerIdResolver, model_user_ids: list[str] | tuple[str, ...]):
        if not isinstance(resolver, CustomerIdResolver):
            raise TypeError("CustomerIdResolver required")
        if not isinstance(model_user_ids, (list, tuple)):
            raise TypeError("A model ID sequence is required")
        users = tuple(model_user_ids)
        user_mapping_digest(users)
        if not users or any(not _valid_id(u) for u in users):
            raise ValueError("Invalid model user IDs")
        groups = {}
        unchanged = 0
        for user in users:
            canonical = resolver.resolve(user)
            groups.setdefault(canonical, []).append(user)
            unchanged += int(canonical == user)
        self._resolver = resolver
        self._groups = MappingProxyType({key: tuple(users) for key, users in groups.items()})
        ambiguous = [len(users) for users in self._groups.values() if len(users) > 1]
        self.diagnostics = CustomerIdentityDiagnostics(
            len(users), unchanged, len(users) - unchanged,
            len(self._groups) - len(ambiguous), len(ambiguous), sum(ambiguous),
        )

    def resolve(self, customer_id: str) -> CustomerModelIdentityResult:
        if not _valid_id(customer_id):
            return _failure(CustomerModelIdentityStatus.INVALID_ID)
        canonical = self._resolver.resolve(customer_id)
        candidates = self._groups.get(canonical, ())
        if not candidates:
            return CustomerModelIdentityResult(canonical, None, CustomerModelIdentityStatus.NOT_IN_MODEL, 0)
        if len(candidates) > 1:
            # Exact text equality never overrides an ambiguous merged group.
            return CustomerModelIdentityResult(canonical, None, CustomerModelIdentityStatus.AMBIGUOUS, len(candidates))
        user = candidates[0]  # Only the sole confirmed candidate can be selected.
        status = (CustomerModelIdentityStatus.MATCHED_EXACT if user == canonical
                  else CustomerModelIdentityStatus.MATCHED_VIA_MERGE)
        return CustomerModelIdentityResult(canonical, user, status, 1)


def _failure(status):
    return CustomerModelIdentityResult(None, None, status, 0)


class _SourcesChanged(ValueError):
    pass


def _file_version(path):
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _manifest_version(path):
    try:
        return _file_version(path)
    except FileNotFoundError:
        return None  # Legacy model pair; missing canonical catalog is never optional.


@dataclass(frozen=True, slots=True, repr=False)
class _Sources:
    key: tuple
    generation: str | None
    revision: str
    mappings_path: Path
    merges_directory: Path

    @property
    def version(self):
        return hashlib.sha256(repr(self.key).encode("utf-8")).hexdigest()


def _capture_sources(model_dir, raw_root):
    # Reuse the established generation resolver; do not load checkpoint tensors.
    from .BPRMF import _resolve_model_artifact_paths

    model_root, root = Path(model_dir).resolve(), Path(raw_root).resolve()
    manifest = model_root / "current.json"
    catalog_path = root / "canonical/catalog.json"
    manifest_before, catalog_before = _manifest_version(manifest), _file_version(catalog_path)
    if manifest_before is not None:
        with manifest.open(encoding="utf-8") as stream:
            json.load(stream, object_pairs_hook=_unique_object)
    # catalog() wraps I/O failures; probe access first to classify unavailability.
    with catalog_path.open("rb"):
        pass
    data = catalog(root)
    if type(data["schema_version"]) is not int:
        raise ValueError("Invalid canonical schema version")
    revision = data["revision"]
    if not _valid_id(revision):
        raise ValueError("Invalid canonical revision")
    entry = data["customer_merges"]
    if entry is None:
        raise FileNotFoundError("No published CustomerMerges")
    if type(entry["parts"]) is not int or entry["parts"] < 1:
        raise ValueError("Invalid merge part count")
    since, until = (datetime.fromisoformat(entry[k]) for k in ("since", "until"))
    if (since >= until or any(d.tzinfo is None or d.utcoffset() != timedelta(0) for d in (since, until))):
        raise ValueError("Invalid merge coverage")
    directory = checked_directory(root, entry["directory"], "customer_merges")
    # Direct listing preserves the unavailable/inconsistent distinction.
    if not tuple(directory.iterdir()):
        raise FileNotFoundError("No published merge parts")
    parts = part_files(directory, "customer_merges")
    if len(parts) != entry["parts"]:
        raise ValueError("Merge part count differs from catalog")
    mapping_path, checkpoint_path = map(Path, _resolve_model_artifact_paths(str(model_root)))
    generation = None if mapping_path.parent == model_root else mapping_path.parent.name
    key = (model_root, root, manifest_before, generation, mapping_path, _file_version(mapping_path),
           _file_version(checkpoint_path), catalog_before, revision,
           json.dumps(entry, sort_keys=True), tuple((p, _file_version(p)) for p in parts))
    if (_manifest_version(manifest) != manifest_before or _file_version(catalog_path) != catalog_before):
        raise _SourcesChanged("Identity source metadata changed")
    return _Sources(key, generation, revision, mapping_path, directory)


def _build_index(sources):
    with sources.mappings_path.open(encoding="utf-8") as stream:
        mappings = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(mappings, dict) or set(mappings) != {"idx2user", "idx2item"}:
        raise ValueError("Invalid model mappings schema")
    for values in mappings.values():
        if (not isinstance(values, list) or not values or any(not _valid_id(u) for u in values)
                or len(set(values)) != len(values)):
            raise ValueError("Invalid model mappings identifiers")
    resolver = CustomerIdResolver(adapt_customer_merge(record) for record in
                                  iter_export("customer_merges", input_dir=sources.merges_directory))
    return CustomerIdentityIndex(resolver, mappings["idx2user"])


class CustomerIdentityBridge:
    """One lazy index; no disk cache, Qt, profile reads or recommendation calls.

    Requests serialize index construction and publication. Source failures clear
    the cache, return a technical status, and never masquerade as NOT_IN_MODEL.
    """

    def __init__(self):
        self._lock = Lock()
        self._cached = None

    def _get_index(self, model_dir, raw_root):
        # Retry once only when a publication is detected during construction.
        for _ in range(2):
            try:
                sources = _capture_sources(model_dir, raw_root)
                if self._cached is not None and self._cached[0].key == sources.key:
                    index = self._cached[1]
                else:
                    self._cached = None
                    index = _build_index(sources)
                if _capture_sources(model_dir, raw_root).key != sources.key:
                    raise _SourcesChanged("Identity sources changed during construction")
                self._cached = sources, index
                return self._cached
            except _SourcesChanged:
                self._cached = None
        raise _SourcesChanged("Identity sources remain unstable")

    def resolve(self, customer_id: str, *, model_dir: str | Path = "model",
                raw_root: str | Path = DEFAULT_RAW_ROOT) -> CustomerModelIdentityResult:
        """Resolve a current ID or alias. This does not check profile existence."""
        if not _valid_id(customer_id):
            return _failure(CustomerModelIdentityStatus.INVALID_ID)
        with self._lock:
            try:
                sources, index = self._get_index(model_dir, raw_root)
                result = index.resolve(customer_id)
                return replace(result, model_generation=sources.generation,
                               canonical_revision=sources.revision, source_version=sources.version)
            except Exception as exc:
                self._cached = None
                # Deliberately return only a status: parser/OS diagnostics may
                # contain source payloads or paths. Never log the exception.
                unavailable = isinstance(exc, OSError) or (
                    isinstance(exc, RawExportError) and exc.category in ("file_missing", "file_access", "file_read"))
                return _failure(CustomerModelIdentityStatus.SOURCE_UNAVAILABLE if unavailable
                                else CustomerModelIdentityStatus.SOURCE_INCONSISTENT)

    def is_current(self, result: CustomerModelIdentityResult, *, model_dir: str | Path = "model",
                   raw_root: str | Path = DEFAULT_RAW_ROOT) -> bool:
        """Guard the handoff to Raw BPR without rereading merge records.

        Only matched results may call Raw BPR. Before displaying its response,
        require response.model_generation == result.model_generation AND this
        check. Otherwise discard/re-resolve; never display a mixed snapshot.
        """
        if not isinstance(result, CustomerModelIdentityResult) or result.source_version is None:
            return False
        with self._lock:
            try:
                sources = _capture_sources(model_dir, raw_root)
                current = (sources.version == result.source_version
                           and sources.generation == result.model_generation
                           and sources.revision == result.canonical_revision)
                if not current:
                    self._cached = None
                return current
            except Exception:
                self._cached = None
                return False

    def get_diagnostics(self, *, model_dir: str | Path = "model",
                        raw_root: str | Path = DEFAULT_RAW_ROOT) -> CustomerIdentityDiagnostics | None:
        """Safe aggregate counts; None means sources could not be validated."""
        with self._lock:
            try:
                return self._get_index(model_dir, raw_root)[1].diagnostics
            except Exception:
                self._cached = None
                return None


_DEFAULT_BRIDGE = CustomerIdentityBridge()


def resolve_customer_model_identity(customer_id: str, *, model_dir: str | Path = "model",
                                    raw_root: str | Path = DEFAULT_RAW_ROOT) -> CustomerModelIdentityResult:
    """Convenience API reusing one source-versioned in-memory index."""
    return _DEFAULT_BRIDGE.resolve(customer_id, model_dir=model_dir, raw_root=raw_root)
