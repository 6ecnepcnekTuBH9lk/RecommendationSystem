"""CPU-only original BPR ranking, independent of contacts, catalogs and exports.

Only exact IDs from the saved user mapping are accepted. Generation directories
are immutable snapshots; a changed manifest or artifact replaces the one cached
snapshot on the next request. No model is loaded at import time.
"""

from dataclasses import dataclass
from copy import deepcopy
from enum import Enum
import json
import math
from pathlib import Path
from threading import Lock, RLock

import numpy as np
from numpy._core.multiarray import _reconstruct
import torch

from . import BPRMF as core


class RawBprArtifactError(ValueError):
    """Missing, changing, unsafe or incompatible model artifacts; no fallback."""


class RawBprStatus(str, Enum):
    OK = "OK"
    USER_NOT_IN_MODEL = "USER_NOT_IN_MODEL"


@dataclass(frozen=True, slots=True)
class RawBprRecommendation:
    rank: int
    model_item_code: str
    raw_score: float

    def __post_init__(self):
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        _validate_id(self.model_item_code)
        if type(self.raw_score) is not float or not math.isfinite(self.raw_score):
            raise ValueError("raw_score must be a finite float")


@dataclass(frozen=True, slots=True, repr=False)
class RawBprResult:
    model_generation: str | None  # None only for the legacy pair without current.json.
    model_user_id: str
    status: RawBprStatus
    recommendations: tuple[RawBprRecommendation, ...]

    def __post_init__(self):
        _validate_id(self.model_user_id)
        if self.model_generation is not None:
            _validate_id(self.model_generation)
        if not isinstance(self.status, RawBprStatus):
            raise TypeError("status must be a RawBprStatus")
        if (not isinstance(self.recommendations, tuple)
                or any(not isinstance(r, RawBprRecommendation) for r in self.recommendations)):
            raise TypeError("recommendations must be an immutable tuple")
        if self.status is RawBprStatus.USER_NOT_IN_MODEL and self.recommendations:
            raise ValueError("An unknown user cannot have recommendations")


def _validate_id(value):
    if not isinstance(value, str):
        raise TypeError("Model identifiers must be strings")
    if not value or value != value.strip():
        raise ValueError("Model identifiers must be nonempty without edge whitespace")


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate mappings field")
        result[key] = value
    return result


def _validate_checkpoint(mappings, checkpoint):
    core._validate_model_artifacts(mappings, checkpoint)
    for name, dimension in (("idx2user", "num_users"), ("idx2item", "num_items")):
        values = mappings[name]
        for value in values:
            _validate_id(value)
        if len(set(values)) != len(values):
            raise ValueError("Duplicate model identifiers")
        if type(checkpoint[dimension]) is not int or checkpoint[dimension] < 1:
            raise ValueError("Model dimensions must be positive integers")

    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise ValueError("Invalid checkpoint config")
    dim = config.get("embedding_dim")
    if type(dim) is not int or dim < 1:
        raise ValueError("Invalid embedding dimension")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Invalid state_dict")
    for name, count in (("user_emb.weight", len(mappings["idx2user"])),
                        ("item_emb.weight", len(mappings["idx2item"]))):
        value = state.get(name)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.shape != (count, dim) or not torch.isfinite(value).all()):
            raise ValueError("Invalid embedding tensor")

    features = checkpoint.get("feat2idx", {})
    if (not isinstance(features, dict)
            or any(not isinstance(k, str) or not k or type(v) is not int for k, v in features.items())
            or sorted(features.values()) != list(range(len(features)))):
        raise ValueError("Invalid feature mapping")
    matrix = checkpoint.get("item_feat_mat")
    if matrix is not None:
        if (not isinstance(matrix, np.ndarray) or matrix.dtype != np.dtype("int64")
                or matrix.ndim != 2 or matrix.shape[0] != len(mappings["idx2item"])
                or matrix.shape[1] < 1 or np.any(matrix < -1) or np.any(matrix >= len(features))):
            raise ValueError("Invalid feature matrix")
    if features:
        saved_matrix = state.get("item_feat_mat")
        weights = state.get("feat_emb.weight")
        if (matrix is None or not isinstance(saved_matrix, torch.Tensor)
                or saved_matrix.dtype != torch.int64
                or not np.array_equal(matrix, saved_matrix.numpy())
                or not isinstance(weights, torch.Tensor) or weights.dtype != torch.float32
                or weights.shape != (len(features), dim) or not torch.isfinite(weights).all()):
            raise ValueError("Inconsistent saved item features")
        scale = config.get("feature_scale", 0.2)
        norm = config.get("feature_norm", "mean")
        if (type(scale) not in (int, float) or not math.isfinite(scale)
                or not isinstance(norm, str)
                or norm.strip().lower() not in ("sum", "mean", "avg", "sqrt", "sqrt_mean", "sqrt-norm")):
            raise ValueError("Invalid saved feature configuration")


# PyTorch 2.5 safe_globals is process-wide. Serialize this module's temporary
# allowlist contexts, restore them after loading, and never retry with pickle.
_CHECKPOINT_LOAD_LOCK = Lock()


def _load_checkpoint(path):
    # Exactly the NumPy constructors/dtypes used by this project's checkpoints.
    allowed = [_reconstruct, np.ndarray, np.dtype,
               type(np.dtype("int64")), type(np.dtype("float64"))]
    with _CHECKPOINT_LOAD_LOCK:
        existing = torch.serialization.get_safe_globals()
        with torch.serialization.safe_globals([value for value in allowed if value not in existing]):
            return torch.load(path, map_location="cpu", weights_only=True)


def _artifact_signature(paths):
    stats = (p.stat() for p in paths)
    return tuple((s.st_ino, s.st_size, s.st_mtime_ns) for s in stats)


@dataclass(slots=True, repr=False)
class _LoadedModel:
    key: tuple
    generation: str | None
    user2idx: dict[str, int]
    idx2item: tuple[str, ...]
    model: core.BPRMF
    item_vectors: torch.Tensor
    item_metadata: dict


class RawBprInferenceService:
    """One reusable CPU model per service; requests and cache changes are serialized.

    The convenience function below reuses one service. A UI worker can instead
    own an instance. Artifact failures invalidate the cache and raise
    RawBprArtifactError; they never return a previous generation's results.
    """

    def __init__(self):
        self._lock = RLock()
        self._loaded = None

    def _get_model(self, model_dir):
        try:
            root = Path(model_dir).resolve()
            mapping_path, checkpoint_path = map(Path, core._resolve_model_artifact_paths(str(root)))
            generation = None if mapping_path.parent == root else mapping_path.parent.name
            paths = (mapping_path, checkpoint_path)
            key = (root, generation, _artifact_signature(paths))
            if self._loaded is not None and self._loaded.key == key:
                return self._loaded

            # Release the previous large generation before loading its replacement.
            self._loaded = None
            with mapping_path.open(encoding="utf-8") as stream:
                mappings = json.load(stream, object_pairs_hook=_unique_fields)
            checkpoint = _load_checkpoint(checkpoint_path)
            _validate_checkpoint(mappings, checkpoint)
            model, _, _, _ = core._build_model_from_ckpt(checkpoint, torch.device("cpu"))
            model.requires_grad_(False)
            with torch.inference_mode():
                vectors = model.item_vec_all()
                if not torch.isfinite(vectors).all():
                    raise ValueError("Nonfinite saved item vectors")
            if _artifact_signature(paths) != key[2]:
                raise ValueError("Model artifacts changed during loading")
            self._loaded = _LoadedModel(
                key, generation, {u: i for i, u in enumerate(mappings["idx2user"])},
                tuple(mappings["idx2item"]), model, vectors,
                deepcopy(checkpoint.get("train_item_meta", {})),
            )
            return self._loaded
        except Exception as exc:
            self._loaded = None
            raise RawBprArtifactError("Cannot load valid Raw BPR model artifacts") from exc

    def get_recommendations(self, model_user_id: str, *, k: int = 10,
                            model_dir: str | Path = "model") -> RawBprResult:
        """Return up to K original items; ties use their saved mapping index.

        IDs are exact and never resolved through CustomerMerges. K must be a
        positive built-in int. Scores retain model precision, with no catalog,
        seen mask, replacement, rounding or probability transformation.
        """
        _validate_id(model_user_id)
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if not isinstance(model_dir, (str, Path)):
            raise TypeError("model_dir must be a string or Path")
        with self._lock:
            loaded = self._get_model(model_dir)
            user_index = loaded.user2idx.get(model_user_id)
            if user_index is None:
                return RawBprResult(loaded.generation, model_user_id,
                                    RawBprStatus.USER_NOT_IN_MODEL, ())
            with torch.inference_mode():
                scores = (loaded.model.user_emb.weight[user_index] @ loaded.item_vectors.T).numpy()
            if not np.isfinite(scores).all():
                raise RawBprArtifactError("Nonfinite Raw BPR scores")
            # Stable sorting preserves the original mapping index on exact ties.
            indices = np.argsort(-scores, kind="stable")[:k]
            recommendations = tuple(
                RawBprRecommendation(rank, loaded.idx2item[int(i)], float(scores[i]))
                for rank, i in enumerate(indices, 1)
            )
            return RawBprResult(loaded.generation, model_user_id, RawBprStatus.OK, recommendations)

    def get_source_metadata(self, *, model_dir="model"):
        """Detached historical attributes from the same safe generation cache."""
        with self._lock:
            loaded = self._get_model(model_dir)
            return loaded.generation, deepcopy(loaded.item_metadata)

    def get_candidate_sources(self, model_user_id, *, model_dir="model"):
        """All original ranks plus historical metadata from ONE loaded snapshot.

        Holding the service lock also protects the legacy pair without a named
        generation. File changes are checked before/after both operations.
        """
        with self._lock:
            raw = self.get_recommendations(model_user_id, k=2**31 - 1, model_dir=model_dir)
            loaded = self._loaded
            metadata = deepcopy(loaded.item_metadata)
            if self._get_model(model_dir) is not loaded:
                raise RawBprArtifactError("Artifacts changed during candidate lookup")
            return raw, metadata


_DEFAULT_SERVICE = RawBprInferenceService()


def get_raw_bpr_recommendations(model_user_id: str, *, k: int = 10,
                                model_dir: str | Path = "model") -> RawBprResult:
    """Convenience API with one lazy, generation-aware CPU cache."""
    return _DEFAULT_SERVICE.get_recommendations(model_user_id, k=k, model_dir=model_dir)
