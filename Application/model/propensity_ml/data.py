"""Owned numeric caches, one streaming read per role, no raw IDs or JSON copies."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from Application.model.customer_propensity_storage import iter_propensity_rows
from .contracts import COLUMNS, FEATURES, MLDataError, PREPROCESSING_VERSION, canonical, checksum, load_json, write_json
from .preprocessing import numeric_batch


@contextmanager
def writer_lock(root):
    """OS lock releases even on process exit; never infer ownership from file existence."""
    path = Path(root) / ".trainer.lock"
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise MLDataError("TRAINER_ALREADY_RUNNING") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def close_array(array):
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


class NumericCache:
    def __init__(self, directory, manifest):
        self.directory, self.manifest = Path(directory), manifest
        self.x = self.y = self.dates = None
        try:
            for attr, name, dtype, shape in (
                ("x", "x.npy", "float32", (manifest["rows"], 48)),
                ("y", "y.npy", "uint8", (manifest["rows"],)),
                ("dates", "dates.npy", "uint8", (manifest["rows"],))):
                arr = np.load(self.directory / name, mmap_mode="r", allow_pickle=False)
                setattr(self, attr, arr)
                if str(arr.dtype) != dtype or arr.shape != shape:
                    raise MLDataError("INVALID_NUMERIC_CACHE")
        except BaseException:
            self.close()
            raise

    def close(self):
        for arr in (self.x, self.y, self.dates):
            close_array(arr)


def build_cache(directory, dataset, manifest, partition, allowed_dates, batch_size, guard,
                preprocessor=None, progress=lambda value: None, reader=None):
    """No reuse of unverified or incompatible caches. Complete cache is immutable."""
    if partition not in ("train", "validation", "test") or (preprocessor is not None and partition != "train"):
        raise MLDataError("CACHE_FIT_ROLE_FORBIDDEN")
    expected = manifest["counts"]["files"][partition]
    if not expected or not 1 <= batch_size <= 50000:
        raise MLDataError("INVALID_BATCH_OR_COUNT")
    directory = Path(directory)
    reader = iter_propensity_rows if reader is None else reader
    if directory.exists():
        raise MLDataError("CACHE_ALREADY_EXISTS")
    staging = directory.parent / (".cache-building-" + uuid.uuid4().hex)
    # 48 float32 plus labels/date codes; allowance for npy headers and manifests.
    guard.poll(force=True, extra_disk_bytes=expected * 194 + 1024 * 1024)
    staging.mkdir()
    arrays = {}
    source = None
    try:
        for name, dtype, shape in (("x", "float32", (expected, 48)), ("y", "uint8", (expected,)),
                                   ("dates", "uint8", (expected,))):
            arrays[name] = np.lib.format.open_memmap(staging / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
        lookup = {v: j for j, v in enumerate(allowed_dates)}
        positive = 0
        position = 0
        source = reader(dataset, partition, purpose="training", source_version=manifest["source_version"],
                        generation=manifest["generation"], cancelled=lambda: (guard.poll() or False))

        def consume(rows):
            nonlocal position, positive
            guard.poll(force=True)
            x = numeric_batch(rows)
            y = np.fromiter((r["target"] for r in rows), dtype="uint8", count=len(rows))
            if any(type(r["target"]) is not int or r["target"] not in (0, 1) or r["split"] != partition for r in rows):
                raise MLDataError("INVALID_CACHE_LABEL")
            try:
                codes = np.fromiter((lookup[r["prediction_utc"]] for r in rows), dtype="uint8", count=len(rows))
            except KeyError:
                raise MLDataError("FORBIDDEN_CACHE_DATE") from None
            end = position + len(rows)
            if end > expected:
                raise MLDataError("CACHE_COUNT_MISMATCH")
            arrays["x"][position:end] = x
            arrays["y"][position:end] = y
            arrays["dates"][position:end] = codes
            if preprocessor is not None:
                preprocessor.partial_fit(x, role="train")
            positive += int(y.sum(dtype=np.int64))
            position = end
            progress({"stage": "cache_" + partition, "rows": position, "total": expected})

        batch = []
        for row in source:
            batch.append(row)
            if len(batch) >= batch_size:
                consume(batch)
                batch.clear()
        if batch:
            consume(batch)
        if position != expected or positive != manifest["distributions"]["by_split"][partition]["positive"]:
            raise MLDataError("CACHE_COUNT_MISMATCH")
        for arr in arrays.values():
            arr.flush()
            close_array(arr)
        arrays.clear()
        files = {p.name: {"sha256": checksum(p), "bytes": p.stat().st_size} for p in staging.iterdir()}
        spec = {"dataset_version": manifest["version"], "source_version": manifest["source_version"],
                "generation": manifest["generation"], "features": list(FEATURES), "columns": list(COLUMNS),
                "preprocessing_version": PREPROCESSING_VERSION, "role": partition,
                "rows": position, "positive": positive, "dates": list(allowed_dates),
                "storage": "npy-v1-float32-uint8-uint8", "files": files}
        spec["arrays"] = {"x.npy": {"dtype": "float32", "shape": [position, 48]},
                          "y.npy": {"dtype": "uint8", "shape": [position]},
                          "dates.npy": {"dtype": "uint8", "shape": [position]}}
        spec["version"] = hashlib.sha256(canonical(spec)).hexdigest()
        write_json(staging / "manifest.json", spec)
        guard.poll(force=True)
        staging.rename(directory)
        return NumericCache(directory, spec)
    finally:
        if source is not None and hasattr(source, "close"):
            source.close()
        for arr in arrays.values():
            close_array(arr)
        if staging.exists():
            # Only this function's freshly created UUID path is eligible for removal.
            shutil.rmtree(staging)


def load_cache(directory, manifest, partition, dates):
    directory = Path(directory)
    m = load_json(directory / "manifest.json")
    if (m.get("dataset_version") != manifest["version"] or m.get("features") != list(FEATURES)
            or m.get("rows") != manifest["counts"]["files"][partition]
            or m.get("columns") != list(COLUMNS) or m.get("role") != partition or m.get("dates") != list(dates)
            or m.get("preprocessing_version") != PREPROCESSING_VERSION
            or m.get("storage") != "npy-v1-float32-uint8-uint8"
            or m.get("source_version") != manifest["source_version"] or m.get("generation") != manifest["generation"]):
        raise MLDataError("INCOMPATIBLE_CACHE")
    expected_arrays = {"x.npy": {"dtype": "float32", "shape": [m["rows"], 48]},
                       "y.npy": {"dtype": "uint8", "shape": [m["rows"]]},
                       "dates.npy": {"dtype": "uint8", "shape": [m["rows"]]}}
    if m.get("arrays") != expected_arrays:
        raise MLDataError("INCOMPATIBLE_CACHE_ARRAYS")
    base = {k: v for k, v in m.items() if k != "version"}
    if hashlib.sha256(canonical(base)).hexdigest() != m.get("version") or set(m["files"]) != {"x.npy", "y.npy", "dates.npy"}:
        raise MLDataError("CORRUPT_CACHE_MANIFEST")
    for name, f in m["files"].items():
        p = directory / name
        if p.is_symlink() or p.stat().st_size != f["bytes"] or checksum(p) != f["sha256"]:
            raise MLDataError("CORRUPT_CACHE")
    return NumericCache(directory, m)
