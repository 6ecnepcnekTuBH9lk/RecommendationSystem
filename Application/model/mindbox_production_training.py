"""Offline Mindbox production orchestration; no contacts or recommendation export.

Use a dedicated process: legacy training log suppression is process-wide.
"""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from types import MappingProxyType
from collections.abc import Mapping
import uuid

import torch

from Application.mindbox.daily_training_batch import load_chunked_training_batch, prepare_training_data_from_chunked_batch
from . import BPRMF as core
from .mindbox_shadow_training import _atomic_report, _safe_config
from .seen_items import build_seen_items_index, seen_items_from_checkpoint
from .interaction_analytics import analytics_from_checkpoint
from .training_metrics import TrainingRunMetrics
from .training_quality import TrainingQualityReport, TrainingQualityDiagnostics, evaluate_training_quality

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "Модель"
REPORT_ROOT = PROJECT_ROOT / "ВходныеДанные" / "MindboxReports" / "production_training"


@dataclass(frozen=True)
class ProductionTrainingResult:
    batch_id: str | None = None
    quality_report: TrainingQualityReport | None = field(default=None, repr=False)
    dataset: Mapping = field(default_factory=dict, repr=False)
    interaction_window: Mapping = field(default_factory=dict, repr=False)
    training_metrics: TrainingRunMetrics | None = None
    training_started: bool = False
    training_completed: bool = False
    publication_started: bool = False
    published: bool = False
    previous_generation: str | None = None
    published_generation: str | None = None
    prepublish_disk_validation: bool = False
    postpublish_validation: bool = False
    rolled_back: bool = False
    cleanup_failed: bool = False
    cancelled: bool = False
    error_code: str | None = None
    report_path: str | None = field(default=None, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "dataset", MappingProxyType(dict(self.dataset)))
        object.__setattr__(self, "interaction_window", MappingProxyType(dict(self.interaction_window)))


class _Failure(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@contextmanager
def _publication_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".mindbox-production.lock").open("a+b") as stream:
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
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise _Failure("LOCKED") from None
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _current(model_dir):
    try:
        content = (model_dir / "current.json").read_bytes()
    except FileNotFoundError:
        return None, None
    data = json.loads(content)
    # All generations produced by the serializer are UUID hex. Restrict values
    # before including them in an aggregate report or using them as run paths.
    if (not isinstance(data, dict) or set(data) != {"generation"}
            or not isinstance(data["generation"], str) or not re.fullmatch("[0-9a-f]{32}", data["generation"])):
        raise _Failure("ARTIFACT_FAILED")
    return data["generation"], content


def _prepare(manifest, raw_root, catalog, cfg):
    if catalog.name != "Номенклатура.csv" or catalog.parent != Path(cfg.data_dir).resolve():
        raise _Failure("PREPARATION_FAILED")
    batch = load_chunked_training_batch(manifest, raw_root=raw_root, require_complete=True)
    prepared = prepare_training_data_from_chunked_batch(batch, raw_root=raw_root, catalog_path=catalog,
                                                       train_config=cfg, diagnose=True)
    d = prepared.diagnostics
    quality = evaluate_training_quality(prepared.prepared_data, TrainingQualityDiagnostics(
        actions_view=d.actions_view, actions_favorite=d.actions_favorite, malformed_mapped_actions=d.malformed_actions,
        unresolved_products=d.resolution.total.unresolved, unsupported_products=d.resolution.total.unsupported_namespace,
        bpr_events=d.bpr.events_total, unmapped_actions=d.unmapped_actions,
        malformed_action_system_names=d.malformed_action_system_names,
        orders_duplicate_conflicting=d.orders_duplicate_conflicting))
    result = ProductionTrainingResult(batch_id=batch.batch_id, quality_report=quality,
        dataset={**dict(quality.metrics), "complete": prepared.complete, "orders_raw": d.orders_raw,
                 "orders_unique": d.orders_unique, "orders_duplicate_identical": d.orders_duplicate_identical}, interaction_window={
            "since": batch.window.interaction_since.isoformat(), "until": batch.window.interaction_until.isoformat()})
    return prepared.prepared_data, result


def preflight_production_training(training_manifest, *, raw_root, catalog_path, cfg, device=None, progress=None):
    """Read-only preparation/quality; no lock, model writes, report or training."""
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            _, result = _prepare(Path(training_manifest).resolve(), Path(raw_root).resolve(), Path(catalog_path).resolve(), cfg)
        if progress:
            progress(result)
        return replace(result, error_code=None if result.quality_report.training_allowed else "QUALITY_BLOCK")
    except KeyboardInterrupt:
        return ProductionTrainingResult(error_code="CANCELLED", cancelled=True)
    except Exception:
        return ProductionTrainingResult(error_code="PREPARATION_FAILED")


def _verify(model_dir, generation):
    if _current(model_dir)[0] != generation:
        raise _Failure("CURRENT_MODEL_CHANGED")
    mappings, checkpoint = core._load_artifacts(str(model_dir))
    core._validate_model_artifacts(mappings, checkpoint)
    if seen_items_from_checkpoint(checkpoint) is None or analytics_from_checkpoint(checkpoint) is None:
        raise ValueError("Missing embedded inference data")
    core._build_model_from_ckpt(checkpoint, torch.device("cpu"))
    if _current(model_dir)[0] != generation:
        raise _Failure("CURRENT_MODEL_CHANGED")


def _rollback(model_dir, generation, previous):
    """Restore exact previous manifest bytes only while our generation is current."""
    if _current(model_dir)[0] != generation:
        return False
    temporary = model_dir / f".rollback.{uuid.uuid4().hex}.tmp"
    try:
        if previous is None:
            if _current(model_dir)[0] != generation:
                return False
            (model_dir / "current.json").unlink()
        else:
            with temporary.open("xb") as stream:
                stream.write(previous)
                stream.flush()
                os.fsync(stream.fileno())
            if _current(model_dir)[0] != generation:
                return False
            os.replace(temporary, model_dir / "current.json")
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _after_failed_verification(result, model_dir, previous):
    try:
        rolled_back = _rollback(model_dir, result.published_generation, previous)
        still_current = _current(model_dir)[0] == result.published_generation
        result = replace(result, published=still_current, rolled_back=rolled_back,
                         error_code="POST_PUBLISH_VALIDATION_FAILED")
        if not still_current:
            # Only this run's UUID, never any previous or externally published run.
            shutil.rmtree(model_dir / "runs" / result.published_generation)
        return result
    except (Exception, KeyboardInterrupt):
        try:
            current_generation, current_bytes = _current(model_dir)
            still_current = current_generation == result.published_generation
            result = replace(result, rolled_back=result.rolled_back or current_bytes == previous)
        except Exception:
            still_current = result.published
        return replace(result, published=still_current, error_code="ROLLBACK_FAILED", cleanup_failed=True)


def _report(result, config_summary):
    run_id = uuid.uuid4().hex
    q, m = result.quality_report, result.training_metrics
    payload = {"schema_version": 1, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
        "batch_id": result.batch_id, "interaction_window": dict(result.interaction_window),
        "quality": None if q is None else {"level": q.level.value, "training_allowed": q.training_allowed,
            "issues": [{"code": i.code, "count": i.count, "rate": i.rate} for i in q.issues]},
        "dataset": dict(result.dataset), "config": config_summary,
        "training": {"started": result.training_started, "completed": result.training_completed,
            **({name: getattr(m, name) for name in ("epochs_completed", "best_epoch", "best_recall", "best_ndcg", "early_stopped")} if m else {})},
        "publication": {name: getattr(result, name) for name in (
            "published", "previous_generation", "published_generation", "prepublish_disk_validation",
            "postpublish_validation", "rolled_back", "cleanup_failed", "cancelled", "error_code")},
        "artifact": {"embedded_seen": result.prepublish_disk_validation,
                     "embedded_analytics": result.prepublish_disk_validation,
                     "num_users": result.dataset.get("users"), "num_items": result.dataset.get("items")}}
    payload["publication"]["started"] = result.publication_started
    path = REPORT_ROOT / run_id / "report.json"
    _atomic_report(path, payload)
    return str(path)


def train_and_publish_production_model(training_manifest, *, raw_root, catalog_path, cfg, device,
                                        allow_warn=False, model_dir=None, progress=None):
    """Train/publish under an OS lock; failures return safe immutable state.

    model_dir is an API/test seam. The CLI always selects the fixed project model.
    Audit failure never undoes a valid model commit.
    """
    result = ProductionTrainingResult()
    stage = "PREPARATION_FAILED"
    receipt = {}
    previous = None
    config_summary = {}
    try:
        if type(allow_warn) is not bool:
            raise ValueError("Boolean acknowledgement required")
        model_dir = Path(model_dir or MODEL_ROOT).resolve()
        manifest, raw_root, catalog = (Path(p).resolve() for p in (training_manifest, raw_root, catalog_path))
        config_summary = _safe_config(cfg)
        with _publication_lock(model_dir.parent), open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            try:
                previous_generation, previous = _current(model_dir)
                data, prepared_result = _prepare(manifest, raw_root, catalog, cfg)
                result = replace(prepared_result, previous_generation=previous_generation)
                if progress:
                    progress(result)
                if not result.quality_report.training_allowed:
                    raise _Failure("QUALITY_BLOCK")
                if result.quality_report.level.value == "WARN" and not allow_warn:
                    raise _Failure("WARN_NOT_ACKNOWLEDGED")
                if data.analytics is None:
                    raise _Failure("ARTIFACT_FAILED")
                stage = "TRAINING_FAILED"
                result = replace(result, training_started=True)
                core._set_seed(cfg.seed)
                model, _, metrics = core.train_prepared_data_with_metrics(cfg, data, device)
                result = replace(result, training_completed=True, training_metrics=metrics)

                def guard():
                    try:
                        changed = _current(model_dir)[1] != previous
                    except Exception:
                        changed = True
                    if changed:
                        raise _Failure("CURRENT_MODEL_CHANGED")

                guard()
                stage = "ARTIFACT_FAILED"
                seen = build_seen_items_index(data)
                if seen is None or data.analytics is None:
                    raise _Failure("ARTIFACT_FAILED")
                result = replace(result, publication_started=True)
                generation = core._save_artifacts(cfg, data.mappings, model, seen_items=seen, analytics=data.analytics,
                                     model_dir=str(model_dir), _before_commit=guard, _publication_state=receipt)
                result = replace(result, published=True, published_generation=generation,
                                 prepublish_disk_validation=receipt["disk_validated"])
                stage = "POST_PUBLISH_VALIDATION_FAILED"
                _verify(model_dir, result.published_generation)
                result = replace(result, postpublish_validation=True)
            except BaseException as exc:
                if not isinstance(exc, (Exception, KeyboardInterrupt)):
                    raise
                committed = receipt.get("published", False)
                result = replace(result, published=committed,
                    published_generation=receipt.get("generation") if committed else None,
                    prepublish_disk_validation=receipt.get("disk_validated", False),
                    cleanup_failed=receipt.get("cleanup_failed", False),
                    cancelled=isinstance(exc, KeyboardInterrupt),
                    error_code="CANCELLED" if isinstance(exc, KeyboardInterrupt) else
                        (exc.code if isinstance(exc, _Failure) else
                         ("PUBLICATION_FAILED" if stage == "ARTIFACT_FAILED" and receipt.get("disk_validated") else stage)))
                if committed:
                    if isinstance(exc, KeyboardInterrupt):
                        # Disk readback completed before commit. Cancellation alone
                        # must not pretend that an activated model was unpublished.
                        pass
                    elif stage == "POST_PUBLISH_VALIDATION_FAILED":
                        result = _after_failed_verification(result, model_dir, previous)
                    else:
                        try:
                            _verify(model_dir, result.published_generation)
                            result = replace(result, postpublish_validation=True)
                        except KeyboardInterrupt:
                            result = replace(result, error_code="CANCELLED", cancelled=True)
                        except Exception:
                            result = _after_failed_verification(result, model_dir, previous)
    except KeyboardInterrupt:
        result = replace(result, error_code="CANCELLED", cancelled=True)
    except Exception as exc:
        result = replace(result, error_code=exc.code if isinstance(exc, _Failure) else stage)
    try:
        return replace(result, report_path=_report(result, config_summary))
    except KeyboardInterrupt:
        return replace(result, error_code="CANCELLED", cancelled=True)
    except Exception:
        return replace(result, error_code="REPORT_FAILED")
