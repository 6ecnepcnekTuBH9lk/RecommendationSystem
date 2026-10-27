"""Small, aggregate-only GUI history. No torch import or canonical ingestion."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from Application.mindbox.canonical_storage import atomic_json
from Application.mindbox.daily_training_batch import _writer_lock

VERSION = 1
RESEARCH_FIXED = {"seed": 42, "embedding_dim": 128, "batch_size": 256, "n_neg": 10,
                  "lr": .0003, "bpr_reg": .0005, "weight_decay": 0.,
                  "early_stop": False, "use_item_features": False}
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
CONFIG_KEYS = {
    "w_view_item", "w_favorite", "w_purchase", "epochs", "seed", "embedding_dim",
    "batch_size", "n_neg", "lr", "bpr_reg", "weight_decay", "early_stop", "use_item_features",
}


class HistoryError(ValueError):
    """Existing history is unreadable; never overwrite it silently."""


def now():
    return datetime.now(timezone.utc).isoformat()


def clean_record(record):
    """Allowlist scalar aggregates; never copy arbitrary logs, exceptions or payloads."""
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise HistoryError("Invalid run identifier")
    status = record.get("status")
    if status not in TERMINAL | {"running"}:
        raise HistoryError("Invalid run status")
    result = {"run_id": run_id, "status": status}
    for name in ("started_at", "finished_at"):
        if record.get(name):
            result[name] = datetime.fromisoformat(record[name]).isoformat()
    for name in ("epochs_requested", "epochs_completed", "training_users", "training_items", "training_pairs", "pid"):
        value = record.get(name)
        if type(value) is int and value >= 0:
            result[name] = value
        else:
            result[name] = None
    for name in ("total_seconds", "training_seconds"):
        value = record.get(name)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            result[name] = value
    for name in ("device", "torch_version", "cuda_build"):
        value = record.get(name)
        if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9.+_: -]{1,80}", value):
            result[name] = value
        else:
            result[name] = None
    result["hyperparameters"] = {
        k: v for k, v in record.get("hyperparameters", {}).items()
        if k in CONFIG_KEYS and type(v) in (bool, int, float) and math.isfinite(v)
    }
    result["metrics"] = {}
    for label in ("overall", "VIEW", "FAVORITE", "PURCHASE"):
        for k in ("5", "10", "20"):
            cell = record.get("metrics", {}).get(label, {}).get(k, {})
            safe = {name: value for name, value in cell.items()
                    if name in {"ndcg", "recall", "cases"} and type(value) in (int, float)
                    and math.isfinite(value) and value >= 0}
            if safe:
                result["metrics"].setdefault(label, {})[k] = safe
    # Fixed benchmark identity and dates, never a user-provided test metric.
    result["benchmark"] = {"id": "mindbox-validation-2025-11", "history_end": "2025-11-01T00:00:00Z",
                           "validation_end": "2025-12-01T00:00:00Z", "min_history_events": 10}
    provenance = record.get("provenance", {})
    head = provenance.get("git_head", "")
    result["provenance"] = {
        "git_head": head if isinstance(head, str) and re.fullmatch(r"[0-9a-f]{40,64}", head) else None,
        "working_tree_dirty": provenance.get("working_tree_dirty") if type(provenance.get("working_tree_dirty")) is bool else None,
    }
    result["artifact"] = f"runs/{run_id}/result.json"
    error = record.get("error_summary")
    if error in {"CANCELLED", "INTERRUPTED", "FAILED_TO_START", "RUN_FAILED", "BENCHMARK_UNAVAILABLE", "INVALID_CONFIG"}:
        result["error_summary"] = error
    return result


class ExperimentHistory:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.path = self.directory / "history.json"

    def read(self):
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data["schema_version"] != VERSION or not isinstance(data["runs"], list):
                raise ValueError
            records = [clean_record(r) for r in data["runs"]]
            if len({r["run_id"] for r in records}) != len(records):
                raise ValueError
            return sorted(records, key=lambda r: r.get("started_at", ""), reverse=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise HistoryError("История экспериментов повреждена или недоступна; исходный файл сохранён.") from None

    def upsert(self, record):
        record = clean_record(record)
        self.directory.mkdir(parents=True, exist_ok=True)
        with _writer_lock(self.directory):
            records = {r["run_id"]: r for r in self.read()}
            records[record["run_id"]] = record
            atomic_json(self.path, {"schema_version": VERSION, "runs": sorted(
                records.values(), key=lambda r: r.get("started_at", ""), reverse=True)})

    def recover(self, alive):
        """Reconcile finished worker artifacts, then mark dead workers interrupted."""
        for record in self.read():
            if record["status"] != "running":
                continue
            artifact = self.directory / record["artifact"]
            try:
                completed = clean_record(json.loads(artifact.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError, AttributeError):
                completed = None
            if completed and completed["run_id"] == record["run_id"] and completed["status"] in TERMINAL:
                self.upsert(completed)
            elif not alive(record.get("pid")):
                self.upsert({**record, "status": "interrupted", "finished_at": now(), "error_summary": "INTERRUPTED"})
        return self.read()
