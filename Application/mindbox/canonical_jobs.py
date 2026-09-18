"""Sequential API transport into canonical storage, with durable resume receipts."""

from dataclasses import asdict
from datetime import datetime
import json
import hashlib
from pathlib import Path
import tempfile
import uuid

from .canonical_storage import atomic_json, catalog, collect_unreferenced, merge_target, publish, storage_lock
from .daily_training_batch import _poll_settings, split_daily_windows
from .selection import DEFAULT_SELECTION, MindboxSelectionConfig
from .storage import RawExportStorage
from .training_batch import TrainingBatchError, TrainingBatchWindow, _utc


def _fingerprint(config):
    text = json.dumps([config.api_url, config.endpoint_id, dict(config.operations)], sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _bind(root, client):
    path = root / "canonical/identity.json"
    fingerprint = _fingerprint(client.config)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            if json.load(stream) != {"fingerprint": fingerprint}:
                raise TrainingBatchError("Canonical store belongs to another endpoint/operation configuration")
    else:
        atomic_json(path, {"fingerprint": fingerprint})


def monthly_windows(since, until):
    since, until = _utc(since), _utc(until)
    if since >= until or any(value.second or value.microsecond for value in (since, until)):
        raise TrainingBatchError("Ordered customer period required")
    result = []
    while since < until:
        next_month = since.replace(year=since.year + (since.month == 12), month=since.month % 12 + 1,
                                   day=1, hour=0, minute=0, second=0, microsecond=0)
        end = min(next_month, until)
        result.append((since, end))
        since = end
    return result


def download(client, name, since, until, root, timeout, poll_interval):
    operation = client.config.operations[name]
    payload = {"sinceDateTimeUtc": since.strftime("%Y-%m-%d %H:%M"), "tillDateTimeUtc": until.strftime("%Y-%m-%d %H:%M")}
    export_id = client.start_export(operation, payload)
    urls = client.wait_for_export(operation, export_id, timeout=timeout, poll_interval=poll_interval)
    parts = client.download_export(name, urls, storage=RawExportStorage(root))
    if not parts or len({Path(p).resolve().parent for p in parts}) != 1:
        raise TrainingBatchError("Invalid downloaded parts")
    directory = Path(parts[0]).resolve().parent
    if not directory.is_relative_to(Path(root).resolve()):
        raise TrainingBatchError("Downloaded parts outside staging")
    return directory, export_id, operation


def create_job(client, *, raw_root, window=None, since=None, until=None, selection=DEFAULT_SELECTION,
               customers=False, on_state_created=None, progress=None, timeout=14400., poll_interval=5.):
    _poll_settings(poll_interval, timeout)
    root = Path(raw_root).resolve()
    with storage_lock(root):
        _bind(root, client)
        if customers:
            windows = [("customers", a, b) for a, b in monthly_windows(since, until)]
        else:
            TrainingBatchWindow(**asdict(window))
            days = split_daily_windows(window.interaction_since, window.interaction_until)
            a, b = merge_target(root, window.merge_since, window.interaction_until)
            windows = [("customer_merges", a, b)] + [(name, day.since, day.until) for day in days for name in ("actions", "orders")]
        job = {"schema_version": 1, "kind": "customers" if customers else "interactions", "id": uuid.uuid4().hex,
               "fingerprint": _fingerprint(client.config), "selection": asdict(selection),
               "components": [{"name": name, "since": a.isoformat(), "until": b.isoformat(), "done": False}
                              for name, a, b in windows]}
        path = root / "canonical/jobs" / job["id"] / "state.json"
        atomic_json(path, job)
    if on_state_created:
        on_state_created(path)
    return resume_job(client, raw_root=root, state_path=path, timeout=timeout, poll_interval=poll_interval, progress=progress)


def load_job(path, root):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        job = json.load(stream)
    if (path != Path(root).resolve() / "canonical/jobs" / job["id"] / "state.json"
            or len(job["id"]) != 32 or any(c not in "0123456789abcdef" for c in job["id"])
            or job["schema_version"] != 1 or job["kind"] not in ("customers", "interactions")):
        raise TrainingBatchError("Invalid canonical job")
    MindboxSelectionConfig(**job["selection"])
    for item in job["components"]:
        a, b = datetime.fromisoformat(item["since"]), datetime.fromisoformat(item["until"])
        if _utc(a) >= _utc(b) or type(item["done"]) is not bool:
            raise TrainingBatchError("Invalid job component")
    return job


def resume_job(client, *, raw_root, state_path, timeout=14400., poll_interval=5., progress=None):
    _poll_settings(poll_interval, timeout)
    root = Path(raw_root).resolve()
    with storage_lock(root):
        _bind(root, client)
        job = load_job(state_path, root)
        if job["fingerprint"] != _fingerprint(client.config):
            raise TrainingBatchError("Endpoint/operations differ from saved job")
        collect_unreferenced(root)
        for index, item in enumerate(job["components"]):
            name = item["name"]
            if item["done"]:
                continue
            a, b = datetime.fromisoformat(item["since"]), datetime.fromisoformat(item["until"])
            # Publication receipts survive a process kill between commit and checkpoint.
            if name == "customers":
                from .canonical_customers import month_committed, apply_month
                committed = month_committed(root, job["id"], index)
            else:
                data = catalog(root)
                entry = data[name] if name == "customer_merges" else data[name].get(a.date().isoformat())
                committed = entry is not None and entry["job"] == job["id"]
            if not committed:
                if name == "customer_merges":
                    a, b = merge_target(root, a, b)
                if progress:
                    progress(name, -1, a.date().isoformat())
                with tempfile.TemporaryDirectory(prefix=".transport-", dir=root / "canonical") as temporary:
                    directory, export_id, operation = download(client, name, a, b, temporary, timeout, poll_interval)
                    if name == "customers":
                        apply_month(root, directory, a, b, job=job["id"], index=index)
                    else:
                        publish(root, name, a, b, directory, export_id=export_id, operation=operation,
                                selection=MindboxSelectionConfig(**job["selection"]), job=job["id"])
            item["done"] = True
            atomic_json(state_path, job)
            if progress:
                progress(name, sum(c["done"] for c in job["components"] if c["name"] == name),
                         sum(c["name"] == name for c in job["components"]))
        return root / "canonical" / ("customers.sqlite" if job["kind"] == "customers" else "training.json")
