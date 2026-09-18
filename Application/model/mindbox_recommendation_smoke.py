"""Offline acceptance in a disposable cwd. Run only in a dedicated CLI process.

The process-wide cwd/stdout scopes are intentionally unsuitable for GUI workers.
Only the existing serializer/exporter writes artifacts, all below the temp root.
"""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
import zipfile

from Application.mindbox.daily_training_batch import load_chunked_training_batch, prepare_training_data_from_chunked_batch
from Application.mindbox.customer_profile_snapshot import load_customer_contact_index, load_customer_profile_snapshot
from . import BPRMF as core
from .interaction_analytics import analytics_from_checkpoint
from .seen_items import build_seen_items_index, seen_items_from_checkpoint
from .mindbox_shadow_training import _atomic_report
from .training_quality import TrainingQualityDiagnostics, evaluate_training_quality

REPORT_ROOT = Path(__file__).resolve().parents[2] / "ВходныеДанные" / "MindboxReports" / "recommendation_smoke"
INTERACTION_FILES = ("Заказы.csv", "Просмотры.csv", "Избранное.csv")


def _tree_state(root):
    """Compare listings and file bytes without retaining model contents in memory."""
    if not root.exists():
        return None
    state = {}
    for path in sorted(root.rglob("*")):
        digest = None
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            digest = h.digest()
        state[path.relative_to(root).as_posix()] = digest
    return state


@contextmanager
def _working_directory(root):
    previous = Path.cwd()
    try:
        os.chdir(root)
        yield
    finally:
        os.chdir(previous)


def _validate_outputs(root, k):
    xlsx = root / "recommendations.xlsx"
    core._validate_export_xlsx(str(xlsx))
    header = ["MindboxID", "ДисконтнаяКарта", "Почта", "Телефон"]
    for rank in range(1, k + 1):
        header.extend(f"{name}_{rank}" for name in (
            "КодНоменклатуры", "НазваниеНоменклатуры", "Коллекция", "Коэффициент", "Конверсия", "Остаток"))
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows = 0
    # The production writer uses inline strings. Inspect only header values;
    # subsequent cells are cleared without extracting their contact/item text.
    with zipfile.ZipFile(xlsx) as archive, archive.open("xl/worksheets/sheet1.xml") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag == ns + "row":
                if rows == 0 and [node.text or "" for node in element.iter(ns + "t")] != header:
                    raise ValueError("Invalid XLSX header")
                rows += 1
                element.clear()
            elif rows > 0:
                element.clear()
    if rows == 0:
        raise ValueError("Missing XLSX header")
    counts = {"xlsx_rows": rows - 1}
    for name, expected, key in (
        ("format1.csv", ["CustomerID", "ProductID"], "format1_rows"),
        ("Kanzler.ML.csv", ["CustomerMindboxId", "Quantity", "ProductGroupOffline1C", "CustomFieldKoefficient"], "kanzler_ml_rows"),
    ):
        path = root / name
        core._validate_export_csv(str(path), expected)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter=";")
            next(reader)
            # Parse logical CSV records, never inspect or log field values.
            counts[key] = sum(1 for _ in reader)
    if counts["format1_rows"] != counts["xlsx_rows"] or not (
        counts["xlsx_rows"] <= counts["kanzler_ml_rows"] <= counts["xlsx_rows"] * k
    ):
        raise ValueError("Inconsistent export counts")
    return counts


def run_recommendation_smoke(training_manifest, profile_manifest, *, raw_root, catalog_path,
                             cfg, device, max_export_users=100, settings_source=None, progress=None):
    """Return (safe aggregate report, report path); exceptions never expose inputs."""
    if type(max_export_users) is not int or max_export_users < 1 or cfg.epochs < 1:
        raise ValueError("Positive smoke limits required")
    source_cwd = Path.cwd()
    training_manifest, profile_manifest, raw_root, catalog_path = (
        Path(p).resolve() for p in (training_manifest, profile_manifest, raw_root, catalog_path))
    settings_source = Path(settings_source or source_cwd / "Настройки").resolve()
    production_root = source_cwd / "Модель"
    report_root = REPORT_ROOT.resolve()
    run_id = uuid.uuid4().hex
    report = {"schema_version": 1, "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
              "status": "FAILED", "error_code": None,
              "artifact": {"saved": False, "reloaded": False, "embedded_seen": False,
                           "embedded_analytics": False, "production_model_published": False},
              "training": {"epochs_requested": cfg.epochs, "epochs_completed": 0},
              "export": {"requested_max_users": max_export_users, "filter_seen": True,
                         "interaction_csv_present_in_smoke_env": False, "completed": False},
              "cleanup": {"temporary_artifacts_removed": False}}
    temporary = None
    before = None
    state_captured = False
    stage = "PREPARATION_FAILED"

    def announce(message):
        if progress is not None:
            progress(message)

    try:
        before = _tree_state(production_root)
        state_captured = True
        temporary = tempfile.TemporaryDirectory(prefix="mindbox-recommendation-smoke-")
        temp_root = Path(temporary.name).resolve()
        data_dir = temp_root / "ВходныеДанные"
        data_dir.mkdir()
        shutil.copyfile(catalog_path, data_dir / "Номенклатура.csv")
        if settings_source.exists():
            shutil.copytree(settings_source, temp_root / "Настройки")
        temp_cfg = replace(cfg, data_dir=str(data_dir))
        # Do not retain arbitrary legacy logs in memory/on disk or leak paths/PII.
        with open(os.devnull, "w") as sink, _working_directory(temp_root), redirect_stdout(sink), redirect_stderr(sink):
            announce("Preparation: STARTED")
            batch = load_chunked_training_batch(training_manifest, raw_root=raw_root, require_complete=True)
            report["training_batch_id"] = batch.batch_id
            prepared = prepare_training_data_from_chunked_batch(batch, raw_root=raw_root,
                catalog_path=data_dir / "Номенклатура.csv", train_config=temp_cfg, diagnose=True)
            d = prepared.diagnostics
            quality = evaluate_training_quality(prepared.prepared_data, TrainingQualityDiagnostics(
                actions_view=d.actions_view, actions_favorite=d.actions_favorite,
                malformed_mapped_actions=d.malformed_actions, unresolved_products=d.resolution.total.unresolved,
                unsupported_products=d.resolution.total.unsupported_namespace, bpr_events=d.bpr.events_total,
                unmapped_actions=d.unmapped_actions, malformed_action_system_names=d.malformed_action_system_names,
                orders_duplicate_conflicting=d.orders_duplicate_conflicting))
            report["quality"] = {"level": quality.level.value, "training_allowed": quality.training_allowed,
                                 "issues": [{"code": i.code, "count": i.count} for i in quality.issues]}
            report["dataset"] = {**dict(quality.metrics), "complete": prepared.complete}
            announce(f"Quality: {quality.level.value}; training_allowed={quality.training_allowed}")
            if not quality.training_allowed:
                report["error_code"] = "QUALITY_BLOCK"
            else:
                stage = "TRAINING_FAILED"
                announce("Training: STARTED")
                core._set_seed(temp_cfg.seed)
                model, _, metrics = core.train_prepared_data_with_metrics(temp_cfg, prepared.prepared_data, device)
                report["training"].update({name: getattr(metrics, name) for name in (
                    "epochs_completed", "best_epoch", "best_recall", "best_ndcg", "early_stopped")})
                stage = "ARTIFACT_FAILED"
                announce("Artifact save/reload: STARTED")
                core._save_artifacts(temp_cfg, prepared.prepared_data.mappings, model,
                    seen_items=build_seen_items_index(prepared.prepared_data), analytics=prepared.prepared_data.analytics)
                report["artifact"]["saved"] = True
                mappings, checkpoint = core._load_artifacts(str(temp_root / "Модель"))
                report["artifact"]["reloaded"] = True
                seen = seen_items_from_checkpoint(checkpoint)
                analytics = analytics_from_checkpoint(checkpoint)
                if seen is None or analytics is None:
                    raise ValueError("Missing embedded inference data")
                dimensions = (len(mappings["idx2user"]), len(mappings["idx2item"]))
                if any((obj.num_users, obj.num_items) != dimensions for obj in (seen, analytics)):
                    raise ValueError("Invalid embedded dimensions")
                report["artifact"].update(embedded_seen=True, embedded_analytics=True)
                stage = "CONTACTS_FAILED"
                announce("Contacts: STARTED")
                snapshot = load_customer_profile_snapshot(profile_manifest, raw_root=raw_root)
                merges = next(c.export for c in batch.components if c.name == "customer_merges")
                if (snapshot.customer_merges_directory, snapshot.customer_merges_parts) != (merges.relative_directory, merges.parts_count):
                    raise ValueError("Profile and training merge sources differ")
                report["customer_profile_snapshot_id"] = snapshot.snapshot_id
                contacts = load_customer_contact_index(profile_manifest, mappings, raw_root=raw_root,
                    progress=lambda count: announce(f"Customers processed: {count}"))
                report["contacts"] = {name: getattr(contacts.diagnostics, name) for name in (
                    "model_users", "matched_profiles", "missing_profiles", "with_email", "with_phone", "with_card", "ambiguous_discount_cards")}
                stage = "EXPORT_FAILED"
                present = any((data_dir / name).exists() for name in INTERACTION_FILES)
                report["export"]["interaction_csv_present_in_smoke_env"] = present
                if present:
                    raise ValueError("Interaction CSV isolation failed")
                announce("Export: STARTED")
                core.export_recommendations_excel(out_xlsx=str(temp_root / "recommendations.xlsx"),
                    model_dir=str(temp_root / "Модель"), customer_contacts=contacts, device_str="cpu", filter_seen=True,
                    max_export_users=max_export_users, out_csv_format1=str(temp_root / "format1.csv"),
                    out_csv_kanzler_ml=str(temp_root / "Kanzler.ML.csv"))
                stage = "VALIDATION_FAILED"
                k = max(1, min(max(10, temp_cfg.topk), dimensions[1]))
                counts = _validate_outputs(temp_root, k)
                if counts["xlsx_rows"] > max_export_users:
                    raise ValueError("Export user limit exceeded")
                report["export"].update(counts, completed=True)
                report["status"] = "COMPLETED"
    except KeyboardInterrupt:
        report.update(status="CANCELLED", error_code="CANCELLED")
    except Exception:
        report["error_code"] = stage
    finally:
        if temporary is not None:
            try:
                temporary.cleanup()
                report["cleanup"]["temporary_artifacts_removed"] = not Path(temporary.name).exists()
                if not report["cleanup"]["temporary_artifacts_removed"]:
                    report.update(status="FAILED", error_code="CLEANUP_FAILED")
            except (Exception, KeyboardInterrupt):
                report.update(status="FAILED", error_code="CLEANUP_FAILED")
        else:
            report["cleanup"]["temporary_artifacts_removed"] = True
        if state_captured:
            try:
                unchanged = _tree_state(production_root) == before
            except Exception:
                unchanged = False
            report["artifact"]["production_model_unchanged"] = unchanged
            if not unchanged:
                report.update(status="FAILED", error_code="PRODUCTION_ISOLATION_FAILED")
    report_path = report_root / run_id / "report.json"
    _atomic_report(report_path, report)
    return report, report_path
