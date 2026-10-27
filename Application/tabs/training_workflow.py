"""Cheap metadata readiness and four GUI values; no trainer import in the GUI."""
import json
import math
from pathlib import Path

from Application.mindbox.canonical_storage import current_batch
from Application.evaluation.experiments.gui_benchmark import require_gui_coverage
from Application.model.item_feature_config import normalize_feature_norm


def input_values(window):
    for widget in (window.w_purchase, window.w_favorite, window.w_view_item, window.epochs_input):
        if hasattr(widget, "hasAcceptableInput") and not widget.hasAcceptableInput():
            raise ValueError("Завершите ввод корректных весов и количества эпох.")
    values = {name: getattr(window, name).value() for name in ("w_purchase", "w_favorite", "w_view_item")}
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name == "w_purchase" and value == 0):
            raise ValueError("Вес покупки должен быть больше нуля, остальные веса — неотрицательными и конечными.")
    epochs = window.epochs_input.value()
    if type(epochs) is not int or not 1 <= epochs <= 10000:
        raise ValueError("Количество эпох должно быть целым числом от 1 до 10000.")
    return {**values, "epochs": epochs}


def metadata_readiness(data_dir, research):
    root = Path(data_dir) / "MindboxRaw"
    try:
        if not (Path(data_dir) / "nomenclature.csv").is_file():
            return False, "Нет каталога товаров для обучения", None
        manifest = json.loads((root / "canonical/training.json").read_text(encoding="utf-8"))
        if manifest != {"schema_version": 5, "storage": "canonical"}:
            return False, "Проверка metadata завершилась с ошибкой", None
        batch = current_batch(root)  # Metadata and part filenames only; no raw reads.
        product_stat = (Path(data_dir) / "nomenclature.csv").stat()
        revision = (batch.batch_id, product_stat.st_mtime_ns, product_stat.st_size)
        if research:
            try:
                require_gui_coverage(batch.window.interaction_since, batch.window.interaction_until)
            except ValueError:
                return False, "Нет непрерывного покрытия истории с января 2022 по август 2026 для эксперимента", revision
        return True, "Данные готовы · полная проверка перед обучением", revision
    except FileNotFoundError:
        return False, "Canonical batch или данные для обучения отсутствуют", None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False, "Проверка metadata завершилась с ошибкой", None


def production_config(values, settings_dir, data_dir):
    path = Path(settings_dir) / "train_config.json"
    settings = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    if not isinstance(settings, dict):
        raise ValueError("Некорректные настройки рабочей модели")
    # The CLI's existing typed TrainConfig loader validates hidden fields and
    # supplies defaults. Existing early-stop/feature settings stay intact.
    result = {**settings, **values, "data_dir": str(data_dir)}
    # Validate hidden settings before the GUI starts a process; no ML imports.
    if "feature_norm" in result:
        result["feature_norm"] = normalize_feature_norm(result["feature_norm"])
    return result
