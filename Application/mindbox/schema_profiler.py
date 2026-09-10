"""Raw JSON -> статистика схемы. Значения не сохраняются, кроме systemName действий."""

import html
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import raw_reader
from .raw_reader import DEFAULT_RAW_ROOT, EXPORT_ROOTS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "ВходныеДанные" / "MindboxReports"
FOCUS_FIELDS = {
    "actions": ("actionTemplate", "customer", "product", "products", "productCategories", "order", "customFields"),
    "orders": ("ids", "firstAction", "customer", "lines", "payments", "appliedPromotions",
               "bonusPointsInfoPerBalanceTypes"),
    "customers": ("ids", "sex", "birthDate", "changeDateTimeUtc", "customFields", "lastActivatedCard",
                  "segmentations", "balances", "discountCards", "subscriptions", "email", "mobilePhone",
                  "firstName", "lastName", "middleName"),
    "customer_merges": ("id", "dateTimeUtc", "method", "resultingCustomer", "mergedCustomers"),
}


class SchemaProfileError(Exception):
    """Безопасная диагностика без исходного JSON, его значений и произвольных путей."""


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"  # Остальные значения стандартного JSON — dict.


def _field_path(parent: str, name: str) -> str:
    # Отличаем ключ "a.b" от вложенных a/b и ключ "x[]" от массива x.
    return f"{parent}.{name}" if name.isidentifier() else parent + "[" + json.dumps(name, ensure_ascii=False) + "]"


@dataclass
class _PathStats:
    types: Counter = field(default_factory=Counter)
    occurrences: int = 0
    present_records: int = 0
    null_records: int = 0
    last_record: int = -1
    last_null_record: int = -1
    array_count: int = 0
    empty_arrays: int = 0
    min_length: int | None = None
    max_length: int | None = None
    object_fields: set[str] = field(default_factory=set)
    first_fields: frozenset[str] | None = None
    object_fields_vary: bool = False

    def observe(self, value: Any, record: int) -> None:
        kind = _type_name(value)
        self.types[kind] += 1
        self.occurrences += 1
        if self.last_record != record:
            self.present_records += 1
            self.last_record = record
        if value is None and self.last_null_record != record:
            self.null_records += 1
            self.last_null_record = record
        if isinstance(value, dict):
            keys = frozenset(value)
            if self.first_fields is None:
                self.first_fields = keys
            elif keys != self.first_fields:
                self.object_fields_vary = True
            self.object_fields.update(keys)
        if isinstance(value, list):
            length = len(value)
            self.array_count += 1
            self.empty_arrays += int(length == 0)
            self.min_length = length if self.min_length is None else min(self.min_length, length)
            self.max_length = length if self.max_length is None else max(self.max_length, length)

    def summary(self, denominator: int, *, scope: str = "export_objects") -> dict[str, Any]:
        reasons = []
        if self.present_records < denominator:
            reasons.append("optional_in_export")
        if len(self.types) > 1:
            reasons.append("multiple_types")
        if self.object_fields_vary:
            reasons.append("different_object_fields")
        if self.min_length != self.max_length:
            reasons.append("different_array_lengths")
        return {
            "scope": scope, "types": sorted(self.types), "type_counts": dict(sorted(self.types.items())),
            "occurrences": self.occurrences, "present_records": self.present_records,
            "total_records": denominator,
            "presence_percent": round(100 * self.present_records / denominator, 2) if denominator else 0.0,
            "null_count": self.types["null"], "null_records": self.null_records,
            "array": {"observed": self.array_count > 0, "count": self.array_count,
                      "empty_count": self.empty_arrays, "min_length": self.min_length, "max_length": self.max_length},
            "object_fields": sorted(self.object_fields), "variation_reasons": reasons,
        }


def _part_files(directory: Path, export_name: str) -> list[Path]:
    try:
        return raw_reader.part_files(directory, export_name)
    except raw_reader.RawExportError as exc:
        raise SchemaProfileError(str(exc)) from None


def select_export_directory(
    export_name: str, *, raw_root: str | Path = DEFAULT_RAW_ROOT,
    input_dir: str | Path | None = None,
) -> Path:
    try:
        return raw_reader.select_export_directory(export_name, raw_root=raw_root, input_dir=input_dir)
    except raw_reader.RawExportError as exc:
        raise SchemaProfileError(str(exc)) from None


class _Profiler:
    def __init__(self, export_name: str) -> None:
        self.export_name = export_name
        self.root = EXPORT_ROOTS[export_name]
        self.nodes: dict[str, _PathStats] = {}
        self.total = 0
        self.custom_fields: set[str] = set()
        self.identifiers: set[str] = set()
        self.templates: Counter = Counter()
        self.template_entities: dict[str, dict[str, _PathStats]] = {}
        self.unclassified: Counter = Counter()

    def walk(self, value: Any, path: str, record: int) -> None:
        self.nodes.setdefault(path, _PathStats()).observe(value, record)
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = _field_path(path, key)
                if key == "customFields":
                    self.custom_fields.add(child_path)
                if key == "ids":
                    self.identifiers.add(child_path)
                self.walk(child, child_path, record)
        elif isinstance(value, list):
            for child in value:
                self.walk(child, path + "[]", record)

    def action(self, item: dict[str, Any], record: int) -> None:
        template = item.get("actionTemplate")
        ids = template.get("ids") if isinstance(template, dict) else None
        if not isinstance(ids, dict) or "systemName" not in ids:
            self.unclassified["missing"] += 1
            return
        name = ids["systemName"]
        if not isinstance(name, str):
            self.unclassified["null" if name is None else "invalid_type"] += 1
            return
        # Единственное разрешённое scalar value, переносимое в отчёт.
        self.templates[name] += 1
        entities = self.template_entities.setdefault(name, {})
        for key, value in item.items():
            if key not in ("ids", "actionTemplate"):
                entities.setdefault(key, _PathStats()).observe(value, record)

    def consume_part(self, path: Path, part_number: int) -> None:
        try:
            # M02-01 сохраняет прежние float-типы и формат статистики.
            items = raw_reader.load_export_part(path, self.export_name, part_number, parse_float=float)
            self.nodes.setdefault(self.root, _PathStats()).observe(items, part_number)
            for item in items:
                self.total += 1
                self.walk(item, self.root + "[]", self.total)
                if self.export_name == "actions":
                    self.action(item, self.total)
        except raw_reader.RawExportError as exc:
            raise SchemaProfileError(str(exc)) from None
        except RecursionError:
            raise SchemaProfileError(f"Часть {part_number}: превышена допустимая глубина JSON") from None

    def report(self, file_count: int) -> dict[str, Any]:
        paths = {
            path: stats.summary(file_count if path == self.root else self.total,
                                scope="part_files" if path == self.root else "export_objects")
            for path, stats in sorted(self.nodes.items())
        }
        focus = [_field_path(self.root + "[]", name) for name in FOCUS_FIELDS[self.export_name]]
        if self.export_name == "orders":
            focus += ["orders[].lines[]." + name for name in ("product", "status", "appliedPromotions", "customFields")]
        custom_paths = self.custom_fields | {
            "actions": {"customerActions[].customFields"},
            "customers": {"customers[].customFields"},
            "orders": {"orders[].customFields", "orders[].lines[].customFields"},
            "customer_merges": set(),
        }[self.export_name]
        custom = {}
        for path in sorted(custom_paths):
            stats = self.nodes.get(path)
            custom[path] = {
                "observed": stats is not None,
                "fields": {name: {key: paths[_field_path(path, name)][key]
                                  for key in ("types", "occurrences", "present_records", "null_count", "null_records")}
                           for name in sorted(stats.object_fields) if stats is not None} if stats else {},
            }
        templates = []
        for name, count in sorted(self.templates.items()):
            entities = self.template_entities[name]
            related = {key: stats.summary(count) for key, stats in sorted(entities.items())
                       if "object" in stats.types or "array" in stats.types or key in (
                           "product", "products", "productCategories", "order", "customer", "customFields")}
            templates.append({"system_name": name, "count": count,
                              "percent": round(100 * count / self.total, 2), "related_entities": related})
        return {
            "schema_version": 1, "export_name": self.export_name, "root_key": self.root,
            "file_count": file_count, "total_objects": self.total, "paths": paths,
            "variable_paths": [path for path, stats in paths.items() if stats["variation_reasons"]],
            "sections": {path: {"observed": path in paths,
                                "paths": [child for child in paths if _within(child, path)]} for path in focus},
            "custom_fields": custom,
            "identifier_namespaces": {path: sorted(self.nodes[path].object_fields) for path in sorted(self.identifiers)},
            "action_templates": templates,
            "actions_without_system_name": dict(sorted(self.unclassified.items())),
        }


def _within(path: str, parent: str) -> bool:
    return path == parent or path.startswith((parent + ".", parent + "["))


def profile_export(
    export_name: str, *, input_dir: str | Path | None = None, raw_root: str | Path = DEFAULT_RAW_ROOT,
) -> dict[str, Any]:
    directory = select_export_directory(export_name, raw_root=raw_root, input_dir=input_dir)
    profiler = _Profiler(export_name)
    parts = _part_files(directory, export_name)
    for number, path in enumerate(parts, start=1):
        profiler.consume_part(path, number)
    return profiler.report(len(parts))


def _cell(value: Any) -> str:
    return html.escape(str(value), quote=True).replace("|", "&#124;").replace("\n", "\\n").replace("\r", "\\r").replace("`", "&#96;")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Mindbox schema: {report['export_name']}", "",
             f"Part-файлов: {report['file_count']}. Объектов: {report['total_objects']}.", "",
             "Присутствие считается по исходным объектам экспорта, а не по элементам вложенных массивов.",
             "У корневого массива scope=part_files: знаменатель — число part-файлов.",
             "Occurrences и null — число вхождений; null_records — число объектов с хотя бы одним null.",
             "Пустой массив присутствует, но не создаёт path элементов []. Отсутствие поля не считается null.",
             "Реальные значения исключены; единственное исключение — systemName шаблона действия.", "",
             "## Все paths", "",
             "| Path | Types | Present / total | Presence % | Occurrences | Null | Null records | Arrays / empty / min / max | Variations |",
             "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for path, stats in report["paths"].items():
        array = stats["array"]
        arrays = f"{array['count']} / {array['empty_count']} / {array['min_length']} / {array['max_length']}" if array["observed"] else "—"
        row = (path, ", ".join(stats["types"]), f"{stats['present_records']} / {stats['total_records']}",
               stats["presence_percent"], stats["occurrences"], stats["null_count"], stats["null_records"],
               arrays, ", ".join(stats["variation_reasons"]))
        lines.append("| " + " | ".join(map(_cell, row)) + " |")
    lines += ["", "## Специальные разделы", ""]
    for path, section in report["sections"].items():
        lines += [f"### {_cell(path)}", "", "Встречается." if section["observed"] else "Не встречается.", ""]
        lines += ["- " + _cell(child) + ": " + _cell(", ".join(report["paths"][child]["types"])) for child in section["paths"]]
        lines.append("")
    lines += ["## Identifier namespaces", ""]
    for path, names in report["identifier_namespaces"].items():
        lines.append("- " + _cell(path) + ": " + _cell(", ".join(names)))
    lines += ["", "## Custom fields (без значений)", ""]
    for path, section in report["custom_fields"].items():
        lines += [f"### {_cell(path)}", "", "Встречается." if section["observed"] else "Не встречается.", "",
                  "| Name | Types | Occurrences | Present objects | Null | Null records |", "|---|---|---:|---:|---:|---:|"]
        for name, stats in section["fields"].items():
            lines.append("| " + " | ".join(map(_cell, (name, ", ".join(stats["types"]), stats["occurrences"],
                                                       stats["present_records"], stats["null_count"], stats["null_records"]))) + " |")
        lines.append("")
    if report["export_name"] == "actions":
        lines += ["## Action templates", "", "| SystemName | Count | Percent | Related entities (present / actions) |", "|---|---:|---:|---|"]
        for item in report["action_templates"]:
            entities = "; ".join(f"{key}: {stats['present_records']}/{item['count']}" for key, stats in item["related_entities"].items())
            lines.append("| " + " | ".join(map(_cell, (item["system_name"], item["count"], item["percent"], entities))) + " |")
        lines += ["", "Без строкового systemName: " + _cell(json.dumps(report["actions_without_system_name"], sort_keys=True)), ""]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: str | Path = DEFAULT_REPORT_ROOT) -> list[Path]:
    """Публикует каждый отчёт через временный файл; raw никогда не перезаписывает."""
    name = report["export_name"]
    if name not in EXPORT_ROOTS:
        raise SchemaProfileError("Неизвестный тип отчёта")
    destination = Path(output_dir)
    raw = DEFAULT_RAW_ROOT.resolve()
    resolved = destination.resolve()
    if resolved == raw or raw in resolved.parents:
        raise SchemaProfileError("Отчёты нельзя сохранять внутри MindboxRaw")
    payloads = {"json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                "md": render_markdown(report)}
    result = []
    try:
        destination.mkdir(parents=True, exist_ok=True)
        for extension, text in payloads.items():
            temporary = None
            try:
                target = destination / f"{name}_schema.{extension}"
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=destination,
                                                 prefix=".schema-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                result.append(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    except OSError:
        raise SchemaProfileError("Не удалось сохранить отчёты схемы") from None
    return result
