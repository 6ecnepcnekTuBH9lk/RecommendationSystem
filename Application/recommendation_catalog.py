"""Snapshot of current CSV references for recommendation matching only."""

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType

from Application.paths import INPUT_DATA_DIR


FIELDS = ("Номенклатура", "НазваниеНаСайте", "ВидНоменклатуры", "ВидАссортимента", "Марка",
          "Коллекция", "ПолНоменклатуры", "ГруппаСоставов", "КатегорияНаСайте", "СезонНоски",
          "СтилеваяГруппа", "ТитульнаяФотография")


def text(value):
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in ("nan", "none", "null", "<na>") else value


def signature(path):
    stat = Path(path).stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def read_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|", strict=True)
        header = reader.fieldnames
        if (not header or len(set(header)) != len(header) or not set(required) <= set(header)
                or any(not h for h in header)):
            raise ValueError("Некорректные колонки справочника.")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Некорректная строка справочника.")
            yield row


def stock(value):
    try:
        number = Decimal(value.replace(" ", "").replace("\xa0", "").replace("\u202f", "").replace(",", "."))
        return number if number.is_finite() and number >= 0 else None
    except (InvalidOperation, AttributeError):
        return None


@dataclass(frozen=True, slots=True)
class CatalogItem:
    code: str
    attributes: Mapping[str, str]
    stock: Decimal | None

    def get(self, key):
        return self.attributes.get(key, "")

    @property
    def name(self):
        return self.get("НазваниеНаСайте") or self.get("Номенклатура")


def item_from_metadata(code, values):
    return CatalogItem(code, MappingProxyType({field: text(values.get(field)) for field in FIELDS}), None)


class RecommendationCatalog:
    def __init__(self, path=INPUT_DATA_DIR / "nomenclature.csv", categories_path=INPUT_DATA_DIR / "site_categories.csv"):
        self.path, self.categories_path = Path(path), Path(categories_path)
        self._key = None

    def current(self):
        return self._key == (signature(self.path), signature(self.categories_path))

    def load(self):
        key = (signature(self.path), signature(self.categories_path))
        if key == self._key:
            return self
        items, ambiguous, duplicates = {}, set(), 0
        for row in read_rows(self.path, ("КодНоменклатуры", "ВидНоменклатуры", "Коллекция", "Остаток")):
            code = row["КодНоменклатуры"]
            if not code:
                continue
            if code != code.strip():
                raise ValueError("Некорректный код номенклатуры.")
            item = CatalogItem(code, MappingProxyType({field: text(row.get(field)) for field in FIELDS}), stock(row["Остаток"]))
            if code in items:
                duplicates += 1
                if items[code] != item:
                    ambiguous.add(code)
            else:
                items[code] = item
        parents, labels, bad_categories = {}, {}, set()
        for row in read_rows(self.categories_path, ("КодКатегории", "НазваниеКатегории", "КодРодительскойКатегории")):
            code, label, parent = (text(row[field]) for field in (
                "КодКатегории", "НазваниеКатегории", "КодРодительскойКатегории"))
            if not code:
                continue
            if code in parents and (parents[code], labels[code]) != (parent, label):
                bad_categories.add(code)
            else:
                parents[code], labels[code] = parent, label
        paths = {}
        for code in parents:
            trail, cursor = [], code
            while cursor:
                if cursor in trail or cursor in bad_categories or cursor not in parents:
                    trail = []
                    break
                trail.append(cursor)
                cursor = parents[cursor]
            if trail:
                paths[code] = frozenset(trail)
        by_label = {}
        for code, label in labels.items():
            by_label.setdefault(label, []).append(code)
        # Only uniquely named categories can safely be resolved from a label.
        category_paths = {code: path for code, path in paths.items()}
        for label, codes in by_label.items():
            if label and len(codes) == 1 and label not in parents and codes[0] in paths:
                category_paths[label] = paths[codes[0]]
        if key != (signature(self.path), signature(self.categories_path)):
            raise ValueError("Справочники изменились во время чтения.")
        self.items = MappingProxyType({c: i for c, i in items.items() if c not in ambiguous})
        self.category_paths = MappingProxyType(category_paths)
        # Ambiguous codes still resolve identity, but cannot become targets.
        self.all_codes = frozenset(items)
        self.kinds = frozenset(i.get("ВидНоменклатуры") for i in self.items.values()) - {""}
        self.collections = frozenset(i.get("Коллекция") for i in self.items.values()) - {""}
        self.diagnostics = {"duplicate_rows": duplicates, "ambiguous_codes": len(ambiguous),
                            "invalid_stock": sum(i.stock is None for i in self.items.values()),
                            "invalid_categories": len(parents) - len(paths)}
        self._key = key
        return self
