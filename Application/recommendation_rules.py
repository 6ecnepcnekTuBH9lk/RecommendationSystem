"""Independent, explicit recommendation settings; never read analysis filters."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from Application.paths import USER_SETTINGS_DIR


RULES_PATH = USER_SETTINGS_DIR / "recommendation_rules.json"


@dataclass(frozen=True, slots=True)
class RecommendationRules:
    kinds: tuple[str, ...]
    collections: tuple[str, ...]
    minimum_stock: int  # Zero disables the extra threshold, never the in-stock gate.
    # Explicit user priority; never inferred from the calendar.
    primary_collection: str | None = None

    def __post_init__(self):
        for values in (self.kinds, self.collections):
            if (not isinstance(values, tuple) or not values
                    or any(not isinstance(v, str) or not v or v != v.strip() for v in values)
                    or len(set(values)) != len(values)):
                raise ValueError("Выберите виды номенклатуры и актуальные сезоны.")
        if type(self.minimum_stock) is not int or self.minimum_stock < 0:
            raise ValueError("Допустимый остаток должен быть неотрицательным целым.")
        if self.primary_collection is not None and self.primary_collection not in self.collections:
            raise ValueError("Основная коллекция должна входить в выбранные сезоны.")

    def validate_catalog(self, catalog):
        if set(self.kinds) - catalog.kinds or set(self.collections) - catalog.collections:
            raise ValueError("Выбранные виды или сезоны удалены из каталога. Обновите настройки.")


def load_rules(path=RULES_PATH):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if (not isinstance(value, dict) or type(value.get("version")) is not int or value.get("version") != 1
            or not {"version", "kinds", "collections", "minimum_stock"} <= set(value)
            or set(value) - {"version", "kinds", "collections", "minimum_stock", "primary_collection"}
            or not isinstance(value["kinds"], list) or not isinstance(value["collections"], list)):
        raise ValueError("Некорректная версия настроек рекомендаций.")
    return RecommendationRules(tuple(value["kinds"]), tuple(value["collections"]),
                               value["minimum_stock"], value.get("primary_collection"))


def save_rules(rules, path=RULES_PATH):
    if not isinstance(rules, RecommendationRules):
        raise TypeError("Expected recommendation rules")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".recommendation-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"version": 1, "kinds": rules.kinds, "collections": rules.collections,
                       "minimum_stock": rules.minimum_stock, "primary_collection": rules.primary_collection},
                      stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
