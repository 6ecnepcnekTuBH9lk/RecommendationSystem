"""One catalog read per training run; immutable feature and metadata payloads.

Coverage uses the trainer's aggregated positive pairs, not raw event counts.
No customer IDs, product IDs or attribute values are included in diagnostics.
"""

from dataclasses import dataclass, field
import json

import numpy as np


FEATURE_FIELDS = ("ВидНоменклатуры", "ВидАссортимента", "Марка", "Коллекция", "СезонНоски",
                  "ПолНоменклатуры", "ГруппаСоставов", "КатегорияНаСайте", "СтилеваяГруппа")
META_FIELDS = ("Коллекция", "НазваниеНаСайте", "Номенклатура", "ВидНоменклатуры",
               "ПолНоменклатуры", "КатегорияНаСайте", "СтилеваяГруппа", "Марка",
               "ГруппаСоставов", "ВидАссортимента", "Остаток", "СезонНоски")


def feature_parameters(cfg):
    return (bool(cfg.use_item_features), tuple(cfg.item_feature_cols), cfg.max_item_features,
            cfg.feature_scale, cfg.feature_dropout, cfg.feature_norm, cfg.feat_reg_mult)


def encode_features(catalog, item_codes, cfg):
    """Preserve the legacy exact join, last duplicate and token ordering."""
    matrix = np.full((len(item_codes), cfg.max_item_features), -1, dtype=np.int64)
    if not cfg.use_item_features or catalog is None or "КодНоменклатуры" not in catalog:
        return {}, matrix
    columns = [c for c in cfg.item_feature_cols if c in catalog]
    if not columns:
        return {}, matrix
    rows = catalog[["КодНоменклатуры"] + columns].copy()
    rows["КодНоменклатуры"] = rows["КодНоменклатуры"].astype(str)
    rows = rows.drop_duplicates("КодНоменклатуры", keep="last").set_index("КодНоменклатуры")
    vocab = {}
    for index, code in enumerate(item_codes):
        if code not in rows.index:
            continue
        row = rows.loc[code]
        tokens = []
        for column in columns:
            value = str(row[column]).strip() if row[column] is not None else ""
            if value and value.lower() not in ("nan", "none", "null", "-"):
                tokens.append(f"{column}={value}")
        for offset, token in enumerate(tokens[:cfg.max_item_features]):
            matrix[index, offset] = vocab.setdefault(token, len(vocab))
    return vocab, matrix


@dataclass(frozen=True, eq=False, repr=False)
class ItemFeatureSnapshot:
    item_codes: tuple
    parameters: tuple
    tokens: tuple
    matrix: np.ndarray = field(repr=False)
    metadata: tuple = field(repr=False)
    diagnostics_json: str

    def vocabulary(self):
        return dict(zip(self.tokens, range(len(self.tokens))))

    def historical_metadata(self):
        return {code: dict(values) for code, values in self.metadata}

    def diagnostics(self):
        return json.loads(self.diagnostics_json)


def prepare_snapshot(catalog, item_codes, cfg, splits=None):
    vocab, matrix = encode_features(catalog, item_codes, cfg)
    metadata = ()
    available = set() if catalog is None else set(catalog.columns)
    conflicts = {name: 0 for name in FEATURE_FIELDS}
    model_conflicts = dict(conflicts)
    duplicate_codes = 0
    conflict_codes = 0
    conflict_values = 0
    if catalog is not None:
        if "КодНоменклатуры" not in catalog:
            raise ValueError("Номенклатура.csv не содержит обязательную колонку КодНоменклатуры")
        rows = catalog.copy()
        rows["КодНоменклатуры"] = rows["КодНоменклатуры"].astype(str)
        duplicates = rows.loc[rows["КодНоменклатуры"].duplicated(keep=False)]
        duplicate_codes = int(duplicates["КодНоменклатуры"].nunique())
        columns = [name for name in FEATURE_FIELDS if name in rows]
        if len(duplicates) and columns:
            counts = duplicates.fillna("").groupby("КодНоменклатуры")[columns].nunique()
            conflict_codes = int((counts > 1).any(axis=1).sum())
            conflict_values = int((counts - 1).clip(lower=0).to_numpy().sum())
            model_counts = counts.loc[counts.index.isin(item_codes)]
            conflicts.update({name: int((counts[name] > 1).sum()) for name in columns})
            model_conflicts.update({name: int((model_counts[name] > 1).sum()) for name in columns})
        rows = rows.drop_duplicates("КодНоменклатуры", keep="last").set_index("КодНоменклатуры")
        metadata = tuple((code, tuple((name, "" if row[name] is None else str(row[name]))
                                     for name in META_FIELDS if name in rows))
                         for code in item_codes if code in rows.index
                         for row in (rows.loc[code],))

    # Single O(pairs + items*fields) aggregation; no per-customer diagnostics.
    pair_counts = np.zeros(len(item_codes), dtype=np.int64)
    confidence = np.zeros(len(item_codes), dtype=np.float64)
    if splits is not None:
        pair_counts = np.bincount(splits.train_pairs[:, 1], minlength=len(item_codes))
        confidence = np.bincount(splits.train_pairs[:, 1], weights=splits.train_weights,
                                 minlength=len(item_codes))
    per_field = {}
    for name in FEATURE_FIELDS:
        ids = [index for token, index in vocab.items() if token.startswith(name + "=")]
        covered = np.isin(matrix, ids).any(axis=1)
        count = int(covered.sum())
        pairs = int(pair_counts[covered].sum())
        mass = float(confidence[covered].sum())
        per_field[name] = {"items": count, "item_fraction": count / len(item_codes) if item_codes else 0.,
                           "train_pairs": pairs, "train_pair_fraction": pairs / int(pair_counts.sum()) if pair_counts.sum() else 0.,
                           "confidence_fraction": mass / float(confidence.sum()) if confidence.sum() else 0.}
    diagnostics = {
        "coverage_unit": "aggregated_train_pairs", "items": len(item_codes),
        "training_coverage_available": splits is not None,
        "available_fields": [name for name in FEATURE_FIELDS if name in cfg.item_feature_cols and name in available],
        "missing_fields": [name for name in FEATURE_FIELDS if name in cfg.item_feature_cols and name not in available],
        "other_selected_fields_count": sum(name not in FEATURE_FIELDS for name in cfg.item_feature_cols),
        "items_without_features": int((matrix < 0).all(axis=1).sum()), "unique_tokens": len(vocab),
        "train_pairs": int(pair_counts.sum()), "per_field": per_field,
        "duplicate_codes": duplicate_codes, "conflicting_fields": conflicts,
        "conflicting_codes": conflict_codes, "conflicting_value_excess": conflict_values,
        "model_conflicting_fields": model_conflicts,
        "category_conflict_codes": conflicts["КатегорияНаСайте"],
    }
    # Bytes-backed array cannot be made writable by a consumer.
    frozen_matrix = np.frombuffer(matrix.tobytes(), dtype=np.int64).reshape(matrix.shape)
    return ItemFeatureSnapshot(tuple(item_codes), feature_parameters(cfg), tuple(vocab), frozen_matrix,
                               metadata, json.dumps(diagnostics, ensure_ascii=False))
