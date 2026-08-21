from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpyxl import Workbook
import csv
import re
from difflib import SequenceMatcher

# --- make CPU BLAS usage predictable (often important for UI apps on Windows) ---
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ============================= Config =============================

@dataclass
class TrainConfig:
    # processed csv folder (created in your "dataset processing" tab)
    data_dir: str = "ВходныеДанные"

    # implicit feedback weights
    w_view_item: float = 0.1
    w_favorite: float = 2.0
    w_purchase: float = 10.0

    # BPR-MF
    embedding_dim: int = 128
    epochs: int = 200
    batch_size: int = 256
    lr: float = 3e-4  # 0.0003
    weight_decay: float = 0.0
    bpr_reg: float = 5e-4  # 0.0005
    n_neg: int = 10  # negatives per positive in BPR (1..20 works well)
    seed: int = 42

    # eval
    topk: int = 10
    min_user_interactions_for_eval: int = 10

    # early stopping
    early_stop: bool = True
    early_stop_metric: str = "ndcg"
    early_stop_patience: int = 8
    early_stop_min_delta: float = 5e-4  # 0.0005
    early_stop_min_epochs: int = 30

    # item side-features from Номенклатура.csv (no cold-start: only for items seen in events)
    use_item_features: bool = True
    item_feature_cols: List[str] = field(default_factory=lambda: [
        "ВидНоменклатуры",
        "ВидАссортимента",
        "Марка",
        "Коллекция",
        "СезонНоски",
        "ПолНоменклатуры",
        "ГруппаСоставов",
        "КатегорияНаСайте",
        "СтилеваяГруппа",
    ])

    max_item_features: int = 32
    feature_dropout: float = 0.10
    feature_scale: float = 0.20
    feature_norm: str = "mean"
    feat_reg_mult: float = 1.00


# ============================= Helpers =============================
def _now() -> str:
    return time.strftime("%d-%m-%Y %H:%M:%S")


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _path_csv(data_dir: str, name: str) -> str:
    return os.path.join(data_dir, f"{name}.csv")


def _read_csv_pipe(path: str) -> pd.DataFrame:
    # your processed files are pipe-separated with utf-8-sig (BOM-safe)
    return pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")


def _parse_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NaT] * len(df))
    return pd.to_datetime(df[col], errors="coerce")


# ============================= Data prep =============================

@dataclass
class Mappings:
    user2idx: Dict[str, int]
    idx2user: List[str]
    item2idx: Dict[str, int]
    idx2item: List[str]


@dataclass
class Splits:
    train_pairs: np.ndarray  # [N,2] (u,i)
    train_weights: np.ndarray  # [N]
    eval_users: np.ndarray  # [M]
    eval_items: np.ndarray  # [M]
    user_pos_train: List[set]  # per user: set(items)


def _build_mappings(orders: pd.DataFrame, views: pd.DataFrame, fav: pd.DataFrame) -> Mappings:
    users = pd.concat(
        [
            orders.get("MindboxID", pd.Series(dtype=str)),
            views.get("MindboxID", pd.Series(dtype=str)),
            fav.get("MindboxID", pd.Series(dtype=str)),
        ],
        axis=0,
    ).dropna().astype(str).unique().tolist()

    item_series = []
    # purchases + favourites
    for df in (orders, fav):
        if "КодНоменклатуры" in df.columns:
            item_series.append(df["КодНоменклатуры"])

    # views only for "Номенклатура"
    if {"ТипТовара", "КодНоменклатуры"}.issubset(views.columns):
        item_series.append(views.loc[views["ТипТовара"] == "Номенклатура", "КодНоменклатуры"])

    items = pd.concat(item_series, axis=0).dropna().astype(str).unique().tolist()

    return Mappings(
        user2idx={u: i for i, u in enumerate(users)},
        idx2user=users,
        item2idx={it: i for i, it in enumerate(items)},
        idx2item=items,
    )


def _collect_user_item_events(
        orders: pd.DataFrame,
        views: pd.DataFrame,
        fav: pd.DataFrame,
        maps: Mappings,
        cfg: TrainConfig,
) -> pd.DataFrame:

    frames: List[pd.DataFrame] = []

    # purchases
    if len(orders) and {"MindboxID", "КодНоменклатуры"}.issubset(orders.columns):
        o = orders[["MindboxID", "КодНоменклатуры"]].copy()
        o["ts"] = _parse_date_col(orders, "Дата")
        qty = pd.to_numeric(orders.get("Количество", 1), errors="coerce").fillna(1).astype(float).clip(1, 10)
        o["w"] = cfg.w_purchase * qty
        o = o.dropna(subset=["MindboxID", "КодНоменклатуры"])
        o["u_idx"] = o["MindboxID"].astype(str).map(maps.user2idx)
        o["i_idx"] = o["КодНоменклатуры"].astype(str).map(maps.item2idx)
        o = o.dropna(subset=["u_idx", "i_idx"])
        frames.append(o[["u_idx", "i_idx", "ts", "w"]])

    # favourites
    if len(fav) and {"MindboxID", "КодНоменклатуры"}.issubset(fav.columns):
        f = fav[["MindboxID", "КодНоменклатуры"]].copy()
        f["ts"] = _parse_date_col(fav, "Дата")
        f["w"] = cfg.w_favorite
        f = f.dropna(subset=["MindboxID", "КодНоменклатуры"])
        f["u_idx"] = f["MindboxID"].astype(str).map(maps.user2idx)
        f["i_idx"] = f["КодНоменклатуры"].astype(str).map(maps.item2idx)
        f = f.dropna(subset=["u_idx", "i_idx"])
        frames.append(f[["u_idx", "i_idx", "ts", "w"]])

    # views
    if len(views) and {"MindboxID", "КодНоменклатуры", "ТипТовара"}.issubset(views.columns):
        v = views.loc[views["ТипТовара"] == "Номенклатура", ["MindboxID", "КодНоменклатуры"]].copy()
        v["ts"] = _parse_date_col(views.loc[views["ТипТовара"] == "Номенклатура"], "Дата")
        v["w"] = cfg.w_view_item
        v = v.dropna(subset=["MindboxID", "КодНоменклатуры"])
        v["u_idx"] = v["MindboxID"].astype(str).map(maps.user2idx)
        v["i_idx"] = v["КодНоменклатуры"].astype(str).map(maps.item2idx)
        v = v.dropna(subset=["u_idx", "i_idx"])
        frames.append(v[["u_idx", "i_idx", "ts", "w"]])

    if not frames:
        return pd.DataFrame(columns=["u_idx", "i_idx", "ts", "w"])

    ev = pd.concat(frames, axis=0, ignore_index=True)
    ev["u_idx"] = ev["u_idx"].astype(int)
    ev["i_idx"] = ev["i_idx"].astype(int)
    ev["ts"] = ev["ts"].fillna(pd.Timestamp("1970-01-01"))
    ev["w"] = pd.to_numeric(ev["w"], errors="coerce").fillna(1.0).astype(float)
    return ev


def _train_test_split_last_per_user(events: pd.DataFrame, cfg: TrainConfig, num_users: int) -> Splits:

    ev = events.copy()
    if "_row_id" not in ev.columns:
        ev["_row_id"] = np.arange(len(ev), dtype=np.int64)
    events_sorted = ev.sort_values(["u_idx", "ts", "_row_id"], kind="mergesort")

    counts = events_sorted.groupby("u_idx").size()
    eligible_users = counts[counts >= cfg.min_user_interactions_for_eval].index.values

    last = events_sorted.groupby("u_idx").tail(1)
    last = last[last["u_idx"].isin(eligible_users)]

    eval_users = last["u_idx"].astype(int).to_numpy()
    eval_items = last["i_idx"].astype(int).to_numpy()

    train_ev = events_sorted.drop(index=last.index)

    # aggregate multiple events to one weighted pair
    train_agg = train_ev.groupby(["u_idx", "i_idx"], as_index=False)["w"].sum()
    train_pairs = train_agg[["u_idx", "i_idx"]].astype(int).to_numpy()
    train_weights = train_agg["w"].astype(float).to_numpy()

    user_pos_train = [set() for _ in range(num_users)]
    for u, i in train_pairs:
        user_pos_train[int(u)].add(int(i))

    return Splits(
        train_pairs=train_pairs,
        train_weights=train_weights,
        eval_users=eval_users,
        eval_items=eval_items,
        user_pos_train=user_pos_train,
    )


def _sample_batch(
        train_pairs: np.ndarray,
        train_weights: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if len(train_pairs) <= batch_size:
        return train_pairs[:, 0], train_pairs[:, 1], train_weights.astype(np.float64)

    w = train_weights.astype(np.float64)
    w = w / (w.sum() + 1e-12)
    idx = rng.choice(len(train_pairs), size=batch_size, replace=False, p=w)
    return train_pairs[idx, 0], train_pairs[idx, 1], train_weights[idx].astype(np.float64)


def _sample_negatives(
        users: np.ndarray,
        num_items: int,
        user_pos_train: List[set],
        rng: np.random.Generator,
        n_neg: int = 10,
        max_tries: int = 25,
) -> np.ndarray:

    n_neg = max(1, int(n_neg))
    if len(users) == 0:
        return np.zeros((0, n_neg), dtype=np.int64)

    neg = rng.integers(0, num_items, size=(len(users), n_neg), dtype=np.int64)

    # rejection sampling (keep it deterministic given rng)
    for _ in range(max_tries):
        bad = np.zeros_like(neg, dtype=bool)
        for bi, u in enumerate(users):
            pos = user_pos_train[int(u)]
            if pos:
                # np.isin is fast for small n_neg; convert set -> list once per user
                bad[bi] = np.isin(neg[bi], list(pos))
        if not bad.any():
            break
        neg[bad] = rng.integers(0, num_items, size=int(bad.sum()), dtype=np.int64)

    return neg


# ============================= Item features (Номенклатура.csv) =============================

def _build_item_feature_matrix(
        data_dir: str,
        maps: Mappings,
        cfg: TrainConfig,
) -> Tuple[Dict[str, int], np.ndarray]:

    num_items = len(maps.idx2item)
    max_f = int(getattr(cfg, "max_item_features", 32))
    item_feat_mat = np.full((num_items, max_f), fill_value=-1, dtype=np.int64)

    if not bool(getattr(cfg, "use_item_features", True)):
        return {}, item_feat_mat

    nom_path = os.path.join(data_dir, "Номенклатура.csv")
    if not os.path.isfile(nom_path):
        return {}, item_feat_mat

    try:
        nom = _read_csv_pipe(nom_path)
    except Exception:
        return {}, item_feat_mat

    if "КодНоменклатуры" not in nom.columns:
        return {}, item_feat_mat

    cols = [c for c in getattr(cfg, "item_feature_cols", []) if c in nom.columns]
    if not cols:
        return {}, item_feat_mat

    nom = nom[["КодНоменклатуры"] + cols].copy()
    nom["КодНоменклатуры"] = nom["КодНоменклатуры"].astype(str)
    nom = nom.drop_duplicates("КодНоменклатуры", keep="last").set_index("КодНоменклатуры")

    feat2idx: Dict[str, int] = {}

    def _get_fid(token: str) -> int:
        fid = feat2idx.get(token)
        if fid is None:
            fid = len(feat2idx)
            feat2idx[token] = fid
        return fid

    for i_idx, code in enumerate(maps.idx2item):
        if code not in nom.index:
            continue
        row = nom.loc[code]
        tokens: List[str] = []
        for c in cols:
            v = row.get(c)
            if v is None:
                continue
            v = str(v).strip()
            if not v or v.lower() in ("nan", "none", "null", "-"):
                continue
            tokens.append(f"{c}={v}")

        if not tokens:
            continue
        # cap
        tokens = tokens[:max_f]
        ids = [_get_fid(t) for t in tokens]
        item_feat_mat[i_idx, :len(ids)] = np.asarray(ids, dtype=np.int64)

    return feat2idx, item_feat_mat


class BPRMF(nn.Module):

    def __init__(
            self,
            num_users: int,
            num_items: int,
            emb_dim: int,
            num_item_feats: int = 0,
            item_feat_mat: Optional[torch.Tensor] = None,
            feature_dropout: float = 0.0,
            feature_scale: float = 0.2,
            feature_norm: str = "mean",
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.use_item_features = bool(num_item_feats > 0 and item_feat_mat is not None)
        if self.use_item_features:
            self.feat_emb = nn.Embedding(int(num_item_feats), emb_dim)
            # item_feat_mat: [I, F] with -1 padding
            self.register_buffer("item_feat_mat", item_feat_mat.long())
            self.feature_dropout = float(feature_dropout)
            self.feature_scale = float(feature_scale)
            self.feature_norm = str(feature_norm).strip().lower()
        else:
            self.feat_emb = None
            self.item_feat_mat = None
            self.feature_dropout = 0.0
            self.feature_scale = 0.0
            self.feature_norm = "sum"

        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        if self.use_item_features:
            nn.init.normal_(self.feat_emb.weight, std=0.1)

    def _item_vec(self, i_idx: torch.Tensor) -> torch.Tensor:
        """Returns item vectors with side-features summed in."""
        base = self.item_emb(i_idx)  # [..., d]
        if not self.use_item_features:
            return base

        # fids: [..., F]
        fids = self.item_feat_mat[i_idx]
        mask = fids.ge(0)
        fids = fids.clamp_min(0)

        emb = self.feat_emb(fids)  # [..., F, d]
        emb = emb * mask.unsqueeze(-1)

        # optional dropout by feature token
        if self.training and self.feature_dropout > 0.0:
            keep = (torch.rand(mask.shape, device=mask.device) > self.feature_dropout)
            emb = emb * keep.unsqueeze(-1)

        add = emb.sum(dim=-2)  # sum over F

        # normalize by number of present features (helps stability)
        norm = getattr(self, "feature_norm", "sum")
        if norm in ("mean", "avg"):
            cnt = mask.sum(dim=-1).clamp_min(1).to(add.dtype)
            add = add / cnt.unsqueeze(-1)
        elif norm in ("sqrt", "sqrt_mean", "sqrt-norm"):
            cnt = mask.sum(dim=-1).clamp_min(1).to(add.dtype)
            add = add / torch.sqrt(cnt).unsqueeze(-1)

        scale = float(getattr(self, "feature_scale", 1.0))
        return base + scale * add

    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        """Return dot-product scores for (user, item) pairs.

        Supports:
          - u_idx: [B], i_idx: [B]           -> [B]
          - u_idx: [B], i_idx: [B, n_neg]    -> [B, n_neg]
        """
        u = self.user_emb(u_idx)  # [B, d]
        v = self._item_vec(i_idx)  # [B, d] or [B, n_neg, d]

        if u.dim() == 2 and v.dim() == 3:
            u = u.unsqueeze(1)  # [B, 1, d]

        return (u * v).sum(dim=-1)

    @torch.no_grad()
    def item_vec_all(self) -> torch.Tensor:
        """Returns matrix [I, d] of item vectors (with side-features)."""
        if not self.use_item_features:
            return self.item_emb.weight

        fids = self.item_feat_mat  # [I, F]
        mask = fids.ge(0)
        fids = fids.clamp_min(0)
        emb = self.feat_emb(fids) * mask.unsqueeze(-1)  # [I, F, d]
        add = emb.sum(dim=1)
        norm = getattr(self, "feature_norm", "sum")
        if norm in ("mean", "avg"):
            cnt = mask.sum(dim=1).clamp_min(1).to(add.dtype)
            add = add / cnt.unsqueeze(-1)
        elif norm in ("sqrt", "sqrt_mean", "sqrt-norm"):
            cnt = mask.sum(dim=1).clamp_min(1).to(add.dtype)
            add = add / torch.sqrt(cnt).unsqueeze(-1)
        scale = float(getattr(self, "feature_scale", 1.0))
        return self.item_emb.weight + scale * add


@torch.no_grad()
def _eval_bprmf_recall_ndcg(
        model: BPRMF,
        splits: Splits,
        num_items: int,
        k: int,
        device: torch.device,
) -> Tuple[float, float]:
    if len(splits.eval_users) == 0:
        return 0.0, 0.0

    item_emb = model.item_vec_all()  # [I, d]
    users = splits.eval_users.astype(np.int64)
    gt = splits.eval_items.astype(np.int64)

    chunk = 512
    recalls: List[float] = []
    ndcgs: List[float] = []

    for start in range(0, len(users), chunk):
        u = users[start:start + chunk]
        g = gt[start:start + chunk]

        u_t = torch.tensor(u, dtype=torch.long, device=device)
        u_emb = model.user_emb(u_t)  # [B, d]
        scores = u_emb @ item_emb.t()  # [B, I]

        # filter train positives (so we don't recommend already seen items)
        # IMPORTANT: if the evaluation target was also seen in train (repeat interaction),
        # we do NOT mask the target item so it remains rankable.
        for bi, uu in enumerate(u):
            target = int(g[bi])
            pos = splits.user_pos_train[int(uu)]
            if pos:
                if target in pos:
                    idx_list = [x for x in pos if x != target]
                else:
                    idx_list = list(pos)

                if idx_list:
                    idx = torch.tensor(idx_list, dtype=torch.long, device=device)
                    scores[bi, idx] = -1e9

        topk_idx = torch.topk(scores, k=min(k, num_items), dim=1).indices.cpu().numpy()

        for bi in range(len(u)):
            target = int(g[bi])
            row = topk_idx[bi]
            if target in row:
                rank = int(np.where(row == target)[0][0]) + 1
                recalls.append(1.0)
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def train_bprmf(maps: Mappings, events: pd.DataFrame, cfg: TrainConfig, device: torch.device) -> Tuple[BPRMF, Splits]:
    # keep PyTorch thread usage stable (important for desktop apps)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    num_users = len(maps.idx2user)
    num_items = len(maps.idx2item)
    splits = _train_test_split_last_per_user(events, cfg, num_users)

    # build item side-features from Номенклатура.csv (no cold-start)
    feat2idx, item_feat_np = _build_item_feature_matrix(cfg.data_dir, maps, cfg)
    item_feat_t = torch.tensor(item_feat_np, dtype=torch.long, device=device)

    # --- diagnostics: verify item features are actually used ---
    use_feats = bool(getattr(cfg, "use_item_features", True))
    enabled = bool(use_feats)
    applied = False

    if enabled:
        tokens = len(feat2idx)
        covered_items = int((item_feat_np[:, 0] >= 0).sum()) if getattr(item_feat_np, "size", 0) else 0
        applied = (tokens > 0) and (covered_items > 0)

    if enabled:
        status = "включены, применяются" if applied else "включены, не применяются"
    else:
        status = "не включены, не применяются"

    print(f"[{_now()}] Признаки номенклатуры: вид номенклатуры, вид ассортимента, марка, коллекция, сезон носки, пол, "
          f"группа составов, категория на сайте, стилевая группа --> {status}\n")

    # --- ЛОГ: входные параметры (итоговый cfg, реально ушедший в обучение) ---
    def _fmt(v):
        if isinstance(v, bool):
            return "да" if v else "нет"
        if isinstance(v, float):
            return f"{v:.10g}"
        if isinstance(v, (list, tuple)):
            return ", ".join(map(str, v))
        return str(v)

    # нормализуем метрику и norm так же, как дальше в обучении
    metric_raw = str(getattr(cfg, "early_stop_metric", "ndcg")).strip().lower()
    metric_norm = metric_raw if metric_raw in ("ndcg", "recall") else "ndcg"

    feat_norm_raw = str(getattr(cfg, "feature_norm", "mean")).strip().lower()
    feat_norm_norm = feat_norm_raw if feat_norm_raw in ("sum", "mean", "sqrt") else "mean"

    print(f"[{_now()}] Входные параметры:")
    for name, val in [
        ("Вес покупки", getattr(cfg, "w_purchase", 0.0)),
        ("Вес избранного", getattr(cfg, "w_favorite", 0.0)),
        ("Вес просмотра", getattr(cfg, "w_view_item", 0.0)),
        ("Количество рекомендаций", getattr(cfg, "topk", 10)),
        ("Количество эпох", getattr(cfg, "epochs", 0)),
        ("Скорость обучения", getattr(cfg, "lr", 0.0)),
        ("Количество отрицательных примеров", getattr(cfg, "n_neg", 10)),
        ("Инициализатор случайных чисел", getattr(cfg, "seed", 42)),
        ("Регуляризация BPR", getattr(cfg, "bpr_reg", 0.0)),
        ("Регуляризация L2", getattr(cfg, "weight_decay", 0.0)),
        ("Минимум действий для оценки", getattr(cfg, "min_user_interactions_for_eval", 0)),
        ("Размер обучающего пакета", getattr(cfg, "batch_size", 0)),
        ("Размерность векторов", getattr(cfg, "embedding_dim", 0)),
        ("Количество эпох без улучшения", getattr(cfg, "early_stop_patience", 0)),
        ("Минимальный прирост метрики", getattr(cfg, "early_stop_min_delta", 0.0)),
        ("Минимум эпох до остановки", getattr(cfg, "early_stop_min_epochs", 0)),
        ("Метрика ранней остановки", metric_norm.upper()),
        ("Максимальное число признаков номенклатуры", getattr(cfg, "max_item_features", 0)),
        ("Случайное отключение признаков номенклатуры", getattr(cfg, "feature_dropout", 0.0)),
        ("Вес признаков номенклатуры в модели", getattr(cfg, "feature_scale", 0.0)),
        ("Сила регуляризации признаков номенклатуры", getattr(cfg, "feat_reg_mult", 1.0)),
        ("Способ объединения признаков номенклатуры", feat_norm_norm.upper()),
    ]:
        print(f"- {name}: {_fmt(val)}")

    print("")  # пустая строка перед итерациями обучения

    model = BPRMF(
        num_users,
        num_items,
        cfg.embedding_dim,
        num_item_feats=len(feat2idx),
        item_feat_mat=item_feat_t,
        feature_dropout=float(getattr(cfg, "feature_dropout", 0.0)),
        feature_scale=float(getattr(cfg, "feature_scale", 0.2)),
        feature_norm=str(getattr(cfg, "feature_norm", "mean")),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)

    # early stopping configuration
    metric_name = str(getattr(cfg, "early_stop_metric", "ndcg")).strip().lower()
    if metric_name not in ("ndcg", "recall"):
        metric_name = "ndcg"

    use_early_stop = bool(getattr(cfg, "early_stop", True))
    patience = max(1, int(getattr(cfg, "early_stop_patience", 2)))
    min_delta = float(getattr(cfg, "early_stop_min_delta", 1e-4))
    min_epochs = max(1, int(getattr(cfg, "early_stop_min_epochs", 1)))

    best = {"metric": -1e9, "RECALL": -1.0, "NDCG": -1.0, "epoch": -1, "state": None}
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        steps = max(1, int(np.ceil(len(splits.train_pairs) / cfg.batch_size)))
        total_loss = 0.0

        for _ in range(steps):
            u_pos, i_pos, w_pos = _sample_batch(splits.train_pairs, splits.train_weights, cfg.batch_size, rng)
            n_neg = max(1, int(getattr(cfg, "n_neg", 10)))
            i_neg = _sample_negatives(u_pos, num_items, splits.user_pos_train, rng, n_neg=n_neg)

            u_t = torch.tensor(u_pos, dtype=torch.long, device=device)
            ip_t = torch.tensor(i_pos, dtype=torch.long, device=device)
            in_t = torch.tensor(i_neg, dtype=torch.long, device=device)

            s_pos = model.score(u_t, ip_t)  # [B]
            s_neg = model.score(u_t, in_t)  # [B, n_neg]
            delta = s_pos.unsqueeze(1) - s_neg
            per = -F.logsigmoid(delta).mean(dim=1)  # [B]

            w_t = torch.tensor(w_pos, dtype=torch.float32, device=device)
            loss = (per * w_t).sum() / w_t.sum().clamp_min(1e-6)
            if cfg.bpr_reg > 0:
                u_reg = model.user_emb(u_t).pow(2).sum(dim=-1)  # [B]
                p_reg = model.item_emb(ip_t).pow(2).sum(dim=-1)  # [B]
                n_reg = model.item_emb(in_t).pow(2).sum(dim=-1).mean(dim=1)  # [B]
                reg = (u_reg + p_reg + n_reg).mean()
                if getattr(model, "use_item_features", False) and getattr(model, "feat_emb", None) is not None:
                    reg = reg + float(getattr(cfg, "feat_reg_mult", 1.0)) * model.feat_emb.weight.pow(2).sum(
                        dim=-1).mean()
                loss = loss + cfg.bpr_reg * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())

        model.eval()
        recall, ndcg = _eval_bprmf_recall_ndcg(model, splits, num_items, cfg.topk, device)

        print(
            f"[{_now()}] Итерация {epoch} из {cfg.epochs}: "
            f"loss={total_loss / steps:.4f}  RECALL@{cfg.topk}={recall:.4f}  NDCG@{cfg.topk}={ndcg:.4f}"
        )

        cur_metric = ndcg if metric_name == "ndcg" else recall
        improved = cur_metric > best["metric"] + min_delta

        if improved:
            best["metric"] = float(cur_metric)
            best["RECALL"], best["NDCG"], best["epoch"] = float(recall), float(ndcg), epoch
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            if use_early_stop and epoch >= min_epochs:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(
                        f"[{_now()}] Преждевременная остановка на итерации {epoch}: {metric_name}@{cfg.topk} не улучшается "
                        f"{patience} итерации подряд (лучший показатель = {best['metric']:.4f})"
                    )
                    break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    print(
        f"[{_now()}] Лучшие показатели метрики {metric_name}@{cfg.topk} на итерации {best['epoch']}: "
        f"RECALL@{cfg.topk}={best['RECALL']:.4f} NDCG@{cfg.topk}={best['NDCG']:.4f}"
    )
    return model, splits


# ============================= Saving / Loading =============================
def _save_artifacts(cfg: TrainConfig, maps: Mappings, model: BPRMF) -> None:

    out_dir = os.path.join(os.getcwd(), "Модель")
    _ensure_dir(out_dir)

    with open(os.path.join(out_dir, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump({"idx2user": maps.idx2user, "idx2item": maps.idx2item}, f, ensure_ascii=False)

    ckpt = {
        "model_type": "bprmf",
        "config": cfg.__dict__,
        "num_users": len(maps.idx2user),
        "num_items": len(maps.idx2item),
        "state_dict": model.state_dict(),
    }

    # save item features for consistent inference/evaluation
    feat2idx, item_feat_np = _build_item_feature_matrix(cfg.data_dir, maps, cfg)
    ckpt["feat2idx"] = feat2idx
    ckpt["item_feat_mat"] = item_feat_np
    ckpt["item_feature_cols"] = getattr(cfg, "item_feature_cols", [])
    ckpt["max_item_features"] = int(getattr(cfg, "max_item_features", 32))

    # --- сохраняем метаданные товаров из Номенклатура.csv на момент обучения ---
    # Нужно для маппинга "старый сезон -> актуальная коллекция" при экспорте рекомендаций.
    try:
        train_item_meta: Dict[str, Dict[str, str]] = {}
        nom_path = os.path.join(cfg.data_dir, "Номенклатура.csv")
        if os.path.isfile(nom_path):
            nom = _read_csv_pipe(nom_path)
            nom.columns = [str(c).replace("\ufeff", "").strip() for c in nom.columns]

            want_cols = [
                "КодНоменклатуры", "Коллекция", "НазваниеНаСайте", "Номенклатура",
                "ВидНоменклатуры", "ПолНоменклатуры", "КатегорияНаСайте", "СтилеваяГруппа",
                "Марка", "ГруппаСоставов", "ВидАссортимента", "Остаток"
            ]
            cols = [c for c in want_cols if c in nom.columns]
            if "КодНоменклатуры" in cols:
                sub = nom[cols].copy()
                sub["КодНоменклатуры"] = sub["КодНоменклатуры"].astype(str)
                sub = sub.drop_duplicates("КодНоменклатуры", keep="last").set_index("КодНоменклатуры")

                for code in maps.idx2item:
                    if code in sub.index:
                        row = sub.loc[code].to_dict()
                        # гарантируем строковые значения
                        train_item_meta[str(code)] = {k: ("" if row.get(k) is None else str(row.get(k))) for k in
                                                      row.keys()}

        ckpt["train_item_meta"] = train_item_meta

    except Exception:

        ckpt["train_item_meta"] = {}

    torch.save(ckpt, os.path.join(out_dir, "bprmf.pt"))


def _load_artifacts(model_dir: str = "Модель") -> Tuple[Dict[str, List[str]], dict]:
    out_dir = os.path.join(os.getcwd(), model_dir)
    mappings_path = os.path.join(out_dir, "mappings.json")
    ckpt_path = os.path.join(out_dir, "bprmf.pt")

    if not (os.path.isfile(mappings_path) and os.path.isfile(ckpt_path)):
        raise FileNotFoundError("Не найдена обученная модель, необходимо выполнить обучение.")

    with open(mappings_path, "r", encoding="utf-8") as f:
        maps_json = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    return maps_json, ckpt


# ============================= Recommendation =============================
NON_SEASONAL_COLLECTIONS = {
    "NOS",
    "БЕЗ СЕЗОНА",
    "ВСЕСЕЗОННЫЙ",
    "КОМИССИЯ",
    "КОМИССИЯ ДЕМИ",
    "КОМИССИЯ ЗИМА",
    "КОМИССИЯ ЛЕТО",
}


def _norm_text(x) -> str:
    s = "" if x is None else str(x)
    s = s.replace("\ufeff", "").replace("\xa0", " ")
    s = " ".join(s.split()).strip()
    s = s.replace("–", "-").replace("—", "-")
    return s


def _is_nonseasonal_collection(coll: str) -> bool:
    return _norm_text(coll).upper() in NON_SEASONAL_COLLECTIONS


def _parse_season_base_and_year(coll: str) -> Tuple[Optional[str], Optional[int]]:
    """
    "Весна-Лето 2025" -> ("ВЕСНА-ЛЕТО", 2025)
    "Осень-Зима 2026" -> ("ОСЕНЬ-ЗИМА", 2026)
    """
    s = _norm_text(coll).upper()
    m = re.search(r"(19|20)\d{2}", s)
    year = int(m.group(0)) if m else None

    if "ВЕСНА" in s and "ЛЕТО" in s:
        return "ВЕСНА-ЛЕТО", year
    if "ОСЕНЬ" in s and "ЗИМА" in s:
        return "ОСЕНЬ-ЗИМА", year

    return None, year


def _load_selected_collections_from_settings() -> List[str]:
    """
    Берём выбранные пользователем актуальные коллекции из Настройки/filter_settings.json.
    Ключ: collections_selected (или старый seasons_selected).
    """
    path = os.path.join(os.getcwd(), "Настройки", "filter_settings.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vals = data.get("collections_selected") or data.get("seasons_selected") or []
        out = [_norm_text(v) for v in vals if _norm_text(v)]
        return out
    except Exception:
        return []


def _similarity_score(old_meta: Dict[str, str], new_meta: Dict[str, str]) -> float:
    """
    Скоринг похожести "старый товар" -> "новый товар" по реквизитам + названию.
    """
    score = 0.0

    # совпадения по реквизитам (можно тюнить веса)
    for key, w in [
        ("ВидНоменклатуры", 3.0),
        ("ПолНоменклатуры", 2.0),
        ("КатегорияНаСайте", 2.0),
        ("СтилеваяГруппа", 1.5),
        ("Марка", 1.0),
        ("ГруппаСоставов", 0.8),
        ("ВидАссортимента", 0.8),
    ]:
        a = _norm_text(old_meta.get(key, ""))
        b = _norm_text(new_meta.get(key, ""))
        if a and b and a == b:
            score += w

    # похожесть названия
    name_a = _norm_text(old_meta.get("НазваниеНаСайте") or old_meta.get("Номенклатура") or "")
    name_b = _norm_text(new_meta.get("НазваниеНаСайте") or new_meta.get("Номенклатура") or "")
    if name_a and name_b:
        score += 1.2 * SequenceMatcher(None, name_a, name_b).ratio()

    return score


def _load_item_names(data_dir: str) -> Dict[str, str]:
    nom_path = os.path.join(data_dir, "Номенклатура.csv")
    if not os.path.isfile(nom_path):
        return {}
    try:
        nom = _read_csv_pipe(nom_path)
    except Exception:
        return {}

    if "КодНоменклатуры" not in nom.columns:
        return {}

    have_site = "НазваниеНаСайте" in nom.columns
    have_nom = "Номенклатура" in nom.columns
    if not (have_site or have_nom):
        return {}

    cols = ["КодНоменклатуры"]
    if have_site: cols.append("НазваниеНаСайте")
    if have_nom: cols.append("Номенклатура")

    sub = nom[cols].copy()
    sub["КодНоменклатуры"] = sub["КодНоменклатуры"].astype(str)

    def _clean(v) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null", "<na>", "-"):
            return ""
        return s

    # выбираем имя построчно: НазваниеНаСайте -> Номенклатура
    if have_site:
        sub["__site"] = sub["НазваниеНаСайте"].map(_clean)
    else:
        sub["__site"] = ""

    if have_nom:
        sub["__nom"] = sub["Номенклатура"].map(_clean)
    else:
        sub["__nom"] = ""

    sub["__name"] = sub["__site"]
    sub.loc[sub["__name"] == "", "__name"] = sub.loc[sub["__name"] == "", "__nom"]

    sub = sub[sub["__name"] != ""]
    sub = sub.drop_duplicates("КодНоменклатуры", keep="last")

    return dict(zip(sub["КодНоменклатуры"].tolist(), sub["__name"].tolist()))


def _format_stock_value(v) -> str:
    """
    Приводит Остаток к целому строковому виду:
      10.0 -> "10"
      10,0 -> "10"
      пусто/NaN -> "0"
    """
    if v is None:
        return "0"

    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", "<na>", "-"):
        return "0"

    s = s.replace(",", ".").replace(" ", "")

    try:
        num = pd.to_numeric(s, errors="coerce")
        if pd.isna(num):
            return "0"
        return str(int(round(float(num))))
    except Exception:
        return "0"


def _load_item_stocks(data_dir: str) -> Dict[str, str]:
    """
    Загружает остатки из ВходныеДанные/Номенклатура.csv:
      КодНоменклатуры -> Остаток
    """
    nom_path = os.path.join(data_dir, "Номенклатура.csv")
    if not os.path.isfile(nom_path):
        return {}

    try:
        nom = _read_csv_pipe(nom_path)
    except Exception:
        return {}

    nom.columns = [str(c).replace("\ufeff", "").strip() for c in nom.columns]

    if "КодНоменклатуры" not in nom.columns or "Остаток" not in nom.columns:
        return {}

    sub = nom[["КодНоменклатуры", "Остаток"]].copy()

    sub["КодНоменклатуры"] = (
        sub["КодНоменклатуры"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    sub["Остаток"] = sub["Остаток"].map(_format_stock_value)

    sub = sub[sub["КодНоменклатуры"] != ""]
    sub = sub.drop_duplicates("КодНоменклатуры", keep="last")

    return dict(zip(sub["КодНоменклатуры"].tolist(), sub["Остаток"].tolist()))


def _user_seen_items_from_processed(data_dir: str, mindbox_id: str, item2idx: Dict[str, int],
                                    cfg: TrainConfig) -> np.ndarray:
    """
    Returns indices of items that user has already interacted with (for filtering).
    """
    orders_path = _path_csv(data_dir, "Заказы")
    views_path = _path_csv(data_dir, "Просмотры")
    fav_path = _path_csv(data_dir, "Избранное")

    seen = set()

    if os.path.isfile(orders_path):
        o = _read_csv_pipe(orders_path)
        if {"MindboxID", "КодНоменклатуры"}.issubset(o.columns):
            oo = o[o["MindboxID"].astype(str) == str(mindbox_id)]
            for code in oo.get("КодНоменклатуры", pd.Series(dtype=str)).dropna().astype(str).tolist():
                idx = item2idx.get(code)
                if idx is not None:
                    seen.add(idx)

    if os.path.isfile(fav_path):
        f = _read_csv_pipe(fav_path)
        if {"MindboxID", "КодНоменклатуры"}.issubset(f.columns):
            ff = f[f["MindboxID"].astype(str) == str(mindbox_id)]
            for code in ff.get("КодНоменклатуры", pd.Series(dtype=str)).dropna().astype(str).tolist():
                idx = item2idx.get(code)
                if idx is not None:
                    seen.add(idx)

    if os.path.isfile(views_path):
        v = _read_csv_pipe(views_path)
        if {"MindboxID", "КодНоменклатуры", "ТипТовара"}.issubset(v.columns):
            vv = v[(v["MindboxID"].astype(str) == str(mindbox_id)) & (v["ТипТовара"] == "Номенклатура")]
            for code in vv.get("КодНоменклатуры", pd.Series(dtype=str)).dropna().astype(str).tolist():
                idx = item2idx.get(code)
                if idx is not None:
                    seen.add(idx)

    if not seen:
        return np.array([], dtype=np.int64)

    return np.fromiter(seen, dtype=np.int64)


def print_recommendations(mindbox_id: str, k: int = 20) -> None:
    """
    Prints top-K recommendations to console using saved artifacts (BPR-MF only).
    """
    cfg = TrainConfig()
    maps_json, ckpt = _load_artifacts()

    idx2user = maps_json["idx2user"]
    idx2item = maps_json["idx2item"]
    user2idx = {u: i for i, u in enumerate(idx2user)}
    item2idx = {code: i for i, code in enumerate(idx2item)}

    if str(mindbox_id) not in user2idx:
        print(f"[{_now()}] User {mindbox_id} not found in mappings.json.")
        return

    seen_idx = _user_seen_items_from_processed(cfg.data_dir, mindbox_id, item2idx, cfg)
    names = _load_item_names(cfg.data_dir)

    num_users = int(ckpt["num_users"])
    num_items = int(ckpt["num_items"])
    emb_dim = int(ckpt["config"]["embedding_dim"])

    feat2idx = ckpt.get("feat2idx", {})
    item_feat_np = ckpt.get("item_feat_mat", None)
    if item_feat_np is not None and isinstance(item_feat_np, np.ndarray) and len(feat2idx) > 0:
        item_feat_t = torch.tensor(item_feat_np, dtype=torch.long)
        model = BPRMF(
            num_users,
            num_items,
            emb_dim,
            num_item_feats=len(feat2idx),
            item_feat_mat=item_feat_t,
            feature_dropout=0.0,
            feature_scale=float(ckpt.get("config", {}).get("feature_scale", 0.2)),
            feature_norm=str(ckpt.get("config", {}).get("feature_norm", "mean")),
        )
    else:
        model = BPRMF(num_users, num_items, emb_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    u_idx = user2idx[str(mindbox_id)]
    u = model.user_emb.weight[u_idx]  # [d]
    item_mat = model.item_vec_all()  # [I, d]
    scores = (u @ item_mat.t()).cpu().numpy()  # [n_items]

    if len(seen_idx):
        scores[seen_idx] = -1e9

    top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    top = top[np.argsort(-scores[top])]

    print(f"[{_now()}] Recommendations (BPR-MF) for MindboxID={mindbox_id} top{k}:")
    for rank, ii in enumerate(top, start=1):
        code = idx2item[int(ii)]
        nm = names.get(code, "")
        if nm:
            print(f"{rank:02d}. {code} | {nm} | score={scores[int(ii)]:.4f}")
        else:
            print(f"{rank:02d}. {code} | score={scores[int(ii)]:.4f}")


# ============================= Training entry point (UI button) =============================

def _train_in_this_process(cfg: Optional[TrainConfig] = None) -> bool:
    cfg = cfg or TrainConfig()
    _set_seed(cfg.seed)

    data_dir = cfg.data_dir
    orders_path = _path_csv(data_dir, "Заказы")
    views_path = _path_csv(data_dir, "Просмотры")
    fav_path = _path_csv(data_dir, "Избранное")

    required = [orders_path, views_path, fav_path]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        print(f"[{_now()}] Отсутствуют следующие необходимые файлы для обучения:")
        for p in missing:
            print(f"  - {p}")
        print("\nДля начала нужно загрузить датасеты на вкладке 'Обработка датасета'.")
        return False

    orders = _read_csv_pipe(orders_path)
    views = _read_csv_pipe(views_path)
    fav = _read_csv_pipe(fav_path)

    print(f"[{_now()}] Загружено 3 файла:")
    print(f"- Заказы.csv: {len(orders):,}".replace(",", ".") + " строк")
    print(f"- Просмотры.csv: {len(views):,}".replace(",", ".") + " строк")
    print(f"- Избранное.csv: {len(fav):,}".replace(",", ".") + " строк\n")

    maps = _build_mappings(orders, views, fav)
    print(f"[{_now()}] Общее количество уникальных пользователей: {len(maps.idx2user):,}".replace(",", "."))
    print(f"[{_now()}] Общее количество уникальных товаров: {len(maps.idx2item):,}".replace(",", "."))

    events = _collect_user_item_events(orders, views, fav, maps, cfg)
    if len(events) == 0:
        print(f"[{_now()}] Не найдено взаимодействий пользователь-товар.")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_label = "GPU (графический процессор, видеокарта)" if device.type == "cuda" else "CPU (центральный процессор)"
    print(f"[{_now()}] Устройство для обучения: {device_label}\n")

    model, _splits = train_bprmf(maps, events, cfg, device)
    _save_artifacts(cfg, maps, model)
    return True


def train_recommender(*_args, **_kwargs) -> None:
    """
    Called by your PyQt button.

    Runs training in a subprocess to avoid occasional native crashes during Python shutdown on Windows.
    """
    py = sys.executable
    script = os.path.abspath(__file__)
    cmd = [py, script, "--train"]
    print(f"[{_now()}] Starting training in subprocess:")
    print(" ", " ".join(cmd))
    try:
        subprocess.Popen(cmd, cwd=os.getcwd())
    except Exception as e:
        print(f"[{_now()}] Failed to start subprocess: {repr(e)}")


# ============================= CLI =============================

TRAIN_EXIT_SUCCESS = 0
TRAIN_EXIT_NO_DATA = 2


def _load_train_config_from_json(path: str) -> TrainConfig:
    cfg = TrainConfig()
    if not path:
        return cfg
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Apply only known fields (ignore everything else)
        allowed = set(cfg.__dict__.keys())
        for k, v in data.items():
            if k in allowed:
                setattr(cfg, k, v)
    except Exception as e:
        print(f"[{_now()}] WARNING: failed to load train config from '{path}': {e}. Using defaults.")
    return cfg


def _parse_cli(argv: List[str]) -> Tuple[bool, Optional[str], int, Optional[str]]:
    do_train = "--train" in argv
    mindbox = None
    k = 20
    config_path: Optional[str] = None

    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 < len(argv):
            config_path = argv[i + 1]

    if "--recommend" in argv:
        i = argv.index("--recommend")
        if i + 1 < len(argv):
            mindbox = argv[i + 1]
    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 < len(argv):
            try:
                k = int(argv[i + 1])
            except Exception:
                pass

    return do_train, mindbox, k, config_path


def _load_cfg_from_ckpt(ckpt: dict) -> TrainConfig:
    """
    Берём фактические параметры обучения из checkpoint["config"].
    Если каких-то полей нет — остаются дефолты TrainConfig.
    """
    cfg = TrainConfig()
    cfg_dict = ckpt.get("config", {}) or {}
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def _build_model_from_ckpt(ckpt: dict, device: torch.device) -> Tuple[BPRMF, TrainConfig, int, int]:
    """
    Восстанавливаем модель BPRMF + признаки так же, как при обучении.
    """
    if ckpt.get("model_type", "bprmf") != "bprmf":
        raise ValueError(f"Неожиданный model_type={ckpt.get('model_type')}. Ожидается 'bprmf'.")

    cfg = _load_cfg_from_ckpt(ckpt)

    num_users = int(ckpt["num_users"])
    num_items = int(ckpt["num_items"])

    feat2idx = ckpt.get("feat2idx", {}) or {}
    item_feat_np = ckpt.get("item_feat_mat", None)

    # В инференсе dropout признаков отключаем
    num_item_feats = int(len(feat2idx))
    item_feat_t = None
    if item_feat_np is not None:
        item_feat_t = torch.tensor(np.asarray(item_feat_np), dtype=torch.long, device=device)

    model = BPRMF(
        num_users=num_users,
        num_items=num_items,
        emb_dim=int(getattr(cfg, "embedding_dim", 64)),
        num_item_feats=num_item_feats,
        item_feat_mat=item_feat_t,
        feature_dropout=0.0,  # важно: в инференсе 0
        feature_scale=float(getattr(cfg, "feature_scale", 0.2)),
        feature_norm=str(getattr(cfg, "feature_norm", "mean")),
    ).to(device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, cfg, num_users, num_items


def _read_csv_pipe_chunks(path: str, chunksize: int = 500_000):
    """
    Потоковое чтение ваших pipe-separated файлов.
    """
    return pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig", chunksize=chunksize)


def _build_user_seen_sets(
    data_dir: str,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    chunksize: int = 500_000,
) -> List[set]:
    """
    Собираем для каждого пользователя множество i_idx всех товаров,
    с которыми он взаимодействовал (заказы/избранное/просмотры номенклатуры).
    """
    num_users = len(user2idx)
    seen: List[set] = [set() for _ in range(num_users)]

    def _process_frame(df: pd.DataFrame, filter_views: bool = False) -> None:
        if df is None or df.empty:
            return
        if filter_views:
            if "ТипТовара" not in df.columns:
                return
            df = df[df["ТипТовара"] == "Номенклатура"]
            if df.empty:
                return

        if "MindboxID" not in df.columns or "КодНоменклатуры" not in df.columns:
            return

        u = df["MindboxID"].astype(str).map(user2idx)
        i = df["КодНоменклатуры"].astype(str).map(item2idx)

        tmp = pd.DataFrame({"u": u, "i": i}).dropna()
        if tmp.empty:
            return

        tmp["u"] = tmp["u"].astype(int)
        tmp["i"] = tmp["i"].astype(int)

        # обновляем множества пачками (уникальные товары на пользователя в рамках chunk)
        grp = tmp.groupby("u")["i"].unique()
        for uu, arr in grp.items():
            seen[int(uu)].update(arr.tolist())

    # Заказы
    orders_path = _path_csv(data_dir, "Заказы")
    if os.path.isfile(orders_path):
        for chunk in _read_csv_pipe_chunks(orders_path, chunksize=chunksize):
            _process_frame(chunk, filter_views=False)

    # Избранное
    fav_path = _path_csv(data_dir, "Избранное")
    if os.path.isfile(fav_path):
        for chunk in _read_csv_pipe_chunks(fav_path, chunksize=chunksize):
            _process_frame(chunk, filter_views=False)

    # Просмотры (только ТипТовара == "Номенклатура")
    views_path = _path_csv(data_dir, "Просмотры")
    if os.path.isfile(views_path):
        for chunk in _read_csv_pipe_chunks(views_path, chunksize=chunksize):
            _process_frame(chunk, filter_views=True)

    return seen


def _load_historical_item_conversion(
    data_dir: str,
    item_kind_by_code: Dict[str, str],
    window_days: int = 30,
    prior_strength: float = 20.0,
    chunksize: int = 500_000,
) -> Tuple[Dict[str, float], float]:
    """
    Рассчитывает историческую конверсию товара из просмотра в покупку.

    Конверсия считается по уникальным парам пользователь-товар:
      - пользователь просмотрел карточку номенклатуры;
      - затем купил этот же товар не позднее window_days дней.

    Для устойчивости применяется байесовское сглаживание относительно
    средней конверсии вида номенклатуры. Для новых товаров без просмотров
    используется средняя конверсия их вида, а при её отсутствии — общая.

    Возвращает:
      КодНоменклатуры -> конверсия в процентах;
      общая конверсия в процентах.
    """
    views_path = _path_csv(data_dir, "Просмотры")
    orders_path = _path_csv(data_dir, "Заказы")
    nom_path = _path_csv(data_dir, "Номенклатура")

    if not os.path.isfile(views_path) or not os.path.isfile(orders_path):
        print(f"[{_now()}] Историческая конверсия не рассчитана: отсутствуют Просмотры.csv или Заказы.csv.")
        return {}, 0.0

    def _mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    cache_key = (
        os.path.abspath(views_path), _mtime(views_path),
        os.path.abspath(orders_path), _mtime(orders_path),
        os.path.abspath(nom_path), _mtime(nom_path),
        int(window_days), float(prior_strength),
    )

    cache = getattr(_load_historical_item_conversion, "_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache.get("values", {}), float(cache.get("global", 0.0))

    def _clean_id(s: pd.Series) -> pd.Series:
        return (
            s.astype("string")
            .fillna("")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    def _parse_dates(s: pd.Series) -> pd.Series:
        raw = s.astype("string").fillna("").str.strip()
        try:
            return pd.to_datetime(raw, errors="coerce", dayfirst=True, format="mixed")
        except TypeError:
            return pd.to_datetime(raw, errors="coerce", dayfirst=True)

    print(f"\n[{_now()}] Расчёт исторической конверсии товаров:")
    print(f"- Окно после просмотра: {int(window_days)} дней")
    print(f"- Сглаживание: {float(prior_strength):g} условных наблюдений")

    # 1. Первая дата просмотра для каждой уникальной пары пользователь-товар.
    view_parts: List[pd.DataFrame] = []
    view_columns = {"MindboxID", "КодНоменклатуры", "Дата", "ТипТовара"}

    for chunk in pd.read_csv(
        views_path,
        sep="|",
        dtype=str,
        encoding="utf-8-sig",
        usecols=lambda c: c in view_columns,
        chunksize=chunksize,
    ):
        if chunk.empty or not {"MindboxID", "КодНоменклатуры", "Дата"}.issubset(chunk.columns):
            continue

        if "ТипТовара" in chunk.columns:
            chunk = chunk[
                chunk["ТипТовара"].astype("string").fillna("").str.strip().eq("Номенклатура")
            ]
            if chunk.empty:
                continue

        part = pd.DataFrame({
            "MindboxID": _clean_id(chunk["MindboxID"]),
            "КодНоменклатуры": _clean_id(chunk["КодНоменклатуры"]),
            "ДатаПросмотра": _parse_dates(chunk["Дата"]),
        })

        part = part[
            part["MindboxID"].ne("")
            & part["КодНоменклатуры"].ne("")
            & part["ДатаПросмотра"].notna()
        ]
        if part.empty:
            continue

        part = (
            part.groupby(["MindboxID", "КодНоменклатуры"], as_index=False, sort=False)["ДатаПросмотра"]
            .min()
        )
        view_parts.append(part)

    if not view_parts:
        print(f"[{_now()}] Историческая конверсия не рассчитана: нет корректных просмотров номенклатуры.")
        return {}, 0.0

    first_views = pd.concat(view_parts, ignore_index=True)
    del view_parts

    first_views = (
        first_views.groupby(["MindboxID", "КодНоменклатуры"], as_index=False, sort=False)["ДатаПросмотра"]
        .min()
    )

    viewer_counts = first_views.groupby("КодНоменклатуры", sort=False).size().astype(np.int64)
    first_view_series = first_views.set_index(["MindboxID", "КодНоменклатуры"])["ДатаПросмотра"]

    # 2. Ищем покупку после просмотра в пределах заданного окна.
    converted_parts: List[pd.DataFrame] = []
    order_columns = {"MindboxID", "КодНоменклатуры", "Дата"}
    max_delta = pd.Timedelta(days=int(window_days))

    for chunk in pd.read_csv(
        orders_path,
        sep="|",
        dtype=str,
        encoding="utf-8-sig",
        usecols=lambda c: c in order_columns,
        chunksize=chunksize,
    ):
        if chunk.empty or not order_columns.issubset(chunk.columns):
            continue

        part = pd.DataFrame({
            "MindboxID": _clean_id(chunk["MindboxID"]),
            "КодНоменклатуры": _clean_id(chunk["КодНоменклатуры"]),
            "ДатаПокупки": _parse_dates(chunk["Дата"]),
        })

        part = part[
            part["MindboxID"].ne("")
            & part["КодНоменклатуры"].ne("")
            & part["ДатаПокупки"].notna()
        ]
        if part.empty:
            continue

        keys = pd.MultiIndex.from_frame(part[["MindboxID", "КодНоменклатуры"]])
        part["ДатаПросмотра"] = first_view_series.reindex(keys).to_numpy()

        valid = (
            part["ДатаПросмотра"].notna()
            & part["ДатаПокупки"].ge(part["ДатаПросмотра"])
            & part["ДатаПокупки"].le(part["ДатаПросмотра"] + max_delta)
        )

        if valid.any():
            converted_parts.append(
                part.loc[valid, ["MindboxID", "КодНоменклатуры"]].drop_duplicates()
            )

    if converted_parts:
        converted_pairs = pd.concat(converted_parts, ignore_index=True).drop_duplicates()
        buyer_counts = converted_pairs.groupby("КодНоменклатуры", sort=False).size().astype(np.int64)
        converted_pair_count = int(len(converted_pairs))
    else:
        buyer_counts = pd.Series(dtype=np.int64)
        converted_pair_count = 0

    # 3. Байесовское сглаживание по виду номенклатуры.
    stats_index = viewer_counts.index.union(buyer_counts.index)
    stats = pd.DataFrame(index=stats_index)
    stats["Просмотры"] = viewer_counts.reindex(stats_index).fillna(0).astype(float)
    stats["Покупатели"] = buyer_counts.reindex(stats_index).fillna(0).astype(float)
    stats["ВидНоменклатуры"] = [
        _norm_text(item_kind_by_code.get(str(code), ""))
        for code in stats.index
    ]

    total_viewers = float(stats["Просмотры"].sum())
    total_buyers = float(stats["Покупатели"].sum())
    global_rate = total_buyers / total_viewers if total_viewers > 0 else 0.0

    kind_stats = (
        stats[stats["ВидНоменклатуры"].ne("")]
        .groupby("ВидНоменклатуры", sort=False)[["Просмотры", "Покупатели"]]
        .sum()
    )
    kind_rates = {
        str(kind): (float(row["Покупатели"]) / float(row["Просмотры"]))
        for kind, row in kind_stats.iterrows()
        if float(row["Просмотры"]) > 0
    }

    stats["PriorRate"] = [
        kind_rates.get(str(kind), global_rate)
        for kind in stats["ВидНоменклатуры"]
    ]
    stats["Conversion"] = (
        stats["Покупатели"] + float(prior_strength) * stats["PriorRate"]
    ) / (
        stats["Просмотры"] + float(prior_strength)
    )

    conversion_by_code: Dict[str, float] = {
        str(code): round(float(rate) * 100.0, 4)
        for code, rate in stats["Conversion"].items()
    }

    # Новая номенклатура без собственной истории получает среднее по своему виду.
    for code, kind in item_kind_by_code.items():
        code_s = str(code).strip()
        if not code_s or code_s in conversion_by_code:
            continue
        kind_n = _norm_text(kind)
        conversion_by_code[code_s] = round(kind_rates.get(kind_n, global_rate) * 100.0, 4)

    global_pct = round(global_rate * 100.0, 4)

    print(f"- Уникальных пар пользователь-товар с просмотром: {len(first_views):,}".replace(",", " "))
    print(f"- Покупок после просмотра в окне: {converted_pair_count:,}".replace(",", " "))
    print(f"- Общая историческая конверсия: {global_pct:.2f}%\n")

    _load_historical_item_conversion._cache = {
        "key": cache_key,
        "values": conversion_by_code,
        "global": global_pct,
    }
    return conversion_by_code, global_pct


# -------------------------------------------ВЫГРУЗКА В ЭКСЕЛЬ----------------------------------------------------------
@torch.no_grad()
def export_recommendations_excel(
    out_xlsx: str = "Модель/Рекомендации.xlsx",
    k: Optional[int] = None,
    model_dir: str = "Модель",
    include_item_names: bool = True,
    include_scores: bool = True,
    include_discount_card: bool = True,
    include_email: bool = True,
    include_phone: bool = True,
    filter_seen: bool = True,
    batch_users: int = 1024,
    chunksize_seen: int = 500_000,
    out_csv_format1: Optional[str] = "Модель/Рекомендации_format1.csv",
    out_csv_kanzler_ml: Optional[str] = "Модель/Kanzler.ML.csv",
    export_item_kinds: Optional[List[str]] = None,
    max_export_users: Optional[int] = 1000,
    device_str: str = "cuda",
) -> str:

    # Обработка колонок в нормальный вид
    def _clean_str_series(s: "pd.Series") -> "pd.Series":
        s = s.astype("string").str.strip()
        s = s.str.replace(r"\.0$", "", regex=True)
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        return s

    def _clean_name(v) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null", "<na>", "-"):
            return ""
        return s

    def _load_user_fields(data_dir: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        orders_path = _path_csv(data_dir, "Заказы")
        if not os.path.isfile(orders_path):
            return {}, {}, {}

        df = _read_csv_pipe(orders_path)
        if df is None or df.empty or "MindboxID" not in df.columns:
            return {}, {}, {}

        card_candidates = ["ДисконтнаяКарта"]
        email_candidates = ["Почта"]
        phone_candidates = ["Телефон"]

        card_col = next((c for c in card_candidates if c in df.columns), None)
        email_col = next((c for c in email_candidates if c in df.columns), None)
        phone_col = next((c for c in phone_candidates if c in df.columns), None)

        cols = ["MindboxID"]
        if card_col: cols.append(card_col)
        if email_col: cols.append(email_col)
        if phone_col: cols.append(phone_col)

        tmp = df[cols].copy()
        tmp["MindboxID"] = _clean_str_series(tmp["MindboxID"])

        cards: Dict[str, str] = {}
        emails: Dict[str, str] = {}
        phones: Dict[str, str] = {}

        if card_col:
            tmp[card_col] = _clean_str_series(tmp[card_col])
            t = tmp[["MindboxID", card_col]].dropna()
            if not t.empty:
                cards = t.groupby("MindboxID")[card_col].first().astype(str).to_dict()

        if email_col:
            tmp[email_col] = _clean_str_series(tmp[email_col])
            t = tmp[["MindboxID", email_col]].dropna()
            if not t.empty:
                emails = t.groupby("MindboxID")[email_col].first().astype(str).to_dict()

        if phone_col:
            tmp[phone_col] = _clean_str_series(tmp[phone_col])
            t = tmp[["MindboxID", phone_col]].dropna()
            if not t.empty:
                phones = t.groupby("MindboxID")[phone_col].first().astype(str).to_dict()

        return cards, emails, phones

    def _normalize_phone(value) -> str:
        """
        Приводит телефон к виду 79XXXXXXXXX.

        Примеры:
          +7 (916) 123-45-67 -> 79161234567
          8 916 123-45-67   -> 79161234567
          9161234567        -> 79161234567
        """
        digits = re.sub(r"\D+", "", str(value or ""))

        if len(digits) == 10:
            digits = "7" + digits
        elif len(digits) == 11 and digits.startswith("8"):
            digits = "7" + digits[1:]

        return digits

    def _rank_users_by_loyalty(
            data_dir: str,
            user2idx: Dict[str, int],
            idx2user: List[str],
            phones: Dict[str, str],
            chunksize: int = 500_000,
            require_phone: bool = True,
    ) -> List[int]:
        """
        Возвращает индексы пользователей, отсортированные от наиболее
        лояльных к наименее лояльным.

        Используется взвешенный показатель:
          покупки   * w_purchase;
          избранное * w_favorite;
          просмотры * w_view_item.

        Для покупок учитывается Количество, ограниченное от 1 до 10.
        Просмотры учитываются только для ТипТовара = Номенклатура.
        """
        users_count = len(idx2user)

        purchase_activity = np.zeros(users_count, dtype=np.float64)
        favorite_activity = np.zeros(users_count, dtype=np.float64)
        view_activity = np.zeros(users_count, dtype=np.float64)

        def _accumulate_activity(
                file_name: str,
                target: np.ndarray,
                use_quantity: bool = False,
                only_nomenclature_views: bool = False,
        ) -> None:
            path = _path_csv(data_dir, file_name)
            if not os.path.isfile(path):
                return

            needed_columns = {"MindboxID"}

            if use_quantity:
                needed_columns.add("Количество")

            if only_nomenclature_views:
                needed_columns.add("ТипТовара")

            reader = pd.read_csv(
                path,
                sep="|",
                dtype=str,
                encoding="utf-8-sig",
                usecols=lambda column: column in needed_columns,
                chunksize=chunksize,
            )

            for chunk in reader:
                if chunk.empty or "MindboxID" not in chunk.columns:
                    continue

                # В просмотрах учитываем только карточки номенклатуры,
                # а не категории и прочие сущности.
                if (
                        only_nomenclature_views
                        and "ТипТовара" in chunk.columns
                ):
                    chunk = chunk[
                        chunk["ТипТовара"]
                        .astype(str)
                        .str.strip()
                        .eq("Номенклатура")
                    ]

                    if chunk.empty:
                        continue

                mapped_users = (
                    _clean_str_series(chunk["MindboxID"])
                    .map(user2idx)
                )

                valid_mask = mapped_users.notna()
                if not valid_mask.any():
                    continue

                user_indices = (
                    mapped_users.loc[valid_mask]
                    .astype(np.int64)
                    .to_numpy()
                )

                if use_quantity and "Количество" in chunk.columns:
                    values = (
                        pd.to_numeric(
                            chunk.loc[valid_mask, "Количество"],
                            errors="coerce",
                        )
                        .fillna(1)
                        .clip(lower=1, upper=10)
                        .astype(float)
                        .to_numpy()
                    )

                    grouped = (
                        pd.DataFrame({
                            "u_idx": user_indices,
                            "value": values,
                        })
                        .groupby("u_idx", sort=False)["value"]
                        .sum()
                    )
                else:
                    grouped = (
                        pd.Series(user_indices)
                        .value_counts(sort=False)
                    )

                grouped_indices = grouped.index.to_numpy(dtype=np.int64)
                grouped_values = grouped.to_numpy(dtype=np.float64)

                target[grouped_indices] += grouped_values

        # Покупки имеют наибольший приоритет.
        _accumulate_activity(
            file_name="Заказы",
            target=purchase_activity,
            use_quantity=True,
        )

        _accumulate_activity(
            file_name="Избранное",
            target=favorite_activity,
        )

        _accumulate_activity(
            file_name="Просмотры",
            target=view_activity,
            only_nomenclature_views=True,
        )

        purchase_weight = float(getattr(cfg, "w_purchase", 10.0))
        favorite_weight = float(getattr(cfg, "w_favorite", 2.0))
        view_weight = float(getattr(cfg, "w_view_item", 0.1))

        loyalty_scores = (
                purchase_activity * purchase_weight
                + favorite_activity * favorite_weight
                + view_activity * view_weight
        )

        total_activity = (
                purchase_activity
                + favorite_activity
                + view_activity
        )

        # Сначала оставляем пользователей хотя бы с одним взаимодействием.
        eligible_indices = np.flatnonzero(total_activity > 0)

        # Для InternetMagazin.csv нужен непустой телефон.
        # Поэтому при формировании этого файла исключаем клиентов,
        # которых интернет-магазин не сможет идентифицировать.
        if require_phone:
            valid_phone_mask = np.fromiter(
                (
                    bool(
                        _normalize_phone(
                            phones.get(str(mindbox_id), "")
                        )
                    )
                    for mindbox_id in idx2user
                ),
                dtype=bool,
                count=users_count,
            )

            eligible_indices = eligible_indices[
                valid_phone_mask[eligible_indices]
            ]

        if eligible_indices.size == 0:
            return []

        # np.lexsort использует последний ключ как главный:
        # 1. LoyaltyScore;
        # 2. покупки;
        # 3. избранное;
        # 4. просмотры.
        order = np.lexsort((
            -view_activity[eligible_indices],
            -favorite_activity[eligible_indices],
            -purchase_activity[eligible_indices],
            -loyalty_scores[eligible_indices],
        ))

        ranked_indices = eligible_indices[order].astype(int).tolist()

        print(f"\n[{_now()}] Отбор наиболее лояльных клиентов:")
        print(
            f"- Пользователей в модели: "
            f"{users_count:,}".replace(",", " ")
        )
        print(
            f"- Пользователей с историей"
            f"{' и номером телефона' if require_phone else ''}: "
            f"{len(ranked_indices):,}".replace(",", " ")
        )

        return ranked_indices

    mappings_path = os.path.join(model_dir, "mappings.json")
    ckpt_path = os.path.join(model_dir, "bprmf.pt")

    if not (os.path.isfile(mappings_path) and os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"Не найдены файлы модели: {mappings_path} и/или {ckpt_path}")

    with open(mappings_path, "r", encoding="utf-8") as f:
        maps_json = json.load(f)

    idx2user: List[str] = maps_json["idx2user"]
    idx2item: List[str] = maps_json["idx2item"]
    user2idx = {u: i for i, u in enumerate(idx2user)}
    item2idx = {it: i for i, it in enumerate(idx2item)}

    ckpt = torch.load(ckpt_path, map_location="cpu")

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    model, cfg, num_users, num_items = _build_model_from_ckpt(ckpt, device)

    # --------- подготовка данных для сезонного маппинга ---------
    train_item_meta: Dict[str, Dict[str, str]] = ckpt.get("train_item_meta", {}) or {}

    # выбранные пользователем актуальные коллекции (например: "Весна-Лето 2026", "Осень-Зима 2026")
    selected_active_collections = _load_selected_collections_from_settings()

    # строим target по базовому сезону: "ВЕСНА-ЛЕТО" -> (коллекция, год)
    active_by_base: Dict[str, Tuple[str, int]] = {}
    for coll in selected_active_collections:
        base, year = _parse_season_base_and_year(coll)
        if base and year:
            prev = active_by_base.get(base)
            if prev is None or year > prev[1]:
                active_by_base[base] = (coll, year)

    # читаем актуальную номенклатуру (текущая ВходныеДанные)
    catalog_path = os.path.join(os.getcwd(), "ВходныеДанные", "Номенклатура.csv")
    new_meta_by_code: Dict[str, Dict[str, str]] = {}
    codes_by_collection: Dict[str, List[str]] = {}
    index_by_collection_key: Dict[str, Dict[Tuple[str, str, str], List[str]]] = {}

    if os.path.isfile(catalog_path):
        nom_cur = _read_csv_pipe(catalog_path)
        nom_cur.columns = [str(c).replace("\ufeff", "").strip() for c in nom_cur.columns]

        if "КодНоменклатуры" in nom_cur.columns:
            want_cols = [
                "КодНоменклатуры", "Коллекция", "НазваниеНаСайте", "Номенклатура",
                "ВидНоменклатуры", "ПолНоменклатуры", "КатегорияНаСайте", "СтилеваяГруппа",
                "Марка", "ГруппаСоставов", "ВидАссортимента", "Остаток"
            ]
            cols = [c for c in want_cols if c in nom_cur.columns]
            sub = nom_cur[cols].copy()
            sub["КодНоменклатуры"] = sub["КодНоменклатуры"].astype(str)
            sub = sub.drop_duplicates("КодНоменклатуры", keep="last").set_index("КодНоменклатуры")
            new_meta_by_code = sub.to_dict(orient="index")

            # индекс: коллекция -> коды
            for code, meta in new_meta_by_code.items():
                coll = _norm_text(meta.get("Коллекция", ""))
                if coll:
                    codes_by_collection.setdefault(coll, []).append(code)

            # индекс для ускорения: (ВидНоменклатуры, ПолНоменклатуры, КатегорияНаСайте) -> list[codes]
            for coll, codes in codes_by_collection.items():
                d = {}
                for code in codes:
                    meta = new_meta_by_code.get(code, {}) or {}
                    key = (
                        _norm_text(meta.get("ВидНоменклатуры", "")),
                        _norm_text(meta.get("ПолНоменклатуры", "")),
                        _norm_text(meta.get("КатегорияНаСайте", "")),
                    )
                    d.setdefault(key, []).append(code)
                index_by_collection_key[coll] = d

    # кэш "старый код -> новый код"
    old_to_new_cache: Dict[str, str] = {}

    def _map_old_code_to_active(code_old: str) -> str:
        """
        Правила:
          - несезонные коллекции -> оставляем как есть
          - сезонные (ВЛ/ОЗ) с годом меньше активного -> маппим в активную коллекцию того же типа сезона
          - если данных не хватает/кандидатов нет -> оставляем как есть
        """
        if code_old in old_to_new_cache:
            return old_to_new_cache[code_old]

        old_meta = train_item_meta.get(code_old, {}) or {}
        coll_old = _norm_text(old_meta.get("Коллекция", ""))

        # несезонные коллекции не трогаем
        if _is_nonseasonal_collection(coll_old):
            old_to_new_cache[code_old] = code_old
            return code_old

        base_old, year_old = _parse_season_base_and_year(coll_old)
        if not base_old or not year_old:
            old_to_new_cache[code_old] = code_old
            return code_old

        tgt = active_by_base.get(base_old)
        if not tgt:
            old_to_new_cache[code_old] = code_old
            return code_old

        coll_new, year_new = tgt
        if year_old >= year_new:
            old_to_new_cache[code_old] = code_old
            return code_old

        # кандидаты из нужной актуальной коллекции
        coll_new_n = _norm_text(coll_new)
        cand_codes = codes_by_collection.get(coll_new_n, [])
        if not cand_codes:
            old_to_new_cache[code_old] = code_old
            return code_old

        # ускоряем: сначала пытаемся подобрать среди кандидатов с тем же (вид, пол, категория)
        key = (
            _norm_text(old_meta.get("ВидНоменклатуры", "")),
            _norm_text(old_meta.get("ПолНоменклатуры", "")),
            _norm_text(old_meta.get("КатегорияНаСайте", "")),
        )
        cand_fast = index_by_collection_key.get(coll_new_n, {}).get(key, [])
        candidates = cand_fast if cand_fast else cand_codes

        best_code = None
        best_score = -1e9
        for cc in candidates:
            new_meta = new_meta_by_code.get(cc, {}) or {}
            sc = _similarity_score(old_meta, new_meta)
            if sc > best_score:
                best_score = sc
                best_code = cc

        out = best_code if best_code else code_old
        old_to_new_cache[code_old] = out
        return out

    def _kind_for_export(
            code_out: str,
            code_old: str = "",
    ) -> str:
        """
        Возвращает ВидНоменклатуры итогового товара после сезонного
        сопоставления.
        """
        code_out = str(code_out or "").strip()
        code_old = str(code_old or "").strip()

        if code_out:
            current_meta = new_meta_by_code.get(code_out, {}) or {}
            kind = _clean_name(
                current_meta.get("ВидНоменклатуры")
            )
            if kind:
                return kind

            trained_meta = train_item_meta.get(code_out, {}) or {}
            kind = _clean_name(
                trained_meta.get("ВидНоменклатуры")
            )
            if kind:
                return kind

        if code_old:
            old_meta = train_item_meta.get(code_old, {}) or {}
            kind = _clean_name(
                old_meta.get("ВидНоменклатуры")
            )
            if kind:
                return kind

        return ""

    if k is None:
        k = int(getattr(cfg, "topk", 10))
    csv_min_k = 10 if (out_csv_format1 or out_csv_kanzler_ml) else 1
    k = max(int(csv_min_k), int(k))
    k = max(1, min(int(k), num_items))

    data_dir = getattr(cfg, "data_dir", "ВходныеДанные")

    data_dir = getattr(cfg, "data_dir", "ВходныеДанные")

    # Нормализуем выбранные пользователем виды номенклатуры.
    # Пустое множество означает, что ограничение не применяется.
    export_kind_names = sorted({
        _norm_text(value)
        for value in (export_item_kinds or [])
        if _norm_text(value)
    })

    export_kind_keys = {
        value.upper()
        for value in export_kind_names
    }

    if export_kind_names:
        print(
            f"[{_now()}] Виды номенклатуры в итоговой выгрузке: "
            + ", ".join(export_kind_names)
        )
    else:
        print(
            f"[{_now()}] Ограничение по виду номенклатуры "
            f"в итоговой выгрузке не установлено."
        )

    item_names: Dict[str, str] = (
        _load_item_names(data_dir)
        if include_item_names
        else {}
    )

    item_names: Dict[str, str] = _load_item_names(data_dir) if include_item_names else {}

    # Остатки берём из текущей номенклатуры, потому что после сезонного сопоставления
    # код товара может быть заменён на товар из актуальной коллекции.
    current_data_dir = os.path.join(os.getcwd(), "ВходныеДанные")
    item_stocks: Dict[str, str] = _load_item_stocks(current_data_dir)

    # fallback на cfg.data_dir
    if not item_stocks:
        item_stocks = _load_item_stocks(data_dir)

    # Вид номенклатуры нужен для сглаживания конверсии и fallback новых товаров.
    item_kind_by_code: Dict[str, str] = {}
    for code, meta in train_item_meta.items():
        item_kind_by_code[str(code)] = _norm_text((meta or {}).get("ВидНоменклатуры", ""))
    for code, meta in new_meta_by_code.items():
        item_kind_by_code[str(code)] = _norm_text((meta or {}).get("ВидНоменклатуры", ""))

    # Для расчёта используем только текущие обработанные файлы.
    conversion_data_dir = current_data_dir
    if not (
        os.path.isfile(_path_csv(conversion_data_dir, "Просмотры"))
        and os.path.isfile(_path_csv(conversion_data_dir, "Заказы"))
    ):
        conversion_data_dir = data_dir

    historical_conversion_by_code, historical_conversion_global = _load_historical_item_conversion(
        data_dir=conversion_data_dir,
        item_kind_by_code=item_kind_by_code,
        window_days=30,
        prior_strength=20.0,
        chunksize=chunksize_seen,
    )

    discount_cards: Dict[str, str] = {}
    emails: Dict[str, str] = {}
    phones: Dict[str, str] = {}
    need_csv = bool(out_csv_format1) or bool(out_csv_kanzler_ml)

    if include_discount_card or include_email or include_phone or need_csv:
        discount_cards, emails, phones = _load_user_fields(data_dir)

    # Ограничение количества клиентов в итоговой выгрузке.
    # None или значение <= 0 означает выгрузку всех подходящих клиентов.
    export_user_limit: Optional[int] = None

    if max_export_users is not None:
        try:
            parsed_limit = int(max_export_users)
            if parsed_limit > 0:
                export_user_limit = parsed_limit
        except (TypeError, ValueError):
            export_user_limit = None

    # Клиенты располагаются от наиболее лояльных к наименее лояльным.
    # Если создаётся InternetMagazin.csv, участвуют только клиенты
    # с корректным номером телефона.
    ranked_user_indices = _rank_users_by_loyalty(
        data_dir=data_dir,
        user2idx=user2idx,
        idx2user=idx2user,
        phones=phones,
        chunksize=chunksize_seen,
        require_phone=bool(out_csv_format1),
    )

    if not ranked_user_indices:
        raise ValueError(
            "Не найдены пользователи с историей взаимодействий "
            "и корректным номером телефона."
        )

    user_seen: Optional[List[set]] = None

    if filter_seen:
        user_seen = _build_user_seen_sets(
            data_dir=data_dir,
            user2idx=user2idx,
            item2idx=item2idx,
            chunksize=chunksize_seen,
        )

    item_vec = model.item_vec_all().to(device)  # [I, d]

    # Маска товаров, разрешённых выбранным видом номенклатуры.
    # Она применяется до torch.topk, поэтому модель будет выбиратьАМ
    # лучшие рекомендации именно среди выбранных видов.
    export_kind_mask = None

    if export_kind_keys:
        allowed_items = np.zeros(
            num_items,
            dtype=np.bool_
        )

        for item_index, old_code_value in enumerate(idx2item):
            code_old = str(old_code_value)
            code_out = _map_old_code_to_active(code_old)

            kind_value = _kind_for_export(
                code_out,
                code_old
            )

            kind_key = _norm_text(kind_value).upper()

            if kind_key in export_kind_keys:
                allowed_items[item_index] = True

        allowed_count = int(allowed_items.sum())

        if allowed_count == 0:
            raise ValueError(
                "В модели не найдено товаров выбранных видов "
                "номенклатуры: "
                + ", ".join(export_kind_names)
            )

        export_kind_mask = torch.tensor(
            allowed_items,
            dtype=torch.bool,
            device=device,
        )

        print(
            f"[{_now()}] Товаров выбранных видов, доступных "
            f"для ранжирования: {allowed_count:,}"
            .replace(",", " ")
        )

    os.makedirs(os.path.dirname(out_xlsx) or ".", exist_ok=True)
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Рекомендации")

    header = ["MindboxID"]
    if include_discount_card:
        header.append("ДисконтнаяКарта")
    if include_email:
        header.append("Почта")
    if include_phone:
        header.append("Телефон")

    for r in range(1, k + 1):
        header.append(f"КодНоменклатуры_{r}")
        if include_item_names:
            header.append(f"НазваниеНоменклатуры_{r}")
        header.append(f"Коллекция_{r}")
        if include_scores:
            header.append(f"Коэффициент_{r}")
        header.append(f"Конверсия_{r}")
        header.append(f"Остаток_{r}")

    ws.append(header)

    # --- Статистика по коллекциям в итоговом Excel-файле ---
    collection_counts: Dict[str, int] = {}

    def _add_collection_to_stats(collection_value: str) -> None:
        coll = _clean_name(collection_value)

        if not coll:
            coll = "Без коллекции"

        collection_counts[coll] = collection_counts.get(coll, 0) + 1

    # --- CSV outputs (additional to Excel) ---
    csv1_f = csvml_f = None
    csv1_w = csvml_w = None

    if out_csv_format1:
        os.makedirs(os.path.dirname(out_csv_format1) or ".", exist_ok=True)
        csv1_f = open(out_csv_format1, "w", newline="", encoding="utf-8-sig")
        csv1_w = csv.writer(csv1_f, delimiter=";")
        csv1_w.writerow(["CustomerID", "ProductID"])

    if out_csv_kanzler_ml:
        os.makedirs(os.path.dirname(out_csv_kanzler_ml) or ".", exist_ok=True)
        csvml_f = open(out_csv_kanzler_ml, "w", newline="", encoding="utf-8-sig")
        csvml_w = csv.writer(csvml_f, delimiter=";")
        csvml_w.writerow(["CustomerMindboxId", "Quantity", "ProductGroupOffline1C", "CustomFieldKoefficient"])

    exported_users = 0
    stop_export = False

    for batch_start in range(
            0,
            len(ranked_user_indices),
            batch_users,
    ):
        if stop_export:
            break

        batch_user_indices = ranked_user_indices[
                             batch_start:batch_start + batch_users
                             ]

        if not batch_user_indices:
            continue

        u_idx = torch.tensor(
            batch_user_indices,
            device=device,
            dtype=torch.long,
        )

        u_emb = model.user_emb(u_idx)  # [B, d]
        scores = u_emb @ item_vec.t()  # [B, I]

        if filter_seen and user_seen is not None:
            for bi, uu in enumerate(batch_user_indices):
                seen_items = user_seen[uu]

                if seen_items:
                    scores[
                        bi,
                        torch.tensor(
                            list(seen_items),
                            device=device,
                            dtype=torch.long,
                        )
                    ] = -1e9

        # Убираем из ранжирования все виды номенклатуры,
        # которые пользователь не выбрал для выгрузки.
        if export_kind_mask is not None:
            scores[:, ~export_kind_mask] = -1e9

        # Берём больше кандидатов, потому что часть товаров будет отсеяна по остатку < 100
        cand_k = min(scores.shape[1], max(k * 100, 1000, k))
        top = torch.topk(scores, k=cand_k, dim=1)
        top_idx = top.indices.detach().cpu().numpy()
        top_val = top.values.detach().cpu().numpy()

        for bi, uu in enumerate(batch_user_indices):
            if (
                    export_user_limit is not None
                    and exported_users >= export_user_limit
            ):
                stop_export = True
                break

            mindbox_id = str(idx2user[uu])

            row: List[object] = [mindbox_id]

            if include_discount_card:
                row.append(str(discount_cards.get(mindbox_id, "")))
            if include_email:
                row.append(str(emails.get(mindbox_id, "")))
            if include_phone:
                row.append(str(phones.get(mindbox_id, "")))

            rec_items = top_idx[bi].tolist()
            rec_scores = [float(x) for x in top_val[bi].tolist()]

            used = set()
            out_codes: List[str] = []
            out_names: List[str] = []
            out_collections: List[str] = []
            out_scores: List[float] = []
            out_conversions: List[float] = []
            out_stocks: List[str] = []

            def _stock_for_code(code: str) -> str:
                code = str(code or "").strip()
                if not code:
                    return ""

                # основной источник — текущая Номенклатура.csv
                val = item_stocks.get(code)
                if val is not None:
                    return _format_stock_value(val)

                # fallback — метаданные текущего каталога
                meta = new_meta_by_code.get(code, {}) or {}
                return _format_stock_value(meta.get("Остаток", 0))

            def _stock_to_int(stock_value) -> int:
                """
                Преобразует остаток к int для фильтрации:
                  "150" -> 150
                  "150.0" -> 150
                  "150,0" -> 150
                  пусто/NaN -> 0
                """
                try:
                    return int(_format_stock_value(stock_value))
                except Exception:
                    return 0

            def _collection_for_code(code_out: str, code_old: str = "") -> str:
                """
                Возвращает коллекцию итогового товара после сезонного сопоставления.
                Сначала ищем по итоговому коду в текущей Номенклатура.csv,
                затем в метаданных обучения.
                """
                code_out = str(code_out or "").strip()
                code_old = str(code_old or "").strip()

                if code_out:
                    meta = new_meta_by_code.get(code_out, {}) or {}
                    coll = _clean_name(meta.get("Коллекция"))
                    if coll:
                        return coll

                    meta = train_item_meta.get(code_out, {}) or {}
                    coll = _clean_name(meta.get("Коллекция"))
                    if coll:
                        return coll

                if code_old:
                    meta = train_item_meta.get(code_old, {}) or {}
                    coll = _clean_name(meta.get("Коллекция"))
                    if coll:
                        return coll

                return ""

            def _kind_for_code(
                code_out: str,
                code_old: str = "",
            ) -> str:
                return _kind_for_export(
                    code_out,
                    code_old
                )

            def _is_autumn_winter_collection(collection_value: str) -> bool:
                """
                True для любых коллекций Осень-Зима любого года:
                  Осень-Зима 2024
                  Осень-Зима 2025
                  Осень-Зима 2025 переходящий остаток
                  Осень-Зима 2026
                """
                coll = _norm_text(collection_value).upper()
                return "ОСЕНЬ" in coll and "ЗИМА" in coll

            def _is_gift_card_kind(kind_value: str) -> bool:
                """
                True для вида номенклатуры Подарочные карты.
                """
                kind = _norm_text(kind_value).upper()
                return kind == "ПОДАРОЧНЫЕ КАРТЫ" or ("ПОДАРОЧН" in kind and "КАРТ" in kind)


            ptr = 0

            while len(out_codes) < k and ptr < len(rec_items):
                code_old = str(idx2item[int(rec_items[ptr])])
                score_old = float(rec_scores[ptr])

                # Значение -1e9 используется для исключённых,
                # уже просмотренных или запрещённых товаров.
                if score_old <= -1e8:
                    ptr += 1
                    continue

                code_out = _map_old_code_to_active(code_old)

                if code_out in used:
                    ptr += 1
                    continue

                collection_out = _collection_for_code(code_out, code_old)
                kind_out = _kind_for_code(code_out, code_old)

                # Отбор по видам номенклатуры, выбранным пользователем.
                # Пустой список означает отсутствие ограничения.
                if (
                        export_kind_keys
                        and _norm_text(kind_out).upper()
                        not in export_kind_keys
                ):
                    ptr += 1
                    continue

                # Фильтр по коллекции: убираем всю Осень-Зиму любого года
                if _is_autumn_winter_collection(collection_out):
                    ptr += 1
                    continue

                # Фильтр по виду номенклатуры: убираем подарочные карты
                if _is_gift_card_kind(kind_out):
                    ptr += 1
                    continue

                stock_out = _stock_for_code(code_out)

                # Фильтр по остатку: в выгрузку попадают только товары с остатком >= 100
                if _stock_to_int(stock_out) < 100:
                    ptr += 1
                    continue

                used.add(code_out)
                out_codes.append(code_out)
                out_scores.append(score_old)
                out_collections.append(collection_out)
                out_conversions.append(
                    float(historical_conversion_by_code.get(code_out, historical_conversion_global))
                )
                out_stocks.append(stock_out)

                if include_item_names:
                    nm = _clean_name(item_names.get(code_out, ""))

                    if not nm:
                        if code_out == code_old:
                            om = train_item_meta.get(code_old, {}) or {}
                            nm = _clean_name(om.get("НазваниеНаСайте")) or _clean_name(om.get("Номенклатура"))
                        else:
                            nm = _clean_name((new_meta_by_code.get(code_out, {}) or {}).get("НазваниеНаСайте")) \
                                 or _clean_name((new_meta_by_code.get(code_out, {}) or {}).get("Номенклатура"))

                    out_names.append(nm)

                ptr += 1

            if not out_codes:
                continue

            # добиваем до k, чтобы структура Excel не ломалась
            while len(out_codes) < k:
                out_codes.append("")
                out_scores.append(0.0)
                out_collections.append("")
                out_conversions.append(0.0)
                out_stocks.append("")
                if include_item_names:
                    out_names.append("")
            # --- CSV format #1: CustomerID=discount card, ProductID=comma-separated codes ---
            # --- CSV InternetMagazin:
            # CustomerID = номер телефона,
            # ProductID = коды рекомендаций через запятую.
            if csv1_w is not None:
                customer_id = _normalize_phone(
                    phones.get(mindbox_id, "")
                )

                product_codes_csv = [
                    str(code).strip()
                    for code in out_codes[:k]
                    if str(code or "").strip()
                ]

                product_id = ",".join(product_codes_csv)

                # Пустые идентификаторы и пустые рекомендации не записываем.
                if customer_id and product_id:
                    csv1_w.writerow([
                        customer_id,
                        product_id,
                    ])

            # --- CSV Kanzler ML: one row per recommended item ---
            if csvml_w is not None:
                def _fmt_coef_ru(val: float) -> str:
                    try:
                        return f"{float(val):.2f}".replace(".", ",")
                    except Exception:
                        return ""
                for code_val, sc_val in zip(out_codes[:k], out_scores[:k]):
                    code_val = str(code_val or "")
                    if not code_val:
                        continue
                    csvml_w.writerow([mindbox_id, 1, code_val, _fmt_coef_ru(sc_val)])

            for j in range(k):
                row.append(out_codes[j])
                if include_item_names:
                    row.append(out_names[j])
                row.append(out_collections[j])
                if include_scores:
                    row.append(round(out_scores[j], 2))
                # Историческая конверсия хранится только в Excel, в процентах.
                row.append(round(out_conversions[j], 2) if out_codes[j] else "")
                row.append(out_stocks[j])

                # Считаем коллекции только по реально выгружаемым рекомендациям
            for code_val, coll_val in zip(out_codes[:k], out_collections[:k]):
                code_val = str(code_val or "").strip()

                if not code_val:
                    continue

                _add_collection_to_stats(coll_val)

            ws.append(row)
            exported_users += 1

    print(
        f"[{_now()}] В итоговые файлы выгружено клиентов: "
        f"{exported_users:,}".replace(",", " ")
    )

    if (
            export_user_limit is not None
            and exported_users < export_user_limit
    ):
        print(
            f"[{_now()}] Предупреждение: требовалось "
            f"{export_user_limit:,}, но удалось сформировать рекомендации "
            f"только для {exported_users:,} клиентов."
            .replace(",", " ")
        )
    wb.save(out_xlsx)

    print("\n[RECS COLLECTIONS] Количество рекомендаций по коллекциям в Excel:")

    if collection_counts:
        total_recs = sum(collection_counts.values())

        for coll, cnt in sorted(collection_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {coll}: {cnt}")

        print(f"  Итого рекомендаций: {total_recs}")
    else:
        print("  Нет рекомендаций для подсчёта.")

    print()

    # close CSV files
    if csv1_f is not None:
        csv1_f.close()
    if csvml_f is not None:
        csvml_f.close()

    return out_xlsx


def main(argv: Optional[List[str]] = None) -> int:
    do_train, mindbox, k, config_path = _parse_cli(
        sys.argv[1:] if argv is None else argv
    )

    if do_train:
        cfg = _load_train_config_from_json(config_path) if config_path else TrainConfig()
        trained = _train_in_this_process(cfg)
        return TRAIN_EXIT_SUCCESS if trained else TRAIN_EXIT_NO_DATA

    if mindbox is not None:
        print_recommendations(mindbox, k=k)

    return TRAIN_EXIT_SUCCESS


if __name__ == "__main__":
    cli_args = sys.argv[1:]
    if "--train" in cli_args:
        # hard-exit helps avoid rare native crashes during Python shutdown on Windows
        os._exit(main(cli_args))
    raise SystemExit(main(cli_args))
