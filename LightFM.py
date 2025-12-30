from __future__ import annotations
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import json
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scipy.sparse as sp
except Exception:
    sp = None


# ----------------------------- Config -----------------------------

@dataclass
class TrainConfig:
    data_dir: str = "ВходныеДанные"
    use_selection_if_exists: bool = True

    # Веса для обучения
    w_view_item: float = 1.0
    w_favorite: float = 3.0
    w_purchase: float = 5.0

    # BPR-MF
    embedding_dim: int = 64
    epochs: int = 20
    batch_size: int = 4096
    lr: float = 2e-3
    weight_decay: float = 1e-6
    bpr_reg: float = 1e-4
    seed: int = 42

    # Eval
    topk: int = 20
    min_user_interactions_for_eval: int = 2

    # EASE^R
    ease_lambda: float = 200.0
    max_items_for_ease: int = 15000  # guardrail


# -------------------------------------------Вспомогательные функции----------------------------------------------------

# Функция для получения текущего времени в читаемом формате.
def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# Устанавливает случайное зерно для обеспечения воспроизводимости: для NumPy, Python, PyTorch и CUDA.
def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Создаёт директорию, если её нет.
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# Функция для выбора правильного файла из каталога данных: сначала проверяется наличие файла с префиксом "Отбор".
def _pick_file(data_dir: str, base: str, selection: bool) -> str:
    if selection:
        p = os.path.join(data_dir, f"{base}Отбор.csv")
        if os.path.isfile(p):
            return p
    return os.path.join(data_dir, f"{base}Оригинал.csv")


# Читает CSV файл с разделителем | и кодировкой utf-8-sig (чтобы корректно работать с BOM).
def _read_csv_pipe(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="|", dtype=str, encoding="utf-8-sig")


# Преобразует колонку с датами в формат datetime. Если ошибка — заменяет на NaT.
def _parse_date_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NaT] * len(df))
    return pd.to_datetime(df[col], errors="coerce")


@dataclass
class Mappings:
    user2idx: Dict[str, int]
    idx2user: List[str]
    item2idx: Dict[str, int]
    idx2item: List[str]


@dataclass
class Splits:
    train_pairs: np.ndarray       # [N,2] (u,i)
    train_weights: np.ndarray     # [N]
    eval_users: np.ndarray        # [M]
    eval_items: np.ndarray        # [M]
    user_pos_train: List[set]     # per user: set(items)


# Создаёт маппинги для пользователей и товаров. Конвертирует их в индексы
def _build_mappings(orders: pd.DataFrame, views: pd.DataFrame, fav: pd.DataFrame) -> Mappings:
    users = pd.concat(
        [
            orders.get("MindboxID", pd.Series(dtype=str)),
            views.get("MindboxID", pd.Series(dtype=str)),
            fav.get("MindboxID", pd.Series(dtype=str)),
        ],
        axis=0,
    ).dropna().astype(str).unique().tolist()

    item_codes = []
    for df in (orders, fav):
        if "КодНоменклатуры" in df.columns:
            item_codes.append(df["КодНоменклатуры"])
    if {"ТипТовара", "КодНоменклатуры"}.issubset(views.columns):
        item_codes.append(views.loc[views["ТипТовара"] == "Номенклатура", "КодНоменклатуры"])

    items = pd.concat(item_codes, axis=0).dropna().astype(str).unique().tolist()

    return Mappings(
        user2idx={u: i for i, u in enumerate(users)},
        idx2user=users,
        item2idx={it: i for i, it in enumerate(items)},
        idx2item=items,
    )


# Собирает данные о взаимодействиях пользователей с товарами: покупки, избранное, просмотры.
def _collect_user_item_events(
    orders: pd.DataFrame,
    views: pd.DataFrame,
    fav: pd.DataFrame,
    maps: Mappings,
    cfg: TrainConfig,
) -> pd.DataFrame:
    frames = []

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

    if len(fav) and {"MindboxID", "КодНоменклатуры"}.issubset(fav.columns):
        f = fav[["MindboxID", "КодНоменклатуры"]].copy()
        f["ts"] = _parse_date_col(fav, "Дата")
        f["w"] = cfg.w_favorite
        f = f.dropna(subset=["MindboxID", "КодНоменклатуры"])
        f["u_idx"] = f["MindboxID"].astype(str).map(maps.user2idx)
        f["i_idx"] = f["КодНоменклатуры"].astype(str).map(maps.item2idx)
        f = f.dropna(subset=["u_idx", "i_idx"])
        frames.append(f[["u_idx", "i_idx", "ts", "w"]])

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


# Делит данные на обучающие и тестовые выборки на основе времени последнего взаимодействия каждого пользователя.
def _train_test_split_last_per_user(events: pd.DataFrame, cfg: TrainConfig, num_users: int) -> Splits:
    events_sorted = events.sort_values(["u_idx", "ts"])
    last = events_sorted.groupby("u_idx").tail(1)

    counts = events_sorted.groupby("u_idx").size()
    eligible_users = counts[counts >= cfg.min_user_interactions_for_eval].index.values

    last = last[last["u_idx"].isin(eligible_users)]
    eval_users = last["u_idx"].astype(int).to_numpy()
    eval_items = last["i_idx"].astype(int).to_numpy()

    train_ev = events_sorted.drop(index=last.index)
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


def _sample_batch(train_pairs: np.ndarray, train_weights: np.ndarray, batch_size: int, rng: np.random.Generator):
    if len(train_pairs) <= batch_size:
        return train_pairs[:, 0], train_pairs[:, 1]
    w = train_weights.astype(np.float64)
    w = w / (w.sum() + 1e-12)
    idx = rng.choice(len(train_pairs), size=batch_size, replace=False, p=w)
    return train_pairs[idx, 0], train_pairs[idx, 1]


def _sample_negatives(users: np.ndarray, num_items: int, user_pos_train: List[set], rng: np.random.Generator, max_tries: int = 25):
    neg = rng.integers(0, num_items, size=len(users), dtype=np.int64)
    for _ in range(max_tries):
        bad = np.array([n in user_pos_train[int(u)] for u, n in zip(users, neg)], dtype=bool)
        if not bad.any():
            break
        neg[bad] = rng.integers(0, num_items, size=bad.sum(), dtype=np.int64)
    return neg


# ----------------------------- BPR-MF -----------------------------

class BPRMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(u_idx)
        i = self.item_emb(i_idx)
        return (u * i).sum(dim=-1)


@torch.no_grad()
def _eval_bprmf_recall_ndcg(model: BPRMF, splits: Splits, num_items: int, k: int, device: torch.device) -> Tuple[float, float]:
    if len(splits.eval_users) == 0:
        return 0.0, 0.0

    item_emb = model.item_emb.weight  # [I, d]
    users = splits.eval_users.astype(np.int64)
    gt = splits.eval_items.astype(np.int64)

    chunk = 512
    recalls, ndcgs = [], []

    for start in range(0, len(users), chunk):
        u = users[start:start + chunk]
        g = gt[start:start + chunk]

        u_t = torch.tensor(u, dtype=torch.long, device=device)
        u_emb = model.user_emb(u_t)
        scores = u_emb @ item_emb.t()

        # filter train positives
        for bi, uu in enumerate(u):
            pos = splits.user_pos_train[int(uu)]
            if pos:
                idx = torch.tensor(list(pos), dtype=torch.long, device=device)
                scores[bi, idx] = -1e9

        topk_idx = torch.topk(scores, k=min(k, num_items), dim=1).indices.cpu().numpy()

        for bi in range(len(u)):
            target = int(g[bi])
            if target in topk_idx[bi]:
                rank = int(np.where(topk_idx[bi] == target)[0][0]) + 1
                recalls.append(1.0)
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def train_bprmf(maps: Mappings, events: pd.DataFrame, cfg: TrainConfig, device: torch.device):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    num_users = len(maps.idx2user)
    num_items = len(maps.idx2item)
    splits = _train_test_split_last_per_user(events, cfg, num_users)

    model = BPRMF(num_users, num_items, cfg.embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)

    best = {"recall": -1.0, "ndcg": -1.0, "epoch": -1, "state": None}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        steps = max(1, int(np.ceil(len(splits.train_pairs) / cfg.batch_size)))
        total_loss = 0.0

        for _ in range(steps):
            u_pos, i_pos = _sample_batch(splits.train_pairs, splits.train_weights, cfg.batch_size, rng)
            i_neg = _sample_negatives(u_pos, num_items, splits.user_pos_train, rng)

            u_t = torch.tensor(u_pos, dtype=torch.long, device=device)
            ip_t = torch.tensor(i_pos, dtype=torch.long, device=device)
            in_t = torch.tensor(i_neg, dtype=torch.long, device=device)

            s_pos = model.score(u_t, ip_t)
            s_neg = model.score(u_t, in_t)
            loss = -F.logsigmoid(s_pos - s_neg).mean()

            if cfg.bpr_reg > 0:
                reg = (model.user_emb(u_t).pow(2).sum(dim=1) +
                       model.item_emb(ip_t).pow(2).sum(dim=1) +
                       model.item_emb(in_t).pow(2).sum(dim=1)).mean()
                loss = loss + cfg.bpr_reg * reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())

        model.eval()
        recall, ndcg = _eval_bprmf_recall_ndcg(model, splits, num_items, cfg.topk, device)

        print(f"[{_now()}] BPR-MF epoch {epoch:02d}/{cfg.epochs}: "
              f"loss={total_loss/steps:.4f}  recall@{cfg.topk}={recall:.4f}  ndcg@{cfg.topk}={ndcg:.4f}")

        if recall > best["recall"]:
            best["recall"], best["ndcg"], best["epoch"] = recall, ndcg, epoch
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # restore best weights (optional; here last epoch is usually best, but keep it correct)
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    print(f"[{_now()}] BPR-MF best: epoch={best['epoch']} recall@{cfg.topk}={best['recall']:.4f} ndcg@{cfg.topk}={best['ndcg']:.4f}")
    return model, splits


# ----------------------------- EASE^R -----------------------------

def _ease_solve(X: "sp.csr_matrix", l2: float) -> np.ndarray:
    G = (X.T @ X).toarray().astype(np.float32)
    diag = np.arange(G.shape[0])
    G[diag, diag] += float(l2)
    P = np.linalg.inv(G).astype(np.float32)
    d = np.diag(P)
    B = -P / d.reshape(1, -1)
    np.fill_diagonal(B, 0.0)
    return B


def _eval_ease_one_positive(X_train: "sp.csr_matrix", B: np.ndarray, splits: Splits, k: int) -> Tuple[float, float]:
    if len(splits.eval_users) == 0:
        return 0.0, 0.0

    users = splits.eval_users.astype(np.int64)
    gt = splits.eval_items.astype(np.int64)

    recalls, ndcgs = [], []

    for u, g in zip(users, gt):
        x = X_train[int(u)]  # 1 x n
        scores = np.asarray(x @ B).ravel()  # <- safe for ndarray/matrix
        seen = x.indices
        if len(seen):
            scores[seen] = -1e9
        topk = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        topk = topk[np.argsort(-scores[topk])]
        target = int(g)
        if target in topk:
            rank = int(np.where(topk == target)[0][0]) + 1
            recalls.append(1.0)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            recalls.append(0.0)
            ndcgs.append(0.0)

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def train_ease_r(splits: Splits, num_users: int, num_items: int, cfg: TrainConfig):
    if sp is None:
        raise RuntimeError("scipy is not installed")

    if num_items > cfg.max_items_for_ease:
        raise RuntimeError(f"EASE guardrail: items={num_items} > {cfg.max_items_for_ease}")

    rows = splits.train_pairs[:, 0].astype(np.int32)
    cols = splits.train_pairs[:, 1].astype(np.int32)
    vals = splits.train_weights.astype(np.float32)

    X = sp.csr_matrix((vals, (rows, cols)), shape=(num_users, num_items))
    B = _ease_solve(X, cfg.ease_lambda)
    recall, ndcg = _eval_ease_one_positive(X, B, splits, cfg.topk)
    return B, recall, ndcg


# ----------------------------- Saving / Loading -----------------------------

def _save_artifacts(cfg: TrainConfig, maps: Mappings, model: BPRMF, splits: Splits, ease_B: Optional[np.ndarray]):
    out_dir = "Models"
    _ensure_dir(out_dir)

    with open(os.path.join(out_dir, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"idx2user": maps.idx2user, "idx2item": maps.idx2item},
            f,
            ensure_ascii=False,
        )

    ckpt = {
        "model_type": "bprmf",
        "config": cfg.__dict__,
        "num_users": len(maps.idx2user),
        "num_items": len(maps.idx2item),
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, os.path.join(out_dir, "lightgcn.pt"))

    if ease_B is not None:
        np.save(os.path.join(out_dir, "ease_B.npy"), ease_B)


def _load_artifacts():
    out_dir = "Models"
    mappings_path = os.path.join(out_dir, "mappings.json")
    ckpt_path = os.path.join(out_dir, "lightgcn.pt")
    ease_path = os.path.join(out_dir, "ease_B.npy")

    if not (os.path.isfile(mappings_path) and os.path.isfile(ckpt_path)):
        raise FileNotFoundError("Models not found. Train first (press the train button).")

    with open(mappings_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ease_B = np.load(ease_path) if os.path.isfile(ease_path) else None
    return maps, ckpt, ease_B


# ----------------------------- Recommendation (console) -----------------------------

def _load_item_names(data_dir: str, idx2item: List[str]) -> Dict[str, str]:
    """
    Returns mapping item_code -> name (if nomenclature file exists).
    """
    nom_path = os.path.join(data_dir, "Номенклатура.csv")
    if not os.path.isfile(nom_path):
        return {}
    try:
        nom = _read_csv_pipe(nom_path)
    except Exception:
        return {}

    if "КодНоменклатуры" not in nom.columns:
        return {}
    name_col = "НазваниеНаСайте" if "НазваниеНаСайте" in nom.columns else None
    if name_col is None:
        return {}

    sub = nom[["КодНоменклатуры", name_col]].dropna()
    sub["КодНоменклатуры"] = sub["КодНоменклатуры"].astype(str)
    sub[name_col] = sub[name_col].astype(str)
    return dict(zip(sub["КодНоменклатуры"].tolist(), sub[name_col].tolist()))


def _user_profile_from_processed(data_dir: str, mindbox_id: str, idx2item: List[str], item2idx: Dict[str, int], cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a single-user sparse profile (indices + weights) from processed CSVs.
    Used for EASE inference and for filtering 'seen' items.
    """
    orders_path = _pick_file(data_dir, "Заказы", True)
    views_path = _pick_file(data_dir, "Просмотры", True)
    fav_path = _pick_file(data_dir, "Избранное", True)

    # fallback to originals if selection not found
    if not os.path.isfile(orders_path):
        orders_path = _pick_file(data_dir, "Заказы", False)
    if not os.path.isfile(views_path):
        views_path = _pick_file(data_dir, "Просмотры", False)
    if not os.path.isfile(fav_path):
        fav_path = _pick_file(data_dir, "Избранное", False)

    weights = {}

    if os.path.isfile(orders_path):
        o = _read_csv_pipe(orders_path)
        if {"MindboxID", "КодНоменклатуры"}.issubset(o.columns):
            oo = o[o["MindboxID"].astype(str) == str(mindbox_id)]
            if len(oo):
                qty = pd.to_numeric(oo.get("Количество", 1), errors="coerce").fillna(1).astype(float).clip(1, 10)
                for code, q in zip(oo["КодНоменклатуры"].astype(str).tolist(), qty.tolist()):
                    if code in item2idx:
                        weights[item2idx[code]] = weights.get(item2idx[code], 0.0) + cfg.w_purchase * float(q)

    if os.path.isfile(fav_path):
        f = _read_csv_pipe(fav_path)
        if {"MindboxID", "КодНоменклатуры"}.issubset(f.columns):
            ff = f[f["MindboxID"].astype(str) == str(mindbox_id)]
            for code in ff.get("КодНоменклатуры", pd.Series(dtype=str)).dropna().astype(str).tolist():
                if code in item2idx:
                    weights[item2idx[code]] = weights.get(item2idx[code], 0.0) + cfg.w_favorite

    if os.path.isfile(views_path):
        v = _read_csv_pipe(views_path)
        if {"MindboxID", "КодНоменклатуры", "ТипТовара"}.issubset(v.columns):
            vv = v[(v["MindboxID"].astype(str) == str(mindbox_id)) & (v["ТипТовара"] == "Номенклатура")]
            for code in vv.get("КодНоменклатуры", pd.Series(dtype=str)).dropna().astype(str).tolist():
                if code in item2idx:
                    weights[item2idx[code]] = weights.get(item2idx[code], 0.0) + cfg.w_view_item

    if not weights:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    idx = np.fromiter(weights.keys(), dtype=np.int64)
    val = np.fromiter(weights.values(), dtype=np.float32)
    return idx, val


def print_recommendations(mindbox_id: str, k: int = 20, use: str = "ease") -> None:
    """
    Prints top-K recommendations to console using saved artifacts.
    use: "ease" | "bprmf"
    """
    cfg = TrainConfig()  # for profile weights + paths
    maps_json, ckpt, ease_B = _load_artifacts()

    idx2user = maps_json["idx2user"]
    idx2item = maps_json["idx2item"]
    item2idx = {code: i for i, code in enumerate(idx2item)}

    names = _load_item_names(cfg.data_dir, idx2item)

    # Build user seen profile for filtering (and for EASE scoring)
    seen_idx, seen_w = _user_profile_from_processed(cfg.data_dir, str(mindbox_id), idx2item, item2idx, cfg)

    if use.lower() == "ease":
        if ease_B is None:
            print(f"[{_now()}] EASE model not found (Models/ease_B.npy). Use use='bprmf' or retrain.")
            return
        if len(seen_idx) == 0:
            print(f"[{_now()}] User {mindbox_id}: no history in processed files. Cannot score with EASE.")
            return

        # score = w @ B[seen, :]
        B_sub = ease_B[seen_idx]                       # [m, n_items]
        scores = (seen_w.astype(np.float32) @ B_sub)   # [n_items]
        scores = np.asarray(scores).ravel()

        # filter seen
        scores[seen_idx] = -1e9

        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]

        print(f"[{_now()}] Recommendations (EASE) for MindboxID={mindbox_id} top{k}:")
        for rank, ii in enumerate(top, start=1):
            code = idx2item[int(ii)]
            nm = names.get(code, "")
            if nm:
                print(f"{rank:02d}. {code} | {nm} | score={scores[int(ii)]:.4f}")
            else:
                print(f"{rank:02d}. {code} | score={scores[int(ii)]:.4f}")

    else:
        # BPR-MF
        num_users = int(ckpt["num_users"])
        num_items = int(ckpt["num_items"])
        model = BPRMF(num_users, num_items, int(ckpt["config"]["embedding_dim"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Map mindbox -> idx
        user2idx = {u: i for i, u in enumerate(idx2user)}
        if str(mindbox_id) not in user2idx:
            print(f"[{_now()}] User {mindbox_id} not found in mappings.json.")
            return
        u_idx = user2idx[str(mindbox_id)]

        u = model.user_emb.weight[u_idx]                 # [d]
        scores = (u @ model.item_emb.weight.t()).numpy() # [n_items]
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


# ----------------------------- Training entry point (UI button) -----------------------------

def _train_in_this_process() -> None:
    cfg = TrainConfig()
    _set_seed(cfg.seed)

    data_dir = cfg.data_dir
    orders_path = _pick_file(data_dir, "Заказы", cfg.use_selection_if_exists)
    views_path = _pick_file(data_dir, "Просмотры", cfg.use_selection_if_exists)
    fav_path = _pick_file(data_dir, "Избранное", cfg.use_selection_if_exists)

    required = [orders_path, views_path, fav_path]
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        print(f"[{_now()}] ERROR: missing required processed files in '{data_dir}':")
        for p in missing:
            print(f"  - {p}")
        print("Сначала загрузите/обработайте датасеты во вкладке 'Обработка датасета'.")
        return

    orders = _read_csv_pipe(orders_path)
    views = _read_csv_pipe(views_path)
    fav = _read_csv_pipe(fav_path)

    print(f"[{_now()}] Loaded:")
    print(f"  - Orders:     {len(orders):,} rows  ({os.path.basename(orders_path)})")
    print(f"  - Views:      {len(views):,} rows  ({os.path.basename(views_path)})")
    print(f"  - Favorites:  {len(fav):,} rows    ({os.path.basename(fav_path)})")

    maps = _build_mappings(orders, views, fav)
    num_users = len(maps.idx2user)
    num_items = len(maps.idx2item)
    print(f"[{_now()}] Universe sizes: users={num_users:,} items={num_items:,}")

    events = _collect_user_item_events(orders, views, fav, maps, cfg)
    if len(events) == 0:
        print(f"[{_now()}] ERROR: no user-item events found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{_now()}] Device: {device}")

    model, splits = train_bprmf(maps, events, cfg, device)

    print(f"[{_now()}] Saving artifacts to ./Models/ ...")
    ease_B = None
    try:
        B, r, n = train_ease_r(splits, num_users, num_items, cfg)
        ease_B = B
        print(f"[{_now()}] EASE^R: recall@{cfg.topk}={r:.4f} ndcg@{cfg.topk}={n:.4f}")
    except Exception as e:
        print(f"[{_now()}] EASE^R failed (non-fatal): {repr(e)}")

    _save_artifacts(cfg, maps, model, splits, ease_B)
    print(f"[{_now()}] Done. Exiting training process.")


def train_recommender(*_args, **_kwargs) -> None:
    """
    Called by your PyQt button.

    Runs training in a subprocess to avoid Windows native shutdown crashes.
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


# ----------------------------- CLI -----------------------------

def _parse_cli(argv: List[str]) -> Tuple[bool, Optional[str], int, str]:
    do_train = "--train" in argv
    mindbox = None
    k = 20
    model = "ease"

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
    if "--model" in argv:
        i = argv.index("--model")
        if i + 1 < len(argv):
            model = argv[i + 1]

    return do_train, mindbox, k, model


if __name__ == "__main__":
    do_train, mindbox, k, model = _parse_cli(sys.argv[1:])
    if do_train:
        _train_in_this_process()
        # hard-exit helps avoid rare native crashes during Python shutdown on Windows
        os._exit(0)

    if mindbox is not None:
        print_recommendations(mindbox, k=k, use=model)
