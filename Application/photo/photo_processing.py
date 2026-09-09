import os
import pandas as pd
import re
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtNetwork import QNetworkRequest, QNetworkReply
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QLabel, QTableWidget)


_PHOTO_URL_NOT_PROVIDED = object()


def _clean_code(value) -> str:
    s = str(value or "").strip()
    s = re.sub(r"\.0$", "", s)

    if s.lower() in ("", "nan", "none", "<na>", "null"):
        return ""

    return s


def _ensure_photo_map(aboba):
    nom_path = os.path.join(os.getcwd(), "ВходныеДанные", "Номенклатура.csv")

    if not os.path.isfile(nom_path):
        aboba._photo_by_code = {}
        return

    mtime = os.path.getmtime(nom_path)

    if (
        aboba._photo_by_code is not None
        and getattr(aboba, "_photo_map_mtime", None) == mtime
    ):
        return

    aboba._photo_by_code = {}
    aboba._photo_map_mtime = mtime

    df = pd.read_csv(nom_path, sep="|", encoding="utf-8-sig", dtype=str)

    if "КодНоменклатуры" not in df.columns or "ТитульнаяФотография" not in df.columns:
        return

    df["КодНоменклатуры"] = df["КодНоменклатуры"].map(_clean_code)
    df["ТитульнаяФотография"] = (
        df["ТитульнаяФотография"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    df = df[
        (df["КодНоменклатуры"] != "")
        & (df["ТитульнаяФотография"] != "")
    ].copy()

    aboba._photo_by_code = (
        df.drop_duplicates(subset=["КодНоменклатуры"], keep="first")
        .set_index("КодНоменклатуры")["ТитульнаяФотография"]
        .to_dict()
    )
    

def _set_photo_cell(
        aboba,
        table: QTableWidget,
        r: int,
        code: str,
        gen: int,
        photo_url=_PHOTO_URL_NOT_PROVIDED,
) -> None:
    PAD = 6
    PHOTO_W = 100
    ASPECT = 1600 / 1066
    PHOTO_H = int(PHOTO_W * ASPECT)

    lbl = QLabel()
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setContentsMargins(0, 0, 0, 0)
    lbl.setFixedSize(PHOTO_W + PAD * 2, PHOTO_H + PAD * 2)
    lbl.setStyleSheet(f"QLabel {{ padding: {PAD}px; background: transparent; }}")

    ph = QPixmap(PHOTO_W, PHOTO_H)
    ph.fill(Qt.GlobalColor.transparent)
    lbl.setPixmap(ph)

    table.setCellWidget(r, 0, lbl)
    table.setRowHeight(r, PHOTO_H + PAD * 2)
    table.setColumnWidth(0, PHOTO_W + PAD * 2)

    if photo_url is _PHOTO_URL_NOT_PROVIDED:
        url = _photo_url_for_code(aboba, code)
    else:
        url = photo_url

    if not url:
        lbl.clear()
        lbl.setText("Нет фото")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return

    cached = aboba._img_cache.get(url, "MISS")
    if cached != "MISS":
        if cached is not None:
            _set_row_pixmap(table, r, cached)
        return

    aboba._img_targets.setdefault(url, []).append((table, r))

    if url not in aboba._img_inflight and url not in aboba._img_queue:
        aboba._img_queue.append(url)

    _pump_image_queue(aboba, gen)
    

def _photo_url_for_code(aboba, code: str) -> str:
    _ensure_photo_map(aboba)

    key = _clean_code(code)
    if not key:
        return ""

    raw = (aboba._photo_by_code or {}).get(key, "")
    return _photo_to_url(raw)


def _photo_to_url(raw: str) -> str:
    if raw is None:
        return ""

    s = str(raw).strip().strip('"').strip("'")

    if not s or s.lower() in ("nan", "none", "<na>", "null"):
        return ""

    # если в строке несколько значений через ; или перенос строки — берём первое
    s = re.split(r"[;\n\r]+", s)[0].strip()

    # если внутри строки есть полноценная ссылка — вытащим её
    m = re.search(r"https?://[^\s\"';]+", s)
    if m:
        return m.group(0)

    if s.startswith("//"):
        return "https:" + s

    if s.startswith("/"):
        return "https://kanzler-style.ru" + s

    if s.startswith("upload/"):
        return "https://kanzler-style.ru/" + s

    return s


def _set_row_pixmap(table: QTableWidget, row: int, pm: QPixmap) -> None:
    w = table.cellWidget(row, 0)
    if not isinstance(w, QLabel):
        return

    w.setAlignment(Qt.AlignmentFlag.AlignCenter)
    w.setScaledContents(False)

    rect = w.contentsRect()  # учитывает padding
    if rect.width() <= 0 or rect.height() <= 0:
        return

    scaled = pm.scaled(
        rect.width(), rect.height(),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    w.setPixmap(scaled)
    

def _request_image(aboba, url: str, gen: int) -> None:
    if not url:
        return

    req = QNetworkRequest(QUrl(url))
    req.setAttribute(QNetworkRequest.Attribute.Http2AllowedAttribute, False)
    req.setAttribute(
        QNetworkRequest.Attribute.RedirectPolicyAttribute,
        QNetworkRequest.RedirectPolicy.NoLessSafeRedirectPolicy
    )

    req.setRawHeader(b"User-Agent", b"Mozilla/5.0")
    req.setRawHeader(b"Accept", b"image/jpeg,image/png,image/webp,image/*,*/*;q=0.8")
    req.setRawHeader(b"Referer", b"https://kanzler-style.ru/")

    reply = aboba.net.get(req)
    reply.setProperty("img_url", str(url).strip())
    reply.setProperty("img_gen", gen)
    reply.finished.connect(lambda r=reply: _on_image_loaded(aboba, r))


def _retry_image_url(aboba, url: str, gen: int) -> None:
    if gen != aboba._img_gen:
        return
    if url in aboba._img_inflight:
        return
    aboba._img_queue.appendleft(url)
    _pump_image_queue(aboba, gen)


def _on_image_loaded(aboba, reply) -> None:
    url = str(reply.property("img_url") or "").strip()
    gen = int(reply.property("img_gen") or 0)

    status_raw = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
    status = int(status_raw) if status_raw is not None else 0

    ctype_raw = reply.header(QNetworkRequest.KnownHeaders.ContentTypeHeader)
    ctype = str(ctype_raw or "").lower()

    err = reply.error()

    data = bytes(reply.readAll())
    reply.deleteLater()

    # Всегда снимаем inflight
    aboba._img_inflight.discard(url)

    # Старое поколение: подчистили и вышли
    if gen != aboba._img_gen:
        aboba._img_targets.pop(url, None)
        aboba._img_retry_count.pop(url, None)
        _pump_image_queue(aboba, aboba._img_gen)
        return

    # Проверка успешности
    pm = QPixmap()
    decode_ok = pm.loadFromData(data)
    http_ok = (status == 0) or (200 <= status < 300)
    ctype_ok = (not ctype) or ctype.startswith("image/")

    ok = (
            err == QNetworkReply.NetworkError.NoError
            and http_ok
            and ctype_ok
            and decode_ok
            and not pm.isNull()
    )

    # Ретрай для временных проблем
    retriable_status = {408, 425, 429, 500, 502, 503, 504}
    retriable_errors = {
        QNetworkReply.NetworkError.TemporaryNetworkFailureError,
        QNetworkReply.NetworkError.TimeoutError,
        QNetworkReply.NetworkError.RemoteHostClosedError,
        QNetworkReply.NetworkError.UnknownNetworkError,
    }

    attempt = aboba._img_retry_count.get(url, 0)
    should_retry = (
            not ok
            and attempt < aboba._img_retry_max
            and (status in retriable_status or err in retriable_errors or len(data) == 0)
    )

    if should_retry:
        aboba._img_retry_count[url] = attempt + 1
        delay_ms = 250 * (2 ** attempt)  # 250, 500, 1000...
        QTimer.singleShot(delay_ms, lambda u=url, g=gen: _retry_image_url(aboba, u, g))
        _pump_image_queue(aboba, gen)
        return

    targets = aboba._img_targets.pop(url, [])
    aboba._img_retry_count.pop(url, None)

    if not ok:
        aboba._img_cache.pop(url, None)

        reason = "Ошибка"
        if status:
            reason = f"HTTP {status}"
        elif not decode_ok:
            reason = "Формат"
        elif err != QNetworkReply.NetworkError.NoError:
            reason = "Сеть"

        for table, r in targets:
            w = table.cellWidget(int(r), 0)
            if isinstance(w, QLabel):
                w.clear()
                w.setText(reason)
                w.setAlignment(Qt.AlignmentFlag.AlignCenter)

        _pump_image_queue(aboba, gen)
        return

    aboba._img_cache[url] = pm
    for table, r in targets:
        _set_row_pixmap(table, int(r), pm)

    _pump_image_queue(aboba, gen)


def _pump_image_queue(aboba, gen: int) -> None:
    while len(aboba._img_inflight) < aboba._img_max_inflight and aboba._img_queue:
        url = aboba._img_queue.popleft()

        if url in aboba._img_inflight:
            continue

        aboba._img_inflight.add(url)
        _request_image(aboba, url, gen)
