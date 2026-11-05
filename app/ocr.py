"""
ocr.py — 「讀字」模式的 OCR 引擎
================================
[資服版] 原本用 RapidOCR（中國團隊開發，權重來自百度 PaddleOCR），依競賽規定換成
EasyOCR（JaidedAI，Apache-2.0，https://github.com/JaidedAI/EasyOCR）。
文字偵測用 CRAFT，辨識用 EasyOCR 的繁體中文 (ch_tra) + 英文模型。

為什麼讀字要用 OCR 而不是 VLM：論文 4.3 節量過，SmolVLM-500M 在 17 張圖裡讀對 0 張。

第一次執行會自動下載模型（需要連網一次），之後完全離線。
有 CUDA 顯示卡時自動使用，沒有就用 CPU。
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from app import config
from app.messages import t

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _ENGINE = easyocr.Reader(list(config.OCR_LANGS), gpu=gpu, verbose=False)
    return _ENGINE


# ---------------------------------------------------------------------------
# [KNEO Pi] EasyOCR 放在子程序（app/ocr_worker.py），要跑 Gemma 前可以關掉把記憶體還回來
# ---------------------------------------------------------------------------
_WORKER = None
_WORKER_READY = False


def preload():
    """在背景開始載入 EasyOCR，不等它完成。"""
    global _WORKER, _WORKER_READY
    if config.OCR_WORKER and (_WORKER is None or _WORKER.poll() is not None):
        _WORKER = subprocess.Popen([sys.executable, "-m", "app.ocr_worker"], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True, encoding="utf-8")
        _WORKER_READY = False


def release():
    """關掉 EasyOCR 子程序，把記憶體還給系統。"""
    global _WORKER, _WORKER_READY
    if _WORKER is not None:
        _WORKER.kill()
        _WORKER.stdin.close()
        _WORKER.stdout.close()
        _WORKER.wait()
    _WORKER, _WORKER_READY = None, False


def _worker_line(prefix):
    for line in _WORKER.stdout:
        if line.startswith(prefix):
            return line
    release()
    raise RuntimeError("EasyOCR 子程序意外結束")


def _worker_read(image):
    global _WORKER_READY
    preload()
    if not _WORKER_READY:
        _worker_line("READY")
        _WORKER_READY = True
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        image.convert("RGB").save(path)
        _WORKER.stdin.write(path + "\n")
        _WORKER.stdin.flush()
        return json.loads(_worker_line("RESULT ")[len("RESULT "):])
    finally:
        os.unlink(path)


def read_detail(image):
    """PIL.Image → (文字, 夠有把握的字數, 偵測到的總字數)。
    文字 = 信心達標的部分（由上到下、由左到右）；完全沒有達標的字時為空字串。"""
    if config.OCR_WORKER:
        result = _worker_read(image)
    else:
        arr = np.array(image.convert("RGB"))
        result = _engine().readtext(arr, **config.OCR_READ_KWARGS)   # [(四個角點, 文字, 信心)]
    items, total = [], 0
    for box, txt, score in result:
        if not txt or not txt.strip():
            continue
        total += len(txt.strip())
        if score < config.OCR_MIN_CONF:
            continue
        top = min(pt[1] for pt in box)
        left = min(pt[0] for pt in box)
        items.append((top, left, txt.strip()))
    items.sort(key=lambda x: (x[0], x[1]))
    kept = sum(len(txt) for _, _, txt in items)
    return " ".join(txt for _, _, txt in items), kept, total


def read_text(image) -> str:
    """PIL.Image → 讀到的文字，沒有文字時回傳「沒有看到文字。」"""
    text, _kept, _total = read_detail(image)
    return text or t("no_text")
