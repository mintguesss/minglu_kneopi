"""
detect.py — 物件偵測（YOLOv8n），供過馬路模式的混合式流程使用
================================================================
為什麼需要：小型 VLM 分不出「車是停著還是在開」（§5.5.2／5.5.1），還會在空的路上幻想出車。
專門的偵測模型能準確框出「車／人」的位置 → 再用位置做規則判斷
（app/crossing.py），結果可預期。這就是本研究主打的「混合式流程」
（VLM＝情境／燈色，偵測模型＝路徑上的車）。

離線：YOLOv8n 權重（約 6MB）只需下載一次並快取 — 推論時不需要網路。
容錯：沒有安裝 ultralytics → 回傳 []（程式照常執行，不會當掉）。
"""

from app import config

# 過馬路模式關心的類別（名稱依 YOLO 的 COCO 類別）
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
PERSON_CLASS = "person"
_RELEVANT = VEHICLE_CLASSES | {PERSON_CLASS}

_MODEL = None


def _detect_kneron(image, conf, classes):
    from app import detect_kneron
    try:
        dets = detect_kneron.infer(image, conf)
    except Exception as e:
        print(f"[warn] NPU 偵測失敗：{e}")
        return []
    keep = _RELEVANT if classes is None else classes
    return [d for d in dets if keep == "all" or d["cls"] in keep]


def _model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO("yolov8n.pt")
    return _MODEL


def detect(image, conf: float = 0.35, classes=None):
    """
    在 PIL.Image 上執行偵測。回傳 (dets, (w, h))。
    dets = list of {"cls": str, "conf": float, "box": (x1,y1,x2,y2)}.
    classes：要保留的類別名稱集合（預設 = 車＋人），"all" = 所有 COCO 類別（router 使用）。
    沒有 ultralytics → ([], size)（容錯）。
    """
    w, h = image.size
    if config.DETECTOR_ENGINE == "kneron":
        return _detect_kneron(image, conf, classes), (w, h)
    try:
        m = _model()
    except Exception:
        return [], (w, h)

    keep = _RELEVANT if classes is None else classes
    import numpy as np
    res = m(np.asarray(image.convert("RGB")), conf=conf, verbose=False)[0]
    dets = []
    for b in res.boxes:
        cls = m.names[int(b.cls)]
        if keep != "all" and cls not in keep:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        dets.append({"cls": cls, "conf": float(b.conf), "box": (x1, y1, x2, y2)})
    return dets, (w, h)
