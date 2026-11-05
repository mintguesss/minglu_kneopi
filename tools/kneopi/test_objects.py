"""在真 NPU 上比較 80 類與 601 類 YOLOv8n：python tools/kneopi/test_objects.py <模型名稱> <類別檔或 coco> <圖片...>"""

import json
import os
import sys
import time

os.environ.setdefault("MINGLU_PROFILE", "kneopi")
sys.path.insert(0, os.getcwd())

from PIL import Image

from app import config, detect_kneron

name, names_path, files = sys.argv[1], sys.argv[2], sys.argv[3:]
config.YOLO_NEF = os.path.join(config.MODEL_DIR, name + "_730.nef")
names = detect_kneron.COCO_NAMES if names_path == "coco" else json.load(open(names_path))
kp_postprocess = detect_kneron.postprocess
detect_kneron.postprocess = lambda *a, **k: kp_postprocess(*a, names=names, **k)

for path in files:
    image = Image.open(path).convert("RGB")
    t0 = time.perf_counter()
    dets = detect_kneron.infer(image, 0.10)
    ms = (time.perf_counter() - t0) * 1000
    dets.sort(key=lambda d: -d["conf"])
    w, h = image.size
    top = [f"{d['cls']} {d['conf']:.2f} ({(d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]) / (w * h):.0%})"
           for d in dets[:5]]
    print(f"{os.path.basename(path)} {ms:.0f}ms: {'; '.join(top) or '（沒有偵測到）'}", flush=True)
