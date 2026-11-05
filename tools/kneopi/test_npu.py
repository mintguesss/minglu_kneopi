"""在 KNEO Pi 上用真的 NPU 跑 YOLOv8n：python tools/kneopi/test_npu.py <圖片>"""

import json
import os
import sys
import time

os.environ.setdefault("MINGLU_PROFILE", "kneopi")
sys.path.insert(0, os.getcwd())

from PIL import Image

from app import detect_kneron

image = Image.open(sys.argv[1])
t0 = time.perf_counter()
detect_kneron._connect()
print(f"連線與載入模型：{time.perf_counter() - t0:.1f} 秒")

for i in range(5):
    t0 = time.perf_counter()
    dets = detect_kneron.infer(image, 0.25)
    print(f"第 {i + 1} 次：{(time.perf_counter() - t0) * 1000:.0f} ms，{len(dets)} 個物件")
print(json.dumps([{"cls": d["cls"], "conf": round(d["conf"], 3), "box": [round(v) for v in d["box"]]}
                  for d in dets], ensure_ascii=False, indent=1))
