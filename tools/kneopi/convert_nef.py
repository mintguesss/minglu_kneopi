"""第 2 步：把切好的 ONNX 量化、編譯成 KL730 的 NEF，再用 NEF 模擬器確認偵測結果。

在 kneron/toolchain 容器裡執行，專案根目錄掛在 /data1：
    source /workspace/miniconda/bin/activate onnx1.13 && python /data1/tools/kneopi/convert_nef.py
校正圖片放在 tools/kneopi/work/calib/，輸出 models/yolov8n_730.nef。
"""

import glob
import json
import os
import shutil
import sys

import kneronnxopt
import ktc
import numpy as np
import onnx
from PIL import Image

ROOT = "/data1"
WORK = os.path.join(ROOT, "tools/kneopi/work")
sys.path.insert(0, ROOT)
os.environ.setdefault("MINGLU_PROFILE", "kneopi")

from app import detect_kneron  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"
model = onnx.load(os.path.join(WORK, NAME + "_cut.onnx"))
input_name = model.graph.input[0].name
optimized = kneronnxopt.optimize(model)
input_name = optimized.graph.input[0].name

config = ktc.ModelConfig(11111, "0001", "730", onnx_model=optimized)
print("NPU 效能估計：", config.evaluate())

calibration = []
for path in sorted(glob.glob(os.path.join(WORK, "calib", "*.jpg"))):
    data, _, _ = detect_kneron.letterbox(Image.open(path))
    calibration.append(data)
print("校正圖片：", len(calibration))
bie = config.analysis({input_name: calibration})
nef = ktc.compile([config])

target = os.path.join(ROOT, "models", NAME + "_730.nef")
shutil.copyfile(nef, target)
print("輸出：", target, os.path.getsize(target), "bytes")

image = Image.open(os.path.join(WORK, "bus.jpg"))
data, ratio, pad = detect_kneron.letterbox(image)
outputs = ktc.kneron_inference([data], input_names=[input_name], nef_file=target, platform=730)
print("輸出形狀：", [np.asarray(o).shape for o in outputs])
names_path = os.path.join(WORK, NAME + "_names.json")
names = json.load(open(names_path)) if os.path.exists(names_path) else detect_kneron.COCO_NAMES
dets = detect_kneron.postprocess(outputs, 0.25, ratio, pad, image.size, names=names)
with open(os.path.join(WORK, NAME + "_nef_bus.json"), "w") as f:
    json.dump(dets, f, indent=1)
print(json.dumps(dets, indent=1))
