"""第 1 步：把 YOLOv8n 匯出成 ONNX，並切在偵測頭的最後一層卷積。

在一般的 Python 容器裡執行（需要 ultralytics、onnx），工作目錄是 tools/kneopi/work：
    pip install ultralytics onnx onnxslim
    python export_onnx.py [yolov8n-oiv7]
輸出：<名稱>_cut.onnx（6 個輸出）、<名稱>_bus.json（之後用來比對 NPU 結果）、<名稱>_names.json（類別名稱）
"""

import json
import sys

import onnx
from ultralytics import YOLO

name = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"
model = YOLO(name + ".pt")
path = model.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)

graph = onnx.load(path).graph
outputs = []
for level in range(3):
    outputs.append(f"/model.22/cv2.{level}/cv2.{level}.2/Conv_output_0")
    outputs.append(f"/model.22/cv3.{level}/cv3.{level}.2/Conv_output_0")
names = {o for node in graph.node for o in node.output}
missing = [o for o in outputs if o not in names]
if missing:
    raise SystemExit(f"找不到這些節點：{missing}")
onnx.utils.extract_model(path, f"{name}_cut.onnx", [graph.input[0].name], outputs)
print("輸入節點：", graph.input[0].name, "類別數：", len(model.names))

with open(f"{name}_names.json", "w", encoding="utf-8") as f:
    json.dump([model.names[i] for i in range(len(model.names))], f, indent=0)

res = model("bus.jpg", imgsz=640, conf=0.25, verbose=False)[0]
expected = [{"cls": res.names[int(b.cls)], "conf": float(b.conf), "box": [float(v) for v in b.xyxy[0]]}
            for b in res.boxes]
with open(f"{name}_bus.json", "w") as f:
    json.dump(expected, f, indent=1)
print(json.dumps(expected, indent=1))
