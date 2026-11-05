"""第 1 步：把 yolov8n.pt 匯出成 ONNX，並切在偵測頭的最後一層卷積。

在一般的 Python 容器裡執行（需要 ultralytics、onnx），工作目錄是 tools/kneopi/work：
    pip install ultralytics onnx onnxslim
    python export_onnx.py
輸出：yolov8n_cut.onnx（6 個輸出）、ultralytics_bus.json（之後用來比對 NPU 結果）
"""

import json

import onnx
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
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
onnx.utils.extract_model(path, "yolov8n_cut.onnx", [graph.input[0].name], outputs)
print("輸入節點：", graph.input[0].name)

res = model("bus.jpg", imgsz=640, conf=0.25, verbose=False)[0]
expected = [{"cls": res.names[int(b.cls)], "conf": float(b.conf), "box": [float(v) for v in b.xyxy[0]]}
            for b in res.boxes]
with open("ultralytics_bus.json", "w") as f:
    json.dump(expected, f, indent=1)
print(json.dumps(expected, indent=1))
