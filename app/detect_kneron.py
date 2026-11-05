"""
detect_kneron.py — [KNEO Pi] 用 KL730 NPU 跑專案原本的 YOLOv8n
==============================================================
模型由 tools/kneopi/ 從 yolov8n.pt 轉成 models/yolov8n_730.nef（COCO 80 類）。
轉換時把模型切在偵測頭的最後一層卷積，DFL 解碼與 NMS 留在這裡用 numpy 做，
因為 NPU 主要加速卷積，而且 KNEO Pi 上不必為了後處理載入 torch。

流程參考 Kneron 官方範例 kneron/kneopi-examples 的 KL730End2EndTutorialYoloV7.py。
"""

import os
import sys

import numpy as np
from PIL import Image

from app import config

INPUT_SIZE = 640
IOU_THRES = 0.45
REG_MAX = 16
NUM_CLASSES = 80

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

_DEVICE = None      # (device_group, model_descriptor)


# ---------------------------------------------------------------------------
# 前處理與後處理（純 numpy，不需要 NPU，可以在電腦上測試）
# ---------------------------------------------------------------------------
def letterbox(image, size=INPUT_SIZE):
    """PIL.Image → (1×3×size×size float32 0~1 RGB, 縮放比例, (左邊留白, 上方留白))"""
    w, h = image.size
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.convert("RGB").resize((nw, nh), Image.BILINEAR), (left, top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None], ratio, (left, top)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _decode_level(box_logits, cls_logits, input_size=INPUT_SIZE):
    """一個尺度的 (1,64,H,W) 與 (1,80,H,W) → (N×4 的 xyxy, N×80 的類別分數)，座標是輸入圖的像素。"""
    _, _, h, w = box_logits.shape
    stride = input_size / h
    dist = box_logits[0].reshape(4, REG_MAX, h, w)
    dist = (_softmax(dist, axis=1) * np.arange(REG_MAX, dtype=np.float32).reshape(1, -1, 1, 1)).sum(axis=1)
    cy, cx = np.meshgrid(np.arange(h, dtype=np.float32) + 0.5,
                         np.arange(w, dtype=np.float32) + 0.5, indexing="ij")
    boxes = np.stack((cx - dist[0], cy - dist[1], cx + dist[2], cy + dist[3]), axis=-1) * stride
    return boxes.reshape(-1, 4), _sigmoid(cls_logits[0]).reshape(NUM_CLASSES, -1).T


def _nms(boxes, scores, iou_thres):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
        order = rest[iou <= iou_thres]
    return keep


def postprocess(outputs, conf, ratio, pad, image_size, iou_thres=IOU_THRES):
    """六個輸出張量（順序不拘）→ [{"cls", "conf", "box"}]（box 是原圖像素座標）。"""
    levels = {}
    for out in outputs:
        out = np.asarray(out, dtype=np.float32)
        kind = {REG_MAX * 4: "box", NUM_CLASSES: "cls"}[out.shape[1]]
        levels.setdefault(out.shape[2], {})[kind] = out
    decoded = [_decode_level(lv["box"], lv["cls"]) for lv in levels.values()]
    boxes = np.concatenate([d[0] for d in decoded])
    scores = np.concatenate([d[1] for d in decoded])

    cls_id = scores.argmax(axis=1)
    score = scores[np.arange(len(cls_id)), cls_id]
    mask = score >= conf
    if not mask.any():
        return []
    boxes, cls_id, score = boxes[mask], cls_id[mask], score[mask]

    # 各類別分開做 NMS：位移到不重疊的座標區間
    keep = _nms(boxes + (cls_id[:, None] * 4096.0), score, iou_thres)

    w, h = image_size
    left, top = pad
    dets = []
    for i in keep:
        x1 = float(np.clip((boxes[i, 0] - left) / ratio, 0, w))
        y1 = float(np.clip((boxes[i, 1] - top) / ratio, 0, h))
        x2 = float(np.clip((boxes[i, 2] - left) / ratio, 0, w))
        y2 = float(np.clip((boxes[i, 3] - top) / ratio, 0, h))
        dets.append({"cls": COCO_NAMES[int(cls_id[i])], "conf": float(score[i]),
                     "box": (x1, y1, x2, y2)})
    return dets


# ---------------------------------------------------------------------------
# NPU（只有這段需要 KNEO Pi）
# ---------------------------------------------------------------------------
def _connect():
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    root = config.KNEOPI_EXAMPLES_DIR
    if root not in sys.path:
        sys.path.insert(1, root)
    import kp
    device_group = kp.core.connect_devices(usb_port_ids=[0])
    kp.core.load_firmware_from_file(
        device_group=device_group,
        scpu_fw_path=os.path.join(root, "res/firmware/KL730/kp_firmware.tar"),
        ncpu_fw_path="")
    kp.core.set_timeout(device_group=device_group, milliseconds=5000)
    model = kp.core.load_model_from_file(device_group=device_group, file_path=config.YOLO_NEF)
    _DEVICE = (device_group, model)
    return _DEVICE


def infer(image, conf):
    """PIL.Image → 偵測結果。NPU 或 kp 有問題時丟出例外，由 detect.py 處理。"""
    device_group, model = _connect()      # 順便把官方範例加進 sys.path，下面才 import 得到
    import kp
    from utils.ExampleHelper import convert_onnx_data_to_npu_data
    data, ratio, pad = letterbox(image)
    buffer = convert_onnx_data_to_npu_data(
        tensor_descriptor=model.models[0].input_nodes[0], onnx_data=data)
    kp.inference.generic_data_inference_send(
        device_group=device_group,
        generic_inference_input_descriptor=kp.GenericDataInferenceDescriptor(
            model_id=model.models[0].id, inference_number=0,
            input_node_data_list=[kp.GenericInputNodeData(buffer=buffer)]))
    raw = kp.inference.generic_data_inference_receive(device_group=device_group)
    outputs = [
        kp.inference.generic_inference_retrieve_float_node(
            node_idx=i, generic_raw_result=raw,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT).ndarray
        for i in range(raw.header.num_output_node)
    ]
    return postprocess(outputs, conf, ratio, pad, image.size)
