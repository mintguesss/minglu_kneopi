"""
scene.py — [KNEO Pi] 把偵測結果組成固定句型的繁中描述（約 1 秒內回答）
======================================================================
Gemma 在 KNEO Pi 上一次可能要等好幾分鐘，所以周遭、室內、物品模式先用偵測結果回答。
只說偵測到的東西，不做推論；偵測不到時回傳 None，交給 VLM。
"""

from app import detect

MIN_CONF = 0.40
MAX_ITEMS = 4
NEAR_AREA = 0.15          # 框面積 > 畫面 15% 就說「很近」
HELD_AREA = 0.12          # 物品模式：和 pipeline.route 的手持物品門檻相同

COCO_ZH = {
    "person": "人", "bicycle": "腳踏車", "car": "汽車", "motorcycle": "機車", "airplane": "飛機",
    "bus": "公車", "train": "火車", "truck": "卡車", "boat": "船", "traffic light": "號誌",
    "fire hydrant": "消防栓", "stop sign": "停車標誌", "parking meter": "停車收費器",
    "bench": "長椅", "bird": "鳥", "cat": "貓", "dog": "狗", "horse": "馬", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑馬", "giraffe": "長頸鹿",
    "backpack": "背包", "umbrella": "雨傘", "handbag": "手提包", "tie": "領帶",
    "suitcase": "行李箱", "frisbee": "飛盤", "skis": "滑雪板", "snowboard": "單板滑雪板",
    "sports ball": "球", "kite": "風箏", "baseball bat": "球棒", "baseball glove": "棒球手套",
    "skateboard": "滑板", "surfboard": "衝浪板", "tennis racket": "網球拍", "bottle": "瓶子",
    "wine glass": "酒杯", "cup": "杯子", "fork": "叉子", "knife": "刀子", "spoon": "湯匙",
    "bowl": "碗", "banana": "香蕉", "apple": "蘋果", "sandwich": "三明治", "orange": "橘子",
    "broccoli": "花椰菜", "carrot": "紅蘿蔔", "hot dog": "熱狗", "pizza": "披薩",
    "donut": "甜甜圈", "cake": "蛋糕", "chair": "椅子", "couch": "沙發", "potted plant": "盆栽",
    "bed": "床", "dining table": "桌子", "toilet": "馬桶", "tv": "電視", "laptop": "筆電",
    "mouse": "滑鼠", "remote": "遙控器", "keyboard": "鍵盤", "cell phone": "手機",
    "microwave": "微波爐", "oven": "烤箱", "toaster": "烤麵包機", "sink": "水槽",
    "refrigerator": "冰箱", "book": "書", "clock": "時鐘", "vase": "花瓶", "scissors": "剪刀",
    "teddy bear": "玩偶", "hair drier": "吹風機", "toothbrush": "牙刷",
}

_DIGITS = {1: "一", 2: "兩", 3: "三", 4: "四", 5: "五"}


def _area(box, w, h):
    return ((box[2] - box[0]) * (box[3] - box[1])) / (w * h)


def _position(box, w):
    cx = ((box[0] + box[2]) / 2) / w
    if cx < 1 / 3:
        return "左前方"
    if cx > 2 / 3:
        return "右前方"
    return "正前方"


def _count(n):
    return _DIGITS.get(n, "多") + "個"


def _sentence(dets, w, h):
    groups = {}
    for d in dets:
        key = (_position(d["box"], w), d["cls"])
        g = groups.setdefault(key, {"n": 0, "area": 0.0})
        g["n"] += 1
        g["area"] = max(g["area"], _area(d["box"], w, h))
    # 人優先，其次由近到遠（框越大越近）
    ranked = sorted(groups.items(), key=lambda kv: (kv[0][1] != "person", -kv[1]["area"]))
    parts = []
    for (pos, cls), g in ranked[:MAX_ITEMS]:
        near = "，很近" if g["area"] > NEAR_AREA else ""
        parts.append(f"{pos}有{_count(g['n'])}{COCO_ZH.get(cls, cls)}{near}")
    return "；".join(parts) + "。"


def describe(image, mode):
    """回傳描述文字；這張畫面沒有可說的內容時回傳 None。"""
    dets, (w, h) = detect.detect(image, conf=MIN_CONF, classes="all")
    dets = [d for d in dets if d["cls"] != "traffic light"]     # 這個版本不談號誌
    if not dets:
        return None
    if mode == "object":
        held = [d for d in dets if d["cls"] != "person"
                and _area(d["box"], w, h) > HELD_AREA
                and 0.2 < ((d["box"][0] + d["box"][2]) / 2) / w < 0.8]
        if not held:
            return None
        best = max(held, key=lambda d: d["conf"])
        return f"這看起來是{COCO_ZH.get(best['cls'], best['cls'])}。"
    return _sentence(dets, w, h)
