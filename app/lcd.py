"""板子上的 I2C 字元 LCD（HD44780 + PCF8574 轉接板，常見位址 0x27）。

第一行顯示板子在熱點上的 IP（手機要連的網址），第二行顯示目前狀態。
只用標準函式庫（/dev/i2c-N + ioctl），不需要另外裝套件。
沒有接 LCD、沒有權限或寫入失敗時一律安靜略過，不影響明路本身。
"""

from __future__ import annotations

import os
import socket
import threading
import time

from . import config

_I2C_SLAVE = 0x0703

# PCF8574 的腳位對應（市面上大多數轉接板都是這樣接）
_RS, _EN, _BL = 0x01, 0x04, 0x08


class CharLCD:
    def __init__(self, bus: int = 0, addr: int = 0x27, cols: int = 16, rows: int = 2):
        import fcntl                # 只有 Linux 有；放這裡讓電腦版也能 import 這個模組
        self.cols, self.rows = cols, rows
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        fcntl.ioctl(self.fd, _I2C_SLAVE, addr)
        self._init()

    def _raw(self, b: int) -> None:
        os.write(self.fd, bytes([b | _BL]))

    def _pulse(self, b: int) -> None:
        self._raw(b | _EN)
        time.sleep(0.0005)
        self._raw(b & ~_EN)
        time.sleep(0.0001)

    def _nibble(self, n: int, rs: int) -> None:
        b = (n & 0xF0) | rs
        self._raw(b)
        self._pulse(b)

    def _byte(self, v: int, rs: int = 0) -> None:
        self._nibble(v & 0xF0, rs)
        self._nibble((v << 4) & 0xF0, rs)

    def _init(self) -> None:
        time.sleep(0.05)
        for _ in range(3):           # 不管原本在 8 或 4 位元模式，先強制回 8 位元
            self._nibble(0x30, 0)
            time.sleep(0.005)
        self._nibble(0x20, 0)        # 切到 4 位元
        self._byte(0x28)             # 2 行、5x8 字型
        self._byte(0x0C)             # 顯示開、游標關
        self._byte(0x06)             # 寫入後游標右移
        self.clear()

    def clear(self) -> None:
        self._byte(0x01)
        time.sleep(0.002)

    def line(self, row: int, text: str) -> None:
        offsets = (0x00, 0x40, 0x14, 0x54)
        self._byte(0x80 | offsets[row])
        text = text.encode("ascii", "replace").decode()[: self.cols].ljust(self.cols)
        for ch in text:
            self._byte(ord(ch), _RS)

    def close(self) -> None:
        os.close(self.fd)


def hotspot_ip() -> str | None:
    """板子對外（熱點）那張網卡的 IPv4；沒有網路時回傳 None。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # UDP connect 不會真的送封包，只是讓系統選出要用的來源位址
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


class StatusDisplay:
    """背景執行緒：IP 變了就更新第一行；set_status() 更新第二行。"""

    def __init__(self) -> None:
        self._lcd: CharLCD | None = None
        self._status = "Starting"
        self._shown: tuple[str, str] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="lcd").start()

    def set_status(self, text: str) -> None:
        self._status = text
        self._refresh()

    # 每隔這麼久就重新初始化、整個重寫一次。I2C 上只要掉一個位元，LCD 的 4 位元傳輸就會錯位，
    # 之後寫的字全部變成亂碼（實測 CPU 滿載時會出現 ooOO 和日文符號），所以要定期自我修復
    HEAL_SEC = 10

    def _refresh(self, force: bool = False) -> None:
        ip = hotspot_ip()
        lines = (ip or "No network", self._status)
        with self._lock:
            if lines == self._shown and not force:
                return
            try:
                if self._lcd is None:
                    self._lcd = CharLCD(config.LCD_I2C_BUS, config.LCD_I2C_ADDR)
                else:
                    self._lcd._init()          # 每次都重新同步，前一次寫壞也不會延續下去
                self._lcd.line(0, lines[0])
                self._lcd.line(1, lines[1])
                self._shown = lines
            except OSError:
                # 沒接、被拔掉或權限不足：下次再試
                self._lcd, self._shown = None, None

    def _loop(self) -> None:
        last_heal = time.monotonic()
        while True:
            force = time.monotonic() - last_heal >= self.HEAL_SEC
            if force:
                last_heal = time.monotonic()
            self._refresh(force)
            time.sleep(3)


_display: StatusDisplay | None = None


def start() -> None:
    global _display
    if getattr(config, "LCD_ENABLED", False) and _display is None:
        _display = StatusDisplay()
        _display.start()


def status(text: str) -> None:
    if _display is not None:
        _display.set_status(text)


_result: str | None = None      # 這次處理的結果（例如偵測到的物品），做完時取代「Ready (Ns)」


def objects_summary(labels: list[str]) -> str:
    """["person", "car", "person"] → "2 person, 1 car"（多的在前；LCD 只能顯示英文）"""
    counts: dict[str, int] = {}
    for name in labels:
        counts[name] = counts.get(name, 0) + 1
    order = sorted(counts, key=lambda n: (-counts[n], labels.index(n)))
    return ", ".join(f"{counts[n]} {n}" for n in order)


def show_objects(labels: list[str]) -> None:
    """偵測結果出來時呼叫：處理完後第二行顯示偵測到的東西。"""
    global _result
    if labels:
        _result = objects_summary(labels)


# Gemma 的回答是中文，LCD 只能顯示英文：用對照表找出回答裡提到的東西。
# COCO 80 類的中文名稱（app/scene.py）也會用上；這裡是常見但不在 COCO 裡的東西，以及想換短一點的英文
_EXTRA_EN = {
    "電腦": "computer", "筆電": "laptop", "螢幕": "monitor", "顯示器": "monitor", "手機": "phone",
    "鍵盤": "keyboard", "滑鼠": "mouse", "耳機": "headphones", "充電器": "charger", "電線": "cable",
    "插座": "outlet", "遙控器": "remote", "杯": "cup", "水壺": "bottle", "書": "book", "紙": "paper",
    "筆": "pen", "盒子": "box", "箱子": "box", "袋子": "bag", "包包": "bag", "錢包": "wallet",
    "鑰匙": "keys", "眼鏡": "glasses", "帽子": "hat", "衣服": "clothes", "鞋": "shoes",
    "衛生紙": "tissue", "食物": "food", "飲料": "drink", "時鐘": "clock", "燈": "light",
    "植物": "plant", "盆栽": "plant", "桌子": "table", "桌": "table", "椅": "chair", "門": "door", "窗": "window",
    "牆": "wall", "地板": "floor", "天花板": "ceiling", "樓梯": "stairs", "電梯": "elevator",
    "文字": "text", "招牌": "sign", "天空": "sky", "雲": "cloud", "樹": "tree", "車": "car",
    "人": "person", "電視": "TV",
}
_NOTHING_RE = ("沒有看到", "看不清楚", "看不到")


def _zh_to_en() -> list[tuple[str, str]]:
    from app import scene                      # 放這裡避免和 scene 互相 import
    table = {zh: en for en, zh in scene.COCO_ZH.items()}
    table.update(_EXTRA_EN)
    return sorted(table.items(), key=lambda kv: -len(kv[0]))   # 長的先比對（「筆電」優先於「電」）


def answer_keywords(text: str) -> str | None:
    """「這是一台電腦和一些周邊設備。」→ "Computer"；找不到對照時回傳 None。"""
    if any(w in text for w in _NOTHING_RE):
        return "Nothing seen"
    found, rest = [], text
    for zh, en in _zh_to_en():
        pos = rest.find(zh)
        if pos >= 0:
            found.append((pos, en))
            rest = rest.replace(zh, "\0" * len(zh))   # 已比對過的字不要再被短的詞比對到
    words = []
    for _, en in sorted(found):
        if en not in words:
            words.append(en)
    if not words:
        return None
    return ", ".join(w[0].upper() + w[1:] for w in words)


def show_answer(text: str) -> None:
    """Gemma 回答後呼叫：這次還沒有偵測結果時，第二行顯示回答裡提到的東西。"""
    global _result
    if _result is None and text:
        _result = answer_keywords(text)


class busy:
    """with lcd.busy("Thinking"): ...  做事時顯示 text；做完顯示結果，沒有結果就顯示花了幾秒。"""

    def __init__(self, text: str):
        self.text = text

    def __enter__(self):
        global _result
        _result = None
        self.t0 = time.perf_counter()
        status(self.text)
        return self

    def __exit__(self, exc_type, *_):
        dt = time.perf_counter() - self.t0
        status("Error" if exc_type else (_result or f"Ready ({dt:.0f}s)"))
        return False
