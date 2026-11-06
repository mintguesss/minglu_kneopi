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

    def _refresh(self) -> None:
        ip = hotspot_ip()
        lines = (ip or "No network", self._status)
        with self._lock:
            if lines == self._shown:
                return
            try:
                if self._lcd is None:
                    self._lcd = CharLCD(config.LCD_I2C_BUS, config.LCD_I2C_ADDR)
                self._lcd.line(0, lines[0])
                self._lcd.line(1, lines[1])
                self._shown = lines
            except OSError:
                # 沒接、被拔掉或權限不足：下次再試
                self._lcd, self._shown = None, None

    def _loop(self) -> None:
        while True:
            self._refresh()
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
