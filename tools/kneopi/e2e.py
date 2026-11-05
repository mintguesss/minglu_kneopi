"""對跑著的伺服器做端到端測試，模擬手機送來的請求。

用法：python tools/kneopi/e2e.py <測試資料夾>（裡面要有 bus.jpg、sign.jpg、say1.wav、say2.wav、say3.wav）
"""

import base64
import json
import os
import ssl
import sys
import time
import urllib.request
import wave

BASE = "https://127.0.0.1:8443"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
folder = sys.argv[1]


def image(name):
    with open(os.path.join(folder, name), "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def audio(name):
    with wave.open(os.path.join(folder, name)) as w:
        return base64.b64encode(w.readframes(w.getnframes())).decode()


def post(label, path, body):
    t0 = time.perf_counter()
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900, context=CTX) as r:
            out = json.loads(r.read().decode())
    except Exception as e:
        out = {"error": repr(e)}
    print(f"[{label}] {time.perf_counter() - t0:.1f} 秒\n  {json.dumps(out, ensure_ascii=False)[:300]}", flush=True)


with urllib.request.urlopen(BASE + "/ping", context=CTX) as r:
    print("ping", r.read().decode(), flush=True)
with urllib.request.urlopen(BASE + "/", context=CTX) as r:
    page = r.read().decode()
print("頁面有過馬路按鈕：", 'data-mode="street"' in page, "；有讀字按鈕：", 'data-mode="read"' in page, flush=True)

bus, sign = image("bus.jpg"), image("sign.jpg")
post("過馬路（舊手機設定）", "/describe", {"image": bus, "mode": "street"})
post("自動", "/describe", {"image": bus, "mode": "auto"})
post("周遭", "/describe", {"image": bus, "mode": "surrounding"})
post("語音：切換到讀字", "/ask", {"audio": audio("say2.wav"), "image": bus, "mode": "auto", "history": []})
post("讀字", "/describe", {"image": sign, "mode": "read"})
post("語音：前面有什麼（會問 Gemma）", "/ask", {"audio": audio("say1.wav"), "image": bus, "mode": "auto", "history": []})
print("E2E_DONE", flush=True)
