"""在 KNEO Pi 上實測各功能的耗時，並印出尖峰記憶體。

用法：MINGLU_PROFILE=kneopi python tools/kneopi/bench.py <whisper|ocr|gemma|scene> [檔案...]
"""

import os
import resource
import sys
import time
import wave

os.environ.setdefault("MINGLU_PROFILE", "kneopi")
sys.path.insert(0, os.getcwd())

import numpy as np
from PIL import Image


def report(label, t0, text):
    # ru_maxrss 單位是 KB；CHILDREN 是 llama-mtmd-cli、whisper-cli 這些子程序
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    kid = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    print(f"[{label}] {time.perf_counter() - t0:.1f} 秒 | 本程序尖峰 {me} MB，子程序尖峰 {kid} MB\n  → {text}", flush=True)


kind, files = sys.argv[1], sys.argv[2:]

if kind == "whisper":
    from app import voice
    voice.load_asr()
    for path in files:
        with wave.open(path) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        text = voice.transcribe(pcm)
        report(f"whisper {path} {len(pcm) / 16000:.1f}s 音訊", t0, f"{text} | 指令判斷：{voice.parse_command(text)}")

elif kind == "ocr":
    t0 = time.perf_counter()
    from app import ocr
    report("載入 easyocr 模組", t0, "")
    for path in files + files:       # 第二輪才看得出模型載入後的速度
        t0 = time.perf_counter()
        report(f"ocr {path}", t0, ocr.read_detail(Image.open(path)))

elif kind == "scene":
    from app import scene
    for path in files:
        t0 = time.perf_counter()
        report(f"scene {path}", t0, scene.describe(Image.open(path).convert("RGB"), "surrounding"))

elif kind == "gemma":
    from app import config, voice
    from app.assistant import build_backend
    backend = build_backend()
    backend.load()
    for path in files:
        image = Image.open(path).convert("RGB")
        t0 = time.perf_counter()
        report(f"gemma 描述 {path}", t0, backend.describe(image, config.get_prompt("surrounding"),
                                                       max_new_tokens=config.get_max_tokens("surrounding")))
        t0 = time.perf_counter()
        report(f"gemma 問答 {path}", t0, voice.answer(backend, image, "這是什麼交通工具", "auto")[0])
