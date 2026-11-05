"""
ocr_worker.py — [KNEO Pi] 在獨立程序裡跑 EasyOCR
================================================
EasyOCR 載入後佔約 1GB，和 Gemma 同時留在記憶體會嚴重搶資源。放在子程序裡，
要跑 Gemma 前關掉這個程序，記憶體才真的還給系統（見 app/ocr.py）。

協定：從 stdin 每行讀一個圖片路徑，每張輸出一行 "RESULT <json>"；載入完成時輸出 "READY"。
"""

import json
import sys

from app import config


def main():
    import easyocr
    reader = easyocr.Reader(list(config.OCR_LANGS), gpu=False, verbose=False, download_enabled=False)
    print("READY", flush=True)
    for line in sys.stdin:
        items = reader.readtext(line.strip(), **config.OCR_READ_KWARGS)
        result = [[[[float(v) for v in pt] for pt in box], text, float(score)] for box, text, score in items]
        print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
