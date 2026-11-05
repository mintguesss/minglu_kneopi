"""從 Gemma 的 GGUF 取出聊天範本，並關掉思考模式，存成 llama-mtmd-cli 要用的檔案。

llama-mtmd-cli 沒有關閉思考的參數，不關的話 token 全花在英文的思考過程上，等不到答案。
用法：python tools/kneopi/extract_template.py <模型.gguf> <輸出.jinja>
"""

import struct
import sys

KEY = b"tokenizer.chat_template"
THINKING_ON = "{%- set enable_thinking = enable_thinking | default(false) -%}"
THINKING_OFF = "{%- set enable_thinking = false -%}"


def extract(path):
    with open(path, "rb") as f:
        head = f.read(64 * 1024 * 1024)     # 中繼資料都在檔案開頭
    i = head.find(KEY)
    if i < 0:
        raise SystemExit("模型檔裡找不到聊天範本")
    i += len(KEY)
    length = struct.unpack("<Q", head[i + 4:i + 12])[0]
    return head[i + 12:i + 12 + length].decode("utf-8")


def disable_thinking(template):
    if THINKING_ON not in template:
        raise SystemExit("範本格式和預期不同，無法關閉思考模式")
    return template.replace(THINKING_ON, THINKING_OFF)


if __name__ == "__main__":
    with open(sys.argv[2], "w", encoding="utf-8") as out:
        out.write(disable_thinking(extract(sys.argv[1])))
