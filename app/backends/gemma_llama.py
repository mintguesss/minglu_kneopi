"""
gemma_llama.py — [KNEO Pi] Gemma 4 E2B（低量化）透過 llama.cpp 跑在 CPU 上
==========================================================================
Gemma 家族能看圖的最小版本，會中文，所以提示詞和電腦版的 Gemma 共用。
模型加影像模組約 2.9GB，超過 KNEO Pi 的 2GB：模型檔用 mmap 從 microSD 讀，需要時才進記憶體，
所以放得下，但會很慢。每次呼叫都開一個 llama-mtmd-cli，結束後記憶體就還給系統。

出錯或逾時回傳一句說明，不丟例外，讓使用者聽得到發生什麼事。
"""

import os
import re
import subprocess
import tempfile

from app import config, postprocess
from app.base import VLMBackend
from app.messages import t

_THINK_RE = re.compile(r"<\|?(?:think|channel)[^>]*>.*?(?:</think>|<\|?channel\|?>)", re.S)


class GemmaLlamaBackend(VLMBackend):
    name = "gemma-4-e2b-llama.cpp"

    def __init__(self):
        self.model = config.GEMMA_GGUF
        self.mmproj = config.GEMMA_MMPROJ
        self.binary = config.LLAMA_MTMD_BIN

    def load(self) -> None:
        for path in (self.model, self.mmproj, self.binary):
            if not os.path.exists(path):
                print(f"[warn] 找不到 {path}，這次啟動無法使用 Gemma")

    @staticmethod
    def _build_prompt(prompt, system_prompt, history):
        lines = [system_prompt or config.GEMMA_SYSTEM_ZH]
        for q, a in history or []:
            lines += [f"使用者：{q}", f"你：{a}"]
        lines.append(prompt)
        return "\n".join(lines)

    def _generate(self, image, prompt, max_new_tokens):
        image = image.convert("RGB")
        image.thumbnail((config.GEMMA_IMAGE_EDGE, config.GEMMA_IMAGE_EDGE))
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            image.save(path, quality=90)
            r = subprocess.run(
                [self.binary, "-m", self.model, "--mmproj", self.mmproj, "--image", path,
                 "-p", prompt, "-n", str(max_new_tokens), "-c", str(config.GEMMA_CONTEXT),
                 "-t", str(config.CPU_THREADS), "--temp", "0",
                 "--repeat-penalty", str(config.GEMMA_GEN_PARAMS["repetition_penalty"])],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=config.VLM_TIMEOUT_SEC)
        finally:
            os.unlink(path)
        if r.returncode != 0:
            raise RuntimeError(f"llama-mtmd-cli 結束碼 {r.returncode}：{r.stderr[-300:]}")
        return _THINK_RE.sub("", r.stdout).strip()

    def describe(self, image, prompt: str, max_new_tokens: int | None = None,
                 system_prompt: str | None = None, history=None) -> str:
        if not hasattr(image, "convert"):            # OpenCV 的 BGR 陣列 → PIL
            from PIL import Image
            image = Image.fromarray(image[:, :, ::-1])
        try:
            text = self._generate(image, self._build_prompt(prompt, system_prompt, history),
                                  max_new_tokens or config.GEMMA_GEN_PARAMS["max_new_tokens"])
        except subprocess.TimeoutExpired:
            print(f"[warn] Gemma 超過 {config.VLM_TIMEOUT_SEC} 秒")
            return t("vlm_timeout")
        except Exception as e:
            print(f"[warn] Gemma 失敗：{e}")
            return t("vlm_unavailable")
        # 這個版本不做過馬路：回答裡提到燈號或能否過馬路的句子一律刪掉，小模型這方面不可靠
        return postprocess.strip_light_claims(text) or t("vlm_unavailable")
