"""
gemma_litert.py — [KNEO Pi] Gemma 4 E2B 透過 LiteRT-LM 跑在 CPU 上
==================================================================
為什麼用 LiteRT-LM：同一個模型，llama.cpp 在 KNEO Pi 上一題要約 9 分鐘（模型比記憶體大，
每生成一個 token 都要從 microSD 重讀）。LiteRT-LM 版可以常駐記憶體，實測一題約 22 秒，
其中約 19.5 秒是影像編碼，所以影像解析度用模型提供的最低檔（70 tokens）。

模型載入一次就常駐，佔約 1.4GB。出錯時回傳一句說明，不丟例外，讓使用者聽得到發生什麼事。
"""

import os
import tempfile

from app import config, postprocess
from app.base import VLMBackend
from app.messages import t


class GemmaLitertBackend(VLMBackend):
    name = "gemma-4-e2b-litert-lm"

    def __init__(self):
        self.engine = None

    def load(self) -> None:
        if not os.path.exists(config.GEMMA_LITERTLM):
            print(f"[warn] 找不到 {config.GEMMA_LITERTLM}，這次啟動無法使用 Gemma")
            return
        import litert_lm
        os.makedirs(config.GEMMA_CACHE_DIR, exist_ok=True)
        cpu = litert_lm.Backend.CPU(thread_count=config.CPU_THREADS)
        self.engine = litert_lm.Engine(
            config.GEMMA_LITERTLM, backend=cpu, vision_backend=cpu, cache_dir=config.GEMMA_CACHE_DIR,
            max_vision_tokens_per_image=config.GEMMA_VISION_TOKENS, max_num_images=1)
        self.engine.__enter__()

    def warmup(self) -> None:
        """第一題比之後慢 5~25 秒，開機時先跑掉。"""
        from PIL import Image
        if self.engine is not None:
            self._generate(Image.new("RGB", (384, 384), (127, 127, 127)), "這是什麼？",
                           config.GEMMA_SYSTEM_ZH, None, 4)

    def _generate(self, image, prompt, system_prompt, history, max_new_tokens):
        import litert_lm
        if self.engine is None:
            raise RuntimeError("Gemma 沒有載入")
        if not hasattr(image, "convert"):            # OpenCV 的 BGR 陣列 → PIL
            from PIL import Image
            image = Image.fromarray(image[:, :, ::-1])
        messages = []
        for q, a in history or []:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            image.convert("RGB").save(path, quality=90)
            conversation = self.engine.create_conversation(
                messages=messages or None, system_message=system_prompt,
                visual_token_budget=config.GEMMA_VISION_TOKENS, max_output_tokens=max_new_tokens)
            with conversation:
                response = conversation.send_message(litert_lm.Contents.of(
                    prompt, litert_lm.Content.ImageFile(absolute_path=path)))
        finally:
            os.unlink(path)
        return "".join(part.get("text", "") for part in response["content"]).strip()

    def describe(self, image, prompt: str, max_new_tokens: int | None = None,
                 system_prompt: str | None = None, history=None) -> str:
        try:
            text = self._generate(image, prompt, system_prompt or config.GEMMA_SYSTEM_ZH, history,
                                  max_new_tokens or config.ASK_MAX_TOKENS)
        except Exception as e:
            print(f"[warn] Gemma 失敗：{e}")
            return t("vlm_unavailable")
        # 這個版本不做過馬路：回答裡提到燈號或能否過馬路的句子一律刪掉，小模型這方面不可靠
        return postprocess.strip_light_claims(text) or t("vlm_unavailable")

    def read_text(self, image) -> str:
        """讀字：照原樣唸，不套用燈號過濾（告示上可能真的寫著「過馬路」）。"""
        try:
            return self._generate(image, "圖片裡寫了什麼？", config.GEMMA_SYSTEM_READ, None,
                                  config.GEMMA_READ_MAX_TOKENS) or t("no_text")
        except Exception as e:
            print(f"[warn] Gemma 讀字失敗：{e}")
            return t("vlm_unavailable")
