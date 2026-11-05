"""KNEO Pi 版的單元測試：不需要 NPU 或模型檔。執行：python -m unittest tests.test_kneopi -v"""
import os
import re
import subprocess
import unittest
from unittest import mock

os.environ["MINGLU_PROFILE"] = "kneopi"

import numpy as np
from PIL import Image

from app import config, detect, detect_kneron, ocr, pipeline, postprocess, scene, server, voice
from app.backends.gemma_llama import GemmaLlamaBackend
from app.messages import t

IMAGE = Image.new("RGB", (100, 100))


def _outputs(hit=None):
    """六個輸出張量；hit = (格數, row, col, class, (l, t, r, b))，距離以 stride 為單位。"""
    out = []
    for n in (80, 40, 20):
        box = np.zeros((1, 64, n, n), dtype=np.float32)
        cls = np.full((1, 80, n, n), -20.0, dtype=np.float32)
        if hit and hit[0] == n:
            _, r, c, k, dist = hit
            cls[0, k, r, c] = 10.0
            for side, d in enumerate(dist):
                box[0, side * 16 + d, r, c] = 30.0
        out += [box, cls]
    return out


class NpuLayoutTest(unittest.TestCase):
    def test_same_as_element_by_element_rearrange(self):
        shape, s_onnx, s_npu = (1, 3, 4, 5), (60, 20, 5, 1), (480, 16, 80, 4)
        data = np.linspace(-1, 1, 60, dtype=np.float32).reshape(shape)
        offsets = detect_kneron.npu_offsets(shape, s_onnx, s_npu)
        got = np.frombuffer(detect_kneron.quantize_int8(data, 64.0, offsets), dtype=np.int8)
        self.assertEqual(len(got), 480)
        want = np.zeros(480, dtype=np.int8)
        q = np.clip(np.round(data * 64.0), -128, 127).astype(np.int8).ravel()
        for idx in np.ndindex(shape):
            want[np.dot(idx, s_npu)] = q[np.dot(idx, s_onnx)]
        np.testing.assert_array_equal(got, want)


def _det(cls, box, conf=0.9):
    return {"cls": cls, "conf": conf, "box": box}


class DecoderTest(unittest.TestCase):
    def test_box_position_and_class(self):
        dets = detect_kneron.postprocess(_outputs((40, 10, 20, 2, (1, 2, 3, 4))), 0.25, 1.0, (0, 0), (640, 640))
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["cls"], "car")
        # 中心 (20.5, 10.5) 格；x1 = (20.5 - 1) * 16 ...
        np.testing.assert_allclose(dets[0]["box"], (312, 136, 376, 232), atol=1e-2)

    def test_output_order_does_not_matter(self):
        a = _outputs((20, 5, 6, 0, (1, 1, 1, 1)))
        first = detect_kneron.postprocess(a, 0.25, 1.0, (0, 0), (640, 640))
        second = detect_kneron.postprocess(a[::-1], 0.25, 1.0, (0, 0), (640, 640))
        self.assertEqual(first[0]["cls"], "person")
        np.testing.assert_allclose(first[0]["box"], second[0]["box"])

    def test_nothing_above_threshold(self):
        self.assertEqual(detect_kneron.postprocess(_outputs(), 0.25, 1.0, (0, 0), (640, 640)), [])

    def test_nms(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
        self.assertEqual(detect_kneron._nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45), [0, 2])

    def test_letterbox_maps_back_to_original(self):
        image = Image.new("RGB", (1280, 720))
        data, ratio, pad = detect_kneron.letterbox(image)
        self.assertEqual(data.shape, (1, 3, 640, 640))
        self.assertEqual((ratio, pad), (0.5, (0, 140)))
        dets = detect_kneron.postprocess(_outputs((40, 19, 19, 2, (1, 1, 1, 1))), 0.25, ratio, pad, image.size)
        x1, y1, x2, y2 = dets[0]["box"]
        self.assertAlmostEqual((x1 + x2) / 2, 19.5 * 16 / 0.5, places=2)
        self.assertAlmostEqual((y1 + y2) / 2, (19.5 * 16 - 140) / 0.5, places=2)

    def test_npu_failure_means_no_detections(self):
        with mock.patch.object(detect_kneron, "infer", side_effect=RuntimeError("no npu")):
            self.assertEqual(detect.detect(IMAGE)[0], [])

    def test_filters_requested_classes(self):
        with mock.patch.object(detect_kneron, "infer", return_value=[_det("car", (0, 0, 1, 1)), _det("cup", (0, 0, 1, 1))]):
            self.assertEqual([d["cls"] for d in detect.detect(IMAGE, classes={"cup"})[0]], ["cup"])


class NoStreetTest(unittest.TestCase):
    def test_profile_has_no_street(self):
        self.assertFalse(config.STREET_MODE)

    def test_stale_street_mode_is_refused(self):
        backend = mock.Mock()
        with mock.patch.object(detect, "detect", return_value=([], (100, 100))) as d:
            text, _ = pipeline.describe(backend, IMAGE, "street")
        self.assertEqual(text, t("street_unsupported"))
        backend.describe.assert_not_called()
        d.assert_not_called()

    def test_auto_never_routes_to_street(self):
        dets = [_det("car", (40, 40, 60, 60)), _det("traffic light", (0, 0, 5, 5))]
        with mock.patch.object(detect, "detect", return_value=(dets, (100, 100))):
            self.assertNotEqual(pipeline.route(IMAGE)[0], "street")

    def test_crossing_questions_are_refused_before_gemma(self):
        for q in ("現在可以過馬路嗎", "現在是紅燈嗎", "切換到過馬路模式", "號誌是什麼顏色", "過馬路"):
            self.assertEqual(voice.parse_command(q), ("street_unsupported", None), q)
        self.assertEqual(voice.command_reply("street_unsupported", None), t("street_unsupported"))

    def test_car_question_is_a_normal_question(self):
        self.assertIsNone(voice.parse_command("前面那是什麼車子"))
        self.assertEqual(voice.question_mode("前面那是什麼車子"), "vqa")

    def test_other_modes_still_switch(self):
        self.assertEqual(voice.parse_command("切換到讀字模式"), ("switch", "read"))

    def test_help_does_not_offer_street(self):
        self.assertNotIn("過馬路", voice.HELP_TEXT)
        self.assertNotIn("紅燈", voice.HELP_TEXT)
        self.assertNotIn("過馬路", voice.command_reply("switch_unknown", ""))

    def test_strip_light_claims(self):
        self.assertEqual(postprocess.strip_light_claims("前面有一輛車。號誌是綠燈。可以過馬路！"), "前面有一輛車。")
        self.assertEqual(postprocess.strip_light_claims("現在是綠燈。"), "")

    def test_page_has_no_street_button(self):
        with mock.patch.object(server, "PAGE", server.PAGE):
            server._apply_profile_to_page()
            page = server.PAGE
        self.assertNotIn('data-mode="street"', page)
        self.assertNotIn("行人號誌", page)
        self.assertIn('data-mode="read"', page)
        self.assertIn('data-mode="indoor"', page)
        self.assertIn("if(MODE==='street') MODE='auto'", page)
        self.assertIn(str(config.PHONE_ASK_TIMEOUT_MS), page)
        self.assertNotIn("電腦", page)


class SceneTest(unittest.TestCase):
    def _scene(self, dets, mode="surrounding"):
        with mock.patch.object(detect, "detect", return_value=(dets, (100, 100))):
            return scene.describe(IMAGE, mode)

    def test_positions_and_near(self):
        text = self._scene([_det("chair", (0, 0, 20, 20)), _det("person", (30, 0, 70, 90))])
        self.assertEqual(text, "正前方有一個人，很近；左前方有一個椅子。")

    def test_vehicle_measure_word(self):
        self.assertIn("一輛公車", self._scene([_det("bus", (40, 0, 60, 20))]))

    def test_counts(self):
        self.assertIn("兩個人", self._scene([_det("person", (40, 0, 50, 10)), _det("person", (50, 0, 60, 10))]))

    def test_nothing_falls_through_to_gemma(self):
        self.assertIsNone(self._scene([]))

    def test_traffic_light_is_not_mentioned(self):
        self.assertIsNone(self._scene([_det("traffic light", (40, 0, 50, 10))]))

    def test_object_needs_large_central_item(self):
        self.assertEqual(self._scene([_det("bottle", (30, 30, 70, 70))], "object"), "這看起來是瓶子。")
        self.assertIsNone(self._scene([_det("bottle", (0, 0, 10, 10))], "object"))

    def test_every_coco_class_has_chinese(self):
        self.assertEqual(set(detect_kneron.COCO_NAMES) - set(scene.COCO_ZH), set())

    def test_pipeline_prefers_detector(self):
        backend = mock.Mock()
        with mock.patch.object(detect, "detect", return_value=([_det("chair", (40, 0, 50, 10))], (100, 100))):
            text, _ = pipeline.describe(backend, IMAGE, "indoor")
        backend.describe.assert_not_called()
        self.assertIn("椅子", text)

    def test_pipeline_asks_gemma_when_nothing_detected(self):
        backend = mock.Mock()
        backend.describe.return_value = "有一面牆。"
        with mock.patch.object(detect, "detect", return_value=([], (100, 100))):
            text, _ = pipeline.describe(backend, IMAGE, "indoor")
        self.assertEqual(text, "有一面牆。")
        self.assertRegex(backend.describe.call_args[0][1], "[一-鿿]")   # 中文提示詞


class GemmaBackendTest(unittest.TestCase):
    def _run(self, stdout="一張椅子。", returncode=0, side_effect=None, **kwargs):
        result = mock.Mock(stdout=stdout, stderr="", returncode=returncode)
        with mock.patch("subprocess.run", return_value=result, side_effect=side_effect) as run, \
             mock.patch.object(GemmaLlamaBackend, "_template", return_value="TEMPLATE"):
            text = GemmaLlamaBackend().describe(Image.new("RGB", (800, 600)), "這是什麼？", 40, **kwargs)
        return text, run

    def test_passes_model_and_prompt(self):
        text, run = self._run()
        self.assertEqual(text, "一張椅子。")
        args = run.call_args[0][0]
        self.assertEqual(args[args.index("-m") + 1], config.GEMMA_GGUF)
        self.assertEqual(args[args.index("--mmproj") + 1], config.GEMMA_MMPROJ)
        self.assertEqual(args[args.index("-n") + 1], "40")
        self.assertIn("這是什麼？", args[args.index("-p") + 1])

    def test_history_and_system_prompt_are_sent(self):
        _, run = self._run(system_prompt="系統提示", history=[("前面有什麼", "一張桌子")])
        prompt = run.call_args[0][0][run.call_args[0][0].index("-p") + 1]
        self.assertEqual(prompt, "系統提示\n使用者：前面有什麼\n你：一張桌子\n請看這張照片回答：這是什麼？")

    def test_light_claim_removed(self):
        self.assertEqual(self._run(stdout="現在是綠燈。")[0], t("vlm_unavailable"))

    def test_think_block_removed(self):
        self.assertEqual(self._run(stdout="<think>想一下</think>一張椅子。")[0], "一張椅子。")

    def test_failure_and_timeout_speak(self):
        self.assertEqual(self._run(returncode=1)[0], t("vlm_unavailable"))
        self.assertEqual(self._run(side_effect=subprocess.TimeoutExpired("x", 1))[0], t("vlm_timeout"))

    def test_voice_uses_system_prompt_and_history(self):
        backend = mock.Mock()
        backend.describe.return_value = "一張椅子。"
        text, target = voice.answer(backend, IMAGE, "這是什麼", "auto", [{"q": "前面", "a": "桌子", "t": "vqa"}])
        self.assertEqual((text, target), ("一張椅子。", "vqa"))
        kwargs = backend.describe.call_args[1]
        self.assertEqual(kwargs["system_prompt"], config.GEMMA_SYSTEM_ASK)
        self.assertEqual(kwargs["history"], [("前面", "桌子")])


class AudioCtxTest(unittest.TestCase):
    def test_follows_recording_length(self):
        self.assertEqual(voice._audio_ctx(16000), 256)
        self.assertEqual(voice._audio_ctx(160000), 564)
        self.assertEqual(voice._audio_ctx(16000 * 60), 1500)


class WhisperTest(unittest.TestCase):
    def _run(self, stdout):
        result = mock.Mock(stdout=stdout, stderr="", returncode=0)
        with mock.patch("subprocess.run", return_value=result) as run:
            return voice.transcribe(np.zeros(16000, dtype=np.float32)), run

    def test_passes_language_and_returns_text(self):
        text, run = self._run(" 前面有什麼\n")
        self.assertEqual(text, "前面有什麼")
        args = run.call_args[0][0]
        self.assertEqual(args[args.index("-l") + 1], "zh")
        self.assertEqual(args[args.index("-m") + 1], config.ASR_MODEL)

    def test_blank_audio_is_empty(self):
        self.assertEqual(self._run("[BLANK_AUDIO]\n")[0], "")

    def test_hallucination_is_empty(self):
        self.assertEqual(self._run("謝謝觀看\n")[0], "")


class OcrWorkerTest(unittest.TestCase):
    """用真的子程序跑一個假的 worker，驗證 EasyOCR 子程序的協定與記憶體釋放。"""

    FAKE = ("import sys,json\nprint('torch warning noise',flush=True)\nprint('READY',flush=True)\n"
            "for line in sys.stdin:\n"
            "    print('RESULT '+json.dumps([[[[0,0],[9,0],[9,9],[0,9]],'出口',0.9],[[[0,20],[9,20],[9,29],[0,29]],'糊',0.1]]),flush=True)\n")

    def setUp(self):
        ocr.release()
        import sys
        real = subprocess.Popen
        self.addCleanup(ocr.release)
        self.patcher = mock.patch("subprocess.Popen", side_effect=lambda cmd, **kw: real([sys.executable, "-c", self.FAKE], **kw))
        self.popen = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_reads_filters_and_keeps_worker(self):
        self.assertEqual(ocr.read_detail(IMAGE), ("出口", 2, 3))
        self.assertEqual(ocr.read_detail(IMAGE), ("出口", 2, 3))
        self.assertEqual(self.popen.call_count, 1)

    def test_release_then_read_restarts(self):
        ocr.read_detail(IMAGE)
        worker = ocr._WORKER
        ocr.release()
        self.assertIsNotNone(worker.poll())
        self.assertIsNone(ocr._WORKER)
        ocr.read_detail(IMAGE)
        self.assertEqual(self.popen.call_count, 2)

    def test_gemma_releases_ocr_first_and_preloads_after(self):
        calls = []
        result = mock.Mock(stdout="一張椅子。", stderr="", returncode=0)
        def run(*a, **k):
            calls.append("gemma")
            return result
        with mock.patch("subprocess.run", side_effect=run), \
             mock.patch.object(GemmaLlamaBackend, "_template", return_value="T"), \
             mock.patch.object(ocr, "release", side_effect=lambda: calls.append("release")), \
             mock.patch.object(ocr, "preload", side_effect=lambda: calls.append("preload")):
            GemmaLlamaBackend().describe(IMAGE, "這是什麼", 8)
        self.assertEqual(calls, ["release", "gemma", "preload"])


class ProfileTest(unittest.TestCase):
    def test_uses_original_model_families(self):
        self.assertIn("gemma", config.GEMMA_GGUF)
        self.assertIn("yolov8n", config.YOLO_NEF)
        self.assertIn("ggml-base", config.ASR_MODEL)
        self.assertEqual(config.READ_ENGINE, "easyocr")

    def test_no_leftover_replacement_models(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("app/config.py", "app/detect_kneron.py", "deploy/kneopi/install.sh", "deploy/kneopi/requirements.txt"):
            with open(os.path.join(root, name), encoding="utf-8") as f:
                self.assertIsNone(re.search(r"smolvlm_llama|SmolVLM-256M|tesseract|opus-mt|YoloV7_tiny", f.read()), name)


if __name__ == "__main__":
    unittest.main()
