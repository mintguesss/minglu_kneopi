# 明路：KNEO Pi 版

把電腦換成 Kneron KNEO Pi（KL730、2GB 記憶體），手機一樣用瀏覽器連線。

**這個版本沒有過馬路功能，也不判斷紅綠燈，請不要靠它過馬路。** 目前還沒有在實機上測試過。

## 和電腦版的差別

| 功能 | 電腦版 | KNEO Pi 版 |
|---|---|---|
| 過馬路、紅綠燈 | 有 | **移除**；問到時回答不提供 |
| 物件偵測 | YOLOv8n（CUDA） | YOLOv8n（KL730 NPU，INT8 量化） |
| 周遭、室內、物品 | Gemma 3 4B | 先用偵測結果組句；偵測不到才問 Gemma 4 E2B |
| 語音提問與追問 | Gemma 3 4B | Gemma 4 E2B（Q3_K_S 量化） |
| 語音辨識 | Whisper large-v3-turbo | Whisper base（whisper.cpp，q5_1 量化） |
| 讀字 | EasyOCR，必要時由 Gemma 補讀 | EasyOCR（CPU），不用 Gemma 補讀 |

Gemma 模型加影像模組約 2.9GB，超過 2GB 記憶體，靠 mmap 和 swap 才放得下，**一次回答可能要好幾分鐘**。
偵測只認得 COCO 80 類物品。「沒有偵測到」不代表前方沒有障礙物。

## 安裝

需要 KNEO Pi、32GB 以上的 microSD、USB Wi-Fi 網卡（官方驗證過 TP-Link T2UB Nano、T3U Nano、T2U Nano）。
安裝時 KNEO Pi 要連網，之後可以離線使用。

1. 確認 `models/yolov8n_730.nef` 存在；沒有的話照下面「轉換 YOLOv8n」產生。
2. 用隨身碟或 `scp` 把專案資料夾複製到 KNEO Pi。
3. 在 KNEO Pi 的專案資料夾裡執行：

```bash
sudo bash deploy/kneopi/install.sh
```

腳本會裝好系統套件、建立 4GB swap、編譯 llama.cpp 與 whisper.cpp、下載模型，並設定開機自動啟動。
編譯可能要很久。中途失敗可以重新執行，已完成的步驟會略過。

## 使用

1. 讓 KNEO Pi 和手機在同一個網路（見下面）
2. 手機用 Chrome 打開 `https://minglu.local:8443`；打不開時改用 IP（`nmcli device show wlan0 | grep IP4.ADDRESS`）
3. 第一次會出現憑證警告，選擇「進階」→「繼續前往」，並允許使用相機和麥克風

### 開發測試：用目前的網路

KNEO Pi 接網路線，或用下面的指令連 Wi-Fi，手機連同一個網路。如果網路擋裝置互連（常見於公司、學校或訪客網路），就要換網路。

```bash
nmcli device wifi connect "<目前網路的名稱>" password "<密碼>"
```

### 比賽現場：Android 手機熱點

出發前先用同一支手機設定好：

1. 手機開熱點，固定名稱和密碼，頻段選 2.4GHz（USB 網卡比較容易連上）。
2. KNEO Pi 連一次熱點，之後開機會自動連：

```bash
nmcli device wifi connect "<熱點名稱>" password "<密碼>"
```

3. 測試 `minglu.local` 能不能打開。不行的話，查 IP 並改用 IP 連線；手機的熱點設定頁可能也看得到已連線裝置。IP 可能變動，所以現場要先確認一次。

現場不需要外網，手機的行動數據關掉也可以試試，先確認這樣熱點還能用。

## 轉換 YOLOv8n

在有 Docker 的電腦上做，KNEO Pi 上不需要。把 `yolov8n.pt`、`bus.jpg` 放進 `tools/kneopi/work/`，校正圖片放進 `tools/kneopi/work/calib/`（這裡用 COCO128 的 128 張）。

```bash
docker run --rm -v "$PWD/tools/kneopi:/w" -w /w/work python:3.11-slim bash -c "apt-get update && apt-get install -y libgl1 libglib2.0-0 && pip install ultralytics onnx onnxslim onnxruntime && python ../export_onnx.py"
```

```bash
docker run --rm -v "$PWD:/data1" kneron/toolchain:latest bash -c "source /workspace/miniconda/bin/activate onnx1.13 && python /data1/tools/kneopi/convert_nef.py"
```

## 上機後要確認的事

- [ ] `python -c "import kp"` 可以執行，NPU 連得上（程式用 `usb_port_ids=[0]`，和官方範例預設相同）
- [ ] 偵測框位置正確，物品名稱合理
- [ ] Gemma 真的跑得動：實際秒數、記憶體尖峰（`free -h`）、中文品質。太慢的話可以關掉語音提問
- [ ] Gemma 的影像模組（ggml-org）能和模型檔（unsloth）搭配使用
- [ ] EasyOCR 和 PyTorch 能在這個 Python 版本安裝，讀字的速度可以接受
- [ ] Whisper base 的中文準確度；不夠的話試 `ggml-small`，但記憶體用量會增加
- [ ] 手機熱點下能否用 `minglu.local` 連線
- [ ] microSD 的耐用度（swap 會頻繁寫入）和長時間運作的溫度（工作溫度上限 50°C）

## 測試

```bash
python -m unittest tests.test_kneopi -v
```

不需要 NPU 或模型檔，只驗證程式邏輯。
