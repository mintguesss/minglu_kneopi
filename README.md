# 明路 KNEO Pi 版

**MingLu on KNEO Pi：視障者的繁中隨身視覺助理，不用電腦，手機連一塊開發板就能用**

這是[明路](https://github.com/kobocawaee/minglu)的 KNEO Pi 版本。原本由筆電負責辨識，這個版本改由
Kneron KNEO Pi（KL730，2GB 記憶體）負責。手機仍然是眼睛和喇叭：手機用瀏覽器拍照、錄音，傳給板子，
板子回傳文字，再由手機唸出來。所有模型都在板子上離線執行，影像和聲音不會送到第三方雲端。

**這個版本沒有過馬路功能，也不判斷紅綠燈，請不要靠它過馬路。** 問到過馬路或燈號時，它會直接說不提供。

原本的電腦版說明見 [README_電腦版.md](README_電腦版.md)。

## 功能

| 操作 | 做什麼 | 用的模型 | 實測時間 |
|---|---|---|---|
| 點畫面（周遭、室內） | 說出前方有什麼、在哪個方位、多近 | YOLOv8n（KL730 NPU） | 約 0.5 秒（開機後第一次約 7 秒） |
| 點畫面（物品） | 說出拿在鏡頭前的東西是什麼 | Gemma 4 E2B | 約 25~36 秒 |
| 點畫面（讀字） | 照原樣唸出畫面上的文字 | Gemma 4 E2B | 約 40~48 秒 |
| 語音指令 | 切換模式、調語速字體、連續描述 | Whisper base | 約 6 秒 |
| 語音提問 | 「這是什麼」「前面有什麼」，可以追問 | Whisper base + Gemma 4 E2B | 約 30~40 秒 |

語音提問辨識完（約 2~5 秒）會先唸「聽到了，正在看」，等待時每 30 秒提示一次「還在處理」。

上表是 2026-11-06 在 KNEO Pi 上透過伺服器量測的結果，用公開照片、一張合成的中文告示和電腦合成語音測試。
另外用 Samsung S24 實際連線操作過，但真實場景的準確度還要更多測試。

## 系統架構

```
手機（Chrome）                        KNEO Pi
┌────────────────┐   HTTPS   ┌───────────────────────────────────────────┐
│ 相機、麥克風     │ ────────► │ app/server.py                              │
│ 唸出結果（TTS）  │ ◄──────── │  ├─ YOLOv8n → KL730 NPU（周遭、室內）       │
└────────────────┘   文字     │  ├─ Whisper base → whisper-server（常駐）   │
                             │  └─ Gemma 4 E2B → LiteRT-LM（常駐，CPU）    │
                             └───────────────────────────────────────────┘
```

| 模型 | 執行方式 | 授權 |
|---|---|---|
| [YOLOv8n](https://github.com/ultralytics/ultralytics)（COCO 80 類） | 轉成 KL730 的 NEF，在 NPU 上跑 | AGPL-3.0 |
| [Gemma 4 E2B](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm) | [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)，CPU，常駐記憶體 | Apache-2.0 |
| [Whisper base](https://huggingface.co/ggerganov/whisper.cpp)（q5_1） | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) 的 whisper-server，常駐 | MIT |

## 和電腦版的差別

| 功能 | 電腦版 | KNEO Pi 版 |
|---|---|---|
| 過馬路、紅綠燈 | 有 | **移除** |
| 周遭、室內 | Gemma 3 4B | YOLOv8n 組句；偵測不到才問 Gemma |
| 物品、語音提問 | Gemma 3 4B | Gemma 4 E2B |
| 語音辨識 | Whisper large-v3-turbo | Whisper base |
| 讀字 | EasyOCR，必要時由 Gemma 補讀 | Gemma 4 E2B |

## 為什麼這樣選

KNEO Pi 只有 2GB 記憶體和 4 核 1.1GHz 的 Cortex-A55，NPU 只能加速 CNN（YOLO 這類），不能跑 Gemma。以下是實測後的取捨：

- **Gemma 用 LiteRT-LM，不用 llama.cpp：** llama.cpp 的模型比記憶體大，每生成一個字都要從 microSD 重讀，一題約 9 分鐘。LiteRT-LM 版可以常駐，一題約 22 秒。
- **影像用最低解析度（70 tokens）：** 影像編碼固定約 19.5 秒，是這塊板子最大的瓶頸。140 tokens 要多等約 30 秒。
- **Whisper 常駐：** 每次重新啟動 whisper-cli 會把 Gemma 擠出記憶體，Gemma 每題要多讀 300~700MB 的 microSD。
- **讀字改用 Gemma：** EasyOCR 加 PyTorch 在這塊板子上要 90~160 秒，記憶體也不夠和 Gemma 同時常駐。
- **物品交給 Gemma：** YOLOv8n 會把耳機認成滑鼠。601 類的 Open Images 版 YOLOv8n 也試過，耳機 3 張只認對 1 張。
- **`vm.swappiness` 設 180：** Gemma 的權重是 mmap 的檔案快取。調低成 10 時，物品從約 27 秒變成 42 秒。
- **沒用的方法：** Whisper tiny（更慢，還會重複句子）、先送圖再問（多 6 秒）、縮小圖片、speculative decoding、YNNPACK。

**還能再快的地方：** microSD 只有約 35MB/s。把模型放到 USB 3.0 的隨身碟或固態硬碟，應該能再省幾秒，但還沒實測。

## 需要的東西

- Kneron KNEO Pi，官方系統（Arch Linux ARM，已內建 Kneron PLUS 的 `kp`）
- 32GB 以上的 microSD，根分割區要擴大到整張卡
- USB Wi-Fi 網卡（官方驗證過 TP-Link T2UB Nano、T3U Nano、T2U Nano），或網路線
- Android 手機（測過 Samsung S24）
- 一台有 Docker 的電腦：只有重新轉換 YOLOv8n 時需要，`models/yolov8n_730.nef` 已附在 repo 裡

## 安裝

安裝時板子要能上網，之後可以離線使用。

1. 把這個 repo 複製到 KNEO Pi（`git clone` 或隨身碟）。
2. 用 root 建立 4GB swap：

```bash
dd if=/dev/zero of=/swapfile bs=1M count=4096 && chmod 600 /swapfile && mkswap /swapfile && swapon --priority 10 /swapfile && echo '/swapfile none swap defaults,pri=10 0 0' >> /etc/fstab
```

3. 用 `alarm` 帳號在 repo 資料夾裡安裝。會建立 Python 環境、編譯 whisper-server、下載約 2.7GB 的模型：

```bash
bash deploy/kneopi/install.sh
```

4. 用 root 設定開機自動啟動。這一步會關閉桌面環境（省約 900MB）並設定 swappiness。`alarm` 不在 sudoers 裡，要先 `su - root`：

```bash
MINGLU_HOME=/home/alarm/minglu bash /home/alarm/<repo 資料夾>/deploy/kneopi/install-service.sh
```

查看狀態和紀錄：

```bash
systemctl status minglu
```

```bash
journalctl -u minglu -f
```

開機後要約 1 分鐘才能使用（載入 Gemma 並熱身一次）。要恢復桌面：用 root 執行 `systemctl set-default graphical.target` 後重開機。

板子不能上網時：在電腦上下載模型、套件和原始碼，複製到板子的 `~/minglu`，再設定 `PIP_NO_INDEX=1` 和 `PIP_FIND_LINKS=<套件資料夾>` 後執行 `install.sh`，已存在的檔案會略過。

## 使用

1. 讓板子和手機在同一個網路。
2. 手機用 Chrome 打開 `https://<板子的 IP>:8443`。查 IP：`ip -4 -o addr show wlan0`（網路線是 `eth0`）。
3. 第一次會出現憑證警告，選擇「進階」→「繼續前往」，並允許相機和麥克風。

不要靠 `kneo-pi.local` 這個名稱找板子：同一個網路上如果有別塊 KNEO Pi，名稱會一樣。

### 比賽現場：Android 手機熱點

出發前先用同一支手機設定好：

1. 手機開熱點，固定名稱和密碼，頻段選 2.4GHz。
2. 板子連一次熱點，之後開機會自動連：

```bash
nmcli device wifi connect "<熱點名稱>" password "<熱點密碼>"
```

3. 查板子的 IP，用手機連線。IP 可能變動，現場要先確認一次；手機的熱點設定頁可能也看得到已連線裝置。

### 板子有 Tailscale 時

板子能上網並登入 Tailscale 後，伺服器會自動使用 Tailscale 的正式憑證，手機裝 Tailscale 後在哪都能連，也不會有憑證警告。

## 專案結構

```
app/                        系統本體（和電腦版共用，用 MINGLU_PROFILE=kneopi 切換）
  config.py                 kneopi 設定檔的所有參數
  server.py                 手機網頁與 API（/describe、/ask 分兩階段）
  detect_kneron.py          YOLOv8n 在 KL730 NPU 上推論、後處理
  scene.py                  把偵測結果組成繁中句子
  backends/gemma_litert.py  Gemma 4 E2B（LiteRT-LM）
  voice.py                  whisper-server、語音指令與提問
deploy/kneopi/              安裝腳本、開機服務、Python 套件清單
models/yolov8n_730.nef      轉好的 YOLOv8n NPU 模型
tools/kneopi/               模型轉換與實機測試工具
tests/test_kneopi.py        單元測試（不需要板子）
```

## 轉換 YOLOv8n

在有 Docker 的電腦上做。把 `yolov8n.pt`、`bus.jpg` 放進 `tools/kneopi/work/`，校正圖片放進 `tools/kneopi/work/calib/`（這裡用 COCO128 的 128 張）：

```bash
docker run --rm -v "$PWD/tools/kneopi:/w" -w /w/work python:3.11-slim bash -c "apt-get update && apt-get install -y libgl1 libglib2.0-0 && pip install ultralytics onnx onnxslim onnxruntime && python ../export_onnx.py"
```

```bash
docker run --rm -v "$PWD:/data1" kneron/toolchain:latest bash -c "source /workspace/miniconda/bin/activate onnx1.13 && python /data1/tools/kneopi/convert_nef.py"
```

模型切在偵測頭的最後一層卷積，DFL 解碼與 NMS 在 `app/detect_kneron.py` 用 numpy 做。用工具鏈模擬器比對公車照片，結果和原模型接近（信心略低，邊緣的小物件可能漏掉）。

## 測試

單元測試不需要板子：

```bash
MINGLU_PROFILE=kneopi python -m unittest tests.test_kneopi -v
```

在板子上：

| 工具 | 用途 |
|---|---|
| `tools/kneopi/test_npu.py` | NPU 推論速度與結果 |
| `tools/kneopi/test_objects.py` | 比較不同 YOLOv8n 模型 |
| `tools/kneopi/bench.py` | 各模型單獨的耗時與記憶體 |
| `tools/kneopi/e2e.py` | 對跑著的伺服器做端到端測試（模擬手機） |

## 已知限制與待辦

- [ ] 語音提問約 30~40 秒，目標是 20 秒內。板子上幾乎不可能；最有希望的是讓手機自己跑 Gemma
- [ ] 試 USB 3.0 儲存裝置能省多少時間
- [ ] Android 手機熱點下的連線
- [ ] 板子的 LCD 顯示狀態
- [ ] 重開機後服務能自動恢復
- [ ] 真實場景的物品辨識和讀字準確度（特別是小字）
- [ ] microSD 的耐用度（swap 會寫入）和長時間運作的溫度（工作溫度上限 50°C）
- YOLOv8n 只認得 COCO 80 類，「沒有偵測到」不代表前方沒有障礙物
- Gemma 回答很短，有時會漏掉重點（例如把公車照片說成「兩位男士在停車場」）

## 授權

本專案以 **[GNU AGPL-3.0](LICENSE)** 授權（因使用 Ultralytics YOLOv8）。各模型依其各自的授權條款使用。
`external/ImVisible/` 保留其原本的 MIT 授權。
