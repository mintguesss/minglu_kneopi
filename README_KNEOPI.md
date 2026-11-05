# 明路：KNEO Pi 版

把電腦換成 Kneron KNEO Pi（KL730、2GB 記憶體），手機一樣用瀏覽器連線。畫面和聲音都由手機傳給板子，板子只回傳文字，由手機唸出來。

**這個版本沒有過馬路功能，也不判斷紅綠燈，請不要靠它過馬路。**

## 和電腦版的差別

| 功能 | 電腦版 | KNEO Pi 版 |
|---|---|---|
| 過馬路、紅綠燈 | 有 | **移除**；問到時回答不提供 |
| 物件偵測 | YOLOv8n（CUDA） | YOLOv8n（KL730 NPU，INT8 量化） |
| 周遭、室內、物品 | Gemma 3 4B | 先用偵測結果組句；偵測不到才問 Gemma 4 E2B |
| 語音提問與追問 | Gemma 3 4B | Gemma 4 E2B（Q3_K_S 量化） |
| 語音辨識 | Whisper large-v3-turbo | Whisper base（whisper.cpp，q5_1 量化） |
| 讀字 | EasyOCR，必要時由 Gemma 補讀 | EasyOCR（CPU），不用 Gemma 補讀 |

偵測只認得 COCO 80 類物品。「沒有偵測到」不代表前方沒有障礙物。

## 實測結果

2026-11-05 在 KNEO Pi 上量測，桌面環境已關閉。測試圖只有公車照片和一張合成的中文告示，語音是電腦合成的，真實相機畫面和人聲還沒測過。

| 功能 | 秒數 | 結果 |
|---|---|---|
| 周遭（YOLOv8n 組句） | 第一次約 7 秒，之後約 0.5 秒 | 正確 |
| 語音辨識 | 3~5 秒 | 3 句都聽得懂；「讀字」被聽成「獨自」，靠拼音比對修正 |
| 語音指令（辨識加判斷） | 約 13 秒內 | 正確 |
| 讀字 | 約 90~160 秒 | 文字全對 |
| Gemma 語音問答 | **約 520~550 秒** | 正確，回答很短 |

記憶體：Gemma 尖峰約 1.5GB，EasyOCR 約 1.1GB，所以兩者不會同時留在記憶體。

Gemma 慢的主因是模型比實體記憶體大：每生成一個字都要從 microSD 重讀權重，約 10 秒一個 token。這點調參數解決不了。

## 安裝

需要 KNEO Pi、32GB 的 microSD，並且要先把根分割區擴大到整張卡。Wi-Fi 要另接 USB 網卡（官方驗證過 TP-Link T2UB Nano、T3U Nano、T2U Nano）。安裝時要連網，之後可以離線使用。

1. 確認 `models/yolov8n_730.nef` 存在；沒有的話照下面「轉換 YOLOv8n」產生。
2. 用隨身碟或 `scp` 把專案資料夾複製到 KNEO Pi。
3. 用 root 建立 swap（編譯和 Gemma 都需要）：

```bash
dd if=/dev/zero of=/swapfile bs=1M count=4096 && chmod 600 /swapfile && mkswap /swapfile && swapon --priority 10 /swapfile && echo '/swapfile none swap defaults,pri=10 0 0' >> /etc/fstab
```

4. 用 `alarm` 帳號在專案資料夾裡安裝。編譯要幾個小時，中途失敗可以重新執行，已完成的步驟會略過：

```bash
bash deploy/kneopi/install.sh
```

5. 用 root 設定開機自動啟動。`alarm` 不在 sudoers 裡，要先 `su - root`：

```bash
MINGLU_HOME=/home/alarm/minglu bash /home/alarm/minglu-src/deploy/kneopi/install-service.sh
```

查看狀態和紀錄：

```bash
systemctl status minglu
```

```bash
journalctl -u minglu -f
```

## 省記憶體：關閉桌面

桌面環境佔約 900MB。用 root 暫時關閉，重開機後會自動回來：

```bash
systemctl stop x11vnc x-server local-display
```

上面的實測都是在關閉桌面的狀態下量的。

## 使用

1. 讓 KNEO Pi 和手機在同一個網路（見下面）
2. 手機用 Chrome 打開 `https://<板子的 IP>:8443`。查 IP：`ip -4 -o addr show eth0`（Wi-Fi 是 `wlan0`）
3. 第一次會出現憑證警告，選擇「進階」→「繼續前往」，並允許使用相機和麥克風

不要靠 `kneo-pi.local` 認板子：同一個網路上如果有別塊 KNEO Pi，它們的名稱相同，可能連到別人的板子。

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

3. 查 IP 並用手機連線。IP 可能變動，現場要先確認一次；手機的熱點設定頁可能也看得到已連線裝置。

## 轉換 YOLOv8n

在有 Docker 的電腦上做，KNEO Pi 上不需要。把 `yolov8n.pt`、`bus.jpg` 放進 `tools/kneopi/work/`，校正圖片放進 `tools/kneopi/work/calib/`（這裡用 COCO128 的 128 張）。

```bash
docker run --rm -v "$PWD/tools/kneopi:/w" -w /w/work python:3.11-slim bash -c "apt-get update && apt-get install -y libgl1 libglib2.0-0 && pip install ultralytics onnx onnxslim onnxruntime && python ../export_onnx.py"
```

```bash
docker run --rm -v "$PWD:/data1" kneron/toolchain:latest bash -c "source /workspace/miniconda/bin/activate onnx1.13 && python /data1/tools/kneopi/convert_nef.py"
```

## 還要確認的事

- [ ] 用真的手機連線：相機、麥克風、語音播放、等待時的「還在處理」提示
- [ ] 真實相機畫面的偵測和讀字品質（目前只測過一張照片和合成圖）
- [ ] 真人語音的辨識準確度
- [ ] Android 手機熱點下的連線
- [ ] 重開機後服務能自動恢復（重開機後桌面也會回來，Gemma 可能更慢）
- [ ] microSD 的耐用度（swap 會頻繁寫入）和長時間運作的溫度（工作溫度上限 50°C）

## 測試

單元測試不需要 NPU 或模型檔，只驗證程式邏輯：

```bash
python -m unittest tests.test_kneopi -v
```

在板子上實測：`tools/kneopi/test_npu.py`（NPU）、`tools/kneopi/bench.py`（各模型）、`tools/kneopi/e2e.py`（對跑著的伺服器做端到端測試）。
