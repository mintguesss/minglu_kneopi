# 明路 KNEO Pi 版：交接文件

最後更新：2026-11-06。這份文件記錄比賽用的板子現在怎麼設定、出問題時怎麼查，以及開發過程中試過、不要再走的路。
功能、架構和安裝步驟請看 [README.md](README.md)；這裡只寫 README 沒有的現場資訊。

## 1. 現況

- 比賽用的版本是這個 repo 的 `main`，板子上跑的就是這一版。
- 實測通過：手機熱點連線、LCD 顯示、整台重開機自動恢復、手機重開網頁相機恢復。
- 速度（Samsung S24 + 手機熱點實測）：

| 操作 | 時間 |
|---|---|
| 點畫面（周遭、室內），YOLO 有偵測到東西 | 約 0.5~7 秒 |
| 點畫面，YOLO 沒偵測到、改問 Gemma | 約 45~50 秒 |
| 語音指令（切換模式等） | 約 6 秒 |
| 語音提問（Whisper 2~5 秒 + Gemma） | 約 30 秒 |
| 讀字 | 約 45 秒 |
| 開機或服務重啟後的第一題 | 會再慢 10~20 秒 |

## 2. 比賽當天

### 要帶的東西

- KNEO Pi + 電源
- **USB Wi-Fi 網卡**（Realtek RTL8821CU，USB ID `0bda:c820`）。板子本身沒有 Wi-Fi，沒有網卡手機就連不上。
- 1602 I2C LCD（已接在板子上）
- Samsung S24（熱點名稱 `ken`，密碼只存在板子和手機裡，不寫在這裡）
- 備用：USB 轉 UART 線（見第 5 節），出事時可以直接進板子的終端機

### 流程

1. 插好 Wi-Fi 網卡，手機開熱點 `ken`（2.4GHz）。
2. 板子開機。LCD 第一行會出現 IP，第二行從 `Loading...` 變成 `Ready :8443`（約 1~2 分鐘）。
3. 手機 Chrome 打開 `https://<LCD 第一行的 IP>:8443`。憑證警告按「進階」→「繼續前往」，允許相機和麥克風。
4. 上台前先問一題熱身，讓 Gemma 的模型讀進記憶體。

**熱點每次開關，IP 都可能變**，一律以 LCD 為準，不要用書籤裡舊的 IP。

### LCD 第二行的意思

| 顯示 | 意思 |
|---|---|
| `Loading...` | 正在載入模型 |
| `Ready :8443` | 可以用了 |
| `Listening` | Whisper 正在辨識語音 |
| `Thinking` | Gemma 正在回答 |
| `Look: <模式>` | 正在處理點畫面 |
| `2 person, 1 car` | YOLO 偵測到的東西 |
| `Computer`、`Nothing seen` | Gemma 回答裡提到的東西（中英對照表轉的，在 `app/lcd.py`） |
| `Ready (30s)` | 做完了，花了 30 秒（回答裡沒有對照表認得的詞） |
| `Error` | 這次處理失敗，看第 3 節查紀錄 |
| 第一行 `No network` | 板子沒連上熱點 |

## 3. 出問題時

| 症狀 | 原因與處理 |
|---|---|
| LCD 第一行 `No network` | 網卡沒插好，或熱點沒開、名稱不是 `ken`。熱點重開後等約 30 秒。還是不行就用 root 執行 `nmcli con up ken`。 |
| LCD 一直停在 `Loading...` | 模型還在載入，等 2 分鐘。超過 5 分鐘就看服務紀錄。 |
| LCD 完全沒字 | 服務沒起來，或 LCD 沒接好。背光亮但沒字時，調轉接板背面的藍色旋鈕。 |
| LCD 出現 ooOO、日文或奇怪符號 | I2C 傳輸錯位。程式每 10 秒會自動重寫，等一下就好。常常發生的話，把 LCD 的 VCC 改接 3.3V 試試（轉接板 5V、板子 I2C 3.3V，訊號在臨界邊緣）。 |
| 手機打不開網頁 | IP 變了（看 LCD），或手機沒開熱點。網址要有 `https://` 和 `:8443`。 |
| 手機相機全黑 | 重新整理網頁。還是黑的就把 Chrome 從最近使用的 App 滑掉再開（舊頁面佔著相機）。 |
| 手機唸成廣東話 | 手機的文字轉語音沒有國語（臺灣）語音，到系統設定裝「中文（臺灣）」語音。 |
| 語音很慢 | 看服務紀錄裡的 `[ask] 聽到：…（排隊 X 秒，辨識 Y 秒）`。排隊久：前一個請求（通常是 Gemma 描述）還在跑。辨識久：檢查是不是有兩個 whisper-server（`pgrep -c -x whisper-server` 應該是 1）。 |
| Gemma 一直答「沒有看到」 | 板子會把最後一張提問的畫面存在 `/tmp/minglu-last-ask.jpg`，拿出來看是不是鏡頭沒對準。 |
| `[warnning] The version of firmware ... is not the corresponding version` | 正常，NPU 照樣能用，不用理。 |
| `NPU accelerator could not be loaded` | 正常，是 LiteRT-LM 找不到 NPU 外掛，Gemma 本來就用 CPU。 |
| `取得 Tailscale 憑證失敗，改用自簽憑證` | 正常，現場用自簽憑證。 |

### 常用指令

看服務狀態和紀錄（一般帳號就能看）：

```bash
systemctl status minglu
```

```bash
journalctl -u minglu -f
```

重新啟動服務（root）：

```bash
systemctl restart minglu
```

沒有 root 時的替代做法：強制結束主程式，systemd 會在 5 秒後自動重啟（`Restart=on-failure`）：

```bash
kill -9 $(systemctl show -p MainPID --value minglu)
```

查板子在熱點上的 IP：

```bash
ip -4 -br addr show wlan0
```

## 4. 板子上的設定

| 項目 | 內容 |
|---|---|
| 系統 | KNEO Pi 官方 Arch Linux ARM，Python 3.14 |
| 主機名稱 | `minglu`（原本是 `kneo-pi`，改掉是因為每塊 KNEO Pi 都叫這個名字，會撞名） |
| 帳號 | `alarm`：明路服務用這個帳號跑，**沒有 sudo**。要 root 就 `su -` 輸入 root 密碼。 |
| 開機服務 | `/etc/systemd/system/minglu.service`（由 `deploy/kneopi/install-service.sh` 產生），已 enable |
| 桌面 | 已關閉（`multi-user.target`），省約 900MB 記憶體 |
| 記憶體 | zram 1GB + `/swapfile` 4GB（priority 10），`vm.swappiness=180`（`/etc/sysctl.d/99-minglu.conf`） |
| Wi-Fi | NetworkManager 連線 `ken`，autoconnect 開著 |
| LCD | 1602 + PCF8574，I2C 匯流排 0、位址 `0x27`；`alarm` 已加入 `i2c` 群組 |
| mDNS | `/etc/avahi/avahi-daemon.conf` 設 `allow-interfaces=wlan0`（避免在有線網路上自己跟自己撞名）。Android 手機解析不到 `.local`，現場用不到。 |
| Tailscale | 板子已登入（會耗一點 CPU）。比賽不需要；不想掛在帳號上可以 `tailscale logout`。 |

### 檔案位置（`/home/alarm/minglu`）

| 路徑 | 內容 |
|---|---|
| `app/app/` | 明路程式（這個 repo 的 `app/`） |
| `venv/` | Python 環境（`--system-site-packages`，才看得到系統裝的 Kneron `kp`） |
| `models/` | `yolov8n_730.nef`、`gemma-4-E2B-it.litertlm`（2.6GB）、`ggml-base-q5_1.bin`、`litert-cache/` |
| `bin/whisper-server` | whisper.cpp 編出來的伺服器 |
| `src/whisper.cpp/build/` | **不要刪**：whisper-server 從這裡載入共用函式庫 |
| `kneopi-examples/` | Kneron 官方範例，NPU 韌體從這裡讀 |

## 5. 怎麼連進板子

**先確認是自己的板子**：`cat /etc/machine-id` 應該以 `e9adf9fa` 開頭。
實驗室裡還有別塊 KNEO Pi，`kneo-pi.local` 會連到別人的板子。開發時曾經誤連過一次，還弄停了那塊板子的 NPU 服務，要請板主用 root 重啟 `nnm-usb-loopback`。

| 方法 | 怎麼做 |
|---|---|
| 手機熱點 | 電腦也連 `ken`，`ssh alarm@<LCD 上的 IP>` |
| Tailscale | 電腦和板子都在同一個 tailnet 時，`ssh alarm@<板子的 Tailscale IP>`（板子上 `tailscale ip -4` 查） |
| 網路線 | 接實驗室路由器，`eth0` 會拿到 DHCP 位址 |
| UART | 3.3V TTL（**不能接 5V 的 USB 轉 TTL**）：板子 pin 8 = TX、pin 10 = RX、pin 6 = GND；USB 轉 TTL 的 RX 接板子 TX、TX 接 RX。115200 bps、8N1。沒反應就對調 TX、RX。 |

Windows 上用 git bash 時，要用 `/c/Windows/System32/OpenSSH/ssh.exe`：git 內建的 ssh 讀不到中文路徑下的金鑰。

## 6. 更新程式

1. 在電腦上改這個 repo，先跑單元測試（板子上跑最準，見 README「測試」）。
2. 把 `app/` 複製到板子的 `~/minglu/app/app/`（整個資料夾換掉）。
3. 重啟服務（見第 3 節）。重啟後第一題會比較慢。
4. 推上 GitHub。

```bash
git push kneopi feat/kneopi-port:main
```

Git remote：`kneopi` = `mintguesss/minglu_kneopi`（這個 repo）。`origin` 是原本的 `kobocawaee/minglu`，**不要推到 origin**。

## 7. 試過、不要再走的路

| 試過的做法 | 結果 |
|---|---|
| llama.cpp 跑 Gemma | 模型比記憶體大，每個字都要從 microSD 重讀，一題約 9 分鐘。改用 LiteRT-LM。 |
| LiteRT-LM 設 `max_num_tokens=512` | 配置記憶體失敗，不要設。 |
| LiteRT-LM 的 YNNPACK、推測解碼、FP16 | 不支援或沒有比較快。 |
| `vm.swappiness=10` | 反而更慢：Gemma 權重是檔案快取，會被優先丟掉、重讀 microSD。改回 180。 |
| 每次用 whisper-cli 辨識 | 每次都把 Gemma 擠出記憶體，改成常駐的 whisper-server。 |
| Whisper 用預設 audio_ctx（1500） | 2.5 秒語音要算 19.5 秒；依錄音長度設定後約 3.6 秒，結果相同。 |
| EasyOCR 讀字 | 板子上要 90~160 秒，改用 Gemma。 |
| whisper-server 用 `PR_SET_PDEATHSIG` 跟著主程式結束 | 它跟著的是「建立它的執行緒」，每題結束就被關掉，辨識變成 29 秒。已拿掉。 |
| 手機用 `minglu.local` 連板子 | 板子廣播正常，但 Android 不解析連在自己熱點上的裝置的 `.local` 名稱。改用 LCD 顯示 IP。 |
| 手機 USB 網路共用、核心內建 Wi-Fi 驅動 | 核心沒有 RNDIS 驅動；但 RTL8821CU 的驅動可以另外載入，插上就能用。 |
| ADATA UV100 隨身碟放模型 | 它是 USB 2.0，速度跟 microSD 差不多，沒有用。 |

## 8. 待辦

- [ ] 語音提問約 30 秒，目標 20 秒內。板子 CPU 上幾乎不可能：光是 Gemma 看圖就要約 20 秒。可以試的：
  - USB 3.0 隨身碟或外接 SSD 放模型，插板子的 USB 3.0 孔（5000Mbps），預估省幾秒到十幾秒，還沒實測
  - 「預先看圖」：閒置時先讓 Gemma 看最新畫面，發問時只剩文字部分，預估 10~15 秒，但畫面可能是十幾秒前的
  - 讓手機自己跑 Gemma
- [ ] 真實場景的準確度（特別是小字、物品）
- [ ] 長時間運作的溫度（工作溫度上限 50°C）和 microSD 耐用度（swap 會寫入）
- [ ] 請另一塊 KNEO Pi 的板主重啟 `nnm-usb-loopback`
