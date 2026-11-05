#!/usr/bin/env bash
# 在 KNEO Pi 上安裝明路（以 root 執行）。可重複執行，已完成的步驟會略過。
#   用法：把專案複製到 KNEO Pi 後，在專案根目錄執行 sudo bash deploy/kneopi/install.sh
set -euo pipefail

HOME_DIR=${MINGLU_HOME:-/opt/minglu}
SRC_DIR=$(cd "$(dirname "$0")/../.." && pwd)
HF=https://huggingface.co
NEF="$SRC_DIR/models/yolov8n_730.nef"

[ -f "$NEF" ] || { echo "[錯誤] 找不到 $NEF，請先在電腦上照 README_KNEOPI.md 轉換模型。" >&2; exit 1; }

echo "[1/8] 安裝系統套件"
pacman -S --needed --noconfirm base-devel git cmake curl python avahi

mkdir -p "$HOME_DIR"/{bin,models,src}

echo "[2/8] 建立 4GB swap（Gemma 超過實體記憶體，需要 microSD 上的空間）"
if ! swapon --show=NAME --noheadings | grep -qx /swapfile; then
    [ -f /swapfile ] || { dd if=/dev/zero of=/swapfile bs=1M count=4096 status=progress; chmod 600 /swapfile; mkswap /swapfile; }
    swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap defaults 0 0' >> /etc/fstab
fi

echo "[3/8] 複製程式與 YOLOv8n 模型"
mkdir -p "$HOME_DIR/app"
cp -r "$SRC_DIR/app" "$HOME_DIR/app/"
cp "$NEF" "$HOME_DIR/models/"

echo "[4/8] 取得 Kneron 官方範例（NPU 韌體與 kp 輔助程式）"
[ -d "$HOME_DIR/kneopi-examples" ] || \
    git clone --depth 1 https://github.com/kneron/kneopi-examples "$HOME_DIR/kneopi-examples"

echo "[5/8] 建立 Python 環境"
# 要看得到系統已裝的 kp（Kneron PLUS）
[ -d "$HOME_DIR/venv" ] || python3 -m venv --system-site-packages "$HOME_DIR/venv"
"$HOME_DIR/venv/bin/pip" install -r "$SRC_DIR/deploy/kneopi/requirements.txt"
"$HOME_DIR/venv/bin/python" -c "import kp" || {
    echo "[錯誤] 找不到 kp（Kneron PLUS Python）。請先照官方文件安裝後再執行一次。" >&2; exit 1; }

echo "[6/8] 編譯 llama.cpp 與 whisper.cpp（只用 2 個核心，避免記憶體不足）"
build() {  # build <名稱> <git 網址> <目標與執行檔>
    [ -x "$HOME_DIR/bin/$3" ] && return
    [ -d "$HOME_DIR/src/$1" ] || git clone --depth 1 "$2" "$HOME_DIR/src/$1"
    cmake -S "$HOME_DIR/src/$1" -B "$HOME_DIR/src/$1/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME_DIR/src/$1/build" --target "$3" -j2
    cp "$HOME_DIR/src/$1/build/bin/$3" "$HOME_DIR/bin/"
}
build llama.cpp https://github.com/ggml-org/llama.cpp llama-mtmd-cli
build whisper.cpp https://github.com/ggml-org/whisper.cpp whisper-cli

echo "[7/8] 下載模型（約 3GB）"
fetch() {  # fetch <網址> <目的地>
    [ -s "$2" ] || { curl -fL --retry 3 -o "$2.part" "$1"; mv "$2.part" "$2"; }
}
M="$HOME_DIR/models"
fetch "$HF/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q3_K_S.gguf" "$M/gemma-4-E2B-it-Q3_K_S.gguf"
fetch "$HF/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mmproj-gemma-4-E2B-it-Q8_0.gguf" "$M/mmproj-gemma-4-E2B-it-Q8_0.gguf"
fetch "$HF/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin" "$M/ggml-base-q5_1.bin"
# EasyOCR 第一次使用時才會下載模型，先在有網路的時候載好
"$HOME_DIR/venv/bin/python" -c "import easyocr; easyocr.Reader(['ch_tra', 'en'], gpu=False, verbose=False)"

echo "[8/8] 設定開機自動啟動，網址 https://minglu.local:8443"
hostnamectl set-hostname minglu
systemctl enable --now avahi-daemon
cp "$SRC_DIR/deploy/kneopi/minglu.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable minglu
systemctl restart minglu
echo "完成。查看狀態：systemctl status minglu；查看紀錄：journalctl -u minglu -f"
