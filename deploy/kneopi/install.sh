#!/usr/bin/env bash
# 在 KNEO Pi 上安裝明路（一般使用者即可，不需要 root）。可重複執行，已完成的步驟會略過。
#   用法：在專案根目錄執行 bash deploy/kneopi/install.sh
#   裝好後再用 root 執行 deploy/kneopi/install-service.sh，設定開機自動啟動。
set -euo pipefail

HOME_DIR=${MINGLU_HOME:-$HOME/minglu}
SRC_DIR=$(cd "$(dirname "$0")/../.." && pwd)
HF=https://huggingface.co
NEF="$SRC_DIR/models/yolov8n_730.nef"

[ -f "$NEF" ] || { echo "[錯誤] 找不到 $NEF，請先在電腦上照 README_KNEOPI.md 轉換模型。" >&2; exit 1; }
for tool in gcc g++ make cmake git curl python3; do
    command -v "$tool" >/dev/null || { echo "[錯誤] 缺少 $tool，請先用 pacman 安裝。" >&2; exit 1; }
done

mkdir -p "$HOME_DIR"/{bin,models,src,app}

echo "[1/6] 複製程式與 YOLOv8n 模型"
if [ "$SRC_DIR" != "$HOME_DIR/app" ]; then
    rm -rf "$HOME_DIR/app/app"
    cp -r "$SRC_DIR/app" "$HOME_DIR/app/"
fi
cp "$NEF" "$HOME_DIR/models/"

echo "[2/6] 取得 Kneron 官方範例（NPU 韌體與 kp 輔助程式）"
[ -d "$HOME_DIR/kneopi-examples" ] || \
    git clone --depth 1 https://github.com/kneron/kneopi-examples "$HOME_DIR/kneopi-examples"

echo "[3/6] 建立 Python 環境"
# 要看得到系統已裝的 kp（Kneron PLUS）
[ -d "$HOME_DIR/venv" ] || python3 -m venv --system-site-packages "$HOME_DIR/venv"
"$HOME_DIR/venv/bin/python" -c "import kp" || {
    echo "[錯誤] 找不到 kp（Kneron PLUS Python）。請先照官方文件安裝後再執行一次。" >&2; exit 1; }
# /tmp 在記憶體裡，放不下 PyTorch，所以暫存檔改放磁碟；不留下載快取以節省空間
mkdir -p "$HOME_DIR/tmp"
# 先裝 CPU 版 PyTorch：PyPI 的 aarch64 版會拉好幾 GB 的 NVIDIA CUDA 套件，板子用不到
TMPDIR="$HOME_DIR/tmp" "$HOME_DIR/venv/bin/pip" install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
TMPDIR="$HOME_DIR/tmp" "$HOME_DIR/venv/bin/pip" install --no-cache-dir -r "$SRC_DIR/deploy/kneopi/requirements.txt"
rm -rf "$HOME_DIR/tmp"

echo "[4/6] 編譯 llama.cpp 與 whisper.cpp（用 2 個核心，避免記憶體不足）"
build() {  # build <名稱> <git 網址> <目標與執行檔>
    [ -x "$HOME_DIR/bin/$3" ] && return
    [ -d "$HOME_DIR/src/$1" ] || git clone --depth 1 "$2" "$HOME_DIR/src/$1"
    cmake -S "$HOME_DIR/src/$1" -B "$HOME_DIR/src/$1/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME_DIR/src/$1/build" --target "$3" -j2
    cp "$HOME_DIR/src/$1/build/bin/$3" "$HOME_DIR/bin/"
}
build whisper.cpp https://github.com/ggml-org/whisper.cpp whisper-cli
build llama.cpp https://github.com/ggml-org/llama.cpp llama-mtmd-cli

echo "[5/6] 下載模型（約 3GB）"
fetch() {  # fetch <網址> <目的地>
    [ -s "$2" ] || { curl -fL --retry 3 -o "$2.part" "$1"; mv "$2.part" "$2"; }
}
M="$HOME_DIR/models"
fetch "$HF/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin" "$M/ggml-base-q5_1.bin"
fetch "$HF/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q3_K_S.gguf" "$M/gemma-4-E2B-it-Q3_K_S.gguf"
fetch "$HF/ggml-org/gemma-4-E2B-it-GGUF/resolve/main/mmproj-gemma-4-E2B-it-Q8_0.gguf" "$M/mmproj-gemma-4-E2B-it-Q8_0.gguf"

"$HOME_DIR/venv/bin/python" "$SRC_DIR/tools/kneopi/extract_template.py" "$M/gemma-4-E2B-it-Q3_K_S.gguf" "$M/gemma-nothink.jinja"

echo "[6/6] 下載 EasyOCR 模型（第一次使用時才會抓，先在有網路時載好）"
"$HOME_DIR/venv/bin/python" -c "import easyocr; easyocr.Reader(['ch_tra', 'en'], gpu=False, verbose=False)"

echo "完成。接著用 root 執行 deploy/kneopi/install-service.sh"
