#!/usr/bin/env bash
# 在 KNEO Pi 上安裝明路（一般使用者即可，不需要 root）。可重複執行，已完成的步驟會略過。
#   用法：在專案根目錄執行 bash deploy/kneopi/install.sh
#   裝好後再用 root 執行 deploy/kneopi/install-service.sh，設定開機自動啟動。
#   板子不能上網時：先把模型放進 ~/minglu/models、原始碼放進 ~/minglu/src、官方範例放進
#   ~/minglu/kneopi-examples，再設定 PIP_NO_INDEX=1 和 PIP_FIND_LINKS=<套件資料夾> 後執行。
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

echo "[1/5] 複製程式與 YOLOv8n 模型"
if [ "$SRC_DIR" != "$HOME_DIR/app" ]; then
    rm -rf "$HOME_DIR/app/app"
    cp -r "$SRC_DIR/app" "$HOME_DIR/app/"
fi
cp "$NEF" "$HOME_DIR/models/"

echo "[2/5] 取得 Kneron 官方範例（NPU 韌體）"
[ -d "$HOME_DIR/kneopi-examples" ] || \
    git clone --depth 1 https://github.com/kneron/kneopi-examples "$HOME_DIR/kneopi-examples"

echo "[3/5] 建立 Python 環境"
# 要看得到系統已裝的 kp（Kneron PLUS）
[ -d "$HOME_DIR/venv" ] || python3 -m venv --system-site-packages "$HOME_DIR/venv"
"$HOME_DIR/venv/bin/python" -c "import kp" || {
    echo "[錯誤] 找不到 kp（Kneron PLUS Python）。請先照官方文件安裝後再執行一次。" >&2; exit 1; }
"$HOME_DIR/venv/bin/pip" install --no-cache-dir -r "$SRC_DIR/deploy/kneopi/requirements.txt"

echo "[4/5] 編譯 whisper.cpp（用 2 個核心，避免記憶體不足）"
if [ ! -x "$HOME_DIR/bin/whisper-cli" ]; then
    [ -d "$HOME_DIR/src/whisper.cpp" ] || \
        git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$HOME_DIR/src/whisper.cpp"
    cmake -S "$HOME_DIR/src/whisper.cpp" -B "$HOME_DIR/src/whisper.cpp/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HOME_DIR/src/whisper.cpp/build" --target whisper-cli -j2
    cp "$HOME_DIR/src/whisper.cpp/build/bin/whisper-cli" "$HOME_DIR/bin/"
fi

echo "[5/5] 下載模型（約 2.7GB）"
fetch() {  # fetch <網址> <目的地>
    [ -s "$2" ] || { curl -fL --retry 3 -o "$2.part" "$1"; mv "$2.part" "$2"; }
}
M="$HOME_DIR/models"
fetch "$HF/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin" "$M/ggml-base-q5_1.bin"
fetch "$HF/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm" "$M/gemma-4-E2B-it.litertlm"

echo "完成。接著用 root 執行 deploy/kneopi/install-service.sh"
