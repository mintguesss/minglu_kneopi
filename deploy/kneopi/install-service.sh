#!/usr/bin/env bash
# 設定明路開機自動啟動（以 root 執行）。先跑完 install.sh 再執行這個。
#   用法：su - root 之後執行 bash deploy/kneopi/install-service.sh
set -euo pipefail

RUN_USER=${SUDO_USER:-alarm}     # alarm 不在 sudoers 裡時，先 su - root 再執行
HOME_DIR=${MINGLU_HOME:-$(getent passwd "$RUN_USER" | cut -d: -f6)/minglu}
SRC_DIR=$(cd "$(dirname "$0")/../.." && pwd)

[ -x "$HOME_DIR/venv/bin/python" ] || { echo "[錯誤] 找不到 $HOME_DIR/venv，請先執行 install.sh。" >&2; exit 1; }

sed -e "s|@HOME@|$HOME_DIR|g" -e "s|@USER@|$RUN_USER|g" \
    "$SRC_DIR/deploy/kneopi/minglu.service" > /etc/systemd/system/minglu.service
systemctl daemon-reload
systemctl enable minglu
systemctl restart minglu
echo "完成。查看狀態：systemctl status minglu；查看紀錄：journalctl -u minglu -f"
