#!/usr/bin/env bash
# 設定明路開機自動啟動（以 root 執行）。先跑完 install.sh 再執行這個。
#   用法：su - root 之後執行 bash deploy/kneopi/install-service.sh
#   會關閉桌面環境（佔約 900MB），要恢復：systemctl set-default graphical.target 後重開機。
set -euo pipefail

RUN_USER=${SUDO_USER:-alarm}     # alarm 不在 sudoers 裡時，先 su - root 再執行
HOME_DIR=${MINGLU_HOME:-$(getent passwd "$RUN_USER" | cut -d: -f6)/minglu}
SRC_DIR=$(cd "$(dirname "$0")/../.." && pwd)

[ -x "$HOME_DIR/venv/bin/python" ] || { echo "[錯誤] 找不到 $HOME_DIR/venv，請先執行 install.sh。" >&2; exit 1; }

# Gemma 的權重是 mmap 的檔案快取，被丟掉就要重讀 microSD。swappiness 調高，閒置資料才會先進 zram，
# 實測調低成 10 反而更慢（每題多讀數百 MB）
systemctl set-default multi-user.target
systemctl stop x11vnc x-server local-display 2>/dev/null || true
echo 'vm.swappiness=180' > /etc/sysctl.d/99-minglu.conf
sysctl -q vm.swappiness=180

# 每塊 KNEO Pi 預設都叫 kneo-pi，同一個網路上會撞名；改成 minglu，手機熱點的裝置清單也比較好認
hostnamectl set-hostname "${MINGLU_HOSTNAME:-minglu}"
# LCD 走 I2C（/dev/i2c-0 屬於 i2c 群組）
usermod -aG i2c "$RUN_USER"

# 測試時可能有手動啟動的伺服器佔著連接埠和 NPU
pkill -u "$RUN_USER" -f '[a]pp\.server' 2>/dev/null || true
sleep 2

sed -e "s|@HOME@|$HOME_DIR|g" -e "s|@USER@|$RUN_USER|g" \
    "$SRC_DIR/deploy/kneopi/minglu.service" > /etc/systemd/system/minglu.service
systemctl daemon-reload
systemctl enable minglu
systemctl restart minglu
echo "完成。查看狀態：systemctl status minglu；查看紀錄：journalctl -u minglu -f"
