#!/usr/bin/env bash
# Chạy app trong tmux để không chết khi Mac sleep hoặc SSH đứt.
#   bash run.sh              # khởi động
#   bash run.sh logs         # xem log
#   bash run.sh stop         # dừng
set -euo pipefail

SESSION="${SESSION:-tts}"
PORT="${PORT:-7860}"
APP_DIR="${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

case "${1:-start}" in
  stop)
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Đã dừng." || echo "Không có phiên nào."
    exit 0
    ;;
  logs)
    exec tmux attach -t "$SESSION"
    ;;
esac

if [ -z "${APP_PASSWORD:-}" ]; then
  echo "CẢNH BÁO: chưa đặt APP_PASSWORD — trang web sẽ mở công khai."
  echo "  export APP_USER=admin APP_PASSWORD='mat-khau-cua-ban'"
  echo
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Phiên '$SESSION' đang chạy. Xem log: bash run.sh logs"
  exit 0
fi

tmux new-session -d -s "$SESSION" -c "$APP_DIR" \
  "HF_HOME='$HF_HOME' APP_USER='${APP_USER:-}' APP_PASSWORD='${APP_PASSWORD:-}' \
   python app.py --port $PORT ${EXTRA_ARGS:-}; echo; echo '--- app đã thoát, Enter để đóng ---'; read"

echo "Đang khởi động trong tmux '$SESSION'."
echo "  Xem log:  bash run.sh logs      (thoát ra: Ctrl+B rồi D)"
echo "  Dừng:     bash run.sh stop"
echo
echo "Trên Mac mở: http://localhost:$PORT"
echo "Chưa có tunnel thì chạy trên Mac:"
echo "  ssh -p <PORT> root@<HOST> -L $PORT:localhost:$PORT"
