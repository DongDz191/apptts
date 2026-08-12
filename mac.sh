#!/usr/bin/env bash
# Chạy trên Mac, không phải trên máy vast.
#
# Sửa 2 dòng dưới cho khớp thông tin instance của bạn, rồi:
#   bash mac.sh push      # đẩy code lên máy vast
#   bash mac.sh tunnel    # mở đường hầm, giữ cửa sổ này
#   bash mac.sh shell     # SSH vào kèm sẵn tunnel
#   bash mac.sh pull      # tải file .wav về ~/Downloads
set -euo pipefail

VAST_PORT="${VAST_PORT:-16885}"
VAST_HOST="${VAST_HOST:-ssh7.vast.ai}"

REMOTE_DIR="/workspace/qwen3-tts-vastai"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PORT="${APP_PORT:-7860}"
API_PORT="${API_PORT:-8000}"

case "${1:-help}" in
  push)
    echo "Đẩy $LOCAL_DIR -> $VAST_HOST:$REMOTE_DIR"
    ssh -p "$VAST_PORT" "root@$VAST_HOST" "mkdir -p $REMOTE_DIR"
    # scp dùng -P viết hoa, khác ssh dùng -p thường
    scp -P "$VAST_PORT" \
      "$LOCAL_DIR"/*.py "$LOCAL_DIR"/*.sh "$LOCAL_DIR"/requirements.txt "$LOCAL_DIR"/README.md \
      "root@$VAST_HOST:$REMOTE_DIR/"
    echo "Xong. Tiếp theo: bash mac.sh shell"
    ;;

  tunnel)
    echo "Mở tunnel $APP_PORT và $API_PORT. Giữ cửa sổ này, mở http://localhost:$APP_PORT"
    ssh -N -p "$VAST_PORT" "root@$VAST_HOST" \
      -L "$APP_PORT:localhost:$APP_PORT" \
      -L "$API_PORT:localhost:$API_PORT"
    ;;

  shell)
    ssh -p "$VAST_PORT" "root@$VAST_HOST" \
      -L "$APP_PORT:localhost:$APP_PORT" \
      -L "$API_PORT:localhost:$API_PORT" \
      -L 8080:localhost:8080
    ;;

  pull)
    mkdir -p "$HOME/Downloads/qwen3-tts-out"
    scp -P "$VAST_PORT" "root@$VAST_HOST:$REMOTE_DIR/*.wav" "$HOME/Downloads/qwen3-tts-out/" \
      || echo "Không có file .wav nào trong $REMOTE_DIR"
    echo "Đã tải về ~/Downloads/qwen3-tts-out"
    ;;

  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}"
    ;;
esac
