#!/usr/bin/env bash
# Cài đặt trên máy vast.ai vừa thuê. Chạy một lần:
#   bash setup.sh
set -euo pipefail

WORK="${WORK:-/workspace}"
export HF_HOME="${HF_HOME:-$WORK/hf}"        # cache model nằm trên disk của instance
export PIP_ROOT_USER_ACTION=ignore
export DEBIAN_FRONTEND=noninteractive

echo "==> Kiểm tra GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "    KHÔNG thấy nvidia-smi. Instance này hỏng driver — hủy và thuê máy khác."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> Gói hệ thống"
apt-get update -qq || echo "    apt-get update lỗi, bỏ qua"
apt-get install -y -qq ffmpeg libsndfile1 git curl tmux >/dev/null 2>&1 \
  || echo "    một số gói không cài được, thử tiếp"

echo "==> Thư viện Python"
pip install -qU pip wheel
pip install -qU qwen-tts gradio soundfile fastapi "uvicorn[standard]"

echo "==> Kiểm tra torch thấy CUDA"
python - <<'PY'
import sys, torch
print(f"    torch {torch.__version__} · cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("    LỖI: torch không thấy CUDA. Image này thiếu bản torch GPU.")
    print("    Cài lại theo hướng dẫn tại https://pytorch.org rồi chạy lại setup.sh")
    sys.exit(1)
print(f"    GPU: {torch.cuda.get_device_name(0)}")
PY

echo "==> flash-attn (tùy chọn, có thể mất 5–15 phút)"
if MAX_JOBS=4 pip install -qU flash-attn --no-build-isolation 2>/dev/null; then
  echo "    ok — engine sẽ tự dùng flash_attention_2"
else
  echo "    bỏ qua — engine tự rơi về sdpa, vẫn chạy bình thường"
fi

echo "==> Tải trọng số về $HF_HOME"
mkdir -p "$HF_HOME"
python - <<'PY'
from huggingface_hub import snapshot_download

# Bỏ bớt dòng nào bạn không dùng để tiết kiệm disk và thời gian tải.
repos = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
]
for r in repos:
    print(f"    {r}")
    snapshot_download(r, max_workers=8)
PY

SUGGESTED=$(head -c 12 /dev/urandom | base64 | tr -d '/+=' | head -c 12)
cat <<EOF

Cài xong. Chạy app:

  export HF_HOME=$HF_HOME
  export APP_USER=admin
  export APP_PASSWORD='$SUGGESTED'
  bash run.sh

Rồi trên Mac mở http://localhost:7860
EOF
