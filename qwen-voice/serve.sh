#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -gt 3 ]]; then
    printf '用法：bash serve.sh [模型目录] [端口] [cuda:0|cpu]\n' >&2
    exit 2
fi

model=${1:-models/base-0.6b}
port=${2:-8000}
device=${3:-cuda:0}
case "$device" in
    cuda:0) dtype=bfloat16 ;;
    cpu) dtype=float32 ;;
    *) printf '设备应为 cuda:0 或 cpu\n' >&2; exit 2 ;;
esac

for file in config.json model.safetensors speech_tokenizer/config.json speech_tokenizer/model.safetensors; do
    if [[ ! -s $model/$file ]]; then
        printf '模型文件缺失：%s\n' "$model/$file" >&2
        exit 1
    fi
done

export HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False
exec .venv/bin/qwen-tts-demo "$model" --ip 127.0.0.1 --port "$port" \
    --device "$device" --dtype "$dtype" --no-flash-attn
