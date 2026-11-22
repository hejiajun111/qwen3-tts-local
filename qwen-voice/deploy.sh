#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

model_source=${1:-hf}
if [[ $# -gt 1 || ( $model_source != hf && $model_source != modelscope ) ]]; then
    printf '用法：bash deploy.sh [hf|modelscope]\n' >&2
    exit 2
fi

mkdir -p models data outputs runs
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi
python=.venv/bin/python
"$python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "部署需使用 Python 3.12")'
if ! "$python" -m pip --version >/dev/null 2>&1; then
    "$python" -m ensurepip --default-pip
fi

if ! command -v sox >/dev/null; then
    printf '系统缺少 SoX。按手册可在 Arch Linux 上用 sudo pacman -S sox 安装。\n' >&2
fi

export PIP_TIMEOUT=${PIP_TIMEOUT:-120}
"$python" -m pip install --index-url https://pypi.org/simple 'pip>=26.1'
export PIP_RESUME_RETRIES=${PIP_RESUME_RETRIES:-10}

# Use PyPI's wheels for the packages whose mirror downloads failed.
"$python" -m pip install --index-url https://pypi.org/simple --only-binary=:all: \
    'triton==3.5.1' 'nvidia-cudnn-cu12==9.10.2.21' 'nvidia-cublas-cu12==12.8.4.1'
"$python" -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.9.1+cu128' 'torchaudio==2.9.1+cu128'
"$python" -m pip install 'qwen-tts==0.1.1' 'gradio>=5,<6' \
    'torch==2.9.1+cu128' 'torchaudio==2.9.1+cu128'
"$python" -m pip check
"$python" - <<'PY'
import torch
import torchaudio

print(f"PyTorch {torch.__version__}, TorchAudio {torchaudio.__version__}, CUDA {torch.version.cuda}")
device = torch.empty(1, device="cuda:0").device
print(device, torch.cuda.get_device_name(device))
PY

model_id=Qwen/Qwen3-TTS-12Hz-0.6B-Base
model_dir=./models/base-0.6b
if [[ $model_source == modelscope ]]; then
    "$python" -m pip install modelscope
    .venv/bin/modelscope download --model "$model_id" --local_dir "$model_dir"
else
    .venv/bin/hf download "$model_id" --local-dir "$model_dir"
fi

for file in config.json model.safetensors speech_tokenizer/config.json speech_tokenizer/model.safetensors; do
    if [[ ! -s $model_dir/$file ]]; then
        printf '模型文件缺失或为空：%s\n' "$model_dir/$file" >&2
        exit 1
    fi
done

export HF_HUB_OFFLINE=1
exec .venv/bin/qwen-tts-demo "$model_dir" --ip 127.0.0.1 --port 8000 --no-flash-attn
