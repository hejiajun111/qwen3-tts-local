# Qwen3‑TTS 本地语音克隆

使用 Qwen3‑TTS‑12Hz‑0.6B‑Base 从短参考录音合成中文语音。项目提供零样本克隆网页、WAV 合成脚本和单人微调入口。已在 Arch Linux 和 RTX 4070 Laptop（8 GB 显存）上打通合成、音色保存与加载、训练和检查点导出。[Qwen3‑TTS 官方项目](https://github.com/QwenLM/Qwen3-TTS)

## 部署

准备 Linux x86‑64、支持 CUDA 12.8 的 NVIDIA 驱动，以及 `uv`、`git`、`ffmpeg` 和 `sox`。从仓库根目录进入运行目录。脚本会创建 Python 3.12 虚拟环境，安装 PyTorch 2.9.1、TorchAudio 2.9.1、`qwen-tts` 0.1.1 和 Gradio 5，从 Hugging Face 下载模型并启动本地网页。

```bash
cd qwen-voice
bash deploy.sh
```

也可用 `bash deploy.sh modelscope` 从 ModelScope 下载。部署后可离线重启，运行 `bash serve.sh`，然后打开 [127.0.0.1:8000](http://127.0.0.1:8000)。已下载模型时，`bash serve.sh models/base-0.6b 8000 cpu` 可在 CPU 上启动。

## 零样本克隆

在「克隆并合成」页面录制或上传约 10 秒的单人录音，填入逐字原文，再输入新的中文句子。语种选 `Chinese`，保持「仅用说话人向量」关闭。生成的 WAV 可直接试听或下载。可在「保存/加载克隆音色」页面导出自己的 `.pt` 音色文件，供下次使用。[官方模型和参考音频](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)

命令行默认读取 `data/ref.wav` 和 `data/ref.txt`。把手机录音转换为单声道 WAV，并将录音原文以 UTF‑8 保存。

```bash
ffmpeg -i data/recording.m4a -ac 1 -ar 24000 -c:a pcm_s16le data/ref.wav
.venv/bin/python synthesize.py --text '请把会议改到周四下午三点。' --output outputs/base.wav
```

## 单人微调

从 `qwen-voice` 目录获取经过验证的官方数据集代码版本。已有 `Qwen3-TTS` 源码时，先核对其版本。

```bash
git init Qwen3-TTS
git -C Qwen3-TTS fetch --depth 1 https://github.com/QwenLM/Qwen3-TTS.git 022e286b98fbec7e1e916cb940cdf532cd9f488e
git -C Qwen3-TTS switch --detach FETCH_HEAD
```

录制数段同一说话人的短句，分别转为 24 kHz 单声道 PCM WAV。把文件路径和实际原文写入 `data/train_raw.jsonl`。所有样本填写同一个参考文件，路径相对于 `qwen-voice`。

```jsonl
{"audio":"data/utt0001.wav","text":"我们周四下午在实验楼碰面。","ref_audio":"data/ref.wav"}
{"audio":"data/utt0002.wav","text":"请把采集到的语音保存到本地。","ref_audio":"data/ref.wav"}
```

```bash
.venv/bin/python -u train.py --input data/train_raw.jsonl \
  --output runs/my-voice-smoke --speaker my_voice --epochs 1
bash serve.sh runs/my-voice-smoke/checkpoint-epoch-0 8001
```

在 [127.0.0.1:8001](http://127.0.0.1:8001) 选择 `My Voice` 和 `Chinese`。显存不足时，可先退出零样本网页，或给训练命令加 `--device cpu`。每次训练使用新的输出目录。

训练入口先在 CPU 提取音频编码，微调 talker 的最后四个块、末端归一化和码本输出层，再导出 CustomVoice 检查点。用四段 Qwen 公开参考音频的切片完成了一轮小试，CUDA 峰值分配约 2.45 GiB。实际音色相似度、读音和停顿以独立测试句的试听为准。

## 本地数据和上游许可

录音、文本标注、音色文件、权重、日志和生成结果存放在本地运行目录。模型和上游训练代码按上面的步骤获取。

`train.py` 参考 Qwen3‑TTS 的官方训练脚本，使用上游数据集和导出格式，并适配 0.6B 模型与有限显存环境。[上游训练代码](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e/finetuning) · [Apache 2.0 许可](licenses/Qwen3-TTS-Apache-2.0.txt)
