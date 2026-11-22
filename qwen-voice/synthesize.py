import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="使用本地 Qwen3-TTS 生成 WAV")
    parser.add_argument("--model", type=Path, default=Path("models/base-0.6b"))
    parser.add_argument("--reference", type=Path, default=Path("data/ref.wav"))
    parser.add_argument("--transcript", type=Path, default=Path("data/ref.txt"))
    parser.add_argument("--speaker", help="微调检查点中保存的说话人名称")
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.text.strip() or args.output.suffix.lower() != ".wav":
        parser.error("请填写目标文本，并指定 .wav 输出路径")
    if not (args.model / "model.safetensors").is_file():
        parser.error(f"模型权重缺失：{args.model / 'model.safetensors'}")
    if not args.speaker:
        if not args.reference.is_file() or not args.transcript.is_file():
            parser.error("请提供参考录音和同一段录音的 UTF-8 原文")
        transcript = args.transcript.read_text(encoding="utf-8-sig").strip()
        if not transcript:
            parser.error("参考原文为空")

    os.environ["HF_HUB_OFFLINE"] = "1"
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    if args.device == "cuda:0" and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请确认运行环境可访问 NVIDIA 显卡，或使用 --device cpu")
    model = Qwen3TTSModel.from_pretrained(
        str(args.model.resolve()), device_map=args.device,
        dtype=torch.bfloat16 if args.device == "cuda:0" else torch.float32,
        attn_implementation="sdpa", local_files_only=True,
    )
    if args.speaker:
        samples, sample_rate = model.generate_custom_voice(
            text=args.text.strip(), language=args.language, speaker=args.speaker,
        )
    else:
        samples, sample_rate = model.generate_voice_clone(
            text=args.text.strip(), language=args.language,
            ref_audio=str(args.reference), ref_text=transcript,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, samples[0], sample_rate, subtype="PCM_16")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
