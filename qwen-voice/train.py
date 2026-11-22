# Adapted from Qwen3-TTS/finetuning/sft_12hz.py.
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def read_manifest(path, sf):
    rows = []
    checked = set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("audio", "text", "ref_audio")
        ):
            raise ValueError(f"第 {number} 行需要填写 audio、text 和 ref_audio")
        for key in ("audio", "ref_audio"):
            audio_path = Path(row[key]).resolve(strict=True)
            row[key] = str(audio_path)
            if audio_path not in checked:
                info = sf.info(audio_path)
                if info.format != "WAV" or info.samplerate != 24000 or info.channels != 1:
                    raise ValueError(f"录音应为 24 kHz 单声道 WAV：{audio_path}")
                if info.frames < 1920:
                    raise ValueError(f"录音太短，无法提取音频编码：{audio_path}")
                checked.add(audio_path)
        row["text"] = row["text"].strip()
        rows.append(row)
    if not rows:
        raise ValueError("训练清单为空")
    if len({row["ref_audio"] for row in rows}) != 1:
        raise ValueError("单人训练请为所有样本填写同一份 ref_audio")
    return rows


def save_checkpoint(base, output, speaker, model, speaker_embedding, dtype, save_file):
    def ignore_weights(directory, names):
        ignored = {".cache"}
        if Path(directory) == base:
            ignored.add("model.safetensors")
        return ignored.intersection(names)

    with tempfile.TemporaryDirectory(prefix=".export-", dir=output.parent) as tmp:
        checkpoint = Path(tmp) / output.name
        shutil.copytree(base, checkpoint, ignore=ignore_weights)
        config_path = checkpoint / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["tts_model_type"] = "custom_voice"
        config["talker_config"]["spk_id"] = {speaker: 3000}
        config["talker_config"]["spk_is_dialect"] = {speaker: False}
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        weights = {
            key: value.detach().to(device="cpu", dtype=dtype if value.is_floating_point() else value.dtype)
            for key, value in model.state_dict().items()
            if not key.startswith("speaker_encoder.")
        }
        codec_key = "talker.model.codec_embedding.weight"
        weights[codec_key] = weights[codec_key].clone()
        weights[codec_key][3000] = speaker_embedding[0].to(device="cpu", dtype=dtype)
        save_file(weights, checkpoint / "model.safetensors")
        checkpoint.rename(output)


def training_loss(talker, batch, speaker_embedding):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    codec_mask = batch["codec_mask"]
    device = codec_ids.device
    text = talker.get_text_embeddings()(input_ids[:, :, 0]).to(device)
    text = talker.text_projection(text)
    text = text * batch["text_embedding_mask"]
    codec = talker.get_input_embeddings()(input_ids[:, :, 1].to(device)) * batch["codec_embedding_mask"]
    codec[:, 6, :] = speaker_embedding
    inputs = text + codec
    for group, embedding in enumerate(talker.code_predictor.get_input_embeddings(), start=1):
        inputs = inputs + embedding(codec_ids[:, :, group]) * codec_mask.unsqueeze(-1)

    outputs = talker(
        inputs_embeds=inputs[:, :-1, :], attention_mask=batch["attention_mask"][:, :-1],
        labels=batch["codec_0_labels"][:, 1:], output_hidden_states=True, use_cache=False,
    )
    hidden = outputs.hidden_states[0][-1][codec_mask[:, :-1]]
    _, sub_loss = talker.forward_sub_talker_finetune(codec_ids[codec_mask], hidden)
    return outputs.loss + 0.3 * sub_loss


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 0.6B 单人微调及 CustomVoice 导出")
    parser.add_argument("--input", required=True, type=Path, help="训练 JSONL，音频路径相对于当前目录")
    parser.add_argument("--output", required=True, type=Path, help="新的训练输出目录")
    parser.add_argument("--model", type=Path, default=Path("models/base-0.6b"))
    parser.add_argument("--speaker", default="my_voice")
    parser.add_argument("--device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()
    speaker = args.speaker.strip().lower()
    if not speaker or args.epochs < 1 or not 0 < args.lr < float("inf") or args.cpu_threads < 1:
        parser.error("需要非空说话人名称，以及大于零的 epochs、lr 和 cpu-threads")

    base = args.model.resolve()
    if args.output.exists() or args.output.resolve().is_relative_to(base):
        parser.error("请指定模型目录之外的新输出目录")
    dataset_dir = Path(__file__).resolve().parent / "Qwen3-TTS" / "finetuning"
    if not (dataset_dir / "dataset.py").is_file():
        parser.error("项目缺少 Qwen3-TTS 官方训练源码")

    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.device == "cuda:0":
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(dataset_dir))
    from dataset import TTSDataset

    try:
        rows = read_manifest(args.input, sf)
        config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if config.get("tts_model_type") != "base" or config.get("tts_model_size") != "0b6":
        parser.error("训练入口需要 Qwen3-TTS-12Hz-0.6B-Base")
    cuda = args.device == "cuda:0"
    if cuda and not torch.cuda.is_available():
        parser.error("CUDA 不可用，请使用宿主机终端或传入 --device cpu")

    args.output.mkdir(parents=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(0)
    dtype = torch.bfloat16 if cuda else torch.float32
    tts = Qwen3TTSModel.from_pretrained(
        str(base), device_map="cpu", dtype=dtype,
        attn_implementation="sdpa", local_files_only=True,
    )
    model = tts.model
    tokenizer = model.speech_tokenizer
    print(f"在 CPU 上编码 {len(rows)} 段录音", flush=True)
    for row in rows:
        codes = tokenizer.encode(row["audio"]).audio_codes[0]
        if codes.ndim != 2 or not codes.shape[0] or codes.shape[1] != model.talker.config.num_code_groups:
            raise ValueError(f"录音未生成有效的音频编码：{row['audio']}")
        row["audio_codes"] = codes.cpu().tolist()
    (args.output / "train_with_codes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    model.speech_tokenizer = None
    del tokenizer

    dataset = TTSDataset(rows, tts.processor, model.config)
    batches = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    model.requires_grad_(False).to(args.device)
    talker = model.talker
    # The frozen text vocabulary is large and each batch looks up only a few rows.
    talker.get_text_embeddings().cpu()
    for module in (talker.model.layers[-4:], talker.model.norm, talker.codec_head, talker.code_predictor.lm_head):
        module.float().requires_grad_(True)
    talker.code_predictor.config.use_cache = False
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, foreach=False)
    print(f"训练设备 {args.device}，可更新参数 {sum(param.numel() for param in params):,}", flush=True)
    with torch.no_grad():
        ref_mel = dataset[0]["ref_mel"].to(device=args.device, dtype=dtype)
        speaker_embedding = model.speaker_encoder(ref_mel)
    talker.train()

    accumulated = 4
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(batches):
            for key in batch.keys() - {"input_ids", "ref_mels"}:
                batch[key] = batch[key].to(args.device)
            group_size = min(accumulated, len(batches) - (step // accumulated) * accumulated)
            with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16, enabled=cuda):
                loss = training_loss(talker, batch, speaker_embedding)
            loss_value = loss.item()
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"第 {epoch + 1} 轮第 {step + 1} 步 loss 非有限值")
            (loss / group_size).backward()
            if (step + 1) % accumulated == 0 or step + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss_value
            if step % 10 == 0 or step + 1 == len(batches):
                print(f"轮次 {epoch + 1}，样本 {step + 1}/{len(batches)}，loss {loss_value:.4f}", flush=True)

        checkpoint = args.output / f"checkpoint-epoch-{epoch}"
        save_checkpoint(base, checkpoint, speaker, model, speaker_embedding, torch.bfloat16, save_file)
        print(f"平均 loss {total_loss / len(batches):.4f}，检查点 {checkpoint.resolve()}", flush=True)
    if cuda:
        print(f"CUDA 峰值分配 {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
