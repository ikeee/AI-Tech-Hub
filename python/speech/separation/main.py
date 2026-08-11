"""音频分离演示：基于 Demucs 将混音拆分为人声/鼓/贝斯/伴奏。

使用预训练的 htdemucs 等模型，对输入音频文件做分轨分离，
输出 drums、bass、other、vocals 四个 stem 文件。

依赖安装：
    pip install demucs torchaudio

用法：
    python main.py input.wav output_dir/ [--model htdemucs] [--two-stems vocals]

--model      模型名称：htdemucs（默认）/ htdemucs_ft / mdx / mdx_extra
--two-stems  只分离指定声部及其伴奏（vocals/drums/bass/other）
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

# 限制 torch 线程数，避免分离时吃满 CPU 导致系统卡顿。
# 可通过环境变量 SEPARATION_THREADS 覆盖（默认 6）。
torch.set_num_threads(int(os.environ.get("SEPARATION_THREADS", "6")))
torch.set_num_interop_threads(int(os.environ.get("SEPARATION_INTEROP_THREADS", "1")))

# stem 顺序与 demucs 模型输出一致
STEM_NAMES = ("drums", "bass", "other", "vocals")


def separate(input_path: str, output_dir: str, model_name: str = "htdemucs", two_stems: str | None = None) -> list[str]:
    """加载音频、分离音轨、保存各 stem，返回输出文件路径列表。"""
    model = get_model(model_name)
    model.eval()

    # 直接用 soundfile 读取（不依赖 torchaudio / TorchCodec / 系统 FFmpeg DLL）
    data, sr = sf.read(input_path, dtype="float32", always_2d=True)  # [samples, channels]
    wav = torch.from_numpy(data.T.copy())  # [channels, samples]
    # demucs 只接受立体声：单声道复制为双声道
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    # 分离需要批次维度，apply_model 返回 [batch, tracks, channels, samples]
    sources = apply_model(model, wav.unsqueeze(0))[0]

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    outputs: list[str] = []

    stem_map = dict(zip(STEM_NAMES, sources))

    if two_stems:
        # 双轨模式：指定声部 + 其余部分（伴奏 = 输入 - 指定声部）
        if two_stems not in stem_map:
            raise ValueError(f"Unknown two-stems value: {two_stems} (expected one of {', '.join(STEM_NAMES)})")
        stem = stem_map[two_stems].cpu()
        accompaniment = wav - stem

        stem_path = os.path.join(output_dir, f"{base}_{two_stems}.wav")
        acc_path = os.path.join(output_dir, f"{base}_no_{two_stems}.wav")
        sf.write(stem_path, stem.numpy().T, sr, subtype="PCM_16")
        sf.write(acc_path, accompaniment.numpy().T, sr, subtype="PCM_16")
        outputs = [stem_path, acc_path]
    else:
        # 四轨模式：drums / bass / other / vocals
        for name, stem in zip(STEM_NAMES, sources):
            out_path = os.path.join(output_dir, f"{base}_{name}.wav")
            sf.write(out_path, stem.cpu().numpy().T, sr, subtype="PCM_16")
            outputs.append(out_path)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Demucs 音频分离")
    parser.add_argument("input", help="输入音频文件路径")
    parser.add_argument("output_dir", nargs="?", default="separated", help="输出目录")
    parser.add_argument("--model", default="htdemucs", help="模型名称 (htdemucs/htdemucs_ft/mdx/mdx_extra)")
    parser.add_argument("--two-stems", default=None, help="双轨分离指定声部 (vocals/drums/bass/other)")
    args = parser.parse_args()

    results = separate(args.input, args.output_dir, args.model, args.two_stems)
    for path in results:
        print(path)


if __name__ == "__main__":
    main()
