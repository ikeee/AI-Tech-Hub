"""音频分离演示：基于 demucs 将混音拆分为人声/鼓/贝斯/伴奏。

使用预训练的 htdemucs 模型，对输入 wav 文件做分轨分离，
输出 drums、bass、other、vocals 四个 stem 文件。

依赖安装：
    pip install demucs torchaudio

用法：
    python main.py input.wav output_dir/
"""

import os
import sys

import torchaudio
from demucs.pretrained import get_model

# stem 顺序与 demucs 模型输出一致
STEM_NAMES = ("drums", "bass", "other", "vocals")


def separate(input_path: str, output_dir: str) -> list[str]:
    """加载音频、分离音轨、保存各 stem，返回输出文件路径列表。"""
    model = get_model("htdemucs")
    model.eval()

    wav, sr = torchaudio.load(input_path)  # [channels, samples]
    # 分离需要批次维度，apply 返回 [batch, tracks, channels, samples]
    sources = model.apply(wav.unsqueeze(0))[0]

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    outputs: list[str] = []
    for name, stem in zip(STEM_NAMES, sources):
        out_path = os.path.join(output_dir, f"{base}_{name}.wav")
        torchaudio.save(out_path, stem.cpu(), sr)
        outputs.append(out_path)
    return outputs


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python main.py <input.wav> [output_dir]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "separated"
    results = separate(input_path, output_dir)
    for path in results:
        print(path)


if __name__ == "__main__":
    main()
