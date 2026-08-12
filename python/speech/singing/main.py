"""歌声合成（Python 示例）：DiffSinger 使用说明。

⚠️ 该功能标注为「规划中」：DiffSinger 需要 GPU 训练声库（LJSpeech/VCTK 或自制数据集），
本机无 NVIDIA GPU，故仅提供最简实现说明，不注册为可运行 demo。

依赖安装（Linux + NVIDIA GPU 推荐）：
    pip install torch torchaudio
    git clone https://github.com/openvpi/DiffSinger.git

用法（需先准备声库与训练数据，详见官方 README）：
    cd DiffSinger
    python scripts/binarize.py --config configs/<your_config>.yaml
    python scripts/train.py --exp-dir exp/<your_experiment>
    python scripts/infer.py --exp-dir exp/<your_experiment> \
        --input data/notes.json --output output.wav

说明：
- 输入：乐谱（音符序列 + 歌词/音素）
- 输出：合成歌声 wav（44.1kHz）
- 中文社区活跃（openvpi），支持中文声库；需要至少 1 小时高质量歌声训练数据
- 训练与推理均需 GPU（CPU 推理极慢，数十分钟/句）

替代方向（本机可行）：
- 文生音乐（MusicGen）：见 speech/musicgen（已实现）
- 歌声/音色转换（Seed-VC）：需要额外权重与依赖，见 docs/RESEARCH-SPEECH.md S8/S15 说明
"""

import sys


def main() -> None:
    print("歌声合成（DiffSinger）为规划中功能：需要 GPU 与声库训练数据。", flush=True)
    print("请参考 docs/RESEARCH-SPEECH.md 与 openvpi/DiffSinger 官方文档。", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
