"""音频可视化（Python 示例）：用 librosa 提取波形包络与频谱数据。

与浏览器端音频可视化（app/pages/speech/visualizer.vue）对应的「最简 Python 实现」，
仅供学习/对照展示，不参与 nuxt_AI 服务端调用（可视化在浏览器完成）。

依赖安装（Python 3.9+）：
    pip install librosa numpy

用法：
    python main.py input.mp3 --bins 200

输出（stdout，JSON）：
    {"duration": 6.3, "sample_rate": 22050, "waveform": [...], "spectrogram": {...}}
"""

import argparse
import json
import sys


def analyze(input_path: str, bins: int) -> None:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import librosa
    import numpy as np

    print(f"[1/2] 读取音频: {input_path}", flush=True)
    y, sr = librosa.load(input_path, sr=None, mono=True)
    duration = len(y) / sr

    print(f"[2/2] 提取波形/频谱 (bins={bins})…", flush=True)
    # 波形包络：降采样到 bins 个点（取绝对值 + 平均）
    frame = max(1, len(y) // bins)
    envelope = [
        float(np.mean(np.abs(y[i * frame : (i + 1) * frame]))) if len(y[i * frame : (i + 1) * frame]) else 0.0
        for i in range(min(bins, len(y) // frame))
    ]
    # 频谱：STFT 平均幅值谱（取前 128 个频点）
    S = np.abs(librosa.stft(y, n_fft=2048))
    spectrum = np.mean(S, axis=1)[:128].tolist()

    result = {
        "duration": round(duration, 2),
        "sample_rate": sr,
        "waveform": envelope,
        "spectrum": [round(v, 6) for v in spectrum],
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="音频可视化数据提取（librosa）")
    parser.add_argument("input", help="输入音频路径")
    parser.add_argument("--bins", type=int, default=200, help="波形包络点数")
    args = parser.parse_args()

    analyze(args.input, args.bins)


if __name__ == "__main__":
    main()
