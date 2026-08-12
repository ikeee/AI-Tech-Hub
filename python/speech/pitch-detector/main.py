"""实时音高检测（Python 示例）：基于 librosa 的 PYIN 基频估计。

与浏览器端实时音高检测（app/pages/speech/pitch-detector.vue）对应的「最简 Python 实现」，
仅供学习/对照展示，不参与 nuxt_AI 服务端调用。

依赖安装（Python 3.9+，CPU 即可）：
    pip install librosa numpy

用法：
    python main.py input.wav [--min-freq 80] [--max-freq 1200] [--top 10]

--min-freq / --max-freq  基频搜索范围（Hz）
--top                    输出前 N 个有声帧
"""

import argparse
import math
import sys

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_note(freq: float) -> tuple[str, int]:
    """频率 -> (音名如 C4, 音分偏差)"""
    midi = 69 + 12 * math.log2(freq / 440.0)
    r = round(midi)
    name = NOTE_NAMES[r % 12]
    octave = r // 12 - 1
    cents = round((midi - r) * 100)
    return f"{name}{octave}", cents


def analyze(audio: str, min_freq: float, max_freq: float, top: int) -> None:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import librosa

    print(f"[1/2] 读取音频: {audio}", flush=True)
    y, sr = librosa.load(audio, sr=None, mono=True)

    print(f"[2/2] PYIN 基频估计 (fmin={min_freq}, fmax={max_freq})…", flush=True)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=min_freq, fmax=max_freq, sr=sr, hop_length=512
    )
    times = librosa.times_like(f0, sr=sr, hop_length=512)

    rows: list[tuple[float, float, str, int]] = []
    for t, f, v in zip(times, f0, voiced_flag):
        if v and f == f:  # 排除 NaN（无声段）
            note, cents = freq_to_note(f)
            rows.append((t, f, note, cents))

    if not rows:
        print("未检测到有效音高（请检查音频是否包含人声/乐器声）", flush=True)
        return

    print(f"检测到 {len(rows)} 个有声帧，前 {min(top, len(rows))} 帧：", flush=True)
    for t, f, note, cents in rows[:top]:
        sign = "+" if cents > 0 else ""
        print(f"  t={t:7.2f}s  {f:7.1f} Hz  {note:5s} ({sign}{cents}¢)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="实时音高检测（librosa PYIN）")
    parser.add_argument("audio", help="输入音频（wav/mp3...）")
    parser.add_argument("--min-freq", type=float, default=80, help="最低频率 Hz")
    parser.add_argument("--max-freq", type=float, default=1200, help="最高频率 Hz")
    parser.add_argument("--top", type=int, default=10, help="输出前 N 个有声帧")
    args = parser.parse_args()

    analyze(args.audio, args.min_freq, args.max_freq, args.top)


if __name__ == "__main__":
    main()
