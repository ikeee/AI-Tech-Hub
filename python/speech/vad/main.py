"""语音活动检测（VAD）Python 实现：基于 Silero VAD。

与浏览器端语音活动检测（app/pages/speech/vad.vue）对应的服务端实现，
由 nuxt_AI 通过异步队列调用（server/utils/vad-queue.ts）。
本文件同时也是该功能的「Python 最简实现」示例。

依赖安装（Python 3.9+，CPU 即可，模型首次运行自动下载 ~2MB）：
    pip install torch silero-vad soundfile

用法：
    python main.py input.wav [--threshold 0.5] [--min-speech-ms 250] [--min-silence-ms 100]

输出（stdout，JSON）：
    {"speech_segments": [{"start": 0.25, "end": 2.10}, ...], "speech_seconds": 1.85, "total_seconds": 6.3}
"""

import argparse
import json
import sys


def detect(input_path: str, threshold: float, min_speech_ms: int, min_silence_ms: int) -> None:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import soundfile as sf
    from silero_vad import get_speech_timestamps, load_silero_vad

    print(f"[1/3] 加载 Silero VAD 模型…", flush=True)
    model = load_silero_vad()

    print(f"[2/3] 读取音频: {input_path} (16kHz)", flush=True)
    audio, sr = sf.read(input_path, dtype="float32")
    if sr != 16000:
        import numpy as np
        # 简单线性重采样到 16kHz
        ratio = sr / 16000
        n = int(len(audio) / ratio)
        audio = np.interp(np.arange(n) * ratio, np.arange(len(audio)), audio).astype("float32")
        sr = 16000

    print("[3/3] 检测语音段…", flush=True)
    stamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
    )
    segments = [
        {"start": round(s["start"] / sr, 2), "end": round(s["end"] / sr, 2)}
        for s in stamps
    ]
    result = {
        "speech_segments": segments,
        "speech_seconds": round(sum(s["end"] - s["start"] for s in segments), 2),
        "total_seconds": round(len(audio) / sr, 2),
        "segment_count": len(segments),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="语音活动检测（Silero VAD）")
    parser.add_argument("input", help="输入音频路径（wav/mp3...）")
    parser.add_argument("--threshold", type=float, default=0.5, help="语音概率阈值 0-1")
    parser.add_argument("--min-speech-ms", type=int, default=250, help="最短语音段毫秒")
    parser.add_argument("--min-silence-ms", type=int, default=100, help="最小静音间隔毫秒")
    args = parser.parse_args()

    detect(args.input, args.threshold, args.min_speech_ms, args.min_silence_ms)


if __name__ == "__main__":
    main()
