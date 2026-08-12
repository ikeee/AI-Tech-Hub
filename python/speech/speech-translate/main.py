"""语音翻译（Python 实现）：faster-whisper task=translate（语音 → 英文文本）。

与浏览器端语音翻译（app/pages/speech/speech-translate.vue）对应的服务端实现，
由 nuxt_AI 通过异步队列调用（server/utils/speech-translate-queue.ts）。
本文件同时也是该功能的「Python 最简实现」示例。

依赖安装（Python 3.9+，CPU 即可，模型首次运行自动下载）：
    pip install faster-whisper soundfile

用法：
    python main.py input.wav [--model base]

输出（stdout，JSON）：
    {"translation": "...", "language": "zh", "segments": [...]}
"""

import argparse
import json
import sys


def translate(input_path: str, model_name: str) -> None:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from faster_whisper import WhisperModel

    print(f"[1/3] 加载模型: {model_name}", flush=True)
    wmodel = WhisperModel(model_name, device="cpu", compute_type="int8")

    print(f"[2/3] 语音转写并翻译为英文: {input_path}", flush=True)
    segments, info = wmodel.transcribe(
        input_path,
        task="translate",  # 关键：translate 任务输出英文
        vad_filter=True,
        beam_size=5,
    )
    segs = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]
    print("[3/3] 完成", flush=True)
    result = {
        "translation": "\n".join(s["text"] for s in segs),
        "language": info.language,
        "segments": segs,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="语音翻译（faster-whisper translate）")
    parser.add_argument("input", help="输入音频路径")
    parser.add_argument("--model", default="base", help="tiny/base/small")
    args = parser.parse_args()

    translate(args.input, args.model)


if __name__ == "__main__":
    main()
