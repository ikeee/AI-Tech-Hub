"""会议纪要（Python 实现）：faster-whisper 转写 + WeSpeaker 说话人分离 + 合并标注。

与浏览器端会议纪要（app/pages/speech/meeting.vue）对应的服务端实现，
由 nuxt_AI 通过异步队列调用（server/utils/meeting-queue.ts）。
本文件同时也是该功能的「Python 最简实现」示例。

依赖安装（Python 3.10-3.12，CPU 即可，模型首次运行自动下载）：
    pip install torch torchaudio faster-whisper soundfile
    pip install git+https://github.com/wenet-e2e/wespeaker.git

用法：
    python main.py input.wav [--model base] [--lang zh] [--out transcript.json]

输出（stdout，JSON）：
    {"segments": [{"start":0.0,"end":2.3,"speaker":"SPEAKER_00","text":"..."}],
     "text": "SPEAKER_00: ...\nSPEAKER_01: ...", "summary_source": "..."}
"""

import argparse
import json
import os
import sys


def run(input_path: str, model_name: str, lang: str, out_path: str) -> None:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    os.environ.setdefault("OMP_NUM_THREADS", "6")

    # 1) ASR 转写（带时间戳）
    print("[1/4] 加载 faster-whisper 模型…", flush=True)
    from faster_whisper import WhisperModel

    wmodel = WhisperModel(model_name, device="cpu", compute_type="int8")
    print("[2/4] 转写中…", flush=True)
    segments, info = wmodel.transcribe(
        input_path,
        language=lang or None,
        vad_filter=True,
        beam_size=5,
    )
    asr = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]

    # 2) 说话人分离
    print("[3/4] 说话人分离（WeSpeaker）…", flush=True)
    import wespeaker

    spk_model = wespeaker.load_model("chinese")
    diar = spk_model.diarize(input_path)  # [{start, end, speaker}]

    # 3) 合并：按时间重叠最多的说话人标注每个 ASR 段
    # WeSpeaker diarize 返回 [(start, end, speaker), ...]
    def seg_bounds(d):
        # WeSpeaker: (utt, begin, end, label)
        if isinstance(d, dict):
            return d["start"], d["end"], d.get("speaker", "SPEAKER_00")
        if len(d) >= 4:
            return float(d[1]), float(d[2]), f"SPEAKER_{d[3]}"
        if len(d) >= 3:
            return float(d[0]), float(d[1]), str(d[2])
        return float(d[0]), float(d[1]), "SPEAKER_00"

    def best_speaker(start: float, end: float) -> str:
        best, best_overlap = "SPEAKER_00", 0.0
        for d in diar:
            ds, de, spk = seg_bounds(d)
            overlap = max(0.0, min(end, de) - max(start, ds))
            if overlap > best_overlap:
                best, best_overlap = spk, overlap
        return best

    merged = [
        {
            "start": s["start"],
            "end": s["end"],
            "speaker": best_speaker(s["start"], s["end"]),
            "text": s["text"],
        }
        for s in asr
    ]

    # 4) 输出
    print("[4/4] 生成结果…", flush=True)
    full_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in merged)
    result = {
        "segments": merged,
        "text": full_text,
        "language": info.language,
        "summary_source": full_text,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="会议纪要（转写 + 说话人分离）")
    parser.add_argument("input", help="输入音频（wav/mp3...）")
    parser.add_argument("--model", default="base", help="whisper 模型 tiny/base/small")
    parser.add_argument("--lang", default=None, help="语言代码，如 zh/en，缺省自动检测")
    parser.add_argument("--out", default="transcript.json", help="输出 JSON 路径")
    args = parser.parse_args()

    run(args.input, args.model, args.lang, args.out)


if __name__ == "__main__":
    main()
