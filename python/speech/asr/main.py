"""语音识别（ASR）Python 示例：基于 faster-whisper 的文件离线转写。

与浏览器端 Whisper 离线转写（app/pages/speech/asr.vue）对应的「最简 Python 实现」，
仅供学习/对照展示，不参与 nuxt_AI 服务端调用（ASR 为纯浏览器功能）。

依赖安装（Python 3.9+，CPU 即可，模型首次运行自动下载）：
    pip install faster-whisper

用法：
    python main.py input.mp3 [--model small] [--lang zh] [--task transcribe] [--out transcript]

--model   模型大小：tiny / base / small / medium / large-v3（默认 small，中文建议 base 及以上）
--lang    音频语言（zh / en / ja ...），缺省自动检测
--task    transcribe=转写 / translate=翻译为英文
--out     输出前缀，生成 <out>.txt 与 <out>.srt（默认 transcript）
"""

import argparse
import sys


def format_ts(seconds: float) -> str:
    """秒 -> SRT 时间戳 00:00:00,000"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe(audio: str, model: str, lang: str | None, task: str, out: str) -> str:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from faster_whisper import WhisperModel

    # CPU 量化（int8）速度快、内存小；有 GPU 可改 device="cuda", compute_type="float16"
    print(f"[1/3] 加载模型: {model} (device=cpu, int8)", flush=True)
    wmodel = WhisperModel(model, device="cpu", compute_type="int8")

    print(f"[2/3] 正在转写: {audio} (lang={lang or 'auto'}, task={task})", flush=True)
    segments, info = wmodel.transcribe(
        audio,
        language=lang,
        task=task,
        vad_filter=True,          # 过滤静音段
        beam_size=5,
    )
    print(f"      检测语言: {info.language} (p={info.language_probability:.2f})", flush=True)

    lines: list[str] = []
    srt_blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text.strip()
        lines.append(text)
        srt_blocks.append(
            f"{i}\n{format_ts(seg.start)} --> {format_ts(seg.end)}\n{text}\n"
        )
        print(f"  [{seg.start:7.2f} -> {seg.end:7.2f}] {text}", flush=True)

    txt_path = f"{out}.txt"
    srt_path = f"{out}.srt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks))

    print(f"[3/3] 完成: {txt_path} / {srt_path}", flush=True)
    return txt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper 离线转写（faster-whisper）")
    parser.add_argument("audio", help="输入音频（mp3/wav/m4a...）")
    parser.add_argument("--model", default="small", help="tiny/base/small/medium/large-v3")
    parser.add_argument("--lang", default=None, help="语言代码，如 zh/en/ja，缺省自动检测")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--out", default="transcript", help="输出前缀")
    args = parser.parse_args()

    transcribe(args.audio, args.model, args.lang, args.task, args.out)


if __name__ == "__main__":
    main()
