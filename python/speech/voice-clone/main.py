"""语音克隆（XTTS-v2 零样本克隆）—— 供 nuxt_AI 服务端调用。

用法:
    python main.py --ref 参考音频.wav --text "要合成的话" --lang zh-cn --out output.wav

说明:
- 一段 5-15 秒参考录音即可克隆音色，无需训练
- 首次运行自动下载 XTTS-v2 模型 (~1.8GB)，可用 HF_ENDPOINT 指定镜像
- CPU 可运行，句子越长越慢；通过环境变量 VC_THREADS 限制线程数
"""

import argparse
import os
import sys

# XTTS-v2 使用 Coqui CPML 许可（仅限非商业用途），自动同意许可确认
os.environ.setdefault("COQUI_TOS_AGREED", "1")

SUPPORTED_LANGS = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
]


def clone_voice(text: str, ref_wav: str, lang: str = "zh-cn", out_path: str = "output.wav") -> str:
    """用一段参考音频克隆音色并合成语音，返回输出文件路径。"""
    import torch
    from TTS.api import TTS

    # 限制线程数，避免吃满 CPU 拖慢系统
    threads = int(os.environ.get("VC_THREADS", "6"))
    torch.set_num_threads(threads)

    # 检测模型是否已缓存（下载目录），提示"加载"而非"下载"
    from pathlib import Path
    model_hint = "正在加载 XTTS-v2 模型（已缓存）…"
    try:
        cache = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "tts" / "tts_models--multilingual--multi-dataset--xtts_v2"
        if not (cache / "model.pth").exists():
            model_hint = "首次使用：正在下载 XTTS-v2 模型（约 1.8GB，请耐心等待）…"
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] {model_hint} (设备: {device}, 线程: {threads})", flush=True)
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print(f"[2/3] 正在合成: {text!r} (语言: {lang})", flush=True)
    tts.tts_to_file(
        text=text,
        speaker_wav=ref_wav,
        language=lang,
        file_path=out_path,
    )
    print(f"[3/3] 完成! 音频已保存到: {os.path.abspath(out_path)}", flush=True)
    return out_path


def main() -> None:
    # 让 Windows 控制台正常显示中文
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="本地语音克隆 (XTTS-v2, 零样本)")
    parser.add_argument("--ref", required=True, help="参考音频（你的声音样本），wav/mp3 均可")
    parser.add_argument("--text", required=True, help="要合成的文字")
    parser.add_argument("--lang", default="zh-cn", choices=SUPPORTED_LANGS, help="合成语言")
    parser.add_argument("--out", default="output.wav", help="输出音频路径")
    args = parser.parse_args()

    clone_voice(args.text, args.ref, args.lang, args.out)


if __name__ == "__main__":
    main()
