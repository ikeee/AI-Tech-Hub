"""语音克隆常驻 Worker：启动时加载 XTTS-v2 模型一次，循环处理 stdin JSON 任务。

协议（每行一个 JSON）：
  -> {"ref": "...", "text": "...", "lang": "zh-cn", "out": "..."}
  <- {"type": "ready" | "loaded" | "done", "out": "..."} 或 {"type": "error", "error": "..."}

这样多次合成只需加载一次模型（~1.8GB），避免每次重新加载等待。
"""

import json
import os
import sys
import traceback

# XTTS-v2 使用 Coqui CPML 许可（仅限非商业用途），自动同意
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")  # Windows 下 stdin 默认 GBK，Node 发送的是 UTF-8
    except Exception:
        pass

    import torch
    import torchaudio

    # CPU 环境无 torchcodec（pip 版依赖 CUDA torch，且 CPU torchcodec 不兼容 FFmpeg 8）：
    # torchaudio 默认后端会走 torchcodec 报错，这里显式切到 soundfile 后端。
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

    from TTS.api import TTS

    torch.set_num_threads(int(os.environ.get("VC_THREADS", "6")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"type": "ready", "device": device}), flush=True)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(json.dumps({"type": "loaded", "device": device}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
            tts.tts_to_file(
                text=task["text"],
                speaker_wav=task["ref"],
                language=task.get("lang", "zh-cn"),
                file_path=task["out"],
            )
            print(json.dumps({"type": "done", "out": task["out"]}), flush=True)
        except Exception as e:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
