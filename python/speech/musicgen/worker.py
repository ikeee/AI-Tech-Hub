"""文生音乐常驻 Worker：启动时加载 MusicGen-small 模型一次，循环处理 stdin JSON 任务。

协议（每行一个 JSON）：
  -> {"prompt": "...", "duration": 5, "out": "..."}
  <- {"type": "ready" | "loaded"} 或 {"type": "done", "out": "..."} 或 {"type": "error", "error": "..."}

多次生成只加载一次模型（~1.5GB），避免每次重新加载等待。
"""

import json
import os
import sys
import traceback


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")  # Windows 下 stdin 默认 GBK，Node 发送的是 UTF-8
    except Exception:
        pass


    import soundfile as sf
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    torch.set_num_threads(int(os.environ.get("MUSICGEN_THREADS", "6")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"type": "ready", "device": device}), flush=True)

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model.eval()
    print(json.dumps({"type": "loaded", "device": device}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
            tokens = max(1, int(float(task.get("duration", 5)) * 50))
            inputs = processor(text=[task["prompt"]], padding=True, return_tensors="pt")
            with torch.no_grad():
                audio_values = model.generate(**inputs, max_new_tokens=tokens)
            audio = audio_values[0, 0].cpu().numpy()
            sf.write(task["out"], audio, 32000)
            print(json.dumps({"type": "done", "out": task["out"]}), flush=True)
        except Exception as e:
            print(json.dumps({"type": "error", "error": str(e)}), flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
