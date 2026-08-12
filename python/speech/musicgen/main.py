"""文生音乐（MusicGen）Python 实现：基于 Meta MusicGen（transformers 版本）。

与浏览器端文生音乐（app/pages/speech/musicgen.vue）对应的服务端实现，
由 nuxt_AI 通过常驻 worker 队列调用（server/utils/musicgen-queue.ts）。
本文件同时也是该功能的「Python 最简实现」示例。

依赖安装（Python 3.10-3.12，CPU 即可，模型首次运行自动下载 ~1.5GB）：
    pip install torch transformers soundfile

用法：
    python main.py --prompt "upbeat electronic dance music" --duration 5 --out music.wav

说明：
- 模型 facebook/musicgen-small（Meta 非商用许可，页面需标注）
- 32kHz 采样率，约 50 tokens = 1 秒音频
- 建议使用英文提示词（模型以英文训练为主）
"""

import argparse
import os
import sys


def generate(prompt: str, duration: float, out_path: str) -> str:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


    import soundfile as sf
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    # 限制 CPU 线程，避免吃满系统
    torch.set_num_threads(int(os.environ.get("MUSICGEN_THREADS", "6")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/3] 加载模型: facebook/musicgen-small (device={device})", flush=True)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    model.eval()

    tokens = max(1, int(duration * 50))  # 32kHz 下约 50 tokens/秒
    print(f"[2/3] 生成中: prompt={prompt!r} duration={duration}s tokens={tokens}", flush=True)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=tokens)

    print("[3/3] 保存音频…", flush=True)
    audio = audio_values[0, 0].cpu().numpy()
    sf.write(out_path, audio, 32000)
    print(out_path, flush=True)  # 供队列解析输出路径
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="文生音乐（MusicGen）")
    parser.add_argument("--prompt", required=True, help="音乐描述（建议英文）")
    parser.add_argument("--duration", type=float, default=5.0, help="音频时长（秒）")
    parser.add_argument("--out", default="music.wav", help="输出 wav 路径")
    args = parser.parse_args()

    generate(args.prompt, args.duration, args.out)


if __name__ == "__main__":
    main()
