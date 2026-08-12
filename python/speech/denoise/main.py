"""音频降噪增强（Python 实现）：基于 DeepFilterNet 的实时语音增强。

与浏览器端音频降噪（app/pages/speech/denoise.vue）对应的服务端实现，
由 nuxt_AI 通过异步队列调用（server/utils/denoise-queue.ts）。
本文件同时也是该功能的「Python 最简实现」示例。

依赖安装（Python 3.10-3.12，CPU 即可，模型首次运行自动下载 ~30MB）：
    pip install deepfilternet soundfile numpy

用法：
    python main.py input.wav output.wav [--model DeepFilterNet3]

--model   DeepFilterNet3（默认，效果最好）/ DeepFilterNet2（更快更小）
"""

import argparse
import os
import sys


def denoise(input_path: str, output_path: str, model_name: str) -> str:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import librosa
    import soundfile as sf
    import torch
    from df import enhance, init_df

    # 限制 CPU 线程，避免吃满系统
    os.environ.setdefault("OMP_NUM_THREADS", "6")

    print(f"[1/3] 加载模型: {model_name}", flush=True)
    model, df_state, _ = init_df(model_name or "DeepFilterNet3")

    print(f"[2/3] 读取音频: {input_path} (重采样 48kHz)", flush=True)
    audio, sr = librosa.load(input_path, sr=48000, mono=False)
    if audio.ndim == 1:
        audio = audio[None, :]  # [1, T] 单声道补通道维

    print("[3/3] 增强中…", flush=True)
    enhanced = enhance(model, df_state, torch.from_numpy(audio))
    enhanced_np = enhanced.cpu().numpy()

    sf.write(output_path, enhanced_np.T, sr)  # soundfile 约定 (frames, channels)
    print(output_path, flush=True)  # 供队列解析输出路径
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="音频降噪增强（DeepFilterNet）")
    parser.add_argument("input", help="输入音频路径（wav/mp3...）")
    parser.add_argument("output", help="输出音频路径（wav）")
    parser.add_argument("--model", default="DeepFilterNet3", help="DeepFilterNet3 / DeepFilterNet2")
    args = parser.parse_args()

    denoise(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
