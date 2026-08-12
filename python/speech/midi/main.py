"""音频转 MIDI（Python 实现）：基于 Spotify basic-pitch（多乐器转录）。

与浏览器端音频转 MIDI（app/pages/speech/midi.vue）对应的服务端实现，
由 nuxt_AI 通过异步队列调用（server/utils/midi-queue.ts）。
本文件同时也是该功能的「Python 最简实现」示例。

依赖安装（Python 3.10-3.12，CPU 即可，模型首次运行自动下载 ~100MB）：
    pip install basic-pitch soundfile

用法：
    python main.py input.wav output.mid

说明：
- basic-pitch 支持钢琴/吉他等多乐器，输出 MIDI 到指定文件
"""

import argparse
import json
import os
import sys


def transcribe(input_path: str, output_path: str) -> None:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import shutil
    import tempfile
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    print(f"[1/3] 读取音频: {input_path}", flush=True)
    print("[2/3] 转写中（basic-pitch，首次运行需下载模型）…", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        predict_and_save(
            [input_path],
            tmp,
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
        # 输出文件: <name>_basic_pitch.mid
        base = os.path.splitext(os.path.basename(input_path))[0]
        cand = os.path.join(tmp, f"{base}_basic_pitch.mid")
        if not os.path.exists(cand):
            # 兜底：找目录下任意 .mid
            mids = [f for f in os.listdir(tmp) if f.endswith(".mid")]
            if not mids:
                print(json.dumps({"error": "no midi generated"}), flush=True)
                sys.exit(1)
            cand = os.path.join(tmp, mids[0])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(cand, output_path)

    print(json.dumps({"midi_path": output_path, "size": os.path.getsize(output_path)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="音频转 MIDI（basic-pitch）")
    parser.add_argument("input", help="输入音频路径")
    parser.add_argument("output", help="输出 MIDI 路径")
    args = parser.parse_args()

    transcribe(args.input, args.output)


if __name__ == "__main__":
    main()
