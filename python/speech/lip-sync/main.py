"""视频口型同步（Python 实现）：Wav2Lip（含 GAN 增强）。

与浏览器端口型同步（app/pages/speech/lip-sync.vue）对应的服务端实现，
由 nuxt_AI 通过异步队列调用（server/utils/lip-sync-queue.ts）。
本文件同时也是该功能的「Python 最简实现」示例。

依赖安装（Python 3.10-3.12，CPU 即可；权重已就绪：wav2lip_gan.pth + s3fd.pth）：
    pip install torch torchvision opencv-python numpy librosa tqdm numba

用法：
    python main.py input.mp4 audio.wav output.mp4

说明：
- 输入：含人脸的短视频（<60s，≤720p）+ 任意音频（wav/mp3）
- 输出：口型与音频同步的视频
- 模型权重为研究用途（非商用许可），页面需标注
"""

import argparse
import os
import subprocess
import sys

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")


def lip_sync(face_video: str, audio: str, output: str, checkpoint: str) -> str:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"[1/3] 转码音频为 wav…", flush=True)
    wav_path = os.path.join(REPO_DIR, "temp", "input_audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_path],
        check=True,
        timeout=120,
        capture_output=True,
    )
    print(f"[2/3] 人脸检测 + 口型同步中（首次运行加载模型）…", flush=True)
    subprocess.run(
        [
            sys.executable,
            os.path.join(REPO_DIR, "inference.py"),
            "--checkpoint_path", checkpoint,
            "--face", face_video,
            "--audio", wav_path,
            "--outfile", output,
            "--nosmooth",
        ],
        cwd=REPO_DIR,
        check=True,
        timeout=900,
    )
    print(f"[3/3] 输出: {output}", flush=True)
    print(output, flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="视频口型同步（Wav2Lip）")
    parser.add_argument("face", help="含人脸的输入视频")
    parser.add_argument("audio", help="输入音频")
    parser.add_argument("output", help="输出视频路径")
    parser.add_argument("--checkpoint", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "wav2lip_gan.pth"))
    args = parser.parse_args()

    lip_sync(args.face, args.audio, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
