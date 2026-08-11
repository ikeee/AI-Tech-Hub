"""下载 XTTS-v2 模型（带断点续传 + 自动重试，网络不好也能下完）。

默认从 HuggingFace 官方下载；网络不通时设置镜像再运行:
    $env:HF_ENDPOINT = "https://hf-mirror.com"   # Windows PowerShell
    export HF_ENDPOINT=https://hf-mirror.com      # Linux / macOS
"""

import os
import time

from huggingface_hub import hf_hub_download
from TTS.utils.manage import get_user_data_dir

REPO = "coqui/XTTS-v2"
FILES = ["model.pth", "config.json", "vocab.json", "hash.md5", "speakers_xtts.pth"]


def main():
    target = os.path.join(get_user_data_dir("tts"), "tts_models--multilingual--multi-dataset--xtts_v2")
    os.makedirs(target, exist_ok=True)
    print(f"下载目录: {target}")
    print(f"镜像: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")

    for name in FILES:
        for attempt in range(1, 101):  # 最多重试 100 次，断点续传
            try:
                path = hf_hub_download(
                    repo_id=REPO, filename=name, revision="main", local_dir=target
                )
                size_mb = os.path.getsize(path) / 1024 / 1024
                print(f"完成: {name} ({size_mb:.1f} MB)")
                break
            except Exception as e:
                print(f"重试 {name} 第 {attempt} 次失败: {type(e).__name__}, 3 秒后继续...")
                time.sleep(3)
        else:
            print(f"下载 {name} 失败次数过多，请检查网络后重新运行。")
            return

    # 写一个许可确认标记，省得运行时再问
    with open(os.path.join(target, "tos_agreed.txt"), "w", encoding="utf-8") as f:
        f.write("I have read, understood and agreed to the Terms and Conditions.")
    print("全部下载完成!")


if __name__ == "__main__":
    main()
