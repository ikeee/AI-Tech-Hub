"""文生图/图生图（SD-Turbo）模型下载说明。

stabilityai/sd-turbo 由 diffusers from_pretrained 自动下载到
~/.cache/huggingface（可通过 HF_ENDPOINT=https://hf-mirror.com 加速），
无需手动下载。本文件保留作为统一入口：
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    python download_model.py
"""

import os


def main():
    target = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    print("SD-Turbo 模型由 diffusers 自动下载，缓存目录:")
    print(f"  {target}")
    print("若首次生成失败，请设置 HF_ENDPOINT=https://hf-mirror.com 后重试。")


if __name__ == "__main__":
    main()