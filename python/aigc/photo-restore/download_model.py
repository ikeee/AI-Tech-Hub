"""下载老照片修复模型（断点续传 + 重试）。

模型：
- CodeFormer:      https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth (~370MB)
- RealESRGAN_x2plus: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth (~64MB)
- 人脸检测权重：首次运行由 facexlib 自动下载（retinaface_resnet50）

用法：
    python download_model.py
"""

import os
import time
import urllib.request

CACHE_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.expanduser("~"), ".cache", "photo-restore"))
URLS = {
    os.path.join("CodeFormer", "codeformer.pth"): "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "RealESRGAN_x2plus.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
}


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"已存在: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, 101):
        try:
            print(f"下载: {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"完成: {dest} ({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")
            return
        except Exception as e:
            print(f"重试 {attempt} 次失败: {type(e).__name__}, 3 秒后继续...")
            time.sleep(3)
    raise SystemExit("下载失败次数过多")


def prepare_codeformer_src():
    """从 repo/ 拷贝 facelib 与 arch 文件到模块目录（避免与 pip basicsr 冲突）。"""
    import shutil
    module_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(module_dir, "repo")

    src_facelib = os.path.join(repo_dir, "facelib")
    dst_facelib = os.path.join(module_dir, "facelib")
    if os.path.isdir(src_facelib) and not os.path.isdir(dst_facelib):
        shutil.copytree(src_facelib, dst_facelib)
        print("已拷贝 facelib/")

    for name in ("vqgan_arch.py", "codeformer_arch.py"):
        src = os.path.join(repo_dir, "basicsr", "archs", name)
        dst = os.path.join(module_dir, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            content = open(src, encoding="utf-8").read()
            content = content.replace(
                "from basicsr.archs.vqgan_arch import *", "from vqgan_arch import *"
            )
            open(dst, "w", encoding="utf-8").write(content)
            print(f"已拷贝 {name}")


def main():
    print(f"下载目录: {CACHE_DIR}")
    for rel, url in URLS.items():
        download(url, os.path.join(CACHE_DIR, rel))
    prepare_codeformer_src()
    print("全部完成! 模型 + CodeFormer 源码就绪。")


if __name__ == "__main__":
    main()