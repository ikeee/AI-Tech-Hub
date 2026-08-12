"""老照片修复（Real-ESRGAN + CodeFormer）— Python 服务端实现。

流程：
1. Real-ESRGAN x2 对整图背景超分增强
2. CodeFormer 人脸检测 + 细节修复（fidelity 滑杆控制保真度）
3. 修复后的人脸贴回超分背景

依赖安装（Python 3.11，CPU 版 torch 必须先行）：
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install realesrgan basicsr facexlib opencv-python-headless pillow numpy
    git clone --depth 1 https://github.com/sczhou/CodeFormer.git repo   # 在 python/aigc/photo-restore/ 下
    python download_model.py                                            # 下载 codeformer.pth + RealESRGAN_x2plus.pth

用法：
    python main.py input.jpg --fidelity 0.5 --upscale 2 --out output
"""

import argparse
import os
import sys

import torch

# basicsr 1.4.2 兼容新版 torchvision：functional_tensor 已合并进 functional，
# 以 sys.modules 注入兼容模块（属性注入对子模块 import 无效）
import sys
import types
import torchvision.transforms as _tv_transforms
if "torchvision.transforms.functional_tensor" not in sys.modules:
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.rgb_to_grayscale = _tv_transforms.functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _ft

CACHE_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.expanduser("~"), ".cache", "photo-restore"))
CODE_FORMER_WEIGHTS = os.path.join(CACHE_DIR, "CodeFormer", "codeformer.pth")
ESRGAN_WEIGHTS = os.path.join(CACHE_DIR, "RealESRGAN_x2plus.pth")

# facelib（CodeFormer 人脸检测/对齐/解析）+ vqgan/codeformer arch
# 由 download_model.py 从 repo/ 拷贝到本目录（避免与 pip basicsr 冲突）
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

# pip basicsr 的 utils.misc 缺少 get_device（新版 basicSR 才有），注入兼容实现
import basicsr.utils.misc as _bmisc
if not hasattr(_bmisc, "get_device"):
    def _get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _bmisc.get_device = _get_device

_restorer = None


def get_restorer():
    """加载一次：RealESRGAN + CodeFormer + 人脸检测器，之后常驻复用。"""
    global _restorer
    if _restorer is not None:
        return _restorer

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    device = torch.device("cpu")

    if not os.path.exists(CODE_FORMER_WEIGHTS) or not os.path.exists(ESRGAN_WEIGHTS):
        raise FileNotFoundError(
            f"模型缺失，请先运行: python download_model.py\n  {CODE_FORMER_WEIGHTS}\n  {ESRGAN_WEIGHTS}"
        )

    # ---- RealESRGAN x2（背景超分）----
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan.utils import RealESRGANer

    esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path=ESRGAN_WEIGHTS,
        model=esrgan_model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=False,
    )

    # 注册 CodeFormer 结构（拷贝自 CodeFormer repo，兼容 pip basicsr registry）
    import codeformer_arch  # noqa: F401

    # ---- CodeFormer（人脸修复）----
    from basicsr.utils.registry import ARCH_REGISTRY

    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]
    ).to(device)
    checkpoint = torch.load(CODE_FORMER_WEIGHTS, map_location="cpu")
    net.load_state_dict(checkpoint["params_ema"])
    net.eval()

    # ---- 人脸检测/对齐（facexlib；权重首次运行自动下载）----
    from facelib.utils.face_restoration_helper import FaceRestoreHelper

    face_helper = FaceRestoreHelper(
        2, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
        save_ext="png", use_parse=True, device=device,
    )

    _restorer = {"net": net, "helper": face_helper, "bg": bg_upsampler, "device": device}
    return _restorer


def run(input_path: str, params: dict, out_path: str) -> str:
    import cv2
    import numpy as np
    from basicsr.utils import imwrite, img2tensor, tensor2img
    from torchvision.transforms.functional import normalize

    if not input_path or not os.path.exists(input_path):
        raise ValueError("请上传需要修复的图片")

    fidelity = float(params.get("fidelity", 0.5))
    upscale = int(params.get("upscale", 2))

    r = get_restorer()
    net, face_helper, bg_upsampler, device = r["net"], r["helper"], r["bg"], r["device"]

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法读取图片，请使用 PNG / JPG / WebP")

    # 限制输入尺寸，控制 CPU 耗时
    h, w = img.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    face_helper.clean_all()
    face_helper.read_image(img)
    num_faces = face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    face_helper.align_warp_face()

    # 逐个人脸用 CodeFormer 修复
    for cropped_face in face_helper.cropped_faces:
        face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        face_t = face_t.unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                output = net(face_t, w=fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        except Exception:
            restored_face = tensor2img(face_t, rgb2bgr=True, min_max=(-1, 1))
        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face, cropped_face)

    # 背景超分 + 人脸贴回
    if upscale > 1 and bg_upsampler is not None:
        bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
    else:
        bg_img = None
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=False)

    out_file = f"{out_path}.png"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    imwrite(restored_img, out_file)
    print(f"检测到 {num_faces} 张人脸，输出: {out_file}", flush=True)
    return out_file


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="老照片修复（Real-ESRGAN + CodeFormer）")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("--fidelity", type=float, default=0.5, help="保真度 0-1（越小越增强）")
    parser.add_argument("--upscale", type=int, default=2, help="放大倍数（1 或 2）")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    out = run(args.input, {"fidelity": args.fidelity, "upscale": args.upscale}, args.out)
    print(f"完成: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()