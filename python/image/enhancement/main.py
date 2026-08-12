"""噪声与增强（Noise & Enhancement）：加噪/去噪/直方图/均衡化/增强/超分。

用法：
  python main.py add-noise <input> [--type gaussian|salt-pepper] [--sigma 25] [--amount 5] [--output out.png]
  python main.py denoise <input> [--method median|gaussian] [--size 3] [--output out.png]
  python main.py histogram <input> [--output hist.png]
  python main.py histogram-equalize <input> [--output out.png]
  python main.py enhance <input> [--output out.png]
  python main.py super-res <input> --scale 2 [--amount 0.8] [--output out.png]
"""

import argparse

import cv2
import numpy as np


def load(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    return img


def save_or_print(img, args: argparse.Namespace) -> None:
    if args.output:
        cv2.imwrite(args.output, img)
        print(f"已保存: {args.output}")
    else:
        print(f"result: {img.shape[1]}x{img.shape[0]}")


def cmd_add_noise(args):
    img = load(args.input)
    noise = np.zeros(img.shape[:2], np.float32)
    cv2.randn(noise, 0, args.sigma)
    out = np.clip(img.astype(np.float32) + noise[..., None] if img.ndim == 3 else img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if args.type == "salt-pepper":
        out = img.copy()
        n = int(img.shape[0] * img.shape[1] * args.amount / 100)
        for _ in range(n):
            y, x = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
            v = 0 if np.random.rand() < 0.5 else 255
            out[y, x] = v
    save_or_print(out, args)


def cmd_denoise(args):
    img = load(args.input)
    if args.method == "median":
        out = cv2.medianBlur(img, args.size)
    else:
        out = cv2.GaussianBlur(img, (args.size * 2 + 1,) * 2, 0)
    save_or_print(out, args)


def cmd_histogram(args):
    img = load(args.input)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    h = np.zeros((200, 256, 3), np.uint8)
    for i in range(256):
        cv2.line(h, (i, 200), (i, 200 - int(hist[i])), (96, 165, 250), 1)
    save_or_print(h, args)


def cmd_histogram_equalize(args):
    img = load(args.input)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    eq = cv2.equalizeHist(gray)
    if img.ndim == 3:
        # 保持颜色比例：对亮度做映射
        gray_f = gray.astype(np.float32)
        eq_f = eq.astype(np.float32)
        scale = np.divide(eq_f, gray_f + 1e-6, out=np.ones_like(gray_f, np.float32), where=gray_f > 0)
        out = np.clip(img[:, :, :3].astype(np.float32) * scale[..., None], 0, 255).astype(np.uint8)
        if img.shape[2] == 4:
            out = cv2.merge([out[:, :, 0], out[:, :, 1], out[:, :, 2], img[:, :, 3]])
    else:
        out = eq
    save_or_print(out, args)


def cmd_enhance(args):
    img = load(args.input)
    mean = img.astype(np.float32).mean()
    out = np.clip(img.astype(np.float32) + (128 - mean), 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
    lo, hi = np.percentile(gray, [1, 99])
    if hi - lo > 1:
        alpha = 255 / (hi - lo)
        out = np.clip((out.astype(np.float32) - lo) * alpha, 0, 255).astype(np.uint8)
    save_or_print(out, args)


def cmd_super_res(args):
    img = load(args.input)
    h, w = img.shape[:2]
    up = cv2.resize(img, (w * args.scale, h * args.scale), interpolation=cv2.INTER_LANCZOS4)
    if args.amount > 0:
        blur = cv2.GaussianBlur(up, (5, 5), 0)
        up = np.clip(up.astype(np.float32) + args.amount * (up.astype(np.float32) - blur.astype(np.float32)), 0, 255).astype(np.uint8)
    save_or_print(up, args)


def main() -> None:
    p = argparse.ArgumentParser(description="噪声与增强参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    p_n = sub.add_parser("add-noise")
    p_n.add_argument("input")
    p_n.add_argument("--type", default="gaussian", choices=["gaussian", "salt-pepper"])
    p_n.add_argument("--sigma", type=float, default=25)
    p_n.add_argument("--amount", type=float, default=5)
    p_n.add_argument("--output")

    p_d = sub.add_parser("denoise")
    p_d.add_argument("input")
    p_d.add_argument("--method", default="median", choices=["median", "gaussian"])
    p_d.add_argument("--size", type=int, default=3)
    p_d.add_argument("--output")

    p_h = sub.add_parser("histogram")
    p_h.add_argument("input")
    p_h.add_argument("--output")

    p_e = sub.add_parser("histogram-equalize")
    p_e.add_argument("input")
    p_e.add_argument("--output")

    p_en = sub.add_parser("enhance")
    p_en.add_argument("input")
    p_en.add_argument("--output")

    p_sr = sub.add_parser("super-res")
    p_sr.add_argument("input")
    p_sr.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p_sr.add_argument("--amount", type=float, default=0.8)
    p_sr.add_argument("--output")

    args = p.parse_args()
    {
        "add-noise": cmd_add_noise,
        "denoise": cmd_denoise,
        "histogram": cmd_histogram,
        "histogram-equalize": cmd_histogram_equalize,
        "enhance": cmd_enhance,
        "super-res": cmd_super_res,
    }[args.tool](args)


if __name__ == "__main__":
    main()
