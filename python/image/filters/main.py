"""图像滤镜（Image Filters）：模糊/锐化/浮雕/高通等卷积滤波。

用法：
  python main.py box-blur <input> --radius 3 [--output out.png]
  python main.py gaussian-blur <input> --radius 3 [--sigma 1.5] [--output out.png]
  python main.py median-blur <input> --size 5 [--output out.png]
  python main.py motion-blur <input> --length 10 --angle 45 [--output out.png]
  python main.py sharpen <input> --amount 1.0 [--output out.png]
  python main.py unsharp-mask <input> --radius 3 --amount 1.0 [--output out.png]
  python main.py emboss <input> [--angle 45] [--output out.png]
  python main.py high-pass <input> [--output out.png]
"""

import argparse
import math

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


def motion_kernel(length: int, angle: float):
    size = length
    k = np.zeros((size, size), np.float32)
    cx = (size - 1) / 2
    rad = math.radians(angle)
    for i in range(length):
        t = i - (length - 1) / 2
        x = int(round(cx + t * math.cos(rad)))
        y = int(round(cx + t * math.sin(rad)))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1
    return k / k.sum()


def cmd_box_blur(args):
    img = load(args.input)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (args.radius * 2 + 1,) * 2)
    out = cv2.filter2D(img, -1, k / k.sum())
    save_or_print(out, args)


def cmd_gaussian_blur(args):
    img = load(args.input)
    ksize = args.radius * 2 + 1
    out = cv2.GaussianBlur(img, (ksize, ksize), args.sigma)
    save_or_print(out, args)


def cmd_median_blur(args):
    img = load(args.input)
    out = cv2.medianBlur(img, args.size)
    save_or_print(out, args)


def cmd_motion_blur(args):
    img = load(args.input)
    out = cv2.filter2D(img, -1, motion_kernel(args.length, args.angle))
    save_or_print(out, args)


def cmd_sharpen(args):
    img = load(args.input)
    lap = cv2.Laplacian(img, cv2.CV_32F)
    out = np.clip(img.astype(np.float32) + args.amount * lap, 0, 255).astype(np.uint8)
    save_or_print(out, args)


def cmd_unsharp_mask(args):
    img = load(args.input)
    ksize = args.radius * 2 + 1
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
    out = np.clip(img.astype(np.float32) + args.amount * (img.astype(np.float32) - blur.astype(np.float32)), 0, 255).astype(np.uint8)
    save_or_print(out, args)


def cmd_emboss(args):
    img = load(args.input)
    rad = math.radians(args.angle)
    k = np.array([
        [round(math.cos(rad) * -1 + math.sin(rad) * -1), round(math.sin(rad) * -1), round(math.cos(rad) * 1 + math.sin(rad) * -1)],
        [round(math.cos(rad) * -1), 0, round(math.cos(rad) * 1)],
        [round(math.cos(rad) * -1 + math.sin(rad) * 1), round(math.sin(rad) * 1), round(math.cos(rad) * 1 + math.sin(rad) * 1)],
    ], np.float32)
    out = cv2.filter2D(img, -1, k, delta=128)
    save_or_print(out, args)


def cmd_high_pass(args):
    img = load(args.input)
    k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
    out = cv2.filter2D(img, -1, k)
    save_or_print(out, args)


def main() -> None:
    p = argparse.ArgumentParser(description="图像滤镜参考实现")
    sub = p.add_subparsers(dest="tool", required=True)
    for name, add in {
        "box-blur": lambda s: (s.add_argument("--radius", type=int, default=3), s.add_argument("--output")),
        "gaussian-blur": lambda s: (s.add_argument("--radius", type=int, default=3), s.add_argument("--sigma", type=float, default=1.5), s.add_argument("--output")),
        "median-blur": lambda s: (s.add_argument("--size", type=int, default=5), s.add_argument("--output")),
        "motion-blur": lambda s: (s.add_argument("--length", type=int, default=10), s.add_argument("--angle", type=float, default=45), s.add_argument("--output")),
        "sharpen": lambda s: (s.add_argument("--amount", type=float, default=1.0), s.add_argument("--output")),
        "unsharp-mask": lambda s: (s.add_argument("--radius", type=int, default=3), s.add_argument("--amount", type=float, default=1.0), s.add_argument("--output")),
        "emboss": lambda s: (s.add_argument("--angle", type=float, default=45), s.add_argument("--output")),
        "high-pass": lambda s: s.add_argument("--output"),
    }.items():
        sp = sub.add_parser(name)
        sp.add_argument("input")
        add(sp)
    args = p.parse_args()
    {
        "box-blur": cmd_box_blur,
        "gaussian-blur": cmd_gaussian_blur,
        "median-blur": cmd_median_blur,
        "motion-blur": cmd_motion_blur,
        "sharpen": cmd_sharpen,
        "unsharp-mask": cmd_unsharp_mask,
        "emboss": cmd_emboss,
        "high-pass": cmd_high_pass,
    }[args.tool](args)


if __name__ == "__main__":
    main()
