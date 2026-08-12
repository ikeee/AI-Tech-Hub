"""阈值与形态学（Threshold & Morphology）：二值化/自适应/Otsu/腐蚀膨胀/开闭/梯度。

用法：
  python main.py binary-threshold <input> --thresh 128 [--output out.png]
  python main.py adaptive-threshold <input> [--block 15 --c 10 --method mean|gaussian] [--output out.png]
  python main.py otsu-threshold <input> [--output out.png]
  python main.py erode <input> --size 3 [--iter 1] [--output out.png]
  python main.py dilate <input> --size 3 [--iter 1] [--output out.png]
  python main.py opening <input> --size 3 [--output out.png]
  python main.py closing <input> --size 3 [--output out.png]
  python main.py morph-gradient <input> --size 3 [--output out.png]
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


def to_gray3(img):
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.merge([gray, gray, gray])


def cmd_binary_threshold(args):
    img = load(args.input)
    _, out = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img, args.thresh, 255, cv2.THRESH_BINARY)
    save_or_print(cv2.merge([out, out, out]), args)


def cmd_adaptive_threshold(args):
    img = load(args.input)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if args.method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    out = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, args.block, args.c)
    save_or_print(cv2.merge([out, out, out]), args)


def cmd_otsu_threshold(args):
    img = load(args.input)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    t, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu threshold = {t}")
    save_or_print(cv2.merge([out, out, out]), args)


def kernel(args):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (args.size, args.size))


def cmd_erode(args):
    img = load(args.input)
    out = cv2.erode(img, kernel(args), iterations=args.iter)
    save_or_print(out, args)


def cmd_dilate(args):
    img = load(args.input)
    out = cv2.dilate(img, kernel(args), iterations=args.iter)
    save_or_print(out, args)


def cmd_opening(args):
    img = load(args.input)
    save_or_print(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel(args)), args)


def cmd_closing(args):
    img = load(args.input)
    save_or_print(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel(args)), args)


def cmd_morph_gradient(args):
    img = load(args.input)
    save_or_print(cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel(args)), args)


def main() -> None:
    p = argparse.ArgumentParser(description="阈值与形态学参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    p_b = sub.add_parser("binary-threshold")
    p_b.add_argument("input")
    p_b.add_argument("--thresh", type=int, default=128)
    p_b.add_argument("--output")

    p_a = sub.add_parser("adaptive-threshold")
    p_a.add_argument("input")
    p_a.add_argument("--block", type=int, default=15)
    p_a.add_argument("--c", type=float, default=10)
    p_a.add_argument("--method", default="mean", choices=["mean", "gaussian"])
    p_a.add_argument("--output")

    p_o = sub.add_parser("otsu-threshold")
    p_o.add_argument("input")
    p_o.add_argument("--output")

    for name in ("erode", "dilate"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--size", type=int, default=3)
        sp.add_argument("--iter", type=int, default=1)
        sp.add_argument("--output")

    for name in ("opening", "closing", "morph-gradient"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--size", type=int, default=3)
        sp.add_argument("--output")

    args = p.parse_args()
    {
        "binary-threshold": cmd_binary_threshold,
        "adaptive-threshold": cmd_adaptive_threshold,
        "otsu-threshold": cmd_otsu_threshold,
        "erode": cmd_erode,
        "dilate": cmd_dilate,
        "opening": cmd_opening,
        "closing": cmd_closing,
        "morph-gradient": cmd_morph_gradient,
    }[args.tool](args)


if __name__ == "__main__":
    main()
