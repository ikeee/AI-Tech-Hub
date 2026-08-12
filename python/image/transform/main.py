"""图像变换（Image Transform）：缩放/裁剪/旋转/翻转/比例/边距/透视/仿射。

用法：
  python main.py resize <input> --width 800 --height 600 [--keep] [--output out.png]
  python main.py crop <input> --x 0 --y 0 --w 300 --h 200 [--output out.png]
  python main.py rotate <input> --angle 90 [--bg black|white|transparent] [--output out.png]
  python main.py flip <input> --dir horizontal|vertical|both [--output out.png]
  python main.py scale <input> --factor 1.5 [--output out.png]
  python main.py pad <input> --top 20 --right 20 --bottom 20 --left 20 [--color white] [--output out.png]
  python main.py perspective <input> [--top-inset 0.1] ... [--output out.png]
  python main.py affine <input> [--rotate-deg 15] ... [--output out.png]
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
        print(f"已保存: {args.output} ({img.shape[1]}x{img.shape[0]})")
    else:
        print(f"result: {img.shape[1]}x{img.shape[0]}")


def cmd_resize(args: argparse.Namespace) -> None:
    img = load(args.input)
    w = args.width
    h = args.height
    if args.keep:
        h = round(img.shape[0] * w / img.shape[1])
    save_or_print(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), args)


def cmd_crop(args: argparse.Namespace) -> None:
    img = load(args.input)
    save_or_print(img[args.y:args.y + args.h, args.x:args.x + args.w], args)


def cmd_rotate(args: argparse.Namespace) -> None:
    img = load(args.input)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), args.angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = cv2.BORDER_CONSTANT
    value = (0, 0, 0) if args.bg == "black" else (255, 255, 255)
    if img.ndim == 3 and img.shape[2] == 4:
        value = tuple(list(value) + [0])
    out = cv2.warpAffine(img, m, (nw, nh), borderMode=border, borderValue=value)
    save_or_print(out, args)


def cmd_flip(args: argparse.Namespace) -> None:
    img = load(args.input)
    code = {"horizontal": 1, "vertical": 0, "both": -1}[args.dir]
    save_or_print(cv2.flip(img, code), args)


def cmd_scale(args: argparse.Namespace) -> None:
    img = load(args.input)
    save_or_print(cv2.resize(img, None, fx=args.factor, fy=args.factor, interpolation=cv2.INTER_LINEAR), args)


def cmd_pad(args: argparse.Namespace) -> None:
    img = load(args.input)
    color = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}
    value = color.get(args.color, (255, 255, 255))
    if img.ndim == 3 and img.shape[2] == 4 and args.color == "transparent":
        value = (0, 0, 0, 0)
    out = cv2.copyMakeBorder(img, args.top, args.bottom, args.left, args.right, cv2.BORDER_CONSTANT, value=value)
    save_or_print(out, args)


def cmd_perspective(args: argparse.Namespace) -> None:
    img = load(args.input)
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * args.top_inset, h * args.left_inset],
        [w * (1 - args.top_inset), h * args.left_inset],
        [w * (1 - args.bottom_inset), h * (1 - args.right_inset)],
        [w * args.bottom_inset, h * (1 - args.right_inset)],
    ])
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    save_or_print(out, args)


def cmd_affine(args: argparse.Namespace) -> None:
    img = load(args.input)
    h, w = img.shape[:2]
    rad = np.deg2rad(args.rotate_deg)
    m = cv2.getRotationMatrix2D((0, 0), 0, 1.0)
    # 旋转 + 缩放 + 错切
    a = np.cos(rad) * args.scale_x + np.sin(rad) * args.shear_y
    b = -np.sin(rad) * args.scale_x + np.cos(rad) * args.shear_y
    c = np.sin(rad) * args.scale_y + np.cos(rad) * args.shear_x
    d = np.cos(rad) * args.scale_y - np.sin(rad) * args.shear_x
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = a, c, b, d
    m[0, 2] = args.tx * w
    m[1, 2] = args.ty * h
    out = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    save_or_print(out, args)


def main() -> None:
    p = argparse.ArgumentParser(description="图像变换参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    p_resize = sub.add_parser("resize")
    p_resize.add_argument("input")
    p_resize.add_argument("--width", type=int, default=800)
    p_resize.add_argument("--height", type=int, default=600)
    p_resize.add_argument("--keep", action="store_true")
    p_resize.add_argument("--output")

    p_crop = sub.add_parser("crop")
    p_crop.add_argument("input")
    p_crop.add_argument("--x", type=int, default=0)
    p_crop.add_argument("--y", type=int, default=0)
    p_crop.add_argument("--w", type=int, required=True)
    p_crop.add_argument("--h", type=int, required=True)
    p_crop.add_argument("--output")

    p_rotate = sub.add_parser("rotate")
    p_rotate.add_argument("input")
    p_rotate.add_argument("--angle", type=float, default=90)
    p_rotate.add_argument("--bg", default="transparent", choices=["black", "white", "transparent"])
    p_rotate.add_argument("--output")

    p_flip = sub.add_parser("flip")
    p_flip.add_argument("input")
    p_flip.add_argument("--dir", default="horizontal", choices=["horizontal", "vertical", "both"])
    p_flip.add_argument("--output")

    p_scale = sub.add_parser("scale")
    p_scale.add_argument("input")
    p_scale.add_argument("--factor", type=float, default=1.5)
    p_scale.add_argument("--output")

    p_pad = sub.add_parser("pad")
    p_pad.add_argument("input")
    p_pad.add_argument("--top", type=int, default=20)
    p_pad.add_argument("--right", type=int, default=20)
    p_pad.add_argument("--bottom", type=int, default=20)
    p_pad.add_argument("--left", type=int, default=20)
    p_pad.add_argument("--color", default="white", choices=["black", "white", "gray", "transparent"])
    p_pad.add_argument("--output")

    p_persp = sub.add_parser("perspective")
    p_persp.add_argument("input")
    p_persp.add_argument("--top-inset", type=float, default=0.0)
    p_persp.add_argument("--bottom-inset", type=float, default=0.0)
    p_persp.add_argument("--left-inset", type=float, default=0.0)
    p_persp.add_argument("--right-inset", type=float, default=0.0)
    p_persp.add_argument("--output")

    p_affine = sub.add_parser("affine")
    p_affine.add_argument("input")
    p_affine.add_argument("--rotate-deg", type=float, default=0)
    p_affine.add_argument("--scale-x", type=float, default=1)
    p_affine.add_argument("--scale-y", type=float, default=1)
    p_affine.add_argument("--shear-x", type=float, default=0)
    p_affine.add_argument("--shear-y", type=float, default=0)
    p_affine.add_argument("--tx", type=float, default=0)
    p_affine.add_argument("--ty", type=float, default=0)
    p_affine.add_argument("--output")

    args = p.parse_args()
    {
        "resize": cmd_resize,
        "crop": cmd_crop,
        "rotate": cmd_rotate,
        "flip": cmd_flip,
        "scale": cmd_scale,
        "pad": cmd_pad,
        "perspective": cmd_perspective,
        "affine": cmd_affine,
    }[args.tool](args)


if __name__ == "__main__":
    main()
