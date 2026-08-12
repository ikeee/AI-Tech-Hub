"""图像调整（Image Adjustment）：亮度/对比度/伽马/饱和度/色相/曝光/白平衡/自动增强。

用法：
  python main.py brightness <input> --delta 30 [--output out.png]
  python main.py contrast <input> --factor 1.3 [--output out.png]
  python main.py gamma <input> --gamma 1.2 [--output out.png]
  python main.py saturation <input> --factor 1.5 [--output out.png]
  python main.py hue <input> --shift 60 [--output out.png]
  python main.py exposure <input> --ev 0.5 [--output out.png]
  python main.py white-balance <input> [--temp 20] [--tint -10] [--output out.png]
  python main.py auto-contrast <input> [--low 1 --high 99] [--output out.png]
  python main.py auto-brightness <input> [--output out.png]
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


def cmd_brightness(args: argparse.Namespace) -> None:
    img = load(args.input)
    save_or_print(np.clip(img.astype(np.int16) + args.delta, 0, 255).astype(np.uint8), args)


def cmd_contrast(args: argparse.Namespace) -> None:
    img = load(args.input)
    out = np.clip((img.astype(np.float32) - 128) * args.factor + 128, 0, 255).astype(np.uint8)
    save_or_print(out, args)


def cmd_gamma(args: argparse.Namespace) -> None:
    img = load(args.input)
    inv = 1 / max(0.05, args.gamma)
    lut = (np.power(np.arange(256) / 255, inv) * 255).astype(np.uint8)
    save_or_print(cv2.LUT(img, lut), args)


def cmd_saturation(args: argparse.Namespace) -> None:
    img = load(args.input)
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * args.factor, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    save_or_print(cv2.merge([out[:, :, 0], out[:, :, 1], out[:, :, 2]]), args)


def cmd_hue(args: argparse.Namespace) -> None:
    img = load(args.input)
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + args.shift / 2) % 180
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    save_or_print(out, args)


def cmd_exposure(args: argparse.Namespace) -> None:
    img = load(args.input)
    m = 2 ** args.ev
    save_or_print(np.clip(img.astype(np.float32) * m, 0, 255).astype(np.uint8), args)


def cmd_white_balance(args: argparse.Namespace) -> None:
    img = load(args.input).astype(np.float32)
    img[:, :, 2] *= 1 + args.temp / 200   # R
    img[:, :, 0] *= 1 - args.temp / 200   # B
    img[:, :, 1] *= 1 - args.tint / 200   # G
    save_or_print(np.clip(img, 0, 255).astype(np.uint8), args)


def cmd_auto_contrast(args: argparse.Namespace) -> None:
    img = load(args.input)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).ravel()
    lo = int(np.percentile(gray, args.low))
    hi = int(np.percentile(gray, args.high))
    if hi - lo < 1:
        save_or_print(img, args)
        return
    alpha = 255 / (hi - lo)
    out = np.clip((img.astype(np.float32) - lo) * alpha, 0, 255).astype(np.uint8)
    save_or_print(out, args)


def cmd_auto_brightness(args: argparse.Namespace) -> None:
    img = load(args.input)
    mean = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).mean()
    out = np.clip(img.astype(np.float32) + (128 - mean), 0, 255).astype(np.uint8)
    save_or_print(out, args)


def main() -> None:
    p = argparse.ArgumentParser(description="图像调整参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    p_b = sub.add_parser("brightness")
    p_b.add_argument("input")
    p_b.add_argument("--delta", type=float, default=30)
    p_b.add_argument("--output")

    p_c = sub.add_parser("contrast")
    p_c.add_argument("input")
    p_c.add_argument("--factor", type=float, default=1.3)
    p_c.add_argument("--output")

    p_g = sub.add_parser("gamma")
    p_g.add_argument("input")
    p_g.add_argument("--gamma", type=float, default=1.2)
    p_g.add_argument("--output")

    p_s = sub.add_parser("saturation")
    p_s.add_argument("input")
    p_s.add_argument("--factor", type=float, default=1.5)
    p_s.add_argument("--output")

    p_h = sub.add_parser("hue")
    p_h.add_argument("input")
    p_h.add_argument("--shift", type=float, default=60)
    p_h.add_argument("--output")

    p_e = sub.add_parser("exposure")
    p_e.add_argument("input")
    p_e.add_argument("--ev", type=float, default=0.5)
    p_e.add_argument("--output")

    p_w = sub.add_parser("white-balance")
    p_w.add_argument("input")
    p_w.add_argument("--temp", type=float, default=0)
    p_w.add_argument("--tint", type=float, default=0)
    p_w.add_argument("--output")

    p_ac = sub.add_parser("auto-contrast")
    p_ac.add_argument("input")
    p_ac.add_argument("--low", type=float, default=1)
    p_ac.add_argument("--high", type=float, default=99)
    p_ac.add_argument("--output")

    p_ab = sub.add_parser("auto-brightness")
    p_ab.add_argument("input")
    p_ab.add_argument("--output")

    args = p.parse_args()
    {
        "brightness": cmd_brightness,
        "contrast": cmd_contrast,
        "gamma": cmd_gamma,
        "saturation": cmd_saturation,
        "hue": cmd_hue,
        "exposure": cmd_exposure,
        "white-balance": cmd_white_balance,
        "auto-contrast": cmd_auto_contrast,
        "auto-brightness": cmd_auto_brightness,
    }[args.tool](args)


if __name__ == "__main__":
    main()
