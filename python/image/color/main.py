"""颜色处理（Color Processing）：灰度/通道/色彩空间/颜色替换/量化。

用法：
  python main.py grayscale <input> [--method average|luminance|desaturate] [--output out.png]
  python main.py channel-extract <input> --channel r|g|b|a|h|s|v|l|la|lb [--output out.png]
  python main.py channel-merge <input> [--r-src r] [--g-src g] [--b-src b] [--output out.png]
  python main.py color-replace <input> --target FF0000 --tolerance 60 --replacement 0000FF [--output out.png]
  python main.py color-quantize <input> --k 8 [--output out.png]
  python main.py color-space-info <input> --x 10 --y 10
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


def bgr_to_hex(b, g, r):
    return f"#{r:02x}{g:02x}{b:02x}"


def cmd_grayscale(args: argparse.Namespace) -> None:
    img = load(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if args.method == "average":
        gray = np.mean(img[:, :, :3], axis=2).astype(np.uint8)
    elif args.method == "desaturate":
        gray = ((img[:, :, :3].max(axis=2) + img[:, :, :3].min(axis=2)) / 2).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        out = cv2.merge([gray, gray, gray, img[:, :, 3]])
    else:
        out = cv2.merge([gray, gray, gray])
    save_or_print(out, args)


def cmd_channel_extract(args: argparse.Namespace) -> None:
    img = load(args.input)
    b, g, r = cv2.split(img[:, :, :3])
    ch = args.channel
    if ch == "r":
        v = r
    elif ch == "g":
        v = g
    elif ch == "b":
        v = b
    elif ch == "a":
        v = img[:, :, 3] if img.ndim == 3 and img.shape[2] == 4 else np.full(img.shape[:2], 255, np.uint8)
    else:
        hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
        hsl = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2LAB)
        v = {
            "h": hsv[:, :, 0] * 255 // 180,
            "s": hsv[:, :, 1],
            "v": hsv[:, :, 2],
            "l": hsl[:, :, 1],
            "la": lab[:, :, 0],
            "lb": lab[:, :, 2],
        }[ch]
    out = cv2.merge([v, v, v])
    save_or_print(out, args)


def cmd_channel_merge(args: argparse.Namespace) -> None:
    img = load(args.input)
    b, g, r = cv2.split(img[:, :, :3])
    a = img[:, :, 3] if img.ndim == 3 and img.shape[2] == 4 else np.full(img.shape[:2], 255, np.uint8)
    hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HSV)
    hsl = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2LAB)
    table = {
        "r": r, "g": g, "b": b, "a": a,
        "h": (hsv[:, :, 0] * 255 // 180).astype(np.uint8),
        "s": hsv[:, :, 1], "v": hsv[:, :, 2], "l": hsl[:, :, 1],
        "la": lab[:, :, 0], "lb": lab[:, :, 2],
    }
    out = cv2.merge([table[args.b_src], table[args.g_src], table[args.r_src]])
    save_or_print(out, args)


def cmd_color_replace(args: argparse.Namespace) -> None:
    img = load(args.input)
    t = tuple(int(args.target[i:i + 2], 16) for i in (0, 2, 4))  # (r,g,b)
    rep = tuple(int(args.replacement[i:i + 2], 16) for i in (0, 2, 4))
    bgr_t = (t[2], t[1], t[0])
    bgr_r = (rep[2], rep[1], rep[0])
    dist = np.sqrt(np.sum((img[:, :, :3].astype(np.int16) - np.array(bgr_t)) ** 2, axis=2))
    mask = dist <= args.tolerance
    out = img.copy()
    out[:, :, :3][mask] = bgr_r
    save_or_print(out, args)


def cmd_color_quantize(args: argparse.Namespace) -> None:
    img = load(args.input)
    pixels = img[:, :, :3].reshape(-1, 3).astype(np.float32)
    k = max(2, min(32, args.k))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    quantized = centers[labels.flatten()].reshape(img.shape[:2] + (3,)).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        out = cv2.merge([quantized[:, :, 0], quantized[:, :, 1], quantized[:, :, 2], img[:, :, 3]])
    else:
        out = quantized
    save_or_print(out, args)


def cmd_color_space_info(args: argparse.Namespace) -> None:
    img = load(args.input)
    h, w = img.shape[:2]
    x, y = min(args.x, w - 1), min(args.y, h - 1)
    b, g, r = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
    hsv = cv2.cvtColor(img[y:y + 1, x:x + 1, :3], cv2.COLOR_BGR2HSV)[0, 0]
    hsl = cv2.cvtColor(img[y:y + 1, x:x + 1, :3], cv2.COLOR_BGR2HLS)[0, 0]
    lab = cv2.cvtColor(img[y:y + 1, x:x + 1, :3], cv2.COLOR_BGR2LAB)[0, 0]
    print(f"({x},{y}) RGB=({r},{g},{b}) HEX={bgr_to_hex(b, g, r)} "
          f"HSV=({hsv[0] * 2},{hsv[1]},{hsv[2]}) HSL=({hsl[0] * 2},{hsl[1]},{hsl[2]}) "
          f"Lab=({lab[0]},{lab[1]},{lab[2]})")


def main() -> None:
    p = argparse.ArgumentParser(description="颜色处理参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    p_gray = sub.add_parser("grayscale")
    p_gray.add_argument("input")
    p_gray.add_argument("--method", default="luminance", choices=["average", "luminance", "desaturate"])
    p_gray.add_argument("--output")

    p_ext = sub.add_parser("channel-extract")
    p_ext.add_argument("input")
    p_ext.add_argument("--channel", required=True, choices=["r", "g", "b", "a", "h", "s", "v", "l", "la", "lb"])
    p_ext.add_argument("--output")

    p_merge = sub.add_parser("channel-merge")
    p_merge.add_argument("input")
    p_merge.add_argument("--r-src", default="r", choices=["r", "g", "b", "a", "h", "s", "v", "l", "la", "lb"])
    p_merge.add_argument("--g-src", default="g", choices=["r", "g", "b", "a", "h", "s", "v", "l", "la", "lb"])
    p_merge.add_argument("--b-src", default="b", choices=["r", "g", "b", "a", "h", "s", "v", "l", "la", "lb"])
    p_merge.add_argument("--output")

    p_rep = sub.add_parser("color-replace")
    p_rep.add_argument("input")
    p_rep.add_argument("--target", required=True, help="RRGGBB")
    p_rep.add_argument("--tolerance", type=float, default=60)
    p_rep.add_argument("--replacement", required=True, help="RRGGBB")
    p_rep.add_argument("--output")

    p_q = sub.add_parser("color-quantize")
    p_q.add_argument("input")
    p_q.add_argument("--k", type=int, default=8)
    p_q.add_argument("--output")

    p_info = sub.add_parser("color-space-info")
    p_info.add_argument("input")
    p_info.add_argument("--x", type=int, default=0)
    p_info.add_argument("--y", type=int, default=0)

    args = p.parse_args()
    {
        "grayscale": cmd_grayscale,
        "channel-extract": cmd_channel_extract,
        "channel-merge": cmd_channel_merge,
        "color-replace": cmd_color_replace,
        "color-quantize": cmd_color_quantize,
        "color-space-info": cmd_color_space_info,
    }[args.tool](args)


if __name__ == "__main__":
    main()
