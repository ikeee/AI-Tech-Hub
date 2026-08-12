"""像素处理（Pixel Processing）：读取像素 / 像素网格 / 像素运算。

用法：
  python main.py read-pixel <input> --x 10 --y 10
  python main.py pixel-grid <input> --zoom 8 [--cx 0.5 --cy 0.5] [--output grid.png]
  python main.py pixel-math <input> --op add|subtract|multiply|divide --value 30 [--output out.png]
"""

import argparse

import cv2
import numpy as np


def load(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    return img


def cmd_read_pixel(args: argparse.Namespace) -> None:
    img = load(args.input)
    h, w = img.shape[:2]
    x, y = min(args.x, w - 1), min(args.y, h - 1)
    b, g, r = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
    a = int(img[y, x, 3]) if img.ndim == 3 and img.shape[2] == 4 else 255
    print(f"({x},{y})  RGBA=({r},{g},{b},{a})  HEX=#{r:02x}{g:02x}{b:02x}")


def cmd_pixel_grid(args: argparse.Namespace) -> None:
    img = load(args.input)
    h, w = img.shape[:2]
    cx = int(args.cx * w)
    cy = int(args.cy * h)
    patch = 12
    cell = max(2, int(args.zoom))
    half = patch // 2
    x0 = max(0, min(w - patch, cx - half))
    y0 = max(0, min(h - patch, cy - half))
    crop = img[y0:y0 + patch, x0:x0 + patch]
    up = cv2.resize(crop, (patch * cell, patch * cell), interpolation=cv2.INTER_NEAREST)
    if not args.no_grid:
        for i in range(patch + 1):
            cv2.line(up, (i * cell, 0), (i * cell, up.shape[0]), (255, 255, 255), 1)
            cv2.line(up, (0, i * cell), (up.shape[1], i * cell), (255, 255, 255), 1)
    if args.output:
        cv2.imwrite(args.output, up)
        print(f"已保存: {args.output}")
    else:
        print(f"pixel grid: center=({cx},{cy}) zoom={cell} size={up.shape[1]}x{up.shape[0]}")


def cmd_pixel_math(args: argparse.Namespace) -> None:
    img = load(args.input).astype(np.float32)
    v = args.value
    if args.op == "add":
        img += v
    elif args.op == "subtract":
        img -= v
    elif args.op == "multiply":
        img *= 1 + v / 100
    elif args.op == "divide":
        d = max(0.05, 1 + v / 100)
        img /= d
    out = np.clip(img, 0, 255).astype(np.uint8)
    if args.output:
        cv2.imwrite(args.output, out)
        print(f"已保存: {args.output}")
    else:
        print(f"pixel-math: op={args.op} value={v}")


def main() -> None:
    p = argparse.ArgumentParser(description="像素处理参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    p_read = sub.add_parser("read-pixel")
    p_read.add_argument("input")
    p_read.add_argument("--x", type=int, default=0)
    p_read.add_argument("--y", type=int, default=0)

    p_grid = sub.add_parser("pixel-grid")
    p_grid.add_argument("input")
    p_grid.add_argument("--zoom", type=int, default=8)
    p_grid.add_argument("--cx", type=float, default=0.5)
    p_grid.add_argument("--cy", type=float, default=0.5)
    p_grid.add_argument("--no-grid", action="store_true")
    p_grid.add_argument("--output")

    p_math = sub.add_parser("pixel-math")
    p_math.add_argument("input")
    p_math.add_argument("--op", default="add", choices=["add", "subtract", "multiply", "divide"])
    p_math.add_argument("--value", type=float, default=30)
    p_math.add_argument("--output")

    args = p.parse_args()
    if args.tool == "read-pixel":
        cmd_read_pixel(args)
    elif args.tool == "pixel-grid":
        cmd_pixel_grid(args)
    elif args.tool == "pixel-math":
        cmd_pixel_math(args)


if __name__ == "__main__":
    main()
