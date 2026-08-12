"""图像查看器（Image Viewer）：图片信息 / 像素取色。

用法：
  python main.py info <input>
  python main.py pixel <input> --x 120 --y 80
"""

import argparse

import cv2


def load(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    return img


def cmd_info(args: argparse.Namespace) -> None:
    img = load(args.input)
    h, w = img.shape[:2]
    if img.ndim == 2:
        mode = "GRAY"
    elif img.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    print(f"width={w}px  height={h}px  mode={mode}  bytes={img.nbytes}")


def cmd_pixel(args: argparse.Namespace) -> None:
    img = load(args.input)
    h, w = img.shape[:2]
    x, y = min(args.x, w - 1), min(args.y, h - 1)
    # OpenCV 是 BGR 顺序
    b, g, r = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])
    a = int(img[y, x, 3]) if img.ndim == 3 and img.shape[2] == 4 else 255
    print(f"({x},{y})  RGBA=({r},{g},{b},{a})  HEX=#{r:02x}{g:02x}{b:02x}")


def main() -> None:
    p = argparse.ArgumentParser(description="图像查看器参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    sub.add_parser("info", help="图片信息").add_argument("input", help="输入图片路径")

    p_pixel = sub.add_parser("pixel", help="读取像素")
    p_pixel.add_argument("input")
    p_pixel.add_argument("--x", type=int, default=0)
    p_pixel.add_argument("--y", type=int, default=0)

    args = p.parse_args()
    if args.tool == "info":
        cmd_info(args)
    elif args.tool == "pixel":
        cmd_pixel(args)


if __name__ == "__main__":
    main()
