"""边缘与形状检测（Edge & Shape Detection）。

用法：
  python main.py sobel <input> [--ksize 3] [--output out.png]
  python main.py scharr <input> [--output out.png]
  python main.py laplacian <input> [--output out.png]
  python main.py canny <input> [--t1 100 --t2 200] [--output out.png]
  python main.py harris <input> [--thresh 0.01] [--output out.png]
  python main.py hough-lines <input> [--threshold 80] [--output out.png]
  python main.py hough-circles <input> [--param2 100] [--output out.png]
  python main.py polygon <input> [--epsilon 2] [--output out.png]
"""

import argparse

import cv2
import numpy as np


def load(path: str):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    return img


def save_or_print(img, args: argparse.Namespace) -> None:
    if args.output:
        cv2.imwrite(args.output, img)
        print(f"已保存: {args.output}")
    else:
        print(f"result: {img.shape[1]}x{img.shape[0]}")


def cmd_sobel(args):
    img = load(args.input)
    dx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=args.ksize)
    dy = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=args.ksize)
    save_or_print(cv2.addWeighted(dx, 0.5, dy, 0.5, 0), args)


def cmd_scharr(args):
    img = load(args.input)
    dx = cv2.Scharr(img, cv2.CV_8U, 1, 0)
    dy = cv2.Scharr(img, cv2.CV_8U, 0, 1)
    save_or_print(cv2.addWeighted(dx, 0.5, dy, 0.5, 0), args)


def cmd_laplacian(args):
    img = load(args.input)
    save_or_print(cv2.Laplacian(img, cv2.CV_8U, 3), args)


def cmd_canny(args):
    img = load(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_or_print(cv2.Canny(gray, args.t1, args.t2), args)


def cmd_harris(args):
    img = load(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    thresh = dst.max() * args.thresh
    out = img.copy()
    out[dst > thresh] = (0, 0, 255)
    save_or_print(out, args)


def cmd_hough_lines(args):
    img = load(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, args.threshold, minLineLength=30, maxLineGap=10)
    out = img.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    save_or_print(out, args)


def cmd_hough_circles(args):
    img = load(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, max(20, gray.shape[0] // 8), param1=200, param2=args.param2, minRadius=10)
    out = img.copy()
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            cv2.circle(out, (x, y), r, (0, 255, 0), 2)
            cv2.circle(out, (x, y), 2, (0, 0, 255), -1)
    save_or_print(out, args)


def cmd_polygon(args):
    img = load(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img.copy()
    count = 0
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        approx = cv2.approxPolyDP(c, cv2.arcLength(c, True) * args.epsilon / 100, True)
        cv2.polylines(out, [approx], True, (0, 255, 0), 2)
        count += 1
    print(f"polygons={count}")
    save_or_print(out, args)


def main() -> None:
    p = argparse.ArgumentParser(description="边缘与形状检测参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    p_s = sub.add_parser("sobel")
    p_s.add_argument("input")
    p_s.add_argument("--ksize", type=int, default=3, choices=[3, 5, 7])
    p_s.add_argument("--output")

    p_sc = sub.add_parser("scharr")
    p_sc.add_argument("input")
    p_sc.add_argument("--output")

    p_l = sub.add_parser("laplacian")
    p_l.add_argument("input")
    p_l.add_argument("--output")

    p_c = sub.add_parser("canny")
    p_c.add_argument("input")
    p_c.add_argument("--t1", type=int, default=100)
    p_c.add_argument("--t2", type=int, default=200)
    p_c.add_argument("--output")

    p_h = sub.add_parser("harris")
    p_h.add_argument("input")
    p_h.add_argument("--thresh", type=float, default=0.01)
    p_h.add_argument("--output")

    p_hl = sub.add_parser("hough-lines")
    p_hl.add_argument("input")
    p_hl.add_argument("--threshold", type=int, default=80)
    p_hl.add_argument("--output")

    p_hc = sub.add_parser("hough-circles")
    p_hc.add_argument("input")
    p_hc.add_argument("--param2", type=int, default=100)
    p_hc.add_argument("--output")

    p_p = sub.add_parser("polygon")
    p_p.add_argument("input")
    p_p.add_argument("--epsilon", type=float, default=2)
    p_p.add_argument("--output")

    args = p.parse_args()
    {
        "sobel": cmd_sobel,
        "scharr": cmd_scharr,
        "laplacian": cmd_laplacian,
        "canny": cmd_canny,
        "harris": cmd_harris,
        "hough-lines": cmd_hough_lines,
        "hough-circles": cmd_hough_circles,
        "polygon": cmd_polygon,
    }[args.tool](args)


if __name__ == "__main__":
    main()
