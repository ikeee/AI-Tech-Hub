"""OCR 与文档视觉（OCR & Document Vision）。

用法：
  python main.py ocr-text <input> [--lang chi_sim|eng|chi_sim+eng] [--output text.txt]
  python main.py document-scan <input> [--output scan.png]

依赖：
  pip install opencv-python numpy pytesseract pillow
  并安装 Tesseract OCR 本体与语言包（chi_sim）。
"""

import argparse

import cv2
import numpy as np


def load(path: str):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    return img


def cmd_ocr_text(args):
    import pytesseract
    from PIL import Image
    img = load(args.input)
    text = pytesseract.image_to_string(img, lang=args.lang)
    text = text.strip()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已保存: {args.output}")
    else:
        print(text)


def order_points(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.float32([tl, tr, br, bl])


def cmd_document_scan(args):
    img = load(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < img.shape[0] * img.shape[1] * 0.1:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best = approx
            best_area = area
    if best is None:
        print("no document quad found")
        return
    src = order_points(best)
    w = max(400, int(np.linalg.norm(src[1] - src[0])))
    h = max(400, int(np.linalg.norm(src[3] - src[0])))
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, m, (w, h))
    save_or_print(out, args)


def save_or_print(img, args: argparse.Namespace) -> None:
    if args.output:
        cv2.imwrite(args.output, img)
        print(f"已保存: {args.output}")
    else:
        print(f"result: {img.shape[1]}x{img.shape[0]}")


def main() -> None:
    p = argparse.ArgumentParser(description="OCR 与文档视觉参考实现")
    sub = p.add_subparsers(dest="tool", required=True)
    p_o = sub.add_parser("ocr-text")
    p_o.add_argument("input")
    p_o.add_argument("--lang", default="eng")
    p_o.add_argument("--output")
    p_s = sub.add_parser("document-scan")
    p_s.add_argument("input")
    p_s.add_argument("--output")
    args = p.parse_args()
    if args.tool == "ocr-text":
        cmd_ocr_text(args)
    else:
        cmd_document_scan(args)


if __name__ == "__main__":
    main()
