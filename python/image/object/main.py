"""颜色与物体检测（Color & Object Detection）。

用法：
  python main.py color-mask <input> [--h-min 0 --h-max 179 --s-min 60 --s-max 255 --v-min 60 --v-max 255] [--output out.png]
  python main.py color-segment <input> [同上] [--output out.png]
  python main.py contour-detect <input> [--thresh 128] [--output out.png]
  python main.py object-count <input> [--thresh 128 --min-area 100] [--output out.png]
  python main.py bounding-box <input> [--thresh 128 --min-area 100] [--output out.png]
  python main.py centroid <input> [--thresh 128 --min-area 100] [--output out.png]
  python main.py area-perimeter <input> [--thresh 128] [--output out.png]
  python main.py shape-recognize <input> [--thresh 128 --epsilon 2] [--output out.png]
"""

import argparse

import cv2


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


def hsv_mask(args, img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low = (args.h_min, args.s_min, args.v_min)
    high = (args.h_max, args.s_max, args.v_max)
    return cv2.inRange(hsv, low, high)


def find_contours(args, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, args.thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def cmd_color_mask(args):
    img = load(args.input)
    save_or_print(hsv_mask(args, img), args)


def cmd_color_segment(args):
    img = load(args.input)
    out = cv2.bitwise_and(img, img, mask=hsv_mask(args, img))
    save_or_print(out, args)


def cmd_contour_detect(args):
    img = load(args.input)
    out = img.copy()
    count = 0
    for c in find_contours(args, img):
        if cv2.contourArea(c) >= 30:
            cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
            count += 1
    print(f"contours={count}")
    save_or_print(out, args)


def cmd_object_count(args):
    img = load(args.input)
    count = sum(1 for c in find_contours(args, img) if cv2.contourArea(c) >= args.min_area)
    out = img.copy()
    cv2.putText(out, f"Count: {count}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(f"objects={count}")
    save_or_print(out, args)


def cmd_bounding_box(args):
    img = load(args.input)
    out = img.copy()
    count = 0
    for c in find_contours(args, img):
        if cv2.contourArea(c) < args.min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
    print(f"objects={count}")
    save_or_print(out, args)


def cmd_centroid(args):
    img = load(args.input)
    out = img.copy()
    count = 0
    for c in find_contours(args, img):
        if cv2.contourArea(c) < args.min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
        count += 1
    print(f"objects={count}")
    save_or_print(out, args)


def cmd_area_perimeter(args):
    img = load(args.input)
    out = img.copy()
    shown = 0
    for i, c in enumerate(find_contours(args, img)):
        area = cv2.contourArea(c)
        if area < 30:
            continue
        perim = cv2.arcLength(c, True)
        cv2.drawContours(out, [c], -1, (0, 255, 0), 1)
        if shown < 6:
            print(f"object{i + 1}: area={area:.0f} perimeter={perim:.1f}")
            shown += 1
    save_or_print(out, args)


def cmd_shape_recognize(args):
    img = load(args.input)
    out = img.copy()
    shown = 0
    for i, c in enumerate(find_contours(args, img)):
        if cv2.contourArea(c) < 100:
            continue
        approx = cv2.approxPolyDP(c, cv2.arcLength(c, True) * args.epsilon / 100, True)
        n = len(approx)
        shape = {3: "Triangle", 4: "Quad", 5: "Pentagon"}.get(n, "Polygon/Circle")
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(out, [c], -1, (0, 255, 0), 2)
        cv2.putText(out, shape, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if shown < 6:
            print(f"object{i + 1}: {shape}")
            shown += 1
    save_or_print(out, args)


def main() -> None:
    p = argparse.ArgumentParser(description="颜色与物体检测参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    for name in ("color-mask", "color-segment"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--h-min", type=int, default=0)
        sp.add_argument("--h-max", type=int, default=179)
        sp.add_argument("--s-min", type=int, default=60)
        sp.add_argument("--s-max", type=int, default=255)
        sp.add_argument("--v-min", type=int, default=60)
        sp.add_argument("--v-max", type=int, default=255)
        sp.add_argument("--output")

    for name in ("contour-detect", "area-perimeter"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--thresh", type=int, default=128)
        sp.add_argument("--output")

    for name in ("object-count", "bounding-box", "centroid"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--thresh", type=int, default=128)
        sp.add_argument("--min-area", type=int, default=100)
        sp.add_argument("--output")

    p_s = sub.add_parser("shape-recognize")
    p_s.add_argument("input")
    p_s.add_argument("--thresh", type=int, default=128)
    p_s.add_argument("--epsilon", type=float, default=2)
    p_s.add_argument("--output")

    args = p.parse_args()
    {
        "color-mask": cmd_color_mask,
        "color-segment": cmd_color_segment,
        "contour-detect": cmd_contour_detect,
        "object-count": cmd_object_count,
        "bounding-box": cmd_bounding_box,
        "centroid": cmd_centroid,
        "area-perimeter": cmd_area_perimeter,
        "shape-recognize": cmd_shape_recognize,
    }[args.tool](args)


if __name__ == "__main__":
    main()
