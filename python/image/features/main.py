"""特征检测（Feature Detection）：ORB/BRISK 关键点与特征匹配（SIFT 在 Python 侧可用）。

用法：
  python main.py orb-keypoints <input> [--max 200] [--output out.png]
  python main.py brisk-keypoints <input> [--output out.png]
  python main.py feature-match <input> --second <img2> [--max 500 --ratio 0.75] [--output out.png]
  python main.py sift-keypoints <input> [--output out.png]   # 需要 opencv-contrib-python
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


def cmd_orb(args):
    img = load(args.input)
    orb = cv2.ORB_create(args.max)
    kp, _ = orb.detectAndCompute(img, None)
    out = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
    print(f"keypoints={len(kp)}")
    save_or_print(out, args)


def cmd_brisk(args):
    img = load(args.input)
    brisk = cv2.BRISK_create()
    kp, _ = brisk.detectAndCompute(img, None)
    out = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
    print(f"keypoints={len(kp)}")
    save_or_print(out, args)


def cmd_sift(args):
    img = load(args.input)
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(img, None)
    out = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
    print(f"keypoints={len(kp)} (SIFT 仅 Python 侧可用，浏览器用 ORB/BRISK)")
    save_or_print(out, args)


def cmd_match(args):
    img1 = load(args.input)
    img2 = load(args.second)
    orb = cv2.ORB_create(args.max)
    kp1, d1 = orb.detectAndCompute(img1, None)
    kp2, d2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < args.ratio * n.distance]
    out = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    print(f"matches={len(good)}")
    save_or_print(out, args)


def main() -> None:
    p = argparse.ArgumentParser(description="特征检测参考实现")
    sub = p.add_subparsers(dest="tool", required=True)

    p_o = sub.add_parser("orb-keypoints")
    p_o.add_argument("input")
    p_o.add_argument("--max", type=int, default=200)
    p_o.add_argument("--output")

    p_b = sub.add_parser("brisk-keypoints")
    p_b.add_argument("input")
    p_b.add_argument("--output")

    p_s = sub.add_parser("sift-keypoints")
    p_s.add_argument("input")
    p_s.add_argument("--output")

    p_m = sub.add_parser("feature-match")
    p_m.add_argument("input")
    p_m.add_argument("--second", required=True)
    p_m.add_argument("--max", type=int, default=500)
    p_m.add_argument("--ratio", type=float, default=0.75)
    p_m.add_argument("--output")

    args = p.parse_args()
    {
        "orb-keypoints": cmd_orb,
        "brisk-keypoints": cmd_brisk,
        "sift-keypoints": cmd_sift,
        "feature-match": cmd_match,
    }[args.tool](args)


if __name__ == "__main__":
    main()
