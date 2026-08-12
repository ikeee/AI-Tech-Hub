"""人脸视觉（Face Vision）：检测/关键点/模糊/马赛克/识别/验证。

检测与关键点用 MediaPipe，识别/验证用 insightface（ArcFace）。

用法：
  python main.py detect <input> [--output out.png]
  python main.py landmark <input> [--output out.png]
  python main.py blur <input> [--output out.png]
  python main.py pixelate <input> [--output out.png]
  python main.py recognition <input> [--output labels.txt]   # insightface
  python main.py verification <input> --second <img2>        # insightface

依赖：
  pip install mediapipe opencv-python numpy
  识别/验证另需：pip install insightface onnxruntime
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


def get_detector():
    import mediapipe as mp
    return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def face_boxes(img, det):
    results = det.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = []
    if results.detections:
        h, w = img.shape[:2]
        for d in results.detections:
            bb = d.location_data.relative_bounding_box
            x = int(bb.xmin * w)
            y = int(bb.ymin * h)
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            boxes.append((x, y, bw, bh))
    return boxes


def cmd_detect(args):
    img = load(args.input)
    det = get_detector()
    boxes = face_boxes(img, det)
    out = img.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 130), 3)
    print(f"faces={len(boxes)}")
    save_or_print(out, args)


def cmd_landmark(args):
    img = load(args.input)
    import mediapipe as mp
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4) as fm:
        res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out = img.copy()
        if res.multi_face_landmarks:
            h, w = img.shape[:2]
            for lm in res.multi_face_landmarks:
                pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                for p in pts:
                    cv2.circle(out, p, 1, (0, 220, 130), -1)
            print(f"faces={len(res.multi_face_landmarks)}")
        save_or_print(out, args)


def cmd_blur(args):
    img = load(args.input)
    det = get_detector()
    out = img.copy()
    for x, y, w, h in face_boxes(img, det):
        pad_x, pad_y = int(w * 0.35), int(h * 0.35)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
        roi = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (51, 51), 0)
    save_or_print(out, args)


def cmd_pixelate(args):
    img = load(args.input)
    det = get_detector()
    out = img.copy()
    for x, y, w, h in face_boxes(img, det):
        pad_x, pad_y = int(w * 0.35), int(h * 0.35)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
        roi = out[y0:y1, x0:x1]
        cell = max(4, roi.shape[1] // 14)
        small = cv2.resize(roi, (max(1, roi.shape[1] // cell), max(1, roi.shape[0] // cell)), interpolation=cv2.INTER_LINEAR)
        out[y0:y1, x0:x1] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    save_or_print(out, args)


def get_recognizer():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def cmd_recognition(args):
    app = get_recognizer()
    img = load(args.input)
    faces = app.get(img)
    print(f"faces={len(faces)}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for i, face in enumerate(faces):
                emb = ",".join(f"{v:.6f}" for v in face.embedding)
                f.write(f"face{i} {emb}\n")
        print(f"已保存嵌入: {args.output}")


def cmd_verification(args):
    app = get_recognizer()
    img1 = load(args.input)
    img2 = load(args.second)
    f1 = app.get(img1)
    f2 = app.get(img2)
    if not f1 or not f2:
        print("error: no face found in one of the images")
        return
    sim = np.dot(f1[0].embedding, f2[0].embedding) / (
        np.linalg.norm(f1[0].embedding) * np.linalg.norm(f2[0].embedding)
    )
    print(f"cosine similarity = {sim:.4f}")
    print("verdict: " + ("same person" if sim > 0.5 else "different persons"))


def main() -> None:
    p = argparse.ArgumentParser(description="人脸视觉参考实现")
    sub = p.add_subparsers(dest="tool", required=True)
    for name in ("detect", "landmark", "blur", "pixelate"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--output")
    p_r = sub.add_parser("recognition")
    p_r.add_argument("input")
    p_r.add_argument("--output")
    p_v = sub.add_parser("verification")
    p_v.add_argument("input")
    p_v.add_argument("--second", required=True)
    args = p.parse_args()
    {
        "detect": cmd_detect,
        "landmark": cmd_landmark,
        "blur": cmd_blur,
        "pixelate": cmd_pixelate,
        "recognition": cmd_recognition,
        "verification": cmd_verification,
    }[args.tool](args)


if __name__ == "__main__":
    main()
