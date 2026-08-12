"""人脸识别/验证（insightface ArcFace）。

命令行用法：
  python main.py recognize <input>
  python main.py verify <input> --second <img2>

依赖：pip install insightface onnxruntime opencv-python numpy
模型：insightface 首次运行自动下载 buffalo_l（约 300MB，放用户目录 ~/.insightface）。
"""

import argparse
import json

import cv2
import numpy as np


_app = None


def get_app():
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _app = app
    return _app


def recognize(path: str) -> dict:
    app = get_app()
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    faces = app.get(img)
    return {
        "faces": len(faces),
        "dim": 512,
        "embeddings": [f.embedding.tolist() for f in faces],
        "bboxes": [[float(v) for v in f.bbox] for f in faces],
        "det_scores": [float(f.det_score) for f in faces],
    }


def verify(path1: str, path2: str) -> dict:
    app = get_app()
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    if img1 is None or img2 is None:
        raise ValueError("无法读取图片")
    f1 = app.get(img1)
    f2 = app.get(img2)
    if not f1 or not f2:
        return {"similarity": None, "verdict": "no-face", "faces": (len(f1), len(f2))}
    a = f1[0].embedding
    b = f2[0].embedding
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return {"similarity": round(sim, 4), "verdict": "same" if sim > 0.5 else "different", "faces": (len(f1), len(f2))}


def main() -> None:
    p = argparse.ArgumentParser(description="人脸识别/验证")
    sub = p.add_subparsers(dest="tool", required=True)
    p_r = sub.add_parser("recognize")
    p_r.add_argument("input")
    p_v = sub.add_parser("verify")
    p_v.add_argument("input")
    p_v.add_argument("--second", required=True)
    args = p.parse_args()
    if args.tool == "recognize":
        r = recognize(args.input)
        print(json.dumps(r, ensure_ascii=False))
    else:
        r = verify(args.input, args.second)
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
