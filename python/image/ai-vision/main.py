"""AI 目标与图像视觉（AI Object & Image Vision）：分类/检测/分割/抠图/嵌入/相似度。

依赖：pip install mediapipe opencv-python numpy
用法：
  python main.py classify <input> [--top 5]
  python main.py detect <input> [--output out.png]
  python main.py segment <input> [--output out.png]
  python main.py background-removal <input> [--output out.png]
  python main.py embed <input>
  python main.py similarity <input> --second <img2>
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


def cmd_classify(args):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base = python.BaseOptions(model_asset_path="efficientnet_lite0.tflite")
    opts = vision.ImageClassifierOptions(base_options=base, max_results=args.top)
    with vision.ImageClassifier.create_from_options(opts) as c:
        res = c.classify(vision.Image.create_from_file(args.input))
        for cat in res.classifications[0].categories:
            print(f"{cat.category_name}: {cat.score:.4f}")


def cmd_detect(args):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base = python.BaseOptions(model_asset_path="efficientdet_lite0.tflite")
    opts = vision.ObjectDetectorOptions(base_options=base, max_results=5, score_threshold=0.5)
    img = load(args.input)
    with vision.ObjectDetector.create_from_options(opts) as d:
        res = d.detect(vision.Image.create_from_file(args.input))
        out = img.copy()
        for det in res.detections:
            bb = det.bounding_box
            cv2.rectangle(out, (bb.origin_x, bb.origin_y), (bb.origin_x + bb.width, bb.origin_y + bb.height), (0, 220, 130), 3)
            label = det.categories[0].category_name
            cv2.putText(out, label, (bb.origin_x, max(20, bb.origin_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 130), 2)
        print(f"objects={len(res.detections)}")
        save_or_print(out, args)


def cmd_segment(args, removal: bool):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base = python.BaseOptions(model_asset_path="selfie_segmenter.tflite")
    opts = vision.ImageSegmenterOptions(base_options=base, output_category_mask=True)
    img = load(args.input)
    with vision.ImageSegmenter.create_from_options(opts) as s:
        res = s.segment(vision.Image.create_from_file(args.input))
        mask = res.category_mask.numpy_view()
        if removal:
            out = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            out[mask == 0] = (0, 0, 0, 0)
        else:
            overlay = img.copy()
            overlay[mask > 0] = (0, 220, 130)
            out = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        save_or_print(out, args)


def cmd_embed(args):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base = python.BaseOptions(model_asset_path="mobilenet_v3_small.tflite")
    opts = vision.ImageEmbedderOptions(base_options=base, l2_normalize=True)
    with vision.ImageEmbedder.create_from_options(opts) as e:
        emb = e.embed(vision.Image.create_from_file(args.input))
        v = emb.embeddings[0].embedding
        print(f"dim={len(v)} first6={[round(x, 4) for x in v[:6]]}")


def cmd_similarity(args):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    base = python.BaseOptions(model_asset_path="mobilenet_v3_small.tflite")
    opts = vision.ImageEmbedderOptions(base_options=base, l2_normalize=True)
    with vision.ImageEmbedder.create_from_options(opts) as e:
        e1 = e.embed(vision.Image.create_from_file(args.input)).embeddings[0]
        e2 = e.embed(vision.Image.create_from_file(args.second)).embeddings[0]
        sim = vision.ImageEmbedder.cosine_similarity(e1, e2)
        print(f"cosine similarity = {sim:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="AI 目标与图像视觉参考实现")
    sub = p.add_subparsers(dest="tool", required=True)
    p_c = sub.add_parser("classify")
    p_c.add_argument("input")
    p_c.add_argument("--top", type=int, default=5)
    p_d = sub.add_parser("detect")
    p_d.add_argument("input")
    p_d.add_argument("--output")
    for name in ("segment", "background-removal"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--output")
    p_e = sub.add_parser("embed")
    p_e.add_argument("input")
    p_s = sub.add_parser("similarity")
    p_s.add_argument("input")
    p_s.add_argument("--second", required=True)
    args = p.parse_args()
    {
        "classify": cmd_classify,
        "detect": cmd_detect,
        "segment": lambda a: cmd_segment(a, False),
        "background-removal": lambda a: cmd_segment(a, True),
        "embed": cmd_embed,
        "similarity": cmd_similarity,
    }[args.tool](args)


if __name__ == "__main__":
    main()
