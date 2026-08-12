"""AI 视觉与多模态（AI Vision & Multimodal）：图像描述 / 深度估计。

依赖：pip install opencv-python numpy transformers torch pillow
用法：
  python main.py caption <input> [--max-tokens 40]
  python main.py depth <input> [--output depth.png]
  python main.py qa <input> --question "图中有什么？"
  python main.py inpaint <input> --mask mask.png [--output out.png]
"""

import argparse

import cv2
import numpy as np


def load(path: str):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"无法读取图片: {path}")
    return img


def cmd_caption(args):
    from transformers import pipeline
    p = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
    out = p(args.input, max_new_tokens=args.max_tokens)
    print(out[0]["generated_text"])


def cmd_depth(args):
    from transformers import pipeline
    import torch
    p = pipeline("depth-estimation", model="Xenova/depth-anything-small-hf")
    out = p(args.input)
    depth = out["depth"]
    arr = np.array(depth)
    norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    save_or_print(cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR), args)


def cmd_qa(args):
    from transformers import pipeline
    p = pipeline("visual-question-answering", model="dandelin/ViLT-B32-finetuned-VQA")
    out = p(image=args.input, question=args.question, top_k=3)
    for item in out:
        print(f"{item['answer']}: {item['score']:.3f}")


def cmd_inpaint(args):
    from transformers import pipeline
    p = pipeline("image-to-image", model="Xenova/paint-by-text")
    # 简化：mask 图 + 原图做简单 inpainting（教学示例）
    img = load(args.input)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"无法读取 mask: {args.mask}")
    out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    save_or_print(out, args)


def save_or_print(img, args: argparse.Namespace) -> None:
    if args.output:
        cv2.imwrite(args.output, img)
        print(f"已保存: {args.output}")
    else:
        print(f"result: {img.shape[1]}x{img.shape[0]}")


def main() -> None:
    p = argparse.ArgumentParser(description="AI 视觉与多模态参考实现")
    sub = p.add_subparsers(dest="tool", required=True)
    p_c = sub.add_parser("caption")
    p_c.add_argument("input")
    p_c.add_argument("--max-tokens", type=int, default=40)
    p_d = sub.add_parser("depth")
    p_d.add_argument("input")
    p_d.add_argument("--output")
    p_q = sub.add_parser("qa")
    p_q.add_argument("input")
    p_q.add_argument("--question", required=True)
    p_i = sub.add_parser("inpaint")
    p_i.add_argument("input")
    p_i.add_argument("--mask", required=True)
    p_i.add_argument("--output")
    args = p.parse_args()
    {
        "caption": cmd_caption,
        "depth": cmd_depth,
        "qa": cmd_qa,
        "inpaint": cmd_inpaint,
    }[args.tool](args)


if __name__ == "__main__":
    main()
