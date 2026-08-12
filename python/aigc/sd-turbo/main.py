"""文生图/图生图（SD-Turbo）— Python 服务端实现。

用 diffusers 加载 stabilityai/sd-turbo（基于 SD2.1 的 1 步蒸馏模型，CPU 约 4-10 秒/张），
支持两种模式：
- text2img：提示词直接生成图片（默认 512x512）
- img2img：上传图片 + 提示词，用 strength 控制重绘程度

依赖安装（Python 3.11，CPU 版 torch 必须先行）：
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install diffusers transformers accelerate safetensors pillow numpy

用法：
    python main.py "" --text "a cat" --out output          # 文生图
    python main.py input.png --text "..." --out output     # 图生图
"""

import argparse
import json
import os
import sys

import torch

MODEL_ID = "stabilityai/sd-turbo"

_pipe_t2i = None
_pipe_i2i = None


def get_pipe():
    global _pipe_t2i
    if _pipe_t2i is None:
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        from diffusers import AutoPipelineForText2Image
        _pipe_t2i = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            requires_safety_checker=False,
        )
    return _pipe_t2i


def get_i2i_pipe():
    global _pipe_i2i
    if _pipe_i2i is None:
        from diffusers import AutoPipelineForImage2Image
        _pipe_i2i = AutoPipelineForImage2Image.from_pipe(get_pipe())
    return _pipe_i2i


def run(input_path: str, params: dict, out_path: str) -> str:
    mode = params.get("mode", "text2img")
    prompt = (params.get("prompt") or "").strip()
    negative = (params.get("negative") or "").strip()
    steps = max(1, min(8, int(params.get("steps", 2))))
    guidance = float(params.get("guidance", 0.0))
    seed = int(params.get("seed", -1))
    batch = max(1, min(4, int(params.get("batch", 1))))
    size = max(256, min(768, int(params.get("size", 512))))
    strength = max(0.05, min(1.0, float(params.get("strength", 0.75))))

    if not prompt:
        raise ValueError("提示词不能为空")

    generator = torch.Generator().manual_seed(seed) if seed >= 0 else None
    common = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        num_images_per_prompt=batch,
    )

    if mode == "img2img":
        if not input_path or not os.path.exists(input_path):
            raise ValueError("图生图模式需要上传输入图片")
        from PIL import Image
        init = Image.open(input_path).convert("RGB").resize((size, size), Image.LANCZOS)
        images = get_i2i_pipe()(image=init, strength=strength, **common).images
    else:
        images = get_pipe()(
            negative_prompt=negative or None,
            height=size,
            width=size,
            **common,
        ).images

    files = []
    for i, img in enumerate(images):
        p = f"{out_path}.png" if len(images) == 1 else f"{out_path}_{i}.png"
        img.save(p)
        files.append(p)
    return json.dumps({"files": files})


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="文生图/图生图（SD-Turbo）")
    parser.add_argument("input", nargs="?", default="", help="图生图输入图片（文生图可留空）")
    parser.add_argument("--mode", default="text2img", choices=["text2img", "img2img"])
    parser.add_argument("--text", default="", help="提示词")
    parser.add_argument("--negative", default="", help="负提示词")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    params = {
        "mode": args.mode,
        "prompt": args.text,
        "negative": args.negative,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "batch": args.batch,
        "size": args.size,
        "strength": args.strength,
    }
    out = run(args.input, params, args.out)
    print(f"完成: {out}", flush=True)


if __name__ == "__main__":
    main()