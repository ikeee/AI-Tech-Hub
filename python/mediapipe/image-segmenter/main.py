"""图像分割最简实现：基于 MediaPipe Tasks Image Segmenter。

使用方法：
    python main.py <图片路径> [输出目录]

说明：使用 selfie_segmenter 分割人像前景，将分类掩码保存为 PNG。
模型文件：public/model/mediapipe/models/selfie_segmenter.tflite
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "selfie_segmenter.tflite"


def segment(image_path: str, output_dir: str = "output") -> None:
    options = vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        image = mp.Image.create_from_file(image_path)
        results = segmenter.segment(image)

        mask = results.category_mask.numpy_view().squeeze()  # (H,W,1) -> HxW, 0=背景 255=前景
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_img = PILImage.fromarray(((mask > 0) * 255).astype(np.uint8))
        out_path = out_dir / f"{Path(image_path).stem}_mask.png"
        mask_img.save(out_path)
        print(f"掩码已保存: {out_path}（前景像素占比 {(mask == 1).mean():.1%}）")


if __name__ == "__main__":
    segment(sys.argv[1] if len(sys.argv) > 1 else "input.jpg",
            sys.argv[2] if len(sys.argv) > 2 else "output")
