"""交互式分割最简实现：基于 MediaPipe Tasks Interactive Segmenter。

使用方法：
    python main.py <图片路径> <点击x> <点击y> [输出目录]

说明：点击图片上的目标点（归一化坐标 0-1），分割出该目标区域并保存掩码 PNG。
模型文件：models/magic_touch.tflite（官方 float32 版，兼容 MediaPipe Python tasks）
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
from mediapipe.tasks.python.vision.interactive_segmenter import RegionOfInterest

MODEL_PATH = Path(__file__).resolve().parent / "models" / "magic_touch.tflite"


def segment(image_path: str, x: float, y: float, output_dir: str = "output") -> None:
    options = vision.InteractiveSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        output_category_mask=True,
    )

    with vision.InteractiveSegmenter.create_from_options(options) as segmenter:
        image = mp.Image.create_from_file(image_path)
        roi = vision.InteractiveSegmenterRegionOfInterest(
            format=RegionOfInterest.Format.KEYPOINT,
            keypoint=NormalizedKeypoint(x=x, y=y),
        )
        results = segmenter.segment(image, roi)

        mask = results.category_mask.numpy_view().squeeze()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_img = PILImage.fromarray(((mask > 0) * 255).astype(np.uint8))
        out_path = out_dir / f"{Path(image_path).stem}_interactive_mask.png"
        mask_img.save(out_path)
        print(f"掩码已保存: {out_path}（目标像素占比 {(mask == 1).mean():.1%}）")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python main.py <图片路径> <点击x> <点击y> [输出目录]")
        sys.exit(1)
    segment(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]),
            sys.argv[4] if len(sys.argv) > 4 else "output")
