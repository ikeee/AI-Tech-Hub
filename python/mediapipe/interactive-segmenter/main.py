"""交互式分割最简实现：基于 MediaPipe Tasks InteractiveSegmenter。

使用方法：
    python main.py <图片路径> <归一化x> <归一化y>

前置准备：
    下载 magic_touch.tflite 模型文件并放置于本目录：
    https://storage.googleapis.com/mediapipe-models/image_segmenter/magic_touch/float32/latest/magic_touch.tflite

依赖安装：pip install mediapipe
"""

import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint


def segment(image_path: str, x: float, y: float) -> None:
    """给定归一化坐标点，输出该位置目标的分割掩码。"""
    base_options = python.BaseOptions(model_asset_path="magic_touch.tflite")
    options = vision.InteractiveSegmenterOptions(
        base_options=base_options,
        output_type=vision.InteractiveSegmenterOptions.OutputType.CATEGORY_MASK,
    )

    with vision.InteractiveSegmenter.create_from_options(options) as segmenter:
        image = mp.Image.create_from_file(image_path)
        # 构造 Roi（Region of Interest），指定点击位置
        roi = NormalizedKeypoint(x=x, y=y)
        result = segmenter.segment(image, roi=roi)

        mask = result.category_mask
        if mask is None:
            print("未生成掩码")
            return

        print(f"点击位置: ({x:.4f}, {y:.4f})")
        print(f"输入图片尺寸: {image.width}x{image.height}")
        print(f"掩码尺寸: {mask.width}x{mask.height}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 3:
        print("用法: python main.py <图片路径> <归一化x> <归一化y>")
        sys.exit(1)
    segment(args[0], float(args[1]), float(args[2]))
