"""图像前景分割最简实现：基于 MediaPipe Tasks ImageSegmenter。

使用方法：
    python main.py <图片路径>

前置准备：
    下载 selfie_segmenter.tflite 模型文件并放置于本目录：
    https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite

依赖安装：pip install mediapipe
"""

import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def segment(image_path: str) -> None:
    """读取图片并分割前景，输出分割掩码信息。"""
    base_options = python.BaseOptions(model_asset_path="selfie_segmenter.tflite")
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        output_type=vision.ImageSegmenterOptions.OutputType.CATEGORY_MASK,
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        image = mp.Image.create_from_file(image_path)
        result = segmenter.segment(image)

        # CATEGORY_MASK 模式下返回单张分类掩码
        mask = result.category_mask
        if mask is None:
            print("未生成掩码")
            return

        # mask 为 mp.Image，可通过 mask.numpy_view() 获取 ndarray
        print(f"输入图片尺寸: {image.width}x{image.height}")
        print(f"掩码尺寸: {mask.width}x{mask.height}")
        print(f"掩码数据类型: {mask.numpy_view().dtype}")


if __name__ == "__main__":
    segment(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
