"""单目深度估计演示：基于 HuggingFace transformers pipeline。

使用方法：
    python main.py
将根据示例图片路径输出深度图 PNG 文件。

依赖：
    pip install transformers torch pillow
"""

from transformers import pipeline


def estimate_depth(image_path: str, output_path: str) -> None:
    """对输入图片执行深度估计，并将结果保存为 PNG。

    参数：
        image_path: 输入图片路径
        output_path: 深度图保存路径
    """
    # 加载深度估计 pipeline，使用原版 Depth-Anything-Small-hf
    depth = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-Small-hf",
    )

    # 推理并保存深度图
    result = depth(image_path)
    depth_image = result["depth"]
    depth_image.save(output_path)
    print(f"深度图已保存至：{output_path}")


if __name__ == "__main__":
    # 示例：替换为实际图片路径
    estimate_depth("input.jpg", "depth.png")
