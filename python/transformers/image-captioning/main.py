"""图像描述生成演示：基于 HuggingFace transformers pipeline。

使用方法：
    python main.py
将根据示例图片路径输出对应的文字描述。

依赖：
    pip install transformers torch pillow
"""

from transformers import pipeline


def caption_image(image_path: str) -> str:
    """对输入图片生成文字描述。

    参数：
        image_path: 输入图片路径

    返回：
        生成的图片描述文本
    """
    # 加载图像转文本 pipeline，使用原版 vit-gpt2 模型
    captioner = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning",
    )

    # 推理并提取生成的文本
    result = captioner(image_path)
    text = result[0]["generated_text"]
    return text


if __name__ == "__main__":
    # 示例：替换为实际图片路径
    description = caption_image("input.jpg")
    print(f"图片描述：{description}")
