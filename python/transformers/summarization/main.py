"""文本摘要演示：基于 HuggingFace transformers pipeline。

使用方法：
    python main.py
将对示例长文本输出对应的摘要。

依赖：
    pip install transformers torch
"""

from transformers import pipeline


def summarize(text: str, max_length: int = 60, min_length: int = 15) -> str:
    """对输入长文本生成摘要。

    参数：
        text: 待摘要的长文本
        max_length: 摘要最大长度
        min_length: 摘要最小长度

    返回：
        生成的摘要文本
    """
    # 加载摘要 pipeline，使用原版 distilbart 模型
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
    )

    # 推理并提取摘要
    result = summarizer(text, max_length=max_length, min_length=min_length)
    summary = result[0]["summary_text"]
    return summary


if __name__ == "__main__":
    # 示例长文本
    long_text = (
        "The Apollo program was a series of space missions undertaken by the "
        "United States in the 1960s and 1970s. Its goal was to land humans on "
        "the Moon and bring them safely back to Earth. Apollo 11, launched in "
        "1969, was the first mission to achieve this historic milestone."
    )
    summary = summarize(long_text)
    print(f"摘要：{summary}")
