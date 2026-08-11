"""完形填空（掩码预测）演示：基于 HuggingFace transformers pipeline。

使用方法：
    python main.py
将对含 [MASK] 的示例文本输出候选词及概率。

依赖：
    pip install transformers torch
"""

from transformers import pipeline


def fill_mask(text: str, top_k: int = 5) -> list[dict]:
    """对含 [MASK] 的文本预测候选词。

    参数：
        text: 含 [MASK] 占位符的文本
        top_k: 返回的候选词数量

    返回：
        候选词列表，每个元素包含 token_str 与 score 字段
    """
    # 加载 fill-mask pipeline，使用原版 bert-base-uncased
    filler = pipeline(
        "fill-mask",
        model="bert-base-uncased",
    )

    # 推理并返回候选词
    results = filler(text, top_k=top_k)
    return results


if __name__ == "__main__":
    # 示例：含 [MASK] 的文本
    sample = "The capital of France is [MASK]."
    candidates = fill_mask(sample)
    for cand in candidates:
        print(f"候选词：{cand['token_str']} | 概率：{cand['score']:.4f}")
