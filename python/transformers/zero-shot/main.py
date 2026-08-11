"""零样本文本分类演示：基于 HuggingFace transformers pipeline。

使用方法：
    python main.py
将对示例文本和候选标签输出各标签的置信度。

依赖：
    pip install transformers torch
"""

from transformers import pipeline


def classify_zero_shot(text: str, labels: list[str]) -> dict:
    """对输入文本执行零样本分类。

    参数：
        text: 待分类的文本
        labels: 候选标签列表

    返回：
        包含 labels 与 scores 的字典
    """
    # 加载零样本分类 pipeline，使用原版 bart-large-mnli
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )

    # 推理并返回分类结果
    result = classifier(text, candidate_labels=labels)
    return result


if __name__ == "__main__":
    # 示例：文本与候选标签
    sample_text = "I love playing basketball with my friends on weekends."
    candidate_labels = ["sports", "technology", "cooking", "travel"]
    result = classify_zero_shot(sample_text, candidate_labels)
    for label, score in zip(result["labels"], result["scores"]):
        print(f"标签：{label} | 置信度：{score:.4f}")
