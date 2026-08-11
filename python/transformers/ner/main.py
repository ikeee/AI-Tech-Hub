"""命名实体识别演示：基于 HuggingFace transformers pipeline。

使用方法：
    python main.py
将对示例文本输出识别到的实体列表（文本、类型、置信度）。

依赖：
    pip install transformers torch
"""

from transformers import pipeline


def recognize_entities(text: str) -> list[dict]:
    """对输入文本执行命名实体识别。

    参数：
        text: 待识别的文本

    返回：
        实体列表，每个实体包含 word、entity_group、score 字段
    """
    # 加载 NER pipeline，使用原版 BERT 模型
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
    )

    # 推理并返回实体列表
    entities = ner(text)
    return entities


if __name__ == "__main__":
    # 示例文本
    sample = "HuggingFace is based in New York City and was founded in 2016."
    results = recognize_entities(sample)
    for ent in results:
        print(f"实体：{ent['word']} | 类型：{ent['entity_group']} | 置信度：{ent['score']:.4f}")
