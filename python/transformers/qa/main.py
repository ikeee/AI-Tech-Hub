"""问答抽取演示：基于 HuggingFace transformers pipeline。

使用方法：
    python main.py
将对示例问题和上下文输出答案及置信度。

依赖：
    pip install transformers torch
"""

from transformers import pipeline


def answer_question(question: str, context: str) -> dict:
    """根据上下文回答问题。

    参数：
        question: 待回答的问题
        context: 包含答案的上下文

    返回：
        包含 answer 与 score 的字典
    """
    # 加载问答 pipeline，使用原版 distilbert 模型
    qa = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
    )

    # 推理并返回答案
    result = qa(question=question, context=context)
    return result


if __name__ == "__main__":
    # 示例：问题与上下文
    question = "Where is HuggingFace based?"
    context = (
        "HuggingFace is a company based in New York City. "
        "It is known for its open-source machine learning libraries."
    )
    result = answer_question(question, context)
    print(f"答案：{result['answer']} | 置信度：{result['score']:.4f}")
