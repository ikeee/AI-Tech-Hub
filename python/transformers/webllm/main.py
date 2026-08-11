"""LLM 对话演示：基于 transformers 的多轮对话实现。

对应浏览器端 WebLLM 的 Python 参考，使用 Qwen2.5-0.5B-Instruct
演示系统提示词 + 多轮对话 + 流式输出的完整流程。

依赖安装：
    pip install transformers torch

用法：
    python main.py
"""

import threading

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def chat() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map="auto"
    )

    messages = [
        {"role": "system", "content": "你是一个简洁友好的助手，用中文回答。"},
    ]

    print("输入 'exit' 退出。")
    while True:
        user = input("用户: ").strip()
        if not user or user.lower() == "exit":
            break
        messages.append({"role": "user", "content": user})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 用迭代流式器边生成边打印，同时收集回复文本
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {
            **inputs,
            "max_new_tokens": 512,
            "do_sample": True,
            "use_cache": True,
            "streamer": streamer,
        }
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        print("助手: ", end="", flush=True)
        reply_parts: list[str] = []
        for piece in streamer:
            print(piece, end="", flush=True)
            reply_parts.append(piece)
        thread.join()
        print()

        messages.append({"role": "assistant", "content": "".join(reply_parts)})


if __name__ == "__main__":
    chat()
