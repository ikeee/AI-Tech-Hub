"""语音情感识别（SER）Python 示例：基于 wav2vec2 的情感分类。

与浏览器端语音情感识别（app/pages/speech/emotion.vue）对应的「最简 Python 实现」，
仅供学习/对照展示，不参与 nuxt_AI 服务端调用（SER 为纯浏览器功能）。

依赖安装（Python 3.9+，CPU 即可，模型首次运行自动下载）：
    pip install transformers torch librosa

用法：
    python main.py input.wav [--model DunnBC22/wav2vec2-base-Speech_Emotion_Recognition] [--topk 3]

--model   情感分类模型（浏览器端用的是其 ONNX 转换版 onnx-community/...-ONNX）
--topk    输出概率最高的前 N 个情感
"""

import argparse
import sys


def predict(audio: str, model_id: str, topk: int) -> None:
    # Windows 控制台中文乱码防护
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import librosa
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    print(f"[1/3] 加载模型: {model_id}", flush=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.eval()

    print(f"[2/3] 读取音频: {audio} (重采样 16kHz)", flush=True)
    waveform, sr = librosa.load(audio, sr=16000, mono=True)
    inputs = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True)

    print("[3/3] 推理中…", flush=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

    # 优先使用模型自带标签映射，缺失时回退到常见顺序
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        id2label = {i: str(i) for i in range(probs.shape[0])}

    ranked = sorted(range(probs.shape[0]), key=lambda i: probs[i].item(), reverse=True)[:topk]
    for i in ranked:
        print(f"  {str(id2label.get(i, i)):12s} {probs[i].item() * 100:5.1f}%", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="语音情感识别（wav2vec2）")
    parser.add_argument("audio", help="输入音频（wav/mp3...）")
    parser.add_argument(
        "--model",
        default="DunnBC22/wav2vec2-base-Speech_Emotion_Recognition",
        help="Hugging Face 情感分类模型",
    )
    parser.add_argument("--topk", type=int, default=3, help="输出前 N 个情感")
    args = parser.parse_args()

    predict(args.audio, args.model, args.topk)


if __name__ == "__main__":
    main()
