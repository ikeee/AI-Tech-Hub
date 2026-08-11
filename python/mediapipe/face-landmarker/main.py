"""人脸关键点最简实现：基于 MediaPipe Tasks Face Landmarker。

使用方法：
    python main.py <图片路径>

输出：每张人脸 478 个关键点（归一化坐标）以及前 5 个表情混合值。
模型文件：public/model/mediapipe/models/face_landmarker.task
"""

import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).resolve().parents[3] / "public" / "model" / "mediapipe" / "models" / "face_landmarker.task"


def detect(image_path: str) -> None:
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(image_path)
        results = landmarker.detect(image)

        if not results.face_landmarks:
            print("未检测到人脸")
            return

        for i, landmarks in enumerate(results.face_landmarks, 1):
            print(f"人脸 {i}: {len(landmarks)} 个关键点")
            # 打印鼻尖（第 1 个点）坐标作示例
            nose = landmarks[1]
            print(f"  鼻尖: x={nose.x:.4f}, y={nose.y:.4f}, z={nose.z:.4f}")
            if results.face_blendshapes and results.face_blendshapes[i - 1]:
                top = sorted(results.face_blendshapes[i - 1], key=lambda b: b.score, reverse=True)[:5]
                print("  表情混合值:", ", ".join(f"{b.category_name}={b.score:.3f}" for b in top))


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
