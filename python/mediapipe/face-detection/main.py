"""人脸检测最简实现：基于 MediaPipe Face Detection。

使用方法：
    python main.py <图片路径>

说明：
- model_selection=0 适用于 2 米以内的人脸（短距离）
- model_selection=1 适用于 5 米以内的人脸（长距离）
- 模型文件随 mediapipe 包内置，无需额外下载

依赖安装：pip install mediapipe opencv-python
"""

import sys

import cv2
import mediapipe as mp


def detect(image_path: str) -> None:
    """读取图片并检测人脸，输出归一化坐标和置信度。"""
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=0,  # 0=短距离，1=长距离
        min_detection_confidence=0.5,
    ) as detector:
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        results = detector.process(rgb)

        if not results.detections:
            print("未检测到人脸")
            return

        for i, det in enumerate(results.detections, 1):
            box = det.location_data.relative_bounding_box
            print(f"人脸 {i}:")
            print(f"  置信度: {det.score[0]:.4f}")
            print(f"  归一化边界框: x={box.xmin:.4f} y={box.ymin:.4f} "
                  f"w={box.width:.4f} h={box.height:.4f}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
