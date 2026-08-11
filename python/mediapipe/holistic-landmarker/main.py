"""整体检测最简实现：基于 MediaPipe Holistic（Legacy Solutions）。

使用方法：
    python main.py <图片路径>

说明：
- 同时检测人体姿态、人脸、左手、右手关键点
- 使用 Legacy Solutions API，一次调用获取全部结果
- 模型文件随 mediapipe 包内置

依赖安装：pip install mediapipe opencv-python
"""

import sys

import cv2
import mediapipe as mp


def detect(image_path: str) -> None:
    """读取图片并整体检测，输出 pose/face/left_hand/right_hand 关键点数量。"""
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as holistic:
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        results = holistic.process(rgb)

        # 姿态 33 关键点
        pose = results.pose_landmarks.landmark if results.pose_landmarks else []
        print(f"Pose 关键点: {len(pose)}")

        # 人脸 468 关键点
        face = results.face_landmarks.landmark if results.face_landmarks else []
        print(f"Face 关键点: {len(face)}")

        # 左手 21 关键点
        left = results.left_hand_landmarks.landmark if results.left_hand_landmarks else []
        print(f"Left hand 关键点: {len(left)}")

        # 右手 21 关键点
        right = results.right_hand_landmarks.landmark if results.right_hand_landmarks else []
        print(f"Right hand 关键点: {len(right)}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
