"""人体姿态关键点检测最简实现：基于 MediaPipe Pose（Legacy Solutions）。

使用方法：
    python main.py <图片路径>

说明：
- 使用 Legacy Solutions API，比 Tasks API 更简洁
- 共检测 33 个关键点
- 模型文件随 mediapipe 包内置，无需额外下载

依赖安装：pip install mediapipe opencv-python
"""

import sys

import cv2
import mediapipe as mp


def detect(image_path: str) -> None:
    """读取图片并检测人体姿态，输出 33 个关键点坐标。"""
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        min_detection_confidence=0.5,
    ) as pose:
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        results = pose.process(rgb)

        if not results.pose_landmarks:
            print("未检测到人体姿态")
            return

        landmarks = results.pose_landmarks.landmark
        print(f"检测到 {len(landmarks)} 个关键点:")
        for i, lm in enumerate(landmarks):
            print(f"  点 {i}: x={lm.x:.4f} y={lm.y:.4f} z={lm.z:.4f} "
                  f"可见度={lm.visibility:.4f}")


if __name__ == "__main__":
    detect(sys.argv[1] if len(sys.argv) > 1 else "input.jpg")
