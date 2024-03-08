import cv2
import mediapipe as mp
import numpy as np
import os

# 初始化 MediaPipe Pose 模型，用于计算图片的平均姿态
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7)

# 从文件夹中读取图像的函数
def read_images(image_folder):
    images = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
    return images

# 计算平均姿态的函数
def calculate_average_pose(images):
    all_landmarks = []
    for image in images:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = np.array([[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark])
            all_landmarks.append(landmarks)
    if all_landmarks:
        average_landmarks = np.mean(all_landmarks, axis=0)
        return average_landmarks
    else:
        return None

# 包含图像的文件夹
image_folder = r"D:\python project\jaoben\CeleAHQ_subset_200 mirror image"

# 从文件夹中读取图像
images = read_images(image_folder)

# 计算平均姿态
average_pose = calculate_average_pose(images)

if average_pose is not None:
    print("计算得到的平均姿态为:")
    print(average_pose)
else:
    print("未在提供的图像中检测到姿态。")