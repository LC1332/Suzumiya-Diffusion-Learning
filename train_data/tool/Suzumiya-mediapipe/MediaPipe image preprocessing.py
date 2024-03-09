import cv2
import mediapipe as mp
import numpy as np

# 初始化mediapipe人脸检测模型
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3, min_detection_confidence=0.5)

def detect_face(image_path):
    # 读取输入的图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查图像路径！")
        return None, None

    # 转换图像为RGB格式（MediaPipe人脸检测器要求输入为RGB图像）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 使用MediaPipe进行人脸检测
    results = face_mesh.process(image_rgb)

    # 检测是否成功
    if results.multi_face_landmarks:
        # 获取每张人脸的关键点，并转换为NumPy数组
        face_landmarks_list = [np.array([[lm.x, lm.y] for lm in face_landmarks.landmark]) for face_landmarks in results.multi_face_landmarks]

        # 计算每张人脸的面积，并选择最大的一个
        max_face_area = 0
        max_face_index = 0
        for i, landmarks_np in enumerate(face_landmarks_list):
            # 计算凸包
            hull = cv2.convexHull(landmarks_np.astype(np.int32))
            # 计算凸包面积
            face_area = cv2.contourArea(hull)
            # 更新最大面积和索引
            if face_area > max_face_area:
                max_face_area = face_area
                max_face_index = i

        # 获取最大的人脸关键点
        largest_face_landmarks_np = face_landmarks_list[max_face_index]

        # 返回最大的人脸关键点及其面积
        return largest_face_landmarks_np, max_face_area
    else:
        print("未检测到人脸！")
        return None, None

def align_face(image, face_landmarks_np):
    # 计算人脸中心
    center_x = np.mean(face_landmarks_np[:, 0])
    center_y = np.mean(face_landmarks_np[:, 1])
    center = (int(center_x), int(center_y))

    # 计算两眼之间的角度
    left_eye = face_landmarks_np[36]
    right_eye = face_landmarks_np[45]
    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 对图像进行旋转
    aligned_face = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return aligned_face

if __name__ == "__main__":
    image_path = 'D:\\python project\\jaoben\\CeleAHQ_subset_200 picture summary\\000004.jpg.jpg'  # 输入图像路径

    # 检测人脸并获取最大人脸的关键点及其面积
    face_landmarks_np, max_face_area = detect_face(image_path)
    
    if face_landmarks_np is not None:
        # 读取原始图像
        image = cv2.imread(image_path)
        
        # 对齐人脸
        aligned_face = align_face(image, face_landmarks_np)  # 注意这里我们只取第一个人脸的关键点
        
        # 显示原始图像和对齐后的人脸图像
        cv2.imshow('Original Image', image)
        cv2.imshow('Aligned Face', aligned_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()