import cv2
import mediapipe as mp

# 初始化MediaPipe人脸检测
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# 加载图像
image = cv2.imread('D://python project//jaoben//civitai image//124526.jpg')

# 进行人脸检测
results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# 获取检测结果
faces = results.detections

# 找到最大的人脸
max_face_location = None
max_area = 0
if faces:
    for face in faces:
        if face.score[0] > 0.5:  # 确保置信度的正确获取
            # 获取人脸的边界框
            normalized_rect = face.location_data.relative_bounding_box

            # 将归一化的边界框转换为图像坐标
            height, width, _ = image.shape
            top_left = (int(normalized_rect.xmin * width), int(normalized_rect.ymin * height))
            bottom_right = (int(normalized_rect.xmin * width + normalized_rect.width * width), 
                            int(normalized_rect.ymin * height + normalized_rect.height * height))
            face_width = bottom_right[0] - top_left[0]
            face_height = bottom_right[1] - top_left[1]
            face_area = face_width * face_height

            # 更新最大的人脸
            if face_area > max_area:
                max_area = face_area
                max_face_location = normalized_rect

# 如果检测到人脸，计算平移和缩放变换
if max_face_location:
    # 计算中心点
    center_x = int(max_face_location.xmin * width + max_face_location.width * width / 2)
    center_y = int(max_face_location.ymin * height + max_face_location.height * height / 2)

    # 计算裁剪区域
    left = int(center_x - max_face_location.width * width * 0.5)
    top = int(center_y - max_face_location.height * height * 0.5)
    right = int(center_x + max_face_location.width * width * 0.5)
    bottom = int(center_y + max_face_location.height * height * 0.5)

    # 确保裁剪区域在图像范围内
    left = max(0, min(left, width))
    top = max(0, min(top, height))
    right = min(width, max(right, 0))
    bottom = min(height, max(bottom, 0))

    # 裁剪人脸
    cropped_face = image[top:bottom, left:right]

    # 缩放人脸到1024x1024
    resized_face = cv2.resize(cropped_face, (1024, 1024), interpolation=cv2.INTER_AREA)

    # 创建一个1024x1024的图像并放置裁剪后的人脸
    output_image = resized_face

    # 保存结果到指定路径
    output_image_path = 'D://python project//jaoben//civitai image2//124526-1.jpg'  # 这里替换为你想要保存的路径
    cv2.imwrite(output_image_path, output_image)

    # 显示结果
    cv2.imshow('Output', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()