import cv2
import mediapipe as mp

def align_face(image):
    # 初始化MediaPipe的FaceMesh模型
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.6)

    # 读取图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 运行面部关键点检测
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            aligned_landmarks = []
            for landmark in face_landmarks.landmark:
                aligned_x = int(landmark.x * image.shape[1])
                aligned_y = int(landmark.y * image.shape[0])
                aligned_landmarks.append((aligned_x, aligned_y))

            # 在图像上绘制对齐后的关键点
            for landmark in aligned_landmarks:
                cv2.circle(image, landmark, 2, (0, 255, 0), -1)

    # 关闭MediaPipe FaceMesh模型
    face_mesh.close()

    return image

if __name__ == "__main__":
    # 读取输入图像
    input_image = cv2.imread("D://python project//Suzumiya-Diffusion-Learning//train_data//data//image//CeleAHQ_subset_200//000004.jpg.jpg")

    # 对齐面部关键点
    aligned_image = align_face(input_image)

    # 显示对齐后的图像
    cv2.imshow("Aligned Face", aligned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存完成以上步骤后的图片
    output_path = "D://python project//jaoben//CeleAHQ_subset_200_aligned_face//aligned_face1.jpg"
    cv2.imwrite(output_path, aligned_image)