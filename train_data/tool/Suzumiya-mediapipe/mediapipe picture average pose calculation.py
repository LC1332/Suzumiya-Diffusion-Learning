import cv2
import mediapipe as mp
import numpy as np
import os

#初始化人脸检测模型

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)