from MPCropAndNorm import MPCropAndNorm
import cv2
from tqdm import tqdm
import pickle

import os
import re

# 确保output目录存在
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 指定保存文件的路径
celeb2images_path = os.path.join(output_dir, 'celeb2images.pkl')

with open( celeb2images_path, 'rb' ) as f:
    celeb2images = pickle.load(f)

print(len(celeb2images))

img_list = []

for item in celeb2images:
    # print(item)
    img_list += celeb2images[item]

detector = MPCropAndNorm()

save_path = "AllCrop"

failed_to_read = 0
failed_to_detect = 0
failed_to_crop = 0
count = 0

for img_name in tqdm( img_list ):

    new_path = re.sub(r"FaceData/Data/imdbface_loose_crop_subset_[0-9]/imdbface_loose_crop_subset_[0-9]", 'AllCrop', img_name)

    # skip if already exists
    if os.path.exists( new_path ):
        continue

    try:
        img = cv2.imread( img_name )
    except:
        failed_to_read += 1
        continue

    if img is None:
        failed_to_read += 1
        continue

    try:
        faces = detector.detect_face(img)
    except:
        failed_to_detect+=1
        continue

    if len(faces) == 0:
        win_size = int((img.shape[0] + img.shape[1] ) // 2)
        faces.append((0,0,win_size,win_size))

    if len(faces)>1:
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        faces = [largest_face]

    try:
        cropped_images = detector.crop_and_norm_with_face(img, faces)
    except:
        failed_to_crop+=1
        continue

    count += 1

    

    if len( cropped_images ) == 0:
        continue

    cv2.imwrite( new_path, cropped_images[0] )

    # break

print("failed to read: ", failed_to_read)
print("failed to detect: ", failed_to_detect)
print("count: ", count)
    