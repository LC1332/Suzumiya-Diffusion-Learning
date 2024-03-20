import cv2
from tqdm import tqdm
import pickle

import os
import re

# 确保output目录存在
output_dir = 'output'

# 指定保存 Parquet 文件的路径
output_parquet_path = 'output/celeb_features_CLIP-H-14_50k.parquet'

processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


# 指定保存文件的路径
celeb2images_path = os.path.join(output_dir, 'celeb2images.pkl')

with open( celeb2images_path, 'rb' ) as f:
    celeb2images = pickle.load(f)

img_list = []

img_name2celeb = {}

for item in celeb2images:
    img_list += celeb2images[item]
    for img_name in celeb2images[item]:
        img_name2celeb[img_name] = item

cropped_img_list = []

for img_name in tqdm( img_list ):

    new_path = re.sub(r"FaceData/Data/imdbface_loose_crop_subset_[0-9]/imdbface_loose_crop_subset_[0-9]", 'AllCrop', img_name)

    # skip if already exists
    if not os.path.exists( new_path ):
        continue

    celeb_name = img_name2celeb[img_name]

    cropped_img_list.append( (celeb_name, new_path ) )

    # for debug
    if len(cropped_img_list) > 50000:
        break

print(len(cropped_img_list))

# initialize extractor
from CLIPExtractor import CLIPExtractor

if processor_name is not None and model_name is not None:
    extractor = CLIPExtractor(processor_name, model_name)
else:
    extractor = CLIPExtractor()

celeb_names = [item[0] for item in cropped_img_list]
img_list = [item[1] for item in cropped_img_list]

features = extractor.extract(img_list, batch_size = 32)

prefix_len = len("AllCrop") + 1

import pandas as pd

# 初始化存储数据的列表
data = []

# 遍历之前收集的信息
for celeb_name, crop_image_name, feature in zip(celeb_names, img_list, features):
    image_name = crop_image_name[prefix_len:]
    # 将特征向量转换为列表形式
    feature_list = feature.tolist()  # 假设 feature 是 NumPy 数组
    # 将这一行的数据添加到列表中
    data.append([celeb_name, image_name, feature_list])

# 创建 DataFrame
df = pd.DataFrame(data, columns=['celeb_name', 'image_name', 'FaRL'])


# 保存到 Parquet 文件，确保 pyarrow 已安装
df.to_parquet(output_parquet_path, index=False)

# 我这里希望实现一个parquet文件，每一行都是celeb_name ,image_name和feature 现在循环中的feature是一个numpy格式的vector