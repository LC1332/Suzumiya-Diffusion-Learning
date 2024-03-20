# scan each celebrity data
# project all feature into 512 dimension using lda_model
# get the average feature vector
# determinate which image is the most close to the mean vector

from tqdm import tqdm

lda_model_path = "output/lda_openai_clip_model.pkl"

# 指定Feature 存储的路径
output_parquet_path = 'output/celeb_features-full.parquet'
feature_column_name = 'feature'

crop_image_path = 'AllCrop'

if_debug = True

print("加载LDA模型")

import pickle
with open(lda_model_path, 'rb') as f:
    lda = pickle.load(f)

def project_to_lda(feature):
    return lda.transform([feature])

import numpy as np
def normalize_feature(feature):
    return feature / (1e-10 + np.linalg.norm(feature))

import pandas as pd

print("读取特征数据")

# 读取特征数据
df = pd.read_parquet(output_parquet_path)

# show all column names
print(df.columns)

celeb2datas =  {}

nrows = df.shape[0]

print("投影以及归并")

count = 0

for index, row in tqdm( df.iterrows(), total = nrows ):
    feature = row[feature_column_name]
    feature = project_to_lda(feature)
    feature = normalize_feature(feature)

    celeb_name = row['celeb_name']

    if celeb_name not in celeb2datas:
        celeb2datas[celeb_name] = {
            "features": [],
            "image_names": []
        }

    if len(feature.shape) != 2 or feature.shape[1] != 512:
        print("feature shape error", feature.shape)
        continue

    celeb2datas[celeb_name]["features"].append(feature)
    celeb2datas[celeb_name]["image_names"].append(row['image_name'])

    count += 1

    if if_debug and count > 1000:
        break

print("计算平均特征")

save_datas = []

from sklearn.metrics.pairwise import cosine_similarity

for celeb_name in celeb2datas:
    features = celeb2datas[celeb_name]["features"] # here features is a list of (1,512) feature
    average_feature = np.mean( features , axis = 0 )
    average_feature = normalize_feature(average_feature)

    # print(average_feature.shape)

    similarities = [cosine_similarity(average_feature, feature) for feature in features]

    # 找到最相似的index
    most_similar_index = np.argmax(similarities)

    image_name = celeb2datas[celeb_name]["image_names"][most_similar_index]

    # print(image_name)

    save_data = {
        "celeb_name": celeb_name,
        "image_name": image_name,
        "image": None,
        "average_feature": average_feature.tolist()
    }

    save_datas.append(save_data)


from PIL import Image

from PIL import Image

import io

from datasets import Image as datasets_Image

datasets_encoder = datasets_Image()

def load_and_transform_image(image_path):
    image = Image.open(image_path)
    if max(image.size) > 150:
        image = image.resize((150, 150))
    
    in_memory_file = io.BytesIO()
    image.save(in_memory_file, format='PNG')
    in_memory_file.seek(0)
    png_image = Image.open(in_memory_file)
    return datasets_encoder.encode_example(png_image)

# 50个celeb是2.5M
# 1000个celeb是50M
# 10000个celeb是500M


import os

for save_data in save_datas:
    image_path = os.path.join(crop_image_path, save_data["image_name"])
    pixel_value = load_and_transform_image(image_path)
    save_data["image"] = pixel_value

    

save_df = pd.DataFrame(save_datas)

save_name = "output/celeb_average_feature.parquet"

save_df.to_parquet(save_name)

print(f"{len(save_datas)} celeb extracted ")

# 获取文件大小
file_size = os.path.getsize(save_name)

# 打印文件大小，转换为KB
print(f"文件大小: {file_size / 1024:.2f} KB")
