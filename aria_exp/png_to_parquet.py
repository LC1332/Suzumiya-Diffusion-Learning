#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
__info: ""
"""

crop_mode = False
import cv2
from tqdm import tqdm
import pickle

import glob
import re
import os, json

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from sklearn.metrics.pairwise import cosine_similarity

import torch
from typing import Tuple
from torch import Tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pickle
from CLIPExtractor import CLIPExtractor
from MPCropAndNorm import MPCropAndNorm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import glob
import csv
import gradio as gr
import datetime

TIME = datetime.datetime.now().strftime('%m%d_%H%M_%S')
model_method = "identity"

import socket

HOST_NAME = socket.gethostname()

output_dir = "/mnt/data/MLSFT/aria_exp/" + 'output'
comfy_output_dir = "/mnt/data/MLSFT/aria_exp/comfy_output"
tmp_dir = "/mnt/data/MLSFT/aria_exp/tmp/"
output_parquet_path = "/mnt/data/MLSFT/aria_exp" + "/" + model_method + ".parquet"
ERROR_TXT = "/mnt/data/MLSFT/aria_exp/" + 'error_' + model_method + '.txt'
ROOT_DIR = "/home/aria/lattecodes/aria_exp/"
RESOURCES_DIR = "/mnt/data/MLSFT/resources/Models"
MODEL_ID = RESOURCES_DIR + "/embed/img/simple-face-recognition/lda_openai_clip_model.pkl"
MODEL_EXP = RESOURCES_DIR + "/embed/img/RAF_clip_LDA.pkl"
MODEL_OAI = "/mnt/data/MLSFT/resources/Models/embed/img/clip-vit-base-patch16"
ROOT_PATH = "/home/aria/lattecodes/aria_exp/data/comfyoutput/"
OUT_CSV_PATH = ROOT_DIR + 'similarity_scores_expression.csv'
OUT_JSON_PATH = "/home/aria/lattecodes/aria_exp/" + 'similarity_scores_' + model_method + '.json'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def get_png_files(folder_path):
    os.chdir(folder_path)
    png_files = glob.glob('*.png') + glob.glob('*.jpg')
    file_names = [os.path.basename(file) for file in png_files]
    return file_names


def get_filename_list(folder_path):
    png_files_info = get_png_files(folder_path)
    all_png_file_names_list = []
    all_png_file_names_list.extend(png_files_info)
    print("\033[0;30;43m all_png_file_names_list: \033[0m ", len(all_png_file_names_list), "\n",
          all_png_file_names_list)
    return all_png_file_names_list


def path_to_name(img_path):
    filename = os.path.basename(img_path)
    return filename


def lushu():
    output_parquet_path = '20.parquet'
    processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    celeb2images_path = os.path.join(output_dir, 'celeb2images.pkl')
    with open(celeb2images_path, 'rb') as f:
        celeb2images = pickle.load(f)

    img_list = []

    img_name2celeb = {}

    for item in celeb2images:
        img_list += celeb2images[item]
        for img_name in celeb2images[item]:
            img_name2celeb[img_name] = item

    cropped_img_list = []
    img_list = []
    for img_name in tqdm(img_list):

        new_path = re.sub(r"FaceData/Data/imdbface_loose_crop_subset_[0-9]/imdbface_loose_crop_subset_[0-9]", 'AllCrop',
                          img_name)

        # skip if already exists
        if not os.path.exists(new_path):
            continue

        celeb_name = img_name2celeb[img_name]

        cropped_img_list.append((celeb_name, new_path))

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

    features = extractor.extract(img_list, batch_size=32)

    prefix_len = len("AllCrop") + 1

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
    df = pd.DataFrame(data, columns=['celeb_name', 'image_name', 'feature'])

    # 保存到 Parquet 文件，确保 pyarrow 已安装
    df.to_parquet(output_parquet_path, index=False)

    # 我这里希望实现一个parquet文件，每一行都是celeb_name ,image_name和feature 现在循环中的feature是一个numpy格式的vector


class ImageSearch:
    def __init__(self):
        self.model_oai = CLIPModel.from_pretrained(MODEL_OAI)
        self.model_oai_processor = CLIPProcessor.from_pretrained(MODEL_OAI)
        self.extractor = CLIPExtractor(model_name=MODEL_OAI, processor_name=MODEL_OAI)
        self.detector = MPCropAndNorm()

    def img2embed(self, image_path):
        image = Image.open(image_path)
        inputs = self.model_oai_processor(text=["dummy"], images=image, return_tensors="pt", padding=True)
        outputs = self.model_oai(**inputs)
        embed = outputs["image_embeds"]

        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        # print("\033[0;30;43m outputs: \033[0m ", logits_per_image, '\n')
        print("\033[0;30;43m outputs: \033[0m ", outputs.image_embeds.shape, '\n')
        return embed

    def img2vec(self, data):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_OAI, device=DEVICE)
        embedding = model.encode(data)
        if_normalization = True
        if if_normalization:
            sentence_embeddings = embedding / (1e-10 + np.linalg.norm(embedding))
        # embedding = embedding + 1e-10
        # sentence_embeddings = embedding / np.linalg.norm(embedding, axis=0)
        return sentence_embeddings.tolist()

    def cal_similarity_openai(self, image_path_1, image_path_2):
        image_1 = Image.open(image_path_1)
        image_2 = Image.open(image_path_2)

        inputs_1 = self.model_oai_processor(text=["dummy"], images=image_1, return_tensors="pt", padding=True)
        inputs_2 = self.model_oai_processor(text=["dummy"], images=image_2, return_tensors="pt", padding=True)

        outputs_1 = self.model_oai(**inputs_1)
        outputs_2 = self.model_oai(**inputs_2)

        similarity_score = torch.nn.functional.cosine_similarity(outputs_1["image_embeds"],
                                                                 outputs_2["image_embeds"]).item()
        return similarity_score

    def calculate_similarity_lda(self, image_paths, method):
        if method == "identity":
            model_path = MODEL_ID
        elif method == "expression":
            model_path = MODEL_EXP
        else:
            raise ValueError("Invalid method. Choose 'id' or 'exp'.")

        with open(model_path, 'rb') as f:
            lda_processor = pickle.load(f)

        images = [cv2.imread(path) for path in image_paths]
        cropped_faces = [self.detector.crop_and_norm(image)[0] for image in images]

        raw_clip_features = [self.extractor.extract([face])[0] for face in cropped_faces]

        lda_features = [lda_processor.transform([feature])[0] for feature in raw_clip_features]

        similarity_score = cosine_similarity([lda_features[0]], [lda_features[1]])[0][0]
        return similarity_score

    def cal_similarity_lda(self, image_path_1, image_path_2, method):
        if method == "identity":
            MODEL_ = MODEL_ID
        if method == "expression":
            MODEL_ = MODEL_EXP
        with open(MODEL_, 'rb') as f:
            self.simple_processor = pickle.load(f)
        image1 = cv2.imread(image_path_1)
        image2 = cv2.imread(image_path_2)

        try:
            faces_1 = self.detector.crop_and_norm(image1)
            faces_2 = self.detector.crop_and_norm(image2)
            cropped_face_1 = faces_1[0]
            save_name1 = tmp_dir + path_to_name(image_path_1) + "_cropped_face.jpg"
            cv2.imwrite(save_name1, cropped_face_1)

            cropped_face_2 = faces_2[0]
            save_name2 = tmp_dir + path_to_name(image_path_2) + "__cropped_face.jpg"
            cv2.imwrite(save_name2, cropped_face_2)
            features_1 = self.extractor.extract([save_name1])
            raw_clip_feature_1 = features_1[0]
            features_2 = self.extractor.extract([save_name2])
            raw_clip_feature_2 = features_2[0]

            def project_to_lda(feature):

                return self.simple_processor.transform([feature])

            face_1 = project_to_lda(raw_clip_feature_1)
            face_2 = project_to_lda(raw_clip_feature_2)

            similarity_score = cosine_similarity(face_1, face_2)[0]
            return similarity_score[0]
        except:
            print("\033[0;30;43m image_path_1 : \033[0m ", path_to_name(image_path_1))
            print("\033[0;30;43m image_path_2 : \033[0m ", path_to_name(image_path_2))
            print(f"IndexError: Could not calculate similarity for {image_path_1} and {image_path_2}")
            with open(ROOT_DIR + 'error_log_' + model_method + '.txt', 'a') as f:
                f.write(image_path_1 + '\n')
                f.write(image_path_2 + '\n')
            return -1000

    def cal_feature(self, image_path, method):
        MODEL_ = None
        if method == "identity":
            MODEL_ = MODEL_ID
        if method == "expression":
            MODEL_ = MODEL_EXP
        if MODEL_ is not None:
            with open(MODEL_, 'rb') as f:
                self.simple_processor = pickle.load(f)
            #

            # try:

            if crop_mode:
                image = cv2.imread(image_path)
                faces_ = self.detector.crop_and_norm(image)
                cropped_face_ = faces_[0]
                save_name = tmp_dir + path_to_name(image_path) + "_cropped_face.jpg"
                cv2.imwrite(save_name, cropped_face_)
                features_ = self.extractor.extract([save_name])
            else:
                features_ = self.extractor.extract([image_path])
            raw_clip_feature_ = features_[0]

            def project_to_lda(feature):
                if_normalization = True  # l2 normalization for each feature
                if if_normalization:
                    feature = feature / (1e-10 + np.linalg.norm(feature))
                return self.simple_processor.transform([feature])

            face_ = project_to_lda(raw_clip_feature_)
            # print("\033[0;30;43m =========face_.shape: =============\033[0m ",type(face_),face_.shape)
            return face_

            # except:
            #     print("\033[0;30;43m image_path : \033[0m ", path_to_name(image_path))
            #     # print("\033[0;30;43m image_path_2 : \033[0m ", path_to_name(image_path_2))
            #     with open(ERROR_TXT, 'a') as f:
            #         f.write( method + path_to_name(image_path)+ '\n')
            #     return np.empty((1, 512), dtype=np.float32)




def img2feature(image_path, method):
    image_name = path_to_name(image_path)
    print(f"\033[0;30;43m img_name: \033[0m {image_name}")

    feature_ndarray = ImageSearch().cal_feature(image_path, method)

    return feature_ndarray


def image_search_and_save(img_list, comfy_output_dir, output_parquet_path, method, batch_size):
    data = []
    json_file_path = os.path.splitext(output_parquet_path)[0] + ".json"
    for i in tqdm(range(0, len(img_list), batch_size), desc="Processing images"):
        batch_img_names = img_list[i:i + batch_size]
        batch_features = []

        for img_namess in tqdm(batch_img_names, desc="Processing batch images"):
            image_path = os.path.join(comfy_output_dir, img_namess)
            image_name = path_to_name(image_path)
            print(f"\033[0;30;43m img_name: \033[0m {image_name}")

            feature_ndarray = ImageSearch().cal_feature(image_path, method)
            feature = feature_ndarray.tolist()
            print(f"\033[0;30;43m  type : \033[0m {type(feature)}")

            if isinstance(feature, list):
                feature = feature[0]
                print(f"\033[0;30;43m list len : \033[0m {len(feature)}")
            else:
                print(f"\033[0;30;43m shape : \033[0m {feature.shape}")

            batch_features.append([image_name, feature, method])

        # 将batch结果保存到 JSON 文件
        with open(json_file_path, "a") as f:
            for row in batch_features:
                json.dump(row, f)
                f.write("\n")

        # 释放内存
        del batch_features
        import gc
        gc.collect()

    # 将 JSON 数据转换为 Parquet 格式
    df = pd.DataFrame(data, columns=['image_name', 'feature', 'model'])
    df.to_parquet(output_parquet_path, index=False)


def cosine_similarity0(a: Tensor, b: Tensor):
    """
    computes similarity between two embeddings
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    score = torch.mm(a_norm, b_norm.transpose(0, 1)).item()
    return score


def find_top_k_similar(input_embedding: Tensor, embeddings_df: pd.DataFrame, k: int) -> Tuple[
    pd.DataFrame, pd.Series]:
    """
    Find the top k embeddings from the DataFrame that are most similar to the input embedding.

    Args:
        input_embedding (Tensor): Input embedding tensor.
        embeddings_df (pd.DataFrame): DataFrame containing the embeddings to compare with.
        k (int): Number of top similar embeddings to return.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrame containing the top k similar embeddings, and a Series containing the corresponding cosine similarities.
    """

    #
    # similarities = embeddings_df["feature"].apply(
    #     lambda x: cosine_similarity(input_embedding, torch.tensor(x)))
    similarities = embeddings_df["feature"].apply(
        lambda x: cos_dot(input_embedding, x))

    top_k_indices = similarities.nlargest(k).index
    top_k_df = embeddings_df.loc[top_k_indices, ["celeb_name", "image_name"]]
    top_k_df["similarity"] = similarities[top_k_indices]
    return top_k_df[["celeb_name", "image_name", "similarity"]].reset_index(drop=True)  # Rearrange and drop index


def utils_cos_sim(a: Tensor, b: Tensor):
    """
    computes similarity between two embeddings
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    score = torch.mm(a_norm, b_norm.transpose(0, 1)).item()
    return score


def cos_dot(array1, array2):
    array1 = np.squeeze(array1)
    dot_product = np.dot(array1, array2)
    norm_a = np.linalg.norm(array1)
    norm_b = np.linalg.norm(array2)
    similarity = dot_product / (norm_a * norm_b)
    # print("Cosine Similarity:", similarity)
    return similarity


if __name__ == '__main__':
    # print("\033[0;30;43m after: \033[0m ", "", '\n')
    print("=".center(60, "="))
    print("\n \033[0;7m ---------------- \033[0m")
    # folder_path= comfy_output_dir
    folder_path = "/home/fyy/lilynRepos/embed4face/aria_exp/data/原图Identity/"

    print("\n \033[0;7m ---------------- \033[0m")

    # img_list = ['Result_2_00023_.png', 'Result_13_00018_.png']
    #

    print("\n \033[0;7m ---------------- \033[0m")

    # image_search_and_save(img_list, comfy_output_dir, output_parquet_path, method,batch_size=6)
    method = "identity"


    print(f"\n \033[0;7m ------------------------------- \033[0m")


    path1 = "tmp/identity_original_map_clipall.parquet"
    path2 = "tmp/identity_original_map.parquet"
    output_merged_path = path1
    df = pd.read_parquet(output_merged_path)
    df = df[['celeb_name', 'image_name', 'feature']]
    print("df : ", df.columns.values.tolist(), df.head(1)["feature"])

    print(f"\n \033[0;7m ------------------------{method}模型 embedding------------------------ \033[0m")
    # top_k_df = find_top_k_similar(input_embedding , df, k=3)
    # print(top_k_df)
    img_list = []
    for img_path in tqdm(img_list, desc="Processing images"):
        print("\033[0;30;43m img_path: \033[0m ", path_to_name(img_path))
        input_embedding = ImageSearch().cal_feature(img_path, method)
        top_k_df = find_top_k_similar(input_embedding, df, k=3)
        print(top_k_df)
        with open('/home/fyy/lilynRepos/embed4face/aria_exp/output/tmp.txt', 'a') as f:
            f.write(path_to_name(img_path) + '\n')
            f.write(top_k_df.to_string() + '\n\n')
