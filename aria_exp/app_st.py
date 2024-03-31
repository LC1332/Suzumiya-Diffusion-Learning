#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import requests
import streamlit as st
from PIL import Image
from config import *
import numpy as np
from img_embed import ImgEmbed
from img_search import ImgSearch
import pandas as pd
import cv2



@st.cache_resource()
def get_input_embedding(IMG_PATH):
    id_embedding = ImgEmbed(model_paths).img2id_embed(IMG_PATH)
    input_embedding = id_embedding[0]
    return input_embedding


def find_similar(image, k):
    df = pd.read_parquet(ID_PATH)
    input_embedding = get_input_embedding(image)
    top_k_df = ImgSearch().find_top_k_similar(input_embedding, df, k=int(k))
    return top_k_df


st.sidebar.markdown('## AI')
genre = st.sidebar.radio('Fuctions', ('identy', 'expression'))
st.sidebar.markdown('## paras')
k = st.sidebar.slider("topk", 1, 10, 2, 1)
uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg', 'tif'])
if genre == 'identy':
    if uploaded_file is not None:
        # 将文件对象转换为字节对象
        bytes_data = uploaded_file.getvalue()

        # 将字节对象转换为 OpenCV 图像对象
        arr = np.frombuffer(bytes_data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # 将 OpenCV 图像对象转换为 PIL 图像对象
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        st.sidebar.image(pil_image, caption=f"Uploaded Image", use_column_width=True)
        # 保存图像到本地文件夹
        save_folder = "uploaded_images"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        save_path = os.path.join(save_folder, uploaded_file.name)
        pil_image.save(save_path)

        input_embedding = get_input_embedding(save_path)
        top_k_df = ImgSearch().find_top_k_similar(input_embedding, df, k=int(k))
        st.dataframe(top_k_df)