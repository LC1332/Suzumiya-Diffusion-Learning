#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
__info: "img to identity_embedding and expression_embedding"
id represents for identity
exp represents for expression
"""

import torch
from typing import Optional, Any
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import numpy as np
import cv2
from CLIPExtractor import CLIPExtractor
from MPCropAndNorm import MPCropAndNorm
import pandas as pd
import pickle
from PIL import Image
from config import *


from typing import Optional, Any
import logging

logging.basicConfig(level=logging.INFO)


class ImgEmbed:
    def __init__(self, model_path: dict[str, str]):
        self.model_path = model_path
        self.detector = MPCropAndNorm()
        self.extractor = CLIPExtractor()

    def _load_model(self, model_type: str) -> Optional[Any]:
        model_path = self.model_path.get(model_type)
        if model_path:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        logging.warning(f"Model path for {model_type} not found.")
        return None

    def _preprocess_image(self, img_path: str) -> np.ndarray:
        image = cv2.imread(img_path)
        # image = Image.open(img_path)
        faces = self.detector.crop_and_norm(image)
        cropped_face = faces[0]
        save_name = f"{img_path}_cropped_face.jpg"
        cv2.imwrite(save_name, cropped_face)
        return save_name

    def _extract_clip_feature(self, img_path: str) -> np.ndarray:
        features = self.extractor.extract([img_path])
        raw_clip_feature = features[0]
        return raw_clip_feature

    def _project_to_lda(self, feature: np.ndarray, lda_model) -> np.ndarray:
        return lda_model.transform([feature])

    def img2id_embed(self, img_path: str) -> list[float]:
        return self.ImgEmbed(img_path,"id")

    def ImgEmbed(self, img_path: str, model_type: str) -> list[float]:
        sav_name = self._preprocess_image(img_path)
        raw_clip_feature = self._extract_clip_feature(sav_name)
        idORexp_model = self._load_model(model_type)
        if idORexp_model:
            idORexp_feature = self._project_to_lda(raw_clip_feature, idORexp_model)
            print("\033[0;30;43m{}: \033[0m".format(model_type),  type(idORexp_feature), idORexp_feature.shape)
            return idORexp_feature
        return []

    def img2exp_embed(self, img_path: str) -> list[float]:
        return self.ImgEmbed(img_path,"exp")


if __name__ == '__main__':
    IMG_PATH = "1.png"
    ImgEmbed = ImgEmbed(model_paths)
    id_embedding = ImgEmbed.img2id_embed(IMG_PATH)
    print("id : ", id_embedding,"\n\n")
    exp_embedding = ImgEmbed.img2exp_embed(IMG_PATH)
    print(" exp:", exp_embedding)

