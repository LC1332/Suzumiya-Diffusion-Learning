#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
__info: "openai 原始模型 gr"
"""

import torch
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

temp_folder = "aria_exp/tmp"
MODEL_PATH = "openai/clip-vit-base-patch16"


class ImageSimilarity:
    def __init__(self, model_path):
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def calculate_similarity(self, image_path_1, image_path_2, text=["pic1", "pic2"]):
        image_1 = Image.open(image_path_1)
        image_2 = Image.open(image_path_2)

        inputs_1 = self.processor(text=text, images=image_1, return_tensors="pt", padding=True)
        inputs_2 = self.processor(text=text, images=image_2, return_tensors="pt", padding=True)

        outputs_1 = self.model(**inputs_1)
        outputs_2 = self.model(**inputs_2)
        print("len:", len(outputs_1["image_embeds"][0]))
        similarity_score = torch.nn.functional.cosine_similarity(outputs_1["image_embeds"],
                                                                 outputs_2["image_embeds"]).item()
        return similarity_score


def calculate_similarity(image1, image2):
    similarity_score = calculator.calculate_similarity(image1, image2)
    return similarity_score


calculator = ImageSimilarity(MODEL_PATH)
image1 = gr.Image(label="Image 1", type="filepath")
image2 = gr.Image(label="Image 2", type="filepath")

if __name__ == '__main__':
    gr.Interface(
        fn=calculate_similarity,
        inputs=[image1, image2],
        outputs="text",
        title="OAI Image Similarity",
        description=" "
    ).launch(share=True)