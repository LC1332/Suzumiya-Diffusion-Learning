#!/usr/bin/env python
# -*- coding:utf-8 -*-


ID_PATH = "one_clip_per_identity.parquet"
MODEL_PATH = "openai/clip-vit-base-patch16"
EMBED_IMG_MODEL_identity = RESOURCES_DIR + "lda_openai_clip_model.pkl"
EMBED_IMG_MODEL_expression = RESOURCES_DIR + "RAF_clip_LDA.pkl"
model_paths = {
    "id": EMBED_IMG_MODEL_identity,
    "exp": EMBED_IMG_MODEL_expression
}