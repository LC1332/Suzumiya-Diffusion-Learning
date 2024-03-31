#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
__info: " img search"

"""

import pandas as pd
from typing import Tuple
import torch
from torch import Tensor
from config import ID_PATH 


class ImgSearch(object):
    def __init__(self):
        pass

    def load_embeddings(self, parquet_file: str) -> pd.DataFrame:
        """
        Load embeddings from a Parquet file.

        Args:
            parquet_file (str): Path to the Parquet file.

        Returns:
            pd.DataFrame: DataFrame containing the embeddings.

        """
        ID_PATH = parquet_file
        df = pd.read_parquet(ID_PATH)
        return df

        """
        Compute the cosine similarity between two embeddings.

        Args:
            a (Tensor): First embedding tensor.
            b (Tensor): Second embedding tensor.

        Returns:
            float: Cosine similarity between the embeddings.
        """
        a_norm = torch.nn.functional.normalize(a, p=2, dim=0)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=0)
        return torch.dot(a_norm, b_norm).item()

    def cosine_similarity(self, a: Tensor, b: Tensor):
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

    def find_top_k_similar(self, input_embedding: Tensor, embeddings_df: pd.DataFrame, k: int) -> Tuple[
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
        similarities = embeddings_df["feature"].apply(
            lambda x: self.cosine_similarity(input_embedding, torch.tensor(x)))
        top_k_indices = similarities.nlargest(k).index
        top_k_df = embeddings_df.loc[top_k_indices, ["celeb_name", "image_name", "feature"]]
        top_k_df["similarity"] = similarities[top_k_indices]
        return top_k_df[["celeb_name", "image_name", "similarity"]].reset_index(drop=True)  # Rearrange and drop index


if __name__ == '__main__':
    from utils import utils_cos_sim
    import pandas as pd
    df = pd.read_parquet(ID_PATH)
    input_embedding = id_embedding[0]
    top_k_df = ImgSearch().find_top_k_similar(input_embedding, df, k=2)
    print(top_k_df)
