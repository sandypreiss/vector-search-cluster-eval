from typing import List
from fire import Fire
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from umap import UMAP
import zarr


def reduce_dims(embeddings, n_components: int) -> np.ndarray:
    reducer = UMAP(n_components=n_components)
    return reducer.fit_transform(embeddings)


def embed(model_name: str, docs: List[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs)
    return np.vstack(embeddings)


def main(model_name: str, n_components: int):
    """
    asdf

    Args:
        model_name (str): _description_
    """
    df = pl.read_parquet("data/bbc.parquet")
    docs = df["text"].to_list()
    embeddings = embed(model_name, docs)
    reduced = reduce_dims(embeddings, n_components)
    zarr.save("data/embeddings.zarr", reduced)


if __name__ == "__main__":
    Fire(main)
