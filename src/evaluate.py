import polars as pl
from pynndescent import NNDescent
import numpy as np
import json
import zarr
import fire
from sentence_transformers import SentenceTransformer


def load_cluster_names() -> pl.DataFrame:
    with open("data/cluster_names.json") as f:
        names = json.load(f)
    rows = []
    for model, clusters in names.items():
        for cluster, name in clusters.items():
            rows.append({"model": model, "cluster": cluster, "name": name})
    return pl.DataFrame(rows)


def embed_cluster_names(model_name: str, df: pl.DataFrame, text_col: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df[text_col].to_list(), prompt_name="query")
    return np.vstack(embeddings)


def get_cluster_counts(df: pl.DataFrame) -> pl.DataFrame:
    df = pl.read_parquet("data/bbc_clustered.parquet")

    cluster_cols = [col for col in df.columns if col.startswith("cluster_from_")]

    counts = pl.DataFrame()

    for col in cluster_cols:
        cluster_count = (
            df[col]
            .value_counts()
            .rename({col: "cluster"})
            .with_columns(
                pl.lit(col.replace("cluster_from_", "")).alias("model"),
                pl.col("cluster").cast(str),
            )
        )
        counts = (
            counts.vstack(cluster_count) if not counts.is_empty() else cluster_count
        )

    return counts


def get_true_labels(df: pl.DataFrame, label_col: str) -> pl.DataFrame:
    label_counts = (
        df[label_col]
        .value_counts()
        .rename({label_col: "name"})
        .with_columns(pl.lit("true_labels").alias("model"))
    )
    label_lookup = (
        df["label", "label_text"]
        .unique()
        .rename({"label": "cluster"})
        .with_columns(pl.col("cluster").cast(str))
    )

    return label_counts.join(label_lookup, left_on="name", right_on="label_text")


def get_cluster_indices(df: pl.DataFrame, cluster_names: pl.DataFrame) -> dict:
    rename_cols = {col: col.replace("cluster_from_", "") for col in df.columns}
    rename_cols["label"] = "true_labels"
    df = df.rename(rename_cols).with_row_index()
    cluster_indices = {}

    for row in cluster_names.iter_rows(named=True):
        dff = df.filter(pl.col(row["model"]) == int(row["cluster"]))
        cluster_indices[row["index"]] = dff["index"].to_numpy()

    return cluster_indices


def calculate_metrics(
    docs_in_clusters: dict[int, np.ndarray], query_results: dict[int, np.ndarray]
) -> pl.DataFrame:
    rows = []

    for i, cluster_indices in docs_in_clusters.items():
        query_indices = query_results[i]

        true_positives = np.intersect1d(cluster_indices, query_indices)
        false_positives = np.setdiff1d(query_indices, cluster_indices)
        false_negatives = np.setdiff1d(cluster_indices, query_indices)

        recall = len(true_positives) / (len(true_positives) + len(false_negatives))
        try:
            precision = len(true_positives) / (
                len(true_positives) + len(false_positives)
            )
        except ZeroDivisionError:
            precision = 0
        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0

        rows.append(
            {
                "index": i,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pl.DataFrame(rows).with_columns(pl.col("index").cast(pl.datatypes.UInt32))


def main(embedding_model_name: str, distance_threshold: float):
    cluster_names = load_cluster_names()
    df = pl.read_parquet("data/bbc_clustered.parquet")
    cluster_counts = get_cluster_counts(df)
    cluster_names = cluster_names.join(cluster_counts, on=["model", "cluster"])
    true_labels = get_true_labels(df, "label_text")
    cluster_names = pl.concat(
        [cluster_names, true_labels], how="diagonal"
    ).with_row_index()

    name_embeddings = embed_cluster_names(embedding_model_name, cluster_names, "name")

    doc_embeddings = zarr.open("data/embeddings.zarr")[:]
    index = NNDescent(doc_embeddings, metric="cosine")

    # To do the query in one batch, get the most neighbors we'll need for any cluster
    # for all of them. Then filter to the correct number per cluster and apply the
    # distance threshold later.
    k = cluster_names["count"].max()
    nn_indices, nn_distances = index.query(name_embeddings, k=k)

    count_mask = np.zeros_like(nn_indices, dtype=bool)
    for i, count in enumerate(cluster_names["count"]):
        count_mask[i, :count] = True

    distance_mask = nn_distances < distance_threshold

    mask = count_mask & distance_mask

    query_results = {
        i: row[mask_row] for i, (row, mask_row) in enumerate(zip(nn_indices, mask))
    }

    docs_in_clusters = get_cluster_indices(df, cluster_names)

    metrics = calculate_metrics(docs_in_clusters, query_results)
    cluster_names = cluster_names.join(metrics, on="index")

    macro_avg = cluster_names.group_by("model").agg(
        pl.col("precision").mean().alias("macro_avg_precision"),
        pl.col("recall").mean().alias("macro_avg_recall"),
        pl.col("f1").mean().alias("macro_avg_f1"),
    )

    cluster_names = cluster_names.with_columns(
        (pl.col("count") / len(df)).alias("weight")
    )
    cluster_names = cluster_names.with_columns(
        (pl.col("precision") * pl.col("weight")).alias("weighted_precision"),
        (pl.col("recall") * pl.col("weight")).alias("weighted_recall"),
        (pl.col("f1") * pl.col("weight")).alias("weighted_f1"),
    )
    weighted_avg = cluster_names.group_by("model").agg(
        pl.col("weighted_precision").sum().alias("weighted_avg_precision"),
        pl.col("weighted_recall").sum().alias("weighted_avg_recall"),
        pl.col("weighted_f1").sum().alias("weighted_avg_f1"),
    )

    results = macro_avg.join(weighted_avg, on="model")

    cluster_names.write_parquet("data/scored_cluster_names.parquet")
    results.write_parquet("data/eval_results.parquet")


if __name__ == "__main__":
    fire.Fire(main)
