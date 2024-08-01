from sklearn.cluster import KMeans
import polars as pl
import zarr


if __name__ == "__main__":

    embeddings = zarr.load("data/embeddings.zarr")

    n_clusters = [2, 3, 4, 5, 6, 7, 8]
    n_cluster_labels = {}

    # The flat module in hdbscan allows you to extract a flat clustering (ie, a specific
    # number of clusters) from a fitted HDBSCAN model's cluster hierarchy.
    for n in n_clusters:
        print(f"Extracting {n} clusters from hierarchy")
        clusterer = KMeans(n_clusters=n, n_init=10, random_state=999)
        n_cluster_labels[f"cluster_from_{n}_cluster_model"] = clusterer.fit_predict(
            embeddings
        )

    labels_df = pl.from_dict(n_cluster_labels)

    df = pl.read_parquet("data/bbc.parquet")
    df = pl.concat([df, labels_df], how="horizontal")
    df.write_parquet("data/bbc_clustered.parquet")
