from pathlib import Path
from typing import List

import datasets.table
import polars as pl
import datasets


def extract_table(ds: datasets.DatasetDict, tables: List[str]) -> pl.DataFrame:
    """
    Convert a datasets.Dataset to a polars.DataFrame
    """
    dfs = []

    for table in tables:
        ds_dict = ds[table].to_dict()
        ds_df = pl.DataFrame(ds_dict)
        dfs.append(ds_df)

    return pl.concat(dfs, how="vertical")


if __name__ == "__main__":

    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()

    ds = datasets.load_dataset("SetFit/bbc-news")

    df = extract_table(ds, ["train", "test"])

    # keep docs in middle range of length to avoid fitting to length
    df = df.with_columns(
        pl.col("text").str.len_chars().alias("doc_length"),
    )
    quantile_low = df["doc_length"].quantile(0.2)
    quantile_hi = df["doc_length"].quantile(0.8)
    print(f"Keep docs between length {quantile_low} and {quantile_hi}")
    df = df.filter(df["doc_length"].is_between(quantile_low, quantile_hi))

    print(df["label_text"].value_counts())

    df.write_parquet(data_dir / "bbc.parquet")
