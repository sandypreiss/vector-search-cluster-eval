from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
import polars as pl


if __name__ == "__main__":

    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir()

    data = fetch_20newsgroups(
        data_home=data_dir,
        subset="all",
        random_state=123,
        remove=("headers", "footers", "quotes"),
    )

    df = pl.DataFrame(
        {
            "text": data["data"],
            "class": data["target"],
        }
    )
    class_lookup = {i: name for i, name in enumerate(data["target_names"])}

    df = df.with_columns(pl.col("class").replace(class_lookup).alias("class_name"))

    df.write_parquet(data_dir / "20newsgroups.parquet")
