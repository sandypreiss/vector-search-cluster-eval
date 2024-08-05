import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt


if __name__ == "__main__":
    results = (
        pl.read_parquet("data/eval_results.parquet")
        .select(["model", "macro_avg_f1", "weighted_avg_f1"])
        .rename(
            {
                "model": "Model",
                "macro_avg_f1": "Macro F1",
                "weighted_avg_f1": "Weighted F1",
            }
        )
        .melt(id_vars="Model", variable_name="Metric", value_name="Score")
        .sort("Score", descending=True)
        .with_columns(
            pl.col("Model").str.replace("_", " ", literal=True, n=2).str.to_titlecase()
        )
    )

    fig = sns.barplot(data=results, x="Score", y="Model", hue="Metric", orient="h")

    fig.set_ylabel(None)

    plt.savefig("data/f1_plot.png", bbox_inches="tight", dpi=600)
