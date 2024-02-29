import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs




def plot(
    df: pd.DataFrame,
    metric: str="valid/accuracy",
):
    # model_columns = [c for c in df.columns if c.startswith("model.")]
    # for c in model_columns:
    #     df[c] = df[c].ffill()
    idx = df.groupby(
        ["state_size", "model.name"]
    )[metric].idxmax(skipna=True).dropna()
    plot_df = df.loc[idx]
    # plot_df = df


    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        y=metric,
        x="state_size",
        hue="model.name",
        kind="scatter",
        marker="o",
        height=5,
        aspect=1,
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="State Size")

    # # Set custom x-ticks
    # ticks = [64, 128, 256, 512] # Modify this list as needed
    # for ax in g.axes.flat:
    #     ax.set_xticks(ticks)
    #     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # This will keep the tick labels as integers rather than in scientific notation

    # # Set custom y-ticks
    # y_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    # for ax in g.axes.flat:
    #     ax.set_yticks(y_ticks)

    # for ax, title in zip(g.axes.flat, g.col_names):
    #     ax.set_title(f"Sequence Length: {title}")


if __name__ == "__main__" :
    df = fetch_wandb_runs(
        launch_id=[
            # "default-2024-02-09-04-11-25"
            "default-2024-02-09-05-44-06",
            "default-2024-02-09-14-59-58",
            
            "default-2024-02-09-22-11-46",
            "default-2024-02-09-22-35-00",
            "default-2024-02-09-23-19-31"
        ], 
        project_name="zoology"
    )

    # # df["data.input_seq_len"] = df["data.input_seq_len"].fillna(df["data.0.input_seq_len"])
    plot(df=df)

    plt.savefig("results.png")
    print("results.png")
