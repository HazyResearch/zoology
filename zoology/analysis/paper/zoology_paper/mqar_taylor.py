import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs


model_tag_2_name = {
    "zoology.mixers.based.Based": "Linear Attn.",
}

def plot(
    df: pd.DataFrame,
    max_seq_len: int = 512,
):
    
    plot_df = df.groupby([
        "model.sequence_mixer.name",
        "model.d_model",
        "data.input_seq_len",
        "model.sequence_mixer.kwargs.feature_dim",
        "model.sequence_mixer.kwargs.num_key_value_heads",
    ])["valid/accuracy"].max().reset_index()

    # Convert "model.sequence_mixer.name" to human-readable names
    plot_df['model'] = plot_df['model.sequence_mixer.name'].map(model_tag_2_name) + " (K=" + plot_df['model.sequence_mixer.kwargs.feature_dim'].astype(str) + ")"  + " (H=" + plot_df['model.sequence_mixer.kwargs.num_key_value_heads'].astype(str) + ")"


    run_dir = "/var/cr05_data/sim_data/code/clean/zoology/"
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df[plot_df["data.input_seq_len"] <= max_seq_len],
        y="valid/accuracy",
        col="data.input_seq_len",
        x="model.d_model",
        hue="model",
        kind="line",
        marker="o",
        height=2.25,
        aspect=1,
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="Model dimension")

    # Set custom x-ticks
    ticks = [64, 128, 256, 512] # Modify this list as needed
    for ax in g.axes.flat:
        ax.set_xticks([], minor=True)
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # This will keep the tick labels as integers rather than in scientific notation

    # Set custom y-ticks
    y_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    for ax in g.axes.flat:
        ax.set_yticks(y_ticks)

    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f"Sequence Length: {title}")


if __name__ == "__main__" :
    df = fetch_wandb_runs(
        launch_id=[
            # "default-2024-01-22-09-42-49",

            # 4 layers
            "default-2024-01-22-20-47-22",
            "default-2024-01-23-17-55-32",
            "default-2024-01-23-18-10-38",
            "default-2024-01-23-17-49-04",
            "default-2024-01-23-21-07-35",
            "default-2024-01-24-09-09-09",
        ], 
        project_name="zoology"
    )

    print(f"Found {len(df)} runs")

    # breakpoint()

    plot(df=df, max_seq_len=1024)
    plt.savefig("results.png")
