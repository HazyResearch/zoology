import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs




def plot(
    df: pd.DataFrame,
    max_seq_len: int = 512,
):
    
    plot_df = df.groupby([
        "model.sequence_mixer.name",
        "model.d_model",
        "data.input_seq_len",
    ])["valid/accuracy"].max().reset_index()

    run_dir = "/var/cr05_data/sim_data/code/petting-zoo/"
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df[plot_df["data.input_seq_len"] <= max_seq_len],
        y="valid/accuracy",
        col="data.input_seq_len",
        x="model.d_model",
        hue="model.sequence_mixer.name",
        kind="line",
        marker="o",
        height=2.25,
        aspect=1,
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="Model dimension")

    # Set custom x-ticks
    ticks = [64, 128, 256, 512] # Modify this list as needed
    for ax in g.axes.flat:
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
        sweep_id=["fact_retrievale430da"], 
        project_name="zoology"
    )

    plot(df=df, max_seq_len=1024)
    plt.savefig("results.pdf")
