import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs

FEATURE_MAPS = {
    "zoology.mixers.feature_maps.taylor.TaylorExp": "Taylor",
    "zoology.mixers.feature_maps.base.Identity": "Identity",
    "zoology.mixers.feature_maps.base.PosELU": "PosELU",
    "zoology.mixers.feature_maps.base.ReLU": "ReLU",
    "zoology.mixers.feature_maps.base.Square": "Square",
    "zoology.mixers.feature_maps.cosformer.CosFormerFeatureMap": "CosFormer",
    "zoology.mixers.feature_maps.performer.PerformerFeatureMap": "Performer",
}


def plot(
    df: pd.DataFrame,
    metric: str="valid/num_kv_pairs/accuracy-256",
):
    feature_col = "model.sequence_mixer.kwargs.configs.1.kwargs.feature_name"
    df["feature_map"] = df[feature_col].map(FEATURE_MAPS)


    for x_col in ["num_parameters", "state_size"]:
        idx = df.groupby(
            [x_col, "feature_map"]
        )[metric].idxmax(skipna=True).dropna()
        plot_df = df.loc[idx]

        sns.set_theme(style="whitegrid")
        g = sns.relplot(
            data=plot_df,
            y=metric,
            x=x_col,
            hue="feature_map",
            kind="line",
            marker="o",
            height=5,
            aspect=1,   
        )
        g.set(xscale="log", ylabel="Accuracy", xlabel="Parameter Count" if x_col == "num_parameters" else "State Size")

        plt.savefig(f"results_{x_col}.png")
        print(f"Saved results_{x_col}.png")

if __name__ == "__main__" :
    df = fetch_wandb_runs(
        launch_id=["default-2024-02-14-23-41-12"],
        project_name="zoology"
    )

    # filter out some of the taylor runs
    df = df[
        (df["model.sequence_mixer.kwargs.configs.1.kwargs.feature_dim"] != 16) |
        (df["model.sequence_mixer.kwargs.configs.1.kwargs.feature_name"] != "zoology.mixers.feature_maps.taylor.TaylorExp")
    ]

    plot(df=df)

