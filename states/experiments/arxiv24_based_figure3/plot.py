import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from states.analysis.utils import fetch_wandb_runs

FEATURE_MAPS = {
    "zoology.mixers.feature_maps.taylor.TaylorExp": "Taylor",
    "zoology.mixers.feature_maps.base.Identity": "Identity",
    "zoology.mixers.feature_maps.base.PosELU": "PosELU",
    "zoology.mixers.feature_maps.base.ReLU": "ReLU",
    "zoology.mixers.feature_maps.base.Square": "Square",
    "zoology.mixers.feature_maps.cosformer.CosFormerFeatureMap": "CosFormer",
    "zoology.mixers.feature_maps.performer.PerformerFeatureMap": "Performer",
    "all_poly": "All Poly",
}


def plot(
    df: pd.DataFrame,
    metric: str="valid/num_kv_pairs/accuracy-256",
    run_dir: str="."
):
    feature_col = "model.sequence_mixer.kwargs.configs.1.kwargs.feature_name"
    df["feature_map"] = df[feature_col].map(FEATURE_MAPS)

    idx = df.groupby(
        ["num_parameters", "state_size", "feature_map"]
    )[metric].idxmax(skipna=True).dropna()
    plot_df = df.loc[idx]


    for x_col in ["num_parameters", "state_size"]:
    
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
        g.set(
            ylabel="Accuracy", 
            xlabel="Parameter Count" if x_col == "num_parameters" else "State Size"
        )
        g.ax.set_xscale("log", base=2)

        path = os.path.join(run_dir, f"results_{x_col}.png")
        plt.savefig(path)
        print(f"Saved to {path}")


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

    run_dir = os.path.join(
        os.path.dirname(__file__),
        "outputs"
    )
    os.makedirs(run_dir, exist_ok=True)
    plot(df=df, run_dir=run_dir)

