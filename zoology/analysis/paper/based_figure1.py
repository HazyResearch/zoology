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

    idx = df.groupby(
        ["state_size", "model.name"]
    )[metric].idxmax(skipna=True).dropna()
    plot_df = df.loc[idx]

    # upper case the model names first letter
    plot_df["model.name"] = plot_df["model.name"].str.capitalize()
    # replace "-" and "_" with " "
    plot_df["model.name"] = plot_df["model.name"].str.replace("-", " ")
    plot_df["model.name"] = plot_df["model.name"].str.replace("_", " ")
    # replace model column name with "Model"
    plot_df = plot_df.rename(columns={"model.name": "Model"})

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        y=metric,
        x="state_size",
        hue="Model",
        kind="scatter",
        marker="o",
        height=5,
        aspect=1,
    )
    g.set(xscale="log", ylabel="MQAR Accuracy", xlabel="State Size")

if __name__ == "__main__" :
    df = fetch_wandb_runs(
        launch_id=[
            # "default-2024-02-09-04-11-25"
            "default-2024-02-09-05-44-06",
            "default-2024-02-09-14-59-58",
            "default-2024-12-28-14-12-35",
        ], 
        project_name="zoology"
    )

    df2 = fetch_wandb_runs(
        launch_id=[
            # Adding RWKV-v7
            "default-2025-03-04-16-43-26",
            "default-2025-03-04-15-55-12",
            "default-2025-03-04-15-11-23"

            # Adding NSA

            # Adding DeltaNet
            "default-2025-03-05-14-30-11",
            "default-2025-03-05-14-07-18",
            "default-2025-03-05-14-59-58",

            # Adding Gated DeltaNet
        ], 
        project_name="0325_zoology"
    )

    # add the new runs to the df
    df = pd.concat([df, df2]).reset_index(drop=True)

    plot(df=df)

    plt.savefig("results.png")
    print("results.png")


