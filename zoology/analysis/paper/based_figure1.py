import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs


model2color = {
    "Hyena": "#BAB0AC",
    "H3": "#B07AA1",
    
    'Attention': "black", 
    'Sliding window attention': "black", 

    'Based': "#59A14F", # "#F28E2B"
    "Mamba2": "#4E79A7",
    'Gated delta net': "#9C755F", 
    'Rwkv7': "#EDC948", 
    'Mamba': "#76B7B2", 
    'Delta net': "#E15759", 
    'Gla': "#F28E2B",
}

order = [
    "Attention",
    "Sliding Window",
    "H3",
    "Hyena",
    "Based",
    "Mamba",
    "GLA",
    "Mamba-2",
    "DeltaNet",
    "RWKV-7",
    "Gated DeltaNet",
]

name_replacements = {
    "Mamba2": "Mamba-2",
    "Gla": "GLA",
    "Rwkv7": "RWKV-7",
    "Mamba": "Mamba",
    "Delta net": "DeltaNet",
    "Gated delta net": "Gated DeltaNet",
    "Sliding window attention": "Sliding Window",
}

def _normalize_model_key(s: str) -> str:
    """
    Normalize raw model names to match keys in name_replacements.
    We preserve hyphens as given in the raw name (e.g., Mamba2 -> Mamba2),
    but convert underscores to spaces and standardize case for lookup.
    """
    if not isinstance(s, str):
        return s
    s2 = s.strip().replace("_", " ")
    # Don't force hyphen removal—our replacements decide final punctuation.
    # Use case-insensitive lookup by lowercasing.
    return s2

def _apply_name_replacements(series: pd.Series) -> pd.Series:
    # case-insensitive map using provided name_replacements keys
    lower_map = {k.lower(): v for k, v in name_replacements.items()}
    return series.apply(lambda x: lower_map.get(_normalize_model_key(x).lower(), _normalize_model_key(x)))

def _mapped_palette(base_palette: dict, replacements: dict) -> dict:
    """
    Map your existing model2color keys through name_replacements so
    colors line up with the final display names.
    """
    lower_map = {k.lower(): v for k, v in replacements.items()}
    mapped = {}
    for k, color in base_palette.items():
        key_norm = _normalize_model_key(k)
        final_name = lower_map.get(key_norm.lower(), key_norm)
        mapped[final_name] = color
    return mapped


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
    plot_df["Model"] = _apply_name_replacements(plot_df["model.name"])
    # plot_df = plot_df.rename(columns={"model.name": "Model"})


    # (06/05) adjust the state sizes for rwkv v7
    rwkv_mask = (plot_df["Model"] == "Rwkv7")
    rwkv_mask_128 = (plot_df["Model"] == "Rwkv7") & (plot_df["model.d_model"] == 128)
    rwkv_mask_256 = (plot_df["Model"] == "Rwkv7") & (plot_df["model.d_model"] == 256)
    print(plot_df[['Model', 'state_size', 'model.d_model']][rwkv_mask_128 | rwkv_mask_256])
    plot_df.loc[rwkv_mask_128, "state_size"] /= 4
    plot_df.loc[rwkv_mask_256, "state_size"] /= 16
    print(plot_df[['Model', 'state_size', 'model.d_model']][rwkv_mask])

    palette = _mapped_palette(model2color, name_replacements)

    # sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        y=metric,
        x="state_size",
        hue="Model",
        kind="scatter",
        marker="o",
        hue_order=order,           # enforce your order
        height=5,
        aspect=1,
        palette=palette,
        s=60,
        edgecolor="black",    # <-- thin black border
        linewidth=0.5,        # <-- thickness of the border
    )
    g.set(xscale="log", ylabel="Recall Accuracy", xlabel="State Size (log scale)")

    ax = g.ax
    ax.set_xlabel("State Size (log scale)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Recall Accuracy", fontsize=16, fontweight="bold")
    if g._legend is not None:
        g._legend.set_title("Model", prop={"weight": "bold", "size": 16})

    #title
    ax.set_title("Recall-Memory Tradeoff", fontsize=16, fontweight="bold")


    ax = g.ax

    # --- Find the leftmost point (smallest state_size) with accuracy near 1.0 ---
    # tweak tolerance if needed (here: >= 0.99 accuracy)
    point = plot_df.loc[
        plot_df[metric] >= 0.99, ["state_size", metric]
    ].sort_values("state_size").iloc[0]

    x_val, y_val = point["state_size"], point[metric]

    # --- Draw vertical dashed line ---
    ax.axvline(
        x=x_val,
        ymin=0, ymax=y_val,  # scale 0–1 relative to axis
        linestyle="--",
        color="black",
        linewidth=1,
    )




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
            "default-2025-03-05-16-20-42",
            "default-2025-03-05-16-41-32",

            # Adding Gated Linear Attention (GLA)
            "default-2025-03-05-16-01-15",
        ], 
        project_name="0325_zoology"
    )

    # add the new runs to the df
    df = pd.concat([df, df2]).reset_index(drop=True)

    plot(df=df)

    # save in high resolution
    plt.savefig("results.png", dpi=300, bbox_inches="tight")
    print("results.png")


