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
    data_key="data.input_seq_len",
    model_key="model.sequence_mixer.1.name",
    x_lab="Sequence Length",
):
    
    plot_df = df.groupby([
        model_key,
        "model.d_model",
        data_key,
    ])["valid/accuracy"].max().reset_index()

    run_dir = "/var/cr05_data/sim_data/code/petting-zoo/"

    # remove nan 'valid/accuracy' 
    plot_df = plot_df[~plot_df["valid/accuracy"].isna()] 
    
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df[plot_df[data_key] <= max_seq_len],
        y="valid/accuracy",
        col=data_key,
        x="model.d_model",
        hue=model_key,
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
        ax.set_title(f"{x_lab}: {title}")


if __name__ == "__main__" :
    experiment = 'no_random'
    experiment = 'random'
    experiment = 'num_kvs'
    experiment = 'num_heads'

    for experiment in ['no_random', 'num_kvs', 'num_heads']:  #'random', 

        launch_ids = []
        if experiment == 'no_random':
            launch_ids = [
                "default-2023-10-25-22-20-38",    # MHA
                "default-2023-10-26-19-09-31",    # Hyena
                "default-2023-10-27-04-13-56",
                "default-2023-10-29-17-31-26",
                "default-2023-11-12-00-31-44",
                "default-2023-11-13-00-31-15",
                "default-2023-11-13-00-42-27",
                
                "default-2023-12-09-21-02-38",
                "default-2023-12-12-04-47-47",

                # based 
                "default-2023-12-06-05-56-13",
                "default-2023-12-06-06-26-18",
                "default-2023-12-06-06-29-22"
            ]
        elif experiment == 'random':
            launch_ids = [
                "default-2023-12-10-08-47-05",
                "default-2023-12-10-08-39-31",
            ]
        elif experiment == 'num_kvs':
            launch_ids = [
                "default-2023-12-10-09-20-16",
                "default-2023-12-10-09-20-48",
                "default-2023-12-10-08-47-05",
                "default-2023-12-10-08-39-31",

                "default-2023-12-11-17-34-25",
            ]
        elif experiment == 'num_heads':
            launch_ids = [
                "default-2023-12-10-09-22-50",
                "default-2023-12-10-09-21-52",
                "default-2023-12-10-09-21-26",
                "default-2023-12-10-09-19-19",
                # "default-2023-10-26-19-09-31",   # Hyena

                "default-2023-12-11-07-42-08",
                "default-2023-12-11-17-50-00",
                "default-2023-12-12-04-19-03",
                "default-2023-12-12-04-18-50",
            ]
        
        df = fetch_wandb_runs(
            launch_id=launch_ids, 
            project_name="zoology"
        )

        model_key1 = "model.sequence_mixer.name"
        model_key2 = "model.sequence_mixer.1.name"

        data_key1 = "data.input_seq_len"
        data_key2 = 'data.0.input_seq_len'
        # set keys
        if experiment in ['random', 'num_kvs']:
            data_key = data_key2
            model_key = model_key1
            df[f"{model_key1}"] = df[f"{model_key1}"].fillna(df[f"{model_key2}"])
        elif experiment in ['num_heads']:
            data_key = data_key2
            head_key = "model.sequence_mixer.kwargs.num_heads"        
            df[f"{head_key}"] = df[f"{head_key}"].fillna(1)
            df['model_name'] = df[f'{model_key1}'].astype(str) + df[f'{head_key}'].astype(str)
            model_key = 'model_name'
        else:
            model_key = model_key1
            data_key = data_key1
            df[f"{model_key1}"] = df[f"{model_key1}"].fillna(df[f"{model_key2}"])

            
        # set plotting info
        if experiment in ['num_kvs']:
            x_key = "data.0.builder.kwargs.num_kv_pairs"
            x_lab = "Num. KV Pairs"
            df[f"{data_key}"] = df[f"{data_key}"].fillna(df[f"{data_key}"])
        elif experiment in ['random', 'num_heads']:
            x_key = data_key
            x_lab = "Sequence Length"
            try:
                df[f"{data_key}"] = df[f"{data_key}"].fillna(df[f"{data_key1}"])
            except:
                pass
            df[f"{data_key}"] = df[f"{data_key}"].fillna(df[f"{data_key}"])
            
        else:
            x_key = data_key
            x_lab = "Sequence Length"
            try:
                df[f"{data_key1}"] = df[f"{data_key1}"].fillna(df[f"{data_key2}"])
            except:
                pass

        plot(df=df, max_seq_len=1024, data_key=x_key, model_key=model_key, x_lab=x_lab)
        print(f"Length of DF = {len(df)}")
        output_file = f"results_{experiment}.png"
        print(f"{output_file}")
        plt.savefig(output_file)
