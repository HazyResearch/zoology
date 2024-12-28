################################
#  Compare quality as a function of wall clock time for different sequence mixers 
#  While state size is an important proxy, we can also directly put the wall clock time on the x-axis 
################################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from zoology.analysis.utils import fetch_wandb_runs


import torch
import time
import thunderkittens as tk
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from flash_attn import flash_attn_func
from fla.ops.based import fused_chunk_based
from flashfftconv import FlashFFTConv


from torch import nn
import torch.nn.functional as F
from einops import rearrange


def profile_mamba2(batch, seq_len, num_heads, head_dim, dstate, chunk_size=64, ngroups=1):
    dtype = torch.bfloat16
    device = torch.device("cuda")

    # Initialize tensors
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    dt = torch.randn(batch, seq_len, num_heads, dtype=dtype, device=device)
    A = torch.randn(num_heads, dtype=dtype, device=device)
    B = torch.randn(batch, seq_len, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seq_len, ngroups, dstate, dtype=dtype, device=device)

    # Warmup
    for _ in range(10):
        y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)
    
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = ( t1 - t0 ) / 10
    return elapsed


def profile_mamba1(batch, seq_len, num_heads, head_dim, dstate, chunk_size=64, ngroups=1):

    device = torch.device("cuda")
    b, h, n = batch, num_heads, seq_len
    dv = head_dim
    dstate = 16 # from Mamba paper, note state is 8x smaller then Based

    dmodel = dv*h*2
    A = torch.randn(dmodel, dstate, dtype=torch.float32, device=device)
    x = torch.randn(b, dmodel, n, dtype=torch.bfloat16, device=device)
    dt = torch.randn(b, dmodel,n, dtype=torch.bfloat16, device=device)    
    B = torch.randn(b, dstate, n, dtype=torch.bfloat16, device=device)
    C = torch.randn(b, dstate, n, dtype=torch.bfloat16, device=device)
    D = torch.randn(dmodel, dtype=torch.bfloat16, device=device)
    z = torch.randn(b, dmodel, n, dtype=torch.bfloat16, device=device)
    dt_proj_bias = torch.randn(dmodel, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(10):
        y = selective_scan_fn(
            x, dt, A, B, C, D.float(), z=z,
            delta_bias=dt_proj_bias.float(),
            delta_softplus=True, return_last_state=False,
        )
    
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = selective_scan_fn(
            x, dt, A, B, C, D.float(), z=z,
            delta_bias=dt_proj_bias.float(),
            delta_softplus=True, return_last_state=False,
        )
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = ( t1 - t0 ) / 10
    return elapsed


def profile_attn(batch, seq_len, num_heads, head_dim, block_size):

    shape = (batch, seq_len, num_heads, head_dim)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(10):
        y = flash_attn_func( q, k, v, softmax_scale=0.5, causal=True, window_size=(int(block_size), int(block_size)) )

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = flash_attn_func( q, k, v, softmax_scale=0.5, causal=True, window_size=(int(block_size), int(block_size)) )
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = ( t1 - t0 ) / 10

    return elapsed


def profile_based(batch, seq_len, num_heads, head_dim, feature_dim, is_tk=False):
    qk_shape = (batch, num_heads, seq_len, int(feature_dim))
    shape = (batch, num_heads, seq_len, int(head_dim))

    q = torch.randn(qk_shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(qk_shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    
    # Warmup
    for _ in range(10):
        if is_tk:
            y, kv_state = tk.based( q, k, v )
        else:
            y = fused_chunk_based( q, k, v, True, True)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        if is_tk:
            y, kv_state = tk.based( q, k, v )
        else:
            y = fused_chunk_based( q, k, v, True, True)
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = ( t1 - t0 ) / 10
    return elapsed


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u * D.unsqueeze(-1)
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)
    

def fftconv_tk(
    u, k, 
    u_real, u_imag, kfT_real, kfT_imag, 
    f_real, f_imag, finv_real, finv_imag, 
    tw_real, tw_imag, twinv_real, twinv_imag, 
    o_real, 
    B, H, N, N1
):
    out = tk.fftconv(u_real, kfT_real, kfT_imag, f_real, f_imag, finv_real, finv_imag, tw_real, tw_imag, twinv_real, twinv_imag, B, H, N, N1)


def profile_h3(batch, seq_len, d_model):
    u = torch.randn(batch, d_model, seq_len, dtype=torch.float16, device="cuda")
    k = torch.randn(d_model, seq_len, dtype=torch.float16, device="cuda")
    D = torch.randn(d_model, dtype=torch.float16, device="cuda")

    import math
    from fftconv import get_inputs
    B, H, N = u.shape
    N1 = int(math.sqrt(N))

    # ( 
    #     u_real, u_imag, kfT_real, kfT_imag, 
    #     f_real, f_imag, finv_real, finv_imag, 
    #     tw_real, tw_imag, twinv_real, twinv_imag, 
    #     o_real 
    # ) = get_inputs(
    #     u, k, B, H, N, N1, 
    # )

    # conv_flashfft = FlashFFTConv(int(N), dtype=torch.float16).to(u.device)

    # Warmup
    for _ in range(10):
        y = fftconv_ref(u, k, D, None)
        # y = fftconv_tk(
        #     u, k, 
        #     u_real, u_imag, kfT_real, kfT_imag, 
        #     f_real, f_imag, finv_real, finv_imag, 
        #     tw_real, tw_imag, twinv_real, twinv_imag, 
        #     o_real, 
        #     B, H, N, N1
        # )
        # y = conv_flashfft(u, k)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = fftconv_ref(u, k, D, None)
        # y = fftconv_tk(
        #     u, k, 
        #     u_real, u_imag, kfT_real, kfT_imag, 
        #     f_real, f_imag, finv_real, finv_imag, 
        #     tw_real, tw_imag, twinv_real, twinv_imag, 
        #     o_real, 
        #     B, H, N, N1
        # )
        # y = conv_flashfft(u, k)
    torch.cuda.synchronize()
    t1 = time.time()
    elapsed = ( t1 - t0 ) / 10
    return elapsed


def profile_hyena():
    return 0



def plot(
    df: pd.DataFrame,
    metric: str="valid/accuracy",
):
    idx = df.groupby(
        ["state_size", "model.name"]
    )[metric].idxmax(skipna=True).dropna()
    plot_df = df.loc[idx]

    model_name_2_func = {
        "h3": profile_h3,
        # "hyena": profile_hyena,
        "based": profile_based,
        "mamba": profile_mamba1,
        # "mamba2": profile_mamba2,
        "sliding-window-attention": profile_attn,
        "attention": profile_attn,
    }

    for i, row in plot_df.iterrows():
        model_name = row["model.name"]
        if model_name in model_name_2_func:
            batch_size = 256
            seq_len = 8192
            num_heads = 1
            d_model = row['model.d_model']
            head_dim = d_model // num_heads

            if model_name == "based":
                feature_dim = row['model.sequence_mixer.kwargs.configs.1.kwargs.feature_dim']
                if feature_dim != 16 or head_dim != 64: 
                    elapsed = 0
                else: 
                    args = (batch_size, seq_len, num_heads, head_dim, feature_dim)
                    elapsed = model_name_2_func[model_name](*args)
            elif model_name == "mamba":
                dstate = row['model.sequence_mixer.kwargs.d_state']
                args = (batch_size, seq_len, num_heads, head_dim, dstate)
                elapsed = model_name_2_func[model_name](*args)
            elif model_name == "mamba2":
                dstate = row['model.sequence_mixer.kwargs.d_state']
                args = (batch_size, seq_len, num_heads, head_dim, dstate)
                elapsed = model_name_2_func[model_name](*args)
            elif model_name == "sliding-window-attention":
                block_size = row['model.sequence_mixer.kwargs.configs.1.kwargs.block_size']
                args = (batch_size, seq_len, num_heads, head_dim, block_size)
                elapsed = model_name_2_func[model_name](*args)
            elif model_name == "attention":
                args = (batch_size, seq_len, num_heads, head_dim, -1)
                elapsed = model_name_2_func[model_name](*args)
            elif model_name == "h3":
                args = (batch_size, seq_len, d_model)
                elapsed = model_name_2_func[model_name]( *args )
            elif model_name == "hyena":
                elapsed = model_name_2_func[model_name]()
            else:
                import pdb; pdb.set_trace()
            plot_df.loc[i, "wall_clock"] = elapsed
        
        else:
            print(f"Skipping {model_name}")
            pass

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # convert elapsed time to milliseconds
    plot_df["wall_clock"] = plot_df["wall_clock"] * 1e3

    # set colors 
    model2color = {
        "based": "#DD8452",
        "mamba": "#F8EC41",
        "mamba2": "green",
        "sliding-window-attention": "#35A0CD",
        "attention": "#4B689D",
        "h3": "#B44A4E",
        "hyena": "#4F9F62",
    }

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        y=metric,
        x="wall_clock",
        hue="model.name",
        kind="scatter",
        edgecolor="black",
        marker="o",
        height=5,
        aspect=1,
        palette=model2color,
    )
    g.set(xscale="log", ylabel="Associative Recall Accuracy", xlabel="Wall clock time (ms)")


if __name__ == "__main__" :
    df = fetch_wandb_runs(
        launch_id=[
            "default-2024-02-09-05-44-06",
            "default-2024-02-09-14-59-58"
        ], 
        project_name="zoology"
    )

    plot(df=df)

    # save in high resolution
    plt.savefig("wall_clock_results.png", dpi=300)
    print("Saved wall clock results to wall_clock_results.png")


