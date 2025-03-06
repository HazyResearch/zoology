

import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig


# Attention
def add_attention(models, conv_mixer, input_seq_len, model_factory_kwargs):
    for d_model in [64, 128]:
        attention_mixer = dict(
            name="zoology.mixers.attention.MHA",
            kwargs={
                "dropout": 0.1,
                "num_heads": 1
            },
        )
        mixer = ModuleConfig(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, attention_mixer]}
        )
        model = ModelConfig(
            block_type = "TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="attention",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# BASED
def add_based(models, conv_mixer, input_seq_len, model_factory_kwargs):
    for d_model in [
        48,
        64, 
        128, 
        # 256
    ]:
        for ftr_dim in [
            8, 
            16, 
            24,
            # 32, 
            # 64
        ]:
            lin_attn = dict(
                name="zoology.mixers.based.Based",
                kwargs={
                    "l_max": input_seq_len,
                    "feature_dim": ftr_dim,
                    "feature_name": "taylor_exp",
                    "num_key_value_heads": 1,
                    "num_heads": 1,
                    "train_view": "quadratic",
                }
            )
            mixer = dict(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={"configs": [conv_mixer, lin_attn]}
            )
            name = f"based"
            model = ModelConfig(
                block_type="TransformerBlock",
                d_model=d_model,
                n_layers=2,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name=name,
                **model_factory_kwargs
            )
            models.append(model)
    return models


# Sliding window 
def add_sliding_window(models, conv_mixer, input_seq_len, model_factory_kwargs):
    for d_model in [128]:
        for slide_width in [8, 16, 32, 64, 128, 256, 512, 1024]:
            slide_attn = dict(
                name="zoology.mixers.slide_attn.SlidingAttn",
                kwargs={
                    "block_size": slide_width,
                    "attention_dropout": 0.0
                }
            )
            mixer = dict(
                name="zoology.mixers.hybrid.Hybrid",
                kwargs={"configs": [conv_mixer, slide_attn]}
            )
            name = f"sliding-window-attention"
            n_layers = 2
            model = ModelConfig(
                block_type="TransformerBlock",
                d_model=d_model,
                n_layers=2,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name=name,
                **model_factory_kwargs
            )
            models.append(model)
    return models


# Mamba 
def add_mamba(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "MambaBlock"
    for d_model in [64, 128, 256]:
        for d_state in [8, 16, 24]:
            mixer = dict(
                name="zoology.mixers.mamba.Mamba",
                kwargs={"d_state": d_state}
            )
            model = ModelConfig(
                block_type="MambaBlock",
                d_model=d_model,
                n_layers=2,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name="mamba",
                **model_factory_kwargs
            )
            models.append(model)
    return models


# Mamba2
def add_mamba2(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "Mamba2Block"
    for d_model in [64, 128, 256]:
        for d_state in [8, 16, 24]:
            mixer = dict(
                name="zoology.mixers.mamba2.Mamba2",
                kwargs={"d_state": d_state}
            )
            model = ModelConfig(
                block_type="Mamba2Block",
                d_model=d_model,
                n_layers=2,
                sequence_mixer=mixer,
                max_position_embeddings=0,
                name="mamba2",
                **model_factory_kwargs
            )
            models.append(model)
    return models


# Hyena 
def add_hyena(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]:
        mixer = dict(
            name="zoology.mixers.hyena.Hyena",
            kwargs={"l_max": input_seq_len}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="hyena",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# H3 
def add_h3(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]:
        mixer = dict(
            name="zoology.mixers.h3.H3",
            kwargs={
                "l_max": input_seq_len,
                "d_state": d_model / 4,
                "head_dim": 2
            }
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="h3",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# RWKV7
def add_rwkv7(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]:
        rwkv7_mixer = dict(
            name="zoology.mixers.rwkv7.RWKV7Attention",
            kwargs={
                "l_max": input_seq_len,
                "head_dim": 64, 
                "decay_low_rank_dim": 16,    # Same as head dim? 
                "gate_low_rank_dim": 64,     # Tune
                "a_low_rank_dim": 16,        # Tune
                "v_low_rank_dim": 16,        # Tune
            }
        )
        mixer = dict(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, rwkv7_mixer]}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="rwkv7",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# DeltaNet
def add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        delta_net_mixer = dict(
            name="zoology.mixers.delta_net.DeltaNet",
            kwargs={
                "l_max": input_seq_len,
                "num_heads": 2,         # Tune
                "use_beta": True,       # Tune
                "use_gate": False,      # Tune
                "use_short_conv": True, # Tune
                "conv_size": 4
            }
        )
        mixer = dict(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, delta_net_mixer]}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="delta_net",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# DeltaNet
def add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        delta_net_mixer = dict(
            name="zoology.mixers.gated_delta_net.GatedDeltaNet",
            kwargs={
                "l_max": input_seq_len,
                "num_heads": 2,         # Tune
                "use_gate": False,      # Tune
                "use_short_conv": True, # Tune
                "conv_size": 4
            }
        )
        mixer = dict(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, delta_net_mixer]}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="gated_delta_net",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# Gated linear attention
def add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        delta_net_mixer = dict(
            name="zoology.mixers.gla.GatedLinearAttention",
            kwargs={
                "num_heads": 2,          # Tune
                "use_short_conv": False, # Tune (False default)
            }
        )
        mixer = dict(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, delta_net_mixer]}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="gla",
            **model_factory_kwargs
        )
        models.append(model)
    return models


# Deepseek NSA
def add_deepseek_nsa(models, conv_mixer, input_seq_len, model_factory_kwargs):
    block_type = "TransformerBlock"
    for d_model in [64, 128, 256]: 
        delta_net_mixer = dict(
            name="zoology.mixers.deepseek_nsa.SparseAttention",
            kwargs={
                "num_heads": 2,            # Tune
                "sliding_window_size": 16, # Tune
                "compress_block_size": 8, # Tune
                "selection_block_size": 8, # Tune
                "num_selected_blocks": 4,   # Tune
            }
        )
        mixer = dict(
            name="zoology.mixers.hybrid.Hybrid",
            kwargs={"configs": [conv_mixer, delta_net_mixer]}
        )
        model = ModelConfig(
            block_type="TransformerBlock",
            d_model=d_model,
            n_layers=2,
            sequence_mixer=mixer,
            max_position_embeddings=0,
            name="deepseek_nsa",
            **model_factory_kwargs
        )
        models.append(model)
    return models

