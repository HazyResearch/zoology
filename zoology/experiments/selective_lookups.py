import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "monarch_attn" + sweep_id


VOCAB_SIZE = 8_192


configs = []
for input_seq_len, num_kv_pairs in [
    (64, 4),
    # (128, 8),
    # (256, 16),
    # (512, 64),
]:
    if input_seq_len == 1024:
        batch_size = 64
    elif input_seq_len == 512:
        batch_size = 128
    else:
        batch_size = 1024

    data = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        vocab_size=VOCAB_SIZE,
        input_seq_len=input_seq_len,
        batch_size=batch_size,
        cache_dir="/var/cr05_data/sabri_data/zg-synthetics",
        builder={
            "name": "zoology.data.associative_recall.gap_power_distr_ar",
            "kwargs": {
                "num_kv_pairs": num_kv_pairs,
                "train_power_a": 0.01,
                "test_power_a": 0.01,
                "random_non_queries": False
            }
        }   
    )

    for d_model in [
        # 64, 
        # 128, 
        # 256, 
        512
    ]:
        for lr in np.logspace(-4, -2, 8):
            for penalty in [1e-2, 1e-1, 1, 10]: #, 1, 1e1, 0]:
                for num_lookups in [8, 10, 12, 14, 16]:
                        
                    MIXERS = {
                        "attention": dict(
                            name="zoology.mixers.attention.MHA",
                            kwargs={
                                "dropout": 0.1,
                                "num_heads": 1
                            },
                        ),
                        "selective": dict(
                            name="zoology.mixers.selective.SelectiveLookups",
                            kwargs={
                                "dropout": 0.1,
                                "num_heads": 1
                            },
                        ),
                        "sma": dict(
                            name="zoology.mixers.selective.SMA",
                            kwargs={
                                "dropout": 0.1,
                                "num_heads": 1,
                                "alpha": 0.3
                            },
                        ),
                        "sigmoid": dict(
                            name="zoology.mixers.selective.SigmoidLookups",
                            kwargs={
                                "dropout": 0.1,
                                "num_heads": 1,
                                "selection_penalty": penalty, 
                                "n_lookups": num_lookups
                            },
                        ),
                    }

                    for sequence_mixer in [
                        # "attention",
                        # "hyena",
                        # "rwkv",
                        # "base_conv"
                        # "base_conv_explicit",
                        # "selective",
                        # "sma"
                        "sigmoid"
                    ]:
                        model = ModelConfig(
                            d_model=d_model,
                            n_layers=2,
                            max_position_embeddings=input_seq_len,
                            vocab_size=VOCAB_SIZE,
                            sequence_mixer=MIXERS[sequence_mixer],
                            state_mixer=dict(name="torch.nn.Identity", kwargs={})
                        )
                        config = TrainConfig(
                            model=model,
                            data=[data],
                            learning_rate=lr,
                            max_epochs=64,
                            run_id=f"{penalty}-{sequence_mixer}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr}-kv{num_kv_pairs}"
                        )
                        configs.append(config)