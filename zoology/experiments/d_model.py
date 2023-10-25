import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "monarch_attn" + sweep_id


VOCAB_SIZE = 8_192


mixers = {
    "attention": dict(
        name="zoology.mixers.attention.MHA",
        kwargs={
            "dropout": 0.1,
            "num_heads": 1
        },
    )
}


configs = []
for input_seq_len, num_kv_pairs in [
    # (32, 2),
    (64, 4),
    (128, 8),
    # (256, 16),
    # (512, 64),
    # (1024, 128),
]:
    if input_seq_len == 1024:
        batch_size = 64
    elif input_seq_len == 512:
        batch_size = 128
    else:
        batch_size = 256

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

    for d_model in [128]:
        for lr in [1e-3]: # np.logspace(-4, -2, 8):
            for sequence_mixer in [
                "attention",
            ]:
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    max_position_embeddings=input_seq_len,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=mixers[sequence_mixer],
                )

                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epoch=64
                )
                configs.append(config)