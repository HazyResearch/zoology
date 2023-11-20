import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_id = "fact_retrieval" + sweep_id


configs = []
for n in [512, 1024, 2048, 4096, 8192, 16384]:

    vocab_size = 2 * n + 2
    data = DataConfig(
        vocab_size=vocab_size,
        input_seq_len=3,
        batch_size=n,
        cache_dir="/var/cr05_data/sabri_data/zg-synthetics",
        builder={
            "name": "zoology.data.knowledge.fact_retrieval",
            "kwargs": {
                "n_subjects": n,
                "p_predicates": 1,
                "m_objects": n
            }
        }   
    )

    for d_model in [32, 64, 128, 256, 512, 1024]:
        for lr in  np.logspace(-4, -2, 4):
            
            MIXERS = {
                "attention": dict(
                    name="zoology.mixers.attention.MHA",
                    kwargs={
                        "dropout": 0.1,
                        "num_heads": 1
                    },
                ),
                "mlp": dict(
                    name="zoology.mixers.mlp.MLP",
                    kwargs={
                        "hidden_mult": 1,
                    },
                ),
            }

            for sequence_mixer in [
                "attention",
            ]:
                model = ModelConfig(
                    d_model=d_model,
                    n_layers=2,
                    max_position_embeddings=n if sequence_mixer == "attention" else 0,
                    vocab_size=vocab_size,
                    sequence_mixer=MIXERS[sequence_mixer],
                    state_mixer=MIXERS["mlp"],
                    learnable_word_embeddings=False
                )
                config = TrainConfig(
                    model=model,
                    data=[data],
                    learning_rate=lr,
                    max_epochs=1024,
                    run_id=f"{sequence_mixer}-seqlen{n}-dmodel{d_model}-lr{lr}",
                    sweep_id=sweep_id,

                )
                configs.append(config)