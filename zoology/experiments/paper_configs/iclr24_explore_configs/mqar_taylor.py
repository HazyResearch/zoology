import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, LoggerConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "lin-attn-sweep" + sweep_id

VOCAB_SIZE = 8_192

configs = []
for input_seq_len, num_kv_pairs in [
    # (64, 4),
    # (128, 8),
    (256, 16),
    # (512, 64),
]:
    if input_seq_len == 1024:
        batch_size = 64
    elif input_seq_len == 512:
        batch_size = 64
    elif input_seq_len == 256:
        batch_size = 128
    else:
        batch_size = 256

    data = DataConfig(
        num_train_examples=100_000,
        num_test_examples=3_000,
        vocab_size=VOCAB_SIZE,
        input_seq_len=input_seq_len,
        batch_size=batch_size,
        cache_dir="/var/cr05_data/sim_data/zg-synthetics",
        builder={
            "name": "zoology.data.associative_recall.multiquery_ar",
            "kwargs": {
                "num_kv_pairs": num_kv_pairs,
                "train_power_a": 0.01,
                "test_power_a": 0.01,
                "random_non_queries": False
            }
        }   
    )

    for d_model in [
        64, 
        128, 
        # 256, 
        # 512
    ]:
        for lr in  np.logspace(-4, -2, 4):

            for ftr_dim in [8, 16, 32]:

                for num_heads in [1, 4]:
                
                    MIXERS = {
                        "lin-attn": dict(
                            name="zoology.mixers.based.Based",
                            kwargs={
                                "l_max": input_seq_len,
                                "feature_dim": ftr_dim,
                                "num_key_value_heads": num_heads,
                                "num_heads": num_heads,
                                "feature_name": "taylor_exp"
                            }
                        ),
                    }

                    for sequence_mixer in [
                        "lin-attn"
                    ]:
                        block_type = "TransformerBlock"
                        model = ModelConfig(
                            d_model=d_model,
                            n_layers=4,
                            block_type=block_type,
                            max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                            vocab_size=VOCAB_SIZE,
                            sequence_mixer=MIXERS[sequence_mixer],
                            state_mixer=dict(name="torch.nn.Identity", kwargs={})
                        )
                        run_id = f"01-23-sweep-lin-attn-4depth-L{input_seq_len}-D{d_model}-lr{lr}-kv{num_kv_pairs}-ftr{ftr_dim}-heads{num_heads}"
                        config = TrainConfig(
                            model=model,
                            data=data,
                            learning_rate=lr,
                            max_epochs=64,
                            logger=LoggerConfig(
                                project_name="zoology",
                                entity="hazy-research"
                            ),
                            sweep_id=sweep_name,
                            run_id=run_id,
                            predictions_path=f"/var/cr05_data/sim_data/zg-synthetics/predictions/{run_id}",
                            collect_predictions=True,
                        )
                        configs.append(config)