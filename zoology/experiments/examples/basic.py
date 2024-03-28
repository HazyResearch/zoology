from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig
from zoology.data.associative_recall import MQARConfig

factory_kwargs = {
        "num_kv_pairs": 4,
    }

config = TrainConfig(
    data=DataConfig(
        # cache_dir="/path/to/cache/dir"  TODO: add this
        train_configs=[MQARConfig(num_examples=10_000, vocab_size=256, input_seq_len=64, **factory_kwargs)],
        test_configs=[MQARConfig(num_examples=1_000, vocab_size=256, input_seq_len=64, **factory_kwargs)],
    ),
    model=ModelConfig(
        vocab_size=256,
        max_position_embeddings=64,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 1}
        )
    ),
    
)

configs = [config]