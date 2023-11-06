from zoology.config import TrainConfig, ModelConfig, DataConfig



config = TrainConfig(
    data=DataConfig(
        cache_dir="/var/cr05_data/sabri_data/zg-synthetics",
        vocab_size=256,
        input_seq_len=128,
        num_train_examples=10_000,
        num_test_examples=1_000,
        builder={
            "name": "zoology.data.associative_recall.base_ar",
            "kwargs": {}
        },
        
    ),
    model=ModelConfig(
        vocab_size=256,
        max_position_embeddings=128,
        sequence_mixer={
            "name": "zoology.mixers.attention.MHA",
            "kwargs": {
                "dropout": 0.1,
                "num_heads": 1
            }
        }
    ),
    
)

configs = [config]