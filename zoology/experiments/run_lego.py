from zoology.config import TrainConfig, ModelConfig, DataConfig



config = TrainConfig(
    data=DataConfig(
        cache_dir="/dfs/scratch0/mfchen/.cache",
        vocab_size=26,
        num_train_examples=100,
        num_test_examples=50,
        input_seq_len=15,
        builder={
            "name": "zoology.data.lego.lego",
            "kwargs": {
                "n_var": 4,
                "train_proportions": [1, 0, 0, 0],
                "test_proportions": [1, 0, 0, 0]
            }
        },
        
    ),
    model=dict(
        vocab_size=31,
        max_position_embeddings=15,
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