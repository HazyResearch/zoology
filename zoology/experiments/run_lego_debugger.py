from zoology.config import TrainConfig, ModelConfig, DataConfig



config = TrainConfig(
    data=[DataConfig(
        cache_dir="/dfs/scratch0/mfchen/.cache",
        vocab_size=26,
        num_train_examples=10000,
        num_test_examples=500,
        input_seq_len=12,
        builder={
            "name": "zoology.data.lego.lego",
            "kwargs": {
                "seed": 0,
                "n_var": 3,
                "train_proportions": [1, 0, 0],
                "test_proportions": [1, 0, 0]
            }
        },
    ), DataConfig(
        cache_dir="/dfs/scratch0/mfchen/.cache",
        vocab_size=26,
        num_train_examples=10000,
        num_test_examples=500,
        input_seq_len=12,
        builder={
            "name": "zoology.data.lego.lego",
            "kwargs": {
                "seed": 1,
                "n_var": 3,
                "train_proportions": [0, 1, 0],
                "test_proportions": [0, 1, 0]
            }
        },
    ) 
    ],
    model=dict(
        n_layers=4,
        vocab_size=31,
        max_position_embeddings=12,
        sequence_mixer={
            "name": "zoology.mixers.attention.MHA",
            "kwargs": {
                "dropout": 0.1,
                "num_heads": 4
            }
        }
    ),
    max_epochs=300
)

configs = [config]