from zoology.config import TrainConfig, ModelConfig, DataConfig



config = TrainConfig(
    data=DataConfig(
        cache_dir="/var/cr05_data/sabri_data/zg-synthetics",
        vocab_size=128,
        builder={
            "name": "zoology.data.associative_recall.gap_power_distr_ar",
            "kwargs": {
                "num_kv_pairs": 4
            }
        },
        
    ),
    model=dict(
        vocab_size=128,
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