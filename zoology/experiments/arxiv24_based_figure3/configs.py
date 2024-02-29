import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "kvs-lin-attn-sweep" + sweep_id

VOCAB_SIZE = 8_192

# 1. First we are going to create the data configuration

train_configs = [    
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=100_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=20_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=4),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=8),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=64, num_examples=1_000, num_kv_pairs=16),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=128, num_examples=1_000, num_kv_pairs=32),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=256, num_examples=1_000, num_kv_pairs=64),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=512, num_examples=1_000, num_kv_pairs=128),
    MQARConfig(vocab_size=VOCAB_SIZE, input_seq_len=1024, num_examples=1_000, num_kv_pairs=256),
]

input_seq_len=max([c.input_seq_len for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size / 8),
    cache_dir="/var/cr05_data/sabri_data/zoology",
    force_cache=False
)

# 2. Next, we are going to collect all the different model configs we want to sweep
models = []

model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE,
}

# define this conv outside of if/else block because it is used in multiple models
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)


for d_model in [
    64,
    128,
    192,
    256,
    320,
    384,
    448,
    512
]:
    ftrs = []
    
    # taylor exp
    for ftr_dim in [16, 24]:
        ftrs.append({"feature_name": "zoology.mixers.feature_maps.taylor.TaylorExp", "feature_dim": ftr_dim})
    

    # feature maps with no expansion 
    ftrs.extend([
        {"feature_name": "zoology.mixers.feature_maps.base.PosELU", "feature_dim": d_model},
        {"feature_name": "zoology.mixers.feature_maps.base.Identity", "feature_dim": d_model},
        {"feature_name": "zoology.mixers.feature_maps.base.ReLU", "feature_dim": d_model},
        {"feature_name": "zoology.mixers.feature_maps.base.Square", "feature_dim": d_model},
        {"feature_name": "zoology.mixers.feature_maps.cosformer.CosFormerFeatureMap", "feature_dim": d_model},
        {"feature_name": "zoology.mixers.feature_maps.performer.PerformerFeatureMap", "feature_dim": d_model},
    ])
    
    # all poly 
    ftrs.append({
        "feature_name": "all_poly",
        "feature_kwargs": {
            "output_dim": d_model,
            "learnable": True,
            "init": "kaiming"
        },
        "feature_dim": d_model
    })

    for ftr in ftrs:
        lin_attn = dict(
            name="zoology.mixers.based.Based",
            kwargs={
                "l_max": input_seq_len,
                # "feature_name": "taylor_exp",
                **ftr,
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



# 3. Finally we'll create a train config for each
configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"{model.name}-lr{lr:.1e}"
        config = TrainConfig(
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name="zoology",
                entity="hazy-research"
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"/var/cr05_data/sim_data/zg-synthetics/predictions/{run_id}",
            collect_predictions=True,
        )
        configs.append(config)