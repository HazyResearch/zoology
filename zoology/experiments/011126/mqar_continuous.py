import uuid
import numpy as np
from zoology.config import TrainConfig, DataConfig, LoggerConfig
from zoology.data.stacked_mqar import ContinuousMQARConfig as MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "011026-data=simple_stacked_mqar-sweep" + sweep_id

VOCAB_SIZE = 8_192

# 1. First we are going to create the data configuration

train_configs = [    
    MQARConfig(num_examples=100_000, num_kv_pairs=4),
    MQARConfig(num_examples=20_000, num_kv_pairs=8),
    MQARConfig(num_examples=20_000, num_kv_pairs=16),
    MQARConfig(num_examples=20_000, num_kv_pairs=32),
    MQARConfig(num_examples=20_000, num_kv_pairs=64),
]
test_configs = [
    MQARConfig(num_examples=1_000, num_kv_pairs=4),
    MQARConfig(num_examples=1_000, num_kv_pairs=8),
    MQARConfig(num_examples=1_000, num_kv_pairs=16),
    MQARConfig(num_examples=1_000, num_kv_pairs=32),
    MQARConfig(num_examples=1_000, num_kv_pairs=64),
    MQARConfig(num_examples=1_000, num_kv_pairs=128),
    MQARConfig(num_examples=1_000, num_kv_pairs=256),
]

input_seq_len = max([(c.num_passes + 1) * c.num_kv_pairs for c in train_configs + test_configs])
batch_size = 256
data = DataConfig(
    train_configs=train_configs,
    test_configs=test_configs,
    # can pass a tuple if you want a different batch size for train and test
    batch_size=(batch_size, batch_size // 8),
    cache_dir="/data/sim/zoology"
)

# 2. Next, we are going to collect all the different model configs we want to sweep
models = [] 
model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}), "vocab_size": VOCAB_SIZE,
}
# define this conv outside of if/else block because it is used in multiple models
conv_mixer = None 


from zoology.experiments.models_repo import (
    add_attention, add_sliding_window,add_based, add_mamba2, add_rwkv7, 
    add_delta_net, add_gla, add_gated_delta_net, add_deepseek_nsa, add_ttt
)

models = add_attention(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_based(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_mamba2(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_sliding_window(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_rwkv7(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_deepseek_nsa(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)
models = add_ttt(models, conv_mixer, input_seq_len, model_factory_kwargs, num_layers=1)

# convenience for filtering out 
included = [
    "attention", 
    "sliding_window", 
    # "based", 
    # "delta_net", "gla", 
    "gated_delta_net", 
    # "deepseek_nsa", 
    "ttt_linear", "ttt_mlp"
]
models = [m for m in models if any([i == m.name.lower() for i in included])]

for model in models:
    model.embedding_init_type = "spherical"
    model.learnable_word_embeddings = False


# 3. Finally we'll create a train config for each
configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"{model.name}-lr{lr:.1e}"
        config = TrainConfig(
            input_type="continuous",
            model=model,
            data=data,
            learning_rate=lr,
            max_epochs=32,
            logger=LoggerConfig(
                project_name="0325_zoology",
                entity="hazy-research"
            ),
            slice_keys=["num_kv_pairs"],
            sweep_id=sweep_name,
            run_id=run_id,
            predictions_path=f"/data/sim/zoology/predictions/{run_id}",
            collect_predictions=True,
        )
        configs.append(config)


