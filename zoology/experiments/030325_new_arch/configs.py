import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, ModuleConfig, DataConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig


sweep_id = uuid.uuid4().hex[:6]
sweep_name = "deepseek-v3-blocksize=8-swa-16-selected=4" + sweep_id

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
    cache_dir="/data/sim/zoology"
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


from zoology.experiments.models_repo import add_attention, add_based, add_mamba2, add_rwkv7, add_delta_net, add_gla, add_gated_delta_net, add_deepseek_nsa

models = add_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_rwkv7(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gla(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_gated_delta_net(models, conv_mixer, input_seq_len, model_factory_kwargs)
models = add_deepseek_nsa(models, conv_mixer, input_seq_len, model_factory_kwargs)

# convenience for filtering out 
included = ["deepseek_nsa"]
models = [m for m in models if any([i in m.name for i in included])]


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


