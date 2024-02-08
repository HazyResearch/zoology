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
    cache_dir="/var/cr05_data/sabri_data/zoology"
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


# attention
for d_model in [64, 128]:
    attention_mixer = dict(
        name="zoology.mixers.attention.MHA",
        kwargs={
            "dropout": 0.1,
            "num_heads": 1
        },
    )
    mixer = ModuleConfig(
        name="zoology.mixers.hybrid.Hybrid",
        kwargs={"configs": [conv_mixer, attention_mixer]}
    )
    model = ModelConfig(
        block_type = "TransformerBlock",
        d_model=d_model,
        n_layers=2,
        sequence_mixer=mixer,
        max_position_embeddings=0,
        name="attention-conv",
        **model_factory_kwargs
    )
    models.append(model)



# 3. Finally we'll create a train config for each
configs = []
for model in models:
    for lr in np.logspace(-3, -1.5, 4):
        run_id = f"{model.name}-lr={lr}"
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