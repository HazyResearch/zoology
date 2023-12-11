import numpy as np
from zoology.config import TrainConfig
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig

from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig


configs = []

for lr in np.logspace(-4, -2, 10):

   config = TrainConfig(
      data=DataConfig(
         # cache_dir="/path/to/cache/dir"  TODO: add a directory where data will be cached
         vocab_size=256,
         input_seq_len=64,
         num_train_examples=10_000,
         num_test_examples=1_000,
         builder=FunctionConfig(
               name="zoology.data.associative_recall.multiquery_ar",
               kwargs={"num_kv_pairs": 4}
         ),
         
      ),
      model=ModelConfig(
         vocab_size=256,
         max_position_embeddings=64,
         sequence_mixer=ModuleConfig(
               name="zoology.mixers.attention.MHA",
               kwargs={"dropout": 0.1, "num_heads": 1}
         )
      ), 
      learning_rate=lr
   )
   configs.append(config)
