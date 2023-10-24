from functools import partial
from pydantic import BaseModel

from zoology.utils import import_from_str

class FunctionConfig(BaseModel):
    name: str
    kwargs: dict = None

    def instantiate(self):
        return partial(import_from_str(self.name), **self.kwargs)


class DataConfig(BaseModel):
    builder: FunctionConfig = None
    seed: int = 0

    num_train_examples: int = 10_000
    num_test_examples: int = 1000
    input_seq_len: int = 64
    vocab_size: int = 128
    batch_size: int = 2
    
    cache_dir: str = None
    caching: bool = True
    force_cache: bool = False 


class Config(BaseModel):
    data: DataConfig = DataConfig()

    epochs: int = 100
    learning_rate: float = 1e-3
    train_batch: int = 12
    layers: int = 2


    

    run_id: str = None



