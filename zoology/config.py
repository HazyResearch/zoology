from pydantic import BaseModel

class FunctionConfig(BaseModel):
    name: str
    args: dict = None



class DataConfig(BaseModel):
    num_train_examples: int
    num_test_examples: int
    input_seq_len: int
    vocab_size: int
    builder: FunctionConfig = None
    seed: int = 0
    batch_size: int = 32
    cache_dir: str = None
    caching: bool = True
    force_cache: bool = False 