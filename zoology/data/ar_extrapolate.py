

from typing import List
import numpy as np
import torch
from pydantic import BaseModel

from .utils import SyntheticDataSection
from .associative_recall import _mqar, _ar

class ARConfig(BaseModel):
    input_seq_len: int = 64
    num_examples: int = 1_000
    num_kv_pairs: int = 4
    num_queries: int = 4
    random_non_queries: bool=True


def ar_extrapolate(
    train_configs: List[ARConfig],
    test_configs: List[ARConfig],
    vocab_size: int=8_192,
    num_train_examples: int=100_000,
    num_test_examples: int=3_000,
    input_seq_len: int=64,
    seed: int=0,
) -> SyntheticDataSection:
    
    # input seq len should be the max for all the configs
    assert input_seq_len == max([c.input_seq_len for c in train_configs + test_configs])
    assert num_train_examples == sum([c.num_examples for c in train_configs])
    assert num_test_examples == sum([c.num_examples for c in test_configs])
    
    train_sections = []
    for idx, train_config in enumerate(train_configs):
        inputs, labels = _mqar(
            vocab_size=vocab_size, 
            **train_config.dict(), 
            seed=seed + idx
        )
        slices = [train_config.dict()] * len(inputs)
        train_sections.append(
            SyntheticDataSection(inputs=inputs, labels=labels, slices=slices)
        )
    
    test_sections = []
    for idx, test_config in enumerate(test_configs):
        inputs, labels = _mqar(
            vocab_size=vocab_size, 
            **test_config.dict(), 
            seed=seed + idx + len(train_configs)
        )
        slices = [test_config.dict()] * len(inputs)
        test_sections.append(
            SyntheticDataSection(inputs=inputs, labels=labels, slices=slices)
        )
    return MultiSyntheticData(train=train_sections, test=test_sections)
    





    
    