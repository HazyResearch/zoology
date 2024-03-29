import numpy as np
import torch
from collections import Counter

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

class ParityConfig(DataSegmentConfig):
    name: str="parity"
    def build(self, seed: int) -> DataSegment:
        return parity(**self.model_dump(), seed=seed)

def parity(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate parity sequences.
    """
    np.random.seed(seed)

    one = 1
    zero = 0
    end = 2

    inputs = []
    labels = []
    for _ in range(num_examples):
        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(0, 2, size=input_seq_len-1)
        parity = len([b for b in seq if b == 1]) % 2
        
        # convert to vocab
        seq = [one if s == 1 else zero for s in seq]
        if parity: parity = one
        else: parity = zero

        # Append the parity to the sequence
        input = np.array(seq + [end] + [parity], dtype=np.int64)
        input = torch.tensor(input)
        label = torch.full_like(input[:-1], -100) # -100 for labels, except last position
        label[-1] = input[-1]
        input = input[:-1]

        inputs.append(input)
        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    
    return DataSegment(
        inputs, 
        labels, 
        slices={"input_seq_len": input_seq_len}
    )


class MajorityConfig(DataSegmentConfig):
    name: str="majority"
    def build(self, seed: int) -> DataSegment:
        return majority(**self.model_dump(), seed=seed)

def majority(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate majority sequences.
    """
    np.random.seed(seed)

    one = 1
    zero = 0
    end = 2

    inputs = []
    labels = []
    # slices = []
    for _ in range(num_examples):
        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(0, 2, size=input_seq_len-1)
        seq = [one if s == 1 else zero for s in seq]
        num_ones = sum(seq)
        ratio = num_ones / len(seq)
        most = ratio > 0.5
        ratio = round(ratio * 5) / 5 # round ratio to nearest 0.2

        # Full sequence
        input = np.array(seq + [end] + [most], dtype=np.int64)

        # Inputs and outputs
        input = torch.tensor(input)
        label = torch.full_like(input[:-1], -100) # -100 for labels, except last position
        label[-1] = input[-1]
        input = input[:-1]

        inputs.append(input)
        labels.append(label)
        # slices.append(ratio)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    
    return DataSegment(
        inputs, 
        labels, 
        # slices={"ratio": slices}
    )


class VocabMajorityConfig(DataSegmentConfig):
    name: str="vocab_majority"
    def build(self, seed: int) -> DataSegment:
        return vocab_majority(**self.model_dump(), seed=seed)

def vocab_majority(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate majority sequences.
    """
    np.random.seed(seed)
    end = 0

    inputs = []
    labels = []
    for _ in range(num_examples):

        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(1, vocab_size, size=input_seq_len-1)
        seq = [int(s) for s in seq]
        counter = Counter(seq)
        max_count = max(counter.values())
        keys_with_max_count = [k for k, v in counter.items() if v == max_count]
        most_key = min(keys_with_max_count)
        if len(keys_with_max_count) > 1:
            other_key = [k for k in counter.keys() if k != most_key][0]
            # replace an instance of other_key with most_key
            idx = seq.index(other_key)
            seq[idx] = most_key
        print(seq)

        # Full sequence
        input = np.array(seq + [end] + [most_key], dtype=np.int64)

        # Inputs and outputs
        input = torch.tensor(input)
        label = torch.full_like(input[:-1], -100) # -100 for labels, except last position
        label[-1] = input[-1]
        input = input[:-1]

        inputs.append(input)
        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    
    return DataSegment(
        inputs, 
        labels, 
        slices={"vocab_size": vocab_size}
    )


class CumulativeParityConfig(DataSegmentConfig):
    name: str="cumulative_parity"
    def build(self, seed: int) -> DataSegment:
        return cumulative_parity(**self.model_dump(), seed=seed)

def cumulative_parity(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> DataSegment:
    """
    Generate parity sequences.
    """
    np.random.seed(seed)

    one = 1
    zero = 0
    end = 2

    inputs = []
    labels = []
    for _ in range(num_examples):
        # Generate a random sequence of 0s and 1s
        seq = np.random.randint(0, 2, size=input_seq_len-1)
        parities = []
        cur_parity = 0
        for i in range(input_seq_len-1):
            cur_parity = (cur_parity + seq[i]) % 2
            parities.append(cur_parity)

        # Append the parity to the sequence
        input = np.array(seq, dtype=np.int64)
        input = torch.tensor(input)
        label = torch.tensor(parities)

        inputs.append(input)
        labels.append(label)

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    
    return DataSegment(
        inputs, 
        labels, 
        slices={"input_seq_len": input_seq_len}
        # slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
    )





class CumulativeMajorityConfig(DataSegmentConfig):

    num_examples: int=1_000
    input_seq_len: int=16

    name: str="cumulative_majority"


    def build(self, seed: int) -> DataSegment:
        return cumulative_majority(self, seed=seed)

def cumulative_majority(
    config: CumulativeMajorityConfig,
    seed: int, 
    **kwargs
) -> DataSegment:
    """
    Generate majority sequences.
    """
    np.random.seed(seed)

    inputs = np.random.randint(0, 2, size=(config.num_examples, config.input_seq_len))
    labels = ((inputs * 2 - 1).cumsum(axis=1) >= 0).astype(int)

    return DataSegment(
        torch.tensor(inputs), 
        torch.tensor(labels), 
        slices={"input_seq_len": config.input_seq_len}
    )

