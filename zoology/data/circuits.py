
# def majority(vocab, seqlen, rng, cumulative=False, simple=False):
#     """
#         a b c c e c c ==> c
#     """
#     input_seq_len = seqlen
#     non_special_vocab_size = len(vocab.non_special_vocab) 
#     vocab_toks = vocab.non_special_vocab[:non_special_vocab_size]
#     vocab_seq = []

#     majority_token = rng.choice(list(vocab_toks))

#     tok2count = Counter()
#     while len(vocab_seq) < input_seq_len:
#         k = rng.choice(list(vocab_toks))
#         if tok2count[k] >= tok2count[majority_token] - 1:
#             k = majority_token
#         vocab_seq.append(k) 
#         tok2count[k] += 1

#     if not simple:
#         # shuffle the sequence
#         rng.shuffle(vocab_seq)

#     majority_token = tok2count.most_common(1)[0][0]
    
#     to_copy = [vocab.copy_prefix] + [ majority_token ]
#     vocab_seq = vocab_seq + to_copy
#     kv_map = {
#         "majority": majority_token,
#     }

#     labels = None
#     if cumulative:
#         running_majority = []

#         tok2count = Counter()
#         for i in range(len(vocab_seq)):
#             if vocab_seq[i] == vocab.copy_prefix:
#                 running_majority.extend(vocab_seq[i:])
#                 break
#             else:
#                 cur_tok = vocab_seq[i]
#                 tok2count[cur_tok] += 1
#                 max_tok = tok2count.most_common(1)[0][0]
#                 running_majority.append(max_tok)

#         labels = running_majority
#         labels = " ".join(labels)

#     return " ".join(vocab_seq), kv_map, labels
 

# def binary_majority(vocab, seqlen, rng, cumulative=False, simple=False):
#     """
#         If the number of 1s in the sequence is odd, the parity bit is 1
#     """

#     one_tok = '1' 
#     zero_tok = '0' 

#     # Generate a random sequence of 0s and 1s
#     seq = rng.integers(0, 2, size=seqlen)
#     num_ones = sum(seq)
#     num_zeros = len(seq) - num_ones
#     majority = one_tok if num_ones > num_zeros else zero_tok
    
#     # convert to vocab
#     seq = [one_tok if s == 1 else zero_tok for s in seq]
#     seq = seq + [vocab.copy_prefix] + [majority]

#     kv_map = {
#         "one": one_tok,
#         "zero": zero_tok,
#     }
#     return " ".join(seq), kv_map, None


# def threshold_k_function(vocab, seqlen, rng, cumulative=False, k=5):
#     """
#     A thresholdk function is a boolean function whose output is 1
# depending on whether at least k of its inputs have value 1. 
#     """

#     binary_sequence = [0]*seqlen

#     label = random.randint(0, 1)
#     if label == 0:
#         num_ones = random.randint(k-5, k-1)
#     elif label == 1:
#         num_ones = random.randint(k, k+5)
#     else:
#         raise Exception("Unknown label")

#     positions = random.sample(range(seqlen), num_ones)
#     for pos in positions:
#         binary_sequence[pos] = 1

#     binary_sequence = [str(x) for x in binary_sequence]
#     label = str(label)

#     vocab_seq = binary_sequence + [vocab.copy_prefix] + [label]
#     return " ".join(vocab_seq), {}, None


import numpy as np
import torch

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
        # slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
    )

