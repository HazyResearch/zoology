
import numpy as np
import torch

from .utils import SyntheticData

def gap_power_distr_ar(
    vocab_size: int,
    num_train_examples: int,
    num_test_examples: int,
    input_seq_len: int,
    seed: int,
    num_kv_pairs: int,
    train_power_a: float=0.01,
    test_power_a: float=0.01,
    random_non_queries: bool=True
):
    train_inputs, train_labels = _gap_power_distr_ar(
        vocab_size=vocab_size,
        num_examples=num_train_examples,
        input_seq_len=input_seq_len,
        seed=seed,
        power_a=train_power_a,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=random_non_queries
    )
    test_inputs, test_labels = _gap_power_distr_ar(
        vocab_size=vocab_size,
        num_examples=num_test_examples,
        input_seq_len=input_seq_len,
        seed=seed + 10,  # different seed for test set
        power_a=test_power_a,
        num_kv_pairs=num_kv_pairs,
        random_non_queries=random_non_queries
    )

    data = SyntheticData(
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
    )

    # check for data leakage:
    train_set = set([" ".join(map(str, x)) for x in data.train_inputs.tolist()])
    test_set = set([" ".join(map(str, x)) for x in data.test_inputs.tolist()])
    frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
    if frac_test_in_train > 0.001:
        print(
            "WARNING: Potential data leakage detected. " 
            f"{frac_test_in_train: 0.2f} of test examples are in the train set."
        )
    return data


def _gap_power_distr_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,
    random_non_queries: bool=True
) -> SyntheticData:
    """
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_kv_pairs * 4 <= input_seq_len
    SPECIAL_VOCAB = {"copy_prefix": 0}

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # compute power law
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([
        kvs, 
        # np.full((num_examples, 1), SPECIAL_VOCAB["copy_prefix"], dtype=np.int64), 
        queries
    ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    
    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    return inputs, labels