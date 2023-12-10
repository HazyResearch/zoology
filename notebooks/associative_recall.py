from typing import List, Tuple, Union

import numpy as np
import torch
import random
from multiprocessing import Pool
import random
from typing import Dict
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class SyntheticData:
    """Simple dataclass which specifies the format that should be returned by
    the synthetic data generators.

    All tensors (train_inputs, train_labels, test_inputs, test_labels) should be
    have two axes and share the same second dimension length.

    Args:
        train_inputs (torch.Tensor): Training inputs of shape (num_train_examples, input_seq_len)
        train_labels (torch.Tensor): Training labels of shape (num_train_examples, input_seq_len)
        test_inputs (torch.Tensor): Test inputs of shape (num_test_examples, input_seq_len)
        test_labels (torch.Tensor): Test labels of shape (num_test_examples, input_seq_len)
    """

    train_inputs: torch.Tensor
    train_labels: torch.Tensor
    test_inputs: torch.Tensor
    test_labels: torch.Tensor

    def check_shapes(
        self,
        num_train_examples: int,
        num_test_examples: int,
        input_seq_len: int,
    ):
        """Check that the shapes are correct
        this is useful to catch bugs in the data generation code because
        downstream errors due to incorrectly shaped can be tricky to debug.
        """
        if self.train_labels.shape != (num_train_examples, input_seq_len):
            raise ValueError(
                f"train_labels shape is {self.train_labels.shape} but should be {(num_train_examples, input_seq_len)}"
            )

        if self.train_labels.shape != (num_train_examples, input_seq_len):
            raise ValueError(
                f"train_labels shape is {self.train_labels.shape} but should be {(num_train_examples, input_seq_len)}"
            )

        if self.test_inputs.shape != (num_test_examples, input_seq_len):
            raise ValueError(
                f"test_inputs shape is {self.test_inputs.shape} but should be {(num_test_examples, input_seq_len)}"
            )

        if self.test_labels.shape != (num_test_examples, input_seq_len):
            raise ValueError(
                f"test_labels shape is {self.test_labels.shape} but should be {(num_test_examples, input_seq_len)}"
            )


class SyntheticVocab:
    """Simple integer vocab with optional special tokens.."""

    def __init__(self, vocab_size: int, special_vocabs: Dict, do_padding=False):
        # Special tokens hold copy_prefix and noop/pad token etc
        assert "copy_prefix" in special_vocabs
        self.special_vocabs = special_vocabs

        if do_padding:
            special_vocabs["pad"] = "<pad>"

        vocab_size = vocab_size - len(special_vocabs)

        vocab = [str(v) for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(list(set(vocab + list(self.special_vocabs.values()))))
        self.v2id = {v: i for i, v in enumerate(self.vocab)}
        self.vocab_size = len(vocab)

    def get_next_vocab(self, token: str):
        """Gets next token excluding special_vocabs."""
        id = (self.get_id(token) + 1) % self.vocab_size
        while self.get_vocab(id) in self.special_vocabs:
            id = (id + 1) % self.vocab_size
        return self.get_vocab(id)

    @property
    def copy_prefix(self):
        return self.special_vocabs["copy_prefix"]

    @property
    def noop(self):
        return self.special_vocabs["noop"]

    @property
    def pad(self):
        return self.special_vocabs["pad"]

    @property
    def special_tokens(self):
        return set(self.special_vocabs.values())

    def get_id(self, token: str):
        return self.v2id[token]

    def get_vocab(self, id: int):
        return self.vocab[id]

    def __len__(self):
        return len(self.vocab)


class SyntheticTokenizer:
    """Simple tokenizer that splits on space for our own vocab."""

    def __init__(self, vocab: SyntheticVocab, do_padding=False, max_length=-1):
        self.vocab = vocab
        self.do_padding = do_padding
        self.max_length = max_length

    def tokenize(self, text: str, return_tensor=False, mask_input=False):
        """Note this will perform padding on the left."""
        input_ids = [self.vocab.get_id(t) for t in text.split()]

        # Pad the input sequence on the left
        if self.do_padding:
            # print(f"Adding {self.max_length + 3 - len(input_ids)} padding tokens to input sequence of length {len(input_ids)}")
            input_ids = [self.vocab.get_id(self.vocab.pad)] * (
                self.max_length - len(input_ids)
            ) + input_ids
            # print(f"Input sequence length after padding: {len(input_ids)}")

        if self.vocab.get_id(self.vocab.copy_prefix) not in input_ids:
            raise ValueError("Input text must contain copy_prefix token.")
        copy_prefix_pos = input_ids.index(self.vocab.get_id(self.vocab.copy_prefix))
        labels = input_ids
        if mask_input:
            # Mask the input tokens for loss but do not mask the copied token
            labels = [-100] * (copy_prefix_pos + 1) + labels[copy_prefix_pos + 1 :]
        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)

        # print(f"Length of input_ids: {len(input_ids)}\n")
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def decode(self, ids: list):
        return " ".join([self.vocab.get_vocab(id) for id in ids])


def _generate_example_out(args):
    vocab, seed, input_seq_len, example_generator, tokenizer, kwargs = args
    rng = np.random.default_rng(seed)
    result = example_generator(
        vocab=vocab,
        rng=rng,
        input_seq_len=input_seq_len,
        **kwargs,
    )
    tokenized_example = tokenizer.tokenize(result, return_tensor=True)
    return tokenized_example["input_ids"].tolist()


def generate_examples(
    example_generator: callable,
    num_train_examples: int,
    num_test_examples: int,
    vocab_size: int,
    input_seq_len: int,
    seed: int,
    special_vocabs: Dict = {"copy_prefix": "=>"},
    **kwargs
) -> SyntheticData:
    data = {}
    for split, num_examples in [
        ("train", num_train_examples),
        ("test", num_test_examples),
    ]:
        vocab = SyntheticVocab(
            vocab_size, do_padding=True, special_vocabs=special_vocabs
        )
        tokenizer = SyntheticTokenizer(
            # add one to account for the fact we drop the final token from input
            vocab,
            do_padding=True,
            max_length=input_seq_len + 1,
        )

        original_input_seq_len = input_seq_len
        forbidden_lengths = [original_input_seq_len]

        if example_generator.__name__ in ["train_test_length_shift_ar"]:
            new_input_seq_len = random.randint(input_seq_len // 2, input_seq_len)
            while new_input_seq_len in forbidden_lengths:
                new_input_seq_len = random.randint(input_seq_len // 2, input_seq_len)
            forbidden_lengths.append(new_input_seq_len)
            input_seq_len = new_input_seq_len

        # ensure uniqueness
        examples = []
        np.random.seed(seed + int(split == "test"))
        seeds = np.random.randint(0, 1e9, num_examples)
        with Pool(12) as p:
            args =[
                (vocab, seeds[i], input_seq_len, example_generator, tokenizer, kwargs)
                for i in range(num_examples)
            ]
            print("Generating examples...")
            examples = p.map(
                _generate_example_out, args
            )
            print(f"done generating {len(examples)} examples.")


        # offset the inputs and labels by one for autoregressive language modeling
        data[f"{split}_inputs"] = torch.tensor(examples)[:, :-1]
        data[f"{split}_labels"] = torch.tensor(examples)[:, 1:]

        if split == "test":
            data[f"{split}_labels"][:, :-1] = -100
    return SyntheticData(**data)


def generate_examples_with_labels(
    example_generator: callable,
    num_train_examples: int,
    num_test_examples: int,
    vocab_size: int,
    input_seq_len: int,
    seed: int,
    special_vocabs: Dict = {"copy_prefix": "=>"},
    **kwargs
) -> SyntheticData:
    data = {}
    for split, num_examples in [
        ("train", num_train_examples),
        ("test", num_test_examples),
    ]:
        vocab = SyntheticVocab(
            vocab_size, do_padding=True, special_vocabs=special_vocabs
        )
        tokenizer = SyntheticTokenizer(
            # add one to account for the fact we drop the final token from input
            vocab,
            do_padding=True,
            max_length=input_seq_len,
        )
        # ensure different seed
        rng = np.random.default_rng(seed + int(split == "test"))

        def _generate_example():
            input_ids, labels = example_generator(
                vocab=vocab,
                rng=rng,
                input_seq_len=input_seq_len,
                **kwargs,
            )
            input_ids = tokenizer.tokenize(input_ids, return_tensor=True)["input_ids"]
            labels = tokenizer.tokenize(labels, return_tensor=True)["input_ids"]
            return input_ids, labels

        original_input_seq_len = input_seq_len
        forbidden_lengths = [original_input_seq_len]

        if example_generator.__name__ in ["train_test_length_shift_ar"]:
            new_input_seq_len = random.randint(input_seq_len // 2, input_seq_len)
            while new_input_seq_len in forbidden_lengths:
                new_input_seq_len = random.randint(input_seq_len // 2, input_seq_len)
            forbidden_lengths.append(new_input_seq_len)
            input_seq_len = new_input_seq_len

        # ensure uniqueness
        inputs, labels = [], []
        with tqdm(total=num_examples, desc=f"Generating {split} examples") as pbar:
            while len(inputs) < num_examples:
                input, label = _generate_example()
                input = input.tolist()
                label = label.tolist()
                if input not in inputs:
                    inputs.append(input)
                    labels.append(label)
                    pbar.update(1)

        # offset the inputs and labels by one for autoregressive language modeling
        data[f"{split}_inputs"] = torch.tensor(inputs)
        data[f"{split}_labels"] = torch.tensor(labels)

        if split == "test":
            data[f"{split}_labels"][:, :-1] = -100
    return SyntheticData(**data)


def gap_power_distr_ar(
    vocab_size: int,
    num_train_examples: int,
    num_test_examples: int,
    input_seq_len: int,
    seed: int,
    num_kv_pairs: int,
    train_power_a: float=0.01,
    test_power_a: float=0.01,
):
    train_inputs, train_labels = _gap_power_distr_ar(
        vocab_size=vocab_size,
        num_examples=num_train_examples,
        input_seq_len=input_seq_len,
        seed=seed,
        power_a=train_power_a,
        num_kv_pairs=num_kv_pairs
    )
    test_inputs, test_labels = _gap_power_distr_ar(
        vocab_size=vocab_size,
        num_examples=num_test_examples,
        input_seq_len=input_seq_len,
        seed=seed + 10,  # different seed for test set
        power_a=test_power_a,
        num_kv_pairs=num_kv_pairs
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
    queries = np.zeros((num_examples, input_seq_len - context_size), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
    examples = np.concatenate([
        kvs, 
        np.full((num_examples, 1), SPECIAL_VOCAB["copy_prefix"], dtype=np.int64), 
        queries
    ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 2, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    
    # replace all the 0 with random values
    # inputs[inputs == 0] = np.random.choice(, size=(inputs == 0).sum())
    inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return inputs, labels
