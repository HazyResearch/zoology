'''Synthetic datasets for language modeling.'''
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
from typing import Dict


def base_ar(
    vocab,
    input_seq_len: int,
    rng: np.random.Generator,
):
    """Generate sequence where the input has a sequence of key value pairs
    and the copy prefix at the end, and then a key value pair is inserted
    after the copy prefix."""
    non_special_vocab_size = len(vocab.non_special_vocab) 
    keys = vocab.non_special_vocab[:non_special_vocab_size // 2]
    values = vocab.non_special_vocab[non_special_vocab_size // 2:]
    keys = [ [key] for key in keys ]
    kv_map = {
        tuple(k): rng.choice(values) for k in keys
    }

    key_present = {}
    vocab_seq = []
    pair_length = 2
    for _ in range(input_seq_len // (pair_length)):
        k = tuple(rng.choice(list(kv_map.keys())))
        v = kv_map[k]
        vocab_seq += list(k) + [v]
        key_present[k] = True

    k = tuple(rng.choice(list(kv_map.keys())))
    while k not in key_present:
        k = tuple(rng.choice(list(key_present.keys())))
    to_copy = [vocab.copy_prefix] + list(k) + [ kv_map[k] if k in key_present else vocab.noop ]
    vocab_seq = vocab_seq + to_copy
    return " ".join(vocab_seq)


def generate_ar_dataset(
        num_examples=-1, 
        num_test_examples=-1,
        input_seq_len=None, 
        tokenizer=None, 
        vocab=None, 
        rng=None,
        ignore_train=False,
    ):
    
    train_tensor = test_tensor = None
    all_examples = []
    num_extra_seq_len = 2
    if train_tensor is None or test_tensor is None: 
        for i, (example_count, split) in enumerate(zip([num_examples, num_test_examples], ['train', 'eval'])):
            vocab_seq = base_ar(
                 vocab, input_seq_len, rng
            )
            examples = tokenizer.tokenize(vocab_seq, return_tensor=True)
            examples = torch.stack([examples['input_ids'] for _ in tqdm(range(example_count))])
            examples = torch.unique(examples, dim=0, sorted=False).tolist()
            while len(examples) < example_count:
                vocab_seq = base_ar(
                    vocab, input_seq_len, rng
                )
                example = tokenizer.tokenize(vocab_seq, return_tensor=True)
                example = example['input_ids'].tolist()
                examples.append(example)
            rng.shuffle(examples)
            all_examples.append(torch.LongTensor(examples))
        train_tensor = torch.stack([
            torch.stack([
                example[:-1], 
                example[1:]
            ]) for example in all_examples[0]])
        test_tensor = torch.stack([
            torch.stack([
                example[:-1],
                example[1:]
            ]) for example in all_examples[1]])
        if ignore_train:
            train_tensor[:, 1, :-1 * (num_extra_seq_len - 1)] = -100
        test_tensor[:, 1, :-1 * (num_extra_seq_len - 1)] = -100
    dataset = {
        'train': TensorDataset(train_tensor[:, 0, :], train_tensor[:, 1, :]),
        'test': TensorDataset(test_tensor[:, 0, :], test_tensor[:, 1, :])
    }
    return dataset


class Vocab:
    """Custom vocab."""
    def __init__(self, vocab_size: int, special_vocabs: Dict):
        # Special tokens hold copy_prefix and pad token etc
        assert "copy_prefix" in special_vocabs
        self.special_vocabs = special_vocabs
        vocab_size = vocab_size - len(special_vocabs)

        print(f"Vocab size excluding special vocab: {vocab_size}")
        print(f"Special vocabs size: {len(special_vocabs)}")
        vocab = [str(v) for v in list(range(vocab_size))]
        self.non_special_vocab = sorted(list(vocab))
        self.vocab = sorted(list(set(vocab + list(self.special_vocabs.values()))))
        self.vocab.append('-100')
        self.v2id = {v:i for i,v in enumerate(self.vocab)}
        self.v2id['-100'] = -100
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
    def special_tokens(self):
        return set(self.special_vocabs.values())

    def get_id(self, token: str):
        return self.v2id[token]

    def get_vocab(self, id: int):
        return self.vocab[id]

    def __len__(self):
        return len(self.vocab)


class Tokenizer:
    """Custom Tokenizer for our own vocab."""
    def __init__(self, vocab: Vocab, max_length=-1, 
                 len_label_tokens=1, len_copy_tokens=1):
        self.vocab = vocab
        self.max_length = max_length
        self.len_label_tokens = len_label_tokens
        self.len_copy_tokens = len_copy_tokens

    def tokenize(self, text: str, return_tensor=False, mask_input=False):
        input_ids = [self.vocab.get_id(t) for t in text.split()]
        if self.vocab.get_id(self.vocab.copy_prefix) not in input_ids:
            raise ValueError("Input text must contain copy_prefix token.")
        copy_prefix_pos = input_ids.index(self.vocab.get_id(self.vocab.copy_prefix))
        labels = input_ids
        if mask_input:
            labels = [-100] * (copy_prefix_pos+1) + labels[copy_prefix_pos+1:]
        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(labels)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def decode(self, ids: list):
        return " ".join([self.vocab.get_vocab(id) for id in ids])


class ICLDataModule:
    _name_ = "synthetics_suite"

    def __init__(
        self,
        num_examples: int,
        num_test_examples: int,
        vocab_size: int,
        input_seq_len: int,
        copy_method: str = "base_ar",
        seed: int = 0,
        batch_size: int = 32,
        num_keys: int = 1, 
        ignore_train=False,
        **dataset_kwargs,
    ):
        self.num_examples = num_examples
        self.num_test_examples = num_test_examples
        self.input_seq_len = input_seq_len
        self.vocab_size = vocab_size
        self.copy_method = copy_method
        self.ignore_train=ignore_train
        self.seed = seed
        self.batch_size = batch_size
        self.num_keys = num_keys

        # Special Tokens
        special_vocabs = {
            "copy_prefix": "=>",
        }
        self.special_vocabs = special_vocabs   
        self.vocab = Vocab(
            vocab_size, 
            special_vocabs=special_vocabs, 
        )
        self.tokenizer = Tokenizer(
            self.vocab, 
            max_length=self.input_seq_len,
        )

    def setup(self, stage=None):
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        dataset = generate_ar_dataset(
            num_examples=self.num_examples, 
            num_test_examples=self.num_test_examples,
            input_seq_len=self.input_seq_len,
            tokenizer=self.tokenizer, 
            vocab=self.vocab,
            rng=self.rng,
            ignore_train=self.ignore_train,
        )
        self.dataset = dataset        

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['train'], shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['test'], shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset['test'], shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=shuffle,
            persistent_workers=True
        )
    
