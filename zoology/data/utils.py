import os 
import hashlib
from pathlib import Path
import json
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Union
import numpy as np

import torch 
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, Dataset

from zoology.config import DataConfig, DataSegmentConfig

import random

@dataclass
class DataSegment:
    inputs: torch.Tensor
    labels: torch.Tensor
    slices: Dict[str, any] = None

    def __len__(self):
        assert len(self.inputs) == len(self.labels) 
        return len(self.inputs)

    @classmethod
    def from_config(
        cls, 
        config: DataSegmentConfig,
        cache_dir: str = None,
        force_cache: bool = False,
        seed: int = 123
    ):
        """
        Loads a data segment. 
        This function checks if a cache directory is available and if the data is already 
        cached. If the data is cached, it loads the data from the cache. If not, it 
        generates the data using the provided configuration. The generated data is then 
        saved to the cache for future use. The function also checks if the shapes of the 
        data are correct. Finally, it prepares the data loaders for training and testing.
        
        Args: 
            config (DataConfig): The configuration object containing all the necessary parameters to prepare the data.
        Returns: 
            Tuple[DataLoader, DataLoader]: A tuple containing the training and testing data loaders.
        Raises: 
            ValueError: If the shapes of the data are not correct.
        Example: 
            >>> config = DataConfig(…) 
            >>> train_dl, test_dl = SyntheticData.from_config(config).dataloaders()
        """
        def _get_cache_path(config: DataSegmentConfig):
            if cache_dir is None:
                return None
            config_hash = hashlib.md5(
                json.dumps({**config.model_dump(), "_seed": seed}, sort_keys=True).encode()
            ).hexdigest()

            return os.path.join(
                cache_dir,
                f"data_{config_hash}.pt",
            )
        
        if cache_dir is not None:
            try:
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
            except:
                print(f"Could not create cache directory {cache_dir}")
                cache_dir = None
        cache_path = _get_cache_path(config)
        # check cache
        if cache_dir is not None and os.path.exists(cache_path) and not force_cache:
            # load from cache
            print(f"Loading data from on-disk cache at {cache_path}...") 
            # SE 09-12-23: there's some sporadic issue in torch load that gives
            # RuntimeError: PytorchStreamReader failed reading file data/2: file read failed
            MAX_RETRIES = 10
            for _ in range(MAX_RETRIES):
                try:
                    data = cls(**torch.load(cache_path))
                    break
                except RuntimeError as e:
                    print(e)
        else:
            print(f"Generating dataset...") 
            # generate data
            data: DataSegment = config.build(seed=seed)

            if cache_dir is not None:
                print(f"Saving dataset to on-disk cache at {cache_path}...") 
                torch.save(asdict(data), cache_path)
        return data


def prepare_data(config: DataConfig) -> Tuple[DataLoader, DataLoader]:  
    # support different batch sizes for train and test
    if isinstance(config.batch_size, int):
        train_batch_size, test_batch_size = (config.batch_size, config.batch_size)
    else:
        train_batch_size, test_batch_size = config.batch_size
    
    # We set a different random seed for each data segment. We're careful to avoid using
    # the same seed for the train and test data segments.
    MAX_SEED = 2 ** 32
    np.random.seed(config.seed)
    train_seeds = np.random.randint(0, MAX_SEED // 2, size=len(config.train_configs))
    test_seeds = np.random.randint(MAX_SEED // 2, MAX_SEED, size=len(config.test_configs))
    factory_kwargs = {"cache_dir": config.cache_dir, "force_cache": config.force_cache}
    train_segments = _SyntheticDataset([
        DataSegment.from_config(segment_config, seed=int(seed), **factory_kwargs)
        for segment_config, seed in zip(config.train_configs, train_seeds)
    ], batch_size=train_batch_size)
    test_segments = _SyntheticDataset([
        DataSegment.from_config(segment_config, seed=int(seed), **factory_kwargs)
        for segment_config, seed in zip(config.test_configs, test_seeds)
    ], batch_size=test_batch_size)

    return (
        DataLoader(ds, batch_size=None, num_workers=0,  shuffle=False)
        for ds in [train_segments, test_segments]
    )


class _SyntheticDataset(Dataset):
    """Simple torch dataset that returns batches instead of individual examples. 
    This is needed to support data that contains different data segments not to be
    mixed. 
    """
    def __init__(self, segments: List[DataSegment], batch_size: int):
        self.segments = segments
        self.batch_size = batch_size        
        self.batches = [
            (segment_idx, batch_start)
            for segment_idx, segment in enumerate(self.segments)
            for batch_start in range(0, len(segment), self.batch_size)
        ]

    def __getitem__(self, batch_idx: int):
        segment_idx, batch_start = self.batches[batch_idx]
        segment = self.segments[segment_idx]
        slc = slice(batch_start, batch_start + self.batch_size)
        
        slices = [segment.slices if segment.slices is not None else {}] * self.batch_size
        return segment.inputs[slc], segment.labels[slc], slices      

    def __len__(self):
        return len(self.batches)

# def builder_from_single(single_fn: callable):
#     def _build_from_single(
#         num_train_examples: int,
#         num_test_examples: int,
#         vocab_size: int,
#         input_seq_len: int,
#         seed: int,
#         **kwargs
#     ):
#         result = {}
#         for split, num_examples in [("train", num_train_examples), ("test", num_test_examples)]:
#             # TODO: we probably wanna parallelize this
#             inputs, labels = [], []
#             rng = np.random.default_rng(seed + (0 if split == "train" else 1))
#             for _ in tqdm(range(num_examples)):
#                 input, label = single_fn(
#                     vocab_size=vocab_size,
#                     input_seq_len=input_seq_len,
#                     rng=rng,
#                     **kwargs
#                 )
#                 inputs.append(input)
#                 labels.append(label)
#             result[f"{split}_inputs"] = torch.stack(inputs)
#             result[f"{split}_labels"] = torch.stack(labels)
#         return SyntheticData(**result)
        
#     return _build_from_single

