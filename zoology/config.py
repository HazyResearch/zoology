import argparse
from datetime import datetime
from functools import partial

from pydantic import BaseModel
import yaml


from zoology.utils import import_from_str

class BaseConfig(BaseModel):
    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument('--config', type=str, default=None, help='Path to the config file')
        parser.add_argument('--run_id', type=str, default=None, help='Run ID for the training')
        args, extra_args = parser.parse_known_args()


        if args.config is not None:
            with open(args.config) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            config = {}
        
        # Override with any extra arguments from the command line
        def _nested_update(config, args):
            for key, value in args.items():
                keys = key.split(".")
                for key in keys[:-1]:
                    config = config.setdefault(key, {})
                config[keys[-1]] = value

        extra_args = dict([arg.lstrip("-").split("=") for arg in extra_args])
        extra_args = {k.replace("-", "_"): v for k, v in extra_args.items()}
        _nested_update(config, extra_args)
        config = cls.parse_obj(config)

        if config.run_id is None:
            config.run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return config


class FunctionConfig(BaseConfig):
    name: str
    kwargs: dict = {}

    def instantiate(self):
        return partial(import_from_str(self.name), **self.kwargs)

class ModuleConfig(BaseConfig):
    name: str
    kwargs: dict = {}

    def instantiate(self, **kwargs):
        return import_from_str(self.name)(**kwargs, **self.kwargs)


class DataConfig(BaseConfig):
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

class ModelConfig(BaseConfig):
    sequence_mixer: ModuleConfig = None

    d_model: int = 128
    n_layers: int = 2
    max_position_embeddings: int = 64
    vocab_size: int=128

    # num_heads=12,
    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    drop_path: float = 0.0
    layer_norm_epsilon: float = 1e-5
    pad_vocab_size_multiple: int = 1
    # block=None,


class TrainConfig(BaseConfig):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()

    
    max_epochs: int = 100
    learning_rate: float = 1e-3


    run_id: str = None




