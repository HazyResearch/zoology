<div align="center" >
    <img src="assets/banner.png" height=150 alt="Meerkat logo" style="margin-bottom:px"/> 

[![GitHub](https://img.shields.io/github/license/HazyResearch/meerkat)](https://img.shields.io/github/license/HazyResearch/meerkat)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Understand and test language model architectures on synthetic tasks.**


</div>

Zoology provides machine learning researchers with a simple playground for understanding and testing language model architectures on synthetic tasks. This repository can be used to reproduce the results in our paper *[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)*. See the section on [reproducing paper experiments](#reproducing-paper-experiments) for details.

---

*Why did we make Zoology?* In our research on efficient language models, synthetic tasks have been crucial for understanding and debugging issues before scaling up to expensive pretraining runs. So, we're releasing the code we've used alongside instructions for replicating a lot of our experiments and their WandB logs.  Simplicity is our main design goal: limited dependencies, architecture implementations that are easy to understand, and a straightforward process for adding new synthetic tasks. 

*Is Zoology a good fit for your use case?* If you are looking to actually train a large machine learning model, Zoology's training harness (which is optimized for simplicity) is certainly not a good fit. For our language model research, we've found the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) useful for this. That being said, you might still want to use some of Zoology's layer implementations or maybe even mix the synthetic tasks into your training distribution. 

*I want to explore the Based architecture. How should I get started?* See our repository at [HazyResearch/based](https://github.com/HazyResearch/based) for the code we used to train and evaluate large Based language models. If you would like to reproduce the synthetic experiments from the Based paper, this is the right repository! See [zoology/experiments/arxiv24_based_figure2/README.md]() for instructions on how to reproduce the results. 


## Getting started

**Installation.** First, ensure you have torch installed, or install it following the instructions [here](https://pytorch.org/get-started/locally/). Then, install Zoology with:
 
```bash
git clone https://github.com/HazyResearch/zoology.git
cd zoology
pip install -e .[extra,analysis] 
```
If you want to keep this install as lightweight as possible; the only required dependencies are: `torch, einops, tqdm, pydantic, wandb`. There is some extra functionality (*e.g.* launching sweeps in parallel with Ray) that require additional dependencies. To install without the optional dependencies, run `pip install -e .`.

Then, try running an example experiment with: 
```
python -m zoology.launch zoology/experiments/examples/basic.py
```
This will train a simple two layer transformer on multi-query associative recall. To run a sweep over learning rates, try: 
```
python -m zoology.launch zoology/experiments/examples/basic_sweep.py
```
If you have access to multiple GPUs, you can run the sweep in parallel by adding the `-p` flag.


## Reproducing paper experiments
This repository has been used to produce results in a few papers on efficient language models. 
The configs, instructions and plotting code for reproducing the figures in these papers are provided in the following sub-folders. 

- [Zoology: Measuring and improving recall in efficient language models](https://arxiv.org/abs/2312.04927)
    - [zoology/experiments/iclr24_zoology_figure2]()
- [Based: Simple linear attention balances the recall-throughput tradeoff]()
    - [zoology/experiments/arxiv24_based_figure2]()
    - [zoology/experiments/arxiv24_based_figure3]()

## Configuration, Experiments, and Sweeps
In this section, we'll walk through how to configure an experiment and launch sweeps. 

*Configuration*. Models, data, and training are controlled by configuration objects. For details on available configuration fields, see the configuration definition in [`zoology/config.py`](zoology/config.py). The configuration is a nested Pydantic model, which can be instantiated as follows:
```python
from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig, FunctionConfig

config = TrainConfig(
    max_epochs=20,
    data=DataConfig(
        vocab_size=128,
        builder=FunctionConfig(
            name="zoology.data.associative_recall.gap_power_distr_ar",
            kwargs={"num_kv_pairs": 4}
        ),
        
    ),
    model=ModelConfig(
        vocab_size=128,
        sequence_mixer=ModuleConfig("name": "zoology.mixers.attention.MHA"}
    ),
)
```
Note that the `FunctionConfig` and `ModuleConfig` are special objects that configure partial functions and PyTorch modules, respectively. 
They both have an `instantiate()` method that will import the function or class passed to `name` and partial or instantiate it with `kwargs`.
For example, 
```python
fn_config = FunctionConfig(name="torch.sort", kwargs={"descending": True})
fn = fn_config.instantiate()
fn(torch.tensor([2,4,3])) # [4, 3, 2]
```

*Launching experiments.* To launch an experiment from the command line, define a configuration object in python file and store it in a global variable `configs`:
```python
config = TrainConfig(...)
configs = [config]
```
See [`zoology/experiments/examples/basic.py`](zoology/experiments/examples/basic.py) for an example. 

Then run `python -m zoology.launch zoology/experiments/examples/basic.py`, replacing `basic.py` with the path to your experiment. This will launch a single training job. 


*Launching sweeps.* To launch a sweep, simply add more configuration objects to the `configs` list. For example, here's the content of [`zoology/experiments/examples/basic_sweep.py`](zoology/experiments/examples/basic_sweep.py):
```python
import numpy as np
from zoology.config import TrainConfig

configs = []
for lr in np.logspace(-4, -2, 10):
   configs.append(TrainConfig(learning_rate=lr)) 
```
You can then run `python -m zoology.launch zoology/experiments/examples/basic_sweep.py`. This will launch a sweep with 10 jobs, one for each configuration.

*Launching sweeps in parallel.* If you have multiple GPUs on your machine, you can launch sweeps in parallel across your devices. 
To launch sweeps in parallel, you'll need to install [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html): `pip install -e.[extras]`. 
Then, you can run `python -m zoology.launch zoology/experiments/basic_sweep.py -p`. 
This will run the configurations in parallel using a pool of workers, one per GPU.

*Logging.* Zoology uses [Weights and Biases](https://wandb.ai/site) for logging. You'll need to login with `wandb login` and update the `LoggerConfig` in your configuration to point to your project: 
```python
from zoology.config import TrainConfig, LoggerConfig

TrainConfig(
    logger=LoggerConfig(
        project="my_wandb_project",
        entity="my_wandb_entity",
    ),
    ...
)
```

## Data
In this section, we'll walk through how to create a new synthetic task and discuss some of the tasks that are already implemented.

*Creating a new task.* To create a new task, you'll need to implement a data builder function with the following signature:
```python
from zoology.utils.data import SyntheticData
def my_data_builder(
    num_train_examples: int,
    num_test_examples: int,
    vocab_size: int,
    input_seq_len: int,
    seed: int,
    **kwargs
) -> SyntheticData:
    ...
```
You can add any additional arguments to the function signature, but the ones above are required.

The function must return a `SyntheticData` object, which is a simple dataclass with the following fields:
```python
@dataclass
class SyntheticData:
    """Simple dataclass which specifies the format that should be returned by
    the synthetic data generators.

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
```
The inputs and labels should be integer tensors with values in the range `[0, vocab_size)`. 

You can create this function in any file you want, as long as it's importable. Let's
assume that we've created a file `zoology/data/my_task.py` and written our `my_data_builder` function there.
Then, we can add it to our data configuration with: 
```python
from zoology.config import TrainConfig, DataConfig, FunctionConfig
config = TrainConfig(
    DataConfig(
        vocab_size=128,
        builder=FunctionConfig(name="zoology.data.my_task.my_data_builder"),
            kwargs={"my_data_builder_kwarg": 4}
        ),
    ),
)
```
When you launch an experiment with this configuration, the `my_data_builder` function will be imported and called with the specified arguments, constructing the dataset. 

**Caching dataset creation.** Sometimes it's useful to cache the dataset creation process, especially if it's expensive. To do so you can pass a `cache_dir` to the `DataConfig`: `DataConfig(..., cache_dir="my_cache_dir")`.


## About 

This repo is being developed by members of the HazyResearch group. 

If you use this codebase, or otherwise found our work valuable, please cite:
```
@article{zoology2023,
  title={Zoology: Measuring and Improving Recall in Efficient Language Models},
  author={Arora, Simran and Eyuboglu, Sabri and Timalsina, Aman and Johnson, Isys and Poli, Michael and Zou, James and Rudra, Atri and Ré, Christopher},
  journal={	arXiv:2312.04927},
  year={2023}
}
```



