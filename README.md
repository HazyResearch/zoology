<div align="center" >
    <img src="banner.png" height=150 alt="Meerkat logo" style="margin-bottom:px"/> 

[![GitHub](https://img.shields.io/github/license/HazyResearch/meerkat)](https://img.shields.io/github/license/HazyResearch/meerkat)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Understand and test machine learning architectures on synthetic tasks.**


</div>

Zoology provides machine learning researchers with a simple playground for understanding and testing neural network architectures on synthetic tasks. Simplicity is our main design goal: limited dependencies, architecture implementations that are easy to understand, and a straightforward process for adding new synthetic tasks. 

---

*Why did we make Zoology?* In our research on efficient language models, synthetic tasks have been crucial for understanding and debugging issues before scaling up to expensive pretraining runs. So, we're releasing the code we've used alongside instructions for replicating a lot of our experiments and their WandB logs. 

*Is Zoology a good fit for your use case?* If you are looking to actually train a large machine learning model, Zoology's training harness (which is optimized for simplicity) is certainly not a good fit. For our language model research, we've found the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) useful for this. That being said, you might still want to use some of Zoology's layer implementations or maybe even mix the synthetic tasks into your training distribution. 

## Getting started

**Installation.** First, ensure you have torch installed, or install it following the instructions [here](https://pytorch.org/get-started/locally/). Then, install Zoology with:
 
```bash
git clone https://github.com/HazyResearch/zoology.git
cd zoology
pip install -e . 
```
We want to keep this install as lightweight as possible; the only required dependencies are: `torch, einops, tqdm, pydantic, wandb`. There is some extra functionality (*e.g.* launching sweeps in parallel with Ray) that require additional dependencies. To install these, run `pip install -e .[extra,analysis]`.

**Training your first model.** Follow along in this colab

## Experiments and Sweeps
In this section, we'll walk through how to configure experiment and launch sweeps. 

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
```
fn_config = FunctionConfig(name="torch.sort", kwargs={"descending": True})
fn = fn_config.instantiate()
fn(torch.tensor([2,4,3]) # [4, 3, 2]
```

*Launching experiments.* 

*Launching sweeps.*





## Data
In this section, we'll walk through how to create a new synthetic task and discuss some of the tasks that are already implemented.

## Models
In this section, we'll walk through the proces of implementing a new model architecture. 
The library 




## About 

This repo is being developed by members of the HazyResearch group. 



