<div align="center" >
    <img src="banner.png" height=150 alt="Meerkat logo" style="margin-bottom:px"/> 

[![GitHub](https://img.shields.io/github/license/HazyResearch/meerkat)](https://img.shields.io/github/license/HazyResearch/meerkat)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**A minimal repo for modeling synthetic languages.**

---

</div>

Zoology provides machine learning researchers with a simple playground for testing architectures on synthetic language data. 

## Getting started

**Installation.** First, ensure you have torch installed, or install it following the instructions [here](https://pytorch.org/get-started/locally/).
 
```bash
git clone https://github.com/HazyResearch/zoology.git
cd zoology
pip install -e . 
```
We want to keep this install as lightweight as possible; the only required dependencies are: `torch, einops, tqdm, pydantic, wandb`. There is some extra functionality (*e.g.* launching sweeps in parallel with Ray) that require additional dependencies. To install these, run `pip install -e .[extra,analysis]`.

**Training your first model.** Follow along in this colab

## Experiments and Sweeps
In this section, we'll walk through how to configure experiment and launch sweeps. 

## Data
In this section, we'll walk through how to create a new synthetic task and discuss some of the tasks that are already implemented.

## Models
In this section, we'll walk through the proces of implementing a new model architecture. 
The library 




## About 

This repo is being developed by members of the HazyResearch group. 



