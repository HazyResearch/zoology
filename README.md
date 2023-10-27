<div align="center" style="margin-bottom:-40px">
    <img src="banner.png" height=150 alt="Meerkat logo"/> 

---

![GitHub Workflow Status](https://github.com/HazyResearch/meerkat/actions/workflows/.github/workflows/ci.yml/badge.svg)
[![GitHub](https://img.shields.io/github/license/HazyResearch/meerkat)](https://img.shields.io/github/license/HazyResearch/meerkat)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A minimal repo for modeling synthetic languages. 
</div>

## Getting started

**Installation.** First, ensure you have torch installed, or install it following the instructions [here](https://pytorch.org/get-started/locally/).
 
```bash
git clone https://github.com/HazyResearch/zoology.git
cd zoology
pip install -e . 
```
We want to keep this install as lightweight as possible; the only required dependencies are: `torch, einops, tqdm, pydantic, wandb`. There is some extra functionality (*e.g.* launching sweeps in parallel with Ray) that require additional dependencies. To install these, run `pip install -e .[extra,analysis]`.

**Running an experiment.** Follow along in this notebook  2 

## Data



## Mixer2 





