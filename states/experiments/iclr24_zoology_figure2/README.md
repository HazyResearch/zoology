
## Reproducing paper experiments
 In this section, we'll show how to reproduce the results in our paper *[Zoology: Measuring and Improving Recall in Efficient Language Models](https://arxiv.org/abs/2312.04927)* and [blogpost](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis).
 
 The main synthetic data results in our work are summarized in Figure 2. The x-axis is the model dimension and the y-axis is accuracy on Mqar. Increasing the sequence
length correlates with increased task difficulty. The results shown are the maximum performance for each model over four learning rates.
<div align="center" >
    <img src="assets/figure2.png" height=150 alt="Figure 2" style="margin-bottom:px"/> 
</div>

To reproduce these results, ensure you have WandB setup to log all the results and then run the command:
```
python -m zoology.launch zoology/experiments/paper/figure2.py -p
```
Note that there are 448 model/data configurations in this sweep, so it takes a while to run. We ran most of our experiments on an 8xA100 with the `-p` flag, which launches configurations in parallel. To run a smaller scale experiment, you can modify the loops in `figure2.py` file to only include a subset of the configurations you're interested in (*e.g.* you can drop some models, sequence lengths, or learning rates). For more details on how the experiments are configured, see the [configuration section](#configuration-experiments-and-sweeps).

To produce the plot after the run, see the plotting code `zoology/analysis/paper/figure2.py`.