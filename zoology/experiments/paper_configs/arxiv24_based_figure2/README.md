
## Reproducing paper experiments
In this section, we'll show how to reproduce Figure 2 in our paper [Simple linear attention balanced the recall-throughput tradeoff]().
 
<div align="center" >
    <img src="figure.png" height=150 alt="Figure 2" style="margin-bottom:px"/> 
</div>


To reproduce these results, ensure you have WandB setup to log all the results and then run the command:
```
python -m zoology.launch zoology/experiments/arxiv24_based_figure2/configs.py -p
```

