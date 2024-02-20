from datetime import datetime
import os
import importlib.util

import click
from tqdm import tqdm

from zoology.train import train
from zoology.config import TrainConfig


MAX_WORKERS_PER_GPU = 1


def execute_config(config: TrainConfig):
    try: 
        train(config=config)
    except Exception as e:
        return config, e
    return config, None


@click.command()
@click.argument("python_file", type=click.Path(exists=True))
@click.option(
    "--outdir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default=None,
)
@click.option("--name", type=str, default="default")
@click.option("-p", "--parallelize", is_flag=True)
@click.option("--gpus", default=None, type=str)
def main(python_file, outdir, name: str, parallelize: bool, gpus: str):

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


    # Load the given Python file as a module
    spec = importlib.util.spec_from_file_location("config_module", python_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    configs = config_module.configs
    for config in configs:
        config.launch_id = f"{name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    use_ray = parallelize and len(configs) > 0
    if use_ray:
        import ray
        # ray was killing workers due to OOM, but it didn't seem to be necessary 
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    name = name + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"Running sweep {name} with {len(configs)} configs")

    # Run each script in parallel using Ray
    if not use_ray:
        for config in configs: 
            train(config)
    else:
        completed = 0
        failed = 0
        total = len(configs)
        print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        remote = ray.remote(num_gpus=(1 // MAX_WORKERS_PER_GPU))(execute_config)
        futures = [remote.remote(config) for config in configs]
        
        while futures:
            complete, futures = ray.wait(futures)
            for config, error in ray.get(complete):
                if error is not None:
                    failed += 1
                    config.print()
                    print(error)
                completed += 1
            print(f"Completed: {completed} ({completed / total:0.1%} -- {failed} failed) | Total: {total}")

        ray.shutdown()



if __name__ == "__main__":
    main()
