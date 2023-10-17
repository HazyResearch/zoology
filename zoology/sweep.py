from datetime import datetime
import click
import os
import importlib.util
import tempfile
import subprocess
import ray


MAX_WORKERS_PER_GPU = 1

@ray.remote(num_gpus=(1 // MAX_WORKERS_PER_GPU))
def execute_script_remote(script_path: str, outdir: str, debug: bool):
    return execute_script(script_path, outdir, debug)


def execute_script(script_path: str, outdir: str, debug: bool):
    print(f"Running {script_path}")
    result = subprocess.run(["sh", script_path], capture_output=not debug)
    print("done")

    base_name = os.path.basename(script_path)
    if outdir is not None:
        with open(os.path.join(outdir, f"{base_name}.stdout"), "wb") as out_file:
            out_file.write(result.stdout)

        with open(os.path.join(outdir, f"{base_name}.stderr"), "wb") as err_file:
            err_file.write(result.stderr)


@click.command()
@click.argument("python_file", type=click.Path(exists=True))
@click.option(
    "--outdir",
    type=click.Path(exists=True, file_okay=False, writable=True),
    default=None,
)
@click.option("--name", type=str, default="default")
@click.option("--debug", is_flag=True)
@click.option("--gpus", default=None, type=str)
def main(python_file, outdir, name: str, debug: bool, gpus: str):
    if debug:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # Initialize Ray

    if not debug:
        # ray was killing workers due to OOM, but it didn't seem to be necessary 
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        ray.init(
            ignore_reinit_error=True,

        )

    # Load the given Python file as a module
    spec = importlib.util.spec_from_file_location("config_module", python_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    base_config = config_module.base_config
    configs = config_module.configs()

    name = name + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"Running sweep {name} with {len(configs)} configs")

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_output_directory:
        # Generate a shell script for each config
        scripts = []
        for i, config in enumerate(configs):
            config = {**config, "+sweep": name}
            script_path = os.path.join(tmp_output_directory, f"script_{i}.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/sh\n")
                f.write(f"python run.py experiment={base_config} ")
                f.write(" ".join(f"{k}={v}" for k, v in config.items()))
                f.write("\n")
            scripts.append(script_path)

        # Run each script in parallel using Ray
        if debug:
            for script in scripts: 
                execute_script(script, outdir, debug)
        else:
            futures = [execute_script_remote.remote(script, outdir, debug) for script in scripts]
            ray.get(futures)

    ray.shutdown()


if __name__ == "__main__":
    main()
