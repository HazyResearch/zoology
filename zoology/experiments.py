import argparse
import random

import torch
import numpy as np 
import yaml
from datetime import datetime

from zoology.task import LMSynthetic
from zoology.data.utils import prepare_data
from zoology.config import Config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# def get_args():
#     parser = argparse.ArgumentParser(description = "arguments")

#     # mode
#     parser.add_argument("--do_profile", action="store_true", help="run profiling")

#     # training configs
#     parser.add_argument("-t", "--task", required=True, help = "task name")
#     parser.add_argument("-s", "--seed", default=0, help = "random seed")
#     parser.add_argument("-e", "--epochs", default=20, type=int, help = "number of epochs")
#     parser.add_argument("-lr", "--learning_rate", default=0.0005, type=float, help="learning rate")
#     parser.add_argument("-tb", "--train_batch", default=32, type=int, help="training batch size")
#     parser.add_argument("-l",  "--layers", default=4, type=int, help="number of layers")

#     # dataset
#     parser.add_argument("-n_train", "--n_train", default=10000, type=int, help = "number of training examples")
#     parser.add_argument("-n_test", "--n_test", default=500, type=int, help = "number of test examples")
#     parser.add_argument("-v", "--vocab_size", default=20, type=int, help = "vocab size")
#     parser.add_argument("-is", "--input_seq_len", default=100, type=int, help = "input sequence length")

#     args = parser.parse_args()
#     return args

def set_determinism(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)



def create_task_instance(args):
    task = None
    task_name = args.task
    train_dataloader, test_dataloader = prepare_data(args)

    if(task_name=='lm_synthetic'):
        task = LMSynthetic(
            input_seq_len=args.input_seq_len,
            vocab_size=args.vocab_size,
            device=device, 
            task=args.task, 
            epochs=args.epochs, 
            lr=args.learning_rate, 
            train_batch=args.train_batch,
            train_data=train_dataloader,
            test_data=test_dataloader,
            n_layers=args.layers,
        )
    else:
        assert False, f"Task {task_name} not found"
    return task

# if __name__ == "__main__":
#     args = get_args()
#     set_determinism(args)
#     task = create_task_instance(args)
#     task.load_model()

#     if args.do_profile:
#         task.run_profile()
#     else:
#         task.run()



def main():
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
    config = Config.parse_obj(config)

    if config.run_id is None:
        config.run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print(config)

    train_dataloader, test_dataloader = prepare_data(config.data)

    task = LMSynthetic(
        input_seq_len=config.data.input_seq_len,
        vocab_size=config.data.vocab_size,
        device=device, 
        epochs=config.epochs, 
        lr=config.learning_rate, 
        train_batch=config.train_batch,
        train_data=train_dataloader,
        test_data=test_dataloader,
        n_layers=config.layers,
    )
    task.load_model()
    task.run()

    # train(config)
    

if __name__ == "__main__":
    main()