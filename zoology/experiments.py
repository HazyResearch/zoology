import argparse
import torch
import numpy as np
import random 
from task import LMSynthetic
from zoology.data.utils import prepare_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

def get_args():
    parser = argparse.ArgumentParser(description = "arguments")

    # mode
    parser.add_argument("--do_profile", action="store_true", help="run profiling")

    # training configs
    parser.add_argument("-t", "--task", required=True, help = "task name")
    parser.add_argument("-s", "--seed", default=0, help = "random seed")
    parser.add_argument("-e", "--epochs", default=20, type=int, help = "number of epochs")
    parser.add_argument("-lr", "--learning_rate", default=0.0005, type=float, help="learning rate")
    parser.add_argument("-tb", "--train_batch", default=32, type=int, help="training batch size")
    parser.add_argument("-l",  "--layers", default=4, type=int, help="number of layers")

    # dataset
    parser.add_argument("-n_train", "--n_train", default=10000, type=int, help = "number of training examples")
    parser.add_argument("-n_test", "--n_test", default=500, type=int, help = "number of test examples")
    parser.add_argument("-v", "--vocab_size", default=20, type=int, help = "vocab size")
    parser.add_argument("-is", "--input_seq_len", default=100, type=int, help = "input sequence length")

    args = parser.parse_args()
    return args

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

if __name__ == "__main__":
    args = get_args()
    set_determinism(args)
    task = create_task_instance(args)
    task.load_model()

    if args.do_profile:
        task.run_profile()
    else:
        task.run()
