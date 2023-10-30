import numpy as np
import torch
from collections import defaultdict

from .utils import SyntheticData

def lego(
    vocab_size: int,
    num_train_examples: int,
    num_test_examples: int,
    input_seq_len: int,
    seed: int,
    n_var: int,
    train_proportions,
    test_proportions,
):
    lego_dataset = LEGODataset(n_var, vocab_size)
    train_inputs, train_labels = lego_dataset._generate(
        num_examples=num_train_examples,
        seed=seed,
        proportions=train_proportions
    )
    test_inputs, test_labels = lego_dataset._generate(
        num_examples=num_test_examples,
        seed=seed + 10,  # different seed for test set
        proportions=test_proportions
    )
    test_labels[:, :-1] = -100 # mask out labels except for last token

    data = SyntheticData(
        train_inputs=train_inputs,
        train_labels=train_labels,
        test_inputs=test_inputs,
        test_labels=test_labels,
    )

    # check for data leakage:
    train_set = set([" ".join(map(str, x)) for x in data.train_inputs.tolist()])
    test_set = set([" ".join(map(str, x)) for x in data.test_inputs.tolist()])
    frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
    if frac_test_in_train > 0.001:
        print(
            "WARNING: Potential data leakage detected. " 
            f"{frac_test_in_train: 0.2f} of test examples are in the train set."
        )
    return data


class LEGODataset():
    def __init__(self, n_var, vocab_size):
        self.n_var = n_var 
        self.all_samples = defaultdict(set)
        self.non_special_vocab = [chr(ord('`')+number+1) for number in np.arange(vocab_size)]
        self.special_vocab = {
            "=>" : "=>",
            "val" : "val",
            "not" : "not",
            "0" : "0",
            "1" : "1"
        }
        
        #print(f"Vocab size excluding special vocab: {vocab_size}")
        #print(f"Special vocabs size: {len(special_vocabs)}")
        # vocab = [str(v) for v in list(range(vocab_size))]
        self.vocab = sorted(list(set(self.non_special_vocab + list(self.special_vocab.values()))))
        self.vocab.append('-100')
        print(f"Vocabulary is: {self.vocab}")
        self.v2id = {v:i for i,v in enumerate(self.vocab)}
        self.v2id['-100'] = -100

        
    def _generate(self, num_examples, seed, proportions):
        np.random.seed(seed)
        assert len(proportions) == self.n_var 
        proportions = np.array(proportions).astype(float)
        proportions /= proportions.sum()
        n_per_skill = (proportions * num_examples).astype(int) 
        n_per_skill[-1] = num_examples - n_per_skill[:-1].sum() 
        
        all_examples = [] 
        for i, (n_data, skill_idx) in enumerate(zip(n_per_skill, np.arange(self.n_var))):
            var_examples =self._generate_var(n_data, skill_idx)
            all_examples.extend(var_examples)
            
        np.random.shuffle(all_examples)
        all_examples = torch.LongTensor(all_examples)
        inputs, labels = torch.tensor(all_examples[:, :-1]), torch.tensor(all_examples[:, 1:])
    
        return inputs, labels
    
    def _generate_var(self, n_data, skill_idx):
        count = 0
        tokenized_data = []
        while count < n_data:
            values = np.random.randint(0, 2, (self.n_var, ))
            var_idx = tuple(np.random.permutation(len(self.non_special_vocab)))
            vars = [self.non_special_vocab[i] for i in var_idx]
            clauses = []
            clauses.append("%s val %d " % (vars[0], values[0]))
            
            for i in range(1, self.n_var):
                modifier = "val" if values[i] == values[i-1] else "not"
                clauses.append("%s %s %s " % (vars[i], modifier, vars[i-1]))

            clause_idxs = tuple(np.random.permutation(self.n_var))
            sent = "".join([clauses[idx] for idx in clause_idxs])
            sent += "=>"
            sent += " %s val %d" % (vars[skill_idx], values[skill_idx])
            if sent in self.all_samples[skill_idx]:
                continue 
            else:
                self.all_samples[skill_idx].add(sent)
                if count == 0:
                    print(sent)
                tokenized_input = self.tokenize(sent, return_tensor=True)['input_ids'].tolist()
                tokenized_data.append(tokenized_input)
                count += 1
                
        return tokenized_data



    def tokenize(self, text, return_tensor=False):
        input_ids = [self.v2id[t] for t in text.split()]
        if return_tensor:
            input_ids = torch.LongTensor(input_ids)
        return {
            "input_ids": input_ids,
        }