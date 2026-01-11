import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class CompositionalMQARConfig(DataSegmentConfig):
    name: str = "compositional_mqar"
    power_a: float = 0.01
    num_kv_pairs: int = 4  # Must be a perfect square (4, 9, 16, ...)
    random_non_queries: bool = True
    include_slices: bool = True

    def build(self, seed: int) -> DataSegment:
        return compositional_mqar(**self.model_dump(), seed=seed)


def compositional_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 4,
    random_non_queries: bool = True,
    include_slices: bool = True,
    **kwargs
) -> DataSegment:
    """
    Generates synthetic data for the compositional multi-query associative recall task.
    
    This extends the standard MQAR task by using compound keys (K1, K2) instead of 
    single keys. The model must attend to BOTH key components to retrieve the correct 
    value, testing compositional reasoning and multi-token binding.
    
    Uses a grid structure to guarantee no shortcuts: with num_kv_pairs=4 (2x2 grid),
    we get pairs like (A,X)->v1, (A,Y)->v2, (B,X)->v3, (B,Y)->v4. Neither K1 nor K2
    alone determines the value.

    Example: 
        `compositional_mqar(vocab_size=150, num_kv_pairs=4, input_seq_len=30, random_non_queries=False)` 
        will generate input and label sequences of the form: 
                
                K1   K2  Val  K1   K2  Val  K1   K2  Val  K1   K2  Val        Query K1 K2      
        Inputs: 46   96  101  14   96  128  46   87  109  14   87  112  ...  14   96   ...
        Labels: -100 ...                                                          128  ...

        The -100 labels are ignored by the loss function and metrics.
        
    Key difference from standard MQAR:
        - Standard MQAR: "A" -> 5
        - Compositional MQAR: (14, 96) -> 128, (14, 87) -> 112, (46, 96) -> 101, (46, 87) -> 109
        K1=14 maps to different values depending on K2, forcing true compositional understanding.
    
    We include one important note on the power law distribution. In real language data, 
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again 
    drops the further away from the original mention we are. In our synthetic, we can 
    control this with the power law parameter `power_a`. Setting this to 1.0 will 
    result in a uniform distribution.

    Args:
        vocab_size (int): The size of the vocabulary. Split into thirds: K1s from
            [1, vocab_size//3), K2s from [vocab_size//3, 2*vocab_size//3), 
            Values from [2*vocab_size//3, vocab_size).
        num_examples (int): The number of examples to generate.
        input_seq_len (int): The length of the input sequence.
        seed (int): The seed for the random number generator.
        power_a (float, optional): The power for the power law distribution. 
            Defaults to 0.01.
        num_kv_pairs (int): The number of key-value pairs. Must be a perfect square
            (4, 9, 16, 25, ...) to form the grid structure.
        random_non_queries (bool, optional): If True, replace all the 0's with 
            random values in the input. Defaults to True.
        include_slices (bool, optional): If True, include metadata slices.

    Returns:
        DataSegment: A DataSegment object containing the generated inputs and labels.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    # 3 tokens for context (K1, K2, V) + 3 tokens for query region (K1, K2, gap) per pair
    assert num_kv_pairs * 6 <= input_seq_len, \
        f"Need at least 6 tokens per kv pair. Got {input_seq_len} for {num_kv_pairs} pairs."
    
    # num_kv_pairs must be a perfect square for grid structure
    grid_size = int(np.sqrt(num_kv_pairs))
    assert grid_size * grid_size == num_kv_pairs, \
        f"num_kv_pairs must be a perfect square (e.g., 4, 9, 16). Got {num_kv_pairs}."

    np.random.seed(seed)

    # Three tokens per "pair": K1, K2, Value
    context_size = num_kv_pairs * 3

    # Split vocabulary: K1s, K2s, and Values from separate pools
    third = vocab_size // 3
    k1_choices = np.arange(1, third)
    k2_choices = np.arange(third, 2 * third)
    value_choices = np.arange(2 * third, vocab_size)
    
    # For each example, pick grid_size K1s and grid_size K2s
    # Then form all grid_size^2 combinations
    k1s_unshuffled = np.tile(k1_choices, (num_examples, 1))
    k1_set = np.apply_along_axis(
        np.random.choice, 1, k1s_unshuffled, replace=False, size=grid_size
    )
    
    k2s_unshuffled = np.tile(k2_choices, (num_examples, 1))
    k2_set = np.apply_along_axis(
        np.random.choice, 1, k2s_unshuffled, replace=False, size=grid_size
    )
    
    # Create grid: all combinations of K1 x K2
    # k1s[i] = [A, A, B, B] if k1_set[i] = [A, B] and grid_size = 2
    # k2s[i] = [X, Y, X, Y] if k2_set[i] = [X, Y] and grid_size = 2
    k1s = np.repeat(k1_set, grid_size, axis=1)  # [A, A, B, B, ...]
    k2s = np.tile(k2_set, (1, grid_size))        # [X, Y, X, Y, ...]
    
    # Shuffle the order of pairs within each example (so position doesn't give it away)
    for i in range(num_examples):
        perm = np.random.permutation(num_kv_pairs)
        k1s[i] = k1s[i, perm]
        k2s[i] = k2s[i, perm]
    
    # Assign random unique values to each pair
    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
    )

    # Create context sequences with triplets: K1, K2, V, K1, K2, V, ...
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::3] = k1s
    kvs[:, 1::3] = k2s
    kvs[:, 2::3] = values

    # Compute power law distribution for query placement
    # Each query is 2 tokens (K1, K2), so we divide available space by 3 
    # (2 for query + 1 minimum gap)
    space = (input_seq_len - context_size) // 3
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs
    )

    # Build query region: place K1, K2 pairs at positions determined by gaps
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 3), values=k1s, axis=1)
    np.put_along_axis(queries, (gaps * 3) + 1, values=k2s, axis=1)

    examples = np.concatenate([kvs, queries], axis=1)

    # Labels: value should be predicted after seeing K2 in query
    # Position is (gaps * 3) + 2 relative to start of query region
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 3) + context_size + 2, values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # Replace all zeros with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return DataSegment(
        inputs,
        labels,
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len}
        if include_slices
        else {},
    )


if __name__ == "__main__":
    # Quick test to visualize the task
    result = compositional_mqar(
        vocab_size=150,
        num_examples=3,
        input_seq_len=30,
        seed=42,
        num_kv_pairs=4,  # 2x2 grid
        random_non_queries=False,
    )
    
    print("Compositional MQAR Task Visualization")
    print("=" * 60)
    print("Vocab split: K1s from [1-49], K2s from [50-99], Values from [100-149]")
    print("Grid structure ensures neither K1 nor K2 alone determines value")
    print()
    
    for i in range(min(3, len(result.inputs))):
        print(f"Example {i + 1}:")
        print(f"  Inputs: {result.inputs[i].tolist()}")
        print(f"  Labels: {result.labels[i].tolist()}")
        
        # Find the context triplets
        context = result.inputs[i][:12].reshape(4, 3)
        print(f"  Context mappings (2x2 grid):")
        
        # Group by K1 to show the grid structure
        from collections import defaultdict
        by_k1 = defaultdict(list)
        by_k2 = defaultdict(list)
        for j, (k1, k2, v) in enumerate(context):
            by_k1[k1.item()].append((k2.item(), v.item()))
            by_k2[k2.item()].append((k1.item(), v.item()))
        
        for k1, pairs in sorted(by_k1.items()):
            for k2, v in pairs:
                print(f"    ({k1}, {k2}) -> {v}")
        
        print(f"  Proof it's compositional:")
        for k1, pairs in sorted(by_k1.items()):
            vals = [v for _, v in pairs]
            print(f"    K1={k1} maps to values {vals} (depends on K2)")
        
        print()