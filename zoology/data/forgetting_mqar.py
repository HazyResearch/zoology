import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class ForgettingMQARConfig(DataSegmentConfig):
    name: str = "forgetting_mqar"
    power_a: float = 0.01
    num_kv_pairs: int = 8
    num_updates: int = 4  # How many keys get reassigned
    random_non_queries: bool = True
    include_slices: bool = True

    def build(self, seed: int) -> DataSegment:
        return forgetting_mqar(**self.model_dump(), seed=seed)


def forgetting_mqar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float = 0.01,
    num_kv_pairs: int = 8,
    num_updates: int = 4,
    random_non_queries: bool = True,
    include_slices: bool = True,
    **kwargs
) -> DataSegment:
    """
    Generates synthetic data for the forgetting multi-query associative recall task.
    
    This extends the standard MQAR task by having some keys appear twice with 
    different values. The model must remember the LAST (most recent) value for 
    each key, testing the ability to update/overwrite associations.

    Example: 
        `forgetting_mqar(vocab_size=100, num_kv_pairs=4, num_updates=2, input_seq_len=32, random_non_queries=False)` 
        will generate input and label sequences of the form: 
                
                K   V   K   V   K   V   K   V   K   V   K   V        Queries
        Inputs: A   5   B   3   C   7   D   2   A   9   C   1   ...  A  ...  B  ...
        Labels: -100 ...                                             9       3
        
        Keys A and C were updated, so queries return their NEW values (9, 1).
        Keys B and D were not updated, so queries return original values (3, 2).

    Args:
        vocab_size (int): The size of the vocabulary. First half for keys, second 
            half for values.
        num_examples (int): The number of examples to generate.
        input_seq_len (int): The length of the input sequence.
        seed (int): The seed for the random number generator.
        power_a (float, optional): The power for the power law distribution. 
            Defaults to 0.01.
        num_kv_pairs (int): The number of unique keys.
        num_updates (int): The number of keys that get reassigned a new value.
            Must be <= num_kv_pairs.
        random_non_queries (bool, optional): If True, replace all the 0's with 
            random values in the input. Defaults to True.
        include_slices (bool, optional): If True, include metadata slices.

    Returns:
        DataSegment: A DataSegment object containing the generated inputs and labels.
    """
    assert input_seq_len % 2 == 0, "input_seq_len must be even"
    assert vocab_size > input_seq_len
    assert num_updates <= num_kv_pairs, "num_updates must be <= num_kv_pairs"
    
    # Total KV slots: original pairs + updates
    total_kv_slots = num_kv_pairs + num_updates
    # 2 tokens per KV slot in context, 2 tokens per query (key + gap)
    assert total_kv_slots * 2 + num_kv_pairs * 2 <= input_seq_len, \
        f"Need at least {total_kv_slots * 2 + num_kv_pairs * 2} tokens. Got {input_seq_len}."

    np.random.seed(seed)

    # Context size: original KVs + updated KVs
    context_size = total_kv_slots * 2

    # Split vocabulary
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    # Generate unique keys for each example
    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs
    )

    # Generate original values (one per key)
    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    original_values = np.apply_along_axis(
        np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs
    )

    # Select which keys get updated (first num_updates keys, then we'll shuffle)
    # Generate new values for updated keys (must be different from original)
    updated_values = np.zeros((num_examples, num_updates), dtype=np.int64)
    for i in range(num_examples):
        for j in range(num_updates):
            # Pick a value different from the original
            available = value_choices[value_choices != original_values[i, j]]
            updated_values[i, j] = np.random.choice(available)

    # Build context: [K0 V0 K1 V1 ... Kn Vn | K0 V0' K1 V1' ... (updates)]
    # First, place all original KV pairs
    # Then, place updates for selected keys
    
    # For each example, randomly select which keys get updated
    update_indices = np.stack([
        np.random.choice(num_kv_pairs, size=num_updates, replace=False)
        for _ in range(num_examples)
    ])
    
    # Get the keys and their new values for updates
    update_keys = np.take_along_axis(keys, update_indices, axis=1)
    
    # Final values: start with original, then overwrite updated ones
    final_values = original_values.copy()
    for i in range(num_examples):
        for j, idx in enumerate(update_indices[i]):
            final_values[i, idx] = updated_values[i, j]

    # Build context sequence
    # Structure: [original KVs] [update KVs]
    original_context = np.zeros((num_examples, num_kv_pairs * 2), dtype=np.int64)
    original_context[:, 0::2] = keys
    original_context[:, 1::2] = original_values

    update_context = np.zeros((num_examples, num_updates * 2), dtype=np.int64)
    update_context[:, 0::2] = update_keys
    update_context[:, 1::2] = updated_values

    kvs = np.concatenate([original_context, update_context], axis=1)

    # Compute power law distribution for query placement
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(
        np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs
    )

    # Build query region
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)

    examples = np.concatenate([kvs, queries], axis=1)

    # Labels: should be the FINAL value (after updates)
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=final_values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])

    # Replace zeros with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]

    return DataSegment(
        inputs,
        labels,
        slices={
            "num_kv_pairs": num_kv_pairs,
            "num_updates": num_updates,
            "input_seq_len": input_seq_len,
        }
        if include_slices
        else {},
    )


if __name__ == "__main__":
    # Quick test to visualize the task
    result = forgetting_mqar(
        vocab_size=100,
        num_examples=3,
        input_seq_len=32,
        seed=42,
        num_kv_pairs=4,
        num_updates=2,
        random_non_queries=False,
    )

    print("Forgetting MQAR Task Visualization")
    print("=" * 60)
    print("Vocab: 1-49 are keys, 50-99 are values")
    print("num_kv_pairs=4, num_updates=2 (half get reassigned)")
    print()

    for i in range(min(3, len(result.inputs))):
        print(f"Example {i + 1}:")
        print(f"  Inputs: {result.inputs[i].tolist()}")
        print(f"  Labels: {result.labels[i].tolist()}")

        # Parse original KVs (first 8 tokens)
        original = result.inputs[i][:8].reshape(4, 2)
        # Parse updates (next 4 tokens)
        updates = result.inputs[i][8:12].reshape(2, 2)

        print(f"  Original mappings:")
        for k, v in original:
            print(f"    {k.item()} -> {v.item()}")

        print(f"  Updates (overwrite):")
        for k, v in updates:
            print(f"    {k.item()} -> {v.item()} (NEW)")

        # Find query positions
        query_positions = (result.labels[i] != -100).nonzero(as_tuple=True)[0]
        print(f"  Queries (should return FINAL value):")
        for pos in query_positions:
            key = result.inputs[i][pos].item()
            expected = result.labels[i][pos].item()
            print(f"    Key {key} -> {expected}")
        print()
        