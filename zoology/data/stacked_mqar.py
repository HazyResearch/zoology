import torch
from typing import Any
from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment


class ContinuousMQARConfig(DataSegmentConfig):
    name: str = "continuous_mqar"
    num_kv_pairs: int = 8
    num_passes: int = 1
    per_sequence_mapping: bool = True
    embeddings: Any = None  # Changed from torch.Tensor to Any
    
    model_config = {"arbitrary_types_allowed": True}  # ADD THIS
    
    def build(self, seed: int) -> DataSegment:
        return continuous_mqar(
            num_examples=self.num_examples,
            num_kv_pairs=self.num_kv_pairs,
            num_passes=self.num_passes,
            per_sequence_mapping=self.per_sequence_mapping,
            embeddings=self.embeddings,
            seed=seed,
        )


def continuous_mqar(
    num_examples: int,
    num_kv_pairs: int,
    num_passes: int = 1,
    per_sequence_mapping: bool = True,
    embeddings: torch.Tensor = None,
    seed: int = 42,
) -> DataSegment:
    """
    Generate a continuous MQAR dataset.
    Args:
        num_examples: Number of examples to generate.
        num_kv_pairs (int): The number of unique key-value pairs in the sequence. 
        num_passes (int): The number of passes through the key-value pairs in the sequence
            This follows the JRT approach (https://arxiv.org/abs/2407.05483).
        per_sequence_mapping: Whether to use a unique mapping for each sequence between 
            the key and value embeddings (stateless), or whether to persist them across 
            sequences (stateful).
        embeddings: Embeddings tensor.
        seed: Random seed.
    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test 
            inputs and labels.
    """

    torch.manual_seed(seed)
    
    vocab_size = embeddings.shape[0]
    embed_dim = embeddings.shape[1]
    num_keys = vocab_size // 2
    key_emb = embeddings[:num_keys]
    value_emb = embeddings[num_keys:]
    
    seq_len = (num_passes + 1) * num_kv_pairs
    inputs = torch.zeros(num_examples, seq_len, 2 * embed_dim)
    labels = torch.zeros(num_examples, num_kv_pairs, dtype=torch.long)
    
    for b in range(num_examples):
        # Sample which keys and values to use from the FULL set
        selected_keys = torch.randperm(num_keys)[:num_kv_pairs]    # global key indices
        selected_values = torch.randperm(num_keys)[:num_kv_pairs]  # global value indices
        
        if per_sequence_mapping or b == 0:
            # mapping[i] gives which selected_value to pair with selected_key[i]
            mapping = torch.randperm(num_kv_pairs)
        
        pos = 0
        for _ in range(num_passes):
            perm = torch.randperm(num_kv_pairs)
            for j in range(num_kv_pairs):
                local_idx = perm[j]
                key_idx = selected_keys[local_idx]           # global index into key_emb
                value_idx = selected_values[mapping[local_idx]]  # global index into value_emb
                inputs[b, pos, :embed_dim] = key_emb[key_idx]
                inputs[b, pos, embed_dim:] = value_emb[value_idx]
                pos += 1
        
        query_perm = torch.randperm(num_kv_pairs)
        for j in range(num_kv_pairs):
            local_idx = query_perm[j]
            key_idx = selected_keys[local_idx]
            value_idx = selected_values[mapping[local_idx]]
            inputs[b, pos, :embed_dim] = key_emb[key_idx]
            labels[b, j] = value_idx  # Global index for scoring against value_emb
            pos += 1
    
    return DataSegment(
        inputs=inputs,
        labels=labels,
        slices={"num_kv_pairs": num_kv_pairs, "num_passes": num_passes},
    )