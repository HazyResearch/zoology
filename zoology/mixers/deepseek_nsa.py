# Citation: https://github.com/lucidrains/native-sparse-attention-pytorch

from __future__ import annotations

from copy import deepcopy
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn, arange, stack, cat, tensor, Tensor
from torch.nn import Module, ModuleList

from zoology.mixers.deepseek.local_attention import LocalAttention

try:
    from rotary_embedding_torch import RotaryEmbedding
except ImportError:
    print(f"Please install rotary-embedding-torch: pip install rotary-embedding-torch")

# einstein notation
try:
    import einx
except:
    print(f"Please install einops: pip install einx")
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

def round_down_mult(n, mult):
    return n // mult * mult

def round_up_mult(n, mult):
    return ceil(n / mult) * mult

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

# attend function
def attend(
    q, k, v,
    mask = None,
    return_sim = False,
    scale = None
):
    scale = q.shape[-1] ** -0.5
    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads
    q = rearrange(q, 'b (h qh) ... -> b h qh ...', qh = num_grouped_queries)
    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale
    mask_value = max_neg_value(sim)

    if mask is not None:
        sim = sim.masked_fill(~mask, mask_value)

    attn = sim.softmax(dim = -1)
    attn_out = einsum(attn, v, 'b h qh i j, b h j d -> b h qh i d')
    attn_out = rearrange(attn_out, 'b h qh ... -> b (h qh) ...')

    if not return_sim:
        return attn_out

    sim = rearrange(sim, 'b h qh ... -> b (h qh) ...')
    return attn_out, sim


# classes
class SparseAttention(Module):
    def __init__(
        self,
        d_model,
        num_heads,
        sliding_window_size,
        compress_block_size,
        selection_block_size,
        num_selected_blocks,
        kv_heads = None,
        num_compressed_mem_kv = 1,
        causal = False,
        norm = True,
        use_triton_kernel = False,
        interpolated_importance_score = False,
        query_heads_share_selected_kv = True, # if set to True, importance score is averaged across query heads to select top-n buckets of kv per kv head - but can be set to False for each query head within a group to look at different sets of kv buckets. will be more memory and compute of course
        compress_mlp: Module | None = None,
        compress_mlp_expand_factor = 1.,
        layer_idx = -1,
    ):
        super().__init__()

        dim = d_model
        heads = num_heads
        dim_head = dim // heads
        self.dim = dim

        # attention heads
        # handling gqa if `kv_heads` is set
        kv_heads = kv_heads if kv_heads is not None else heads
        assert kv_heads <= heads and (heads % kv_heads) == 0 

        self.heads = heads
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads

        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # autoregressive or not - will extend this work for long context video / genomics use-cases
        self.causal = True # causal

        # rotary
        self.rotary_emb = RotaryEmbedding(dim_head)

        # qkv
        qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)
        self.to_qkv = nn.Linear(dim, sum(qkv_split), bias = False)
        self.qkv_split = qkv_split

        # sliding window strategy
        self.sliding_window = LocalAttention(
            dim = dim_head,
            window_size = sliding_window_size,
            causal = causal,
            exact_windowsize = True,
            autopad = True,
            use_rotary_pos_emb = False
        )
        self.sliding_window_size = sliding_window_size

        # compress strategy
        self.compress_block_size = compress_block_size
        assert num_compressed_mem_kv > 0
        self.split_compress_window = Rearrange('b h (w n) d -> b h w n d', n = compress_block_size)
        self.num_mem_compress_kv = num_compressed_mem_kv
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head))
        
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))

        compress_dim = compress_block_size * dim_head
        compress_mlp_dim_hidden = int(compress_mlp_expand_factor *compress_dim)
        compress_mlp = nn.Sequential(
            Rearrange('b h w n d -> b h w (n d)'),
            nn.Linear(compress_dim, compress_mlp_dim_hidden),
            nn.ReLU(),
            nn.Linear(compress_mlp_dim_hidden, dim_head),
        )
        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        # selection related
        self.interpolated_importance_score = interpolated_importance_score # in the case fine block size < compressed block size, will weigh space better when selecting

        self.query_heads_share_selected_kv = query_heads_share_selected_kv
        self.selection_block_size = selection_block_size

        assert num_selected_blocks >= 0
        if num_selected_blocks == 0:
            print(f'`num_selected_blocks` should be set greater than 0, unless if you are ablating it for experimental purposes')

        self.num_selected_blocks = num_selected_blocks
        self.use_triton_kernel = use_triton_kernel

        # they combine the three sparse branches through a learned combine with sigmoid activation
        strategy_combine_mlp = nn.Linear(dim, 3 * heads)
        # init to sliding windows first, as network tends to pick up on local patterns first before distant ones
        nn.init.zeros_(strategy_combine_mlp.weight)
        strategy_combine_mlp.bias.data.copy_(tensor([-2., -2., 2.] * heads))
        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )

        # split and merging heads
        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # combining heads
        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)


    def forward(
        self,
        inp,
        cache = None,
        disable_triton_kernel = False,
    ):
        batch, seq_len, scale, heads, device = *inp.shape[:2], self.scale, self.heads, inp.device

        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # maybe prenorm
        inp = self.norm(inp)

        # queries, keys, values
        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        # compressed key / values - variables prepended with `c` stands for compressed
        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)
        k_compress_input = self.split_compress_window(k[..., :compress_divisible_seq_len, :] + k_pos)
        v_compress_input = self.split_compress_window(v[..., :compress_divisible_seq_len, :] + v_pos)
        assert seq_len % compress_divisible_seq_len == 0, 'sequence length must be divisible by compress block size'

        cq = q
        ck = self.k_compress(k_compress_input) 
        cv = self.v_compress(v_compress_input)

        # 1. coarse attention over compressed
        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = batch)
        num_mem_compress_kv = mem_ck.shape[-2]
        ck = cat((mem_ck, ck), dim = -2)
        cv = cat((mem_cv, cv), dim = -2)

        # compressed masking
        cmask = None
        if self.causal:
            cq_seq = arange(seq_len, device = device)
            ck_seq = ((arange(num_compress_blocks, device = device) + 1) * self.compress_block_size) - 1
            ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value = -1)
            cmask = einx.less('j, i -> i j', ck_seq, cq_seq)

        compressed_attn_out, csim = attend(cq, ck, cv, mask = cmask, return_sim = True)

        # for 2. and 3., will give them relative positions with rotary - compressed needs to be handled separately (even if they already have intra block absolute positions)
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # 2. fine attention over selected based on compressed attention logits - variables prepended with `f` stands for the fine attention pathway
        importance_scores = csim[..., num_mem_compress_kv:]
        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0

        # maybe average the compressed attention across each grouped queries (per key / values)
        if self.query_heads_share_selected_kv:
            importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries = self.num_grouped_queries)
            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1

        if has_selected_kv_for_fine_attn:
            assert self.compress_block_size == self.selection_block_size
            importance_scores = F.pad(importance_scores, (1, 0), value = -1e3)
            importance_scores = importance_scores.softmax(dim = -1)
            importance_scores = importance_scores[..., 1:]

        # handle if number of total blocks is less than number to select for fine attention
        fq, fk, fv = q, k, v
        if has_selected_kv_for_fine_attn:
            # get the top-n kv segments for fine attention
            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim = -1)
            gates = None

            if self.use_triton_kernel and not disable_triton_kernel:
                from native_sparse_attention_pytorch.triton_native_sparse_attention import native_sparse_attend
                fmask = selected_importance_values > 1e-10
                fine_attn_out = native_sparse_attend(
                    fq, fk, fv,
                    self.selection_block_size,
                    selected_block_indices,
                    fmask,
                    sel_scale = gates,
                    include_block_causal = self.causal
                )

            else:
                fmask = selected_importance_values > 1e-10
                assert seq_len == fine_divisible_seq_len, 'sequence length must be divisible by selection block size'
                
                if self.causal:
                    # handle block causal diagonal in the diagram, but run experiments without to see
                    fine_window_seq = arange(fine_divisible_seq_len, device = device) // self.selection_block_size
                    fine_window_seq = repeat(fine_window_seq, 'n -> b h n 1', b = batch, h = selected_block_indices.shape[1])
                    selected_block_indices = cat((selected_block_indices, fine_window_seq), dim = -1) # for the block causal diagonal in fig2

                    fmask = repeat(fmask, 'b h i w -> b h i w j', j = self.selection_block_size)

                    causal_mask = torch.ones((self.selection_block_size,) * 2, device = device, dtype = torch.bool).tril()
                    causal_mask = repeat(causal_mask, 'i j -> b h (w i) 1 j', w = num_fine_blocks, b = batch, h = fmask.shape[1])

                    fmask = cat((fmask, causal_mask), dim = -2)
                    fmask = rearrange(fmask, 'b h i w j -> b h 1 i (w j)')

                else:
                    fmask = repeat(fmask, 'b h i w -> b h 1 i (w j)', j = self.selection_block_size)

                # select out the spatial crops of keys / values for fine attention
                fk = rearrange(fk, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
                fv = rearrange(fv, 'b h (w n) d -> b h w n d', w = num_fine_blocks)

                if self.query_heads_share_selected_kv:
                    fk = repeat(fk, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                    fv = repeat(fv, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                else:
                    fk = repeat(fk, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)
                    fv = repeat(fv, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)

                selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fk.shape[-2], d = fk.shape[-1])
                fk = fk.gather(3, selected_block_indices)
                fv = fv.gather(3, selected_block_indices)

                # merge selected key values
                fk, fv = tuple(rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fk, fv))

                # fine attention
                fq = rearrange(fq, 'b (h qh) ... -> b h qh ...', qh = fine_num_grouped_queries)
                fsim = einsum(fq, fk, 'b h qh i d, b h i j d -> b h qh i j') * self.scale
                mask_value = max_neg_value(fsim)
                fsim = fsim.masked_fill(~fmask, mask_value)
                fattn = fsim.softmax(dim = -1)
                fine_attn_out = einsum(fattn, fv,'b h qh i j, b h i j d -> b h qh i d')
                fine_attn_out = rearrange(fine_attn_out,'b h qh ... -> b (h qh) ...')
                fine_attn_out = fine_attn_out[..., :seq_len, :]

        else:
            # if only first block, just do a simple block causal
            seq_len = fk.shape[-2]
            fmask = None
            if self.causal:
                fmask = causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).tril()
            fine_attn_out = attend(fq, fk, fv, mask = fmask)

        # 3. overlapping SWA, this is unsurprising and expected - `s` for sliding
        sq, sk, sv = q, k, v
        sk, sv = tuple(repeat(t, 'b h ... -> b (h num_grouped_queries) ...', num_grouped_queries = self.num_grouped_queries) for t in (sk, sv))
        sliding_window_attn_out = self.sliding_window(sq, sk, sv)

        # combine strategies
        strategy_weighted_combine = self.to_strategy_combine(inp)
        out = einsum(strategy_weighted_combine, stack([compressed_attn_out, fine_attn_out, sliding_window_attn_out]), 'b h n s, s b h n d -> b h n d')

        # merge heads and combine them
        out = self.merge_heads(out)
        out = self.combine_heads(out)
        return out #, (cache_kv, cache_compressed_kv)

    def state_size(self, sequence_length: int=2048):
        # 1. compressed kv cache (grows with sequence length)
        num_compressed_tokens = sequence_length // self.compress_block_size
        state_size = 2 * num_compressed_tokens * self.dim 
        # 2. sliding window kv cache (fixed state size)
        state_size += 2 * self.sliding_window_size * self.dim
        # 3. fine kv cache (fixed state size)
        state_size += 2 * self.selection_block_size * self.dim * self.num_selected_blocks
        return state_size 
