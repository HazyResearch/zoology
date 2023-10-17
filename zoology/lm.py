import math
from functools import partial
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
from einops import rearrange


class GPT2Embeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        device='cuda',
        dtype='torch.float32',
    ):
        """
        If max_position_embeddings <= 0, there's no position embeddings
        Wwe embed to word_embe_proj_dim dimension then project up to embed_dim
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                word_embed_proj_dim,
                padding_idx=padding_idx,
            )
            self.project_in = nn.Linear(
                word_embed_proj_dim, embed_dim, bias=False
            )
        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim
            )

    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=self.device
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


def _init_weights(
    module,
    n_layers,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )
            # If using GLU activation for now, we scale the std by 2
            elif name in ["output_linear.0.weight"]:
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )


class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
        """
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads=1,
        bias=True,
        dropout=0.0,
        layer_idx=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(
            embed_dim, 3 * embed_dim, bias=bias
        )
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, **kwargs):
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        context = self.inner_attn(qkv, **kwargs)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        return_residual=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class TransformerMixerBlock(nn.Module):
    _name_ = 'transformer_mixer'

    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        norm_cls=nn.LayerNorm,
        dropout_cls=nn.Dropout,
        resid_dropout1=0.1,
        resid_dropout2=0.0,
        drop_path1=0.0,
        drop_path2=0.0,
    ):
        super().__init__()
        self.sequence_mixer = MHA(
            d_model,
            num_heads=num_heads,
            dropout=0.1,
        )
        self.state_mixer = Mlp(
            d_model,
            hidden_features=d_model*4,
            out_features=d_model,
            activation=torch.tanh,
        )
        self.dropout1 = dropout_cls(resid_dropout1)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.norm1 = norm_cls(d_model)
        self.dropout2 = dropout_cls(resid_dropout2)
        self.drop_path2 = StochasticDepth(drop_path2, mode="row")
        self.norm2 = norm_cls(d_model)

    def forward(self, hidden_states, residual=None):
        dropped = self.drop_path1(self.dropout1(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
        hidden_states = self.sequence_mixer(hidden_states)
            
        dropped = self.drop_path2(self.dropout2(hidden_states))
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        hidden_states = self.state_mixer(hidden_states)
        return hidden_states, residual


class LMBackbone(nn.Module):
    def __init__(
        self,
        d_model=768,
        n_layers=12,
        vocab_size=50257,
        num_heads=12,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        **kwargs
    ) -> None:

        super().__init__()
        self.embeddings = GPT2Embeddings(
            d_model, vocab_size, max_position_embeddings
        )
        self.layers =  nn.ModuleList(
            [
                TransformerMixerBlock(
                    d_model,
                    num_heads=num_heads,
                    norm_cls=nn.LayerNorm,
                    dropout_cls=nn.Dropout,
                    resid_dropout1=embed_dropout if i == 0 else resid_dropout,
                    resid_dropout2=resid_dropout,
                )
                for i in range(n_layers)
            ]
        )
        self.drop_f = nn.Dropout(resid_dropout)
        self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.apply(partial(_init_weights,n_layers=n_layers,))

    def forward(self, input_ids, position_ids=None):
        hidden_states = self.embeddings(
            input_ids,
            position_ids=position_ids,
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        dropped = self.drop_f(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
        return hidden_states


class LMHeadModel(nn.Module):
    def __init__(
        self,
        d_model=768,
        n_layers=12,
        vocab_size=50257,
        num_heads=12,
        max_position_embeddings=0,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        pad_vocab_size_multiple: int = 1,
        block=None,
        **kwargs
    ) -> None:
        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                vocab_size % pad_vocab_size_multiple
            )

        self.backbone = LMBackbone(
            d_model=d_model,
            n_layers=n_layers,
            vocab_size=vocab_size,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            layer_norm_epsilon=layer_norm_epsilon,
            block=block,
            **kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights,n_layers=n_layers,))
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
        self, input_ids, position_ids=None, state=None
    ): 
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        lm_logits = self.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)
        
