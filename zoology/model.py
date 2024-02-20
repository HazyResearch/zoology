import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from zoology.config import ModelConfig


class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        learnable: bool = True,
        device='cuda',
        dtype='torch.float32',
    ):
        """
        GPT-2 Learnable Token and Position Embeddings.
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
        if not learnable:
            self.word_embeddings.weight.requires_grad = False

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
            if "out_proj.weight" in name or "fc2.weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )
            # If using GLU activation for now, we scale the std by 2
            elif "output_linear.0.weight" in name:
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()

        self.sequence_mixer = config.sequence_mixer.instantiate(
            d_model=config.d_model,
            layer_idx=layer_idx,
        )
        self.state_mixer = config.state_mixer.instantiate(
            d_model=config.d_model,
            layer_idx=layer_idx,
        )
        self.dropout1 = nn.Dropout(config.embed_dropout if layer_idx == 0 else config.resid_dropout)
        self.drop_path1 = StochasticDepth(config.drop_path, mode="row")
        self.norm1 = nn.LayerNorm(config.d_model)
        self.dropout2 = nn.Dropout(config.resid_dropout)
        self.drop_path2 = StochasticDepth(config.drop_path, mode="row")
        self.norm2 = nn.LayerNorm(config.d_model)

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
    def __init__(self, config: ModelConfig):

        super().__init__()
        self.embeddings = TokenEmbeddings(
            config.d_model, 
            config.vocab_size, 
            config.max_position_embeddings,
            learnable=config.learnable_word_embeddings
        )
        if config.block_type == 'TransformerBlock':
            block_cls = TransformerBlock
        elif config.block_type == 'MambaBlock':
            from zoology.mixers.mamba import MambaBlock
            block_cls = MambaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(config=config, layer_idx=i)
                for i in range(config.n_layers)
            ]
        )
        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.apply(partial(_init_weights, n_layers=config.n_layers,))

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


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )

        self.backbone = LMBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights,n_layers=config.n_layers,))

        # tie weights
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
        self, input_ids, position_ids=None, state=None
    ): 
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        return self.lm_head(hidden_states)
        
    import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth

from zoology.config import ModelConfig


class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        learnable: bool = True,
        device='cuda',
        dtype='torch.float32',
    ):
        """
        GPT-2 Learnable Token and Position Embeddings.
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
        if not learnable:
            self.word_embeddings.weight.requires_grad = False

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
            if "out_proj.weight" in name or "fc2.weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )
            # If using GLU activation for now, we scale the std by 2
            elif "output_linear.0.weight" in name:
                nn.init.normal_(
                    p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layers)
                )


class TransformerBlock(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()

        self.sequence_mixer = config.sequence_mixer.instantiate(
            d_model=config.d_model,
            layer_idx=layer_idx,
        )
        self.state_mixer = config.state_mixer.instantiate(
            d_model=config.d_model,
            layer_idx=layer_idx,
        )
        self.dropout1 = nn.Dropout(config.embed_dropout if layer_idx == 0 else config.resid_dropout)
        self.drop_path1 = StochasticDepth(config.drop_path, mode="row")
        self.norm1 = nn.LayerNorm(config.d_model)
        self.dropout2 = nn.Dropout(config.resid_dropout)
        self.drop_path2 = StochasticDepth(config.drop_path, mode="row")
        self.norm2 = nn.LayerNorm(config.d_model)

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
    def __init__(self, config: ModelConfig):

        super().__init__()
        self.embeddings = TokenEmbeddings(
            config.d_model, 
            config.vocab_size, 
            config.max_position_embeddings,
            learnable=config.learnable_word_embeddings
        )
        if config.block_type == 'TransformerBlock':
            block_cls = TransformerBlock
        elif config.block_type == 'MambaBlock':
            from zoology.mixers.mamba import MambaBlock
            block_cls = MambaBlock
        self.layers = nn.ModuleList(
            [
                block_cls(config=config, layer_idx=i)
                for i in range(config.n_layers)
            ]
        )
        self.drop_f = nn.Dropout(config.resid_dropout)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.apply(partial(_init_weights, n_layers=config.n_layers,))

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


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )

        self.backbone = LMBackbone(config=config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.apply(partial(_init_weights,n_layers=config.n_layers,))

        # tie weights
        self.lm_head.weight = self.backbone.embeddings.word_embeddings.weight

    def forward(
        self, input_ids, position_ids=None, state=None
    ): 
        hidden_states = self.backbone(input_ids, position_ids=position_ids)
        return self.lm_head(hidden_states)
    
    def state_size(self, sequence_length: int):
        from zoology.mixers.mamba import MambaBlock

        state_size = 0
        for layer in self.backbone.layers:
            if isinstance(layer, MambaBlock):
                mixer = layer.mixer
            elif isinstance(layer, TransformerBlock):
                mixer = layer.sequence_mixer
            else: 
                return None
            if hasattr(mixer, "state_size"):
                state_size += mixer.state_size(sequence_length=sequence_length)
            else:
                print(f"Layer {type(mixer).__name__} does not have state size")
                return None
            
        return state_size * 4
