from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layers: int = 12
    model_dim: int = 768
    head_dim: int = 64

# Model sizes pre-trained and released by OpenAI.
gpt2_117m = GPTConfig()
gpt2_335m = GPTConfig(n_layers=24, model_dim=1024)
gpt2_774m = GPTConfig(n_layers=36, model_dim=1280)
gpt2_1558m = GPTConfig(n_layers=48, model_dim=1600)

# Custom model sizes.
cpu_model = GPTConfig(max_seq_len=512, n_layers=2, head_dim=16, model_dim=4*16)
tiny_model = GPTConfig(n_layers=6, model_dim=384)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.model_dim, 4 * config.model_dim)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.model_dim, config.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.model_dim % config.head_dim == 0
        self.n_heads = config.model_dim // config.head_dim
        self.model_dim = config.model_dim
        self.head_dim = config.head_dim
        # Fused key, query, value projections.
        self.c_attn = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.c_proj = nn.Linear(config.model_dim, config.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute k, q, v, split out, and reshape each (B, T, C) -> (B, n_heads, T, head_dim).
        q, k, v = self.c_attn(x).split(self.model_dim, dim=2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape (B, n_heads, T, head_dims) -> (B, T, C) and compute output projection.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.model_dim)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.model_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.model_dim),
                wpe=nn.Embedding(config.max_seq_len, config.model_dim),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.model_dim),
            )
        )
        self._initialize()

    def _initialize(self):
        """Initializes the full model.
        
        Note: Some initializations are redundant with default PyTorch but we do them
        anyways here for completeness.
        """
        init_std = 0.002

        def init_layer_norm(layer_norm: nn.LayerNorm) -> None:
            nn.init.ones_(layer_norm.weight)
            nn.init.zeros_(layer_norm.bias)

        def init_linear(linear: nn.Linear) -> None:
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)
            nn.init.zeros_(linear.bias)

        nn.init.normal_(self.model.wte.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.model.wpe.weight, mean=0.0, std=init_std)
        for block in self.model.h:
            init_linear(block.attn.c_attn)
            init_linear(block.attn.c_proj)
            init_linear(block.mlp.c_fc)
            init_linear(block.mlp.c_proj)
            init_layer_norm(block.ln_1)
            init_layer_norm(block.ln_2)
        init_layer_norm(self.model.ln_f)

    @property
    def num_params(self) -> tuple[int, int, int]:
        n_total = sum(p.numel() for p in self.parameters()) 
        n_embed = (self.config.vocab_size + self.config.max_seq_len) * self.config.model_dim
        return n_total, n_total - n_embed, n_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)

        # Forward the GPT model.
        tok_emb = self.model.wte(x)
        pos_emb = self.model.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.model.h:
            x = block(x)
        x = self.model.ln_f(x)

        # GPT-2 models use tied wte and lm_head.
        return F.linear(x, self.model.wte.weight)
