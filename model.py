import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


tiny_model = GPTConfig(max_seq_len=64, n_layer=2, n_head=2, n_embed=2*16)

class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # Key, query, value projections for all heads, but in a batch.
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # Output projection.
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # Causal mask.
        bias = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        bias = bias.view(1, 1, config.max_seq_len, config.max_seq_len)
        self.register_buffer("softmax_bias", bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch size, sequence length, embedding dimensionality (n_embed).
        B, T, C = x.size()
        hs = C // self.n_head

        # Calculate query, key, values for all heads in batch and move head forward to
        # be the batch dim.
        q, k, v = self.c_attn(x).split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        # Manual implementation of attention.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.softmax_bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection.
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
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
                wte=nn.Embedding(config.vocab_size, config.n_embed),
                wpe=nn.Embedding(config.max_seq_len, config.n_embed),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

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
