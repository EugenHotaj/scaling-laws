import math
import os
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Key, query, value projections for all heads, but in a batch.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Causal mask.
        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        bias = bias.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("softmax_bias", bias, persistent=False)

    def forward(self, x):
        # Batch size, sequence length, embedding dimensionality (n_embd).
        B, T, C = x.size()
        hs = C // self.n_head

        # Calculate query, key, values for all heads in batch and move head forward to
        # be the batch dim.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
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
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

    def forward(self, idx):
        _, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        # Forward the GPT model.
        tok_emb = self.model.wte(idx)
        pos_emb = self.model.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.model.h:
            x = block(x)
        x = self.model.ln_f(x)

        # GPT-2 models use tied wte and lm_head.
        return F.linear(x, self.model.wte.weight)
