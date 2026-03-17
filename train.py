import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from data import create_data_loader
from model import GPT, GPTConfig, tiny_model


M = 1_000_000


def train(
    gpt_config: GPTConfig,
    clip_grad_norm: float = 1.0,
    batch_size: int = 8,
    n_steps: int = 100,
) -> None:
    seq_len = gpt_config.max_seq_len
    vocab_size = gpt_config.vocab_size
    
    model = GPT(gpt_config)
    optim = AdamW(model.parameters(), lr=3e-4)
    data_loader = create_data_loader(batch_size, seq_len)

    _, n_body, n_emb = model.num_params
    print(f"Training {n_body / M:.2f}M (+{n_emb/ M:.2f}M emb) paramter model.")
    start_ts = time.monotonic()
    for i, (x, y) in enumerate(data_loader):
        if i >= n_steps:
            break

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)

        step_time = time.monotonic() - start_ts
        tps = (batch_size * seq_len) / step_time
        print(f"[{i}] loss={loss.item():.2f}|norm={unclipped_norm:.2f}|tps={tps:.2f}|sec={step_time:.2f}")
        start_ts = time.monotonic()
    

if __name__ == "__main__":
    train(tiny_model)