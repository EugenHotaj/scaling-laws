import torch
import torch.nn.functional as F
from torch.optim import AdamW

from data_loader import create_data_loader
from model import GPT,tiny_model


def train():
    n_steps = 10
    batch_size = 8
    seq_len = 64
    max_norm = 1.0

    model = GPT(tiny_model)
    optim = AdamW(model.parameters(), lr=1e-5)
    data_loader = create_data_loader(batch_size, seq_len)

    for i, (x, y) in enumerate(data_loader):
        if i >= n_steps:
            break

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, tiny_model.vocab_size), y.view(-1))
        loss.backward()
        unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)

        print(f"[{i}] loss={loss.item():.3}|norm={unclipped_norm:.3}")
    

if __name__ == "__main__":
    train()