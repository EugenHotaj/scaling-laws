from collections.abc import Iterator

import torch
import torch.nn.functional as F
from tqdm import tqdm

from scaling_laws.data import create_data_loader
from scaling_laws.model import GPT


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    else:
        return "cpu", torch.float32


@torch.no_grad()
def generate(model: GPT, prompt: torch.Tensor, new_tokens: int, temp=0.6) -> Iterator[torch.Tensor]:
    assert len(prompt) + new_tokens <= model.config.max_seq_len
    for _ in range(new_tokens):
        logits = model(prompt)[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        yield next_token
        prompt = torch.cat((prompt, next_token), dim=1)


@torch.no_grad()
def valid_nll(
    model: GPT, vocab_size: int, batch_size: int, seq_len: int, device: str, max_iter=50
) -> torch.Tensor:
    data_loader = create_data_loader(batch_size=batch_size, seq_len=seq_len, split="valid") 
    total_loss, total_tokens = 0.0, 0
    for step, (x, y) in tqdm(enumerate(data_loader), desc="Running eval", total=max_iter):
        if step >= max_iter:
            break
        x, y = x.to(device=device), y.to(device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / total_tokens
