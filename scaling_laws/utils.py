import json
from collections.abc import Iterator

import tiktoken
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


def append_to_jsonl(rows: list[dict[str, int | float]], path: str) -> None:
    with open(path, "a") as file_:
        for row in rows:
            file_.write(json.dumps(row) + "\n")


@torch.no_grad()
def valid_nll(model: GPT, batch_size: int, max_iter=50) -> torch.Tensor:
    vocab_size = model.config.vocab_size
    seq_len = model.config.max_seq_len
    device = next(model.parameters()).device

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


@torch.no_grad()
def generate(model: GPT, prompt: str, new_tokens: int, temp=1.0) -> Iterator[str]:
    tokenizer = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device

    tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    tokens = torch.tensor(tokens, device=device).view((1, -1))
    assert len(tokens) + new_tokens <= model.config.max_seq_len

    for _ in range(new_tokens):
        logits = model(tokens)[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        yield tokenizer.decode([next_token.item()])
        tokens = torch.cat((tokens, next_token), dim=1)
