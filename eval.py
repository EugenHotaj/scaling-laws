import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GPT, gpt2_124m
from data import create_data_loader


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


if __name__ == '__main__':
    vocab_size = gpt2_24m.vocab_size
    seq_len = gpt2_124m.max_seq_len
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    state_dict = torch.load("models/124M/model.pt")
    model = GPT(gpt2_124m)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    nll = valid_nll(
        model=model, vocab_size=vocab_size, batch_size=256, seq_len=, device=device
    )
    print(f"Validation loss: {nll:.2f}")