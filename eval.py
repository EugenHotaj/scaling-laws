import torch
import torch.nn.functional as F

from model import GPT, gpt2_117m
from data import create_data_loader

@torch.no_grad()
def valid_nll(model: GPT, vocab_size: int, batch_size: int, seq_len: int) -> torch.Tensor:
    data_loader = create_data_loader(batch_size=batch_size, seq_len=seq_len, split="valid") 
    total_loss, total_tokens = 0.0
    for x, y in data_loader:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
    return total_loss / total_tokens


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(gpt2_117m).to(device=device).eval()
    state_dict = torch.load("models/124M/model.pt")
    model.load_state_dict(state_dict)
    valid_nll(model, vocab_size=gpt2_117m.vocab_size, batch_size=512, seq_len=gpt2_117m.max_seq_len)