import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parameter import Parameter

from data import create_data_loader
from model import GPT, GPTConfig, tiny_model


M = 1_000_000
GiB = 2 ** 30


@torch.compile()
def _compiled_fwdbwd(
    model: GPT, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    vocab_size: int, 
    gradient_accumulation_steps: int,
) -> torch.Tensor:
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss


# TODO(eugen): May not be helpful split out from the fwd/bwd like this, need to investigate.
@torch.compile()
def _compiled_optim_step(optim: AdamW, params: list[Parameter], max_norm: float) -> float:
    unclipped_norm = None
    unclipped_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    optim.step()
    optim.zero_grad(set_to_none=True)
    return unclipped_norm


def train(
    gpt_config: GPTConfig,
    clip_grad_norm: float = 1.0,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    n_steps: int = 100,
) -> None:
    seq_len = gpt_config.max_seq_len
    vocab_size = gpt_config.vocab_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Create data loader, model, optim.
    data_loader = create_data_loader(batch_size, seq_len)
    model = GPT(gpt_config).to(device=device, dtype=dtype)
    optim = AdamW(model.parameters(), lr=3e-4)

    # Print out training config.
    _, n_body, n_emb = model.num_params
    global_batch_size = batch_size * gradient_accumulation_steps
    toks_per_batch = global_batch_size * seq_len
    print( 
        "Starting training:\n"
        f"  Model     : {n_body / M:.2f}M (+{n_emb/ M:.2f}M emb) params\n"
        f"  Batch size: {global_batch_size} ({toks_per_batch / M:.2f}M tokens)"
    )

    # Train.
    start_ts = time.monotonic()
    for step, (x, y) in enumerate(data_loader):
        if step >= n_steps:
            break

        loss, norm = 0.0, 0.0
        for i in range(gradient_accumulation_steps):
            x, y = x.to(device=device), y.to(device=device)
            step_loss = _compiled_fwdbwd(model, x, y, vocab_size, gradient_accumulation_steps)
            loss += step_loss.item()
            if i == gradient_accumulation_steps - 1:
                norm = _compiled_optim_step(optim, model.parameters(), clip_grad_norm)

        step_time = time.monotonic() - start_ts
        tps = toks_per_batch / step_time / M
        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / GiB
            torch.cuda.reset_peak_memory_stats()
        print(f"[{step}] loss={loss:.2f}|norm={norm:.2f}|tps={tps:.2f}M|pm={peak_mem:.2f}G")
        start_ts = time.monotonic()
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bs", type=int, default=8, help="Batch size (defaults to 8).")
    parser.add_argument("--gas", type=int, default=1, help="Gradient accumulation steps (defaults to 1).")
    args = parser.parse_args()

    train(tiny_model, batch_size=args.bs, gradient_accumulation_steps=args.gas)