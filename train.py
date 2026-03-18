import math
import os
import time
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parameter import Parameter
from tqdm import tqdm

from scaling_laws.data import create_data_loader
from scaling_laws.model import GPT, GPTConfig, gpt2_124m
from scaling_laws.utils import (
    append_to_jsonl, generate, get_device_and_dtype, valid_nll 
)


M = 10 ** 6
GiB = 2 ** 30


def create_lr_scheduler(
    optim: AdamW, warmup_steps: int, total_steps: int, min_lr_factor: float
) -> LambdaLR:
    decay_steps = total_steps - warmup_steps

    def cosine_decay_with_warmup(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps 
        else:
            progress = float(step - warmup_steps) / decay_steps
            adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
            adjustment = min_lr_factor + (1 - min_lr_factor) * adjustment 
            return adjustment

    return LambdaLR(optim, cosine_decay_with_warmup)


@torch.compile(dynamic=False, options={"shape_padding": True})
def _compiled_fwdbwd(
    model: GPT, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    vocab_size: int, 
    gradient_accumulation_steps: int,
) -> torch.Tensor:
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size).float(), y.view(-1))
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss


def train(
    name: str,
    gpt_config: GPTConfig,
    batch_size: int,
    gradient_accumulation_steps: int,
    n_warmup_steps: int,
    n_steps: int,
    clip_grad_norm: float = 1.0,
    save_every_n_steps: int = 1000,
) -> None:
    torch.manual_seed(42)
    seq_len = gpt_config.max_seq_len
    vocab_size = gpt_config.vocab_size
    device, dtype = get_device_and_dtype()
    checkpoint_dir = f"models/{name}"
    
    # Create data loader, model, optim, scheduler.
    data_loader = create_data_loader(batch_size, seq_len)
    model = GPT(gpt_config).to(device=device, dtype=dtype)
    model = torch.compile(
        model, dynamic=False, fullgraph=True, options={"triton.cudagraphs": True, "shape_padding": True}
    )
    # Only apply weight decay to 2d weight tensors only.
    optim_groups = [
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": 0.1},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    optim = AdamW(optim_groups, lr=6e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    lr_scheduler = create_lr_scheduler(optim, n_warmup_steps, n_steps, min_lr_factor=0.1)

    # Print out training config.
    _, n_body, n_emb = model.num_params
    global_batch_size = batch_size * gradient_accumulation_steps
    toks_per_batch = global_batch_size * seq_len
    print( 
        "Starting training:\n"
        f"  Model     : {n_body / M:.2f}M (+{n_emb/ M:.2f}M emb) params\n"
        f"  Batch size: {global_batch_size} ({toks_per_batch / M:.2f}M tokens)"
    )

    metrics = []
    start_ts = time.perf_counter()
    for step, (x, y) in enumerate(data_loader):
        if step >= n_steps:
            break

        # Train single step (with gradient accumulation).
        loss = 0.0
        for i in range(gradient_accumulation_steps):
            x, y = x.to(device=device), y.to(device=device)
            step_loss = _compiled_fwdbwd(model, x, y, vocab_size, gradient_accumulation_steps)
            loss += step_loss.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm).item()
        optim.step()
        optim.zero_grad(set_to_none=True)
        lr_scheduler.step()
        torch.cuda.synchronize()
        step_time = time.perf_counter() - start_ts

        # Log metrics.
        lr = lr_scheduler.get_last_lr()[0]
        tps = toks_per_batch / step_time / M
        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / GiB
            torch.cuda.reset_peak_memory_stats()
        metrics.append({"step": step, "loss": loss, "norm": norm, "lr": lr, "tps": tps, "peak_mem": peak_mem})
        print(f"[{step:5}] loss={loss:.2f}|norm={norm:.3f}|lr={lr:.3e}|tps={tps:.3f}M|pm={peak_mem:.2f}G")

        if (step + 1) % save_every_n_steps == 0 or (step + 1) == n_steps:
            # Flush training metrics.
            print(f"Evaluating and saving checkpoint at step {step}.")
            os.makedirs(checkpoint_dir, exist_ok=True)
            append_to_jsonl(metrics, f"{checkpoint_dir}/metrics_train.jsonl")
            metrics = []

            # Estimate NLL on the validation split.
            valid_loss = valid_nll(model, batch_size=128)
            print(f"valid_nll@{step}: {valid_loss}")

            # Sample a few tokens from the current checkpoint.
            prompt = "<|endoftext|>Marcus Aurelius said thus:"
            # NOTE: Use the original, uncompiled model for generations since the sequence length changes.
            generation = list(tqdm(generate(model._orig_mod, prompt, new_tokens=128), desc="Sampling", total=128))
            generation = ''.join(generation)
            print(f"sample@{step}: {prompt}{generation}")

            # Flush validation metrics.
            valid_metrics = {"step": step, "loss": valid_loss, "prompt": prompt, "generation": generation}
            append_to_jsonl([valid_metrics], f"{checkpoint_dir}/metrics_valid.jsonl")

            # Dump checkpoint.
            checkpoint_path = f"{checkpoint_dir}/step_{step}.pt"
            print(f"Saving checkpoint to {checkpoint_path}.")
            torch.save(model._orig_mod.state_dict(), checkpoint_path)

        start_ts = time.perf_counter()


if __name__ == "__main__":
    # Default arguments are set to reproduce gpt124M by training for ~10B tokens.
    parser = ArgumentParser()
    parser.add_argument(
        "--name", type=str, help="Run name. Checkpoints and metrics will be saved to models/<run-name>/."
    )
    parser.add_argument("--bs", type=int, default=128, help="Batch size (defaults to 128).")
    parser.add_argument("--gas", type=int, default=4, help="Gradient accumulation steps (defaults to 4).")
    parser.add_argument("--warmup-steps", type=int, default=250, help="Number of LR warmup steps.")
    parser.add_argument("--steps", type=int, default=19074, help="Number of training steps.")
    args = parser.parse_args()

    train(
        name=args.name,
        gpt_config=gpt2_124m, 
        batch_size=args.bs, 
        gradient_accumulation_steps=args.gas,
        n_warmup_steps=args.warmup_steps,
        n_steps=args.steps,
    )