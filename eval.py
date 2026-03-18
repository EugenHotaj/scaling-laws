from argparse import ArgumentParser

import torch

from scaling_laws.model import GPT, gpt2_124m
from scaling_laws.utils import valid_nll, get_device_and_dtype


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/124M/model.pt")
    args = parser.parse_args()

    device, dtype = get_device_and_dtype()

    state_dict = torch.load(args.checkpoint)
    model = GPT(gpt2_124m)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    nll = valid_nll(model=model, batch_size=256)
    print(f"Validation loss: {nll:.2f}")