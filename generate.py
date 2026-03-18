from argparse import ArgumentParser

import torch

from scaling_laws.model import GPT, gpt2_124m
from scaling_laws.utils import generate, get_device_and_dtype


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/124M/model.pt")
    parser.add_argument("--num-tokens", type=int, default=25)
    args = parser.parse_args()

    device, dtype = get_device_and_dtype()
    model = GPT(gpt2_124m)
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype).eval()

    prompt = "<|endoftext|>Marcus Aurelius said thus:"
    print(prompt, end="", flush=True)
    for token in generate(model, prompt, args.num_tokens):
        print(token, end="", flush=True)
    print()
