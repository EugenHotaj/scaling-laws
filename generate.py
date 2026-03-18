from argparse import ArgumentParser

import torch
import tiktoken

from scaling_laws.model import GPT, gpt2_124m
from scaling_laws.utils import generate, get_device_and_dtype


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=25)
    args = parser.parse_args()

    device, dtype = get_device_and_dtype()
    model = GPT(gpt2_124m)
    state_dict = torch.load("models/124M/model.pt")
    model.load_state_dict(state_dict)
    model = model.to(device=device, dtype=dtype).eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "<|endoftext|>Marcus Aurelius said thus:"
    encoded = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    inputs = torch.tensor(encoded, device=device).view((1, -1))

    print(prompt, end="", flush=True)
    for token in generate(model, inputs, args.num_tokens):
        print(tokenizer.decode([token.item()]), end="", flush=True)
    print()
