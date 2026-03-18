from argparse import ArgumentParser

import torch
import tiktoken
import torch.nn.functional as F

from model import GPT, gpt2_124m

@torch.no_grad()
def generate(model: GPT, tokens: torch.Tensor, new_tokens: int, temp=0.6) -> torch.Tensor:
    assert len(tokens) + new_tokens <= model.config.max_seq_len

    for _ in range(new_tokens):
        logits = model(tokens)[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        yield next_token
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=25)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT(gpt2_124m).to(device=device).eval()
    state_dict = torch.load("models/124M/model.pt")
    model.load_state_dict(state_dict)

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "<|endoftext|>Marcus Aurelius said thus:"
    encoded = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    inputs = torch.tensor(encoded, device=device).view((1, -1))

    print(prompt, end="", flush=True)
    for token in generate(model, inputs, args.num_tokens):
        print(tokenizer.decode([token.item()]), end="", flush=True)
    print()
