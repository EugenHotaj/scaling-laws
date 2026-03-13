import torch
import tiktoken
import torch.nn.functional as F

from model import GPT, GPTConfig

@torch.no_grad()
def generate(model: GPT, tokens: torch.Tensor, new_tokens: int, temp=0.6) -> torch.Tensor:
    assert len(tokens) + new_tokens <= model.config.block_size

    for _ in range(new_tokens):
        logits = model(tokens)[:, -1, :] / temp
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens


if __name__ == '__main__':
    config = GPTConfig()
    model = GPT(config).eval()
    state_dict = torch.load("models/124M/model.pt")
    model.load_state_dict(state_dict)

    encoder = tiktoken.get_encoding("gpt2")
    encoded = encoder.encode(
        "<|endoftext|>Marcus Aurelius said thus:", allowed_special={"<|endoftext|>"}
    )
    inputs = torch.tensor(encoded).view((1, -1))
    generated = generate(model, inputs, 10).tolist()[0]

    print(encoder.decode(generated))
