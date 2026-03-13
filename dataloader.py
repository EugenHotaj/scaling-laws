from collections import deque
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
import torch


def _iterate_parquet() -> Iterator[list[str]]:
    data_dir = Path("data")
    parquet_files = sorted(data_dir.glob("*.parquet"))

    for parquet_file in parquet_files[:-1]:
        pf = pq.ParquetFile(parquet_file)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            yield rg.column("text").to_pylist()


def create_data_loader(batch_size: int, seq_len: int) -> Iterator[torch.Tensor]:
    assert batch_size > 0 and seq_len > 0

    # TODO(eugen): Use (seq_len + 1) because inputs / targets will be shifted by 1. However,
    # this means that we never use the last token as input, does this matter? Probably not?
    max_tokens = batch_size * (seq_len + 1)
    tokenizer = tiktoken.get_encoding("gpt2")
    bos_token = tokenizer.eot_token

    token_queue = deque()
    for rows in _iterate_parquet():
        # Yield batches of tokens when we have enough capacity.
        while len(token_queue) >= max_tokens:
            tokens = torch.tensor([token_queue.popleft() for _ in range(max_tokens)])
            tokens = tokens.view(batch_size, seq_len + 1)
            yield tokens[:, :-1], tokens[:, 1:]

        # Tokenize current rows and add them to the queue. For each row we first
        # add the <BOS> token, then we add the tokenized row.
        tokenized_rows = tokenizer.encode_batch(rows)
        for tokens in tokenized_rows:
            token_queue.append(bos_token)  
            token_queue.extend(tokens)
