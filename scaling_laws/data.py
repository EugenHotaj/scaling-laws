from collections import deque
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
import torch
from torch.utils.data import IterableDataset, DataLoader


def _iterate_parquet(split: str) -> Iterator[list[str]]:
    data_dir = Path("data")
    parquet_files = sorted(data_dir.glob("*.parquet"))
    parquet_files = parquet_files[:-1] if split == "train" else parquet_files[-1:]

    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            yield rg.column("text").to_pylist()


class _Dataset(IterableDataset):
    def __init__(self, batch_size: int, seq_len: int, split: str) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self._max_tokens = self.batch_size * self.seq_len + 1  # Use +1 because inputs/targets are shifted.
        self._tokenizer = tiktoken.get_encoding("gpt2")
        self._bos_token = self._tokenizer.eot_token

    def __iter__(self) -> Iterator[torch.Tensor]:
        token_queue = deque()
        for rows in _iterate_parquet(self.split):
            # Yield batches of tokens when we have enough capacity.
            while len(token_queue) >= self._max_tokens:
                tokens = torch.tensor([token_queue.popleft() for _ in range(self._max_tokens)])
                x = tokens[:-1].view(self.batch_size, self.seq_len)
                y = tokens[1:].view(self.batch_size, self.seq_len)
                yield x, y

            # Tokenize current rows and add them to the queue. For each row we first
            # add the <BOS> token, then we add the tokenized row.
            tokenized_rows = self._tokenizer.encode_batch(rows, disallowed_special=())
            for tokens in tokenized_rows:
                token_queue.append(self._bos_token)  
                token_queue.extend(tokens)


def create_data_loader(batch_size: int, seq_len: int, split: str = "train") -> DataLoader:
    assert batch_size > 0 and seq_len > 0
    assert split in ("train", "valid")
    ds = _Dataset(batch_size, seq_len, split)
    return DataLoader(ds, batch_size=None, num_workers=1, pin_memory=True, prefetch_factor=4)
