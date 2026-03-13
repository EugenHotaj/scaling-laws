"""Downloads dataset.

Modified from https://github.com/karpathy/nanochat/blob/master/nanochat/dataset.py.
"""

import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import requests
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # Last shard is shard_06542.parquet.
DATA_DIR = "data"


def download_single_file(index: int) -> bool:
    """Downloads a single file."""

    # Construct the local filepath for this file and skip if it already exists.
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Download the file.
    url = f"{BASE_URL}/{filename}"
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
    


if __name__ == "__main__":
    parser = ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("--num-files", type=int, default=MAX_SHARD)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Prepare the output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)

    # Download the shards
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(download_single_file, idx) for idx in ids_to_download]
        for future in tqdm(futures):
            future.result()
