# scaling-laws

### Setup

Tested on A100 [Runpod](https://www.runpod.io) GPUs with the official `autoresearch` template.

Setup the environment.
```bash
uv init --python 3.13
uv sync
source .venv/bin/activate
```

Download model weights.
```bash
python scripts/download_weights.py --model-size 124M
```

Verify that the model weights / implementation are correct.
```
python generate.py --num-tokens 20
```

Download data.
```bash
python scripts/download_data.py --num-files 1
```

Train.
```bash
python train.py
```