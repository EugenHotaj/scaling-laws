# scaling-laws

### Setup

Tested on A100 [Runpod](https://www.runpod.io) GPUs with the official `autoresearch` template.

Setup the environment.
```bash
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

### Training

Before we can train, we need to download our pre-training dataset. 
We use Karpathy's [`climbmix-400b-shuffle`](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) which can be
downloaded locally with:
```bash
python scripts/download_data.py --num-files 250 
```

The script above will download ~12B training tokens (of which we will use ~10B) plus the ~50M token validation shard. 
Now we can kick off training:
```bash
python train.py
```

### Evaluation

Once we've trained a checkpoint, we can run evaluations with:
```
python eval.py
```

Here are the training run results

| Model | Validation NLL |
|----------|----------|
| GPT-2 (124M)   | ~3.20  | 
