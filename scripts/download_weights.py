"""Downloads GPT-2 weights and converts them into PyTorch-loadable format.

Modified from https://github.com/openai/gpt-2/blob/master/download_model.py.
"""

import os
import re
from argparse import ArgumentParser

import requests
import tensorflow as tf
import torch
from tqdm import tqdm


def download_model(model_size: str) -> None:
    model = f"models/{model_size}"

    # Download the model weights from OpenAI if they don't already exist.
    if not os.path.exists(model):
        os.makedirs(model)
        filenames = [
            "checkpoint", "model.ckpt.data-00000-of-00001", "model.ckpt.index", "model.ckpt.meta"
        ]
        for filename in tqdm(filenames, desc="Donwloading TensorFlow model"):
            resp = requests.get(
                f"https://openaipublic.blob.core.windows.net/gpt-2/{model}/{filename}",
                stream=True,
            )
            with open(f"{model}/{filename}", "wb") as file_:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes.
                for chunk in resp.iter_content(chunk_size=1000):
                    file_.write(chunk)

    # Create PyTorch-loadable state dict if it does not already exist.
    pt_model_path = f"{model}/model.pt"
    if not os.path.exists(pt_model_path):
        checkpoint = tf.train.load_checkpoint(model)
        variables = sorted(list(checkpoint.get_variable_to_shape_map().keys()))

        state_dict = {}
        for name in tqdm(variables, desc="Converting to PyTorch"):
            tensor = checkpoint.get_tensor(name).squeeze()
            if name.endswith("wpe") or name.endswith("wte"):
                name = name + "/weight"
            elif name.endswith("/w") or name.endswith("/g"):
                name = name[:-2] + "/weight"
                # PyTorch transposes tensors compared to TensorFlow.
                tensor = tensor.T
            elif name.endswith("/b"):
                name = name[:-2] + "/bias"
            name = name.replace("/", ".")
            name = re.sub(r"\.h(\d+)\.", r".h.\1.", name)
            state_dict[name] = torch.tensor(tensor)
        torch.save(state_dict, pt_model_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-size", default="124M", choices=("124M", "355M", "774M", "1558M"))
    args = parser.parse_args()
    download_model(args.model_size)
