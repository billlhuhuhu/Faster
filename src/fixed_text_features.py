from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

from src.networks import TextEncoder


def _make_text_model_args(text_encoder="bert"):
    return SimpleNamespace(
        text_encoder=text_encoder,
        text_pretrained=True,
        text_trainable=False,
    )


@torch.no_grad()
def extract_fixed_text_features(
    texts,
    text_repr_method="bert",
    batch_size=256,
    device="cpu",
    bert_model_path=None,
):
    if text_repr_method != "bert":
        raise ValueError(f"Unsupported fixed text representation method: {text_repr_method}")

    model_args = _make_text_model_args(text_encoder=text_repr_method)
    text_encoder = TextEncoder(model_args).to(device)
    text_encoder.eval()

    outputs = []
    for start in tqdm(range(0, len(texts), int(batch_size)), desc="Extracting fixed text features"):
        batch_texts = texts[start : start + int(batch_size)]
        batch_features = text_encoder(batch_texts, device=device).detach().cpu().float().numpy()
        outputs.append(batch_features.astype(np.float32))

    if not outputs:
        raise RuntimeError("No text features were extracted.")
    return np.concatenate(outputs, axis=0).astype(np.float32)
