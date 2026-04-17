from types import SimpleNamespace

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
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
    tfidf_ngram_max=2,
    tfidf_stop_words="english",
    tfidf_max_features=20000,
    tfidf_min_df=1,
    tfidf_svd_dim=256,
    tfidf_random_state=0,
):
    if text_repr_method == "tfidf":
        stop_words = None if str(tfidf_stop_words).lower() in {"", "none", "null"} else tfidf_stop_words
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            ngram_range=(1, int(max(1, tfidf_ngram_max))),
            max_features=None if tfidf_max_features is None or int(tfidf_max_features) <= 0 else int(tfidf_max_features),
            min_df=max(1, int(tfidf_min_df)),
            norm="l2",
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        if tfidf_matrix.shape[1] <= 1:
            dense = tfidf_matrix.toarray().astype(np.float32, copy=False)
            return dense

        target_dim = min(
            int(tfidf_svd_dim),
            int(tfidf_matrix.shape[0]) - 1,
            int(tfidf_matrix.shape[1]) - 1,
        )
        if target_dim <= 0:
            dense = tfidf_matrix.toarray().astype(np.float32, copy=False)
            return dense

        svd = TruncatedSVD(n_components=target_dim, random_state=int(tfidf_random_state))
        reduced = svd.fit_transform(tfidf_matrix).astype(np.float32, copy=False)
        reduced = normalize(reduced, norm="l2").astype(np.float32, copy=False)
        return reduced

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
