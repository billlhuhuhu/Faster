import importlib.machinery
import sys
import types

import numpy as np


def install_sklearn_metrics_stub_if_broken():
    """Install a tiny sklearn.metrics fallback for optional Transformers imports.

    Some environments can have a NumPy/scikit-learn ABI mismatch. Transformers
    imports sklearn.metrics.roc_curve through generation helpers even when this
    project only needs encoder-only BERT. If sklearn is broken, this fallback
    prevents the optional import from aborting unrelated training runs.
    """
    try:
        from sklearn.metrics import roc_curve  # noqa: F401
        return
    except Exception as exc:
        message = str(exc)
        if "numpy.dtype size changed" not in message and "sklearn" not in message:
            return

    for name in list(sys.modules):
        if name == "sklearn" or name.startswith("sklearn."):
            sys.modules.pop(name, None)

    sklearn_stub = types.ModuleType("sklearn")
    sklearn_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None, is_package=True)
    sklearn_stub.__path__ = []
    metrics_stub = types.ModuleType("sklearn.metrics")
    metrics_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None, is_package=True)
    metrics_stub.__path__ = []
    pairwise_stub = types.ModuleType("sklearn.metrics.pairwise")
    pairwise_stub.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics.pairwise", loader=None)

    def roc_curve(y_true, y_score, *args, **kwargs):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        thresholds = np.unique(y_score)[::-1]
        if thresholds.size == 0:
            thresholds = np.asarray([np.inf], dtype=np.float32)
        fps = np.zeros(thresholds.shape[0], dtype=np.float32)
        tps = np.zeros(thresholds.shape[0], dtype=np.float32)
        positives = max(float(np.sum(y_true == 1)), 1.0)
        negatives = max(float(np.sum(y_true != 1)), 1.0)
        for idx, threshold in enumerate(thresholds):
            pred = y_score >= threshold
            tps[idx] = float(np.sum(pred & (y_true == 1))) / positives
            fps[idx] = float(np.sum(pred & (y_true != 1))) / negatives
        return fps, tps, thresholds

    def cosine_similarity(X, Y=None, dense_output=True):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y, dtype=np.float32)
            if Y.ndim == 1:
                Y = Y.reshape(1, -1)

        x_norm = np.linalg.norm(X, axis=1, keepdims=True)
        y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
        X_normalized = X / np.clip(x_norm, 1e-12, None)
        Y_normalized = Y / np.clip(y_norm, 1e-12, None)
        return X_normalized @ Y_normalized.T

    metrics_stub.roc_curve = roc_curve
    pairwise_stub.cosine_similarity = cosine_similarity
    metrics_stub.pairwise = pairwise_stub
    sklearn_stub.metrics = metrics_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = metrics_stub
    sys.modules["sklearn.metrics.pairwise"] = pairwise_stub
