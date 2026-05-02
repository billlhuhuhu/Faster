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

    def accuracy_score(y_true, y_pred, *args, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0:
            return 0.0
        return float(np.mean(y_true == y_pred))

    def precision_recall_fscore_support(y_true, y_pred, average=None, *args, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)]))
        precisions = []
        recalls = []
        f1s = []
        supports = []
        for label in labels:
            true_pos = float(np.sum((y_true == label) & (y_pred == label)))
            false_pos = float(np.sum((y_true != label) & (y_pred == label)))
            false_neg = float(np.sum((y_true == label) & (y_pred != label)))
            support = float(np.sum(y_true == label))
            precision = true_pos / max(true_pos + false_pos, 1.0)
            recall = true_pos / max(true_pos + false_neg, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

        precisions = np.asarray(precisions, dtype=np.float32)
        recalls = np.asarray(recalls, dtype=np.float32)
        f1s = np.asarray(f1s, dtype=np.float32)
        supports = np.asarray(supports, dtype=np.float32)
        if average in {"macro", "micro", "weighted", "binary"}:
            if average == "weighted" and supports.sum() > 0:
                weights = supports / supports.sum()
                return (
                    float(np.sum(precisions * weights)),
                    float(np.sum(recalls * weights)),
                    float(np.sum(f1s * weights)),
                    None,
                )
            return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s)), None
        return precisions, recalls, f1s, supports.astype(np.int64)

    def f1_score(y_true, y_pred, average="binary", *args, **kwargs):
        return precision_recall_fscore_support(y_true, y_pred, average=average, *args, **kwargs)[2]

    def matthews_corrcoef(y_true, y_pred, *args, **kwargs):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        if labels.size != 2:
            return 0.0
        pos = labels[-1]
        tp = float(np.sum((y_true == pos) & (y_pred == pos)))
        tn = float(np.sum((y_true != pos) & (y_pred != pos)))
        fp = float(np.sum((y_true != pos) & (y_pred == pos)))
        fn = float(np.sum((y_true == pos) & (y_pred != pos)))
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float(((tp * tn) - (fp * fn)) / max(denom, 1e-12))

    def mean_squared_error(y_true, y_pred, *args, **kwargs):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        return float(np.mean((y_true - y_pred) ** 2)) if y_true.size else 0.0

    def r2_score(y_true, y_pred, *args, **kwargs):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        if y_true.size == 0:
            return 0.0
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    def roc_auc_score(y_true, y_score, *args, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        order = np.argsort(fpr)
        return float(np.trapz(tpr[order], fpr[order]))

    def classification_report(*args, **kwargs):
        return ""

    def confusion_matrix(y_true, y_pred, labels=None, *args, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)]))
        labels = list(labels)
        index = {label: idx for idx, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for truth, pred in zip(y_true.reshape(-1), y_pred.reshape(-1)):
            if truth in index and pred in index:
                matrix[index[truth], index[pred]] += 1
        return matrix

    metrics_stub.roc_curve = roc_curve
    metrics_stub.roc_auc_score = roc_auc_score
    metrics_stub.accuracy_score = accuracy_score
    metrics_stub.f1_score = f1_score
    metrics_stub.precision_recall_fscore_support = precision_recall_fscore_support
    metrics_stub.matthews_corrcoef = matthews_corrcoef
    metrics_stub.mean_squared_error = mean_squared_error
    metrics_stub.r2_score = r2_score
    metrics_stub.classification_report = classification_report
    metrics_stub.confusion_matrix = confusion_matrix
    pairwise_stub.cosine_similarity = cosine_similarity
    metrics_stub.pairwise = pairwise_stub
    sklearn_stub.metrics = metrics_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = metrics_stub
    sys.modules["sklearn.metrics.pairwise"] = pairwise_stub
