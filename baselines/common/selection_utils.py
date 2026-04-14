from typing import Iterable, List, Tuple

import numpy as np


def resolve_subset_size(n: int, ratio: float) -> int:
    k = int(round(float(ratio) * float(n)))
    return max(1, min(int(n), int(k)))


def resolve_subset_size_from_budget(n: int, budget: int) -> int:
    k = int(budget)
    return max(1, min(int(n), int(k)))


def resolve_budget_and_ratio(n: int, ratio: float = None, budget: int = None) -> Tuple[int, float]:
    if budget is None and ratio is None:
        raise ValueError("Either budget or ratio must be provided.")
    if budget is not None:
        k = resolve_subset_size_from_budget(n, budget)
        resolved_ratio = float(k) / max(float(n), 1.0)
        return k, resolved_ratio
    k = resolve_subset_size(n, ratio)
    resolved_ratio = float(k) / max(float(n), 1.0)
    return k, resolved_ratio


def topk_indices(scores: np.ndarray, k: int, largest: bool = True) -> List[int]:
    scores = np.asarray(scores)
    if largest:
        order = np.argsort(-scores, kind="stable")
    else:
        order = np.argsort(scores, kind="stable")
    return [int(x) for x in order[: int(k)]]


def random_select_indices(n: int, k: int, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(k), replace=False)
    return [int(x) for x in np.sort(idx)]


def herding_select(features: np.ndarray, k: int) -> List[int]:
    x = np.asarray(features, dtype=np.float32)
    mu = np.mean(x, axis=0, keepdims=True)
    selected: List[int] = []
    running = np.zeros((1, x.shape[1]), dtype=np.float32)
    used = np.zeros(x.shape[0], dtype=bool)

    for _ in range(int(k)):
        target = (len(selected) + 1) * mu - running
        dist = np.linalg.norm(x - target, axis=1)
        dist[used] = np.inf
        idx = int(np.argmin(dist))
        selected.append(idx)
        used[idx] = True
        running += x[idx : idx + 1]
    return selected


def kcenter_greedy_select(features: np.ndarray, k: int, seed: int = 0) -> List[int]:
    x = np.asarray(features, dtype=np.float32)
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, x.shape[0]))
    selected = [first]
    min_dist = np.linalg.norm(x - x[first : first + 1], axis=1)
    for _ in range(1, int(k)):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        d = np.linalg.norm(x - x[idx : idx + 1], axis=1)
        min_dist = np.minimum(min_dist, d)
    return selected


def gradmatch_greedy(sample_grads: np.ndarray, target_grad: np.ndarray, k: int) -> Tuple[List[int], np.ndarray]:
    grads = np.asarray(sample_grads, dtype=np.float32)
    residual = np.asarray(target_grad, dtype=np.float32).copy()
    selected: List[int] = []
    used = np.zeros(grads.shape[0], dtype=bool)

    for _ in range(int(k)):
        scores = grads @ residual
        scores[used] = -np.inf
        idx = int(np.argmax(scores))
        selected.append(idx)
        used[idx] = True
        residual = residual - grads[idx]
    return selected, residual


def glister_greedy(sample_grads: np.ndarray, val_grad: np.ndarray, k: int) -> Tuple[List[int], np.ndarray]:
    grads = np.asarray(sample_grads, dtype=np.float32)
    residual = np.asarray(val_grad, dtype=np.float32).copy()
    selected: List[int] = []
    used = np.zeros(grads.shape[0], dtype=bool)

    for _ in range(int(k)):
        gains = grads @ residual
        gains[used] = -np.inf
        idx = int(np.argmax(gains))
        selected.append(idx)
        used[idx] = True
        residual = residual - grads[idx]
    return selected, residual


def nms_select(scores: np.ndarray, features: np.ndarray, k: int, suppress_radius: float = 0.2) -> List[int]:
    scores = np.asarray(scores, dtype=np.float32)
    x = np.asarray(features, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    order = np.argsort(-scores)
    selected: List[int] = []
    suppressed = np.zeros(x.shape[0], dtype=bool)
    for idx in order:
        idx = int(idx)
        if suppressed[idx]:
            continue
        selected.append(idx)
        if len(selected) >= int(k):
            break
        sim = x @ x[idx]
        suppressed = np.logical_or(suppressed, sim >= (1.0 - float(suppress_radius)))
        suppressed[idx] = True
    if len(selected) < int(k):
        for idx in order:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= int(k):
                break
    return selected
