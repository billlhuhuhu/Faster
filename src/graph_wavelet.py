import math

import numpy as np
from scipy import sparse


def parse_wavelet_scales(scales):
    if scales is None:
        return [1, 2, 4]
    if isinstance(scales, str):
        items = [item.strip() for item in scales.split(",") if item.strip()]
    else:
        items = list(scales)
    parsed = []
    for item in items:
        value = int(round(float(item)))
        if value < 1:
            raise ValueError(f"Wavelet scale must be >= 1, got {item}")
        parsed.append(value)
    if not parsed:
        raise ValueError("wavelet_scales must contain at least one positive scale")
    return sorted(set(parsed))


def row_normalize_sparse_graph(graph, eps=1e-12):
    graph = graph.tocsr().astype(np.float32)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    degree = np.maximum(degree, float(eps))
    inv_degree = sparse.diags(1.0 / degree.astype(np.float32))
    transition = inv_degree @ graph
    transition = transition.tocsr()
    transition.eliminate_zeros()
    return transition


def l2_normalize_rows(features, eps=1e-8):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    return (features / norms).astype(np.float32)


def build_multi_scale_wavelet_signatures(graph, base_embedding, scales, normalize=True, eps=1e-8):
    scales = parse_wavelet_scales(scales)
    base_embedding = np.asarray(base_embedding, dtype=np.float32)
    transition = row_normalize_sparse_graph(graph, eps=eps)
    max_power = max(scales) * 2
    powers = {0: base_embedding.astype(np.float32)}
    current = base_embedding.astype(np.float32)
    for power in range(1, max_power + 1):
        current = transition @ current
        powers[power] = np.asarray(current, dtype=np.float32)

    signatures = {}
    for scale in scales:
        coarse = powers[scale]
        finer = powers[2 * scale]
        signature = coarse - finer
        if normalize:
            signature = l2_normalize_rows(signature, eps=eps)
        signatures[int(scale)] = signature.astype(np.float32)
    return signatures


def resolve_active_scales(scales, step, total_steps, schedule="coarse_to_fine"):
    scales = parse_wavelet_scales(scales)
    coarse_first = sorted(scales, reverse=True)
    schedule = str(schedule or "coarse_to_fine")
    total_steps = max(int(total_steps), 1)
    step = int(step)
    if schedule == "all":
        active = coarse_first
    elif schedule == "coarse_to_fine":
        num_active = min(len(coarse_first), max(1, int(math.ceil(float(step + 1) * len(coarse_first) / float(total_steps)))))
        active = coarse_first[:num_active]
    else:
        raise ValueError(f"Unsupported wavelet schedule: {schedule}")
    weight = 1.0 / max(len(active), 1)
    weights = {int(scale): float(weight) for scale in active}
    return active, weights
