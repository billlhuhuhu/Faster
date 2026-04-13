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


def sparsify_sparse_matrix(matrix, topk=None, threshold=None, use_abs=True):
    matrix = matrix.tocsr().astype(np.float32)
    num_rows = int(matrix.shape[0])
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data
    new_indptr = np.zeros(num_rows + 1, dtype=np.int64)
    kept_indices = []
    kept_data = []
    threshold = None if threshold is None else float(threshold)

    for row_idx in range(num_rows):
        start = int(indptr[row_idx])
        end = int(indptr[row_idx + 1])
        row_indices = indices[start:end]
        row_data = data[start:end]
        if row_data.size == 0:
            new_indptr[row_idx + 1] = new_indptr[row_idx]
            continue

        if threshold is not None and threshold > 0:
            mask = np.abs(row_data) >= threshold if use_abs else row_data >= threshold
            row_indices = row_indices[mask]
            row_data = row_data[mask]

        if row_data.size == 0:
            new_indptr[row_idx + 1] = new_indptr[row_idx]
            continue

        if topk is not None and int(topk) > 0 and row_data.size > int(topk):
            metric = np.abs(row_data) if use_abs else row_data
            top_positions = np.argpartition(-metric, int(topk) - 1)[: int(topk)]
            top_positions = top_positions[np.argsort(-metric[top_positions], kind="stable")]
            row_indices = row_indices[top_positions]
            row_data = row_data[top_positions]

        kept_indices.append(np.asarray(row_indices, dtype=np.int32))
        kept_data.append(np.asarray(row_data, dtype=np.float32))
        new_indptr[row_idx + 1] = new_indptr[row_idx] + row_data.size

    if len(kept_indices) == 0:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)

    new_indices = np.concatenate(kept_indices, axis=0) if kept_indices else np.empty(0, dtype=np.int32)
    new_data = np.concatenate(kept_data, axis=0) if kept_data else np.empty(0, dtype=np.float32)
    pruned = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=matrix.shape, dtype=np.float32)
    pruned.eliminate_zeros()
    return pruned


def build_multi_scale_diffusion_difference_graphs(
    reference_graph,
    source_graph,
    scales,
    postprocess_topk=None,
    postprocess_threshold=None,
    eps=1e-8,
):
    scales = parse_wavelet_scales(scales)
    transition = row_normalize_sparse_graph(reference_graph, eps=eps)
    source_graph = source_graph.tocsr().astype(np.float32)
    max_power = max(scales) * 2
    powers = {0: source_graph}
    current = source_graph
    for power in range(1, max_power + 1):
        current = (transition @ current).tocsr()
        current = sparsify_sparse_matrix(
            current,
            topk=postprocess_topk,
            threshold=postprocess_threshold,
            use_abs=True,
        )
        powers[power] = current

    signatures = {}
    for scale in scales:
        signature = (powers[int(scale)] - powers[int(2 * scale)]).tocsr()
        signature.eliminate_zeros()
        signatures[int(scale)] = signature.astype(np.float32)
    return signatures


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
