import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


def sanitize_name(name):
    return name.replace("\\", "-").replace("/", "-").replace(" ", "_")


def build_feature_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    return Path(args.feature_cache_root) / args.dataset / args.split / model_tag


def build_output_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    graph_tag = f"k{args.k}_{sanitize_name(args.metric)}"
    return Path(args.output_root) / args.dataset / args.split / model_tag / args.modality / graph_tag


def load_feature_tensor(feature_dir, modality):
    filename = {
        "image": "img_features.pt",
        "text": "txt_features.pt",
    }[modality]
    path = Path(feature_dir) / filename
    features = torch.load(path, map_location="cpu")
    if torch.is_tensor(features):
        features = features.float().cpu().numpy()
    else:
        features = np.asarray(features, dtype=np.float32)
    return features


def load_sample_meta(feature_dir):
    meta_path = Path(feature_dir) / "sample_meta.json"
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_truncate(features, sample_meta, max_samples=None):
    if max_samples is None:
        return features, sample_meta
    max_samples = min(int(max_samples), features.shape[0])
    return features[:max_samples], sample_meta[:max_samples]


def compute_knn(features, k, metric, n_jobs=None):
    num_nodes = features.shape[0]
    if num_nodes < 2:
        raise ValueError("At least two feature vectors are required to build a topology graph.")

    n_neighbors = min(int(k) + 1, num_nodes)
    nn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm="auto",
        n_jobs=n_jobs,
    )
    nn_model.fit(features)
    distances, indices = nn_model.kneighbors(features, return_distance=True)

    # Drop self-neighbor if present.
    if np.all(indices[:, 0] == np.arange(num_nodes)):
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        distances = distances[:, : min(k, indices.shape[1])]
        indices = indices[:, : min(k, indices.shape[1])]

    return indices.astype(np.int64), distances.astype(np.float32)


def compute_rho(distances, local_connectivity=1.0):
    if distances.shape[1] == 0:
        return np.zeros(distances.shape[0], dtype=np.float32)

    local_connectivity = max(float(local_connectivity), 0.0)
    integer_part = int(math.floor(local_connectivity))
    interpolation = local_connectivity - integer_part
    max_col = distances.shape[1] - 1

    rho = np.zeros(distances.shape[0], dtype=np.float32)
    for row_idx, row in enumerate(distances):
        if integer_part <= 0:
            base = row[0]
            next_value = row[min(1, max_col)]
            rho[row_idx] = base + interpolation * (next_value - base)
        else:
            base_idx = min(integer_part - 1, max_col)
            base = row[base_idx]
            if interpolation > 0 and base_idx + 1 <= max_col:
                rho[row_idx] = base + interpolation * (row[base_idx + 1] - base)
            else:
                rho[row_idx] = base
    return rho


def _sigma_objective(distances_row, rho_value, sigma_value):
    adjusted = np.maximum(0.0, distances_row - rho_value)
    return float(np.sum(np.exp(-adjusted / sigma_value)))


def compute_sigmas(distances, rho, bandwidth=None, n_iter=64, target=None):
    num_nodes, k = distances.shape
    if k == 0:
        return np.ones(num_nodes, dtype=np.float32)

    if bandwidth is None:
        bandwidth = math.log2(k + 1)
    if target is None:
        target = bandwidth

    sigmas = np.zeros(num_nodes, dtype=np.float32)
    for row_idx in range(num_nodes):
        row = distances[row_idx]
        rho_value = rho[row_idx]

        lo = 1e-6
        hi = 1.0
        while _sigma_objective(row, rho_value, hi) < target:
            hi *= 2.0
            if hi > 1e6:
                break

        for _ in range(int(n_iter)):
            mid = (lo + hi) * 0.5
            value = _sigma_objective(row, rho_value, mid)
            if value > target:
                hi = mid
            else:
                lo = mid

        sigmas[row_idx] = max((lo + hi) * 0.5, 1e-6)
    return sigmas


def build_directed_graph(knn_indices, knn_distances, rho, sigma, num_nodes):
    row_ids = np.repeat(np.arange(num_nodes), knn_indices.shape[1])
    col_ids = knn_indices.reshape(-1)
    adjusted = np.maximum(0.0, knn_distances - rho[:, None])
    weights = np.exp(-adjusted / sigma[:, None]).reshape(-1)
    weights = np.clip(weights, 0.0, 1.0).astype(np.float32)

    graph = sparse.csr_matrix((weights, (row_ids, col_ids)), shape=(num_nodes, num_nodes), dtype=np.float32)
    graph.eliminate_zeros()
    return graph


def symmetrize_graph(directed_graph):
    transpose = directed_graph.transpose().tocsr()
    sym = directed_graph + transpose - directed_graph.multiply(transpose)
    sym = sym.tocsr()
    sym.eliminate_zeros()
    return sym


def row_normalize_graph(graph):
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    degree = np.maximum(degree, 1e-12)
    inv_degree = sparse.diags(1.0 / degree.astype(np.float32))
    transition = inv_degree @ graph
    transition = transition.tocsr()
    transition.eliminate_zeros()
    return transition


def build_laplacian(graph, normalized=True):
    laplacian = csgraph.laplacian(graph, normed=normalized)
    laplacian = laplacian.tocsr()
    laplacian.eliminate_zeros()
    return laplacian


def compute_spectrum(laplacian, num_eigs, return_eigenvectors=False):
    num_nodes = laplacian.shape[0]
    if num_nodes <= 1:
        eigenvalues = np.zeros(1, dtype=np.float32)
        eigenvectors = np.ones((1, 1), dtype=np.float32) if return_eigenvectors else None
        return eigenvalues, eigenvectors

    num_eigs = max(2, min(int(num_eigs), num_nodes - 1))
    if num_nodes <= 4096:
        dense_laplacian = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense_laplacian)
        eigenvalues = eigenvalues[:num_eigs].astype(np.float32)
        eigenvectors = eigenvectors[:, :num_eigs].astype(np.float32) if return_eigenvectors else None
        return eigenvalues, eigenvectors

    eigenvalues, eigenvectors = eigsh(laplacian, k=num_eigs, which="SM", return_eigenvectors=True)
    order = np.argsort(eigenvalues)
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float32)
    if return_eigenvectors:
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float32)
    else:
        eigenvectors = None
    return eigenvalues, eigenvectors


def compute_collapse_metrics(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.shape[0] <= 1:
        return {
            "spectral_entropy": 0.0,
            "collapse_score": 1.0,
            "num_eigs_used": int(eigenvalues.shape[0]),
        }

    nontrivial = np.maximum(eigenvalues[1:], 1e-12)
    weights = nontrivial / nontrivial.sum()
    entropy = float(-(weights * np.log(weights)).sum() / np.log(len(weights)))
    collapse_score = float(1.0 - entropy)
    return {
        "spectral_entropy": entropy,
        "collapse_score": collapse_score,
        "num_eigs_used": int(len(eigenvalues)),
    }


def summarize_results(args, features, knn_indices, sym_graph, eigenvalues, collapse_metrics):
    num_nodes = int(features.shape[0])
    num_edges = int(sym_graph.nnz)
    avg_degree = float(num_edges / max(num_nodes, 1))
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "modality": args.modality,
        "image_encoder": args.image_encoder,
        "text_encoder": args.text_encoder,
        "metric": args.metric,
        "k": int(args.k),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "knn_width": int(knn_indices.shape[1]),
        "feature_dim": int(features.shape[1]),
        "first_eigenvalues": [float(x) for x in eigenvalues[: min(10, len(eigenvalues))]],
        "spectral_entropy": collapse_metrics["spectral_entropy"],
        "collapse_score": collapse_metrics["collapse_score"],
        "num_eigs_used": collapse_metrics["num_eigs_used"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary


def save_outputs(output_dir, knn_indices, knn_distances, rho, sigma, directed_graph, sym_graph, transition_graph, laplacian, eigenvalues, eigenvectors, sample_meta, summary):
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(torch.from_numpy(knn_indices), output_dir / "knn_indices.pt")
    torch.save(torch.from_numpy(knn_distances), output_dir / "knn_distances.pt")
    torch.save(
        {
            "rho": torch.from_numpy(rho),
            "sigma": torch.from_numpy(sigma),
        },
        output_dir / "local_scale.pt",
    )

    sparse.save_npz(output_dir / "directed_graph.npz", directed_graph)
    sparse.save_npz(output_dir / "symmetric_graph.npz", sym_graph)
    sparse.save_npz(output_dir / "transition_graph.npz", transition_graph)
    sparse.save_npz(output_dir / "laplacian_normalized.npz", laplacian)

    torch.save(torch.from_numpy(eigenvalues), output_dir / "eigenvalues.pt")
    if eigenvectors is not None:
        torch.save(torch.from_numpy(eigenvectors), output_dir / "eigenvectors.pt")

    with open(output_dir / "sample_meta.json", "w", encoding="utf-8") as handle:
        json.dump(sample_meta, handle, ensure_ascii=False, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def run_topology_graph(args):
    feature_dir = build_feature_dir(args)
    output_dir = build_output_dir(args)

    features = load_feature_tensor(feature_dir, args.modality)
    sample_meta = load_sample_meta(feature_dir)
    features, sample_meta = maybe_truncate(features, sample_meta, args.max_samples)

    knn_indices, knn_distances = compute_knn(features, args.k, args.metric, args.n_jobs)
    rho = compute_rho(knn_distances, local_connectivity=args.local_connectivity)
    sigma = compute_sigmas(
        knn_distances,
        rho,
        bandwidth=args.bandwidth,
        n_iter=args.sigma_search_steps,
        target=args.bandwidth if args.bandwidth is not None else None,
    )

    directed_graph = build_directed_graph(knn_indices, knn_distances, rho, sigma, features.shape[0])
    sym_graph = symmetrize_graph(directed_graph)
    transition_graph = row_normalize_graph(sym_graph)
    laplacian = build_laplacian(sym_graph, normalized=True)
    eigenvalues, eigenvectors = compute_spectrum(laplacian, args.num_eigs, return_eigenvectors=args.save_eigenvectors)
    collapse_metrics = compute_collapse_metrics(eigenvalues)
    summary = summarize_results(args, features, knn_indices, sym_graph, eigenvalues, collapse_metrics)

    save_outputs(
        output_dir,
        knn_indices,
        knn_distances,
        rho,
        sigma,
        directed_graph,
        sym_graph,
        transition_graph,
        laplacian,
        eigenvalues,
        eigenvectors,
        sample_meta,
        summary,
    )
    return {
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "summary": summary,
    }

