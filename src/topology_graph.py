import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection

try:
    import faiss
except ImportError:
    faiss = None


def sanitize_name(name):
    return name.replace("\\", "-").replace("/", "-").replace(" ", "_")


def parse_multi_scale_ks(k, multi_scale_ks=None):
    if multi_scale_ks is None:
        return [int(k)]
    if isinstance(multi_scale_ks, str):
        values = [item.strip() for item in multi_scale_ks.split(",") if item.strip()]
        ks = [int(item) for item in values]
    else:
        ks = [int(item) for item in multi_scale_ks]
    ks = sorted(set(max(1, int(item)) for item in ks))
    return ks or [int(k)]


def build_feature_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    return Path(args.feature_cache_root) / args.dataset / args.split / model_tag


def build_output_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    k_list = parse_multi_scale_ks(args.k, getattr(args, "multi_scale_ks", None))
    if len(k_list) == 1:
        graph_tag = f"k{k_list[0]}_{sanitize_name(args.metric)}"
    else:
        graph_tag = f"ks{'-'.join(str(item) for item in k_list)}_{sanitize_name(args.metric)}"
    return Path(args.output_root) / args.dataset / args.split / model_tag / args.modality / graph_tag


def load_feature_tensor(feature_dir, modality):
    candidates = {
        "image": ["img_features_selection.pt", "img_features.pt"],
        "text": ["txt_features_selection.pt", "txt_features.pt"],
    }[modality]
    path = None
    for filename in candidates:
        candidate_path = Path(feature_dir) / filename
        if candidate_path.exists():
            path = candidate_path
            break
    if path is None:
        raise FileNotFoundError(f"No cached feature file found for modality={modality} under {feature_dir}")
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


def preprocess_features_for_knn(features, method="none", target_dim=None, random_state=0):
    features = np.asarray(features, dtype=np.float32)
    original_dim = int(features.shape[1])

    if method in {None, "none"} or target_dim is None or int(target_dim) <= 0 or original_dim <= int(target_dim):
        return features, {
            "pre_knn_method": "none",
            "original_dim": original_dim,
        "knn_dim": original_dim,
        }

    target_dim = int(target_dim)
    if method == "pca":
        transformer = PCA(n_components=target_dim, svd_solver="randomized", random_state=int(random_state))
        reduced = transformer.fit_transform(features).astype(np.float32)
    elif method == "random_projection":
        transformer = GaussianRandomProjection(n_components=target_dim, random_state=int(random_state))
        reduced = transformer.fit_transform(features).astype(np.float32)
    else:
        raise ValueError(f"Unsupported pre-kNN reduction method: {method}")

    return reduced, {
        "pre_knn_method": str(method),
        "original_dim": original_dim,
        "knn_dim": int(reduced.shape[1]),
    }


def reduce_graph_features(features, method="pca", target_dim=256, random_state=0):
    return preprocess_features_for_knn(
        features,
        method=method,
        target_dim=target_dim,
        random_state=random_state,
    )


def normalize_for_cosine(features, eps=1e-12):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (features / norms).astype(np.float32)


def compute_knn_sklearn(features, k, metric, n_jobs=None):
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


def compute_knn_faiss(features, k, metric, use_gpu=False):
    if faiss is None:
        raise ImportError("faiss is not installed, but faiss backend was requested.")

    features = np.ascontiguousarray(features.astype(np.float32))
    num_nodes = features.shape[0]
    if num_nodes < 2:
        raise ValueError("At least two feature vectors are required to build a topology graph.")

    query_features = features
    if metric == "cosine":
        query_features = normalize_for_cosine(query_features)
        index = faiss.IndexFlatIP(query_features.shape[1])
    elif metric == "euclidean":
        index = faiss.IndexFlatL2(query_features.shape[1])
    else:
        raise ValueError(f"Unsupported metric for faiss backend: {metric}")

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError("Installed faiss package does not support GPU.")
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)

    index.add(query_features)
    distances, indices = index.search(query_features, min(int(k) + 1, num_nodes))

    if metric == "cosine":
        distances = 1.0 - distances

    if np.all(indices[:, 0] == np.arange(num_nodes)):
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        distances = distances[:, : min(k, indices.shape[1])]
        indices = indices[:, : min(k, indices.shape[1])]

    return indices.astype(np.int64), distances.astype(np.float32)


def resolve_knn_backend(backend, use_gpu):
    backend = (backend or "auto").lower()
    if backend == "auto":
        if faiss is not None:
            if use_gpu and hasattr(faiss, "StandardGpuResources"):
                return "faiss"
            return "faiss"
        return "sklearn"
    if backend == "faiss" and faiss is None:
        raise ImportError("faiss backend requested but faiss is not installed.")
    return backend


def compute_knn(features, k, metric, n_jobs=None, backend="auto", use_gpu=False):
    backend = resolve_knn_backend(backend, use_gpu)
    if backend == "faiss":
        return compute_knn_faiss(features, k, metric, use_gpu=use_gpu), backend
    if backend == "sklearn":
        return compute_knn_sklearn(features, k, metric, n_jobs=n_jobs), backend
    raise ValueError(f"Unsupported kNN backend: {backend}")


def compute_local_scale_knn(
    features,
    k,
    metric,
    n_jobs=None,
    backend="auto",
    use_gpu=False,
    local_connectivity=1.0,
    bandwidth=None,
    sigma_search_steps=64,
):
    (knn_indices, knn_distances), resolved_backend = compute_knn(
        features,
        k,
        metric,
        n_jobs=n_jobs,
        backend=backend,
        use_gpu=use_gpu,
    )
    rho = compute_rho(knn_distances, local_connectivity=local_connectivity)
    sigma = compute_sigmas(
        knn_distances,
        rho,
        bandwidth=bandwidth,
        n_iter=sigma_search_steps,
        target=bandwidth if bandwidth is not None else None,
    )
    directed_graph = build_directed_graph(knn_indices, knn_distances, rho, sigma, features.shape[0])
    symmetric_graph = symmetrize_graph(directed_graph)
    return {
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
        "rho": rho,
        "sigma": sigma,
        "directed_graph": directed_graph,
        "symmetric_graph": symmetric_graph,
        "resolved_backend": resolved_backend,
    }


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


def fuzzy_union_merge(graph_list, merge_mode="mean"):
    if not graph_list:
        raise ValueError("graph_list must not be empty.")
    if len(graph_list) == 1:
        return graph_list[0].tocsr()

    if merge_mode == "max":
        merged = graph_list[0].tocsr(copy=True)
        for graph in graph_list[1:]:
            merged = merged.maximum(graph.tocsr())
    elif merge_mode == "mean":
        merged = graph_list[0].tocsr(copy=True)
        for graph in graph_list[1:]:
            merged = merged + graph.tocsr()
        merged = merged.multiply(1.0 / float(len(graph_list)))
    elif merge_mode == "union":
        merged = graph_list[0].tocsr(copy=True)
        for graph in graph_list[1:]:
            merged = merged + graph - merged.multiply(graph)
    else:
        raise ValueError(f"Unsupported multi-scale merge mode: {merge_mode}")

    merged = merged.tocsr()
    merged.eliminate_zeros()
    return merged


def add_mst_connectivity(graph, reference_distances, sigma=None, weight_scale=1.0):
    graph = graph.tocsr(copy=True)
    distance_graph = reference_distances.tocsr(copy=True)
    mst = csgraph.minimum_spanning_tree(distance_graph)
    mst = mst + mst.transpose()
    mst = mst.tocsr()
    mst.eliminate_zeros()
    if mst.nnz == 0:
        return graph

    mst_weights = np.asarray(mst.data, dtype=np.float32)
    sigma_ref = float(np.mean(sigma)) if sigma is not None and len(sigma) > 0 else 1.0
    sigma_ref = max(sigma_ref, 1e-6)
    mst_weights = np.exp(-(mst_weights / sigma_ref)) * float(weight_scale)
    mst_weights = np.clip(mst_weights, 1e-6, 1.0).astype(np.float32)
    mst_graph = sparse.csr_matrix((mst_weights, mst.indices.copy(), mst.indptr.copy()), shape=mst.shape)
    augmented = graph.maximum(mst_graph)
    augmented = augmented.tocsr()
    augmented.eliminate_zeros()
    return augmented


def build_multiscale_fuzzy_graph(
    features,
    k_list,
    metric,
    n_jobs=None,
    backend="auto",
    use_gpu=False,
    local_connectivity=1.0,
    bandwidth=None,
    sigma_search_steps=64,
    merge_mode="mean",
    use_mst_connectivity=False,
    mst_weight_scale=1.0,
):
    k_list = parse_multi_scale_ks(k_list[0] if isinstance(k_list, list) and k_list else 15, k_list)
    per_scale_outputs = []
    directed_graphs = []
    symmetric_graphs = []
    reference_distances = None
    resolved_backend = None

    for k in k_list:
        scale_output = compute_local_scale_knn(
            features,
            k=k,
            metric=metric,
            n_jobs=n_jobs,
            backend=backend,
            use_gpu=use_gpu,
            local_connectivity=local_connectivity,
            bandwidth=bandwidth,
            sigma_search_steps=sigma_search_steps,
        )
        scale_output["k"] = int(k)
        per_scale_outputs.append(scale_output)
        directed_graphs.append(scale_output["directed_graph"])
        symmetric_graphs.append(scale_output["symmetric_graph"])
        resolved_backend = scale_output["resolved_backend"]

        row_ids = np.repeat(np.arange(features.shape[0]), scale_output["knn_indices"].shape[1])
        col_ids = scale_output["knn_indices"].reshape(-1)
        dist_vals = scale_output["knn_distances"].reshape(-1).astype(np.float32)
        local_reference = sparse.csr_matrix((dist_vals, (row_ids, col_ids)), shape=(features.shape[0], features.shape[0]))
        local_reference = local_reference.minimum(local_reference.transpose()) + local_reference.maximum(local_reference.transpose())
        local_reference = local_reference.tocsr()
        local_reference.eliminate_zeros()
        if reference_distances is None or int(k) == max(k_list):
            reference_distances = local_reference

    merged_directed = fuzzy_union_merge(directed_graphs, merge_mode=merge_mode)
    merged_symmetric = fuzzy_union_merge(symmetric_graphs, merge_mode=merge_mode)
    merged_symmetric = symmetrize_graph(merged_symmetric)

    if use_mst_connectivity:
        merged_symmetric = add_mst_connectivity(
            merged_symmetric,
            reference_distances=reference_distances,
            sigma=per_scale_outputs[-1]["sigma"] if per_scale_outputs else None,
            weight_scale=mst_weight_scale,
        )

    return {
        "k_list": k_list,
        "per_scale_outputs": per_scale_outputs,
        "directed_graph": merged_directed,
        "symmetric_graph": merged_symmetric,
        "resolved_backend": resolved_backend,
    }


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


def build_spectral_embedding(eigenvalues, eigenvectors, embedding_dim=None, drop_first=True):
    if eigenvectors is None:
        return None
    start = 1 if drop_first and eigenvectors.shape[1] > 1 else 0
    if embedding_dim is None:
        end = eigenvectors.shape[1]
    else:
        end = min(eigenvectors.shape[1], start + int(embedding_dim))
    embedding = eigenvectors[:, start:end].astype(np.float32)
    if embedding.size == 0:
        embedding = eigenvectors[:, : min(1, eigenvectors.shape[1])].astype(np.float32)
    return embedding


def build_graph_artifacts(graph, num_eigs, save_eigenvectors=True, embedding_dim=None):
    transition = row_normalize_graph(graph)
    laplacian = build_laplacian(graph, normalized=True)
    eigenvalues, eigenvectors = compute_spectrum(laplacian, num_eigs, return_eigenvectors=save_eigenvectors or embedding_dim is not None)
    spectral_embedding = build_spectral_embedding(eigenvalues, eigenvectors, embedding_dim=embedding_dim) if (save_eigenvectors or embedding_dim is not None) else None
    collapse_metrics = compute_collapse_metrics(eigenvalues)
    return {
        "adjacency": graph,
        "transition": transition,
        "laplacian_sym": laplacian,
        "eigvals": eigenvalues,
        "eigvecs": eigenvectors,
        "spectral_embedding": spectral_embedding,
        "collapse_metrics": collapse_metrics,
    }


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

    try:
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=num_eigs,
            which="SM",
            return_eigenvectors=True,
        )
    except ArpackNoConvergence as exc:
        try:
            retry_ncv = min(num_nodes - 1, max(2 * num_eigs + 1, 32))
            eigenvalues, eigenvectors = eigsh(
                laplacian,
                k=num_eigs,
                which="SM",
                return_eigenvectors=True,
                ncv=retry_ncv,
                maxiter=200000,
                tol=1e-4,
            )
        except ArpackNoConvergence as retry_exc:
            partial_values = retry_exc.eigenvalues if retry_exc.eigenvalues is not None else exc.eigenvalues
            partial_vectors = retry_exc.eigenvectors if retry_exc.eigenvectors is not None else exc.eigenvectors
            if partial_values is None or len(partial_values) < 2:
                raise
            print(
                f"[topology] warning: ARPACK did not fully converge for k={num_eigs}; "
                f"using {len(partial_values)} converged eigenpairs instead.",
                flush=True,
            )
            eigenvalues = partial_values
            eigenvectors = partial_vectors
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


def summarize_results(args, features, knn_indices, sym_graph, eigenvalues, collapse_metrics, per_scale_stats=None):
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
        "knn_backend": getattr(args, "resolved_knn_backend", getattr(args, "knn_backend", "sklearn")),
        "k": int(args.k),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "knn_width": int(knn_indices.shape[1]),
        "feature_dim": int(features.shape[1]),
        "pre_knn_method": getattr(args, "pre_knn_method", "none"),
        "pre_knn_dim": getattr(args, "pre_knn_dim", None),
        "original_feature_dim": getattr(args, "original_feature_dim", int(features.shape[1])),
        "first_eigenvalues": [float(x) for x in eigenvalues[: min(10, len(eigenvalues))]],
        "spectral_entropy": collapse_metrics["spectral_entropy"],
        "collapse_score": collapse_metrics["collapse_score"],
        "num_eigs_used": collapse_metrics["num_eigs_used"],
        "multi_scale_ks": parse_multi_scale_ks(args.k, getattr(args, "multi_scale_ks", None)),
        "multi_scale_merge_mode": getattr(args, "multi_scale_merge_mode", "mean"),
        "use_mst_connectivity": bool(getattr(args, "use_mst_connectivity", False)),
        "mst_weight_scale": float(getattr(args, "mst_weight_scale", 1.0)),
        "per_scale_stats": per_scale_stats or [],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return summary


def save_outputs(
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
    spectral_embedding,
    sample_meta,
    summary,
):
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
    sparse.save_npz(output_dir / "A_directed.npz", directed_graph)
    sparse.save_npz(output_dir / "symmetric_graph.npz", sym_graph)
    sparse.save_npz(output_dir / "adjacency.npz", sym_graph)
    sparse.save_npz(output_dir / "B_graph.npz", sym_graph)
    sparse.save_npz(output_dir / "transition_graph.npz", transition_graph)
    sparse.save_npz(output_dir / "laplacian_normalized.npz", laplacian)
    sparse.save_npz(output_dir / "L_sym.npz", laplacian)

    torch.save(torch.from_numpy(eigenvalues), output_dir / "eigenvalues.pt")
    if eigenvectors is not None:
        torch.save(torch.from_numpy(eigenvectors), output_dir / "eigenvectors.pt")
    if spectral_embedding is not None:
        torch.save(torch.from_numpy(spectral_embedding), output_dir / "spectral_embedding.pt")

    with open(output_dir / "sample_meta.json", "w", encoding="utf-8") as handle:
        json.dump(sample_meta, handle, ensure_ascii=False, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def run_topology_graph(args):
    feature_dir = build_feature_dir(args)
    output_dir = build_output_dir(args)

    print(f"[topology] loading features from {feature_dir} for modality={args.modality}")
    features = load_feature_tensor(feature_dir, args.modality)
    sample_meta = load_sample_meta(feature_dir)
    features, sample_meta = maybe_truncate(features, sample_meta, args.max_samples)
    args.original_feature_dim = int(features.shape[1])
    print(f"[topology] feature shape: {features.shape}")

    graph_reduce_method = getattr(args, "graph_reduce_method", getattr(args, "pre_knn_method", "none"))
    graph_feature_dim = getattr(args, "graph_feature_dim", getattr(args, "pre_knn_dim", None))
    reduced_features, reduction_info = reduce_graph_features(
        features,
        method=graph_reduce_method,
        target_dim=graph_feature_dim,
        random_state=getattr(args, "random_state", 0),
    )
    args.pre_knn_method = reduction_info["pre_knn_method"]
    args.pre_knn_dim = reduction_info["knn_dim"] if reduction_info["pre_knn_method"] != "none" else None
    args.graph_reduce_method = reduction_info["pre_knn_method"]
    args.graph_feature_dim = reduction_info["knn_dim"] if reduction_info["pre_knn_method"] != "none" else None
    args.original_feature_dim = reduction_info["original_dim"]
    if reduction_info["pre_knn_method"] != "none":
        print(
            f"[topology] reduced features for kNN: method={reduction_info['pre_knn_method']} "
            f"{reduction_info['original_dim']} -> {reduction_info['knn_dim']}"
        )

    k_list = parse_multi_scale_ks(args.k, getattr(args, "multi_scale_ks", None))
    print(
        f"[topology] computing kNN: ks={k_list}, metric={args.metric}, "
        f"backend={getattr(args, 'knn_backend', 'auto')}, n_jobs={args.n_jobs}"
    )
    multiscale_outputs = build_multiscale_fuzzy_graph(
        reduced_features,
        k_list=k_list,
        metric=args.metric,
        n_jobs=args.n_jobs,
        backend=getattr(args, "knn_backend", "auto"),
        use_gpu=bool(getattr(args, "faiss_use_gpu", False)),
        local_connectivity=args.local_connectivity,
        bandwidth=args.bandwidth,
        sigma_search_steps=args.sigma_search_steps,
        merge_mode=getattr(args, "multi_scale_merge_mode", "mean"),
        use_mst_connectivity=bool(getattr(args, "use_mst_connectivity", False)),
        mst_weight_scale=getattr(args, "mst_weight_scale", 1.0),
    )
    args.resolved_knn_backend = multiscale_outputs["resolved_backend"]
    print(f"[topology] kNN backend resolved to: {args.resolved_knn_backend}")

    per_scale_stats = []
    for scale_output in multiscale_outputs["per_scale_outputs"]:
        per_scale_stats.append(
            {
                "k": int(scale_output["k"]),
                "knn_width": int(scale_output["knn_indices"].shape[1]),
                "mean_rho": float(np.mean(scale_output["rho"])),
                "mean_sigma": float(np.mean(scale_output["sigma"])),
                "num_edges": int(scale_output["symmetric_graph"].nnz),
            }
        )

    knn_indices = multiscale_outputs["per_scale_outputs"][-1]["knn_indices"]
    knn_distances = multiscale_outputs["per_scale_outputs"][-1]["knn_distances"]
    rho = multiscale_outputs["per_scale_outputs"][-1]["rho"]
    sigma = multiscale_outputs["per_scale_outputs"][-1]["sigma"]
    directed_graph = multiscale_outputs["directed_graph"]
    sym_graph = multiscale_outputs["symmetric_graph"]

    print("[topology] building graph artifacts")
    graph_artifacts = build_graph_artifacts(
        sym_graph,
        args.num_eigs,
        save_eigenvectors=args.save_eigenvectors,
        embedding_dim=getattr(args, "spectral_embedding_dim", None),
    )
    transition_graph = graph_artifacts["transition"]
    laplacian = graph_artifacts["laplacian_sym"]
    eigenvalues = graph_artifacts["eigvals"]
    eigenvectors = graph_artifacts["eigvecs"]
    spectral_embedding = graph_artifacts["spectral_embedding"]
    collapse_metrics = graph_artifacts["collapse_metrics"]
    summary = summarize_results(args, features, knn_indices, sym_graph, eigenvalues, collapse_metrics, per_scale_stats=per_scale_stats)

    print(f"[topology] saving outputs to {output_dir}")
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
        spectral_embedding,
        sample_meta,
        summary,
    )
    return {
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "summary": summary,
    }
