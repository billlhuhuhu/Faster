import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans

from src.graph_wavelet import sparsify_sparse_matrix


def add_topology_visualization_args(parser):
    parser.add_argument("--enable_topology_visualization", action="store_true")
    parser.add_argument("--visualization_output_dir", type=str, default=None)
    parser.add_argument("--visualization_topk_edges", type=int, default=8)
    parser.add_argument(
        "--visualization_node_order_mode",
        type=str,
        default="fused_diffusion",
        choices=["fused_spectral", "fused_diffusion", "fused_cluster", "raw"],
    )
    parser.add_argument(
        "--visualization_layout_mode",
        type=str,
        default="fused_diffusion",
        choices=["fused_spectral", "fused_diffusion", "spring_on_fused"],
    )
    parser.add_argument("--visualization_num_local_cases", type=int, default=6)
    parser.add_argument("--visualization_local_case_topk", type=int, default=12)
    parser.add_argument("--visualization_max_heatmap_nodes", type=int, default=768)
    parser.add_argument("--visualization_show_labels_if_available", action="store_true")
    return parser


def extract_topology_visualization_config(args):
    return {
        "visualization_output_dir": getattr(args, "visualization_output_dir", None),
        "visualization_topk_edges": int(getattr(args, "visualization_topk_edges", 8)),
        "visualization_node_order_mode": str(getattr(args, "visualization_node_order_mode", "fused_diffusion")),
        "visualization_layout_mode": str(getattr(args, "visualization_layout_mode", "fused_diffusion")),
        "visualization_num_local_cases": int(getattr(args, "visualization_num_local_cases", 6)),
        "visualization_local_case_topk": int(getattr(args, "visualization_local_case_topk", 12)),
        "visualization_max_heatmap_nodes": int(getattr(args, "visualization_max_heatmap_nodes", 768)),
        "visualization_show_labels_if_available": bool(getattr(args, "visualization_show_labels_if_available", False)),
    }


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    return plt, LineCollection


def _resolve_existing_path(path_str, base_dir):
    raw = Path(path_str)
    candidates = [
        raw,
        Path.cwd() / raw,
        Path(base_dir) / raw,
        Path(base_dir).parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve path from saved result: {path_str}")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cross_modal_result_bundle(result_dir):
    result_dir = Path(result_dir)
    summary = _load_json(result_dir / "summary.json")
    modality_selection = _load_json(result_dir / "modality_selection.json")
    sample_meta = _load_json(result_dir / "sample_meta.json")

    healthy_modality = str(summary.get("healthy_modality", "image"))
    healthy_graph_dir = _resolve_existing_path(modality_selection["healthy_graph_dir"], result_dir)
    collapsed_graph_dir = _resolve_existing_path(modality_selection["collapsed_graph_dir"], result_dir)

    if healthy_modality == "image":
        image_graph_dir = healthy_graph_dir
        text_graph_dir = collapsed_graph_dir
    else:
        image_graph_dir = collapsed_graph_dir
        text_graph_dir = healthy_graph_dir

    image_graph = sparse.load_npz(image_graph_dir / "symmetric_graph.npz").tocsr().astype(np.float32)
    text_graph = sparse.load_npz(text_graph_dir / "symmetric_graph.npz").tocsr().astype(np.float32)
    corrected_image = sparse.load_npz(result_dir / "corrected_image_graph_symmetric.npz").tocsr().astype(np.float32)
    corrected_text = sparse.load_npz(result_dir / "corrected_text_graph_symmetric.npz").tocsr().astype(np.float32)
    fused_graph = sparse.load_npz(result_dir / "B_star.npz").tocsr().astype(np.float32)

    embedding_path = result_dir / "unified_spectral_embedding.npy"
    embedding = np.load(embedding_path) if embedding_path.exists() else None

    eigvec_path = result_dir / "unified_eigvecs.npy"
    eigvecs = np.load(eigvec_path) if eigvec_path.exists() else None

    return {
        "result_dir": result_dir,
        "summary": summary,
        "sample_meta": sample_meta,
        "embedding": embedding,
        "eigvecs": eigvecs,
        "graphs": {
            "modal_A": image_graph,
            "modal_B": text_graph,
            "corr_A": corrected_image,
            "corr_B": corrected_text,
            "fused": fused_graph,
        },
        "graph_dirs": {
            "modal_A": str(image_graph_dir),
            "modal_B": str(text_graph_dir),
        },
        "modality_name_map": {
            "modal_A": "image",
            "modal_B": "text",
            "corr_A": "corrected_image",
            "corr_B": "corrected_text",
            "fused": "B_star",
        },
    }


def _get_base_embedding(bundle):
    embedding = bundle.get("embedding")
    if embedding is not None and embedding.ndim == 2 and embedding.shape[0] == bundle["graphs"]["fused"].shape[0]:
        return np.asarray(embedding, dtype=np.float32)
    eigvecs = bundle.get("eigvecs")
    if eigvecs is not None and eigvecs.ndim == 2 and eigvecs.shape[0] == bundle["graphs"]["fused"].shape[0]:
        if eigvecs.shape[1] >= 3:
            return np.asarray(eigvecs[:, 1:3], dtype=np.float32)
        return np.asarray(eigvecs[:, : min(2, eigvecs.shape[1])], dtype=np.float32)
    num_nodes = int(bundle["graphs"]["fused"].shape[0])
    return np.stack(
        [
            np.linspace(-1.0, 1.0, num_nodes, dtype=np.float32),
            np.zeros(num_nodes, dtype=np.float32),
        ],
        axis=1,
    )


def _ensure_two_dimensional(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("Expected 2D features for visualization.")
    if features.shape[1] >= 2:
        return features[:, :2].astype(np.float32)
    return np.concatenate([features[:, :1], np.zeros((features.shape[0], 1), dtype=np.float32)], axis=1)


def _compute_cluster_labels(features, max_clusters=12, random_state=0):
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] <= 1:
        return np.zeros(features.shape[0], dtype=np.int32)
    n_clusters = max(2, min(int(max_clusters), max(2, int(np.sqrt(features.shape[0] / 8.0)))))
    if n_clusters >= features.shape[0]:
        return np.arange(features.shape[0], dtype=np.int32)
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=min(4096, features.shape[0]))
    labels = model.fit_predict(features)
    return labels.astype(np.int32)


def _compute_node_order(bundle, order_mode):
    order_mode = str(order_mode)
    base = _get_base_embedding(bundle)
    fused = bundle["graphs"]["fused"]
    num_nodes = int(fused.shape[0])
    if order_mode == "raw":
        order = np.arange(num_nodes, dtype=np.int64)
        return order, None

    features = _ensure_two_dimensional(base)
    if order_mode == "fused_cluster":
        labels = _compute_cluster_labels(features, random_state=0)
        order = np.lexsort((features[:, 1], features[:, 0], labels))
        return order.astype(np.int64), labels

    primary = features[:, 0]
    secondary = features[:, 1]
    order = np.lexsort((secondary, primary))
    labels = _compute_cluster_labels(features, random_state=0)
    return order.astype(np.int64), labels


def _compute_layout_coordinates(bundle, layout_mode, display_indices=None):
    layout_mode = str(layout_mode)
    base = _ensure_two_dimensional(_get_base_embedding(bundle))
    if display_indices is None:
        display_indices = np.arange(base.shape[0], dtype=np.int64)
    display_indices = np.asarray(display_indices, dtype=np.int64)
    coords = base[display_indices].astype(np.float32)

    if layout_mode == "spring_on_fused":
        try:
            import networkx as nx

            fused_graph = bundle["graphs"]["fused"][display_indices][:, display_indices].tocsr()
            fused_graph = sparsify_sparse_matrix(fused_graph, topk=6, threshold=None, use_abs=True)
            graph_nx = nx.from_scipy_sparse_array(fused_graph, edge_attribute="weight")
            seed_positions = {int(idx): coords[pos] for pos, idx in enumerate(range(coords.shape[0]))}
            spring = nx.spring_layout(graph_nx, seed=0, pos=seed_positions, weight="weight", dim=2)
            coords = np.asarray([spring[i] for i in range(coords.shape[0])], dtype=np.float32)
        except Exception:
            pass
    coords = coords - np.mean(coords, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(coords, axis=1))
    if scale > 0:
        coords = coords / scale
    return coords.astype(np.float32)


def _select_display_indices(order, max_nodes):
    order = np.asarray(order, dtype=np.int64)
    max_nodes = max(1, int(max_nodes))
    if order.size <= max_nodes:
        return order
    positions = np.linspace(0, order.size - 1, max_nodes, dtype=np.int64)
    return order[positions]


def _sparse_abs_row_sum(matrix):
    matrix = matrix.tocsr().copy()
    matrix.data = np.abs(matrix.data)
    return np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float32)


def _extract_optional_labels(sample_meta):
    preferred_keys = ["label", "class_id", "class", "category", "cluster"]
    if not sample_meta:
        return None, None
    for key in preferred_keys:
        if key in sample_meta[0]:
            values = [item.get(key) for item in sample_meta]
            return np.asarray(values), key
    return None, None


def _graph_stats(graph):
    graph = graph.tocsr().astype(np.float32)
    degree = np.asarray(graph.sum(axis=1)).reshape(-1).astype(np.float32)
    degree_binary = np.asarray((graph > 0).sum(axis=1)).reshape(-1).astype(np.float32)
    edge_weights = graph.data.astype(np.float32)
    binary_graph = graph.copy()
    binary_graph.data = np.ones_like(binary_graph.data, dtype=np.float32)
    component_count, component_labels = connected_components(binary_graph, directed=False, connection="weak")
    clustering = None
    clustering_mean = None
    if int(graph.shape[0]) <= 2000:
        adjacency = binary_graph.tocsr().astype(np.float32)
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        triangles = (adjacency @ adjacency @ adjacency).diagonal().astype(np.float32) / 2.0
        denom = degree_binary * np.maximum(degree_binary - 1.0, 0.0) / 2.0
        clustering = np.divide(triangles, np.maximum(denom, 1.0), where=denom > 0)
        clustering[denom <= 0] = 0.0
        clustering_mean = float(np.mean(clustering)) if clustering.size else 0.0
    num_nodes = int(graph.shape[0])
    num_edges = int(graph.nnz)
    return {
        "degree": degree,
        "degree_binary": degree_binary,
        "edge_weights": edge_weights,
        "component_labels": component_labels.astype(np.int32),
        "num_components": int(component_count),
        "density": float(num_edges / max(num_nodes * num_nodes, 1)),
        "num_nodes": num_nodes,
        "nnz": num_edges,
        "avg_degree": float(np.mean(degree)) if degree.size else 0.0,
        "max_degree": float(np.max(degree)) if degree.size else 0.0,
        "mean_edge_weight": float(np.mean(edge_weights)) if edge_weights.size else 0.0,
        "clustering": None if clustering is None else clustering.astype(np.float32),
        "clustering_mean": clustering_mean,
    }


def _diff_stats(graph_a, graph_b):
    delta = (graph_a - graph_b).tocsr().astype(np.float32)
    if delta.nnz == 0:
        fro_norm = 0.0
        mean_abs = 0.0
        changed_edge_ratio = 0.0
    else:
        fro_norm = float(np.sqrt(np.sum(delta.data.astype(np.float64) ** 2)))
        mean_abs = float(np.mean(np.abs(delta.data.astype(np.float64))))
        union_nnz = max(int((graph_a != 0).nnz + (graph_b != 0).nnz), 1)
        changed_edge_ratio = float(delta.nnz / union_nnz)
    return {
        "frobenius_norm": fro_norm,
        "mean_abs_diff": mean_abs,
        "changed_edge_ratio": changed_edge_ratio,
    }


def _gather_heatmap_matrix(graph, ordered_indices):
    matrix = graph[ordered_indices][:, ordered_indices].toarray().astype(np.float32)
    return matrix


def _compute_positive_heatmap_scale(matrices, eps=1e-8):
    positives = []
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=np.float32)
        values = matrix[matrix > 0]
        if values.size:
            positives.append(values.astype(np.float32))
    if not positives:
        return {"lower_ref": 1.0, "upper_ref": 1.0, "display_vmax": 1.0}
    merged = np.concatenate(positives, axis=0).astype(np.float32)
    lower_ref = float(np.percentile(merged, 10))
    upper_ref = float(np.percentile(merged, 99))
    lower_ref = max(lower_ref, float(eps))
    upper_ref = max(upper_ref, lower_ref + float(eps))
    display_vmax = float(np.log1p(upper_ref / lower_ref))
    return {
        "lower_ref": lower_ref,
        "upper_ref": upper_ref,
        "display_vmax": max(display_vmax, float(eps)),
    }


def _transform_positive_heatmap(matrix, lower_ref, eps=1e-8):
    matrix = np.asarray(matrix, dtype=np.float32)
    transformed = np.zeros_like(matrix, dtype=np.float32)
    mask = matrix > 0
    if np.any(mask):
        transformed[mask] = np.log1p(matrix[mask] / max(float(lower_ref), float(eps))).astype(np.float32)
    return np.ma.masked_where(~mask, transformed)


def _compute_signed_heatmap_scale(matrices, eps=1e-8):
    nonzero_values = []
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=np.float32)
        values = np.abs(matrix[np.abs(matrix) > 0])
        if values.size:
            nonzero_values.append(values.astype(np.float32))
    if not nonzero_values:
        return {"lower_ref": 1.0, "upper_ref": 1.0, "display_vmax": 1.0}
    merged = np.concatenate(nonzero_values, axis=0).astype(np.float32)
    lower_ref = float(np.percentile(merged, 10))
    upper_ref = float(np.percentile(merged, 99))
    lower_ref = max(lower_ref, float(eps))
    upper_ref = max(upper_ref, lower_ref + float(eps))
    display_vmax = float(np.log1p(upper_ref / lower_ref))
    return {
        "lower_ref": lower_ref,
        "upper_ref": upper_ref,
        "display_vmax": max(display_vmax, float(eps)),
    }


def _transform_signed_heatmap(matrix, lower_ref, eps=1e-8):
    matrix = np.asarray(matrix, dtype=np.float32)
    transformed = np.zeros_like(matrix, dtype=np.float32)
    abs_matrix = np.abs(matrix)
    mask = abs_matrix > 0
    if np.any(mask):
        transformed[mask] = np.sign(matrix[mask]) * np.log1p(abs_matrix[mask] / max(float(lower_ref), float(eps))).astype(np.float32)
    return np.ma.masked_where(~mask, transformed)


def _save_matrix_heatmap(plt, matrix, output_path, title, cmap="magma", vmin=None, vmax=None, colorbar_label=None):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")
    image = ax.imshow(matrix, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Ordered Nodes")
    ax.set_ylabel("Ordered Nodes")
    cbar = fig.colorbar(image, ax=ax, shrink=0.8)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _ordered_unique_labels(labels):
    labels = np.asarray(labels)
    unique_labels, first_positions = np.unique(labels, return_index=True)
    order = np.argsort(first_positions, kind="stable")
    return unique_labels[order]


def _aggregate_matrix_by_labels(matrix, labels):
    matrix = np.asarray(matrix, dtype=np.float32)
    labels = np.asarray(labels)
    unique_labels = _ordered_unique_labels(labels)
    block = np.zeros((unique_labels.size, unique_labels.size), dtype=np.float32)
    block_sizes = np.zeros(unique_labels.size, dtype=np.int32)
    groups = []
    for idx, label in enumerate(unique_labels.tolist()):
        group = np.where(labels == label)[0]
        groups.append(group)
        block_sizes[idx] = int(group.size)
    for i, row_group in enumerate(groups):
        row_scale = max(int(row_group.size), 1)
        for j, col_group in enumerate(groups):
            sub_block = matrix[np.ix_(row_group, col_group)]
            block[i, j] = float(np.sum(sub_block, dtype=np.float64) / row_scale)
    return block.astype(np.float32), unique_labels.astype(np.int32), block_sizes


def _save_dual_heatmap_figure(
    plt,
    raw_matrix,
    block_matrix,
    output_path,
    title,
    cmap="magma",
    vmin=None,
    vmax=None,
    colorbar_label=None,
    block_labels=None,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=160)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="white")

    raw_image = axes[0].imshow(raw_matrix, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    axes[0].set_title("Node-Level")
    axes[0].set_xlabel("Ordered Nodes")
    axes[0].set_ylabel("Ordered Nodes")

    block_image = axes[1].imshow(block_matrix, cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    axes[1].set_title("Cluster-Aggregated")
    axes[1].set_xlabel("Cluster Index")
    axes[1].set_ylabel("Cluster Index")
    if block_labels is not None and len(block_labels) <= 24:
        tick_positions = np.arange(len(block_labels))
        axes[1].set_xticks(tick_positions)
        axes[1].set_yticks(tick_positions)
        axes[1].set_xticklabels([str(x) for x in block_labels], rotation=45, ha="right")
        axes[1].set_yticklabels([str(x) for x in block_labels])

    cbar = fig.colorbar(block_image, ax=axes.tolist(), shrink=0.82)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _prepare_edge_segments(graph, coords, topk_edges):
    if topk_edges is not None and int(topk_edges) > 0:
        graph = sparsify_sparse_matrix(graph, topk=max(1, int(topk_edges)), threshold=None, use_abs=True)
    coo = sparse.triu(graph, k=1).tocoo()
    if coo.nnz == 0:
        return np.empty((0, 2, 2), dtype=np.float32), np.empty(0, dtype=np.float32)
    segments = np.stack(
        [
            np.stack([coords[coo.row], coords[coo.col]], axis=1),
        ],
        axis=0,
    )[0].astype(np.float32)
    weights = coo.data.astype(np.float32)
    return segments, weights


def _save_layout_figure(
    plt,
    LineCollection,
    graph,
    coords,
    node_colors,
    output_path,
    title,
    topk_edges=8,
    colorbar=False,
    cmap="tab20",
    node_vmin=None,
    node_vmax=None,
    colorbar_label=None,
    node_sizes=None,
):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
    graph = graph.tocsr().astype(np.float32)
    segments, weights = _prepare_edge_segments(graph, coords, topk_edges=topk_edges)
    if segments.shape[0] > 0:
        alpha = np.clip(weights / max(np.percentile(weights, 95), 1e-8), 0.05, 1.0)
        lc = LineCollection(segments, colors=[(0.45, 0.45, 0.45, float(a) * 0.6) for a in alpha], linewidths=0.5)
        ax.add_collection(lc)
    if node_sizes is None:
        node_sizes = 12
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=node_colors,
        s=node_sizes,
        cmap=cmap,
        vmin=node_vmin,
        vmax=node_vmax,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    if colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
        if colorbar_label:
            cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_distribution_plot(plt, values_map, output_path, title, xlabel):
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    for name, values in values_map.items():
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            continue
        bins = min(60, max(12, int(np.sqrt(values.size))))
        ax.hist(values, bins=bins, density=True, alpha=0.35, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_graph_overview_plot(plt, graph_stats, output_path):
    names = list(graph_stats.keys())
    metrics = {
        "num_components": [float(graph_stats[name]["num_components"]) for name in names],
        "density": [float(graph_stats[name]["density"]) for name in names],
        "avg_degree": [float(graph_stats[name]["avg_degree"]) for name in names],
        "clustering_mean": [
            0.0 if graph_stats[name]["clustering_mean"] is None else float(graph_stats[name]["clustering_mean"])
            for name in names
        ],
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.reshape(-1)
    for ax, (metric_name, values) in zip(axes, metrics.items()):
        ax.bar(np.arange(len(names)), values, color="#4C72B0")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=20)
        ax.set_title(metric_name)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _compute_positive_vector_scale(values_list, eps=1e-8):
    positives = []
    for values in values_list:
        values = np.asarray(values, dtype=np.float32)
        cur = values[values > 0]
        if cur.size:
            positives.append(cur.astype(np.float32))
    if not positives:
        return {"lower_ref": 1.0, "upper_ref": 1.0, "display_vmax": 1.0}
    merged = np.concatenate(positives, axis=0).astype(np.float32)
    lower_ref = float(np.percentile(merged, 10))
    upper_ref = float(np.percentile(merged, 99))
    lower_ref = max(lower_ref, float(eps))
    upper_ref = max(upper_ref, lower_ref + float(eps))
    return {
        "lower_ref": lower_ref,
        "upper_ref": upper_ref,
        "display_vmax": float(np.log1p(upper_ref / lower_ref)),
    }


def _transform_positive_vector(values, lower_ref, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    transformed = np.zeros_like(values, dtype=np.float32)
    mask = values > 0
    if np.any(mask):
        transformed[mask] = np.log1p(values[mask] / max(float(lower_ref), float(eps))).astype(np.float32)
    return transformed.astype(np.float32)


def _build_local_case_scores(graphs):
    modal_a = graphs["modal_A"]
    modal_b = graphs["modal_B"]
    corr_a = graphs["corr_A"]
    corr_b = graphs["corr_B"]
    fused = graphs["fused"]
    score = (
        _sparse_abs_row_sum(modal_a - modal_b)
        + _sparse_abs_row_sum(corr_a - modal_a)
        + _sparse_abs_row_sum(corr_b - modal_b)
        + _sparse_abs_row_sum(fused - modal_a)
        + _sparse_abs_row_sum(fused - modal_b)
    )
    score = score + np.asarray(fused.sum(axis=1)).reshape(-1).astype(np.float32)
    return score.astype(np.float32)


def _select_local_case_nodes(graphs, num_cases):
    num_cases = max(0, int(num_cases))
    if num_cases <= 0:
        return np.empty(0, dtype=np.int64)
    scores = _build_local_case_scores(graphs)
    fused_degree = np.asarray(graphs["fused"].sum(axis=1)).reshape(-1).astype(np.float32)
    modal_support = (
        np.asarray(graphs["modal_A"].sum(axis=1)).reshape(-1).astype(np.float32)
        + np.asarray(graphs["modal_B"].sum(axis=1)).reshape(-1).astype(np.float32)
    )
    valid_primary = np.where((fused_degree > 0) & (modal_support > 0))[0]
    valid_secondary = np.where(modal_support > 0)[0]

    ordered = []
    if valid_primary.size > 0:
        ordered.extend(valid_primary[np.argsort(-scores[valid_primary], kind="stable")].tolist())
    if len(ordered) < num_cases and valid_secondary.size > 0:
        secondary = valid_secondary[np.argsort(-scores[valid_secondary], kind="stable")].tolist()
        for idx in secondary:
            if idx not in ordered:
                ordered.append(int(idx))
                if len(ordered) >= num_cases:
                    break
    if len(ordered) < num_cases:
        fallback = np.argsort(-scores, kind="stable").tolist()
        for idx in fallback:
            if idx not in ordered:
                ordered.append(int(idx))
                if len(ordered) >= num_cases:
                    break
    return np.asarray(ordered[:num_cases], dtype=np.int64)


def _topk_neighbors(graph, node_idx, topk):
    row = graph.getrow(int(node_idx))
    if row.nnz == 0:
        return np.empty(0, dtype=np.int64)
    values = row.data.astype(np.float32)
    indices = row.indices.astype(np.int64)
    if indices.size > int(topk):
        pos = np.argpartition(-values, int(topk) - 1)[: int(topk)]
        pos = pos[np.argsort(-values[pos], kind="stable")]
        indices = indices[pos]
    else:
        indices = indices[np.argsort(-values, kind="stable")]
    return indices.astype(np.int64)


def _save_local_case_figure(
    plt,
    LineCollection,
    case_idx,
    node_idx,
    graphs,
    coords_full,
    cluster_labels,
    output_path,
    topk,
    node_stats,
):
    graph_keys = ["modal_A", "modal_B", "corr_A", "corr_B", "fused"]
    case_nodes = {int(node_idx)}
    for key in graph_keys:
        case_nodes.update(int(x) for x in _topk_neighbors(graphs[key], node_idx, topk))
    case_nodes = np.asarray(sorted(case_nodes), dtype=np.int64)
    local_index = {int(node): idx for idx, node in enumerate(case_nodes.tolist())}
    coords = coords_full[case_nodes]
    colors = cluster_labels[case_nodes] if cluster_labels is not None else np.zeros(case_nodes.shape[0], dtype=np.int32)

    fig, axes = plt.subplots(1, len(graph_keys), figsize=(24, 5), dpi=160)
    for ax, key in zip(axes, graph_keys):
        subgraph = graphs[key][case_nodes][:, case_nodes].tocsr().astype(np.float32)
        segments, weights = _prepare_edge_segments(subgraph, coords, topk_edges=None)
        if segments.shape[0] > 0:
            alpha = np.clip(weights / max(np.percentile(weights, 95), 1e-8), 0.05, 1.0)
            lc = LineCollection(segments, colors=[(0.4, 0.4, 0.4, float(a) * 0.7) for a in alpha], linewidths=0.7)
            ax.add_collection(lc)
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=28, cmap="tab20", linewidths=0.0)
        center = coords[local_index[int(node_idx)]]
        ax.scatter(center[0], center[1], s=80, facecolors="none", edgecolors="black", linewidths=1.5)
        title = key
        if subgraph.nnz == 0:
            title = f"{key}\n(no local edges)"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    fig.suptitle(
        f"case {case_idx:04d} | node={int(node_idx)} | "
        f"degA={node_stats['degA']:.3f} degB={node_stats['degB']:.3f} "
        f"degCorrA={node_stats['degCorrA']:.3f} degCorrB={node_stats['degCorrB']:.3f} degFused={node_stats['degFused']:.3f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def visualize_cross_modal_topology_results(
    result_dir,
    visualization_output_dir=None,
    visualization_topk_edges=8,
    visualization_node_order_mode="fused_diffusion",
    visualization_layout_mode="fused_diffusion",
    visualization_num_local_cases=6,
    visualization_local_case_topk=12,
    visualization_max_heatmap_nodes=768,
    visualization_show_labels_if_available=False,
):
    plt, LineCollection = _load_matplotlib()
    bundle = load_cross_modal_result_bundle(result_dir)
    result_dir = Path(bundle["result_dir"])
    output_dir = Path(visualization_output_dir) if visualization_output_dir is not None else result_dir / "topology_viz"
    local_case_dir = output_dir / "local_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_case_dir.mkdir(parents=True, exist_ok=True)

    graphs = bundle["graphs"]
    order, cluster_labels = _compute_node_order(bundle, visualization_node_order_mode)
    if cluster_labels is None:
        cluster_labels = _compute_cluster_labels(_ensure_two_dimensional(_get_base_embedding(bundle)), random_state=0)
    display_indices = _select_display_indices(order, visualization_max_heatmap_nodes)
    coords_full = _compute_layout_coordinates(bundle, visualization_layout_mode)
    display_coords = coords_full[display_indices]

    labels, label_key = _extract_optional_labels(bundle["sample_meta"])
    if visualization_show_labels_if_available and labels is not None:
        unique_labels = {str(item): idx for idx, item in enumerate(sorted(set(labels.tolist())))}
        node_colors_full = np.asarray([unique_labels[str(item)] for item in labels.tolist()], dtype=np.int32)
        node_color_source = str(label_key)
        layout_node_color_mode = "label"
        layout_node_color_payload = None
        layout_node_vmax = None
        layout_node_colorbar_label = None
        layout_node_sizes = None
    else:
        node_colors_full = cluster_labels
        node_color_source = "graph_degree_shared_log"
        layout_node_color_mode = "graph_degree_shared_log"
        graph_stats_preview = {key: _graph_stats(graph) for key, graph in graphs.items()}
        degree_scale = _compute_positive_vector_scale([stats["degree"] for stats in graph_stats_preview.values()])
        layout_node_color_payload = {
            key: _transform_positive_vector(stats["degree"], degree_scale["lower_ref"])
            for key, stats in graph_stats_preview.items()
        }
        layout_node_vmax = degree_scale["display_vmax"]
        layout_node_colorbar_label = "log1p(weighted degree / q10)"
        layout_node_sizes = {
            key: 12.0 + 28.0 * np.clip(
                layout_node_color_payload[key] / max(layout_node_vmax, 1e-8),
                0.0,
                1.0,
            )
            for key in graphs.keys()
        }

    graph_matrices = {key: _gather_heatmap_matrix(graph, display_indices) for key, graph in graphs.items()}
    positive_heatmap_scale = _compute_positive_heatmap_scale(list(graph_matrices.values()))
    display_cluster_labels = cluster_labels[display_indices]
    block_graph_matrices = {}
    block_label_reference = None
    for key, matrix in graph_matrices.items():
        block_matrix, block_labels, _ = _aggregate_matrix_by_labels(matrix, display_cluster_labels)
        block_graph_matrices[key] = block_matrix
        if block_label_reference is None:
            block_label_reference = block_labels
    positive_block_scale = _compute_positive_heatmap_scale(list(block_graph_matrices.values()))

    for key, filename in {
        "modal_A": "heatmap_modal_A.png",
        "modal_B": "heatmap_modal_B.png",
        "corr_A": "heatmap_corr_A.png",
        "corr_B": "heatmap_corr_B.png",
        "fused": "heatmap_fused.png",
    }.items():
        _save_dual_heatmap_figure(
            plt,
            _transform_positive_heatmap(graph_matrices[key], positive_heatmap_scale["lower_ref"]),
            _transform_positive_heatmap(block_graph_matrices[key], positive_block_scale["lower_ref"]),
            output_dir / filename,
            title=f"Adjacency Heatmap: {bundle['modality_name_map'][key]} (node + cluster view)",
            cmap="magma",
            vmin=0.0,
            vmax=positive_block_scale["display_vmax"],
            colorbar_label="log1p(weight / q10)",
            block_labels=block_label_reference,
        )

    display_subgraphs = {key: graph[display_indices][:, display_indices].tocsr() for key, graph in graphs.items()}
    if layout_node_color_mode == "label":
        display_layout_node_colors = {key: node_colors_full[display_indices] for key in graphs.keys()}
        display_layout_node_sizes = {key: 12.0 for key in graphs.keys()}
    else:
        display_layout_node_colors = {
            key: layout_node_color_payload[key][display_indices] for key in graphs.keys()
        }
        display_layout_node_sizes = {
            key: layout_node_sizes[key][display_indices] for key in graphs.keys()
        }
    for key, filename in {
        "modal_A": "layout_modal_A.png",
        "modal_B": "layout_modal_B.png",
        "corr_A": "layout_corr_A.png",
        "corr_B": "layout_corr_B.png",
        "fused": "layout_fused.png",
    }.items():
        _save_layout_figure(
            plt,
            LineCollection,
            display_subgraphs[key],
            display_coords,
            display_layout_node_colors[key],
            output_dir / filename,
            title=f"Shared Layout: {bundle['modality_name_map'][key]}",
            topk_edges=visualization_topk_edges,
            colorbar=True,
            cmap="viridis" if layout_node_color_mode != "label" else "tab20",
            node_vmin=0.0 if layout_node_color_mode != "label" else None,
            node_vmax=layout_node_vmax if layout_node_color_mode != "label" else None,
            colorbar_label=layout_node_colorbar_label if layout_node_color_mode != "label" else node_color_source,
            node_sizes=display_layout_node_sizes[key],
        )

    delta_matrices = {
        "delta_corr_A": graph_matrices["corr_A"] - graph_matrices["modal_A"],
        "delta_corr_B": graph_matrices["corr_B"] - graph_matrices["modal_B"],
        "delta_fused_vs_A": graph_matrices["fused"] - graph_matrices["modal_A"],
        "delta_fused_vs_B": graph_matrices["fused"] - graph_matrices["modal_B"],
    }
    block_delta_matrices = {}
    for key, matrix in delta_matrices.items():
        block_matrix, _, _ = _aggregate_matrix_by_labels(matrix, display_cluster_labels)
        block_delta_matrices[key] = block_matrix
    signed_heatmap_scale = _compute_signed_heatmap_scale(list(delta_matrices.values()))
    signed_block_scale = _compute_signed_heatmap_scale(list(block_delta_matrices.values()))

    for key, filename, title in [
        ("delta_corr_A", "delta_corr_A.png", "Delta Corr A = B_hat^A - B^A"),
        ("delta_corr_B", "delta_corr_B.png", "Delta Corr B = B_hat^B - B^B"),
        ("delta_fused_vs_A", "delta_fused_vs_A.png", "Delta Fused vs A = B* - B^A"),
        ("delta_fused_vs_B", "delta_fused_vs_B.png", "Delta Fused vs B = B* - B^B"),
    ]:
        _save_dual_heatmap_figure(
            plt,
            _transform_signed_heatmap(delta_matrices[key], signed_heatmap_scale["lower_ref"]),
            _transform_signed_heatmap(block_delta_matrices[key], signed_block_scale["lower_ref"]),
            output_dir / filename,
            title=f"{title} (node + cluster view)",
            cmap="coolwarm",
            vmin=-signed_block_scale["display_vmax"],
            vmax=signed_block_scale["display_vmax"],
            colorbar_label="sign(x) * log1p(|x| / q10)",
            block_labels=block_label_reference,
        )

    graph_stats = graph_stats_preview if layout_node_color_mode != "label" else {key: _graph_stats(graph) for key, graph in graphs.items()}
    _save_distribution_plot(
        plt,
        {key: stats["degree"] for key, stats in graph_stats.items()},
        output_dir / "stats_degree_distribution.png",
        title="Degree Distribution Comparison",
        xlabel="Weighted Degree",
    )
    _save_distribution_plot(
        plt,
        {key: stats["edge_weights"] for key, stats in graph_stats.items()},
        output_dir / "stats_edge_weight_distribution.png",
        title="Edge Weight Distribution Comparison",
        xlabel="Edge Weight",
    )
    _save_graph_overview_plot(
        plt,
        graph_stats,
        output_dir / "stats_graph_overview.png",
    )

    local_case_scores = _build_local_case_scores(graphs)
    case_nodes = _select_local_case_nodes(graphs, visualization_num_local_cases)
    local_case_records = []
    degA = graph_stats["modal_A"]["degree"]
    degB = graph_stats["modal_B"]["degree"]
    degCorrA = graph_stats["corr_A"]["degree"]
    degCorrB = graph_stats["corr_B"]["degree"]
    degFused = graph_stats["fused"]["degree"]
    for case_rank, node_idx in enumerate(case_nodes.tolist(), start=1):
        node_stats = {
            "degA": float(degA[node_idx]),
            "degB": float(degB[node_idx]),
            "degCorrA": float(degCorrA[node_idx]),
            "degCorrB": float(degCorrB[node_idx]),
            "degFused": float(degFused[node_idx]),
        }
        filename = local_case_dir / f"case_node_{case_rank:04d}.png"
        _save_local_case_figure(
            plt,
            LineCollection,
            case_rank,
            int(node_idx),
            graphs,
            coords_full,
            node_colors_full,
            filename,
            topk=visualization_local_case_topk,
            node_stats=node_stats,
        )
        record = {
            "case_rank": int(case_rank),
            "node_idx": int(node_idx),
            "score": float(local_case_scores[node_idx]),
            **node_stats,
            "path": str(filename),
        }
        local_case_records.append(record)

    diff_summary = {
        "delta_corr_A": _diff_stats(graphs["corr_A"], graphs["modal_A"]),
        "delta_corr_B": _diff_stats(graphs["corr_B"], graphs["modal_B"]),
        "delta_fused_vs_A": _diff_stats(graphs["fused"], graphs["modal_A"]),
        "delta_fused_vs_B": _diff_stats(graphs["fused"], graphs["modal_B"]),
    }

    summary_payload = {
        "result_dir": str(result_dir),
        "visualization_output_dir": str(output_dir),
        "modality_name_map": bundle["modality_name_map"],
        "graph_dirs": bundle["graph_dirs"],
        "config": {
            "visualization_topk_edges": int(visualization_topk_edges),
            "visualization_node_order_mode": str(visualization_node_order_mode),
            "visualization_layout_mode": str(visualization_layout_mode),
            "visualization_num_local_cases": int(visualization_num_local_cases),
            "visualization_local_case_topk": int(visualization_local_case_topk),
            "visualization_max_heatmap_nodes": int(visualization_max_heatmap_nodes),
            "visualization_show_labels_if_available": bool(visualization_show_labels_if_available),
        },
        "node_order": {
            "mode": str(visualization_node_order_mode),
            "display_node_count": int(display_indices.shape[0]),
        },
        "layout": {
            "mode": str(visualization_layout_mode),
            "node_color_source": node_color_source,
            "node_color_mode": layout_node_color_mode,
        },
        "heatmap_transform": {
            "adjacency": {
                "type": "shared_log_nonzero_masked",
                "lower_ref_q10": float(positive_heatmap_scale["lower_ref"]),
                "upper_ref_q99": float(positive_heatmap_scale["upper_ref"]),
            },
            "adjacency_block": {
                "type": "cluster_aggregated_row_normalized_log_nonzero_masked",
                "lower_ref_q10": float(positive_block_scale["lower_ref"]),
                "upper_ref_q99": float(positive_block_scale["upper_ref"]),
                "num_clusters": int(len(block_label_reference)) if block_label_reference is not None else 0,
            },
            "delta": {
                "type": "shared_signed_log_nonzero_masked",
                "lower_ref_q10_abs": float(signed_heatmap_scale["lower_ref"]),
                "upper_ref_q99_abs": float(signed_heatmap_scale["upper_ref"]),
            },
            "delta_block": {
                "type": "cluster_aggregated_row_normalized_signed_log_nonzero_masked",
                "lower_ref_q10_abs": float(signed_block_scale["lower_ref"]),
                "upper_ref_q99_abs": float(signed_block_scale["upper_ref"]),
            },
        },
        "graph_summary": {
            key: {
                "num_nodes": int(stats["num_nodes"]),
                "nnz": int(stats["nnz"]),
                "num_components": int(stats["num_components"]),
                "density": float(stats["density"]),
                "avg_degree": float(stats["avg_degree"]),
                "max_degree": float(stats["max_degree"]),
                "mean_edge_weight": float(stats["mean_edge_weight"]),
                "clustering_mean": None if stats["clustering_mean"] is None else float(stats["clustering_mean"]),
            }
            for key, stats in graph_stats.items()
        },
        "delta_summary": diff_summary,
        "local_cases": local_case_records,
    }
    summary_path = output_dir / "graph_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    return {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "local_case_dir": str(local_case_dir),
    }
