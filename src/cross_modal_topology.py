import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from src.graph_wavelet import (
    build_multi_scale_wavelet_signatures,
    parse_wavelet_scales,
    sparsify_sparse_matrix,
)
from src.topology_graph import (
    build_laplacian,
    build_spectral_embedding,
    compute_collapse_metrics,
    compute_spectrum,
    parse_multi_scale_ks,
)


def log_cross_modal(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[cross-modal][{timestamp}] {message}", flush=True)


def sanitize_name(name):
    return name.replace("\\", "-").replace("/", "-").replace(" ", "_")


def get_modality_metric(args, modality):
    if modality == "image":
        return getattr(args, "image_metric", None) or args.metric
    if modality == "text":
        return getattr(args, "text_metric", None) or args.metric
    raise ValueError(f"Unsupported modality: {modality}")


def build_graph_dir(args, modality):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    modality_metric = get_modality_metric(args, modality)
    k_list = parse_multi_scale_ks(args.k, getattr(args, "multi_scale_ks", None))
    if len(k_list) == 1:
        graph_tag = f"k{k_list[0]}_{sanitize_name(modality_metric)}"
    else:
        graph_tag = f"ks{'-'.join(str(item) for item in k_list)}_{sanitize_name(modality_metric)}"
    return Path(args.topology_root) / args.dataset / args.split / model_tag / modality / graph_tag


def build_output_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    fusion_tag = f"k{args.k}_{sanitize_name(args.metric)}_a{sanitize_name(str(args.alpha))}"
    return Path(args.output_root) / args.dataset / args.split / model_tag / fusion_tag


def load_graph_bundle(graph_dir):
    graph_dir = Path(graph_dir)
    log_cross_modal(f"loading graph bundle from {graph_dir}")
    with open(graph_dir / "summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    graph = sparse.load_npz(graph_dir / "symmetric_graph.npz").tocsr()
    transition = sparse.load_npz(graph_dir / "transition_graph.npz").tocsr()
    sample_meta = json.load(open(graph_dir / "sample_meta.json", "r", encoding="utf-8"))
    return {
        "dir": str(graph_dir),
        "summary": summary,
        "graph": graph,
        "transition": transition,
        "sample_meta": sample_meta,
    }


def validate_modalities(image_bundle, text_bundle):
    image_graph = image_bundle["graph"]
    text_graph = text_bundle["graph"]
    if image_graph.shape != text_graph.shape:
        raise ValueError(f"Image/text graph shape mismatch: {image_graph.shape} vs {text_graph.shape}")
    if len(image_bundle["sample_meta"]) != len(text_bundle["sample_meta"]):
        raise ValueError("Image/text sample_meta length mismatch.")

    image_indices = [item["sample_idx"] for item in image_bundle["sample_meta"]]
    text_indices = [item["sample_idx"] for item in text_bundle["sample_meta"]]
    if image_indices != text_indices:
        raise ValueError("Image/text sample_idx ordering mismatch.")


def choose_healthy_modality(image_summary, text_summary, prefer=None):
    if prefer is not None:
        prefer = prefer.lower()
        if prefer not in {"image", "text"}:
            raise ValueError(f"Invalid preferred modality: {prefer}")
        return prefer

    # Lower collapse_score => healthier. If tied, prefer higher spectral entropy.
    image_score = float(image_summary["collapse_score"])
    text_score = float(text_summary["collapse_score"])
    if image_score < text_score:
        return "image"
    if text_score < image_score:
        return "text"

    image_entropy = float(image_summary["spectral_entropy"])
    text_entropy = float(text_summary["spectral_entropy"])
    if image_entropy >= text_entropy:
        return "image"
    return "text"


def sparse_elementwise_power(matrix, alpha):
    matrix = matrix.tocsr(copy=True)
    matrix.data = np.power(np.clip(matrix.data, 1e-12, None), alpha).astype(np.float32)
    return matrix


def fuzzy_union_symmetrize(graph):
    transpose = graph.transpose().tocsr()
    sym = graph + transpose - graph.multiply(transpose)
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


def compute_global_confidences(image_score, text_score, tau_g=0.5):
    tau_g = max(float(tau_g), 1e-8)
    logits = np.array([-float(image_score) / tau_g, -float(text_score) / tau_g], dtype=np.float64)
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights), 1e-12)
    return float(weights[0]), float(weights[1])


def scale_sparse_graph(graph, scalar):
    graph = graph.tocsr(copy=True)
    graph.data = (graph.data.astype(np.float32) * float(scalar)).astype(np.float32)
    return graph


def compute_entropy_confidence_view(graph, eps=1e-8):
    graph = graph.tocsr().astype(np.float32)
    num_nodes = int(graph.shape[0])
    entropy_score = np.zeros(num_nodes, dtype=np.float32)

    indptr = graph.indptr
    data = graph.data
    eps = float(eps)
    for node_idx in range(num_nodes):
        start = indptr[node_idx]
        end = indptr[node_idx + 1]
        num_neighbors = int(end - start)
        if num_neighbors <= 1:
            continue
        p = np.clip(data[start:end].astype(np.float64), eps, None)
        entropy = -np.sum(p * np.log(p + eps))
        entropy = entropy / max(np.log(float(num_neighbors) + eps), eps)
        entropy_score[node_idx] = np.float32(np.clip(entropy, 0.0, 1.0))
    return entropy_score.astype(np.float32)


def entropy_view_to_kappa(entropy_score, tau_l=0.25, kappa_min=0.05, eps=1e-8):
    tau_l = max(float(tau_l), float(eps))
    kappa_min = float(kappa_min)
    chi = 1.0 - np.asarray(entropy_score, dtype=np.float32)
    kappa = kappa_min + (1.0 - kappa_min) * np.exp(-chi / tau_l)
    return np.clip(kappa.astype(np.float32), kappa_min, 1.0)


def extract_row_topk_neighbors(graph, topk=15):
    graph = graph.tocsr().astype(np.float32)
    indptr = graph.indptr
    indices = graph.indices
    data = graph.data
    neighbors = []
    topk = max(1, int(topk))
    for node_idx in range(int(graph.shape[0])):
        start = indptr[node_idx]
        end = indptr[node_idx + 1]
        row_indices = indices[start:end]
        row_values = data[start:end]
        if row_indices.size == 0:
            neighbors.append(np.empty(0, dtype=np.int64))
            continue
        if row_indices.size > topk:
            top_positions = np.argpartition(-row_values, topk - 1)[:topk]
            top_positions = top_positions[np.argsort(-row_values[top_positions], kind="stable")]
            chosen = row_indices[top_positions]
        else:
            chosen = row_indices[np.argsort(-row_values, kind="stable")]
        neighbors.append(chosen.astype(np.int64))
    return neighbors


def compute_cross_modal_agreement_view(graph, other_graph, topk=15, agreement_type="jaccard"):
    agreement_type = str(agreement_type or "jaccard")
    if agreement_type != "jaccard":
        raise ValueError(f"Unsupported local agreement type: {agreement_type}")
    neighbors_a = extract_row_topk_neighbors(graph, topk=topk)
    neighbors_b = extract_row_topk_neighbors(other_graph, topk=topk)
    score = np.zeros(int(graph.shape[0]), dtype=np.float32)
    for node_idx, (row_a, row_b) in enumerate(zip(neighbors_a, neighbors_b)):
        if row_a.size == 0 and row_b.size == 0:
            score[node_idx] = 0.0
            continue
        set_a = set(int(x) for x in row_a.tolist())
        set_b = set(int(x) for x in row_b.tolist())
        union = len(set_a | set_b)
        if union <= 0:
            score[node_idx] = 0.0
            continue
        score[node_idx] = np.float32(len(set_a & set_b) / float(union))
    return score.astype(np.float32)


def compute_diffusion_stability_view(graph, diffusion_hops=2, diffusion_type="p_vs_p2_cosine", eps=1e-8):
    diffusion_type = str(diffusion_type or "p_vs_p2_cosine")
    if diffusion_type != "p_vs_p2_cosine":
        raise ValueError(f"Unsupported local diffusion stability type: {diffusion_type}")
    graph = graph.tocsr().astype(np.float32)
    if int(diffusion_hops) <= 1:
        graph_h = graph
    else:
        graph_h = graph.copy()
        for _ in range(int(diffusion_hops) - 1):
            graph_h = (graph_h @ graph).tocsr()
        graph_h.eliminate_zeros()
    num_nodes = int(graph.shape[0])
    score = np.zeros(num_nodes, dtype=np.float32)
    eps = float(eps)
    for node_idx in range(num_nodes):
        row_p = graph.getrow(node_idx)
        row_h = graph_h.getrow(node_idx)
        norm_p = float(np.sqrt(row_p.multiply(row_p).sum()))
        norm_h = float(np.sqrt(row_h.multiply(row_h).sum()))
        if norm_p <= eps or norm_h <= eps:
            score[node_idx] = 0.0
            continue
        dot = float(row_p.multiply(row_h).sum())
        score[node_idx] = np.float32(np.clip(dot / max(norm_p * norm_h, eps), 0.0, 1.0))
    return score.astype(np.float32)


def compute_local_node_confidence(
    graph,
    other_graph=None,
    mode="entropy",
    eps=1e-8,
    tau_l=0.25,
    kappa_min=0.05,
    local_conf_weight_entropy=1.0,
    local_conf_weight_agreement=1.0,
    local_conf_weight_diffusion=1.0,
    local_conf_agreement_topk=15,
    local_conf_agreement_type="jaccard",
    local_conf_diffusion_hops=2,
    local_conf_diffusion_type="p_vs_p2_cosine",
):
    mode = str(mode or "entropy")
    if mode == "none":
        num_nodes = int(graph.shape[0])
        diagnostics = {
            "mode": "none",
            "entropy_view_stats": summarize_vector(np.zeros(num_nodes, dtype=np.float32)),
            "agreement_view_stats": summarize_vector(np.zeros(num_nodes, dtype=np.float32)),
            "diffusion_view_stats": summarize_vector(np.zeros(num_nodes, dtype=np.float32)),
        }
        return None, diagnostics

    entropy_score = compute_entropy_confidence_view(graph, eps=eps)
    if mode == "entropy":
        kappa = entropy_view_to_kappa(
            entropy_score,
            tau_l=tau_l,
            kappa_min=kappa_min,
            eps=eps,
        )
        diagnostics = {
            "mode": "entropy",
            "entropy_view_stats": summarize_vector(entropy_score),
            "agreement_view_stats": summarize_vector(np.zeros_like(entropy_score)),
            "diffusion_view_stats": summarize_vector(np.zeros_like(entropy_score)),
        }
        return kappa.astype(np.float32), diagnostics

    if mode != "multi_view":
        raise ValueError(f"Unsupported local node confidence mode: {mode}")
    if other_graph is None:
        raise ValueError("multi_view local node confidence requires the opposite-modality graph.")

    agreement_score = compute_cross_modal_agreement_view(
        graph,
        other_graph,
        topk=local_conf_agreement_topk,
        agreement_type=local_conf_agreement_type,
    )
    diffusion_score = compute_diffusion_stability_view(
        graph,
        diffusion_hops=local_conf_diffusion_hops,
        diffusion_type=local_conf_diffusion_type,
        eps=eps,
    )

    w_entropy = float(local_conf_weight_entropy)
    w_agreement = float(local_conf_weight_agreement)
    w_diffusion = float(local_conf_weight_diffusion)
    weight_sum = max(w_entropy + w_agreement + w_diffusion, float(eps))
    # Multi-view local reliability: weighted average of three normalized views,
    # then linearly squashed into [kappa_min, 1] for reuse in edge confidence.
    combined_score = (
        w_entropy * entropy_score
        + w_agreement * agreement_score
        + w_diffusion * diffusion_score
    ) / weight_sum
    combined_score = np.clip(combined_score.astype(np.float32), 0.0, 1.0)
    kappa = float(kappa_min) + (1.0 - float(kappa_min)) * combined_score
    kappa = np.clip(kappa.astype(np.float32), float(kappa_min), 1.0)
    diagnostics = {
        "mode": "multi_view",
        "entropy_view_stats": summarize_vector(entropy_score),
        "agreement_view_stats": summarize_vector(agreement_score),
        "diffusion_view_stats": summarize_vector(diffusion_score),
        "combined_view_stats": summarize_vector(combined_score),
        "weights": {
            "entropy": float(w_entropy),
            "agreement": float(w_agreement),
            "diffusion": float(w_diffusion),
        },
        "agreement_topk": int(local_conf_agreement_topk),
        "agreement_type": str(local_conf_agreement_type),
        "diffusion_hops": int(local_conf_diffusion_hops),
        "diffusion_type": str(local_conf_diffusion_type),
    }
    return kappa.astype(np.float32), diagnostics


def summarize_vector(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def sanitize_real_spectrum(eigenvalues, eigenvectors, label):
    eigenvalues = np.asarray(eigenvalues)
    max_value_imag = float(np.max(np.abs(np.imag(eigenvalues)))) if eigenvalues.size > 0 else 0.0
    max_vector_imag = 0.0
    if eigenvectors is not None:
        eigenvectors = np.asarray(eigenvectors)
        if eigenvectors.size > 0:
            max_vector_imag = float(np.max(np.abs(np.imag(eigenvectors))))
    max_imag = max(max_value_imag, max_vector_imag)
    if max_imag > 1e-5:
        log_cross_modal(f"{label}: dropping small imaginary component from eigendecomposition (max_imag={max_imag:.3e})")
    real_values = np.real(eigenvalues).astype(np.float32)
    real_vectors = np.real(eigenvectors).astype(np.float32) if eigenvectors is not None else None
    return real_values, real_vectors


def compute_diffusion_spectrum(transition, num_eigs, return_eigenvectors=False, solver_mode="auto"):
    num_nodes = int(transition.shape[0])
    if num_nodes <= 1:
        eigenvalues = np.ones(1, dtype=np.float32)
        eigenvectors = np.ones((1, 1), dtype=np.float32) if return_eigenvectors else None
        return eigenvalues, eigenvectors

    num_eigs = max(2, min(int(num_eigs), num_nodes - 1))
    solver_mode = str(solver_mode or "auto")
    if solver_mode not in {"auto", "dense", "sparse"}:
        raise ValueError(f"Unsupported diffusion eig solver: {solver_mode}")

    use_dense = solver_mode == "dense" or (solver_mode == "auto" and num_nodes <= 4096)
    if use_dense:
        log_cross_modal("using dense eig on diffusion transition matrix")
        dense_matrix = transition.toarray()
        eigenvalues, eigenvectors = np.linalg.eig(dense_matrix)
    else:
        log_cross_modal("using sparse eigs on diffusion transition matrix with which=LR")
        try:
            eig_result = eigs(
                transition,
                k=num_eigs,
                which="LR",
                return_eigenvectors=return_eigenvectors,
            )
        except ArpackNoConvergence as exc:
            log_cross_modal("ARPACK did not fully converge for diffusion spectrum, retrying with relaxed settings")
            retry_ncv = min(num_nodes - 1, max(2 * num_eigs + 1, 32))
            eig_result = eigs(
                transition,
                k=num_eigs,
                which="LR",
                return_eigenvectors=return_eigenvectors,
                ncv=retry_ncv,
                maxiter=200000,
                tol=1e-4,
            )
        if return_eigenvectors:
            eigenvalues, eigenvectors = eig_result
        else:
            eigenvalues = eig_result
            eigenvectors = None

    eigenvalues, eigenvectors = sanitize_real_spectrum(
        eigenvalues,
        eigenvectors if return_eigenvectors else None,
        label="diffusion spectrum",
    )
    order = np.argsort(eigenvalues)[::-1]
    order = order[:num_eigs]
    eigenvalues = eigenvalues[order].astype(np.float32)
    eigenvectors = eigenvectors[:, order].astype(np.float32) if return_eigenvectors and eigenvectors is not None else None
    return eigenvalues, eigenvectors


def build_diffusion_embedding(eigenvalues, eigenvectors, embedding_dim=None, diffusion_time=1.0, drop_first=True):
    if eigenvectors is None:
        return None, np.zeros(0, dtype=np.float32)
    start = 1 if drop_first and eigenvectors.shape[1] > 1 else 0
    if embedding_dim is None:
        end = eigenvectors.shape[1]
    else:
        end = min(eigenvectors.shape[1], start + int(embedding_dim))
    basis = eigenvectors[:, start:end].astype(np.float32)
    used_eigenvalues = eigenvalues[start:end].astype(np.float32)
    if basis.size == 0:
        fallback_end = min(eigenvectors.shape[1], max(1, int(embedding_dim or 1)))
        basis = eigenvectors[:, :fallback_end].astype(np.float32)
        used_eigenvalues = eigenvalues[:fallback_end].astype(np.float32)
    diffusion_time = float(diffusion_time)
    abs_values = np.abs(used_eigenvalues).astype(np.float32)
    lambda_power = np.power(abs_values, diffusion_time).astype(np.float32)
    rounded_time = round(diffusion_time)
    if np.isclose(diffusion_time, rounded_time):
        lambda_power = lambda_power * np.power(np.sign(used_eigenvalues), int(rounded_time)).astype(np.float32)
    elif np.any(used_eigenvalues < 0):
        log_cross_modal("diffusion_time is non-integer; using |lambda|^t for negative diffusion eigenvalues")
    embedding = basis * lambda_power.reshape(1, -1)
    return embedding.astype(np.float32), lambda_power.astype(np.float32)


def scale_sparse_graph_with_node_confidence(graph, scalar, node_confidence=None):
    graph = graph.tocoo(copy=True)
    edge_scale = np.full(graph.data.shape[0], float(scalar), dtype=np.float32)
    if node_confidence is not None:
        node_confidence = np.asarray(node_confidence, dtype=np.float32)
        edge_scale = edge_scale * np.sqrt(node_confidence[graph.row] * node_confidence[graph.col]).astype(np.float32)
    graph.data = (graph.data.astype(np.float32) * edge_scale).astype(np.float32)
    scaled = graph.tocsr()
    scaled.eliminate_zeros()
    return scaled


def build_hard_directional_gate(source_confidence, target_confidence, tau_high=0.6, tau_low=0.3, tau_gap=0.15):
    source_confidence = np.asarray(source_confidence, dtype=np.float32)
    target_confidence = np.asarray(target_confidence, dtype=np.float32)
    gate = (
        (source_confidence > float(tau_high))
        & (target_confidence < float(tau_low))
        & ((source_confidence - target_confidence) > float(tau_gap))
    )
    return gate.astype(np.float32)


def sparse_matrix_from_union_keys(union_keys, values, shape):
    values = np.asarray(values, dtype=np.float32)
    nonzero = values > 0
    if not np.any(nonzero):
        return sparse.csr_matrix(shape, dtype=np.float32)
    num_cols = int(shape[1])
    keys = union_keys[nonzero]
    rows = (keys // num_cols).astype(np.int32)
    cols = (keys % num_cols).astype(np.int32)
    matrix = sparse.csr_matrix((values[nonzero].astype(np.float32), (rows, cols)), shape=shape)
    matrix.eliminate_zeros()
    return matrix


def build_topk_neighbor_average_graph(graph, topk=10):
    neighbors = extract_row_topk_neighbors(graph, topk=topk)
    total_nnz = int(sum(int(row.size) for row in neighbors))
    if total_nnz <= 0:
        return sparse.csr_matrix(graph.shape, dtype=np.float32)

    rows = np.empty(total_nnz, dtype=np.int32)
    cols = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=np.float32)
    cursor = 0
    for node_idx, row_neighbors in enumerate(neighbors):
        count = int(row_neighbors.size)
        if count <= 0:
            continue
        rows[cursor : cursor + count] = int(node_idx)
        cols[cursor : cursor + count] = row_neighbors.astype(np.int32, copy=False)
        data[cursor : cursor + count] = np.float32(1.0 / max(count, 1))
        cursor += count

    avg_graph = sparse.csr_matrix((data, (rows, cols)), shape=graph.shape, dtype=np.float32)
    avg_graph.eliminate_zeros()
    return avg_graph


def robust_unit_scale(values, eps=1e-8, upper_quantile=95.0):
    values = np.asarray(values, dtype=np.float32)
    positive = values[values > 0]
    if positive.size == 0:
        scaled = np.zeros_like(values, dtype=np.float32)
        diagnostics = {
            "raw_stats": summarize_vector(values),
            "normalized_stats": summarize_vector(scaled),
            "normalization": {
                "type": "robust_quantile_clip",
                "upper_quantile": float(upper_quantile),
                "scale": 1.0,
            },
        }
        return scaled, diagnostics

    scale = float(np.percentile(positive.astype(np.float64), float(upper_quantile)))
    if scale <= float(eps):
        scale = float(np.max(positive))
    scale = max(scale, float(eps))
    scaled = np.clip(values / scale, 0.0, 1.0).astype(np.float32)
    diagnostics = {
        "raw_stats": summarize_vector(values),
        "normalized_stats": summarize_vector(scaled),
        "normalization": {
            "type": "robust_quantile_clip",
            "upper_quantile": float(upper_quantile),
            "scale": float(scale),
        },
    }
    return scaled, diagnostics


def build_edge_collapse_scores(
    transition,
    union_keys,
    collapse_score_mode="edge_plus_neighborhood",
    collapse_neighbor_topk=10,
    collapse_score_weight_edge=1.0,
    collapse_score_weight_a2b=1.0,
    collapse_score_weight_b2a=1.0,
    collapse_score_weight_nbr2nbr=1.0,
    eps=1e-8,
):
    transition = transition.tocsr().astype(np.float32)
    collapse_score_mode = str(collapse_score_mode or "edge_plus_neighborhood")
    if collapse_score_mode not in {"edge_only", "edge_plus_neighborhood"}:
        raise ValueError(f"Unsupported collapse score mode: {collapse_score_mode}")

    edge_raw = gather_sparse_data_on_keys(transition, union_keys)
    zero = np.zeros_like(edge_raw, dtype=np.float32)
    a2b_raw = zero
    b2a_raw = zero
    nbr2nbr_raw = zero

    if collapse_score_mode == "edge_plus_neighborhood":
        neighbor_average = build_topk_neighbor_average_graph(
            transition,
            topk=max(1, int(collapse_neighbor_topk)),
        )
        neighbor_to_node = (neighbor_average @ transition).tocsr()
        neighbor_to_node.eliminate_zeros()
        neighbor_to_neighbor = (neighbor_to_node @ neighbor_average.transpose()).tocsr()
        neighbor_to_neighbor.eliminate_zeros()

        a2b_raw = gather_sparse_data_on_keys(neighbor_to_node, union_keys)
        b2a_raw = gather_sparse_data_on_keys(neighbor_to_node.transpose().tocsr(), union_keys)
        nbr2nbr_raw = gather_sparse_data_on_keys(neighbor_to_neighbor, union_keys)

    edge_norm, edge_diag = robust_unit_scale(edge_raw, eps=eps)
    a2b_norm, a2b_diag = robust_unit_scale(a2b_raw, eps=eps)
    b2a_norm, b2a_diag = robust_unit_scale(b2a_raw, eps=eps)
    nbr2nbr_norm, nbr2nbr_diag = robust_unit_scale(nbr2nbr_raw, eps=eps)

    q = (
        float(collapse_score_weight_edge) * edge_norm
        + float(collapse_score_weight_a2b) * a2b_norm
        + float(collapse_score_weight_b2a) * b2a_norm
        + float(collapse_score_weight_nbr2nbr) * nbr2nbr_norm
    ).astype(np.float32)

    diagnostics = {
        "collapse_score_mode": collapse_score_mode,
        "collapse_neighbor_topk": int(collapse_neighbor_topk),
        "collapse_score_input": "transition",
        "collapse_score_weights": {
            "edge": float(collapse_score_weight_edge),
            "a2b": float(collapse_score_weight_a2b),
            "b2a": float(collapse_score_weight_b2a),
            "nbr2nbr": float(collapse_score_weight_nbr2nbr),
        },
        "components": {
            "edge": edge_diag,
            "a2b": a2b_diag,
            "b2a": b2a_diag,
            "nbr2nbr": nbr2nbr_diag,
        },
        "q_stats": summarize_vector(q),
    }
    return q, diagnostics


def build_confidence_based_bidirectional_correction_weights(
    image_transition,
    text_transition,
    rho_img,
    rho_txt,
    eps=1e-8,
    enable_local_node_confidence=False,
    local_node_confidence_mode="multi_view",
    tau_l=0.25,
    kappa_min=0.05,
    local_conf_eps=1e-8,
    local_conf_weight_entropy=1.0,
    local_conf_weight_agreement=1.0,
    local_conf_weight_diffusion=1.0,
    local_conf_agreement_topk=15,
    local_conf_agreement_type="jaccard",
    local_conf_diffusion_hops=2,
    local_conf_diffusion_type="p_vs_p2_cosine",
    enable_directional_correction_gate=True,
    correction_gate_tau_high=0.6,
    correction_gate_tau_low=0.3,
    correction_gate_tau_gap=0.15,
):
    requested_local_mode = str(local_node_confidence_mode or "multi_view")
    effective_local_mode = requested_local_mode
    if bool(enable_local_node_confidence) and requested_local_mode == "none":
        # Legacy compatibility: old scripts may only pass the boolean switch.
        effective_local_mode = "entropy"

    image_kappa = None
    text_kappa = None
    image_local_conf_diagnostics = {
        "mode": effective_local_mode,
        "entropy_view_stats": summarize_vector(np.zeros(image_transition.shape[0], dtype=np.float32)),
        "agreement_view_stats": summarize_vector(np.zeros(image_transition.shape[0], dtype=np.float32)),
        "diffusion_view_stats": summarize_vector(np.zeros(image_transition.shape[0], dtype=np.float32)),
    }
    text_local_conf_diagnostics = {
        "mode": effective_local_mode,
        "entropy_view_stats": summarize_vector(np.zeros(text_transition.shape[0], dtype=np.float32)),
        "agreement_view_stats": summarize_vector(np.zeros(text_transition.shape[0], dtype=np.float32)),
        "diffusion_view_stats": summarize_vector(np.zeros(text_transition.shape[0], dtype=np.float32)),
    }
    if effective_local_mode != "none":
        image_kappa, image_local_conf_diagnostics = compute_local_node_confidence(
            image_transition,
            other_graph=text_transition,
            mode=effective_local_mode,
            eps=local_conf_eps,
            tau_l=tau_l,
            kappa_min=kappa_min,
            local_conf_weight_entropy=local_conf_weight_entropy,
            local_conf_weight_agreement=local_conf_weight_agreement,
            local_conf_weight_diffusion=local_conf_weight_diffusion,
            local_conf_agreement_topk=local_conf_agreement_topk,
            local_conf_agreement_type=local_conf_agreement_type,
            local_conf_diffusion_hops=local_conf_diffusion_hops,
            local_conf_diffusion_type=local_conf_diffusion_type,
        )
        text_kappa, text_local_conf_diagnostics = compute_local_node_confidence(
            text_transition,
            other_graph=image_transition,
            mode=effective_local_mode,
            eps=local_conf_eps,
            tau_l=tau_l,
            kappa_min=kappa_min,
            local_conf_weight_entropy=local_conf_weight_entropy,
            local_conf_weight_agreement=local_conf_weight_agreement,
            local_conf_weight_diffusion=local_conf_weight_diffusion,
            local_conf_agreement_topk=local_conf_agreement_topk,
            local_conf_agreement_type=local_conf_agreement_type,
            local_conf_diffusion_hops=local_conf_diffusion_hops,
            local_conf_diffusion_type=local_conf_diffusion_type,
        )

    c_img = scale_sparse_graph_with_node_confidence(image_transition, rho_img, node_confidence=image_kappa)
    c_txt = scale_sparse_graph_with_node_confidence(text_transition, rho_txt, node_confidence=text_kappa)
    overlap = c_img.multiply(c_txt)
    union_keys, _, _ = build_union_keys(c_img, c_txt)
    c_img_data = gather_sparse_data_on_keys(c_img, union_keys)
    c_txt_data = gather_sparse_data_on_keys(c_txt, union_keys)
    overlap_data = gather_sparse_data_on_keys(overlap, union_keys)

    # Preserve the original alpha formulation and only add a directional gate on top.
    numer_txt_to_img = np.maximum(c_txt_data - overlap_data, 0.0).astype(np.float32)
    numer_img_to_txt = np.maximum(c_img_data - overlap_data, 0.0).astype(np.float32)
    alpha_txt_to_img_data = (numer_txt_to_img / (numer_txt_to_img + float(eps))).astype(np.float32)
    alpha_img_to_txt_data = (numer_img_to_txt / (numer_img_to_txt + float(eps))).astype(np.float32)

    if bool(enable_directional_correction_gate):
        gate_txt_to_img = build_hard_directional_gate(
            c_txt_data,
            c_img_data,
            tau_high=correction_gate_tau_high,
            tau_low=correction_gate_tau_low,
            tau_gap=correction_gate_tau_gap,
        )
        gate_img_to_txt = build_hard_directional_gate(
            c_img_data,
            c_txt_data,
            tau_high=correction_gate_tau_high,
            tau_low=correction_gate_tau_low,
            tau_gap=correction_gate_tau_gap,
        )
    else:
        gate_txt_to_img = np.ones_like(alpha_txt_to_img_data, dtype=np.float32)
        gate_img_to_txt = np.ones_like(alpha_img_to_txt_data, dtype=np.float32)

    alpha_txt_to_img_eff_data = (gate_txt_to_img * alpha_txt_to_img_data).astype(np.float32)
    alpha_img_to_txt_eff_data = (gate_img_to_txt * alpha_img_to_txt_data).astype(np.float32)

    alpha_txt_to_img = sparse_matrix_from_union_keys(union_keys, alpha_txt_to_img_data, c_img.shape)
    alpha_img_to_txt = sparse_matrix_from_union_keys(union_keys, alpha_img_to_txt_data, c_img.shape)
    alpha_txt_to_img_eff = sparse_matrix_from_union_keys(union_keys, alpha_txt_to_img_eff_data, c_img.shape)
    alpha_img_to_txt_eff = sparse_matrix_from_union_keys(union_keys, alpha_img_to_txt_eff_data, c_img.shape)

    diagnostics = {
        "correction_score_mode": "confidence",
        "requested_local_node_confidence_mode": requested_local_mode,
        "effective_local_node_confidence_mode": effective_local_mode,
        "image_local_confidence_diagnostics": image_local_conf_diagnostics,
        "text_local_confidence_diagnostics": text_local_conf_diagnostics,
        "image_kappa_stats": summarize_vector(image_kappa) if image_kappa is not None else None,
        "text_kappa_stats": summarize_vector(text_kappa) if text_kappa is not None else None,
        "gate_activation_ratio_t2i": float(np.mean(gate_txt_to_img)) if gate_txt_to_img.size else 0.0,
        "gate_activation_ratio_i2t": float(np.mean(gate_img_to_txt)) if gate_img_to_txt.size else 0.0,
        "alpha_t2i_stats_before_gate": summarize_vector(alpha_txt_to_img_data),
        "alpha_t2i_stats_after_gate": summarize_vector(alpha_txt_to_img_eff_data),
        "alpha_i2t_stats_before_gate": summarize_vector(alpha_img_to_txt_data),
        "alpha_i2t_stats_after_gate": summarize_vector(alpha_img_to_txt_eff_data),
        "c_img_stats": summarize_vector(c_img_data),
        "c_txt_stats": summarize_vector(c_txt_data),
        "edge_confidence_note": "mode=none uses C=rho*P without node confidence scaling" if effective_local_mode == "none" else "node confidence scaling enabled",
    }
    return (
        alpha_txt_to_img,
        alpha_img_to_txt,
        alpha_txt_to_img_eff,
        alpha_img_to_txt_eff,
        image_kappa,
        text_kappa,
        diagnostics,
    )


def build_collapse_score_bidirectional_correction_weights(
    image_transition,
    text_transition,
    eps=1e-8,
    collapse_score_mode="edge_plus_neighborhood",
    collapse_neighbor_topk=10,
    collapse_score_weight_edge=1.0,
    collapse_score_weight_a2b=1.0,
    collapse_score_weight_b2a=1.0,
    collapse_score_weight_nbr2nbr=1.0,
):
    union_keys, _, _ = build_union_keys(image_transition, text_transition)
    q_img, image_q_diag = build_edge_collapse_scores(
        image_transition,
        union_keys,
        collapse_score_mode=collapse_score_mode,
        collapse_neighbor_topk=collapse_neighbor_topk,
        collapse_score_weight_edge=collapse_score_weight_edge,
        collapse_score_weight_a2b=collapse_score_weight_a2b,
        collapse_score_weight_b2a=collapse_score_weight_b2a,
        collapse_score_weight_nbr2nbr=collapse_score_weight_nbr2nbr,
        eps=eps,
    )
    q_txt, text_q_diag = build_edge_collapse_scores(
        text_transition,
        union_keys,
        collapse_score_mode=collapse_score_mode,
        collapse_neighbor_topk=collapse_neighbor_topk,
        collapse_score_weight_edge=collapse_score_weight_edge,
        collapse_score_weight_a2b=collapse_score_weight_a2b,
        collapse_score_weight_b2a=collapse_score_weight_b2a,
        collapse_score_weight_nbr2nbr=collapse_score_weight_nbr2nbr,
        eps=eps,
    )

    denom = np.abs(q_img) + np.abs(q_txt) + float(eps)
    alpha_txt_to_img_data = (np.maximum(q_img - q_txt, 0.0) / denom).astype(np.float32)
    alpha_img_to_txt_data = (np.maximum(q_txt - q_img, 0.0) / denom).astype(np.float32)

    alpha_txt_to_img = sparse_matrix_from_union_keys(union_keys, alpha_txt_to_img_data, image_transition.shape)
    alpha_img_to_txt = sparse_matrix_from_union_keys(union_keys, alpha_img_to_txt_data, image_transition.shape)

    diagnostics = {
        "correction_score_mode": "collapse_score",
        "collapse_score_mode": str(collapse_score_mode),
        "collapse_neighbor_topk": int(collapse_neighbor_topk),
        "collapse_score_weights": {
            "edge": float(collapse_score_weight_edge),
            "a2b": float(collapse_score_weight_a2b),
            "b2a": float(collapse_score_weight_b2a),
            "nbr2nbr": float(collapse_score_weight_nbr2nbr),
        },
        "image_q_stats": image_q_diag["q_stats"],
        "text_q_stats": text_q_diag["q_stats"],
        "image_q_components": image_q_diag["components"],
        "text_q_components": text_q_diag["components"],
        "alpha_t2i_stats_before_gate": summarize_vector(alpha_txt_to_img_data),
        "alpha_t2i_stats_after_gate": summarize_vector(alpha_txt_to_img_data),
        "alpha_i2t_stats_before_gate": summarize_vector(alpha_img_to_txt_data),
        "alpha_i2t_stats_after_gate": summarize_vector(alpha_img_to_txt_data),
        "gate_activation_ratio_t2i": 1.0,
        "gate_activation_ratio_i2t": 1.0,
        "edge_collapse_note": "collapse-score bidirectional correction uses transition-edge and neighborhood support without gates or thresholds",
        "requested_local_node_confidence_mode": "none",
        "effective_local_node_confidence_mode": "none",
        "image_local_confidence_diagnostics": {
            "mode": "unused_in_collapse_score",
        },
        "text_local_confidence_diagnostics": {
            "mode": "unused_in_collapse_score",
        },
        "image_kappa_stats": None,
        "text_kappa_stats": None,
    }
    return (
        alpha_txt_to_img,
        alpha_img_to_txt,
        alpha_txt_to_img,
        alpha_img_to_txt,
        None,
        None,
        diagnostics,
    )


def build_union_keys(matrix_a, matrix_b):
    matrix_a = matrix_a.tocoo()
    matrix_b = matrix_b.tocoo()
    num_cols = int(matrix_a.shape[1])
    keys_a = matrix_a.row.astype(np.int64) * num_cols + matrix_a.col.astype(np.int64)
    keys_b = matrix_b.row.astype(np.int64) * num_cols + matrix_b.col.astype(np.int64)
    union_keys = np.unique(np.concatenate([keys_a, keys_b], axis=0))
    rows = (union_keys // num_cols).astype(np.int32)
    cols = (union_keys % num_cols).astype(np.int32)
    return union_keys, rows, cols


def gather_sparse_data_on_keys(matrix, union_keys):
    matrix = matrix.tocoo()
    num_cols = int(matrix.shape[1])
    keys = matrix.row.astype(np.int64) * num_cols + matrix.col.astype(np.int64)
    values = np.zeros(union_keys.shape[0], dtype=np.float32)
    if keys.size == 0:
        return values
    positions = np.searchsorted(union_keys, keys)
    values[positions] = matrix.data.astype(np.float32)
    return values


def build_bidirectional_correction_weights(
    image_transition,
    text_transition,
    rho_img,
    rho_txt,
    eps=1e-8,
    enable_local_node_confidence=False,
    local_node_confidence_mode="multi_view",
    tau_l=0.25,
    kappa_min=0.05,
    local_conf_eps=1e-8,
    local_conf_weight_entropy=1.0,
    local_conf_weight_agreement=1.0,
    local_conf_weight_diffusion=1.0,
    local_conf_agreement_topk=15,
    local_conf_agreement_type="jaccard",
    local_conf_diffusion_hops=2,
    local_conf_diffusion_type="p_vs_p2_cosine",
    correction_score_mode="collapse_score",
    collapse_score_mode="edge_plus_neighborhood",
    collapse_neighbor_topk=10,
    collapse_score_weight_edge=1.0,
    collapse_score_weight_a2b=1.0,
    collapse_score_weight_b2a=1.0,
    collapse_score_weight_nbr2nbr=1.0,
    enable_directional_correction_gate=True,
    correction_gate_tau_high=0.6,
    correction_gate_tau_low=0.3,
    correction_gate_tau_gap=0.15,
):
    correction_score_mode = str(correction_score_mode or "collapse_score")
    if correction_score_mode == "confidence":
        return build_confidence_based_bidirectional_correction_weights(
            image_transition,
            text_transition,
            rho_img=rho_img,
            rho_txt=rho_txt,
            eps=eps,
            enable_local_node_confidence=enable_local_node_confidence,
            local_node_confidence_mode=local_node_confidence_mode,
            tau_l=tau_l,
            kappa_min=kappa_min,
            local_conf_eps=local_conf_eps,
            local_conf_weight_entropy=local_conf_weight_entropy,
            local_conf_weight_agreement=local_conf_weight_agreement,
            local_conf_weight_diffusion=local_conf_weight_diffusion,
            local_conf_agreement_topk=local_conf_agreement_topk,
            local_conf_agreement_type=local_conf_agreement_type,
            local_conf_diffusion_hops=local_conf_diffusion_hops,
            local_conf_diffusion_type=local_conf_diffusion_type,
            enable_directional_correction_gate=enable_directional_correction_gate,
            correction_gate_tau_high=correction_gate_tau_high,
            correction_gate_tau_low=correction_gate_tau_low,
            correction_gate_tau_gap=correction_gate_tau_gap,
        )
    if correction_score_mode == "collapse_score":
        return build_collapse_score_bidirectional_correction_weights(
            image_transition,
            text_transition,
            eps=eps,
            collapse_score_mode=collapse_score_mode,
            collapse_neighbor_topk=collapse_neighbor_topk,
            collapse_score_weight_edge=collapse_score_weight_edge,
            collapse_score_weight_a2b=collapse_score_weight_a2b,
            collapse_score_weight_b2a=collapse_score_weight_b2a,
            collapse_score_weight_nbr2nbr=collapse_score_weight_nbr2nbr,
        )
    raise ValueError(f"Unsupported correction score mode: {correction_score_mode}")


def apply_directional_correction(
    image_bundle,
    text_bundle,
    healthy_modality,
    alpha=1.0,
):
    if healthy_modality == "image":
        healthy_transition = image_bundle["transition"]
        collapsed_graph = text_bundle["graph"]
        corrected_collapsed_directed = collapsed_graph.multiply(sparse_elementwise_power(healthy_transition, alpha))
        corrected_collapsed_symmetric = fuzzy_union_symmetrize(corrected_collapsed_directed)
        return (
            image_bundle["graph"],
            image_bundle["graph"],
            corrected_collapsed_directed.tocsr(),
            corrected_collapsed_symmetric.tocsr(),
        )

    healthy_transition = text_bundle["transition"]
    collapsed_graph = image_bundle["graph"]
    corrected_collapsed_directed = collapsed_graph.multiply(sparse_elementwise_power(healthy_transition, alpha))
    corrected_collapsed_symmetric = fuzzy_union_symmetrize(corrected_collapsed_directed)
    return (
        corrected_collapsed_directed.tocsr(),
        corrected_collapsed_symmetric.tocsr(),
        text_bundle["graph"],
        text_bundle["graph"],
    )


def apply_bidirectional_correction(image_graph, text_graph, alpha_txt_to_img, alpha_img_to_txt):
    image_graph = image_graph.tocsr()
    text_graph = text_graph.tocsr()

    corrected_image_directed = image_graph + alpha_txt_to_img.multiply(text_graph - image_graph)
    corrected_text_directed = text_graph + alpha_img_to_txt.multiply(image_graph - text_graph)
    corrected_image_directed = corrected_image_directed.tocsr()
    corrected_text_directed = corrected_text_directed.tocsr()
    corrected_image_directed.eliminate_zeros()
    corrected_text_directed.eliminate_zeros()

    corrected_image_symmetric = fuzzy_union_symmetrize(corrected_image_directed)
    corrected_text_symmetric = fuzzy_union_symmetrize(corrected_text_directed)
    corrected_image_symmetric.eliminate_zeros()
    corrected_text_symmetric.eliminate_zeros()
    return (
        corrected_image_directed,
        corrected_image_symmetric,
        corrected_text_directed,
        corrected_text_symmetric,
    )


def summarize_sparse_matrix_values(matrix):
    matrix = matrix.tocsr()
    data = matrix.data.astype(np.float32)
    summary = summarize_vector(data)
    summary.update(
        {
            "nnz": int(matrix.nnz),
            "density": float(matrix.nnz / max(int(matrix.shape[0]) * int(matrix.shape[1]), 1)),
        }
    )
    return summary


def build_average_reference_graph(image_graph, text_graph):
    union_keys, rows, cols = build_union_keys(image_graph, text_graph)
    image_data = gather_sparse_data_on_keys(image_graph, union_keys)
    text_data = gather_sparse_data_on_keys(text_graph, union_keys)
    ref_data = 0.5 * (image_data + text_data)
    reference = sparse.csr_matrix((ref_data.astype(np.float32), (rows, cols)), shape=image_graph.shape)
    reference.eliminate_zeros()
    reference = fuzzy_union_symmetrize(reference)
    reference.eliminate_zeros()
    return reference


def build_union_reference_graph(image_graph, text_graph):
    reference = (image_graph + text_graph - image_graph.multiply(text_graph)).tocsr()
    reference.eliminate_zeros()
    reference = fuzzy_union_symmetrize(reference)
    reference.eliminate_zeros()
    return reference


def clip_sparse_matrix_values(matrix, min_value=0.0, max_value=1.0):
    matrix = matrix.tocsr(copy=True)
    if matrix.nnz <= 0:
        return matrix
    data = matrix.data.astype(np.float32)
    data = np.clip(data, float(min_value), float(max_value)).astype(np.float32)
    matrix.data = data
    matrix.eliminate_zeros()
    return matrix


def parse_scale_weight_values(scales, raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        items = list(raw)
    if len(items) == 1 and len(scales) > 1:
        return [float(items[0])] * len(scales)
    if len(items) != len(scales):
        raise ValueError(f"Scale weight count mismatch: got {len(items)} weights for {len(scales)} scales ({scales})")
    return [float(item) for item in items]


def build_wavelet_probe_matrix(num_nodes, probe_mode="random", probe_dim=32, random_state=0):
    num_nodes = int(num_nodes)
    probe_dim = max(1, min(int(probe_dim), num_nodes))
    probe_mode = str(probe_mode or "random")
    rng = np.random.default_rng(int(random_state))
    if probe_mode == "random":
        probes = rng.standard_normal((num_nodes, probe_dim)).astype(np.float32)
        probes /= np.sqrt(float(probe_dim))
        return probes
    if probe_mode == "impulse_subset":
        selected = rng.choice(num_nodes, size=probe_dim, replace=False)
        probes = np.zeros((num_nodes, probe_dim), dtype=np.float32)
        probes[selected, np.arange(probe_dim)] = 1.0
        return probes
    raise ValueError(f"Unsupported wavelet_fusion_probe_mode: {probe_mode}")


def compute_response_stats(response):
    response = np.asarray(response, dtype=np.float32)
    flat_summary = summarize_vector(response.reshape(-1))
    row_norms = np.linalg.norm(response, axis=1).astype(np.float32)
    flat_summary["mean_row_norm"] = float(np.mean(row_norms)) if row_norms.size else 0.0
    flat_summary["std_row_norm"] = float(np.std(row_norms)) if row_norms.size else 0.0
    flat_summary["num_rows"] = int(response.shape[0])
    flat_summary["num_cols"] = int(response.shape[1]) if response.ndim == 2 else 0
    return flat_summary


def compute_response_collapse_metrics(response, eps=1e-8):
    response = np.asarray(response, dtype=np.float32)
    if response.ndim != 2 or response.shape[1] <= 1:
        return {"spectral_entropy": 0.0, "collapse_score": 1.0}
    gram = response.T @ response
    eigenvalues = np.linalg.eigvalsh(gram.astype(np.float64))
    eigenvalues = np.maximum(eigenvalues.astype(np.float32), float(eps))
    weights = eigenvalues / max(float(np.sum(eigenvalues)), float(eps))
    entropy = float(-(weights * np.log(weights + float(eps))).sum() / max(np.log(float(len(weights)) + float(eps)), float(eps)))
    entropy = float(np.clip(entropy, 0.0, 1.0))
    return {
        "spectral_entropy": float(entropy),
        "collapse_score": float(1.0 - entropy),
    }


def compute_graph_collapse_metrics(graph, num_eigs=64, spectrum_solver_mode="normalized_adjacency_largest"):
    graph = graph.tocsr().astype(np.float32)
    target_num_eigs = max(2, min(int(num_eigs), max(2, int(graph.shape[0]) - 1)))
    laplacian = build_laplacian(graph, normalized=True)
    eigenvalues, _ = compute_spectrum(
        laplacian,
        target_num_eigs,
        return_eigenvectors=False,
        solver_mode=spectrum_solver_mode,
    )
    collapse_metrics = compute_collapse_metrics(eigenvalues)
    collapse_metrics = dict(collapse_metrics)
    collapse_metrics["first_eigenvalues"] = [float(x) for x in np.asarray(eigenvalues[: min(10, len(eigenvalues))], dtype=np.float32)]
    collapse_metrics["num_nodes"] = int(graph.shape[0])
    collapse_metrics["nnz"] = int(graph.nnz)
    collapse_metrics["spectrum_solver_mode"] = str(spectrum_solver_mode)
    return collapse_metrics


def compute_modal_multiscale_responses(graph, probes, scales, impl="diffusion_difference", eps=1e-8):
    impl = str(impl or "diffusion_difference")
    if impl not in {"diffusion_difference", "graph_wavelet"}:
        raise ValueError(f"Unsupported wavelet_fusion_impl: {impl}")
    responses = build_multi_scale_wavelet_signatures(
        graph,
        probes,
        scales,
        normalize=False,
        eps=eps,
    )
    return {int(scale): np.asarray(responses[int(scale)], dtype=np.float32) for scale in parse_wavelet_scales(scales)}


def resolve_latent_wavelet_scale_weights(
    scales,
    responses_a,
    responses_b,
    weight_mode="fixed_per_scale",
    weights_a=None,
    weights_b=None,
    modal_graph_collapse_a=None,
    modal_graph_collapse_b=None,
    eps=1e-8,
):
    scales = parse_wavelet_scales(scales)
    weight_mode = str(weight_mode or "fixed_per_scale")
    parsed_a = parse_scale_weight_values(scales, weights_a)
    parsed_b = parse_scale_weight_values(scales, weights_b)
    resolved = {}
    collapse_stats = {}

    if weight_mode == "fixed_per_scale":
        for idx, scale in enumerate(scales):
            if parsed_a is None and parsed_b is None:
                alpha = 0.5
                beta = 0.5
            elif parsed_a is None:
                beta = parsed_b[idx]
                alpha = max(0.0, 1.0 - beta)
            elif parsed_b is None:
                alpha = parsed_a[idx]
                beta = max(0.0, 1.0 - alpha)
            else:
                alpha = parsed_a[idx]
                beta = parsed_b[idx]
            denom = max(abs(alpha) + abs(beta), float(eps))
            resolved[int(scale)] = {"alpha": float(alpha / denom), "beta": float(beta / denom)}
        return resolved, collapse_stats

    if weight_mode == "collapse_aware":
        if modal_graph_collapse_a is not None and modal_graph_collapse_b is not None:
            health_a = max(1.0 - float(modal_graph_collapse_a["collapse_score"]), float(eps))
            health_b = max(1.0 - float(modal_graph_collapse_b["collapse_score"]), float(eps))
            denom = health_a + health_b
            alpha = float(health_a / denom)
            beta = float(health_b / denom)
            graph_collapse_stats = {
                "A": dict(modal_graph_collapse_a),
                "B": dict(modal_graph_collapse_b),
                "omega_A": alpha,
                "omega_B": beta,
                "source": "corrected_graph",
            }
            for scale in scales:
                resolved[int(scale)] = {"alpha": alpha, "beta": beta}
                collapse_stats[str(scale)] = {
                    "A": dict(modal_graph_collapse_a),
                    "B": dict(modal_graph_collapse_b),
                    "source": "corrected_graph",
                }
            collapse_stats["_graph"] = graph_collapse_stats
            return resolved, collapse_stats
        for scale in scales:
            metrics_a = compute_response_collapse_metrics(responses_a[int(scale)], eps=eps)
            metrics_b = compute_response_collapse_metrics(responses_b[int(scale)], eps=eps)
            health_a = max(1.0 - float(metrics_a["collapse_score"]), float(eps))
            health_b = max(1.0 - float(metrics_b["collapse_score"]), float(eps))
            denom = health_a + health_b
            resolved[int(scale)] = {"alpha": float(health_a / denom), "beta": float(health_b / denom)}
            collapse_stats[str(scale)] = {"A": metrics_a, "B": metrics_b}
        return resolved, collapse_stats

    raise ValueError(f"Unsupported wavelet_fusion_weight_mode: {weight_mode}")


def build_graph_domain_unified_graph(
    image_graph,
    text_graph,
    mode="confidence_aware",
    correction_fusion_mode="legacy",
    rho_img=0.5,
    rho_txt=0.5,
    lambda_f=1.0,
    mu_f=1.0,
    eps=1e-8,
    alpha_txt_to_img_eff=None,
    alpha_img_to_txt_eff=None,
):
    if mode == "intersection":
        unified = image_graph.multiply(text_graph)
        unified = fuzzy_union_symmetrize(unified)
        unified.eliminate_zeros()
        return unified, {}
    if mode == "confidence_aware":
        if str(correction_fusion_mode or "legacy") == "thresholded_autonomy":
            if alpha_txt_to_img_eff is None or alpha_img_to_txt_eff is None:
                raise ValueError("thresholded_autonomy fusion requires effective bidirectional correction weights.")
            return build_thresholded_autonomy_fusion(
                image_graph,
                text_graph,
                alpha_txt_to_img_eff=alpha_txt_to_img_eff,
                alpha_img_to_txt_eff=alpha_img_to_txt_eff,
                eps=eps,
            )
        unified = build_confidence_aware_fusion(
            image_graph,
            text_graph,
            rho_img=rho_img,
            rho_txt=rho_txt,
            lambda_f=lambda_f,
            mu_f=mu_f,
            eps=eps,
        )
        return unified, {}
    raise ValueError(f"Unsupported fusion mode: {mode}")


def build_wavelet_latent_fusion(
    image_graph,
    text_graph,
    eps=1e-8,
    wavelet_fusion_impl="diffusion_difference",
    wavelet_fusion_scales="1,2,4",
    wavelet_fusion_probe_mode="random",
    wavelet_fusion_probe_dim=32,
    wavelet_fusion_weight_mode="fixed_per_scale",
    wavelet_fusion_weight_a_scales=None,
    wavelet_fusion_weight_b_scales=None,
    modal_graph_collapse_a=None,
    modal_graph_collapse_b=None,
    wavelet_latent_lambda_sparse=0.01,
    wavelet_latent_lambda_sym=1.0,
    wavelet_latent_lambda_nonneg=1.0,
    wavelet_latent_reconstruction_mode="candidate_response_similarity",
    wavelet_latent_postprocess_topk=64,
    wavelet_latent_postprocess_threshold=0.0,
):
    scales = parse_wavelet_scales(wavelet_fusion_scales)
    probe_dim = max(1, min(int(wavelet_fusion_probe_dim), int(image_graph.shape[0])))
    probes = build_wavelet_probe_matrix(
        num_nodes=image_graph.shape[0],
        probe_mode=wavelet_fusion_probe_mode,
        probe_dim=probe_dim,
        random_state=0,
    )
    impl = str(wavelet_fusion_impl or "diffusion_difference")
    response_a = compute_modal_multiscale_responses(image_graph, probes, scales, impl=impl, eps=eps)
    response_b = compute_modal_multiscale_responses(text_graph, probes, scales, impl=impl, eps=eps)
    resolved_scale_weights, collapse_stats = resolve_latent_wavelet_scale_weights(
        scales,
        response_a,
        response_b,
        weight_mode=wavelet_fusion_weight_mode,
        weights_a=wavelet_fusion_weight_a_scales,
        weights_b=wavelet_fusion_weight_b_scales,
        modal_graph_collapse_a=modal_graph_collapse_a,
        modal_graph_collapse_b=modal_graph_collapse_b,
        eps=eps,
    )

    target_responses = {}
    per_scale_stats = {}
    for scale in scales:
        alpha = float(resolved_scale_weights[int(scale)]["alpha"])
        beta = float(resolved_scale_weights[int(scale)]["beta"])
        target = (alpha * response_a[int(scale)] + beta * response_b[int(scale)]).astype(np.float32)
        target_responses[int(scale)] = target
        scale_diag = {
            "A": compute_response_stats(response_a[int(scale)]),
            "B": compute_response_stats(response_b[int(scale)]),
            "target": compute_response_stats(target),
            "omega_A": alpha,
            "omega_B": beta,
        }
        if str(scale) in collapse_stats:
            scale_diag["collapse"] = collapse_stats[str(scale)]
        per_scale_stats[str(scale)] = scale_diag

    union_keys, rows, cols = build_union_keys(image_graph, text_graph)
    scale_edge_scores = []
    for scale in scales:
        target = target_responses[int(scale)]
        row_norm = np.linalg.norm(target, axis=1, keepdims=True)
        row_norm = np.maximum(row_norm, float(eps))
        target_norm = (target / row_norm).astype(np.float32)
        edge_score = np.sum(target_norm[rows] * target_norm[cols], axis=1).astype(np.float32)
        edge_score = np.clip(edge_score, -1.0, 1.0)
        edge_score = np.maximum(edge_score, 0.0)
        scale_edge_scores.append(edge_score)

    if wavelet_latent_reconstruction_mode != "candidate_response_similarity":
        raise ValueError(f"Unsupported wavelet_latent_reconstruction_mode: {wavelet_latent_reconstruction_mode}")

    if scale_edge_scores:
        edge_scores = np.mean(np.stack(scale_edge_scores, axis=0), axis=0).astype(np.float32)
    else:
        edge_scores = np.zeros(union_keys.shape[0], dtype=np.float32)

    pre_nonneg_stats = summarize_vector(edge_scores)
    edge_scores = edge_scores - float(wavelet_latent_lambda_sparse)
    edge_scores = np.maximum(edge_scores, 0.0).astype(np.float32)
    pre_nonneg_nnz = int(np.count_nonzero(edge_scores > 0))

    directed = sparse.csr_matrix((edge_scores.astype(np.float32), (rows, cols)), shape=image_graph.shape, dtype=np.float32)
    directed.eliminate_zeros()
    pre_sym_nnz = int(directed.nnz)
    pre_sym_stats = summarize_sparse_matrix_values(directed)

    sym = ((directed + directed.transpose()) * 0.5).tocsr()
    sym.eliminate_zeros()
    sym_stats = summarize_sparse_matrix_values(sym)
    if float(wavelet_latent_lambda_sym) > 0:
        blend = 1.0 / (1.0 + float(wavelet_latent_lambda_sym))
        directed = (blend * directed + (1.0 - blend) * sym).tocsr()
        directed.eliminate_zeros()
    pre_projection_stats = summarize_sparse_matrix_values(directed)
    directed = clip_sparse_matrix_values(directed, min_value=0.0, max_value=1.0)
    if float(wavelet_latent_lambda_nonneg) >= 0.0:
        directed = clip_sparse_matrix_values(directed, min_value=0.0, max_value=1.0)
    post_projection_stats = summarize_sparse_matrix_values(directed)
    projected_nonneg_nnz = int(directed.nnz)

    final_graph = ((directed + directed.transpose()) * 0.5).tocsr()
    final_graph.eliminate_zeros()
    pre_sparsify_nnz = int(final_graph.nnz)
    final_graph = sparsify_sparse_matrix(
        final_graph,
        topk=wavelet_latent_postprocess_topk,
        threshold=wavelet_latent_postprocess_threshold,
        use_abs=False,
    )
    final_graph = ((final_graph + final_graph.transpose()) * 0.5).tocsr()
    final_graph = clip_sparse_matrix_values(final_graph, min_value=0.0, max_value=1.0)
    final_graph.eliminate_zeros()
    final_graph_stats = summarize_sparse_matrix_values(final_graph)

    response_star = compute_modal_multiscale_responses(final_graph, probes, scales, impl=impl, eps=eps)
    for scale in scales:
        per_scale_stats[str(scale)]["star"] = compute_response_stats(response_star[int(scale)])

    diagnostics = {
        "fusion_domain_mode": "wavelet_latent",
        "wavelet_fusion_impl": impl,
        "wavelet_fusion_scales": [int(scale) for scale in scales],
        "wavelet_fusion_probe_mode": str(wavelet_fusion_probe_mode),
        "wavelet_fusion_probe_dim": int(probe_dim),
        "wavelet_fusion_weight_mode": str(wavelet_fusion_weight_mode),
        "wavelet_scale_weights": {
            str(scale): {
                "omega_A": float(resolved_scale_weights[int(scale)]["alpha"]),
                "omega_B": float(resolved_scale_weights[int(scale)]["beta"]),
            }
            for scale in scales
        },
        "wavelet_latent_lambda_sparse": float(wavelet_latent_lambda_sparse),
        "wavelet_latent_lambda_sym": float(wavelet_latent_lambda_sym),
        "wavelet_latent_lambda_nonneg": float(wavelet_latent_lambda_nonneg),
        "wavelet_latent_reconstruction_mode": str(wavelet_latent_reconstruction_mode),
        "wavelet_latent_postprocess_topk": None if wavelet_latent_postprocess_topk is None else int(wavelet_latent_postprocess_topk),
        "wavelet_latent_postprocess_threshold": float(wavelet_latent_postprocess_threshold),
        "latent_candidate_edge_count": int(len(union_keys)),
        "pre_nonneg_edge_score_stats": pre_nonneg_stats,
        "pre_nonneg_positive_edges": int(pre_nonneg_nnz),
        "pre_sym_directed_nnz": int(pre_sym_nnz),
        "pre_sym_directed_stats": pre_sym_stats,
        "symmetrized_candidate_stats": sym_stats,
        "pre_projection_graph_stats": pre_projection_stats,
        "post_projection_graph_stats": post_projection_stats,
        "projected_nonneg_nnz": int(projected_nonneg_nnz),
        "pre_sparsify_reconstructed_nnz": int(pre_sparsify_nnz),
        "post_sparsify_reconstructed_nnz": int(final_graph.nnz),
        "final_graph_stats": final_graph_stats,
        "per_scale_stats": per_scale_stats,
        "latent_graph_note": "Approximate latent graph fitting: fuse modal multiscale responses first, then infer B* on the union candidate edge set by response-similarity graph learning.",
    }
    if collapse_stats:
        diagnostics["collapse_aware_scale_stats"] = collapse_stats
    if collapse_stats.get("_graph") is not None:
        diagnostics["collapse_aware_graph_stats"] = collapse_stats["_graph"]
    return final_graph, diagnostics


def build_thresholded_autonomy_fusion(
    image_graph,
    text_graph,
    alpha_txt_to_img_eff,
    alpha_img_to_txt_eff,
    eps=1e-8,
):
    image_graph = image_graph.tocsr()
    text_graph = text_graph.tocsr()
    union_keys, rows, cols = build_union_keys(image_graph, text_graph)
    image_data = gather_sparse_data_on_keys(image_graph, union_keys)
    text_data = gather_sparse_data_on_keys(text_graph, union_keys)
    alpha_txt_to_img_eff_data = gather_sparse_data_on_keys(alpha_txt_to_img_eff, union_keys)
    alpha_img_to_txt_eff_data = gather_sparse_data_on_keys(alpha_img_to_txt_eff, union_keys)

    autonomy_img = 1.0 - alpha_txt_to_img_eff_data
    autonomy_txt = 1.0 - alpha_img_to_txt_eff_data
    denom = autonomy_img + autonomy_txt + float(eps)
    omega_img = autonomy_img / denom
    omega_txt = autonomy_txt / denom
    unified_data = omega_img * image_data + omega_txt * text_data

    unified = sparse.csr_matrix((unified_data.astype(np.float32), (rows, cols)), shape=image_graph.shape)
    unified.eliminate_zeros()
    unified = fuzzy_union_symmetrize(unified)
    unified.eliminate_zeros()
    diagnostics = {
        "omega_img_stats": summarize_vector(omega_img),
        "omega_txt_stats": summarize_vector(omega_txt),
    }
    return unified, diagnostics


def build_confidence_aware_fusion(
    image_graph,
    text_graph,
    rho_img,
    rho_txt,
    lambda_f=1.0,
    mu_f=1.0,
    eps=1e-8,
):
    image_graph = image_graph.tocsr()
    text_graph = text_graph.tocsr()

    image_norm = row_normalize_graph(image_graph)
    text_norm = row_normalize_graph(text_graph)
    c_img = scale_sparse_graph(image_norm, rho_img)
    c_txt = scale_sparse_graph(text_norm, rho_txt)

    union_keys, rows, cols = build_union_keys(image_graph, text_graph)
    image_data = gather_sparse_data_on_keys(image_graph, union_keys)
    text_data = gather_sparse_data_on_keys(text_graph, union_keys)
    c_img_data = gather_sparse_data_on_keys(c_img, union_keys)
    c_txt_data = gather_sparse_data_on_keys(c_txt, union_keys)

    lambda_f = float(lambda_f)
    mu_f = float(mu_f)
    eps = float(eps)

    c_img_pow = np.power(np.clip(c_img_data, 0.0, None), lambda_f, dtype=np.float32)
    c_txt_pow = np.power(np.clip(c_txt_data, 0.0, None), lambda_f, dtype=np.float32)
    denom = c_img_pow + c_txt_pow + eps
    alpha = c_img_pow / denom
    beta = c_txt_pow / denom

    fused = alpha * image_data + beta * text_data

    # Old logic used hard intersection. The new default keeps edge support soft by
    # applying confidence-aware weighted fusion plus soft consistency gating.
    gate = np.power(c_img_data * c_txt_data + eps, mu_f, dtype=np.float32)
    unified_data = gate * fused

    unified = sparse.csr_matrix((unified_data.astype(np.float32), (rows, cols)), shape=image_graph.shape)
    unified.eliminate_zeros()
    unified = fuzzy_union_symmetrize(unified)
    unified.eliminate_zeros()
    return unified


def unify_topology(
    image_graph,
    text_graph,
    fusion_domain_mode="wavelet_latent",
    mode="confidence_aware",
    correction_fusion_mode="legacy",
    rho_img=0.5,
    rho_txt=0.5,
    lambda_f=1.0,
    mu_f=1.0,
    eps=1e-8,
    alpha_txt_to_img_eff=None,
    alpha_img_to_txt_eff=None,
    wavelet_fusion_scales="1,2,4",
    wavelet_fusion_impl="diffusion_difference",
    wavelet_fusion_probe_mode="random",
    wavelet_fusion_probe_dim=32,
    wavelet_fusion_weight_mode="fixed_per_scale",
    wavelet_fusion_weight_a_scales=None,
    wavelet_fusion_weight_b_scales=None,
    modal_graph_collapse_a=None,
    modal_graph_collapse_b=None,
    wavelet_latent_lambda_sparse=0.01,
    wavelet_latent_lambda_sym=1.0,
    wavelet_latent_lambda_nonneg=1.0,
    wavelet_latent_reconstruction_mode="candidate_response_similarity",
    wavelet_latent_postprocess_topk=64,
    wavelet_latent_postprocess_threshold=0.0,
):
    fusion_domain_mode = str(fusion_domain_mode or "wavelet_latent")
    if fusion_domain_mode == "graph":
        return build_graph_domain_unified_graph(
            image_graph,
            text_graph,
            mode=mode,
            correction_fusion_mode=correction_fusion_mode,
            rho_img=rho_img,
            rho_txt=rho_txt,
            lambda_f=lambda_f,
            mu_f=mu_f,
            eps=eps,
            alpha_txt_to_img_eff=alpha_txt_to_img_eff,
            alpha_img_to_txt_eff=alpha_img_to_txt_eff,
        )
    if fusion_domain_mode == "wavelet_latent":
        return build_wavelet_latent_fusion(
            image_graph,
            text_graph,
            eps=eps,
            wavelet_fusion_scales=wavelet_fusion_scales,
            wavelet_fusion_impl=wavelet_fusion_impl,
            wavelet_fusion_probe_mode=wavelet_fusion_probe_mode,
            wavelet_fusion_probe_dim=wavelet_fusion_probe_dim,
            wavelet_fusion_weight_mode=wavelet_fusion_weight_mode,
            wavelet_fusion_weight_a_scales=wavelet_fusion_weight_a_scales,
            wavelet_fusion_weight_b_scales=wavelet_fusion_weight_b_scales,
            modal_graph_collapse_a=modal_graph_collapse_a,
            modal_graph_collapse_b=modal_graph_collapse_b,
            wavelet_latent_lambda_sparse=wavelet_latent_lambda_sparse,
            wavelet_latent_lambda_sym=wavelet_latent_lambda_sym,
            wavelet_latent_lambda_nonneg=wavelet_latent_lambda_nonneg,
            wavelet_latent_reconstruction_mode=wavelet_latent_reconstruction_mode,
            wavelet_latent_postprocess_topk=wavelet_latent_postprocess_topk,
            wavelet_latent_postprocess_threshold=wavelet_latent_postprocess_threshold,
        )
    raise ValueError(f"Unsupported fusion_domain_mode: {fusion_domain_mode}")


def summarize_graph(graph, collapse_metrics=None):
    num_nodes = int(graph.shape[0])
    num_edges = int(graph.nnz)
    avg_degree = float(num_edges / max(num_nodes, 1))
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    nonzero_degree = degree[degree > 0]
    summary = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "min_degree": float(degree.min()) if degree.size else 0.0,
        "max_degree": float(degree.max()) if degree.size else 0.0,
        "mean_nonzero_degree": float(nonzero_degree.mean()) if nonzero_degree.size else 0.0,
        "density": float(num_edges / max(num_nodes * num_nodes, 1)),
    }
    if collapse_metrics is not None:
        summary.update(
            {
                "spectral_entropy": float(collapse_metrics.get("spectral_entropy", 0.0)),
                "collapse_score": float(collapse_metrics.get("collapse_score", 1.0)),
                "num_eigs_used": int(collapse_metrics.get("num_eigs_used", 0)),
                "collapse_first_eigenvalues": [float(x) for x in collapse_metrics.get("first_eigenvalues", [])],
            }
        )
    return summary


def build_unified_spectral_artifacts(
    unified_graph,
    num_eigs=64,
    embedding_dim=32,
    save_eigenvectors=True,
    spectrum_solver_mode="normalized_adjacency_largest",
    embedding_type="diffusion",
    diffusion_dim=None,
    diffusion_time=1.0,
    diffusion_eig_solver="auto",
):
    embedding_type = str(embedding_type or "diffusion")
    if embedding_type not in {"laplacian", "diffusion"}:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    target_dim = int(diffusion_dim) if diffusion_dim is not None else int(embedding_dim)
    log_cross_modal(
        f"building unified spectral artifacts for B*: shape={unified_graph.shape}, nnz={unified_graph.nnz}, "
        f"embedding_type={embedding_type}, num_eigs={num_eigs}, embedding_dim={target_dim}"
    )
    laplacian = build_laplacian(unified_graph, normalized=True)
    transition = row_normalize_graph(unified_graph)
    if embedding_type == "laplacian":
        log_cross_modal("computing unified Laplacian spectrum")
        eigenvalues, eigenvectors = compute_spectrum(
            laplacian,
            num_eigs=num_eigs,
            return_eigenvectors=save_eigenvectors or target_dim is not None,
            solver_mode=spectrum_solver_mode,
        )
        log_cross_modal("building unified Laplacian spectral embedding")
        spectral_embedding = build_spectral_embedding(
            eigenvalues,
            eigenvectors,
            embedding_dim=target_dim,
        )
        used_eigenvalues = eigenvalues[1 : 1 + max(0, spectral_embedding.shape[1])] if spectral_embedding is not None else np.zeros(0, dtype=np.float32)
        top_values = [f"{float(x):.6f}" for x in used_eigenvalues[: min(10, len(used_eigenvalues))]]
        log_cross_modal(f"top Laplacian eigenvalues used: {top_values}")
    else:
        log_cross_modal(
            f"computing diffusion spectrum: diffusion_dim={target_dim}, diffusion_time={float(diffusion_time):.4f}, "
            f"diffusion_eig_solver={diffusion_eig_solver}"
        )
        diffusion_num_eigs = max(int(num_eigs), int(target_dim) + 1)
        eigenvalues, eigenvectors = compute_diffusion_spectrum(
            transition,
            num_eigs=diffusion_num_eigs,
            return_eigenvectors=save_eigenvectors or target_dim is not None,
            solver_mode=diffusion_eig_solver,
        )
        log_cross_modal("building unified diffusion embedding")
        spectral_embedding, used_eigenvalues = build_diffusion_embedding(
            eigenvalues,
            eigenvectors,
            embedding_dim=target_dim,
            diffusion_time=diffusion_time,
        )
        top_values = [f"{float(x):.6f}" for x in used_eigenvalues[: min(10, len(used_eigenvalues))]]
        log_cross_modal(f"top diffusion eigenvalues used: {top_values}")
    return {
        "laplacian_sym": laplacian,
        "transition_rw": transition,
        "eigvals": eigenvalues,
        "eigvecs": eigenvectors,
        "spectral_embedding": spectral_embedding,
        "embedding_type": embedding_type,
        "used_eigenvalues": used_eigenvalues.astype(np.float32),
    }


def build_summary(
    args,
    healthy_modality,
    image_bundle,
    text_bundle,
    rho_img,
    rho_txt,
    image_kappa_stats,
    text_kappa_stats,
    corrected_image_summary,
    corrected_text_summary,
    corrected_summary,
    unified_summary,
    unified_spectral_artifacts,
    correction_diagnostics,
    fusion_diagnostics,
    effective_correction_fusion_mode,
):
    return {
        "dataset": args.dataset,
        "split": args.split,
        "image_encoder": args.image_encoder,
        "text_encoder": args.text_encoder,
        "metric": args.metric,
        "image_metric": get_modality_metric(args, "image"),
        "text_metric": get_modality_metric(args, "text"),
        "k": int(args.k),
        "alpha": float(args.alpha),
        "correction_mode": getattr(args, "correction_mode", "bidirectional"),
        "correction_score_mode": str(getattr(args, "correction_score_mode", "collapse_score")),
        "collapse_score_mode": str(getattr(args, "collapse_score_mode", "edge_plus_neighborhood")),
        "collapse_neighbor_topk": int(getattr(args, "collapse_neighbor_topk", 10)),
        "collapse_score_weight_edge": float(getattr(args, "collapse_score_weight_edge", 1.0)),
        "collapse_score_weight_a2b": float(getattr(args, "collapse_score_weight_a2b", 1.0)),
        "collapse_score_weight_b2a": float(getattr(args, "collapse_score_weight_b2a", 1.0)),
        "collapse_score_weight_nbr2nbr": float(getattr(args, "collapse_score_weight_nbr2nbr", 1.0)),
        "tau_g": float(getattr(args, "tau_g", 0.5)),
        "correction_eps": float(getattr(args, "correction_eps", 1e-8)),
        "enable_local_node_confidence": bool(getattr(args, "enable_local_node_confidence", False)),
        "local_node_confidence_mode": str(getattr(args, "local_node_confidence_mode", "multi_view")),
        "tau_l": float(getattr(args, "tau_l", 0.25)),
        "kappa_min": float(getattr(args, "kappa_min", 0.05)),
        "local_conf_eps": float(getattr(args, "local_conf_eps", 1e-8)),
        "local_conf_weight_entropy": float(getattr(args, "local_conf_weight_entropy", 1.0)),
        "local_conf_weight_agreement": float(getattr(args, "local_conf_weight_agreement", 1.0)),
        "local_conf_weight_diffusion": float(getattr(args, "local_conf_weight_diffusion", 1.0)),
        "local_conf_agreement_topk": int(getattr(args, "local_conf_agreement_topk", 15)),
        "local_conf_agreement_type": str(getattr(args, "local_conf_agreement_type", "jaccard")),
        "local_conf_diffusion_hops": int(getattr(args, "local_conf_diffusion_hops", 2)),
        "local_conf_diffusion_type": str(getattr(args, "local_conf_diffusion_type", "p_vs_p2_cosine")),
        "enable_directional_correction_gate": bool(getattr(args, "enable_directional_correction_gate", True)),
        "correction_gate_tau_high": float(getattr(args, "correction_gate_tau_high", 0.6)),
        "correction_gate_tau_low": float(getattr(args, "correction_gate_tau_low", 0.3)),
        "correction_gate_tau_gap": float(getattr(args, "correction_gate_tau_gap", 0.15)),
        "requested_correction_fusion_mode": str(getattr(args, "correction_fusion_mode", "thresholded_autonomy")),
        "effective_correction_fusion_mode": str(effective_correction_fusion_mode),
        "fusion_domain_mode": str(getattr(args, "fusion_domain_mode", "wavelet_latent")),
        "fusion_mode": args.fusion_mode,
        "lambda_f": float(getattr(args, "lambda_f", 1.0)),
        "mu_f": float(getattr(args, "mu_f", 1.0)),
        "fusion_eps": float(getattr(args, "fusion_eps", 1e-8)),
        "wavelet_fusion_scales": [int(scale) for scale in parse_wavelet_scales(getattr(args, "wavelet_fusion_scales", "1,2,4"))],
        "wavelet_fusion_impl": str(getattr(args, "wavelet_fusion_impl", "diffusion_difference")),
        "wavelet_fusion_probe_mode": str(getattr(args, "wavelet_fusion_probe_mode", "random")),
        "wavelet_fusion_probe_dim": int(getattr(args, "wavelet_fusion_probe_dim", 32)),
        "wavelet_fusion_weight_mode": str(getattr(args, "wavelet_fusion_weight_mode", "fixed_per_scale")),
        "wavelet_fusion_weight_a_scales": getattr(args, "wavelet_fusion_weight_a_scales", None),
        "wavelet_fusion_weight_b_scales": getattr(args, "wavelet_fusion_weight_b_scales", None),
        "wavelet_latent_lambda_sparse": float(getattr(args, "wavelet_latent_lambda_sparse", 0.01)),
        "wavelet_latent_lambda_sym": float(getattr(args, "wavelet_latent_lambda_sym", 1.0)),
        "wavelet_latent_lambda_nonneg": float(getattr(args, "wavelet_latent_lambda_nonneg", 1.0)),
        "wavelet_latent_reconstruction_mode": str(getattr(args, "wavelet_latent_reconstruction_mode", "candidate_response_similarity")),
        "wavelet_latent_postprocess_topk": None if getattr(args, "wavelet_latent_postprocess_topk", None) is None else int(getattr(args, "wavelet_latent_postprocess_topk", 64)),
        "wavelet_latent_postprocess_threshold": float(getattr(args, "wavelet_latent_postprocess_threshold", 0.0)),
        "embedding_type": str(getattr(args, "embedding_type", "diffusion")),
        "diffusion_dim": int(getattr(args, "diffusion_dim", None) or getattr(args, "spectral_embedding_dim", 32)),
        "diffusion_time": float(getattr(args, "diffusion_time", 1.0)),
        "diffusion_eig_solver": str(getattr(args, "diffusion_eig_solver", "auto")),
        "healthy_modality": healthy_modality,
        "collapsed_modality": "text" if healthy_modality == "image" else "image",
        "rho_img": float(rho_img),
        "rho_txt": float(rho_txt),
        "image_kappa_stats": image_kappa_stats,
        "text_kappa_stats": text_kappa_stats,
        "image_summary": image_bundle["summary"],
        "text_summary": text_bundle["summary"],
        "corrected_image_summary": corrected_image_summary,
        "corrected_text_summary": corrected_text_summary,
        "corrected_summary": corrected_summary,
        "unified_summary": unified_summary,
        "correction_diagnostics": correction_diagnostics,
        "fusion_diagnostics": fusion_diagnostics,
        "unified_first_eigenvalues": [
            float(x) for x in unified_spectral_artifacts["eigvals"][: min(10, len(unified_spectral_artifacts["eigvals"]))]
        ],
        "unified_used_eigenvalues": [
            float(x)
            for x in unified_spectral_artifacts["used_eigenvalues"][: min(10, len(unified_spectral_artifacts["used_eigenvalues"]))]
        ],
        "unified_embedding_dim": int(unified_spectral_artifacts["spectral_embedding"].shape[1]) if unified_spectral_artifacts["spectral_embedding"] is not None else 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_cross_modal_outputs(
    output_dir,
    healthy_modality,
    image_bundle,
    text_bundle,
    corrected_image_directed,
    corrected_image_symmetric,
    corrected_text_directed,
    corrected_text_symmetric,
    unified_graph,
    unified_spectral_artifacts,
    summary,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_cross_modal(f"saving cross-modal artifacts to {output_dir}")

    sparse.save_npz(output_dir / "B_I_or_health.npz", image_bundle["graph"] if healthy_modality == "image" else text_bundle["graph"])
    sparse.save_npz(output_dir / "B_collapsed_raw.npz", text_bundle["graph"] if healthy_modality == "image" else image_bundle["graph"])
    sparse.save_npz(output_dir / "healthy_graph.npz", image_bundle["graph"] if healthy_modality == "image" else text_bundle["graph"])
    sparse.save_npz(output_dir / "healthy_transition.npz", image_bundle["transition"] if healthy_modality == "image" else text_bundle["transition"])
    sparse.save_npz(output_dir / "collapsed_graph.npz", text_bundle["graph"] if healthy_modality == "image" else image_bundle["graph"])

    sparse.save_npz(output_dir / "corrected_image_graph_directed.npz", corrected_image_directed)
    sparse.save_npz(output_dir / "corrected_image_graph_symmetric.npz", corrected_image_symmetric)
    sparse.save_npz(output_dir / "corrected_text_graph_directed.npz", corrected_text_directed)
    sparse.save_npz(output_dir / "corrected_text_graph_symmetric.npz", corrected_text_symmetric)
    sparse.save_npz(
        output_dir / "corrected_graph_directed.npz",
        corrected_text_directed if healthy_modality == "image" else corrected_image_directed,
    )
    sparse.save_npz(
        output_dir / "corrected_graph_symmetric.npz",
        corrected_text_symmetric if healthy_modality == "image" else corrected_image_symmetric,
    )
    sparse.save_npz(output_dir / "unified_graph.npz", unified_graph)
    sparse.save_npz(output_dir / "B_star.npz", unified_graph)
    sparse.save_npz(output_dir / "unified_transition.npz", row_normalize_graph(unified_graph))
    sparse.save_npz(output_dir / "unified_laplacian_sym.npz", unified_spectral_artifacts["laplacian_sym"])
    sparse.save_npz(output_dir / "L_star.npz", unified_spectral_artifacts["laplacian_sym"])

    np.save(output_dir / "unified_first_eigvals.npy", unified_spectral_artifacts["eigvals"])
    if unified_spectral_artifacts["eigvecs"] is not None:
        np.save(output_dir / "unified_eigvecs.npy", unified_spectral_artifacts["eigvecs"])
    if unified_spectral_artifacts["spectral_embedding"] is not None:
        np.save(output_dir / "unified_spectral_embedding.npy", unified_spectral_artifacts["spectral_embedding"])
        np.save(output_dir / "V_full_multi.npy", unified_spectral_artifacts["spectral_embedding"])

    with open(output_dir / "sample_meta.json", "w", encoding="utf-8") as handle:
        json.dump(image_bundle["sample_meta"], handle, ensure_ascii=False, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    selection = {
        "healthy_modality": healthy_modality,
        "collapsed_modality": "text" if healthy_modality == "image" else "image",
        "healthy_graph_dir": image_bundle["dir"] if healthy_modality == "image" else text_bundle["dir"],
        "collapsed_graph_dir": text_bundle["dir"] if healthy_modality == "image" else image_bundle["dir"],
    }
    with open(output_dir / "modality_selection.json", "w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)


def run_cross_modal_topology(args):
    log_cross_modal(
        f"start cross-modal topology: dataset={args.dataset}, split={args.split}, "
        f"image_metric={get_modality_metric(args, 'image')}, text_metric={get_modality_metric(args, 'text')}, "
        f"k={args.k}, alpha={args.alpha}"
    )
    image_bundle = load_graph_bundle(build_graph_dir(args, "image"))
    text_bundle = load_graph_bundle(build_graph_dir(args, "text"))
    log_cross_modal("validating image/text graph compatibility")
    validate_modalities(image_bundle, text_bundle)

    log_cross_modal("choosing healthy modality")
    healthy_modality = choose_healthy_modality(
        image_bundle["summary"],
        text_bundle["summary"],
        prefer=args.prefer_healthy_modality,
    )
    log_cross_modal(f"healthy modality selected for reporting only: {healthy_modality}")

    log_cross_modal("computing global modality confidences")
    rho_img, rho_txt = compute_global_confidences(
        image_bundle["summary"]["collapse_score"],
        text_bundle["summary"]["collapse_score"],
        tau_g=getattr(args, "tau_g", 0.5),
    )
    log_cross_modal(f"global confidences: rho_img={rho_img:.4f}, rho_txt={rho_txt:.4f}")
    image_kappa_stats = None
    text_kappa_stats = None
    correction_diagnostics = {}
    fusion_diagnostics = {}
    effective_correction_fusion_mode = str(getattr(args, "correction_fusion_mode", "thresholded_autonomy"))

    correction_mode = getattr(args, "correction_mode", "bidirectional")
    if correction_mode == "directional":
        if effective_correction_fusion_mode != "legacy":
            log_cross_modal("directional correction does not support thresholded_autonomy fusion; falling back to legacy fusion path")
            effective_correction_fusion_mode = "legacy"
        log_cross_modal("applying directional healthy-to-collapsed correction")
        (
            corrected_image_directed,
            corrected_image_symmetric,
            corrected_text_directed,
            corrected_text_symmetric,
        ) = apply_directional_correction(
            image_bundle,
            text_bundle,
            healthy_modality=healthy_modality,
            alpha=getattr(args, "alpha", 1.0),
        )
    elif correction_mode == "bidirectional":
        log_cross_modal("applying bidirectional graph correction")
        (
            alpha_txt_to_img,
            alpha_img_to_txt,
            alpha_txt_to_img_eff,
            alpha_img_to_txt_eff,
            image_kappa,
            text_kappa,
            correction_diagnostics,
        ) = build_bidirectional_correction_weights(
            image_bundle["transition"],
            text_bundle["transition"],
            rho_img=rho_img,
            rho_txt=rho_txt,
            eps=getattr(args, "correction_eps", 1e-8),
            enable_local_node_confidence=bool(getattr(args, "enable_local_node_confidence", False)),
            local_node_confidence_mode=str(getattr(args, "local_node_confidence_mode", "multi_view")),
            tau_l=getattr(args, "tau_l", 0.25),
            kappa_min=getattr(args, "kappa_min", 0.05),
            local_conf_eps=getattr(args, "local_conf_eps", 1e-8),
            local_conf_weight_entropy=getattr(args, "local_conf_weight_entropy", 1.0),
            local_conf_weight_agreement=getattr(args, "local_conf_weight_agreement", 1.0),
            local_conf_weight_diffusion=getattr(args, "local_conf_weight_diffusion", 1.0),
            local_conf_agreement_topk=getattr(args, "local_conf_agreement_topk", 15),
            local_conf_agreement_type=getattr(args, "local_conf_agreement_type", "jaccard"),
            local_conf_diffusion_hops=getattr(args, "local_conf_diffusion_hops", 2),
            local_conf_diffusion_type=getattr(args, "local_conf_diffusion_type", "p_vs_p2_cosine"),
            correction_score_mode=str(getattr(args, "correction_score_mode", "collapse_score")),
            collapse_score_mode=str(getattr(args, "collapse_score_mode", "edge_plus_neighborhood")),
            collapse_neighbor_topk=getattr(args, "collapse_neighbor_topk", 10),
            collapse_score_weight_edge=getattr(args, "collapse_score_weight_edge", 1.0),
            collapse_score_weight_a2b=getattr(args, "collapse_score_weight_a2b", 1.0),
            collapse_score_weight_b2a=getattr(args, "collapse_score_weight_b2a", 1.0),
            collapse_score_weight_nbr2nbr=getattr(args, "collapse_score_weight_nbr2nbr", 1.0),
            enable_directional_correction_gate=bool(getattr(args, "enable_directional_correction_gate", True)),
            correction_gate_tau_high=getattr(args, "correction_gate_tau_high", 0.6),
            correction_gate_tau_low=getattr(args, "correction_gate_tau_low", 0.3),
            correction_gate_tau_gap=getattr(args, "correction_gate_tau_gap", 0.15),
        )
        correction_score_mode = str(correction_diagnostics.get("correction_score_mode", getattr(args, "correction_score_mode", "collapse_score")))
        log_cross_modal(
            f"correction_score_mode={correction_score_mode}, "
            f"correction_fusion_mode={effective_correction_fusion_mode}"
        )
        log_cross_modal(
            "alpha T->I stats: "
            f"before(mean={correction_diagnostics.get('alpha_t2i_stats_before_gate', {}).get('mean', 0.0):.4f}, "
            f"std={correction_diagnostics.get('alpha_t2i_stats_before_gate', {}).get('std', 0.0):.4f}) "
            f"after(mean={correction_diagnostics.get('alpha_t2i_stats_after_gate', {}).get('mean', 0.0):.4f}, "
            f"std={correction_diagnostics.get('alpha_t2i_stats_after_gate', {}).get('std', 0.0):.4f})"
        )
        log_cross_modal(
            "alpha I->T stats: "
            f"before(mean={correction_diagnostics.get('alpha_i2t_stats_before_gate', {}).get('mean', 0.0):.4f}, "
            f"std={correction_diagnostics.get('alpha_i2t_stats_before_gate', {}).get('std', 0.0):.4f}) "
            f"after(mean={correction_diagnostics.get('alpha_i2t_stats_after_gate', {}).get('mean', 0.0):.4f}, "
            f"std={correction_diagnostics.get('alpha_i2t_stats_after_gate', {}).get('std', 0.0):.4f})"
        )
        if correction_score_mode == "confidence":
            log_cross_modal(
                f"local_node_confidence_mode={correction_diagnostics.get('effective_local_node_confidence_mode', 'none')} "
                f"(requested={correction_diagnostics.get('requested_local_node_confidence_mode', 'none')})"
            )
            log_cross_modal(
                "directional gate activation: "
                f"T->I={correction_diagnostics.get('gate_activation_ratio_t2i', 0.0):.4f}, "
                f"I->T={correction_diagnostics.get('gate_activation_ratio_i2t', 0.0):.4f}"
            )
            image_local_diag = correction_diagnostics.get("image_local_confidence_diagnostics", {})
            text_local_diag = correction_diagnostics.get("text_local_confidence_diagnostics", {})
            log_cross_modal(
                "edge confidence stats: "
                f"C^I(mean={correction_diagnostics.get('c_img_stats', {}).get('mean', 0.0):.4f}, "
                f"std={correction_diagnostics.get('c_img_stats', {}).get('std', 0.0):.4f}) "
                f"C^T(mean={correction_diagnostics.get('c_txt_stats', {}).get('mean', 0.0):.4f}, "
                f"std={correction_diagnostics.get('c_txt_stats', {}).get('std', 0.0):.4f})"
            )
            log_cross_modal(
                "image local views: "
                f"entropy(mean={image_local_diag.get('entropy_view_stats', {}).get('mean', 0.0):.4f}) "
                f"agreement(mean={image_local_diag.get('agreement_view_stats', {}).get('mean', 0.0):.4f}) "
                f"diffusion(mean={image_local_diag.get('diffusion_view_stats', {}).get('mean', 0.0):.4f})"
            )
            log_cross_modal(
                "text local views: "
                f"entropy(mean={text_local_diag.get('entropy_view_stats', {}).get('mean', 0.0):.4f}) "
                f"agreement(mean={text_local_diag.get('agreement_view_stats', {}).get('mean', 0.0):.4f}) "
                f"diffusion(mean={text_local_diag.get('diffusion_view_stats', {}).get('mean', 0.0):.4f})"
            )
        else:
            collapse_weights = correction_diagnostics.get("collapse_score_weights", {})
            log_cross_modal(
                "collapse-score config: "
                f"mode={correction_diagnostics.get('collapse_score_mode', 'edge_plus_neighborhood')}, "
                f"topk={correction_diagnostics.get('collapse_neighbor_topk', 10)}, "
                f"weights=edge:{collapse_weights.get('edge', 1.0):.4f}, "
                f"a2b:{collapse_weights.get('a2b', 1.0):.4f}, "
                f"b2a:{collapse_weights.get('b2a', 1.0):.4f}, "
                f"nbr2nbr:{collapse_weights.get('nbr2nbr', 1.0):.4f}"
            )
            log_cross_modal(
                "q stats: "
                f"Q^I(mean={correction_diagnostics.get('image_q_stats', {}).get('mean', 0.0):.4f}, "
                f"std={correction_diagnostics.get('image_q_stats', {}).get('std', 0.0):.4f}) "
                f"Q^T(mean={correction_diagnostics.get('text_q_stats', {}).get('mean', 0.0):.4f}, "
                f"std={correction_diagnostics.get('text_q_stats', {}).get('std', 0.0):.4f})"
            )
            image_q_components = correction_diagnostics.get("image_q_components", {})
            text_q_components = correction_diagnostics.get("text_q_components", {})
            log_cross_modal(
                "image q components: "
                f"edge(mean={image_q_components.get('edge', {}).get('normalized_stats', {}).get('mean', 0.0):.4f}) "
                f"a2b(mean={image_q_components.get('a2b', {}).get('normalized_stats', {}).get('mean', 0.0):.4f}) "
                f"b2a(mean={image_q_components.get('b2a', {}).get('normalized_stats', {}).get('mean', 0.0):.4f}) "
                f"nbr2nbr(mean={image_q_components.get('nbr2nbr', {}).get('normalized_stats', {}).get('mean', 0.0):.4f})"
            )
            log_cross_modal(
                "text q components: "
                f"edge(mean={text_q_components.get('edge', {}).get('normalized_stats', {}).get('mean', 0.0):.4f}) "
                f"a2b(mean={text_q_components.get('a2b', {}).get('normalized_stats', {}).get('mean', 0.0):.4f}) "
                f"b2a(mean={text_q_components.get('b2a', {}).get('normalized_stats', {}).get('mean', 0.0):.4f}) "
                f"nbr2nbr(mean={text_q_components.get('nbr2nbr', {}).get('normalized_stats', {}).get('mean', 0.0):.4f})"
            )
        if effective_correction_fusion_mode == "legacy":
            log_cross_modal("legacy correction_fusion_mode detected: bypassing thresholded alpha gate during correction and reusing original confidence-aware fusion")
            alpha_txt_to_img_used = alpha_txt_to_img
            alpha_img_to_txt_used = alpha_img_to_txt
        else:
            alpha_txt_to_img_used = alpha_txt_to_img_eff
            alpha_img_to_txt_used = alpha_img_to_txt_eff
        if correction_diagnostics.get("image_kappa_stats") is not None and correction_diagnostics.get("text_kappa_stats") is not None:
            image_kappa_stats = correction_diagnostics.get("image_kappa_stats")
            text_kappa_stats = correction_diagnostics.get("text_kappa_stats")
            log_cross_modal(
                "local node confidence stats: "
                f"image(mean={image_kappa_stats['mean']:.4f}, std={image_kappa_stats['std']:.4f}, "
                f"min={image_kappa_stats['min']:.4f}, max={image_kappa_stats['max']:.4f}), "
                f"text(mean={text_kappa_stats['mean']:.4f}, std={text_kappa_stats['std']:.4f}, "
                f"min={text_kappa_stats['min']:.4f}, max={text_kappa_stats['max']:.4f})"
            )
        else:
            log_cross_modal(
                str(
                    correction_diagnostics.get(
                        "edge_confidence_note",
                        correction_diagnostics.get("edge_collapse_note", "bidirectional correction diagnostics recorded"),
                    )
                )
            )
        (
            corrected_image_directed,
            corrected_image_symmetric,
            corrected_text_directed,
            corrected_text_symmetric,
        ) = apply_bidirectional_correction(
            image_bundle["graph"],
            text_bundle["graph"],
            alpha_txt_to_img_used,
            alpha_img_to_txt_used,
        )
    else:
        raise ValueError(f"Unsupported correction mode: {correction_mode}")

    log_cross_modal("recomputing corrected modality collapse scores")
    corrected_image_collapse_metrics = compute_graph_collapse_metrics(
        corrected_image_symmetric,
        num_eigs=getattr(args, "num_eigs", 64),
        spectrum_solver_mode=getattr(args, "spectrum_solver_mode", "normalized_adjacency_largest"),
    )
    corrected_text_collapse_metrics = compute_graph_collapse_metrics(
        corrected_text_symmetric,
        num_eigs=getattr(args, "num_eigs", 64),
        spectrum_solver_mode=getattr(args, "spectrum_solver_mode", "normalized_adjacency_largest"),
    )
    log_cross_modal(
        "corrected graph collapse: "
        f"A(collapse={corrected_image_collapse_metrics.get('collapse_score', 1.0):.4f}, "
        f"entropy={corrected_image_collapse_metrics.get('spectral_entropy', 0.0):.4f}) "
        f"B(collapse={corrected_text_collapse_metrics.get('collapse_score', 1.0):.4f}, "
        f"entropy={corrected_text_collapse_metrics.get('spectral_entropy', 0.0):.4f})"
    )

    fusion_domain_mode = str(getattr(args, "fusion_domain_mode", "wavelet_latent"))
    log_cross_modal(f"building unified topology B*: fusion_domain_mode={fusion_domain_mode}, graph_fusion_mode={args.fusion_mode}")
    unified_graph, fusion_diagnostics = unify_topology(
        corrected_image_symmetric,
        corrected_text_symmetric,
        fusion_domain_mode=fusion_domain_mode,
        mode=args.fusion_mode,
        correction_fusion_mode=effective_correction_fusion_mode,
        rho_img=rho_img,
        rho_txt=rho_txt,
        lambda_f=getattr(args, "lambda_f", 1.0),
        mu_f=getattr(args, "mu_f", 1.0),
        eps=getattr(args, "fusion_eps", 1e-8),
        alpha_txt_to_img_eff=alpha_txt_to_img_eff if correction_mode == "bidirectional" else None,
        alpha_img_to_txt_eff=alpha_img_to_txt_eff if correction_mode == "bidirectional" else None,
        wavelet_fusion_scales=getattr(args, "wavelet_fusion_scales", "1,2,4"),
        wavelet_fusion_impl=getattr(args, "wavelet_fusion_impl", "diffusion_difference"),
        wavelet_fusion_probe_mode=getattr(args, "wavelet_fusion_probe_mode", "random"),
        wavelet_fusion_probe_dim=getattr(args, "wavelet_fusion_probe_dim", 32),
        wavelet_fusion_weight_mode=getattr(args, "wavelet_fusion_weight_mode", "fixed_per_scale"),
        wavelet_fusion_weight_a_scales=getattr(args, "wavelet_fusion_weight_a_scales", None),
        wavelet_fusion_weight_b_scales=getattr(args, "wavelet_fusion_weight_b_scales", None),
        modal_graph_collapse_a=corrected_image_collapse_metrics,
        modal_graph_collapse_b=corrected_text_collapse_metrics,
        wavelet_latent_lambda_sparse=getattr(args, "wavelet_latent_lambda_sparse", 0.01),
        wavelet_latent_lambda_sym=getattr(args, "wavelet_latent_lambda_sym", 1.0),
        wavelet_latent_lambda_nonneg=getattr(args, "wavelet_latent_lambda_nonneg", 1.0),
        wavelet_latent_reconstruction_mode=getattr(args, "wavelet_latent_reconstruction_mode", "candidate_response_similarity"),
        wavelet_latent_postprocess_topk=getattr(args, "wavelet_latent_postprocess_topk", 64),
        wavelet_latent_postprocess_threshold=getattr(args, "wavelet_latent_postprocess_threshold", 0.0),
    )
    if fusion_diagnostics:
        if fusion_domain_mode == "graph":
            log_cross_modal(
                "autonomy fusion omega stats: "
                f"omega^I(mean={fusion_diagnostics.get('omega_img_stats', {}).get('mean', 0.0):.4f}, "
                f"std={fusion_diagnostics.get('omega_img_stats', {}).get('std', 0.0):.4f}), "
                f"omega^T(mean={fusion_diagnostics.get('omega_txt_stats', {}).get('mean', 0.0):.4f}, "
                f"std={fusion_diagnostics.get('omega_txt_stats', {}).get('std', 0.0):.4f})"
            )
        else:
            log_cross_modal(
                "wavelet_latent fusion stats: "
                f"impl={fusion_diagnostics.get('wavelet_fusion_impl', 'diffusion_difference')}, "
                f"probe_mode={fusion_diagnostics.get('wavelet_fusion_probe_mode', 'random')}, "
                f"probe_dim={fusion_diagnostics.get('wavelet_fusion_probe_dim', 0)}, "
                f"candidate_edges={fusion_diagnostics.get('latent_candidate_edge_count', 0)}, "
                f"pre_sparsify_nnz={fusion_diagnostics.get('pre_sparsify_reconstructed_nnz', 0)}, "
                f"post_sparsify_nnz={fusion_diagnostics.get('post_sparsify_reconstructed_nnz', 0)}"
            )
            for scale_key, scale_diag in fusion_diagnostics.get("per_scale_stats", {}).items():
                log_cross_modal(
                    f"wavelet_latent scale={scale_key}: "
                    f"omega_A={scale_diag.get('omega_A', 0.0):.4f}, omega_B={scale_diag.get('omega_B', 0.0):.4f}, "
                    f"W_A(mean={scale_diag.get('A', {}).get('mean', 0.0):.4f}, row_norm={scale_diag.get('A', {}).get('mean_row_norm', 0.0):.4f}), "
                    f"W_B(mean={scale_diag.get('B', {}).get('mean', 0.0):.4f}, row_norm={scale_diag.get('B', {}).get('mean_row_norm', 0.0):.4f}), "
                    f"W_target(mean={scale_diag.get('target', {}).get('mean', 0.0):.4f}, row_norm={scale_diag.get('target', {}).get('mean_row_norm', 0.0):.4f}), "
                    f"W_star(mean={scale_diag.get('star', {}).get('mean', 0.0):.4f}, row_norm={scale_diag.get('star', {}).get('mean_row_norm', 0.0):.4f})"
                )
            if "collapse_aware_graph_stats" in fusion_diagnostics:
                graph_diag = fusion_diagnostics.get("collapse_aware_graph_stats", {})
                log_cross_modal(
                    "wavelet_latent corrected-graph collapse-aware weights: "
                    f"omega_A={graph_diag.get('omega_A', 0.0):.4f}, omega_B={graph_diag.get('omega_B', 0.0):.4f}, "
                    f"A(collapse={graph_diag.get('A', {}).get('collapse_score', 0.0):.4f}, entropy={graph_diag.get('A', {}).get('spectral_entropy', 0.0):.4f}), "
                    f"B(collapse={graph_diag.get('B', {}).get('collapse_score', 0.0):.4f}, entropy={graph_diag.get('B', {}).get('spectral_entropy', 0.0):.4f})"
                )
            elif "collapse_aware_scale_stats" in fusion_diagnostics:
                for scale_key, scale_diag in fusion_diagnostics.get("collapse_aware_scale_stats", {}).items():
                    if str(scale_key).startswith("_"):
                        continue
                    log_cross_modal(
                        f"wavelet_latent collapse-aware scale={scale_key}: "
                        f"A(collapse={scale_diag.get('A', {}).get('collapse_score', 0.0):.4f}, entropy={scale_diag.get('A', {}).get('spectral_entropy', 0.0):.4f}), "
                        f"B(collapse={scale_diag.get('B', {}).get('collapse_score', 0.0):.4f}, entropy={scale_diag.get('B', {}).get('spectral_entropy', 0.0):.4f})"
                    )
            log_cross_modal(
                "wavelet_latent graph stats: "
                f"pre_sym_nnz={fusion_diagnostics.get('pre_sym_directed_nnz', 0)}, "
                f"pre_projection_nnz={fusion_diagnostics.get('pre_projection_graph_stats', {}).get('nnz', 0)}, "
                f"post_projection_nnz={fusion_diagnostics.get('post_projection_graph_stats', {}).get('nnz', 0)}, "
                f"final_nnz={fusion_diagnostics.get('final_graph_stats', {}).get('nnz', 0)}"
            )

    log_cross_modal("summarizing corrected and unified graphs")
    corrected_image_summary = summarize_graph(corrected_image_symmetric, collapse_metrics=corrected_image_collapse_metrics)
    corrected_text_summary = summarize_graph(corrected_text_symmetric, collapse_metrics=corrected_text_collapse_metrics)
    corrected_summary = corrected_text_summary if healthy_modality == "image" else corrected_image_summary
    unified_summary = summarize_graph(unified_graph)
    unified_spectral_artifacts = build_unified_spectral_artifacts(
        unified_graph,
        num_eigs=getattr(args, "num_eigs", 64),
        embedding_dim=getattr(args, "spectral_embedding_dim", 32),
        save_eigenvectors=bool(getattr(args, "save_eigenvectors", False)),
        spectrum_solver_mode=getattr(args, "spectrum_solver_mode", "normalized_adjacency_largest"),
        embedding_type=getattr(args, "embedding_type", "diffusion"),
        diffusion_dim=getattr(args, "diffusion_dim", None),
        diffusion_time=getattr(args, "diffusion_time", 1.0),
        diffusion_eig_solver=getattr(args, "diffusion_eig_solver", "auto"),
    )
    summary = build_summary(
        args,
        healthy_modality,
        image_bundle,
        text_bundle,
        rho_img,
        rho_txt,
        image_kappa_stats,
        text_kappa_stats,
        corrected_image_summary,
        corrected_text_summary,
        corrected_summary,
        unified_summary,
        unified_spectral_artifacts,
        correction_diagnostics,
        fusion_diagnostics,
        effective_correction_fusion_mode,
    )

    output_dir = build_output_dir(args)
    save_cross_modal_outputs(
        output_dir,
        healthy_modality,
        image_bundle,
        text_bundle,
        corrected_image_directed,
        corrected_image_symmetric,
        corrected_text_directed,
        corrected_text_symmetric,
        unified_graph,
        unified_spectral_artifacts,
        summary,
    )
    log_cross_modal("cross-modal topology completed")

    return {
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "summary": summary,
    }
