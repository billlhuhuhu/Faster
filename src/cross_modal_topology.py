import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from src.topology_graph import build_laplacian, build_spectral_embedding, compute_spectrum, parse_multi_scale_ks


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


def summarize_graph(graph):
    num_nodes = int(graph.shape[0])
    num_edges = int(graph.nnz)
    avg_degree = float(num_edges / max(num_nodes, 1))
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    nonzero_degree = degree[degree > 0]
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "min_degree": float(degree.min()) if degree.size else 0.0,
        "max_degree": float(degree.max()) if degree.size else 0.0,
        "mean_nonzero_degree": float(nonzero_degree.mean()) if nonzero_degree.size else 0.0,
        "density": float(num_edges / max(num_nodes * num_nodes, 1)),
    }


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
        "fusion_mode": args.fusion_mode,
        "lambda_f": float(getattr(args, "lambda_f", 1.0)),
        "mu_f": float(getattr(args, "mu_f", 1.0)),
        "fusion_eps": float(getattr(args, "fusion_eps", 1e-8)),
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
            enable_directional_correction_gate=bool(getattr(args, "enable_directional_correction_gate", True)),
            correction_gate_tau_high=getattr(args, "correction_gate_tau_high", 0.6),
            correction_gate_tau_low=getattr(args, "correction_gate_tau_low", 0.3),
            correction_gate_tau_gap=getattr(args, "correction_gate_tau_gap", 0.15),
        )
        log_cross_modal(
            f"correction_fusion_mode={effective_correction_fusion_mode}, "
            f"enable_directional_correction_gate={bool(getattr(args, 'enable_directional_correction_gate', True))}, "
            f"tau_high={float(getattr(args, 'correction_gate_tau_high', 0.6)):.4f}, "
            f"tau_low={float(getattr(args, 'correction_gate_tau_low', 0.3)):.4f}, "
            f"tau_gap={float(getattr(args, 'correction_gate_tau_gap', 0.15)):.4f}"
        )
        log_cross_modal(
            "directional gate activation: "
            f"T->I={correction_diagnostics.get('gate_activation_ratio_t2i', 0.0):.4f}, "
            f"I->T={correction_diagnostics.get('gate_activation_ratio_i2t', 0.0):.4f}"
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
        log_cross_modal(
            f"local_node_confidence_mode={correction_diagnostics.get('effective_local_node_confidence_mode', 'none')} "
            f"(requested={correction_diagnostics.get('requested_local_node_confidence_mode', 'none')})"
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
            log_cross_modal(str(correction_diagnostics.get("edge_confidence_note", "mode=none uses C=rho*P")))
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

    log_cross_modal("building unified topology B*")
    unified_graph, fusion_diagnostics = unify_topology(
        corrected_image_symmetric,
        corrected_text_symmetric,
        mode=args.fusion_mode,
        correction_fusion_mode=effective_correction_fusion_mode,
        rho_img=rho_img,
        rho_txt=rho_txt,
        lambda_f=getattr(args, "lambda_f", 1.0),
        mu_f=getattr(args, "mu_f", 1.0),
        eps=getattr(args, "fusion_eps", 1e-8),
        alpha_txt_to_img_eff=alpha_txt_to_img_eff if correction_mode == "bidirectional" else None,
        alpha_img_to_txt_eff=alpha_img_to_txt_eff if correction_mode == "bidirectional" else None,
    )
    if fusion_diagnostics:
        log_cross_modal(
            "autonomy fusion omega stats: "
            f"omega^I(mean={fusion_diagnostics.get('omega_img_stats', {}).get('mean', 0.0):.4f}, "
            f"std={fusion_diagnostics.get('omega_img_stats', {}).get('std', 0.0):.4f}), "
            f"omega^T(mean={fusion_diagnostics.get('omega_txt_stats', {}).get('mean', 0.0):.4f}, "
            f"std={fusion_diagnostics.get('omega_txt_stats', {}).get('std', 0.0):.4f})"
        )

    log_cross_modal("summarizing corrected and unified graphs")
    corrected_image_summary = summarize_graph(corrected_image_symmetric)
    corrected_text_summary = summarize_graph(corrected_text_symmetric)
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
