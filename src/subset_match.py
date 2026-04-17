import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from data.subset_dataset import save_selected_indices
from src.graph_wavelet import build_multi_scale_wavelet_signatures, parse_wavelet_scales
from src.proxy_optimization import l2_normalize, optimize_proxy_points, resolve_proxy_loss_type


def sanitize_name(name):
    return name.replace("\\", "-").replace("/", "-").replace(" ", "_")


def build_feature_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    return Path(args.feature_cache_root) / args.dataset / args.split / model_tag


def build_cross_modal_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    fusion_tag = f"k{args.k}_{sanitize_name(args.metric)}_a{sanitize_name(str(args.alpha))}"
    return Path(args.cross_modal_root) / args.dataset / args.split / model_tag / fusion_tag


def build_output_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    budget_tag = build_budget_tag(args)
    method_tag = sanitize_name(getattr(args, "selection_method", "baseline"))
    if bool(getattr(args, "keep_lsrc", True)) or bool(getattr(args, "enable_lsrc", False)):
        method_tag = f"{method_tag}_lsrc"
    if getattr(args, "diagnostic_experiment_id", None) is not None:
        method_tag = f"{method_tag}_exp{int(args.diagnostic_experiment_id)}"
    seed_tag = f"seed_{int(getattr(args, 'random_state', 0))}"
    return Path(args.output_root) / args.dataset / args.split / model_tag / budget_tag / method_tag / seed_tag


def load_feature_cache(feature_dir):
    feature_dir = Path(feature_dir)
    img_path = feature_dir / "img_features_selection.pt"
    txt_path = feature_dir / "txt_features_selection.pt"
    if not img_path.exists():
        img_path = feature_dir / "img_features.pt"
    if not txt_path.exists():
        txt_path = feature_dir / "txt_features.pt"
    img_features = torch.load(img_path, map_location="cpu")
    txt_features = torch.load(txt_path, map_location="cpu")
    with open(feature_dir / "sample_meta.json", "r", encoding="utf-8") as handle:
        sample_meta = json.load(handle)

    img_features = img_features.float().cpu().numpy()
    txt_features = txt_features.float().cpu().numpy()
    return img_features, txt_features, sample_meta


def load_unified_artifacts(cross_modal_dir):
    cross_modal_dir = Path(cross_modal_dir)
    unified_graph = sparse.load_npz(cross_modal_dir / "unified_graph.npz").tocsr()
    summary = json.load(open(cross_modal_dir / "summary.json", "r", encoding="utf-8"))
    spectral_path = cross_modal_dir / "unified_spectral_embedding.npy"
    spectral_embedding = None
    if spectral_path.exists():
        spectral_embedding = np.load(spectral_path).astype(np.float32)
    image_graph = None
    text_graph = None
    image_graph_path = cross_modal_dir / "corrected_image_graph_symmetric.npz"
    text_graph_path = cross_modal_dir / "corrected_text_graph_symmetric.npz"
    if image_graph_path.exists():
        image_graph = sparse.load_npz(image_graph_path).tocsr()
    if text_graph_path.exists():
        text_graph = sparse.load_npz(text_graph_path).tocsr()
    return unified_graph, spectral_embedding, summary, image_graph, text_graph


def build_unified_representation(img_features, txt_features, mode="concat"):
    if mode != "concat":
        raise ValueError(f"Unsupported representation mode: {mode}")
    img_norm = l2_normalize(img_features.astype(np.float32))
    txt_norm = l2_normalize(txt_features.astype(np.float32))
    representation = np.concatenate([img_norm, txt_norm], axis=1).astype(np.float32)
    return representation


def build_reference_embedding(representation, spectral_embedding=None, mode="hybrid", spectral_weight=1.0):
    representation = l2_normalize(np.asarray(representation, dtype=np.float32))
    if mode == "concat" or spectral_embedding is None:
        return representation.astype(np.float32)

    spectral_embedding = l2_normalize(np.asarray(spectral_embedding, dtype=np.float32))
    if mode == "spectral":
        return spectral_embedding.astype(np.float32)
    if mode == "hybrid":
        hybrid = np.concatenate([representation, float(spectral_weight) * spectral_embedding], axis=1)
        return l2_normalize(hybrid.astype(np.float32))
    raise ValueError(f"Unsupported reference embedding mode: {mode}")


def resolve_subset_size(num_samples, budget_ratio=None, budget_size=None):
    if budget_size is not None:
        subset_size = int(budget_size)
    elif budget_ratio is not None:
        subset_size = int(round(num_samples * float(budget_ratio)))
    else:
        raise ValueError("Either budget_ratio or budget_size must be provided.")
    subset_size = max(1, subset_size)
    subset_size = min(num_samples, subset_size)
    return subset_size


def build_budget_tag(args):
    budget_size = getattr(args, "budget_size", None)
    budget_ratio = getattr(args, "budget_ratio", None)
    if budget_size is not None:
        return f"size_{int(budget_size):04d}"
    if budget_ratio is not None:
        return f"ratio_{int(round(float(budget_ratio) * 100)):02d}"
    raise ValueError("Either budget_ratio or budget_size must be provided.")


def compute_graph_scores(unified_graph):
    degree = np.asarray(unified_graph.sum(axis=1)).reshape(-1).astype(np.float32)
    if degree.size == 0:
        return degree
    max_degree = float(degree.max())
    if max_degree <= 0:
        return np.zeros_like(degree)
    return degree / max_degree


def compute_graph_degree(unified_graph, eps=1e-8):
    degree = np.asarray(unified_graph.sum(axis=1)).reshape(-1).astype(np.float32)
    return np.maximum(degree, float(eps))


def fit_proxy_centers(representation, subset_size, random_state=0, use_minibatch=False, batch_size=2048):
    if use_minibatch:
        clusterer = MiniBatchKMeans(
            n_clusters=subset_size,
            random_state=random_state,
            batch_size=batch_size,
            n_init=10,
        )
    else:
        clusterer = KMeans(
            n_clusters=subset_size,
            random_state=random_state,
            n_init=10,
        )
    labels = clusterer.fit_predict(representation)
    centers = clusterer.cluster_centers_.astype(np.float32)
    return centers, labels


def rank_candidates_for_center(representation, center, graph_scores, degree_weight=0.1):
    distances = np.linalg.norm(representation - center[None, :], axis=1).astype(np.float32)
    adjusted = distances - degree_weight * graph_scores
    order = np.argsort(adjusted, kind="stable")
    return order, distances


def select_representatives(representation, centers, graph_scores, degree_weight=0.1):
    selected = []
    selected_set = set()
    candidate_orders = []

    for center in centers:
        order, distances = rank_candidates_for_center(representation, center, graph_scores, degree_weight=degree_weight)
        candidate_orders.append((order, distances))

    for order, _ in candidate_orders:
        chosen = None
        for idx in order:
            idx = int(idx)
            if idx not in selected_set:
                chosen = idx
                break
        if chosen is None:
            raise RuntimeError("Failed to choose a unique representative for one proxy center.")
        selected.append(chosen)
        selected_set.add(chosen)

    return selected


def row_normalize_graph(graph):
    graph = graph.tocsr().astype(np.float32)
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1)
    row_sum = np.maximum(row_sum, 1e-12)
    inv_row = sparse.diags(1.0 / row_sum.astype(np.float32))
    return inv_row @ graph


def build_topology_targets(unified_graph, representation, hop_weight=0.5):
    transition = row_normalize_graph(unified_graph)
    neighbor_context = transition @ representation
    topology_targets = (1.0 - float(hop_weight)) * representation + float(hop_weight) * neighbor_context
    return l2_normalize(topology_targets.astype(np.float32))


def summarize_numeric(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_topology_node_cost(unified_graph):
    graph_scores = compute_graph_scores(unified_graph)
    topo_cost = 1.0 - graph_scores.astype(np.float32)
    return topo_cost.astype(np.float32), graph_scores.astype(np.float32)


def resolve_wavelet_signature_bundle(proxy_bundle, unified_graph, representation, top_k=8):
    summary = proxy_bundle.get("summary", {})
    if not bool(summary.get("use_wavelet_multiscale", False)):
        return None
    scales = parse_wavelet_scales(summary.get("wavelet_scales", [1, 2, 4]))
    history = summary.get("history", [])
    final_history = history[-1] if history else {}
    active_scales = final_history.get("active_scales", scales)
    if not active_scales:
        active_scales = scales
    raw_weights = final_history.get("wavelet_scale_weights", {})
    if raw_weights:
        scale_weights = {int(scale): float(raw_weights.get(str(scale), raw_weights.get(scale, 0.0))) for scale in active_scales}
    else:
        default_weight = 1.0 / max(len(active_scales), 1)
        scale_weights = {int(scale): float(default_weight) for scale in active_scales}

    all_signatures = build_multi_scale_wavelet_signatures(unified_graph, representation, scales, normalize=True)
    node_signature = np.zeros_like(next(iter(all_signatures.values())), dtype=np.float32)
    for scale in active_scales:
        node_signature += float(scale_weights[int(scale)]) * all_signatures[int(scale)]

    proxy_points = np.asarray(proxy_bundle["proxy_points"], dtype=np.float32)
    reference_points = np.asarray(representation, dtype=np.float32)
    top_k = max(1, min(int(top_k), int(reference_points.shape[0])))
    knn = NearestNeighbors(n_neighbors=top_k, metric="euclidean")
    knn.fit(reference_points)
    distances, indices = knn.kneighbors(proxy_points)
    sqdist = (distances.astype(np.float32)) ** 2
    positive = sqdist[sqdist > 0]
    tau = float(np.median(positive)) if positive.size > 0 else 1.0
    tau = max(tau, 1e-6)
    logits = -sqdist / tau
    logits = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits).astype(np.float32)
    weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
    proxy_signature = np.sum(weights[..., None] * node_signature[indices], axis=1).astype(np.float32)

    return {
        "node_signature": node_signature.astype(np.float32),
        "proxy_signature": proxy_signature.astype(np.float32),
        "active_scales": [int(scale) for scale in active_scales],
        "scale_weights": {str(scale): float(scale_weights[int(scale)]) for scale in active_scales},
        "interpolation_top_k": int(top_k),
        "interpolation_tau": float(tau),
    }


def compute_candidate_neighbors(representation, proxy_points, top_k):
    top_k = max(1, min(int(top_k), int(representation.shape[0])))
    knn = NearestNeighbors(n_neighbors=top_k, metric="euclidean")
    knn.fit(representation)
    distances, indices = knn.kneighbors(proxy_points)
    return distances.astype(np.float32), indices.astype(np.int64)


def compute_proxy_sample_costs(
    proxy_points,
    candidate_points,
    candidate_topo_cost,
    proxy_wavelet_signatures=None,
    candidate_wavelet_signatures=None,
    candidate_q=None,
    cost_alpha_diff=1.0,
    cost_beta_wavelet=0.25,
    cost_gamma_topo=0.1,
    cost_eta_lsrc=0.1,
):
    diffusion_cost = np.sum((proxy_points - candidate_points) ** 2, axis=-1).astype(np.float32)
    if proxy_wavelet_signatures is not None and candidate_wavelet_signatures is not None:
        wavelet_cost = np.sum((proxy_wavelet_signatures - candidate_wavelet_signatures) ** 2, axis=-1).astype(np.float32)
    else:
        wavelet_cost = np.zeros_like(diffusion_cost, dtype=np.float32)
    topology_cost = candidate_topo_cost.astype(np.float32)
    if candidate_q is None:
        lsrc_reward = np.zeros_like(diffusion_cost, dtype=np.float32)
    else:
        lsrc_reward = candidate_q.astype(np.float32)
    total_cost = (
        float(cost_alpha_diff) * diffusion_cost
        + float(cost_beta_wavelet) * wavelet_cost
        + float(cost_gamma_topo) * topology_cost
        - float(cost_eta_lsrc) * lsrc_reward
    )
    return total_cost.astype(np.float32), diffusion_cost, wavelet_cost, topology_cost.astype(np.float32), lsrc_reward.astype(np.float32)


def build_degree_aware_cost_matrix(
    proxy_points,
    reference_points,
    unified_graph,
    topo_cost,
    wavelet_bundle=None,
    lsrc_confidence=None,
    cost_alpha_diff=1.0,
    cost_beta_wavelet=0.25,
    cost_gamma_topo=0.1,
    cost_eta_lsrc=0.1,
    batch_size=64,
    eps=1e-8,
):
    num_proxies = int(proxy_points.shape[0])
    num_nodes = int(reference_points.shape[0])
    diffusion_cost = np.empty((num_proxies, num_nodes), dtype=np.float32)
    wavelet_cost = np.empty((num_proxies, num_nodes), dtype=np.float32)
    topo_matrix = np.broadcast_to(np.asarray(topo_cost, dtype=np.float32)[None, :], (num_proxies, num_nodes)).copy()
    lsrc_reward = np.broadcast_to(np.asarray(lsrc_confidence if lsrc_confidence is not None else np.zeros(num_nodes, dtype=np.float32), dtype=np.float32)[None, :], (num_proxies, num_nodes)).copy()

    for start in range(0, num_proxies, int(batch_size)):
        end = min(num_proxies, start + int(batch_size))
        batch_proxy = proxy_points[start:end, None, :]
        batch_candidates = reference_points[None, :, :]
        batch_proxy_wavelet = None
        batch_candidate_wavelet = None
        if wavelet_bundle is not None:
            batch_proxy_wavelet = wavelet_bundle["proxy_signature"][start:end, None, :]
            batch_candidate_wavelet = wavelet_bundle["node_signature"][None, :, :]
        _, batch_diff, batch_wavelet, _, _ = compute_proxy_sample_costs(
            batch_proxy,
            batch_candidates,
            topo_matrix[start:end],
            proxy_wavelet_signatures=batch_proxy_wavelet,
            candidate_wavelet_signatures=batch_candidate_wavelet,
            candidate_q=lsrc_reward[start:end],
            cost_alpha_diff=cost_alpha_diff,
            cost_beta_wavelet=cost_beta_wavelet,
            cost_gamma_topo=cost_gamma_topo,
            cost_eta_lsrc=cost_eta_lsrc,
        )
        diffusion_cost[start:end] = batch_diff
        wavelet_cost[start:end] = batch_wavelet

    degree = compute_graph_degree(unified_graph, eps=eps)
    total_cost = (
        float(cost_alpha_diff) * diffusion_cost
        + float(cost_beta_wavelet) * wavelet_cost
        + float(cost_gamma_topo) * topo_matrix
        - float(cost_eta_lsrc) * lsrc_reward
    )
    return {
        "total_cost": total_cost.astype(np.float32),
        "diffusion_cost": diffusion_cost.astype(np.float32),
        "wavelet_cost": wavelet_cost.astype(np.float32),
        "topo_cost": topo_matrix.astype(np.float32),
        "lsrc_reward": lsrc_reward.astype(np.float32),
        "geometry_cost": diffusion_cost.astype(np.float32),
        "topology_cost": topo_matrix.astype(np.float32),
        "degree": degree.astype(np.float32),
        "cost_component_stats": {
            "total_cost": summarize_numeric(total_cost),
            "diffusion_cost": summarize_numeric(diffusion_cost),
            "wavelet_cost": summarize_numeric(wavelet_cost),
            "topo_cost": summarize_numeric(topo_matrix),
            "lsrc_reward": summarize_numeric(lsrc_reward),
        },
    }


def build_candidate_costs(
    proxy_points,
    representation,
    topo_cost,
    candidate_indices,
    wavelet_bundle=None,
    lsrc_confidence=None,
    cost_alpha_diff=1.0,
    cost_beta_wavelet=0.25,
    cost_gamma_topo=0.1,
    cost_eta_lsrc=0.1,
    batch_size=128,
):
    num_proxies, top_k = candidate_indices.shape
    total_cost = np.empty((num_proxies, top_k), dtype=np.float32)
    diffusion_cost = np.empty((num_proxies, top_k), dtype=np.float32)
    wavelet_cost = np.empty((num_proxies, top_k), dtype=np.float32)
    topology_cost = np.empty((num_proxies, top_k), dtype=np.float32)
    lsrc_reward = np.empty((num_proxies, top_k), dtype=np.float32)

    for start in range(0, num_proxies, int(batch_size)):
        end = min(num_proxies, start + int(batch_size))
        batch_candidates = candidate_indices[start:end]
        batch_proxy = proxy_points[start:end, None, :]
        batch_repr = representation[batch_candidates]
        batch_topo_cost = topo_cost[batch_candidates]
        batch_q = None if lsrc_confidence is None else lsrc_confidence[batch_candidates]
        batch_proxy_wavelet = None
        batch_candidate_wavelet = None
        if wavelet_bundle is not None:
            batch_proxy_wavelet = wavelet_bundle["proxy_signature"][start:end, None, :]
            batch_candidate_wavelet = wavelet_bundle["node_signature"][batch_candidates]
        batch_total, batch_diff, batch_wavelet, batch_topo, batch_reward = compute_proxy_sample_costs(
            batch_proxy,
            batch_repr,
            batch_topo_cost,
            proxy_wavelet_signatures=batch_proxy_wavelet,
            candidate_wavelet_signatures=batch_candidate_wavelet,
            candidate_q=batch_q,
            cost_alpha_diff=cost_alpha_diff,
            cost_beta_wavelet=cost_beta_wavelet,
            cost_gamma_topo=cost_gamma_topo,
            cost_eta_lsrc=cost_eta_lsrc,
        )
        total_cost[start:end] = batch_total
        diffusion_cost[start:end] = batch_diff
        wavelet_cost[start:end] = batch_wavelet
        topology_cost[start:end] = batch_topo
        lsrc_reward[start:end] = batch_reward

    return {
        "total_cost": total_cost,
        "diffusion_cost": diffusion_cost,
        "wavelet_cost": wavelet_cost,
        "topo_cost": topology_cost,
        "lsrc_reward": lsrc_reward,
        "geometry_cost": diffusion_cost,
        "topology_cost": topology_cost,
        "cost_component_stats": {
            "total_cost": summarize_numeric(total_cost),
            "diffusion_cost": summarize_numeric(diffusion_cost),
            "wavelet_cost": summarize_numeric(wavelet_cost),
            "topo_cost": summarize_numeric(topology_cost),
            "lsrc_reward": summarize_numeric(lsrc_reward),
        },
    }


def build_single_row_cost(
    proxy_point,
    candidate_indices,
    representation,
    topo_cost,
    wavelet_bundle=None,
    lsrc_confidence=None,
    cost_alpha_diff=1.0,
    cost_beta_wavelet=0.25,
    cost_gamma_topo=0.1,
    cost_eta_lsrc=0.1,
    proxy_row_idx=None,
):
    candidate_repr = representation[candidate_indices]
    candidate_topo = topo_cost[candidate_indices]
    candidate_q = None if lsrc_confidence is None else lsrc_confidence[candidate_indices]
    proxy_wavelet = None
    candidate_wavelet = None
    if wavelet_bundle is not None and proxy_row_idx is not None:
        proxy_wavelet = wavelet_bundle["proxy_signature"][int(proxy_row_idx)][None, None, :]
        candidate_wavelet = wavelet_bundle["node_signature"][candidate_indices][None, :, :]
    total_cost, _, _, _, _ = compute_proxy_sample_costs(
        proxy_point[None, :],
        candidate_repr[None, :, :],
        candidate_topo[None, :],
        proxy_wavelet_signatures=proxy_wavelet,
        candidate_wavelet_signatures=candidate_wavelet,
        candidate_q=None if candidate_q is None else candidate_q[None, :],
        cost_alpha_diff=cost_alpha_diff,
        cost_beta_wavelet=cost_beta_wavelet,
        cost_gamma_topo=cost_gamma_topo,
        cost_eta_lsrc=cost_eta_lsrc,
    )
    return total_cost.reshape(-1)


def resolve_duplicate_assignments(
    proxy_points,
    candidate_indices,
    candidate_costs,
    representation,
    topo_cost,
    wavelet_bundle=None,
    lsrc_confidence=None,
    cost_alpha_diff=1.0,
    cost_beta_wavelet=0.25,
    cost_gamma_topo=0.1,
    cost_eta_lsrc=0.1,
):
    num_proxies = candidate_indices.shape[0]
    selected = candidate_indices[np.arange(num_proxies), np.argmin(candidate_costs, axis=1)].astype(np.int64)
    all_indices = np.arange(representation.shape[0], dtype=np.int64)
    iterations = 0
    local_hungarian_calls = 0

    while True:
        unique_vals, counts = np.unique(selected, return_counts=True)
        duplicate_candidates = unique_vals[counts > 1]
        if duplicate_candidates.size == 0:
            break

        conflict_mask = np.isin(selected, duplicate_candidates)
        conflict_rows = np.where(conflict_mask)[0]
        locked_rows = np.where(~conflict_mask)[0]
        locked_candidates = set(int(x) for x in selected[locked_rows].tolist())

        local_candidates = np.unique(candidate_indices[conflict_rows].reshape(-1))
        local_candidates = np.array([idx for idx in local_candidates.tolist() if int(idx) not in locked_candidates], dtype=np.int64)

        if local_candidates.size < conflict_rows.size:
            needed = int(conflict_rows.size - local_candidates.size)
            extras = np.array(
                [idx for idx in all_indices.tolist() if idx not in locked_candidates and idx not in set(local_candidates.tolist())][:needed],
                dtype=np.int64,
            )
            if extras.size > 0:
                local_candidates = np.unique(np.concatenate([local_candidates, extras], axis=0))

        if local_candidates.size < conflict_rows.size:
            raise RuntimeError("Not enough candidates to resolve matching conflicts.")

        local_cost = np.full((conflict_rows.size, local_candidates.size), 1e9, dtype=np.float32)
        candidate_to_col = {int(idx): col for col, idx in enumerate(local_candidates.tolist())}

        for local_row, proxy_row in enumerate(conflict_rows.tolist()):
            precomputed = False
            for cand_idx, cost in zip(candidate_indices[proxy_row].tolist(), candidate_costs[proxy_row].tolist()):
                cand_idx = int(cand_idx)
                if cand_idx in candidate_to_col:
                    local_cost[local_row, candidate_to_col[cand_idx]] = float(cost)
                    precomputed = True

            missing_cols = np.where(local_cost[local_row] >= 1e9)[0]
            if missing_cols.size > 0:
                missing_candidates = local_candidates[missing_cols]
                extra_cost = build_single_row_cost(
                    proxy_points[proxy_row],
                missing_candidates,
                representation,
                topo_cost,
                wavelet_bundle=wavelet_bundle,
                lsrc_confidence=lsrc_confidence,
                cost_alpha_diff=cost_alpha_diff,
                cost_beta_wavelet=cost_beta_wavelet,
                cost_gamma_topo=cost_gamma_topo,
                cost_eta_lsrc=cost_eta_lsrc,
                proxy_row_idx=proxy_row,
            )
            local_cost[local_row, missing_cols] = extra_cost

        row_ind, col_ind = linear_sum_assignment(local_cost)
        for row_offset, col_offset in zip(row_ind.tolist(), col_ind.tolist()):
            selected[int(conflict_rows[row_offset])] = int(local_candidates[col_offset])

        iterations += 1
        local_hungarian_calls += 1
        if iterations > max(10, num_proxies):
            raise RuntimeError("Conflict resolution exceeded iteration limit.")

    return selected.astype(np.int64).tolist(), {
        "duplicate_resolution_rounds": int(iterations),
        "local_hungarian_calls": int(local_hungarian_calls),
    }


def run_hungarian_matching(
    proxy_points,
    candidate_indices,
    candidate_costs,
    representation,
    topo_cost,
    wavelet_bundle=None,
    lsrc_confidence=None,
    cost_alpha_diff=1.0,
    cost_beta_wavelet=0.25,
    cost_gamma_topo=0.1,
    cost_eta_lsrc=0.1,
):
    num_proxies = int(candidate_indices.shape[0])
    unique_candidates = np.unique(candidate_indices.reshape(-1)).astype(np.int64)
    if unique_candidates.size < num_proxies:
        fallback_selected, fallback_debug = resolve_duplicate_assignments(
            proxy_points,
            candidate_indices,
            candidate_costs,
            representation,
            topo_cost,
            wavelet_bundle=wavelet_bundle,
            lsrc_confidence=lsrc_confidence,
            cost_alpha_diff=cost_alpha_diff,
            cost_beta_wavelet=cost_beta_wavelet,
            cost_gamma_topo=cost_gamma_topo,
            cost_eta_lsrc=cost_eta_lsrc,
        )
        return {
            "selected_indices": fallback_selected,
            "assignment_mode": "local_conflict_hungarian",
            "diagnostics": fallback_debug,
        }

    candidate_to_col = {int(idx): col for col, idx in enumerate(unique_candidates.tolist())}
    assignment_cost = np.full((num_proxies, unique_candidates.size), 1e9, dtype=np.float32)
    for row_idx in range(num_proxies):
        for cand_idx, cost in zip(candidate_indices[row_idx].tolist(), candidate_costs[row_idx].tolist()):
            assignment_cost[row_idx, candidate_to_col[int(cand_idx)]] = float(cost)

    row_ind, col_ind = linear_sum_assignment(assignment_cost)
    selected_indices = np.full(num_proxies, -1, dtype=np.int64)
    selected_indices[row_ind] = unique_candidates[col_ind]
    return {
        "selected_indices": selected_indices.astype(np.int64).tolist(),
        "assignment_mode": "global_hungarian",
        "diagnostics": {
            "unique_candidate_pool": int(unique_candidates.size),
            "hungarian_rows": int(len(row_ind)),
            "duplicate_resolution_rounds": 0,
            "local_hungarian_calls": 0,
        },
    }


def compute_match_loss(proxy_points, reference_points, selected_indices):
    selected_points = reference_points[np.asarray(selected_indices, dtype=np.int64)]
    geometry_cost = np.sum((proxy_points - selected_points) ** 2, axis=1).astype(np.float32)
    return float(np.mean(geometry_cost)) if geometry_cost.size > 0 else 0.0


def compute_graph_regularization(proxy_points, unified_graph, selected_indices, eps=1e-8):
    if len(selected_indices) == 0:
        return 0.0
    subgraph = unified_graph[np.asarray(selected_indices, dtype=np.int64)][:, np.asarray(selected_indices, dtype=np.int64)].tocsr()
    degree = np.asarray(subgraph.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, float(eps)))
    degree_mat = sparse.diags(inv_sqrt_degree)
    identity = sparse.eye(subgraph.shape[0], dtype=np.float32, format="csr")
    laplacian_sub = identity - degree_mat @ subgraph @ degree_mat
    y_tensor = np.asarray(proxy_points, dtype=np.float32)
    reg = np.trace(y_tensor.T @ laplacian_sub.toarray() @ y_tensor)
    return float(reg)


def run_proxy_matching(
    proxy_points,
    representation,
    topology_targets,
    unified_graph,
    matching_top_k=64,
    degree_weight=0.1,
    topology_weight=0.5,
    geometry_weight=1.0,
    candidate_batch_size=128,
    wavelet_bundle=None,
    lsrc_confidence=None,
    cost_alpha_diff=1.0,
    cost_beta_wavelet=0.25,
    cost_gamma_topo=0.1,
    cost_eta_lsrc=0.1,
):
    topo_cost, graph_scores = build_topology_node_cost(unified_graph)
    _, candidate_indices = compute_candidate_neighbors(representation, proxy_points, top_k=matching_top_k)
    cost_bundle = build_candidate_costs(
        proxy_points,
        representation,
        topo_cost,
        candidate_indices,
        wavelet_bundle=wavelet_bundle,
        lsrc_confidence=lsrc_confidence,
        cost_alpha_diff=cost_alpha_diff,
        cost_beta_wavelet=cost_beta_wavelet,
        cost_gamma_topo=cost_gamma_topo,
        cost_eta_lsrc=cost_eta_lsrc,
        batch_size=candidate_batch_size,
    )
    match_outputs = run_hungarian_matching(
        proxy_points,
        candidate_indices,
        cost_bundle["total_cost"],
        representation,
        topo_cost,
        wavelet_bundle=wavelet_bundle,
        lsrc_confidence=lsrc_confidence,
        cost_alpha_diff=cost_alpha_diff,
        cost_beta_wavelet=cost_beta_wavelet,
        cost_gamma_topo=cost_gamma_topo,
        cost_eta_lsrc=cost_eta_lsrc,
    )
    selected_indices = match_outputs["selected_indices"]
    selected_q = np.asarray(lsrc_confidence, dtype=np.float32)[np.asarray(selected_indices, dtype=np.int64)] if lsrc_confidence is not None else np.zeros(len(selected_indices), dtype=np.float32)
    selected_topo = topo_cost[np.asarray(selected_indices, dtype=np.int64)]
    return {
        "selected_indices": selected_indices,
        "graph_scores": graph_scores,
        "candidate_indices": candidate_indices,
        "cost_bundle": cost_bundle,
        "matching_debug": {
            **match_outputs["diagnostics"],
            "assignment_mode": match_outputs["assignment_mode"],
            "cost_component_stats": cost_bundle.get("cost_component_stats", {}),
            "selected_q_stats": summarize_numeric(selected_q),
            "selected_topo_cost_stats": summarize_numeric(selected_topo),
            "selected_graph_score_stats": summarize_numeric(graph_scores[np.asarray(selected_indices, dtype=np.int64)]),
            "cost_weights": {
                "alpha_diff": float(cost_alpha_diff),
                "beta_wavelet": float(cost_beta_wavelet),
                "matching_wavelet_weight": float(cost_beta_wavelet),
                "gamma_topo": float(cost_gamma_topo),
                "eta_lsrc": float(cost_eta_lsrc),
            },
            "wavelet_active_scales": [] if wavelet_bundle is None else list(wavelet_bundle.get("active_scales", [])),
        },
    }


def run_degree_aware_global_matching(
    proxy_points,
    reference_points,
    unified_graph,
    wavelet_bundle=None,
    lsrc_confidence=None,
    cost_alpha_diff=1.0,
    cost_beta_wavelet=0.25,
    cost_gamma_topo=0.1,
    cost_eta_lsrc=0.1,
):
    topo_cost, graph_scores = build_topology_node_cost(unified_graph)
    cost_bundle = build_degree_aware_cost_matrix(
        proxy_points,
        reference_points,
        unified_graph,
        topo_cost=topo_cost,
        wavelet_bundle=wavelet_bundle,
        lsrc_confidence=lsrc_confidence,
        cost_alpha_diff=cost_alpha_diff,
        cost_beta_wavelet=cost_beta_wavelet,
        cost_gamma_topo=cost_gamma_topo,
        cost_eta_lsrc=cost_eta_lsrc,
    )
    row_ind, col_ind = linear_sum_assignment(cost_bundle["total_cost"])
    selected_indices = np.full(proxy_points.shape[0], -1, dtype=np.int64)
    selected_indices[row_ind] = col_ind
    selected_q = np.asarray(lsrc_confidence, dtype=np.float32)[selected_indices] if lsrc_confidence is not None else np.zeros(len(selected_indices), dtype=np.float32)
    selected_topo = topo_cost[selected_indices]
    return {
        "selected_indices": selected_indices.astype(np.int64).tolist(),
        "candidate_indices": None,
        "cost_bundle": cost_bundle,
        "graph_scores": graph_scores,
        "matching_debug": {
            "assignment_mode": "global_degree_aware_hungarian",
            "duplicate_resolution_rounds": 0,
            "local_hungarian_calls": 0,
            "hungarian_rows": int(len(row_ind)),
            "cost_component_stats": cost_bundle.get("cost_component_stats", {}),
            "selected_q_stats": summarize_numeric(selected_q),
            "selected_topo_cost_stats": summarize_numeric(selected_topo),
            "selected_graph_score_stats": summarize_numeric(graph_scores[selected_indices]),
            "cost_weights": {
                "alpha_diff": float(cost_alpha_diff),
                "beta_wavelet": float(cost_beta_wavelet),
                "matching_wavelet_weight": float(cost_beta_wavelet),
                "gamma_topo": float(cost_gamma_topo),
                "eta_lsrc": float(cost_eta_lsrc),
            },
            "wavelet_active_scales": [] if wavelet_bundle is None else list(wavelet_bundle.get("active_scales", [])),
        },
    }


def sort_selected_indices(selected_indices):
    return sorted(int(x) for x in selected_indices)


def build_selected_meta(sample_meta, selected_indices):
    return [sample_meta[int(idx)] for idx in selected_indices]


def build_per_proxy_matched_meta(sample_meta, selected_indices):
    matched = []
    for proxy_idx, sample_idx in enumerate(selected_indices):
        item = dict(sample_meta[int(sample_idx)])
        item["proxy_idx"] = int(proxy_idx)
        item["matched_sample_idx"] = int(sample_idx)
        matched.append(item)
    return matched


def compute_selection_summary(
    args,
    num_samples,
    subset_size,
    selected_indices,
    unified_graph,
    cross_modal_summary,
    graph_scores,
    extra_summary=None,
):
    selected_scores = [float(graph_scores[idx]) for idx in selected_indices]
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "image_encoder": args.image_encoder,
        "text_encoder": args.text_encoder,
        "diagnostic_experiment_id": None if getattr(args, "diagnostic_experiment_id", None) is None else int(getattr(args, "diagnostic_experiment_id")),
        "enable_stage2_correction": bool(getattr(args, "enable_stage2_correction", True)),
        "enable_stage3_fusion": bool(getattr(args, "enable_stage3_fusion", True)),
        "enable_stage4_lsrc": bool(getattr(args, "enable_stage4_lsrc", True)),
        "budget_ratio": float(args.budget_ratio) if getattr(args, "budget_ratio", None) is not None else None,
        "budget_size": int(subset_size),
        "requested_budget_size": int(getattr(args, "budget_size", subset_size) or subset_size),
        "subset_size": int(subset_size),
        "num_samples": int(num_samples),
        "representation_mode": args.representation_mode,
        "selection_method": getattr(args, "selection_method", "baseline"),
        "selection_seed": int(getattr(args, "random_state", 0)),
        "cluster_method": getattr(args, "cluster_method", None),
        "degree_weight": float(args.degree_weight),
        "mean_selected_graph_score": float(np.mean(selected_scores)) if selected_scores else 0.0,
        "max_selected_graph_score": float(np.max(selected_scores)) if selected_scores else 0.0,
        "min_selected_graph_score": float(np.min(selected_scores)) if selected_scores else 0.0,
        "healthy_modality": cross_modal_summary.get("healthy_modality"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_summary:
        summary.update(extra_summary)
    return summary


def save_selection_outputs(
    output_dir,
    selected_indices,
    selected_meta,
    summary,
    proxy_bundle=None,
    matching_bundle=None,
    matched_proxy_meta=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_indices_path = output_dir / "selected_indices.json"
    selected_meta_path = output_dir / "selected_meta.json"
    summary_path = output_dir / "summary.json"
    per_proxy_meta_path = output_dir / "matched_proxy_meta.json"

    save_selected_indices(selected_indices_path, selected_indices)
    with open(selected_meta_path, "w", encoding="utf-8") as handle:
        json.dump(selected_meta, handle, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if matched_proxy_meta is not None:
        with open(per_proxy_meta_path, "w", encoding="utf-8") as handle:
            json.dump(matched_proxy_meta, handle, ensure_ascii=False, indent=2)

    saved_paths = {
        "selected_indices": str(selected_indices_path),
        "selected_meta": str(selected_meta_path),
        "summary": str(summary_path),
    }
    if matched_proxy_meta is not None:
        saved_paths["matched_proxy_meta"] = str(per_proxy_meta_path)

    if proxy_bundle is not None:
        proxy_points_path = output_dir / "proxy_points.pt"
        proxy_init_path = output_dir / "proxy_init.pt"
        proxy_debug_path = output_dir / "proxy_debug.json"

        torch.save(torch.tensor(proxy_bundle["proxy_points"], dtype=torch.float32), proxy_points_path)
        torch.save(torch.tensor(proxy_bundle["proxy_init"], dtype=torch.float32), proxy_init_path)
        with open(proxy_debug_path, "w", encoding="utf-8") as handle:
            json.dump(proxy_bundle["summary"], handle, ensure_ascii=False, indent=2)

        saved_paths["proxy_points"] = str(proxy_points_path)
        saved_paths["proxy_init"] = str(proxy_init_path)
        saved_paths["proxy_debug"] = str(proxy_debug_path)

        if proxy_bundle.get("projection_matrix") is not None:
            projection_path = output_dir / "projection_matrix.pt"
            torch.save(torch.tensor(proxy_bundle["projection_matrix"], dtype=torch.float32), projection_path)
            saved_paths["projection_matrix"] = str(projection_path)

        if proxy_bundle.get("frequencies") is not None:
            frequency_path = output_dir / "frequency_points.pt"
            torch.save(torch.tensor(proxy_bundle["frequencies"], dtype=torch.float32), frequency_path)
            saved_paths["frequency_points"] = str(frequency_path)
        if proxy_bundle.get("initial_frequencies") is not None:
            initial_frequency_path = output_dir / "initial_frequency_points.pt"
            torch.save(torch.tensor(proxy_bundle["initial_frequencies"], dtype=torch.float32), initial_frequency_path)
            saved_paths["initial_frequency_points"] = str(initial_frequency_path)
        if proxy_bundle.get("lsrc_outputs") is not None:
            lsrc_direct_path = output_dir / "lsrc_coverage_direct.pt"
            lsrc_relational_path = output_dir / "lsrc_coverage_relational.pt"
            lsrc_q_path = output_dir / "lsrc_confidence_q.pt"
            torch.save(torch.tensor(proxy_bundle["lsrc_outputs"]["coverage_direct"], dtype=torch.float32), lsrc_direct_path)
            torch.save(torch.tensor(proxy_bundle["lsrc_outputs"]["coverage_relational"], dtype=torch.float32), lsrc_relational_path)
            torch.save(torch.tensor(proxy_bundle["lsrc_outputs"]["confidence_q"], dtype=torch.float32), lsrc_q_path)
            saved_paths["lsrc_coverage_direct"] = str(lsrc_direct_path)
            saved_paths["lsrc_coverage_relational"] = str(lsrc_relational_path)
            saved_paths["lsrc_confidence_q"] = str(lsrc_q_path)

    if matching_bundle is not None:
        matching_cost_path = output_dir / "matching_cost.pt"
        matching_debug_path = output_dir / "matching_debug.json"
        payload = {
            "total_cost": torch.tensor(matching_bundle["cost_bundle"]["total_cost"], dtype=torch.float32),
        }
        if matching_bundle.get("candidate_indices") is not None:
            payload["candidate_indices"] = torch.tensor(matching_bundle["candidate_indices"], dtype=torch.long)
        if "diffusion_cost" in matching_bundle["cost_bundle"]:
            payload["diffusion_cost"] = torch.tensor(matching_bundle["cost_bundle"]["diffusion_cost"], dtype=torch.float32)
        if "wavelet_cost" in matching_bundle["cost_bundle"]:
            payload["wavelet_cost"] = torch.tensor(matching_bundle["cost_bundle"]["wavelet_cost"], dtype=torch.float32)
        if "topo_cost" in matching_bundle["cost_bundle"]:
            payload["topo_cost"] = torch.tensor(matching_bundle["cost_bundle"]["topo_cost"], dtype=torch.float32)
        if "lsrc_reward" in matching_bundle["cost_bundle"]:
            payload["lsrc_reward"] = torch.tensor(matching_bundle["cost_bundle"]["lsrc_reward"], dtype=torch.float32)
        if "geometry_cost" in matching_bundle["cost_bundle"]:
            payload["geometry_cost"] = torch.tensor(matching_bundle["cost_bundle"]["geometry_cost"], dtype=torch.float32)
        if "topology_cost" in matching_bundle["cost_bundle"]:
            payload["topology_cost"] = torch.tensor(matching_bundle["cost_bundle"]["topology_cost"], dtype=torch.float32)
        if "degree" in matching_bundle["cost_bundle"]:
            payload["degree"] = torch.tensor(matching_bundle["cost_bundle"]["degree"], dtype=torch.float32)
        torch.save(payload, matching_cost_path)
        with open(matching_debug_path, "w", encoding="utf-8") as handle:
            json.dump(matching_bundle["matching_debug"], handle, ensure_ascii=False, indent=2)
        saved_paths["matching_cost"] = str(matching_cost_path)
        saved_paths["matching_debug"] = str(matching_debug_path)

    return saved_paths


def run_baseline_selection(args, representation, unified_graph):
    subset_size = resolve_subset_size(
        representation.shape[0],
        budget_ratio=getattr(args, "budget_ratio", None),
        budget_size=getattr(args, "budget_size", None),
    )
    graph_scores = compute_graph_scores(unified_graph)
    centers, _ = fit_proxy_centers(
        representation,
        subset_size=subset_size,
        random_state=args.random_state,
        use_minibatch=(args.cluster_method == "minibatch_kmeans"),
        batch_size=args.minibatch_size,
    )
    selected_indices = select_representatives(
        representation,
        centers,
        graph_scores,
        degree_weight=args.degree_weight,
    )
    return {
        "selected_indices": selected_indices,
        "subset_size": subset_size,
        "graph_scores": graph_scores,
        "proxy_bundle": None,
        "matching_bundle": None,
        "extra_summary": {
            "cluster_method": args.cluster_method,
        },
    }


def run_proxy_optimized_selection(args, representation, unified_graph):
    subset_size = resolve_subset_size(
        representation.shape[0],
        budget_ratio=getattr(args, "budget_ratio", None),
        budget_size=getattr(args, "budget_size", None),
    )
    proxy_loss_type = resolve_proxy_loss_type(
        getattr(args, "proxy_loss_type", None),
        objective_mode=getattr(args, "proxy_objective_mode", None),
    )
    stage4_enabled = bool(getattr(args, "enable_stage4_lsrc", True))
    reference_embedding = build_reference_embedding(
        representation,
        spectral_embedding=getattr(args, "_spectral_embedding", None),
        mode=getattr(args, "reference_embedding_mode", "hybrid"),
        spectral_weight=getattr(args, "spectral_weight", 1.0),
    )
    optimization_embedding = reference_embedding
    if proxy_loss_type in {"diffusion_mmd", "diffusion_swd", "diffusion_ms_swd"}:
        optimization_embedding = getattr(args, "_spectral_embedding", None)
        if optimization_embedding is None:
            raise ValueError(f"{proxy_loss_type} requires unified spectral embedding, but none was found in cross-modal artifacts.")
        if str(getattr(args, "_embedding_type", "laplacian")) != "diffusion":
            raise ValueError(
                f"{proxy_loss_type} requires diffusion embedding artifacts. "
                f"Current cross-modal embedding_type={getattr(args, '_embedding_type', 'unknown')}."
            )
    graph_reference = build_topology_targets(
        unified_graph,
        optimization_embedding,
        hop_weight=args.topology_hop_weight,
    )
    proxy_bundle = optimize_proxy_points(
        optimization_embedding,
        subset_size=subset_size,
        device=args.device,
        projection_dim=args.proxy_projection_dim,
        random_state=args.random_state,
        init_method=args.proxy_init_method,
        minibatch_size=args.minibatch_size,
        num_frequencies=args.proxy_num_frequencies,
        frequency_scale=args.proxy_frequency_scale,
        lr=args.proxy_lr,
        num_steps=args.proxy_num_steps,
        reg_weight=args.proxy_reg_weight,
        target_batch_size=args.proxy_target_batch_size,
        proxy_batch_size=args.proxy_batch_size,
        proxy_loss_type=proxy_loss_type,
        objective_mode=getattr(args, "proxy_objective_mode", "pd_cfd"),
        use_pdas=bool(getattr(args, "use_pdas", False)),
        pdas_num_stages=getattr(args, "pdas_num_stages", 4),
        pdas_schedule_mode=getattr(args, "pdas_schedule_mode", "low_to_high"),
        use_dpp=bool(getattr(args, "use_dpp", False)),
        lambda_div=getattr(args, "lambda_div", 0.0),
        lambda_match=getattr(args, "lambda_match", 0.0),
        lambda_graph=getattr(args, "lambda_graph", 0.0),
        lambda_phase=getattr(args, "lambda_phase", 0.1),
        num_freq_pool=getattr(args, "num_freq_pool", None),
        tau_min=getattr(args, "tau_min", 0.1),
        tau_max=getattr(args, "tau_max", None),
        diversity_sigma=getattr(args, "diversity_sigma", 1.0),
        phase_weight_mode=getattr(args, "phase_weight_mode", "uniform"),
        match_reference=optimization_embedding,
        graph_reference=graph_reference,
        enable_lsrc=bool(stage4_enabled and getattr(args, "enable_lsrc", False)),
        keep_lsrc=bool(stage4_enabled and getattr(args, "keep_lsrc", True)),
        lsrc_image_graph=getattr(args, "_lsrc_image_graph", None),
        lsrc_text_graph=getattr(args, "_lsrc_text_graph", None),
        lsrc_k=getattr(args, "lsrc_k", 32),
        lsrc_tau_r=getattr(args, "lsrc_tau_r", 1.0),
        lsrc_tau_c=getattr(args, "lsrc_tau_c", 1.0),
        lsrc_eta=getattr(args, "lsrc_eta", 0.5),
        lsrc_beta=getattr(args, "lsrc_beta", 0.5),
        lambda_lsrc_cov=getattr(args, "lambda_lsrc_cov", 0.0),
        lambda_lsrc_rel=getattr(args, "lambda_lsrc_rel", 0.0),
        lsrc_eps=getattr(args, "lsrc_eps", 1e-8),
        lsrc_batch_size=getattr(args, "lsrc_batch_size", args.proxy_target_batch_size),
        lsrc_rho_img=getattr(args, "_lsrc_rho_img", 0.5),
        lsrc_rho_txt=getattr(args, "_lsrc_rho_txt", 0.5),
        lsrc_use_global_confidence=bool(getattr(args, "lsrc_use_global_confidence", False)),
        lsrc_coverage_mode=getattr(args, "lsrc_coverage_mode", "mean"),
        lsrc_rel_loss_mode=getattr(args, "lsrc_rel_loss_mode", "weight_mean"),
        mmd_kernel=getattr(args, "mmd_kernel", "rbf"),
        mmd_bandwidth=getattr(args, "mmd_bandwidth", None),
        mmd_use_median_heuristic=bool(getattr(args, "mmd_use_median_heuristic", True)),
        swd_num_projections=getattr(args, "swd_num_projections", 64),
        swd_p=getattr(args, "swd_p", 2),
        swd_projection_seed=getattr(args, "swd_projection_seed", None),
        swd_use_fixed_projections=bool(getattr(args, "swd_use_fixed_projections", False)),
        use_wavelet_multiscale=bool(getattr(args, "use_wavelet_multiscale", False)),
        wavelet_graph=unified_graph,
        wavelet_scales=getattr(args, "wavelet_scales", None),
        wavelet_loss_weight=getattr(args, "wavelet_loss_weight", 0.0),
        wavelet_distance_type=getattr(args, "wavelet_distance_type", "mmd"),
        wavelet_schedule=getattr(args, "wavelet_schedule", "coarse_to_fine"),
        wavelet_swd_num_projections=getattr(args, "wavelet_swd_num_projections", None),
        wavelet_swd_p=getattr(args, "wavelet_swd_p", None),
        lambda_main=getattr(args, "lambda_main", 1.0),
        wavelet_main_scales=getattr(args, "wavelet_main_scales", None),
        wavelet_main_scale_weights=getattr(args, "wavelet_main_scale_weights", None),
        wavelet_main_swd_num_projections=getattr(args, "wavelet_main_swd_num_projections", None),
        wavelet_cov_weight=getattr(args, "wavelet_cov_weight", 0.5),
        wavelet_edge_weight=getattr(args, "wavelet_edge_weight", 0.25),
        wavelet_curriculum_schedule=getattr(args, "wavelet_curriculum_schedule", "coarse_to_fine"),
        lambda_diff=getattr(args, "lambda_diff", 1.0),
        lambda_ms=getattr(args, "lambda_ms", None),
        lambda_lsrc=getattr(args, "lambda_lsrc", None),
        lsrc_mu=getattr(args, "lsrc_mu", 1.0),
        lambda_reg=getattr(args, "lambda_reg", 1.0),
        reg_alpha_div=getattr(args, "reg_alpha_div", 1.0),
        reg_beta_topo=getattr(args, "reg_beta_topo", 1.0),
        reg_gamma_init=getattr(args, "reg_gamma_init", 1.0),
    )
    projected_representation = proxy_bundle["projected_representation"]
    topology_targets = build_topology_targets(unified_graph, projected_representation, hop_weight=args.topology_hop_weight)
    wavelet_bundle = resolve_wavelet_signature_bundle(proxy_bundle, unified_graph, projected_representation)
    lsrc_confidence_q = None
    if proxy_bundle.get("lsrc_outputs") is not None:
        lsrc_confidence_q = np.asarray(proxy_bundle["lsrc_outputs"]["confidence_q"], dtype=np.float32)
    cost_alpha_diff = float(getattr(args, "cost_alpha_diff", 0.25))
    cost_beta_wavelet = float(getattr(args, "matching_wavelet_weight", getattr(args, "cost_beta_wavelet", 1.0)))
    cost_gamma_topo = float(getattr(args, "cost_gamma_topo", 0.1))
    cost_eta_lsrc = float(getattr(args, "cost_eta_lsrc", 0.1))
    if not stage4_enabled:
        lsrc_confidence_q = None
        cost_beta_wavelet = 0.0
        cost_gamma_topo = 0.0
        cost_eta_lsrc = 0.0
    matching_bundle = run_proxy_matching(
        proxy_bundle["proxy_points"],
        projected_representation,
        topology_targets,
        unified_graph,
        matching_top_k=args.matching_top_k,
        degree_weight=args.degree_weight,
        topology_weight=args.topology_weight,
        geometry_weight=getattr(args, "geometry_weight", 1.0),
        candidate_batch_size=args.matching_candidate_batch_size,
        wavelet_bundle=wavelet_bundle,
        lsrc_confidence=lsrc_confidence_q,
        cost_alpha_diff=cost_alpha_diff,
        cost_beta_wavelet=cost_beta_wavelet,
        cost_gamma_topo=cost_gamma_topo,
        cost_eta_lsrc=cost_eta_lsrc,
    )
    if getattr(args, "matching_cost_mode", "candidate_topk") == "degree_aware_global":
        matching_bundle = run_degree_aware_global_matching(
            proxy_bundle["proxy_points"],
            projected_representation,
            unified_graph,
            wavelet_bundle=wavelet_bundle,
            lsrc_confidence=lsrc_confidence_q,
            cost_alpha_diff=cost_alpha_diff,
            cost_beta_wavelet=cost_beta_wavelet,
            cost_gamma_topo=cost_gamma_topo,
            cost_eta_lsrc=cost_eta_lsrc,
        )

    match_loss = compute_match_loss(
        proxy_bundle["proxy_points"],
        projected_representation,
        matching_bundle["selected_indices"],
    )
    graph_loss_value = compute_graph_regularization(
        proxy_bundle["proxy_points"],
        unified_graph,
        matching_bundle["selected_indices"],
    )
    matching_bundle["matching_debug"]["match_loss"] = float(match_loss)
    matching_bundle["matching_debug"]["graph_loss"] = float(graph_loss_value)
    return {
        "selected_indices": matching_bundle["selected_indices"],
        "subset_size": subset_size,
        "graph_scores": matching_bundle["graph_scores"],
        "proxy_bundle": proxy_bundle,
        "matching_bundle": matching_bundle,
        "extra_summary": {
            "cluster_method": "proxy_optimization",
            "proxy_projection_dim": int(proxy_bundle["summary"]["projection_dim"]),
            "proxy_num_steps": int(args.proxy_num_steps),
            "proxy_num_frequencies": int(args.proxy_num_frequencies),
            "proxy_lr": float(args.proxy_lr),
            "proxy_reg_weight": float(args.proxy_reg_weight),
            "proxy_loss_type": getattr(args, "proxy_loss_type", None),
            "proxy_loss_type_effective": proxy_loss_type,
            "proxy_objective_mode": getattr(args, "proxy_objective_mode", "pd_cfd"),
            "reference_embedding_mode": getattr(args, "reference_embedding_mode", "hybrid"),
            "enable_lsrc": bool(getattr(args, "enable_lsrc", False)),
            "keep_lsrc": bool(stage4_enabled and getattr(args, "keep_lsrc", True)),
            "lsrc_k": int(getattr(args, "lsrc_k", 32)),
            "lambda_lsrc_cov": float(getattr(args, "lambda_lsrc_cov", 0.0)),
            "lambda_lsrc_rel": float(getattr(args, "lambda_lsrc_rel", 0.0)),
            "lambda_main": float(getattr(args, "lambda_main", 1.0)),
            "wavelet_main_scales": getattr(args, "wavelet_main_scales", None),
            "wavelet_main_scale_weights": getattr(args, "wavelet_main_scale_weights", None),
            "wavelet_cov_weight": float(getattr(args, "wavelet_cov_weight", 0.5)),
            "wavelet_edge_weight": float(getattr(args, "wavelet_edge_weight", 0.25)),
            "wavelet_curriculum_schedule": getattr(args, "wavelet_curriculum_schedule", "coarse_to_fine"),
            "matching_wavelet_weight": float(getattr(args, "matching_wavelet_weight", getattr(args, "cost_beta_wavelet", 1.0))),
            "matching_top_k": int(args.matching_top_k),
            "geometry_weight": float(getattr(args, "geometry_weight", 1.0)),
            "topology_weight": float(args.topology_weight),
            "topology_hop_weight": float(args.topology_hop_weight),
            "duplicate_resolution_rounds": int(matching_bundle["matching_debug"]["duplicate_resolution_rounds"]),
            "local_hungarian_calls": int(matching_bundle["matching_debug"]["local_hungarian_calls"]),
            "assignment_mode": matching_bundle["matching_debug"]["assignment_mode"],
            "matching_cost_mode": getattr(args, "matching_cost_mode", "candidate_topk"),
            "diagnostic_experiment_id": None if getattr(args, "diagnostic_experiment_id", None) is None else int(getattr(args, "diagnostic_experiment_id")),
            "enable_stage2_correction": bool(getattr(args, "enable_stage2_correction", True)),
            "enable_stage3_fusion": bool(getattr(args, "enable_stage3_fusion", True)),
            "enable_stage4_lsrc": bool(stage4_enabled),
            "stage4_matching_geometry_only": bool(not stage4_enabled),
            "match_loss": float(matching_bundle["matching_debug"]["match_loss"]),
            "graph_loss": float(matching_bundle["matching_debug"]["graph_loss"]),
            "proxy_initial_loss": proxy_bundle["summary"]["initial_loss"],
            "proxy_final_loss": proxy_bundle["summary"]["final_loss"],
            "selected_local_coverage_stats": matching_bundle["matching_debug"].get("selected_q_stats", {}),
        },
    }


def run_subset_selection(args):
    feature_dir = build_feature_dir(args)
    cross_modal_dir = build_cross_modal_dir(args)

    img_features, txt_features, sample_meta = load_feature_cache(feature_dir)
    unified_graph, spectral_embedding, cross_modal_summary, lsrc_image_graph, lsrc_text_graph = load_unified_artifacts(cross_modal_dir)

    if img_features.shape[0] != txt_features.shape[0] or img_features.shape[0] != len(sample_meta):
        raise ValueError("Feature cache and sample meta length mismatch.")
    if unified_graph.shape[0] != len(sample_meta):
        raise ValueError("Unified graph node count does not match sample_meta length.")
    if spectral_embedding is not None and spectral_embedding.shape[0] != len(sample_meta):
        raise ValueError("Unified spectral embedding node count does not match sample_meta length.")

    representation = build_unified_representation(img_features, txt_features, mode=args.representation_mode)
    args._spectral_embedding = spectral_embedding
    args._embedding_type = str(cross_modal_summary.get("embedding_type", "laplacian"))
    args._lsrc_image_graph = lsrc_image_graph
    args._lsrc_text_graph = lsrc_text_graph
    args._lsrc_rho_img = float(cross_modal_summary.get("rho_img", 0.5))
    args._lsrc_rho_txt = float(cross_modal_summary.get("rho_txt", 0.5))
    selection_method = getattr(args, "selection_method", "baseline")
    if getattr(args, "diagnostic_experiment_id", None) in {0, 1} and selection_method == "proxy_opt":
        selection_method = "baseline"
        args.selection_method = "baseline"
    if not bool(getattr(args, "enable_stage4_lsrc", True)):
        args.enable_lsrc = False
        args.keep_lsrc = False
    output_dir = build_output_dir(args)

    if selection_method == "proxy_opt":
        outputs = run_proxy_optimized_selection(args, representation, unified_graph)
    else:
        outputs = run_baseline_selection(args, representation, unified_graph)

    selected_indices = sort_selected_indices(outputs["selected_indices"])
    selected_meta = build_selected_meta(sample_meta, selected_indices)
    matched_proxy_meta = build_per_proxy_matched_meta(sample_meta, selected_indices) if outputs["matching_bundle"] is not None else None
    summary = compute_selection_summary(
        args,
        num_samples=representation.shape[0],
        subset_size=outputs["subset_size"],
        selected_indices=selected_indices,
        unified_graph=unified_graph,
        cross_modal_summary=cross_modal_summary,
        graph_scores=outputs["graph_scores"],
        extra_summary=outputs["extra_summary"],
    )
    saved = save_selection_outputs(
        output_dir,
        selected_indices,
        selected_meta,
        summary,
        proxy_bundle=outputs["proxy_bundle"],
        matching_bundle=outputs["matching_bundle"],
        matched_proxy_meta=matched_proxy_meta,
    )

    return {
        "output_dir": str(output_dir),
        "subset_size": outputs["subset_size"],
        "selected_indices": selected_indices,
        "saved": saved,
        "summary": summary,
    }
