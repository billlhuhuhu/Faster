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
from src.proxy_optimization import l2_normalize, optimize_proxy_points


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
    base_dir = Path(args.output_root) / args.dataset / args.split / model_tag / budget_tag
    if getattr(args, "selection_method", "baseline") != "baseline":
        return base_dir / sanitize_name(args.selection_method)
    return base_dir


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
    return unified_graph, spectral_embedding, summary


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


def compute_candidate_neighbors(representation, proxy_points, top_k):
    top_k = max(1, min(int(top_k), int(representation.shape[0])))
    knn = NearestNeighbors(n_neighbors=top_k, metric="euclidean")
    knn.fit(representation)
    distances, indices = knn.kneighbors(proxy_points)
    return distances.astype(np.float32), indices.astype(np.int64)


def compute_proxy_sample_costs(
    proxy_points,
    candidate_points,
    topology_points,
    graph_scores,
    degree_weight,
    topology_weight,
    geometry_weight=1.0,
):
    geometry_cost = np.sum((proxy_points - candidate_points) ** 2, axis=-1).astype(np.float32)
    topology_cost = np.sum((proxy_points - topology_points) ** 2, axis=-1).astype(np.float32)
    total_cost = (
        float(geometry_weight) * geometry_cost
        + float(topology_weight) * topology_cost
        - float(degree_weight) * graph_scores.astype(np.float32)
    )
    return total_cost.astype(np.float32), geometry_cost, topology_cost


def build_degree_aware_cost_matrix(proxy_points, reference_points, unified_graph, eps=1e-8):
    geometry_cost = np.sum((proxy_points[:, None, :] - reference_points[None, :, :]) ** 2, axis=-1).astype(np.float32)
    degree = compute_graph_degree(unified_graph, eps=eps)
    total_cost = geometry_cost / degree[None, :]
    return {
        "total_cost": total_cost.astype(np.float32),
        "geometry_cost": geometry_cost.astype(np.float32),
        "degree": degree.astype(np.float32),
    }


def build_candidate_costs(
    proxy_points,
    representation,
    topology_targets,
    graph_scores,
    candidate_indices,
    degree_weight=0.1,
    topology_weight=0.5,
    geometry_weight=1.0,
    batch_size=128,
):
    num_proxies, top_k = candidate_indices.shape
    total_cost = np.empty((num_proxies, top_k), dtype=np.float32)
    geometry_cost = np.empty((num_proxies, top_k), dtype=np.float32)
    topology_cost = np.empty((num_proxies, top_k), dtype=np.float32)

    for start in range(0, num_proxies, int(batch_size)):
        end = min(num_proxies, start + int(batch_size))
        batch_candidates = candidate_indices[start:end]
        batch_proxy = proxy_points[start:end, None, :]
        batch_repr = representation[batch_candidates]
        batch_topo = topology_targets[batch_candidates]
        batch_graph = graph_scores[batch_candidates]
        batch_total, batch_geom, batch_topo_cost = compute_proxy_sample_costs(
            batch_proxy,
            batch_repr,
            batch_topo,
            batch_graph,
            degree_weight=degree_weight,
            topology_weight=topology_weight,
            geometry_weight=geometry_weight,
        )
        total_cost[start:end] = batch_total
        geometry_cost[start:end] = batch_geom
        topology_cost[start:end] = batch_topo_cost

    return {
        "total_cost": total_cost,
        "geometry_cost": geometry_cost,
        "topology_cost": topology_cost,
    }


def build_single_row_cost(proxy_point, candidate_indices, representation, topology_targets, graph_scores, degree_weight, topology_weight, geometry_weight=1.0):
    candidate_repr = representation[candidate_indices]
    candidate_topo = topology_targets[candidate_indices]
    candidate_graph = graph_scores[candidate_indices]
    total_cost, _, _ = compute_proxy_sample_costs(
        proxy_point[None, :],
        candidate_repr[None, :, :],
        candidate_topo[None, :, :],
        candidate_graph[None, :],
        degree_weight=degree_weight,
        topology_weight=topology_weight,
        geometry_weight=geometry_weight,
    )
    return total_cost.reshape(-1)


def resolve_duplicate_assignments(
    proxy_points,
    candidate_indices,
    candidate_costs,
    representation,
    topology_targets,
    graph_scores,
    degree_weight=0.1,
    topology_weight=0.5,
    geometry_weight=1.0,
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
                    topology_targets,
                    graph_scores,
                    degree_weight=degree_weight,
                    topology_weight=topology_weight,
                    geometry_weight=geometry_weight,
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
    topology_targets,
    graph_scores,
    degree_weight=0.1,
    topology_weight=0.5,
    geometry_weight=1.0,
):
    num_proxies = int(candidate_indices.shape[0])
    unique_candidates = np.unique(candidate_indices.reshape(-1)).astype(np.int64)
    if unique_candidates.size < num_proxies:
        fallback_selected, fallback_debug = resolve_duplicate_assignments(
            proxy_points,
            candidate_indices,
            candidate_costs,
            representation,
            topology_targets,
            graph_scores,
            degree_weight=degree_weight,
            topology_weight=topology_weight,
            geometry_weight=geometry_weight,
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
):
    graph_scores = compute_graph_scores(unified_graph)
    _, candidate_indices = compute_candidate_neighbors(representation, proxy_points, top_k=matching_top_k)
    cost_bundle = build_candidate_costs(
        proxy_points,
        representation,
        topology_targets,
        graph_scores,
        candidate_indices,
        degree_weight=degree_weight,
        topology_weight=topology_weight,
        geometry_weight=geometry_weight,
        batch_size=candidate_batch_size,
    )
    match_outputs = run_hungarian_matching(
        proxy_points,
        candidate_indices,
        cost_bundle["total_cost"],
        representation,
        topology_targets,
        graph_scores,
        degree_weight=degree_weight,
        topology_weight=topology_weight,
        geometry_weight=geometry_weight,
    )
    return {
        "selected_indices": match_outputs["selected_indices"],
        "graph_scores": graph_scores,
        "candidate_indices": candidate_indices,
        "cost_bundle": cost_bundle,
        "matching_debug": {
            **match_outputs["diagnostics"],
            "assignment_mode": match_outputs["assignment_mode"],
        },
    }


def run_degree_aware_global_matching(proxy_points, reference_points, unified_graph):
    cost_bundle = build_degree_aware_cost_matrix(proxy_points, reference_points, unified_graph)
    row_ind, col_ind = linear_sum_assignment(cost_bundle["total_cost"])
    selected_indices = np.full(proxy_points.shape[0], -1, dtype=np.int64)
    selected_indices[row_ind] = col_ind
    return {
        "selected_indices": selected_indices.astype(np.int64).tolist(),
        "candidate_indices": None,
        "cost_bundle": cost_bundle,
        "matching_debug": {
            "assignment_mode": "global_degree_aware_hungarian",
            "duplicate_resolution_rounds": 0,
            "local_hungarian_calls": 0,
            "hungarian_rows": int(len(row_ind)),
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
        "budget_ratio": float(args.budget_ratio) if getattr(args, "budget_ratio", None) is not None else None,
        "budget_size": int(subset_size),
        "requested_budget_size": int(getattr(args, "budget_size", subset_size) or subset_size),
        "subset_size": int(subset_size),
        "num_samples": int(num_samples),
        "representation_mode": args.representation_mode,
        "selection_method": getattr(args, "selection_method", "baseline"),
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

        frequency_path = output_dir / "frequency_points.pt"
        torch.save(torch.tensor(proxy_bundle["frequencies"], dtype=torch.float32), frequency_path)
        saved_paths["frequency_points"] = str(frequency_path)
        if proxy_bundle.get("initial_frequencies") is not None:
            initial_frequency_path = output_dir / "initial_frequency_points.pt"
            torch.save(torch.tensor(proxy_bundle["initial_frequencies"], dtype=torch.float32), initial_frequency_path)
            saved_paths["initial_frequency_points"] = str(initial_frequency_path)

    if matching_bundle is not None:
        matching_cost_path = output_dir / "matching_cost.pt"
        matching_debug_path = output_dir / "matching_debug.json"
        payload = {
            "total_cost": torch.tensor(matching_bundle["cost_bundle"]["total_cost"], dtype=torch.float32),
        }
        if matching_bundle.get("candidate_indices") is not None:
            payload["candidate_indices"] = torch.tensor(matching_bundle["candidate_indices"], dtype=torch.long)
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
    reference_embedding = build_reference_embedding(
        representation,
        spectral_embedding=getattr(args, "_spectral_embedding", None),
        mode=getattr(args, "reference_embedding_mode", "hybrid"),
        spectral_weight=getattr(args, "spectral_weight", 1.0),
    )
    graph_reference = build_topology_targets(
        unified_graph,
        reference_embedding,
        hop_weight=args.topology_hop_weight,
    )
    proxy_bundle = optimize_proxy_points(
        reference_embedding,
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
        match_reference=reference_embedding,
        graph_reference=graph_reference,
    )
    projected_representation = proxy_bundle["projected_representation"]
    topology_targets = build_topology_targets(unified_graph, projected_representation, hop_weight=args.topology_hop_weight)
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
    )
    if getattr(args, "matching_cost_mode", "candidate_topk") == "degree_aware_global":
        matching_bundle = run_degree_aware_global_matching(
            proxy_bundle["proxy_points"],
            projected_representation,
            unified_graph,
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
            "proxy_objective_mode": getattr(args, "proxy_objective_mode", "pd_cfd"),
            "reference_embedding_mode": getattr(args, "reference_embedding_mode", "hybrid"),
            "matching_top_k": int(args.matching_top_k),
            "geometry_weight": float(getattr(args, "geometry_weight", 1.0)),
            "topology_weight": float(args.topology_weight),
            "topology_hop_weight": float(args.topology_hop_weight),
            "duplicate_resolution_rounds": int(matching_bundle["matching_debug"]["duplicate_resolution_rounds"]),
            "local_hungarian_calls": int(matching_bundle["matching_debug"]["local_hungarian_calls"]),
            "assignment_mode": matching_bundle["matching_debug"]["assignment_mode"],
            "matching_cost_mode": getattr(args, "matching_cost_mode", "candidate_topk"),
            "match_loss": float(matching_bundle["matching_debug"]["match_loss"]),
            "graph_loss": float(matching_bundle["matching_debug"]["graph_loss"]),
            "proxy_initial_loss": proxy_bundle["summary"]["initial_loss"],
            "proxy_final_loss": proxy_bundle["summary"]["final_loss"],
        },
    }


def run_subset_selection(args):
    feature_dir = build_feature_dir(args)
    cross_modal_dir = build_cross_modal_dir(args)
    output_dir = build_output_dir(args)

    img_features, txt_features, sample_meta = load_feature_cache(feature_dir)
    unified_graph, spectral_embedding, cross_modal_summary = load_unified_artifacts(cross_modal_dir)

    if img_features.shape[0] != txt_features.shape[0] or img_features.shape[0] != len(sample_meta):
        raise ValueError("Feature cache and sample meta length mismatch.")
    if unified_graph.shape[0] != len(sample_meta):
        raise ValueError("Unified graph node count does not match sample_meta length.")
    if spectral_embedding is not None and spectral_embedding.shape[0] != len(sample_meta):
        raise ValueError("Unified spectral embedding node count does not match sample_meta length.")

    representation = build_unified_representation(img_features, txt_features, mode=args.representation_mode)
    args._spectral_embedding = spectral_embedding
    selection_method = getattr(args, "selection_method", "baseline")

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
