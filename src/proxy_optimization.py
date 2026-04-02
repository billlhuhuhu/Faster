import math

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


def l2_normalize(features, eps=1e-12):
    if isinstance(features, np.ndarray):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return features / norms

    if torch.is_tensor(features):
        norms = torch.linalg.norm(features, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=eps)
        return features / norms

    raise TypeError(f"Unsupported feature type: {type(features)}")


def random_project_representation(representation, projection_dim=None, random_state=0):
    representation = np.asarray(representation, dtype=np.float32)
    if projection_dim is None or projection_dim <= 0 or representation.shape[1] <= projection_dim:
        return representation.astype(np.float32), None

    rng = np.random.default_rng(int(random_state))
    projection = rng.standard_normal((representation.shape[1], int(projection_dim))).astype(np.float32)
    projection /= math.sqrt(float(projection.shape[1]))
    projected = representation @ projection
    return projected.astype(np.float32), projection.astype(np.float32)


def project_reference_points(reference_points, projection_matrix=None):
    if reference_points is None:
        return None
    reference_points = np.asarray(reference_points, dtype=np.float32)
    if projection_matrix is not None:
        if reference_points.shape[1] != projection_matrix.shape[0]:
            raise ValueError(
                "Reference points and projection matrix dimension mismatch: "
                f"{reference_points.shape[1]} vs {projection_matrix.shape[0]}"
            )
        reference_points = reference_points @ projection_matrix
    return l2_normalize(reference_points.astype(np.float32))


def initialize_proxy_points(
    representation,
    subset_size,
    init_method="kmeans",
    random_state=0,
    minibatch_size=2048,
):
    representation = np.asarray(representation, dtype=np.float32)
    if init_method == "sample":
        rng = np.random.default_rng(int(random_state))
        chosen = rng.choice(representation.shape[0], size=int(subset_size), replace=False)
        return representation[chosen].astype(np.float32), {"init_method": "sample"}

    if init_method == "minibatch_kmeans":
        clusterer = MiniBatchKMeans(
            n_clusters=int(subset_size),
            random_state=int(random_state),
            batch_size=int(minibatch_size),
            n_init=10,
        )
    else:
        clusterer = KMeans(
            n_clusters=int(subset_size),
            random_state=int(random_state),
            n_init=10,
        )

    clusterer.fit(representation)
    return clusterer.cluster_centers_.astype(np.float32), {"init_method": init_method}


def sample_frequency_points(dim, num_frequencies, frequency_scale, device, random_state=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_state))
    frequencies = torch.randn((int(num_frequencies), int(dim)), generator=generator, dtype=torch.float32)
    frequencies = frequencies * float(frequency_scale)
    return frequencies.to(device=device)


def sample_frequency_pool(dim, num_frequencies, tau_max, device, random_state=0):
    return sample_frequency_points(
        dim=dim,
        num_frequencies=num_frequencies,
        frequency_scale=tau_max,
        device=device,
        random_state=random_state,
    )


def sample_pdas_frequencies(
    dim,
    num_frequencies,
    frequency_scale,
    stage,
    num_stages,
    schedule_mode="low_to_high",
    device="cpu",
    random_state=0,
):
    stage = max(0, int(stage))
    num_stages = max(1, int(num_stages))
    if schedule_mode == "low_to_high":
        scale_ratio = float(stage + 1) / float(num_stages)
    elif schedule_mode == "uniform":
        scale_ratio = 1.0
    else:
        raise ValueError(f"Unsupported PDAS schedule mode: {schedule_mode}")
    return sample_frequency_points(
        dim=dim,
        num_frequencies=num_frequencies,
        frequency_scale=float(frequency_scale) * max(scale_ratio, 1e-3),
        device=device,
        random_state=random_state + stage,
    )


def compute_empirical_characteristic_components(samples, frequencies, batch_size=4096):
    if samples.ndim != 2 or frequencies.ndim != 2:
        raise ValueError("samples and frequencies must both be 2D tensors.")

    real_sum = torch.zeros(frequencies.shape[0], dtype=torch.float32, device=frequencies.device)
    imag_sum = torch.zeros_like(real_sum)
    total = 0

    for batch in samples.split(int(batch_size), dim=0):
        phases = batch @ frequencies.t()
        real_sum += torch.cos(phases).sum(dim=0)
        imag_sum += torch.sin(phases).sum(dim=0)
        total += batch.shape[0]

    total = max(int(total), 1)
    return real_sum / total, imag_sum / total


def compute_characteristic_function(samples, frequencies, batch_size=4096):
    return compute_empirical_characteristic_components(samples, frequencies, batch_size=batch_size)


def compute_cf_amplitude_phase(real_part, imag_part, eps=1e-12):
    amplitude = torch.sqrt(real_part ** 2 + imag_part ** 2 + float(eps))
    phase = torch.atan2(imag_part, real_part)
    return amplitude, phase


def frequency_alignment_loss(proxy_points, target_real, target_imag, frequencies, batch_size=4096):
    proxy_real, proxy_imag = compute_empirical_characteristic_components(
        proxy_points,
        frequencies,
        batch_size=batch_size,
    )
    loss_real = torch.mean((proxy_real - target_real) ** 2)
    loss_imag = torch.mean((proxy_imag - target_imag) ** 2)
    return loss_real + loss_imag


def cfd_loss(proxy_points, target_samples, frequencies, batch_size=4096):
    target_real, target_imag = compute_characteristic_function(
        target_samples,
        frequencies,
        batch_size=batch_size,
    )
    return frequency_alignment_loss(
        proxy_points,
        target_real,
        target_imag,
        frequencies,
        batch_size=batch_size,
    )


def build_phase_weights(num_frequencies, mode="uniform", device="cpu"):
    if mode == "uniform":
        return torch.ones(int(num_frequencies), dtype=torch.float32, device=device)
    if mode == "linear":
        return torch.linspace(0.5, 1.0, int(num_frequencies), dtype=torch.float32, device=device)
    raise ValueError(f"Unsupported phase weight mode: {mode}")


def pd_cfd_loss(
    proxy_points,
    target_samples,
    frequencies,
    phase_weights=None,
    lambda_phase=0.1,
    batch_size=4096,
):
    proxy_real, proxy_imag = compute_characteristic_function(proxy_points, frequencies, batch_size=batch_size)
    target_real, target_imag = compute_characteristic_function(target_samples, frequencies, batch_size=batch_size)
    if phase_weights is None:
        phase_weights = torch.ones_like(proxy_real)
    phase_weights = phase_weights / torch.clamp(torch.sum(phase_weights), min=1e-12)
    cfd_term = torch.sum(phase_weights * ((proxy_real - target_real) ** 2 + (proxy_imag - target_imag) ** 2))

    proxy_amp, proxy_phase = compute_cf_amplitude_phase(proxy_real, proxy_imag)
    target_amp, target_phase = compute_cf_amplitude_phase(target_real, target_imag)
    phase_mask = torch.minimum(proxy_amp, target_amp)
    phase_distance = 1.0 - torch.cos(proxy_phase - target_phase)
    phase_term = torch.sum(phase_weights * phase_mask * phase_distance)
    return cfd_term + float(lambda_phase) * phase_term


def build_rbf_kernel(points, sigma=1.0):
    sqdist = torch.cdist(points, points, p=2) ** 2
    sigma = max(float(sigma), 1e-6)
    return torch.exp(-sqdist / (2.0 * sigma * sigma))


def dpp_diversity_loss(proxy_points, kernel="rbf", sigma=1.0, eps=1e-6):
    if kernel != "rbf":
        raise ValueError(f"Unsupported diversity kernel: {kernel}")
    kernel_matrix = build_rbf_kernel(proxy_points, sigma=sigma)
    eye = torch.eye(kernel_matrix.shape[0], dtype=kernel_matrix.dtype, device=kernel_matrix.device)
    stabilized = kernel_matrix + float(eps) * eye
    sign, logdet = torch.linalg.slogdet(stabilized)
    if torch.any(sign <= 0):
        return torch.tensor(0.0, dtype=proxy_points.dtype, device=proxy_points.device)
    return -logdet / max(int(proxy_points.shape[0]), 1)


def nearest_reference_loss(proxy_points, reference_points):
    if reference_points is None:
        return torch.tensor(0.0, dtype=proxy_points.dtype, device=proxy_points.device)
    distances = torch.cdist(proxy_points, reference_points, p=2) ** 2
    return torch.mean(torch.min(distances, dim=1).values)


def row_normalize_sparse_graph(graph, eps=1e-12):
    graph = graph.tocsr().astype(np.float32)
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1)
    row_sum = np.maximum(row_sum, float(eps))
    inv_row = 1.0 / row_sum.astype(np.float32)
    return graph.multiply(inv_row[:, None]).tocsr()


def build_union_keys(matrix_a, matrix_b):
    matrix_a = matrix_a.tocoo()
    matrix_b = matrix_b.tocoo()
    num_cols = int(matrix_a.shape[1])
    keys_a = matrix_a.row.astype(np.int64) * num_cols + matrix_a.col.astype(np.int64)
    keys_b = matrix_b.row.astype(np.int64) * num_cols + matrix_b.col.astype(np.int64)
    union_keys = np.unique(np.concatenate([keys_a, keys_b], axis=0))
    rows = (union_keys // num_cols).astype(np.int64)
    cols = (union_keys % num_cols).astype(np.int64)
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


def build_lsrc_relation_graph(
    real_points,
    image_graph,
    text_graph,
    k=32,
    tau_r=1.0,
    eta=0.5,
    rho_img=0.5,
    rho_txt=0.5,
    eps=1e-8,
    use_global_confidence=False,
):
    real_points = np.asarray(real_points, dtype=np.float32)
    num_nodes = int(real_points.shape[0])
    k = max(1, min(int(k), max(num_nodes - 1, 1)))
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, num_nodes), metric="euclidean")
    nbrs.fit(real_points)
    distances, indices = nbrs.kneighbors(real_points)

    if indices.shape[1] > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    rows = np.repeat(np.arange(num_nodes, dtype=np.int64), indices.shape[1])
    cols = indices.reshape(-1).astype(np.int64)
    sqdist = (distances.reshape(-1).astype(np.float32)) ** 2
    g = np.exp(-sqdist / max(float(tau_r), float(eps))).astype(np.float32)

    image_norm = row_normalize_sparse_graph(image_graph, eps=eps).tocsr()
    text_norm = row_normalize_sparse_graph(text_graph, eps=eps).tocsr()
    if bool(use_global_confidence):
        c_img = image_norm.copy()
        c_img.data = (c_img.data * float(rho_img)).astype(np.float32)
        c_txt = text_norm.copy()
        c_txt.data = (c_txt.data * float(rho_txt)).astype(np.float32)
    else:
        c_img = image_norm
        c_txt = text_norm

    union_keys, _, _ = build_union_keys(c_img, c_txt)
    c_img_union = gather_sparse_data_on_keys(c_img, union_keys)
    c_txt_union = gather_sparse_data_on_keys(c_txt, union_keys)
    num_cols = num_nodes
    edge_keys = rows * num_cols + cols
    positions = np.searchsorted(union_keys, edge_keys)

    sI = np.zeros(edge_keys.shape[0], dtype=np.float32)
    sT = np.zeros(edge_keys.shape[0], dtype=np.float32)
    valid = (positions < union_keys.shape[0]) & (union_keys[positions] == edge_keys)
    sI[valid] = c_img_union[positions[valid]]
    sT[valid] = c_txt_union[positions[valid]]

    mixed = float(eta) * np.sqrt(np.maximum(sI, 0.0) * np.maximum(sT, 0.0))
    mixed += (1.0 - float(eta)) * (sI + sT) / 2.0
    weights = (g * mixed).astype(np.float32)

    row_sum = np.bincount(rows, weights=weights, minlength=num_nodes).astype(np.float32)
    row_sum = np.maximum(row_sum, float(eps))
    normalized_weights = weights / row_sum[rows]

    keep_mask = normalized_weights > 0
    return {
        "row_index": rows[keep_mask].astype(np.int64),
        "col_index": cols[keep_mask].astype(np.int64),
        "weights": normalized_weights[keep_mask].astype(np.float32),
        "num_edges": int(np.sum(keep_mask)),
        "num_nodes": int(num_nodes),
    }


def compute_direct_coverage(proxy_points, real_points, tau_c=1.0, batch_size=4096, coverage_mode="mean"):
    tau_c = max(float(tau_c), 1e-8)
    coverages = []
    for batch in real_points.split(int(batch_size), dim=0):
        sqdist = torch.cdist(batch, proxy_points, p=2) ** 2
        coverage = torch.exp(-sqdist / tau_c).sum(dim=1)
        if coverage_mode == "mean":
            coverage = coverage / max(int(proxy_points.shape[0]), 1)
        elif coverage_mode != "sum":
            raise ValueError(f"Unsupported LSRC coverage mode: {coverage_mode}")
        coverages.append(coverage)
    return torch.cat(coverages, dim=0) if coverages else torch.empty(0, dtype=proxy_points.dtype, device=proxy_points.device)


def compute_lsrc_losses(
    proxy_points,
    real_points,
    relation_graph,
    tau_c=1.0,
    beta=0.5,
    eps=1e-8,
    batch_size=4096,
    coverage_mode="mean",
    rel_loss_mode="weight_mean",
):
    coverage = compute_direct_coverage(
        proxy_points,
        real_points,
        tau_c=tau_c,
        batch_size=batch_size,
        coverage_mode=coverage_mode,
    )
    if relation_graph is None or relation_graph["num_edges"] <= 0:
        indirect = coverage
        rel_loss = torch.tensor(0.0, dtype=proxy_points.dtype, device=proxy_points.device)
    else:
        row_index = relation_graph["row_index"].to(device=proxy_points.device)
        col_index = relation_graph["col_index"].to(device=proxy_points.device)
        weights = relation_graph["weights"].to(device=proxy_points.device, dtype=proxy_points.dtype)
        propagated = torch.zeros_like(coverage)
        propagated.index_add_(0, row_index, weights * coverage[col_index])
        indirect = (1.0 - float(beta)) * coverage + float(beta) * propagated
        diff_sq = (coverage[row_index] - coverage[col_index]) ** 2
        if rel_loss_mode == "weight_mean":
            rel_loss = torch.sum(weights * diff_sq) / torch.clamp(torch.sum(weights), min=float(eps))
        elif rel_loss_mode == "edge_mean":
            rel_loss = torch.mean(weights * diff_sq)
        else:
            raise ValueError(f"Unsupported LSRC relation loss mode: {rel_loss_mode}")

    cov_loss = -torch.mean(torch.log(indirect + float(eps)))
    return cov_loss, rel_loss, coverage, indirect


def discrepancy_score(proxy_points, target_samples, frequencies, batch_size=4096):
    with torch.no_grad():
        proxy_real, proxy_imag = compute_characteristic_function(proxy_points, frequencies, batch_size=batch_size)
        target_real, target_imag = compute_characteristic_function(target_samples, frequencies, batch_size=batch_size)
        return (proxy_real - target_real) ** 2 + (proxy_imag - target_imag) ** 2


def schedule_pdas_frequencies(
    omega_pool,
    proxy_points,
    full_points,
    step,
    total_steps,
    tau_min,
    tau_max,
    top_k,
    batch_size=4096,
):
    step = int(step)
    total_steps = max(int(total_steps), 1)
    tau_t = float(tau_min) + (float(tau_max) - float(tau_min)) * float(step) / float(total_steps)
    omega_norm = torch.linalg.norm(omega_pool, dim=1)
    cand_mask = omega_norm <= float(tau_t)
    if not torch.any(cand_mask):
        cand_mask = omega_norm <= float(tau_max)
    candidate_freq = omega_pool[cand_mask]
    if candidate_freq.shape[0] <= int(top_k):
        return candidate_freq

    scores = discrepancy_score(proxy_points, full_points, candidate_freq, batch_size=batch_size)
    selected = torch.topk(scores, k=int(top_k), largest=True).indices
    return candidate_freq[selected]


def optimize_proxy_points(
    representation,
    subset_size,
    device="cpu",
    projection_dim=128,
    random_state=0,
    init_method="kmeans",
    minibatch_size=2048,
    num_frequencies=64,
    frequency_scale=1.0,
    lr=5e-2,
    num_steps=200,
    reg_weight=1e-2,
    target_batch_size=4096,
    proxy_batch_size=4096,
    objective_mode="cfd",
    use_pdas=False,
    pdas_num_stages=4,
    pdas_schedule_mode="low_to_high",
    use_dpp=False,
    lambda_div=0.0,
    lambda_match=0.0,
    lambda_graph=0.0,
    diversity_sigma=1.0,
    diversity_kernel="rbf",
    phase_weight_mode="uniform",
    lambda_phase=0.1,
    num_freq_pool=None,
    tau_min=0.1,
    tau_max=None,
    match_reference=None,
    graph_reference=None,
    enable_lsrc=False,
    lsrc_image_graph=None,
    lsrc_text_graph=None,
    lsrc_k=32,
    lsrc_tau_r=1.0,
    lsrc_tau_c=1.0,
    lsrc_eta=0.5,
    lsrc_beta=0.5,
    lambda_lsrc_cov=0.0,
    lambda_lsrc_rel=0.0,
    lsrc_eps=1e-8,
    lsrc_batch_size=4096,
    lsrc_rho_img=0.5,
    lsrc_rho_txt=0.5,
    lsrc_use_global_confidence=False,
    lsrc_coverage_mode="mean",
    lsrc_rel_loss_mode="weight_mean",
):
    projected_representation, projection_matrix = random_project_representation(
        representation,
        projection_dim=projection_dim,
        random_state=random_state,
    )
    projected_representation = l2_normalize(projected_representation.astype(np.float32))
    projected_match_reference = project_reference_points(match_reference, projection_matrix)
    projected_graph_reference = project_reference_points(graph_reference, projection_matrix)
    lsrc_reference = projected_representation.astype(np.float32)
    lsrc_relation_graph = None
    if bool(enable_lsrc):
        if lsrc_image_graph is None or lsrc_text_graph is None:
            raise ValueError("LSRC is enabled but image/text graphs were not provided.")
        lsrc_relation_graph_np = build_lsrc_relation_graph(
            lsrc_reference,
            lsrc_image_graph,
            lsrc_text_graph,
            k=lsrc_k,
            tau_r=lsrc_tau_r,
            eta=lsrc_eta,
            rho_img=lsrc_rho_img,
            rho_txt=lsrc_rho_txt,
            eps=lsrc_eps,
            use_global_confidence=lsrc_use_global_confidence,
        )
    else:
        lsrc_relation_graph_np = None

    init_points, init_info = initialize_proxy_points(
        projected_representation,
        subset_size=subset_size,
        init_method=init_method,
        random_state=random_state,
        minibatch_size=minibatch_size,
    )

    torch_device = torch.device(device)
    full_repr = torch.tensor(projected_representation, dtype=torch.float32, device=torch_device)
    if lsrc_relation_graph_np is not None:
        lsrc_relation_graph = {
            "row_index": torch.tensor(lsrc_relation_graph_np["row_index"], dtype=torch.long, device=torch_device),
            "col_index": torch.tensor(lsrc_relation_graph_np["col_index"], dtype=torch.long, device=torch_device),
            "weights": torch.tensor(lsrc_relation_graph_np["weights"], dtype=torch.float32, device=torch_device),
            "num_edges": int(lsrc_relation_graph_np["num_edges"]),
            "num_nodes": int(lsrc_relation_graph_np["num_nodes"]),
        }
    proxy_init = torch.tensor(init_points, dtype=torch.float32, device=torch_device)
    proxy_points = torch.nn.Parameter(proxy_init.clone())
    if projected_match_reference is not None:
        match_reference = torch.tensor(projected_match_reference, dtype=torch.float32, device=torch_device)
    else:
        match_reference = None
    if projected_graph_reference is not None:
        graph_reference = torch.tensor(projected_graph_reference, dtype=torch.float32, device=torch_device)
    else:
        graph_reference = None

    if tau_max is None:
        tau_max = frequency_scale
    if num_freq_pool is None:
        num_freq_pool = num_frequencies

    initial_frequencies = sample_frequency_pool(
        dim=projected_representation.shape[1],
        num_frequencies=num_freq_pool,
        tau_max=tau_max,
        device=torch_device,
        random_state=random_state,
    )
    current_frequencies = initial_frequencies
    phase_weights = build_phase_weights(num_frequencies, mode=phase_weight_mode, device=torch_device)

    optimizer = torch.optim.Adam([proxy_points], lr=float(lr))
    history = []

    for step in range(int(num_steps)):
        if use_pdas:
            stage = min(int(step * max(int(pdas_num_stages), 1) / max(int(num_steps), 1)), max(int(pdas_num_stages) - 1, 0))
            if pdas_schedule_mode == "low_to_high":
                current_frequencies = schedule_pdas_frequencies(
                    initial_frequencies,
                    proxy_points.detach(),
                    full_repr,
                    step=step,
                    total_steps=num_steps,
                    tau_min=tau_min,
                    tau_max=tau_max,
                    top_k=num_frequencies,
                    batch_size=proxy_batch_size,
                )
            else:
                current_frequencies = sample_pdas_frequencies(
                    dim=projected_representation.shape[1],
                    num_frequencies=num_frequencies,
                    frequency_scale=frequency_scale,
                    stage=stage,
                    num_stages=pdas_num_stages,
                    schedule_mode=pdas_schedule_mode,
                    device=torch_device,
                    random_state=random_state,
                )
            phase_weights = build_phase_weights(current_frequencies.shape[0], mode=phase_weight_mode, device=torch_device)
        else:
            stage = 0
            if current_frequencies.shape[0] != int(num_frequencies):
                current_frequencies = initial_frequencies[: int(num_frequencies)]
                phase_weights = build_phase_weights(current_frequencies.shape[0], mode=phase_weight_mode, device=torch_device)

        optimizer.zero_grad(set_to_none=True)
        if objective_mode == "pd_cfd":
            freq_loss = pd_cfd_loss(
                proxy_points,
                full_repr,
                current_frequencies,
                phase_weights=phase_weights,
                lambda_phase=lambda_phase,
                batch_size=proxy_batch_size,
            )
        elif objective_mode == "cfd":
            freq_loss = cfd_loss(
                proxy_points,
                full_repr,
                current_frequencies,
                batch_size=proxy_batch_size,
            )
        else:
            raise ValueError(f"Unsupported proxy objective mode: {objective_mode}")

        reg_loss = torch.mean((proxy_points - proxy_init) ** 2)
        div_loss = dpp_diversity_loss(
            proxy_points,
            kernel=diversity_kernel,
            sigma=diversity_sigma,
        ) if use_dpp or float(lambda_div) > 0 else torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
        match_loss = nearest_reference_loss(proxy_points, match_reference)
        graph_loss = nearest_reference_loss(proxy_points, graph_reference)
        if bool(enable_lsrc):
            lsrc_cov_loss, lsrc_rel_loss, _, _ = compute_lsrc_losses(
                proxy_points,
                full_repr,
                lsrc_relation_graph,
                tau_c=lsrc_tau_c,
                beta=lsrc_beta,
                eps=lsrc_eps,
                batch_size=lsrc_batch_size,
                coverage_mode=lsrc_coverage_mode,
                rel_loss_mode=lsrc_rel_loss_mode,
            )
        else:
            lsrc_cov_loss = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
            lsrc_rel_loss = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)

        loss = (
            freq_loss
            + float(reg_weight) * reg_loss
            + float(lambda_div) * div_loss
            + float(lambda_match) * match_loss
            + float(lambda_graph) * graph_loss
            + float(lambda_lsrc_cov) * lsrc_cov_loss
            + float(lambda_lsrc_rel) * lsrc_rel_loss
        )
        loss.backward()
        optimizer.step()

        if step in {0, int(num_steps) - 1} or (step + 1) % max(1, int(num_steps) // 10) == 0:
            history.append(
                {
                    "step": int(step + 1),
                    "total_loss": float(loss.detach().cpu().item()),
                    "frequency_loss": float(freq_loss.detach().cpu().item()),
                    "reg_loss": float(reg_loss.detach().cpu().item()),
                    "div_loss": float(div_loss.detach().cpu().item()),
                    "match_loss": float(match_loss.detach().cpu().item()),
                    "graph_loss": float(graph_loss.detach().cpu().item()),
                    "lsrc_cov": float(lsrc_cov_loss.detach().cpu().item()),
                    "lsrc_rel": float(lsrc_rel_loss.detach().cpu().item()),
                    "lsrc_total": float(
                        (float(lambda_lsrc_cov) * lsrc_cov_loss + float(lambda_lsrc_rel) * lsrc_rel_loss)
                        .detach()
                        .cpu()
                        .item()
                    ),
                    "pdas_stage": int(stage),
                }
            )

    optimized = proxy_points.detach().cpu().numpy().astype(np.float32)
    proxy_init_np = proxy_init.detach().cpu().numpy().astype(np.float32)

    summary = {
        "projection_dim": int(projected_representation.shape[1]),
        "num_frequencies": int(num_frequencies),
        "frequency_scale": float(frequency_scale),
        "num_steps": int(num_steps),
        "lr": float(lr),
        "reg_weight": float(reg_weight),
        "objective_mode": objective_mode,
        "use_pdas": bool(use_pdas),
        "pdas_num_stages": int(pdas_num_stages),
        "pdas_schedule_mode": pdas_schedule_mode,
        "lambda_phase": float(lambda_phase),
        "num_freq_pool": int(num_freq_pool),
        "tau_min": float(tau_min),
        "tau_max": float(tau_max),
        "use_dpp": bool(use_dpp or float(lambda_div) > 0),
        "lambda_div": float(lambda_div),
        "lambda_match": float(lambda_match),
        "lambda_graph": float(lambda_graph),
        "enable_lsrc": bool(enable_lsrc),
        "lsrc_k": int(lsrc_k),
        "lsrc_tau_r": float(lsrc_tau_r),
        "lsrc_tau_c": float(lsrc_tau_c),
        "lsrc_eta": float(lsrc_eta),
        "lsrc_beta": float(lsrc_beta),
        "lambda_lsrc_cov": float(lambda_lsrc_cov),
        "lambda_lsrc_rel": float(lambda_lsrc_rel),
        "lsrc_eps": float(lsrc_eps),
        "lsrc_num_edges": int(lsrc_relation_graph["num_edges"]) if lsrc_relation_graph is not None else 0,
        "lsrc_use_global_confidence": bool(lsrc_use_global_confidence),
        "lsrc_coverage_mode": lsrc_coverage_mode,
        "lsrc_rel_loss_mode": lsrc_rel_loss_mode,
        "init_method": init_info["init_method"],
        "initial_loss": float(history[0]["total_loss"]) if history else None,
        "final_loss": float(history[-1]["total_loss"]) if history else None,
        "history": history,
    }

    return {
        "projected_representation": projected_representation,
        "projection_matrix": projection_matrix,
        "proxy_init": proxy_init_np,
        "proxy_points": optimized,
        "frequencies": current_frequencies.detach().cpu().numpy().astype(np.float32),
        "initial_frequencies": initial_frequencies.detach().cpu().numpy().astype(np.float32),
        "summary": summary,
    }
