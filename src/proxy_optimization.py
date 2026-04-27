import math

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from src.graph_wavelet import build_multi_scale_wavelet_signatures, parse_wavelet_scales, resolve_active_scales


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


def infer_mmd_bandwidth(x, y, eps=1e-8, max_points=1024):
    x = x.detach()
    y = y.detach()
    max_points = max(2, int(max_points))
    if x.shape[0] > max_points:
        x = x[torch.randperm(x.shape[0], device=x.device)[:max_points]]
    if y.shape[0] > max_points:
        y = y[torch.randperm(y.shape[0], device=y.device)[:max_points]]
    combined = torch.cat([x, y], dim=0)
    if combined.shape[0] <= 1:
        return float(1.0)
    sqdist = torch.cdist(combined, combined, p=2) ** 2
    mask = sqdist > float(eps)
    if not torch.any(mask):
        return float(1.0)
    median_sqdist = torch.median(sqdist[mask]).detach().cpu().item()
    bandwidth = math.sqrt(max(float(median_sqdist), float(eps)) / 2.0)
    return float(max(bandwidth, 1e-6))


def rbf_kernel_between(x, y, sigma):
    sigma = max(float(sigma), 1e-6)
    sqdist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-sqdist / (2.0 * sigma * sigma))


def diffusion_mmd_loss(
    proxy_points,
    target_samples,
    kernel="rbf",
    bandwidth=None,
    use_median_heuristic=True,
    eps=1e-8,
):
    if kernel != "rbf":
        raise ValueError(f"Unsupported MMD kernel: {kernel}")
    if bandwidth is None and use_median_heuristic:
        bandwidth = infer_mmd_bandwidth(proxy_points, target_samples, eps=eps)
    bandwidth = 1.0 if bandwidth is None else float(max(bandwidth, eps))

    k_xx = rbf_kernel_between(proxy_points, proxy_points, sigma=bandwidth)
    k_yy = rbf_kernel_between(target_samples, target_samples, sigma=bandwidth)
    k_xy = rbf_kernel_between(proxy_points, target_samples, sigma=bandwidth)

    m = max(int(proxy_points.shape[0]), 1)
    n = max(int(target_samples.shape[0]), 1)
    mean_xx = torch.sum(k_xx) / float(m * m)
    mean_yy = torch.sum(k_yy) / float(n * n)
    mean_xy = torch.sum(k_xy) / float(m * n)
    return mean_xx + mean_yy - 2.0 * mean_xy


def sample_swd_projections(dim, num_projections, device, dtype=torch.float32, random_state=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(random_state))
    projections = torch.randn((int(num_projections), int(dim)), generator=generator, dtype=torch.float32)
    projections = projections / torch.clamp(torch.linalg.norm(projections, dim=1, keepdim=True), min=1e-8)
    return projections.to(device=device, dtype=dtype)


def interpolate_sorted_quantiles(sorted_values, num_quantiles):
    num_quantiles = max(1, int(num_quantiles))
    if sorted_values.shape[1] == 1:
        return sorted_values.expand(-1, num_quantiles)
    q = (torch.arange(num_quantiles, device=sorted_values.device, dtype=sorted_values.dtype) + 0.5) / float(num_quantiles)
    positions = q * float(sorted_values.shape[1] - 1)
    low = torch.floor(positions).long()
    high = torch.ceil(positions).long()
    weight_high = (positions - low.to(dtype=sorted_values.dtype)).reshape(1, -1)
    weight_low = 1.0 - weight_high
    low_values = torch.gather(sorted_values, 1, low.reshape(1, -1).expand(sorted_values.shape[0], -1))
    high_values = torch.gather(sorted_values, 1, high.reshape(1, -1).expand(sorted_values.shape[0], -1))
    return weight_low * low_values + weight_high * high_values


def diffusion_swd_loss(
    proxy_points,
    target_samples,
    num_projections=64,
    p=2,
    projections=None,
):
    if projections is None:
        raise ValueError("diffusion_swd_loss requires sampled projections.")
    p = max(float(p), 1.0)
    proxy_proj = torch.matmul(proxy_points, projections.t()).transpose(0, 1)
    target_proj = torch.matmul(target_samples, projections.t()).transpose(0, 1)
    proxy_sorted, _ = torch.sort(proxy_proj, dim=1)
    target_sorted, _ = torch.sort(target_proj, dim=1)

    # When N != M, align the two 1D empirical measures on a shared quantile grid.
    # This keeps the implementation stable and differentiable without assuming equal set sizes.
    num_quantiles = max(int(proxy_points.shape[0]), int(target_samples.shape[0]))
    proxy_quantiles = interpolate_sorted_quantiles(proxy_sorted, num_quantiles=num_quantiles)
    target_quantiles = interpolate_sorted_quantiles(target_sorted, num_quantiles=num_quantiles)
    diff = torch.abs(proxy_quantiles - target_quantiles)
    return torch.mean(diff ** p)


def maybe_subsample_points(points, batch_size):
    if batch_size is None:
        return points
    batch_size = int(batch_size)
    if batch_size <= 0 or points.shape[0] <= batch_size:
        return points
    indices = torch.randperm(points.shape[0], device=points.device)[:batch_size]
    return points[indices]


def interpolate_proxy_wavelet_signature(proxy_points, anchor_points, anchor_signatures, tau):
    tau = max(float(tau), 1e-8)
    sqdist = torch.cdist(proxy_points, anchor_points, p=2) ** 2
    weights = torch.softmax(-sqdist / tau, dim=1)
    return weights @ anchor_signatures


def parse_scale_weight_values(scales, raw):
    scales = parse_wavelet_scales(scales)
    if raw is None:
        return None
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        items = list(raw)
    if len(items) == 1 and len(scales) > 1:
        return {int(scale): float(items[0]) for scale in scales}
    if len(items) != len(scales):
        raise ValueError(f"Scale weight count mismatch: got {len(items)} for {len(scales)} scales ({scales})")
    return {int(scale): float(item) for scale, item in zip(scales, items)}


def resolve_explicit_active_scale_weights(scales, active_scales, raw_weights, eps=1e-8):
    active_scales = [int(scale) for scale in active_scales]
    parsed = parse_scale_weight_values(scales, raw_weights)
    if parsed is None:
        weight = 1.0 / max(len(active_scales), 1)
        return {int(scale): float(weight) for scale in active_scales}
    selected = {int(scale): float(parsed[int(scale)]) for scale in active_scales}
    total = sum(abs(value) for value in selected.values())
    total = max(float(total), float(eps))
    return {int(scale): float(value / total) for scale, value in selected.items()}


def compute_wavelet_coverage_loss(proxy_signature, target_signature, tau=1.0, eps=1e-8):
    tau = max(float(tau), float(eps))
    sqdist = torch.cdist(target_signature, proxy_signature, p=2) ** 2
    coverage = torch.exp(-sqdist / tau).sum(dim=1) / max(int(proxy_signature.shape[0]), 1)
    loss = -torch.mean(torch.log(coverage + float(eps)))
    return loss, coverage


def compute_wavelet_edge_loss(proxy_signature, target_signature, eps=1e-8):
    sqdist = torch.cdist(target_signature, proxy_signature, p=2) ** 2
    nearest = torch.min(sqdist, dim=1).values
    energy = torch.linalg.norm(target_signature, dim=1)
    if energy.numel() == 0:
        return torch.tensor(0.0, dtype=proxy_signature.dtype, device=proxy_signature.device), energy
    weights = energy / torch.clamp(torch.mean(energy), min=float(eps))
    loss = torch.mean(weights * nearest)
    return loss, weights


def resolve_proxy_loss_type(proxy_loss_type, objective_mode=None):
    active_loss_type = str(proxy_loss_type or "").strip().lower()
    if not active_loss_type:
        objective_mode = str(objective_mode or "").strip().lower()
        if objective_mode == "pd_cfd":
            active_loss_type = "pdcfd"
        elif objective_mode == "cfd":
            active_loss_type = "cfd"
        else:
            active_loss_type = "wavelet_main"
    alias_map = {
        "pd_cfd": "pdcfd",
        "legacy_pdcfd": "pdcfd",
        "legacy_cfd": "cfd",
        "legacy_diffusion_mmd": "diffusion_mmd",
        "legacy_diffusion_swd": "diffusion_swd",
        "legacy_diffusion_ms_swd": "diffusion_ms_swd",
        "legacy_diffusion": "diffusion_ms_swd",
    }
    return alias_map.get(active_loss_type, active_loss_type)


def dpp_diversity_loss(proxy_points, kernel="rbf", sigma=1.0, eps=1e-6, max_points=2048):
    if kernel != "rbf":
        raise ValueError(f"Unsupported diversity kernel: {kernel}")
    # Full DPP logdet scales as O(M^2) memory and O(M^3) compute in the number
    # of proxy points. Large-ratio runs can have tens of thousands of proxies,
    # so cap the regularizer to a random proxy subset to keep it diagnostic
    # without dominating GPU memory.
    if max_points is not None and int(max_points) > 0 and proxy_points.shape[0] > int(max_points):
        indices = torch.randperm(proxy_points.shape[0], device=proxy_points.device)[: int(max_points)]
        proxy_points = proxy_points[indices]
    kernel_matrix = build_rbf_kernel(proxy_points, sigma=sigma)
    eye = torch.eye(kernel_matrix.shape[0], dtype=kernel_matrix.dtype, device=kernel_matrix.device)
    stabilized = kernel_matrix + float(eps) * eye
    sign, logdet = torch.linalg.slogdet(stabilized)
    if torch.any(sign <= 0):
        return torch.tensor(0.0, dtype=proxy_points.dtype, device=proxy_points.device)
    return -logdet / max(int(proxy_points.shape[0]), 1)


def resolve_reference_loss_devices(primary_device):
    primary_device = torch.device(primary_device)
    if primary_device.type != "cuda":
        return [primary_device]

    primary_index = primary_device.index
    if primary_index is None:
        primary_index = torch.cuda.current_device()
    primary_device = torch.device(f"cuda:{int(primary_index)}")

    devices = []
    seen = set()
    for device in [primary_device] + [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]:
        key = str(device)
        if key in seen:
            continue
        seen.add(key)
        devices.append(device)
    return devices


def resolve_safe_proxy_chunk_size(proxy_batch_size=None, lsrc_batch_size=None, default=1024):
    candidates = [int(default)]
    for value in (proxy_batch_size, lsrc_batch_size):
        if value is not None and int(value) > 0:
            candidates.append(int(value))
    return max(1, min(candidates))


def compute_direct_coverage(
    proxy_points,
    real_points,
    tau_c=1.0,
    batch_size=4096,
    coverage_mode="mean",
    proxy_chunk_size=None,
):
    tau_c = max(float(tau_c), 1e-8)
    batch_size = max(1, int(batch_size))
    proxy_chunk_size = resolve_safe_proxy_chunk_size(
        proxy_batch_size=proxy_chunk_size,
        lsrc_batch_size=batch_size,
        default=1024,
    )

    coverages = []
    for batch in real_points.split(batch_size, dim=0):
        coverage = torch.zeros(batch.shape[0], dtype=proxy_points.dtype, device=batch.device)
        for proxy_chunk in proxy_points.split(proxy_chunk_size, dim=0):
            sqdist = torch.cdist(batch, proxy_chunk, p=2) ** 2
            coverage = coverage + torch.exp(-sqdist / tau_c).sum(dim=1)
        if coverage_mode == "mean":
            coverage = coverage / max(int(proxy_points.shape[0]), 1)
        elif coverage_mode != "sum":
            raise ValueError(f"Unsupported LSRC coverage mode: {coverage_mode}")
        coverages.append(coverage)
    return torch.cat(coverages, dim=0) if coverages else torch.empty(0, dtype=proxy_points.dtype, device=proxy_points.device)


def build_reference_shards(reference_points, devices, shard_size=16384):
    if reference_points is None:
        return []
    shard_size = max(1, int(shard_size))
    devices = [torch.device(device) for device in devices] if devices else [reference_points.device]
    shards = []
    shard_index = 0
    for start in range(0, int(reference_points.shape[0]), shard_size):
        end = min(int(reference_points.shape[0]), start + shard_size)
        target_device = devices[shard_index % len(devices)]
        shards.append(reference_points[start:end].to(device=target_device))
        shard_index += 1
    return shards


def nearest_reference_loss(
    proxy_points,
    reference_points=None,
    reference_shards=None,
    proxy_chunk_size=2048,
):
    if reference_points is None and not reference_shards:
        return torch.tensor(0.0, dtype=proxy_points.dtype, device=proxy_points.device)
    if reference_shards is None or len(reference_shards) == 0:
        reference_shards = [reference_points]

    proxy_chunk_size = max(1, int(proxy_chunk_size))
    best_values = []
    for start in range(0, int(proxy_points.shape[0]), proxy_chunk_size):
        end = min(int(proxy_points.shape[0]), start + proxy_chunk_size)
        proxy_chunk = proxy_points[start:end]
        best_chunk = None
        for shard in reference_shards:
            local_proxy = proxy_chunk if shard.device == proxy_chunk.device else proxy_chunk.to(device=shard.device)
            local_distances = torch.cdist(local_proxy, shard, p=2) ** 2
            local_best = torch.min(local_distances, dim=1).values
            local_best = local_best.to(device=proxy_points.device)
            best_chunk = local_best if best_chunk is None else torch.minimum(best_chunk, local_best)
        best_values.append(best_chunk)
    return torch.mean(torch.cat(best_values, dim=0)) if best_values else torch.tensor(0.0, dtype=proxy_points.dtype, device=proxy_points.device)


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
    distance_scale=1.0,
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
    sigma = max(float(tau_r) * float(distance_scale), float(eps))
    g = np.exp(-sqdist / (sigma * sigma)).astype(np.float32)

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
        "distance_scale": float(distance_scale),
    }


def compute_lsrc_losses(
    proxy_points,
    real_points,
    relation_graph,
    tau_c=1.0,
    beta=0.5,
    eps=1e-8,
    batch_size=4096,
    proxy_chunk_size=None,
    coverage_mode="mean",
    rel_loss_mode="weight_mean",
):
    coverage = compute_direct_coverage(
        proxy_points,
        real_points,
        tau_c=tau_c,
        batch_size=batch_size,
        proxy_chunk_size=proxy_chunk_size,
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


def compute_multiscale_lsrc_outputs(
    proxy_points,
    real_points,
    relation_graphs,
    active_scales,
    scale_weights,
    tau_c=1.0,
    beta=0.5,
    eps=1e-8,
    batch_size=4096,
    proxy_chunk_size=None,
    coverage_mode="mean",
    rel_loss_mode="weight_mean",
):
    aggregated_cov_loss = torch.tensor(0.0, dtype=proxy_points.dtype, device=proxy_points.device)
    aggregated_rel_loss = torch.tensor(0.0, dtype=proxy_points.dtype, device=proxy_points.device)
    aggregated_coverage = None
    aggregated_indirect = None
    per_scale = {}

    for scale in active_scales:
        relation_graph = relation_graphs[int(scale)]
        cov_loss, rel_loss, coverage, indirect = compute_lsrc_losses(
            proxy_points,
            real_points,
            relation_graph,
            tau_c=tau_c,
            beta=beta,
            eps=eps,
            batch_size=batch_size,
            proxy_chunk_size=proxy_chunk_size,
            coverage_mode=coverage_mode,
            rel_loss_mode=rel_loss_mode,
        )
        weight = float(scale_weights[int(scale)])
        aggregated_cov_loss = aggregated_cov_loss + weight * cov_loss
        aggregated_rel_loss = aggregated_rel_loss + weight * rel_loss
        if aggregated_coverage is None:
            aggregated_coverage = weight * coverage
            aggregated_indirect = weight * indirect
        else:
            aggregated_coverage = aggregated_coverage + weight * coverage
            aggregated_indirect = aggregated_indirect + weight * indirect
        per_scale[int(scale)] = {
            "cov_loss": float(cov_loss.detach().cpu().item()),
            "rel_loss": float(rel_loss.detach().cpu().item()),
            "weight": weight,
        }

    if aggregated_coverage is None:
        aggregated_coverage = torch.zeros(real_points.shape[0], dtype=proxy_points.dtype, device=proxy_points.device)
        aggregated_indirect = torch.zeros_like(aggregated_coverage)
    return {
        "loss_lsrc_cov": aggregated_cov_loss,
        "loss_lsrc_rel": aggregated_rel_loss,
        "coverage_direct": aggregated_coverage,
        "coverage_relational": aggregated_indirect,
        "confidence_q": aggregated_indirect,
        "per_scale": per_scale,
    }


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
    proxy_loss_type="wavelet_main",
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
    mmd_kernel="rbf",
    mmd_bandwidth=None,
    mmd_use_median_heuristic=True,
    use_wavelet_multiscale=False,
    wavelet_graph=None,
    wavelet_scales=None,
    wavelet_loss_weight=0.0,
    wavelet_distance_type="mmd",
    wavelet_schedule="coarse_to_fine",
    wavelet_swd_num_projections=None,
    wavelet_swd_p=None,
    lambda_diff=1.0,
    lambda_ms=None,
    lambda_lsrc=None,
    lsrc_mu=1.0,
    lambda_reg=1.0,
    reg_alpha_div=1.0,
    reg_beta_topo=1.0,
    reg_gamma_init=1.0,
    swd_num_projections=64,
    swd_p=2,
    swd_projection_seed=None,
    swd_use_fixed_projections=False,
    lambda_main=1.0,
    wavelet_main_scales=None,
    wavelet_main_scale_weights=None,
    wavelet_main_swd_num_projections=None,
    wavelet_cov_weight=0.5,
    wavelet_edge_weight=0.25,
    wavelet_curriculum_schedule="coarse_to_fine",
    keep_lsrc=True,
):
    active_loss_type = resolve_proxy_loss_type(proxy_loss_type, objective_mode=objective_mode)
    if active_loss_type not in {"wavelet_main", "diffusion_mmd", "diffusion_swd", "diffusion_ms_swd", "pdcfd", "cfd"}:
        raise ValueError(f"Unsupported proxy loss type: {active_loss_type}")
    effective_use_wavelet_multiscale = bool(use_wavelet_multiscale or active_loss_type == "wavelet_main")
    if effective_use_wavelet_multiscale and active_loss_type not in {"wavelet_main", "diffusion_mmd", "diffusion_swd", "diffusion_ms_swd"}:
        raise ValueError(
            "Graph wavelet multiscale loss currently requires wavelet_main or a diffusion-space proxy loss "
            "(diffusion_mmd, diffusion_swd, or diffusion_ms_swd)."
        )
    if active_loss_type == "diffusion_ms_swd" and not effective_use_wavelet_multiscale:
        raise ValueError("diffusion_ms_swd requires wavelet multiscale features and an active scale schedule.")
    if active_loss_type == "wavelet_main" and wavelet_graph is None:
        raise ValueError("wavelet_main requires unified graph wavelet signatures, but wavelet_graph is None.")

    lambda_diff = float(lambda_diff)
    lambda_ms_effective = float(wavelet_loss_weight if lambda_ms is None else lambda_ms)
    lambda_main = float(lambda_main)
    lambda_reg = float(lambda_reg)
    reg_alpha_div = float(reg_alpha_div)
    reg_beta_topo = float(reg_beta_topo)
    reg_gamma_init = float(reg_gamma_init)
    swd_num_projections = max(1, int(swd_num_projections))
    swd_p = max(float(swd_p), 1.0)
    swd_projection_seed = int(random_state if swd_projection_seed is None else swd_projection_seed)
    swd_use_fixed_projections = bool(swd_use_fixed_projections)
    wavelet_swd_num_projections = swd_num_projections if wavelet_swd_num_projections is None else max(1, int(wavelet_swd_num_projections))
    wavelet_swd_p = swd_p if wavelet_swd_p is None else max(float(wavelet_swd_p), 1.0)
    wavelet_main_swd_num_projections = (
        wavelet_swd_num_projections if wavelet_main_swd_num_projections is None else max(1, int(wavelet_main_swd_num_projections))
    )
    effective_wavelet_distance_type = "swd" if active_loss_type in {"diffusion_ms_swd", "wavelet_main"} else wavelet_distance_type
    legacy_lsrc_weight_mode = lambda_lsrc is None
    lambda_lsrc_effective = 1.0 if legacy_lsrc_weight_mode else float(lambda_lsrc)
    lsrc_mu_effective = float(lsrc_mu)
    effective_keep_lsrc = bool(keep_lsrc) or bool(enable_lsrc)

    projected_representation, projection_matrix = random_project_representation(
        representation,
        projection_dim=projection_dim,
        random_state=random_state,
    )
    projected_representation = l2_normalize(projected_representation.astype(np.float32))
    projected_match_reference = project_reference_points(match_reference, projection_matrix)
    projected_graph_reference = project_reference_points(graph_reference, projection_matrix)
    lsrc_reference = projected_representation.astype(np.float32)
    wavelet_graph_signatures = None
    wavelet_scales = parse_wavelet_scales(wavelet_scales)
    wavelet_main_scales = parse_wavelet_scales(wavelet_main_scales if wavelet_main_scales is not None else wavelet_scales)
    lsrc_relation_graph = None
    lsrc_relation_graphs = None
    if effective_keep_lsrc:
        if lsrc_image_graph is None or lsrc_text_graph is None:
            raise ValueError("LSRC is enabled but image/text graphs were not provided.")
        lsrc_scale_list = wavelet_main_scales if effective_use_wavelet_multiscale else [1]
        lsrc_relation_graph_np = {
            int(scale): build_lsrc_relation_graph(
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
                distance_scale=float(scale),
            )
            for scale in lsrc_scale_list
        }
    else:
        lsrc_relation_graph_np = None

    if effective_use_wavelet_multiscale:
        if wavelet_graph is None:
            raise ValueError("Wavelet multiscale loss is enabled but unified graph was not provided.")
        if effective_wavelet_distance_type not in {"mmd", "swd"}:
            raise ValueError(f"Unsupported wavelet distance type: {effective_wavelet_distance_type}")
        wavelet_graph_signatures = build_multi_scale_wavelet_signatures(
            wavelet_graph,
            projected_representation,
            wavelet_main_scales if active_loss_type == "wavelet_main" else wavelet_scales,
            normalize=True,
        )

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
        lsrc_relation_graphs = {
            int(scale): {
                "row_index": torch.tensor(graph_payload["row_index"], dtype=torch.long, device=torch_device),
                "col_index": torch.tensor(graph_payload["col_index"], dtype=torch.long, device=torch_device),
                "weights": torch.tensor(graph_payload["weights"], dtype=torch.float32, device=torch_device),
                "num_edges": int(graph_payload["num_edges"]),
                "num_nodes": int(graph_payload["num_nodes"]),
                "distance_scale": float(graph_payload.get("distance_scale", scale)),
            }
            for scale, graph_payload in lsrc_relation_graph_np.items()
        }
        lsrc_relation_graph = lsrc_relation_graphs[min(lsrc_relation_graphs.keys())]
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
    reference_loss_devices = resolve_reference_loss_devices(torch_device)
    reference_loss_shard_size = max(4096, min(16384, int(projected_representation.shape[0]) if projected_representation.shape[0] > 0 else 4096))
    reference_loss_proxy_chunk_size = max(512, int(proxy_batch_size) if proxy_batch_size is not None else 2048)
    lsrc_proxy_chunk_size = resolve_safe_proxy_chunk_size(
        proxy_batch_size=proxy_batch_size,
        lsrc_batch_size=lsrc_batch_size,
        default=1024,
    )
    match_reference_shards = build_reference_shards(
        match_reference,
        reference_loss_devices,
        shard_size=reference_loss_shard_size,
    ) if match_reference is not None else []
    graph_reference_shards = build_reference_shards(
        graph_reference,
        reference_loss_devices,
        shard_size=reference_loss_shard_size,
    ) if graph_reference is not None else []
    if len(reference_loss_devices) > 1 and (match_reference_shards or graph_reference_shards):
        print(
            "[proxy-opt] reference matching uses multi-GPU shards on "
            f"{[str(device) for device in reference_loss_devices]} "
            f"(shard_size={reference_loss_shard_size}, proxy_chunk_size={reference_loss_proxy_chunk_size})",
            flush=True,
        )
    elif match_reference_shards or graph_reference_shards:
        print(
            "[proxy-opt] reference matching uses chunked cdist on "
            f"{str(torch_device)} "
            f"(shard_size={reference_loss_shard_size}, proxy_chunk_size={reference_loss_proxy_chunk_size})",
            flush=True,
        )
    if effective_keep_lsrc:
        print(
            "[proxy-opt] LSRC coverage uses chunked cdist "
            f"(real_batch_size={int(lsrc_batch_size)}, "
            f"proxy_sample_size<={int(lsrc_proxy_chunk_size)}, "
            f"proxy_chunk_size={int(lsrc_proxy_chunk_size)})",
            flush=True,
        )

    if wavelet_graph_signatures is not None:
        anchor_count = min(int(projected_representation.shape[0]), 1024)
        rng = np.random.default_rng(int(random_state))
        anchor_indices_np = rng.choice(projected_representation.shape[0], size=anchor_count, replace=False)
        anchor_indices = torch.tensor(anchor_indices_np, dtype=torch.long, device=torch_device)
        wavelet_anchor_points = full_repr[anchor_indices]
        wavelet_scale_tensors = {
            int(scale): torch.tensor(wavelet_graph_signatures[int(scale)], dtype=torch.float32, device=torch_device)
            for scale in wavelet_scales
        }
        wavelet_anchor_signatures = {
            int(scale): wavelet_scale_tensors[int(scale)][anchor_indices]
            for scale in (wavelet_main_scales if active_loss_type == "wavelet_main" else wavelet_scales)
        }
        wavelet_interp_tau = infer_mmd_bandwidth(wavelet_anchor_points, wavelet_anchor_points)
        if effective_wavelet_distance_type == "swd" and swd_use_fixed_projections:
            wavelet_swd_fixed_projections = {
                int(scale): sample_swd_projections(
                    dim=wavelet_scale_tensors[int(scale)].shape[1],
                    num_projections=wavelet_swd_num_projections,
                    device=torch_device,
                    dtype=wavelet_scale_tensors[int(scale)].dtype,
                    random_state=swd_projection_seed + int(scale),
                )
                for scale in (wavelet_main_scales if active_loss_type == "wavelet_main" else wavelet_scales)
            }
        else:
            wavelet_swd_fixed_projections = None
    else:
        wavelet_anchor_points = None
        wavelet_scale_tensors = None
        wavelet_anchor_signatures = None
        wavelet_interp_tau = None
        wavelet_swd_fixed_projections = None

    if tau_max is None:
        tau_max = frequency_scale
    if num_freq_pool is None:
        num_freq_pool = num_frequencies

    initial_frequencies = None
    current_frequencies = None
    phase_weights = None
    if active_loss_type in {"pdcfd", "cfd"}:
        initial_frequencies = sample_frequency_pool(
            dim=projected_representation.shape[1],
            num_frequencies=num_freq_pool,
            tau_max=tau_max,
            device=torch_device,
            random_state=random_state,
        )
        current_frequencies = initial_frequencies
        phase_weights = build_phase_weights(num_frequencies, mode=phase_weight_mode, device=torch_device)
    if active_loss_type == "diffusion_swd" and swd_use_fixed_projections:
        fixed_swd_projections = sample_swd_projections(
            dim=projected_representation.shape[1],
            num_projections=swd_num_projections,
            device=torch_device,
            dtype=full_repr.dtype,
            random_state=swd_projection_seed,
        )
    else:
        fixed_swd_projections = None

    optimizer = torch.optim.Adam([proxy_points], lr=float(lr))
    history = []
    final_lsrc_state = None

    for step in range(int(num_steps)):
        if effective_use_wavelet_multiscale:
            active_scale_source = wavelet_main_scales if active_loss_type == "wavelet_main" else wavelet_scales
            active_schedule = wavelet_curriculum_schedule if active_loss_type == "wavelet_main" else wavelet_schedule
            active_scales, scale_weights = resolve_active_scales(
                active_scale_source,
                step=step,
                total_steps=num_steps,
                schedule=active_schedule,
            )
            if active_loss_type == "wavelet_main":
                scale_weights = resolve_explicit_active_scale_weights(
                    active_scale_source,
                    active_scales,
                    wavelet_main_scale_weights,
                )
        else:
            active_scales = [1]
            scale_weights = {1: 1.0}

        if active_loss_type in {"pdcfd", "cfd"} and use_pdas:
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
            if active_loss_type in {"pdcfd", "cfd"} and current_frequencies.shape[0] != int(num_frequencies):
                current_frequencies = initial_frequencies[: int(num_frequencies)]
                phase_weights = build_phase_weights(current_frequencies.shape[0], mode=phase_weight_mode, device=torch_device)

        optimizer.zero_grad(set_to_none=True)
        if active_loss_type == "diffusion_mmd":
            target_samples_for_loss = maybe_subsample_points(full_repr, target_batch_size)
            proxy_points_for_loss = maybe_subsample_points(proxy_points, proxy_batch_size)
            freq_loss = diffusion_mmd_loss(
                proxy_points_for_loss,
                target_samples_for_loss,
                kernel=mmd_kernel,
                bandwidth=mmd_bandwidth,
                use_median_heuristic=bool(mmd_use_median_heuristic),
            )
        elif active_loss_type == "diffusion_swd":
            target_samples_for_loss = maybe_subsample_points(full_repr, target_batch_size)
            proxy_points_for_loss = maybe_subsample_points(proxy_points, proxy_batch_size)
            current_swd_projections = fixed_swd_projections
            if current_swd_projections is None:
                current_swd_projections = sample_swd_projections(
                    dim=projected_representation.shape[1],
                    num_projections=swd_num_projections,
                    device=torch_device,
                    dtype=proxy_points.dtype,
                    random_state=swd_projection_seed + int(step),
                )
            freq_loss = diffusion_swd_loss(
                proxy_points_for_loss,
                target_samples_for_loss,
                num_projections=swd_num_projections,
                p=swd_p,
                projections=current_swd_projections,
            )
        elif active_loss_type in {"diffusion_ms_swd", "wavelet_main"}:
            freq_loss = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
        elif active_loss_type == "pdcfd":
            freq_loss = pd_cfd_loss(
                proxy_points,
                full_repr,
                current_frequencies,
                phase_weights=phase_weights,
                lambda_phase=lambda_phase,
                batch_size=proxy_batch_size,
            )
        elif active_loss_type == "cfd":
            freq_loss = cfd_loss(
                proxy_points,
                full_repr,
                current_frequencies,
                batch_size=proxy_batch_size,
            )
        else:
            raise ValueError(f"Unsupported proxy loss type: {active_loss_type}")

        init_loss = torch.mean((proxy_points - proxy_init) ** 2)
        div_loss = dpp_diversity_loss(
            proxy_points,
            kernel=diversity_kernel,
            sigma=diversity_sigma,
        ) if use_dpp or float(lambda_div) > 0 else torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
        match_loss = nearest_reference_loss(
            proxy_points,
            reference_points=match_reference,
            reference_shards=match_reference_shards,
            proxy_chunk_size=reference_loss_proxy_chunk_size,
        )
        graph_loss = nearest_reference_loss(
            proxy_points,
            reference_points=graph_reference,
            reference_shards=graph_reference_shards,
            proxy_chunk_size=reference_loss_proxy_chunk_size,
        )
        if effective_keep_lsrc:
            lsrc_proxy_points = proxy_points
            if proxy_points.shape[0] > int(lsrc_proxy_chunk_size):
                lsrc_proxy_indices = torch.randperm(proxy_points.shape[0], device=torch_device)[: int(lsrc_proxy_chunk_size)]
                lsrc_proxy_points = proxy_points[lsrc_proxy_indices]
            lsrc_outputs = compute_multiscale_lsrc_outputs(
                lsrc_proxy_points,
                full_repr,
                lsrc_relation_graphs,
                active_scales=active_scales,
                scale_weights=scale_weights,
                tau_c=lsrc_tau_c,
                beta=lsrc_beta,
                eps=lsrc_eps,
                batch_size=lsrc_batch_size,
                proxy_chunk_size=lsrc_proxy_chunk_size,
                coverage_mode=lsrc_coverage_mode,
                rel_loss_mode=lsrc_rel_loss_mode,
            )
            lsrc_cov_loss = lsrc_outputs["loss_lsrc_cov"]
            lsrc_rel_loss = lsrc_outputs["loss_lsrc_rel"]
            lsrc_q = lsrc_outputs["confidence_q"]
            final_lsrc_state = {
                "coverage_direct": lsrc_outputs["coverage_direct"].detach().cpu().numpy().astype(np.float32),
                "coverage_relational": lsrc_outputs["coverage_relational"].detach().cpu().numpy().astype(np.float32),
                "confidence_q": lsrc_q.detach().cpu().numpy().astype(np.float32),
                "active_scales": [int(scale) for scale in active_scales],
                "scale_weights": {str(scale): float(scale_weights[int(scale)]) for scale in active_scales},
                "per_scale": lsrc_outputs["per_scale"],
            }
        else:
            lsrc_cov_loss = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
            lsrc_rel_loss = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
            lsrc_q = torch.tensor([], dtype=proxy_points.dtype, device=torch_device)
            lsrc_outputs = None

        if effective_use_wavelet_multiscale:
            sampled_target_indices = None
            if target_batch_size is not None and int(target_batch_size) > 0 and full_repr.shape[0] > int(target_batch_size):
                sampled_target_indices = torch.randperm(full_repr.shape[0], device=torch_device)[: int(target_batch_size)]
            sampled_proxy_indices = None
            if proxy_batch_size is not None and int(proxy_batch_size) > 0 and proxy_points.shape[0] > int(proxy_batch_size):
                sampled_proxy_indices = torch.randperm(proxy_points.shape[0], device=torch_device)[: int(proxy_batch_size)]
            proxy_wavelet_points = proxy_points if sampled_proxy_indices is None else proxy_points[sampled_proxy_indices]

            loss_ms = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
            loss_wavelet_cov = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
            loss_wavelet_edge = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
            per_scale_losses = {}
            for scale in active_scales:
                target_signature = wavelet_scale_tensors[int(scale)]
                if sampled_target_indices is not None:
                    target_signature = target_signature[sampled_target_indices]
                proxy_signature = interpolate_proxy_wavelet_signature(
                    proxy_wavelet_points,
                    wavelet_anchor_points,
                    wavelet_anchor_signatures[int(scale)],
                    tau=wavelet_interp_tau,
                )
                if effective_wavelet_distance_type == "mmd":
                    scale_loss = diffusion_mmd_loss(
                        proxy_signature,
                        target_signature,
                        kernel=mmd_kernel,
                        bandwidth=mmd_bandwidth,
                        use_median_heuristic=bool(mmd_use_median_heuristic),
                    )
                else:
                    current_wavelet_projections = None
                    if wavelet_swd_fixed_projections is not None:
                        current_wavelet_projections = wavelet_swd_fixed_projections[int(scale)]
                    else:
                        current_wavelet_projections = sample_swd_projections(
                            dim=proxy_signature.shape[1],
                            num_projections=wavelet_swd_num_projections,
                            device=torch_device,
                            dtype=proxy_signature.dtype,
                            random_state=swd_projection_seed + int(step) * 997 + int(scale),
                        )
                    scale_loss = diffusion_swd_loss(
                        proxy_signature,
                        target_signature,
                        num_projections=wavelet_swd_num_projections,
                        p=wavelet_swd_p,
                        projections=current_wavelet_projections,
                    )
                per_scale_losses[int(scale)] = float(scale_loss.detach().cpu().item())
                loss_ms = loss_ms + float(scale_weights[int(scale)]) * scale_loss
                if active_loss_type == "wavelet_main":
                    coverage_loss, coverage_values = compute_wavelet_coverage_loss(
                        proxy_signature,
                        target_signature,
                        tau=wavelet_interp_tau,
                        eps=lsrc_eps,
                    )
                    edge_loss, edge_weights = compute_wavelet_edge_loss(
                        proxy_signature,
                        target_signature,
                        eps=lsrc_eps,
                    )
                    per_scale_losses[int(scale)] = {
                        "dist": float(scale_loss.detach().cpu().item()),
                        "cov": float(coverage_loss.detach().cpu().item()),
                        "edge": float(edge_loss.detach().cpu().item()),
                        "coverage_mean": float(coverage_values.detach().mean().cpu().item()),
                        "edge_weight_mean": float(edge_weights.detach().mean().cpu().item()) if edge_weights.numel() > 0 else 0.0,
                    }
                    loss_wavelet_cov = loss_wavelet_cov + float(scale_weights[int(scale)]) * coverage_loss
                    loss_wavelet_edge = loss_wavelet_edge + float(scale_weights[int(scale)]) * edge_loss
        else:
            per_scale_losses = {}
            loss_ms = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
            loss_wavelet_cov = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
            loss_wavelet_edge = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)

        old_global_swd = freq_loss if active_loss_type in {"diffusion_swd", "diffusion_ms_swd"} else torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
        if active_loss_type == "wavelet_main":
            loss_main = loss_ms + float(wavelet_cov_weight) * loss_wavelet_cov + float(wavelet_edge_weight) * loss_wavelet_edge
            loss_diff = loss_main
        elif active_loss_type == "diffusion_ms_swd":
            loss_diff = loss_ms
        else:
            loss_diff = freq_loss
        loss_topo = float(lambda_match) * match_loss + float(lambda_graph) * graph_loss
        loss_init = float(reg_weight) * init_loss
        loss_div = float(lambda_div) * div_loss
        loss_reg = reg_alpha_div * loss_div + reg_beta_topo * loss_topo + reg_gamma_init * loss_init
        loss_reg_block = lambda_reg * loss_reg
        loss_global_alignment = (lambda_main if active_loss_type == "wavelet_main" else lambda_diff) * loss_diff
        if active_loss_type in {"diffusion_ms_swd", "wavelet_main"}:
            loss_multiscale = torch.tensor(0.0, dtype=proxy_points.dtype, device=torch_device)
        else:
            loss_multiscale = lambda_ms_effective * loss_ms
        if legacy_lsrc_weight_mode:
            loss_lsrc = float(lambda_lsrc_cov) * lsrc_cov_loss + float(lambda_lsrc_rel) * lsrc_rel_loss
            loss_lsrc_block = loss_lsrc
        else:
            loss_lsrc = lsrc_cov_loss + lsrc_mu_effective * lsrc_rel_loss
            loss_lsrc_block = lambda_lsrc_effective * loss_lsrc

        loss = loss_global_alignment + loss_multiscale + loss_lsrc_block + loss_reg_block
        loss.backward()
        optimizer.step()

        if step in {0, int(num_steps) - 1} or (step + 1) % max(1, int(num_steps) // 10) == 0:
            if effective_use_wavelet_multiscale:
                print(
                    f"[proxy-opt] step={step + 1}/{int(num_steps)} active_scales={active_scales} "
                    f"loss_diff={float(loss_diff.detach().cpu().item()):.6f} "
                    f"loss_ms={float(loss_ms.detach().cpu().item()):.6f} "
                    f"loss_main={float((loss_main.detach().cpu().item() if active_loss_type == 'wavelet_main' else loss_diff.detach().cpu().item())):.6f} "
                    f"loss_lsrc={float(loss_lsrc_block.detach().cpu().item()):.6f} "
                    f"loss_reg={float(loss_reg_block.detach().cpu().item()):.6f} "
                    f"scale_weights={{{', '.join(f'{int(scale)}:{float(scale_weights[scale]):.3f}' for scale in active_scales)}}}",
                    flush=True,
                )
            history.append(
                {
                    "step": int(step + 1),
                    "total_loss": float(loss.detach().cpu().item()),
                    "loss_diff": float(loss_diff.detach().cpu().item()),
                    "loss_diff_type": active_loss_type,
                    "loss_main": float((loss_main.detach().cpu().item() if active_loss_type == "wavelet_main" else loss_diff.detach().cpu().item())),
                    "loss_ms": float(loss_ms.detach().cpu().item()),
                    "loss_wavelet_cov": float(loss_wavelet_cov.detach().cpu().item()),
                    "loss_wavelet_edge": float(loss_wavelet_edge.detach().cpu().item()),
                    "loss_global_alignment": float(loss_global_alignment.detach().cpu().item()),
                    "loss_multiscale": float(loss_multiscale.detach().cpu().item()),
                    "loss_lsrc_block": float(loss_lsrc_block.detach().cpu().item()),
                    "loss_reg_block": float(loss_reg_block.detach().cpu().item()),
                    "wavelet_distance_type": effective_wavelet_distance_type,
                    "frequency_loss": float(freq_loss.detach().cpu().item()),
                    "old_global_swd": float(old_global_swd.detach().cpu().item()) if active_loss_type in {"diffusion_swd", "diffusion_ms_swd"} else 0.0,
                    "old_loss_ms": float(loss_ms.detach().cpu().item()) if active_loss_type == "diffusion_swd" else 0.0,
                    "loss_reg": float(loss_reg.detach().cpu().item()),
                    "reg_loss": float(init_loss.detach().cpu().item()),
                    "loss_div": float(loss_div.detach().cpu().item()),
                    "loss_topo": float(loss_topo.detach().cpu().item()),
                    "loss_init": float(loss_init.detach().cpu().item()),
                    "div_loss": float(div_loss.detach().cpu().item()),
                    "match_loss": float(match_loss.detach().cpu().item()),
                    "graph_loss": float(graph_loss.detach().cpu().item()),
                    "loss_lsrc_cov": float(lsrc_cov_loss.detach().cpu().item()),
                    "loss_lsrc_rel": float(lsrc_rel_loss.detach().cpu().item()),
                    "loss_lsrc_total": float(loss_lsrc_block.detach().cpu().item()),
                    "lsrc_cov": float(lsrc_cov_loss.detach().cpu().item()),
                    "lsrc_rel": float(lsrc_rel_loss.detach().cpu().item()),
                    "lsrc_total": float(loss_lsrc_block.detach().cpu().item()),
                    "pdas_stage": int(stage),
                    "swd_num_projections": (
                        int(wavelet_swd_num_projections)
                        if active_loss_type == "diffusion_ms_swd"
                        else int(swd_num_projections) if active_loss_type == "diffusion_swd" else 0
                    ),
                    "active_scales": [int(scale) for scale in active_scales] if effective_use_wavelet_multiscale else [],
                    "wavelet_scale_weights": {str(scale): float(scale_weights[scale]) for scale in active_scales} if effective_use_wavelet_multiscale else {},
                    "wavelet_scale_losses": {
                        str(scale): (
                            {k: float(v) for k, v in per_scale_losses[scale].items()}
                            if isinstance(per_scale_losses[scale], dict)
                            else float(per_scale_losses[scale])
                        )
                        for scale in per_scale_losses
                    },
                    "lsrc_scale_weights": {str(scale): float(scale_weights[scale]) for scale in active_scales} if effective_keep_lsrc else {},
                }
            )
            if effective_use_wavelet_multiscale:
                if active_loss_type == "wavelet_main":
                    history[-1].update({f"loss_dist_scale_{int(scale)}": float(per_scale_losses[int(scale)]["dist"]) for scale in per_scale_losses})
                    history[-1].update({f"loss_cov_scale_{int(scale)}": float(per_scale_losses[int(scale)]["cov"]) for scale in per_scale_losses})
                    history[-1].update({f"loss_edge_scale_{int(scale)}": float(per_scale_losses[int(scale)]["edge"]) for scale in per_scale_losses})
                else:
                    history[-1].update({f"loss_ms_scale_{int(scale)}": float(per_scale_losses[int(scale)]) for scale in per_scale_losses})
                    history[-1].update({f"loss_diff_scale_{int(scale)}": float(per_scale_losses[int(scale)]) for scale in per_scale_losses})

    optimized = proxy_points.detach().cpu().numpy().astype(np.float32)
    proxy_init_np = proxy_init.detach().cpu().numpy().astype(np.float32)

    summary = {
        "projection_dim": int(projected_representation.shape[1]),
        "num_frequencies": int(num_frequencies),
        "frequency_scale": float(frequency_scale),
        "num_steps": int(num_steps),
        "lr": float(lr),
        "reg_weight": float(reg_weight),
        "proxy_loss_type": active_loss_type,
        "objective_mode": objective_mode,
        "loss_structure": (
            "lambda_main * L_main_wavelet + lambda_lsrc * L_lsrc + lambda_reg * L_reg"
            if active_loss_type == "wavelet_main"
            else
            "lambda_diff * L_diff_ms_swd + lambda_lsrc * L_lsrc + lambda_reg * L_reg"
            if active_loss_type == "diffusion_ms_swd"
            else "lambda_diff * L_diff + lambda_ms * L_ms + lambda_lsrc * L_lsrc + lambda_reg * L_reg"
        ),
        "lambda_main": float(lambda_main),
        "lambda_diff": float(lambda_diff),
        "lambda_ms": float(lambda_ms_effective),
        "lambda_lsrc": float(lambda_lsrc_effective),
        "lsrc_mu": float(lsrc_mu_effective),
        "lambda_reg": float(lambda_reg),
        "reg_alpha_div": float(reg_alpha_div),
        "reg_beta_topo": float(reg_beta_topo),
        "reg_gamma_init": float(reg_gamma_init),
        "legacy_lsrc_weight_mode": bool(legacy_lsrc_weight_mode),
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
        "reference_loss_multi_gpu": bool(len(reference_loss_devices) > 1),
        "reference_loss_devices": [str(device) for device in reference_loss_devices],
        "reference_loss_shard_size": int(reference_loss_shard_size),
        "reference_loss_proxy_chunk_size": int(reference_loss_proxy_chunk_size),
        "enable_lsrc": bool(enable_lsrc),
        "keep_lsrc": bool(effective_keep_lsrc),
        "lsrc_k": int(lsrc_k),
        "lsrc_tau_r": float(lsrc_tau_r),
        "lsrc_tau_c": float(lsrc_tau_c),
        "lsrc_eta": float(lsrc_eta),
        "lsrc_beta": float(lsrc_beta),
        "lambda_lsrc_cov": float(lambda_lsrc_cov),
        "lambda_lsrc_rel": float(lambda_lsrc_rel),
        "lsrc_eps": float(lsrc_eps),
        "lsrc_batch_size": int(lsrc_batch_size),
        "lsrc_proxy_chunk_size": int(lsrc_proxy_chunk_size),
        "lsrc_proxy_sample_size": int(min(proxy_points.shape[0], int(lsrc_proxy_chunk_size))),
        "lsrc_num_edges": int(sum(graph["num_edges"] for graph in lsrc_relation_graphs.values())) if lsrc_relation_graphs is not None else 0,
        "lsrc_use_global_confidence": bool(lsrc_use_global_confidence),
        "lsrc_coverage_mode": lsrc_coverage_mode,
        "lsrc_rel_loss_mode": lsrc_rel_loss_mode,
        "lsrc_geometry_space": "unified_projected_representation",
        "lsrc_multiscale_enabled": bool(effective_keep_lsrc and effective_use_wavelet_multiscale),
        "lsrc_scales": [int(scale) for scale in (wavelet_main_scales if bool(effective_keep_lsrc and effective_use_wavelet_multiscale) else [1])],
        "mmd_kernel": mmd_kernel,
        "mmd_bandwidth": None if mmd_bandwidth is None else float(mmd_bandwidth),
        "mmd_use_median_heuristic": bool(mmd_use_median_heuristic),
        "loss_diff_type": active_loss_type,
        "multiscale_as_main_alignment": bool(active_loss_type in {"diffusion_ms_swd", "wavelet_main"}),
        "swd_num_projections": int(swd_num_projections),
        "effective_main_swd_num_projections": int(wavelet_main_swd_num_projections) if active_loss_type == "wavelet_main" else int(wavelet_swd_num_projections) if active_loss_type == "diffusion_ms_swd" else int(swd_num_projections),
        "swd_p": float(swd_p),
        "swd_projection_seed": int(swd_projection_seed),
        "swd_use_fixed_projections": bool(swd_use_fixed_projections),
        "use_wavelet_multiscale": bool(effective_use_wavelet_multiscale),
        "wavelet_scales": [int(scale) for scale in wavelet_scales],
        "wavelet_main_scales": [int(scale) for scale in wavelet_main_scales],
        "wavelet_main_scale_weights": parse_scale_weight_values(wavelet_main_scales, wavelet_main_scale_weights),
        "wavelet_main_swd_num_projections": int(wavelet_main_swd_num_projections),
        "wavelet_cov_weight": float(wavelet_cov_weight),
        "wavelet_edge_weight": float(wavelet_edge_weight),
        "wavelet_curriculum_schedule": str(wavelet_curriculum_schedule),
        "wavelet_loss_weight": float(wavelet_loss_weight),
        "wavelet_distance_type": effective_wavelet_distance_type,
        "wavelet_swd_num_projections": int(wavelet_swd_num_projections) if effective_wavelet_distance_type == "swd" else 0,
        "wavelet_swd_p": float(wavelet_swd_p) if effective_wavelet_distance_type == "swd" else None,
        "wavelet_schedule": wavelet_schedule,
        "wavelet_interp_tau": None if wavelet_interp_tau is None else float(wavelet_interp_tau),
        "deprecated_proxy_config": {
            "proxy_objective_mode": objective_mode,
            "proxy_num_frequencies": int(num_frequencies),
            "proxy_frequency_scale": float(frequency_scale),
            "use_pdas": bool(use_pdas),
            "pdas_num_stages": int(pdas_num_stages),
            "pdas_schedule_mode": pdas_schedule_mode,
            "num_freq_pool": int(num_freq_pool),
            "tau_min": float(tau_min),
            "tau_max": float(tau_max),
            "lambda_phase": float(lambda_phase),
            "phase_weight_mode": phase_weight_mode,
            "status": "deprecated_compatibility_only",
        },
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
        "frequencies": None if current_frequencies is None else current_frequencies.detach().cpu().numpy().astype(np.float32),
        "initial_frequencies": None if initial_frequencies is None else initial_frequencies.detach().cpu().numpy().astype(np.float32),
        "lsrc_outputs": final_lsrc_state,
        "summary": summary,
    }
