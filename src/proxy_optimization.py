import math

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans


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
):
    projected_representation, projection_matrix = random_project_representation(
        representation,
        projection_dim=projection_dim,
        random_state=random_state,
    )
    projected_representation = l2_normalize(projected_representation.astype(np.float32))
    projected_match_reference = project_reference_points(match_reference, projection_matrix)
    projected_graph_reference = project_reference_points(graph_reference, projection_matrix)

    init_points, init_info = initialize_proxy_points(
        projected_representation,
        subset_size=subset_size,
        init_method=init_method,
        random_state=random_state,
        minibatch_size=minibatch_size,
    )

    torch_device = torch.device(device)
    full_repr = torch.tensor(projected_representation, dtype=torch.float32, device=torch_device)
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

        loss = (
            freq_loss
            + float(reg_weight) * reg_loss
            + float(lambda_div) * div_loss
            + float(lambda_match) * match_loss
            + float(lambda_graph) * graph_loss
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
