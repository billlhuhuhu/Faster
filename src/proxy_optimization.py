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


def frequency_alignment_loss(proxy_points, target_real, target_imag, frequencies, batch_size=4096):
    proxy_real, proxy_imag = compute_empirical_characteristic_components(
        proxy_points,
        frequencies,
        batch_size=batch_size,
    )
    loss_real = torch.mean((proxy_real - target_real) ** 2)
    loss_imag = torch.mean((proxy_imag - target_imag) ** 2)
    return loss_real + loss_imag


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
):
    projected_representation, projection_matrix = random_project_representation(
        representation,
        projection_dim=projection_dim,
        random_state=random_state,
    )
    projected_representation = l2_normalize(projected_representation.astype(np.float32))

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

    frequencies = sample_frequency_points(
        dim=projected_representation.shape[1],
        num_frequencies=num_frequencies,
        frequency_scale=frequency_scale,
        device=torch_device,
        random_state=random_state,
    )

    with torch.no_grad():
        target_real, target_imag = compute_empirical_characteristic_components(
            full_repr,
            frequencies,
            batch_size=target_batch_size,
        )

    optimizer = torch.optim.Adam([proxy_points], lr=float(lr))
    history = []

    for step in range(int(num_steps)):
        optimizer.zero_grad(set_to_none=True)
        freq_loss = frequency_alignment_loss(
            proxy_points,
            target_real,
            target_imag,
            frequencies,
            batch_size=proxy_batch_size,
        )
        reg_loss = torch.mean((proxy_points - proxy_init) ** 2)
        loss = freq_loss + float(reg_weight) * reg_loss
        loss.backward()
        optimizer.step()

        if step in {0, int(num_steps) - 1} or (step + 1) % max(1, int(num_steps) // 10) == 0:
            history.append(
                {
                    "step": int(step + 1),
                    "total_loss": float(loss.detach().cpu().item()),
                    "frequency_loss": float(freq_loss.detach().cpu().item()),
                    "reg_loss": float(reg_loss.detach().cpu().item()),
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
        "frequencies": frequencies.detach().cpu().numpy().astype(np.float32),
        "summary": summary,
    }
