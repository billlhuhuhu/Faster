import math
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def _open_image_rgb(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((int(image_size), int(image_size)), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def _rgb_to_hsv(image_rgb):
    r = image_rgb[..., 0]
    g = image_rgb[..., 1]
    b = image_rgb[..., 2]

    maxc = np.max(image_rgb, axis=-1)
    minc = np.min(image_rgb, axis=-1)
    v = maxc
    deltac = maxc - minc

    s = np.where(maxc == 0, 0, deltac / np.maximum(maxc, 1e-12))
    h = np.zeros_like(maxc)

    mask = deltac > 1e-12
    rc = (maxc - r) / np.maximum(deltac, 1e-12)
    gc = (maxc - g) / np.maximum(deltac, 1e-12)
    bc = (maxc - b) / np.maximum(deltac, 1e-12)

    r_mask = mask & (r == maxc)
    g_mask = mask & (g == maxc)
    b_mask = mask & (b == maxc)

    h[r_mask] = (bc - gc)[r_mask]
    h[g_mask] = 2.0 + (rc - bc)[g_mask]
    h[b_mask] = 4.0 + (gc - rc)[b_mask]
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1)


def _l2_normalize(feature, eps=1e-12):
    norm = float(np.linalg.norm(feature))
    if norm < eps:
        return feature.astype(np.float32)
    return (feature / norm).astype(np.float32)


def _compute_color_histogram(image_array, color_space="rgb", color_hist_bins=16):
    if color_space == "rgb":
        working = image_array
    elif color_space == "hsv":
        working = _rgb_to_hsv(image_array)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    hist_parts = []
    for channel_idx in range(working.shape[-1]):
        hist, _ = np.histogram(
            working[..., channel_idx].reshape(-1),
            bins=int(color_hist_bins),
            range=(0.0, 1.0),
            density=False,
        )
        hist = hist.astype(np.float32)
        hist = hist / max(float(hist.sum()), 1e-12)
        hist_parts.append(hist)
    return np.concatenate(hist_parts, axis=0).astype(np.float32)


def _rgb_to_gray(image_rgb):
    return (
        0.2989 * image_rgb[..., 0]
        + 0.5870 * image_rgb[..., 1]
        + 0.1140 * image_rgb[..., 2]
    ).astype(np.float32)


def _compute_simple_hog(
    gray_image,
    orientations=9,
    pixels_per_cell=8,
    cells_per_block=2,
    eps=1e-6,
):
    """A lightweight NumPy HOG fallback used when scikit-image is unavailable.

    This is intentionally simpler than skimage.feature.hog, but it preserves the
    same high-level signal: local orientation histograms with block normalization.
    """
    gray_image = np.asarray(gray_image, dtype=np.float32)
    gx = np.zeros_like(gray_image, dtype=np.float32)
    gy = np.zeros_like(gray_image, dtype=np.float32)
    gx[:, 1:-1] = gray_image[:, 2:] - gray_image[:, :-2]
    gx[:, 0] = gray_image[:, 1] - gray_image[:, 0]
    gx[:, -1] = gray_image[:, -1] - gray_image[:, -2]
    gy[1:-1, :] = gray_image[2:, :] - gray_image[:-2, :]
    gy[0, :] = gray_image[1, :] - gray_image[0, :]
    gy[-1, :] = gray_image[-1, :] - gray_image[-2, :]

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = (np.degrees(np.arctan2(gy, gx)) % 180.0).astype(np.float32)

    pixels_per_cell = int(pixels_per_cell)
    cells_per_block = int(cells_per_block)
    orientations = int(orientations)
    if pixels_per_cell <= 0 or cells_per_block <= 0 or orientations <= 0:
        raise ValueError("HOG parameters must be positive integers.")

    n_cells_y = gray_image.shape[0] // pixels_per_cell
    n_cells_x = gray_image.shape[1] // pixels_per_cell
    if n_cells_y < cells_per_block or n_cells_x < cells_per_block:
        raise ValueError("Image is too small for the requested HOG cell/block sizes.")

    hist = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)
    bin_width = 180.0 / float(orientations)

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0 = cy * pixels_per_cell
            y1 = y0 + pixels_per_cell
            x0 = cx * pixels_per_cell
            x1 = x0 + pixels_per_cell
            cell_mag = magnitude[y0:y1, x0:x1].reshape(-1)
            cell_ori = orientation[y0:y1, x0:x1].reshape(-1)
            cell_bins = np.floor(cell_ori / bin_width).astype(np.int64)
            cell_bins = np.clip(cell_bins, 0, orientations - 1)
            for bin_idx in range(orientations):
                mask = cell_bins == bin_idx
                if np.any(mask):
                    hist[cy, cx, bin_idx] = float(cell_mag[mask].sum())

    blocks = []
    for cy in range(n_cells_y - cells_per_block + 1):
        for cx in range(n_cells_x - cells_per_block + 1):
            block = hist[cy:cy + cells_per_block, cx:cx + cells_per_block, :].reshape(-1)
            block = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
            block = np.clip(block, 0.0, 0.2)
            block = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
            blocks.append(block.astype(np.float32))

    if not blocks:
        return hist.reshape(-1).astype(np.float32)
    return np.concatenate(blocks, axis=0).astype(np.float32)


def extract_hog_color_features(
    image_path,
    image_size=128,
    hog_orientations=9,
    hog_pixels_per_cell=8,
    hog_cells_per_block=2,
    color_space="rgb",
    color_hist_bins=16,
):
    image_array = _open_image_rgb(image_path, image_size=image_size)
    try:
        from skimage.color import rgb2gray
        from skimage.feature import hog

        gray = rgb2gray(image_array)
        hog_feat = hog(
            gray,
            orientations=int(hog_orientations),
            pixels_per_cell=(int(hog_pixels_per_cell), int(hog_pixels_per_cell)),
            cells_per_block=(int(hog_cells_per_block), int(hog_cells_per_block)),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)
    except ImportError:
        gray = _rgb_to_gray(image_array)
        hog_feat = _compute_simple_hog(
            gray,
            orientations=int(hog_orientations),
            pixels_per_cell=int(hog_pixels_per_cell),
            cells_per_block=int(hog_cells_per_block),
        )
    color_feat = _compute_color_histogram(
        image_array,
        color_space=color_space,
        color_hist_bins=color_hist_bins,
    )
    features = np.concatenate([hog_feat, color_feat], axis=0)
    # Hellinger 变换：将欧氏距离隐式切换为 Hellinger 距离，修复 HOG/颜色直方图在欧氏空间的局部距离退化问题
    features = np.sqrt(np.clip(features, 0, None))
    return _l2_normalize(features)


def _extract_hog_color_base_features(
    image_path,
    image_size=128,
    hog_orientations=9,
    hog_pixels_per_cell=8,
    hog_cells_per_block=2,
    color_space="rgb",
    color_hist_bins=16,
):
    image_array = _open_image_rgb(image_path, image_size=image_size)
    try:
        from skimage.color import rgb2gray
        from skimage.feature import hog

        gray = rgb2gray(image_array)
        hog_feat = hog(
            gray,
            orientations=int(hog_orientations),
            pixels_per_cell=(int(hog_pixels_per_cell), int(hog_pixels_per_cell)),
            cells_per_block=(int(hog_cells_per_block), int(hog_cells_per_block)),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)
    except ImportError:
        gray = _rgb_to_gray(image_array)
        hog_feat = _compute_simple_hog(
            gray,
            orientations=int(hog_orientations),
            pixels_per_cell=int(hog_pixels_per_cell),
            cells_per_block=int(hog_cells_per_block),
        )
    color_feat = _compute_color_histogram(
        image_array,
        color_space=color_space,
        color_hist_bins=color_hist_bins,
    )
    return np.concatenate([hog_feat, color_feat], axis=0).astype(np.float32, copy=False)


def _extract_raw_pixel_vector(image_path, raw_resize_size=32):
    image_array = _open_image_rgb(image_path, image_size=raw_resize_size)
    vector = image_array.reshape(-1).astype(np.float32)
    return vector


def _open_image_gray_uint8(image_path, image_size):
    image = Image.open(image_path).convert("L")
    image = image.resize((int(image_size), int(image_size)), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _extract_raw_pixel_vector_v2(image_path, raw_pixel_resize=64, raw_pixel_color_mode="rgb", raw_pixel_flatten=True):
    raw_pixel_color_mode = str(raw_pixel_color_mode).lower()
    if raw_pixel_color_mode == "rgb":
        image = _open_image_rgb(image_path, image_size=raw_pixel_resize)
    elif raw_pixel_color_mode in {"gray", "grayscale", "l"}:
        gray = _open_image_gray_uint8(image_path, image_size=raw_pixel_resize).astype(np.float32) / 255.0
        image = gray[..., None]
    else:
        raise ValueError(f"Unsupported raw pixel color mode: {raw_pixel_color_mode}")

    if raw_pixel_flatten:
        return image.reshape(-1).astype(np.float32)
    return image.astype(np.float32)


def _sample_dense_sift_keypoints(image_shape, step=8, patch=16):
    height, width = int(image_shape[0]), int(image_shape[1])
    step = max(1, int(step))
    patch = max(4, int(patch))
    radius = max(1, patch // 2)
    xs = list(range(radius, max(width - radius, radius + 1), step))
    ys = list(range(radius, max(height - radius, radius + 1), step))
    if not xs:
        xs = [max(width // 2, 1)]
    if not ys:
        ys = [max(height // 2, 1)]
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("opencv-python is required for dense_sift_bovw image features.") from exc
    keypoints = [cv2.KeyPoint(float(x), float(y), float(patch)) for y in ys for x in xs]
    return keypoints


def _compute_gradients(gray_image):
    gray_image = np.asarray(gray_image, dtype=np.float32)
    gx = np.zeros_like(gray_image, dtype=np.float32)
    gy = np.zeros_like(gray_image, dtype=np.float32)
    gx[:, 1:-1] = gray_image[:, 2:] - gray_image[:, :-2]
    gx[:, 0] = gray_image[:, 1] - gray_image[:, 0]
    gx[:, -1] = gray_image[:, -1] - gray_image[:, -2]
    gy[1:-1, :] = gray_image[2:, :] - gray_image[:-2, :]
    gy[0, :] = gray_image[1, :] - gray_image[0, :]
    gy[-1, :] = gray_image[-1, :] - gray_image[-2, :]
    magnitude = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    orientation = (np.degrees(np.arctan2(gy, gx)) % 360.0).astype(np.float32)
    return magnitude, orientation


def _dense_grid_centers(image_shape, step=8, patch=16):
    height, width = int(image_shape[0]), int(image_shape[1])
    step = max(1, int(step))
    patch = max(4, int(patch))
    radius = max(1, patch // 2)
    xs = list(range(radius, max(width - radius, radius + 1), step))
    ys = list(range(radius, max(height - radius, radius + 1), step))
    if not xs:
        xs = [max(width // 2, 1)]
    if not ys:
        ys = [max(height // 2, 1)]
    return [(int(y), int(x)) for y in ys for x in xs]


def _compute_dense_sift_like_descriptors(gray_image, step=8, patch=16, spatial_bins=4, orientation_bins=8):
    patch = max(4, int(patch))
    spatial_bins = max(1, int(spatial_bins))
    orientation_bins = max(1, int(orientation_bins))
    patch = int(math.ceil(patch / spatial_bins) * spatial_bins)
    radius = patch // 2

    gray_image = np.asarray(gray_image, dtype=np.float32)
    magnitude, orientation = _compute_gradients(gray_image)
    centers = _dense_grid_centers(gray_image.shape, step=step, patch=patch)
    bin_width = 360.0 / float(orientation_bins)
    descriptors = []

    padded_mag = np.pad(magnitude, radius, mode="reflect")
    padded_ori = np.pad(orientation, radius, mode="reflect")

    for center_y, center_x in centers:
        cy = center_y + radius
        cx = center_x + radius
        mag_patch = padded_mag[cy - radius:cy + radius, cx - radius:cx + radius]
        ori_patch = padded_ori[cy - radius:cy + radius, cx - radius:cx + radius]
        if mag_patch.shape != (patch, patch) or ori_patch.shape != (patch, patch):
            continue

        descriptor_parts = []
        cell_size = patch // spatial_bins
        for sy in range(spatial_bins):
            for sx in range(spatial_bins):
                y0 = sy * cell_size
                y1 = y0 + cell_size
                x0 = sx * cell_size
                x1 = x0 + cell_size
                cell_mag = mag_patch[y0:y1, x0:x1].reshape(-1)
                cell_ori = ori_patch[y0:y1, x0:x1].reshape(-1)
                cell_hist = np.zeros(orientation_bins, dtype=np.float32)
                bin_indices = np.floor(cell_ori / bin_width).astype(np.int64)
                bin_indices = np.clip(bin_indices, 0, orientation_bins - 1)
                for bin_idx in range(orientation_bins):
                    mask = bin_indices == bin_idx
                    if np.any(mask):
                        cell_hist[bin_idx] = float(cell_mag[mask].sum())
                descriptor_parts.append(cell_hist)

        descriptor = np.concatenate(descriptor_parts, axis=0).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(descriptor))
        if norm > 1e-12:
            descriptor = descriptor / norm
            descriptor = np.clip(descriptor, 0.0, 0.2)
            descriptor = descriptor / max(float(np.linalg.norm(descriptor)), 1e-12)
        descriptors.append(descriptor.astype(np.float32, copy=False))

    if not descriptors:
        return np.zeros((0, spatial_bins * spatial_bins * orientation_bins), dtype=np.float32)
    return np.stack(descriptors, axis=0).astype(np.float32, copy=False)


def extract_dense_sift_descriptors(image_path, image_size=128, step=8, patch=16):
    image_gray = _open_image_gray_uint8(image_path, image_size=image_size)
    try:
        import cv2
        if hasattr(cv2, "SIFT_create"):
            sift = cv2.SIFT_create()
            keypoints = _sample_dense_sift_keypoints(image_gray.shape, step=step, patch=patch)
            _, descriptors = sift.compute(image_gray, keypoints)
            if descriptors is None or descriptors.size == 0:
                return np.zeros((0, 128), dtype=np.float32)
            return descriptors.astype(np.float32, copy=False)
    except Exception:
        pass

    # Fallback for headless servers without libGL/OpenCV runtime support:
    # keep the same dense local-gradient + BoVW structure using a NumPy SIFT-like descriptor.
    return _compute_dense_sift_like_descriptors(
        image_gray.astype(np.float32) / 255.0,
        step=step,
        patch=patch,
        spatial_bins=4,
        orientation_bins=8,
    )


def _sample_descriptors(descriptors, max_count, random_state):
    if descriptors.shape[0] <= max_count:
        return descriptors.astype(np.float32, copy=False)
    rng = np.random.default_rng(int(random_state))
    selected = rng.choice(descriptors.shape[0], size=int(max_count), replace=False)
    return descriptors[selected].astype(np.float32, copy=False)


def _l1_normalize_rows(matrix, eps=1e-12):
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.clip(matrix, 0.0, None)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, float(eps))
    return (matrix / row_sums).astype(np.float32, copy=False)


def _transform_hog_color_batch(
    batch_paths,
    image_size,
    hog_orientations,
    hog_pixels_per_cell,
    hog_cells_per_block,
    color_space,
    color_hist_bins,
    transform_mode,
    chi2_sample_steps=2,
):
    raw_batch = np.stack(
        [
            _extract_hog_color_base_features(
                image_path,
                image_size=image_size,
                hog_orientations=hog_orientations,
                hog_pixels_per_cell=hog_pixels_per_cell,
                hog_cells_per_block=hog_cells_per_block,
                color_space=color_space,
                color_hist_bins=color_hist_bins,
            )
            for image_path in batch_paths
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    hist_batch = _l1_normalize_rows(raw_batch)
    if transform_mode == "hellinger":
        return np.sqrt(np.clip(hist_batch, 0.0, None)).astype(np.float32, copy=False)
    if transform_mode == "chi2":
        sampler = AdditiveChi2Sampler(sample_steps=max(1, int(chi2_sample_steps)))
        sampler.fit(hist_batch[:1])
        return sampler.transform(hist_batch).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported hog_color transform mode: {transform_mode}")


def _iter_min_sized_batches(num_items, batch_size, min_batch_size):
    """Yield batch ranges while merging a too-small tail into the previous batch."""
    num_items = int(num_items)
    batch_size = max(1, int(batch_size))
    min_batch_size = max(1, int(min_batch_size))
    start = 0
    while start < num_items:
        end = min(start + batch_size, num_items)
        if end < num_items and (num_items - end) < min_batch_size:
            end = num_items
        yield start, end
        start = end


def _extract_histogram_whitened_features(
    image_paths,
    image_size,
    hog_orientations,
    hog_pixels_per_cell,
    hog_cells_per_block,
    color_space,
    color_hist_bins,
    transform_mode,
    pca_dim=256,
    whitening_eps=1e-5,
    chi2_sample_steps=2,
    batch_size=512,
):
    image_paths = [str(item) for item in image_paths]
    batch_size = max(1, int(batch_size))
    first_batch = _transform_hog_color_batch(
        image_paths[:1],
        image_size=image_size,
        hog_orientations=hog_orientations,
        hog_pixels_per_cell=hog_pixels_per_cell,
        hog_cells_per_block=hog_cells_per_block,
        color_space=color_space,
        color_hist_bins=color_hist_bins,
        transform_mode=transform_mode,
        chi2_sample_steps=chi2_sample_steps,
    )
    feature_dim = int(first_batch.shape[1])
    target_dim = min(int(pca_dim), feature_dim, len(image_paths))
    if target_dim <= 0:
        raise ValueError(f"pca_dim must be positive for {transform_mode} histogram whitening features.")

    ipca = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    fit_ranges = list(_iter_min_sized_batches(len(image_paths), batch_size, target_dim))
    for start, end in tqdm(fit_ranges, desc=f"Fitting {transform_mode} PCA whitening"):
        batch_paths = image_paths[start:end]
        batch = _transform_hog_color_batch(
            batch_paths,
            image_size=image_size,
            hog_orientations=hog_orientations,
            hog_pixels_per_cell=hog_pixels_per_cell,
            hog_cells_per_block=hog_cells_per_block,
            color_space=color_space,
            color_hist_bins=color_hist_bins,
            transform_mode=transform_mode,
            chi2_sample_steps=chi2_sample_steps,
        )
        ipca.partial_fit(batch)

    std = np.sqrt(
        np.asarray(
            getattr(ipca, "explained_variance_", np.ones((target_dim,), dtype=np.float32)),
            dtype=np.float32,
        )
    ) + float(whitening_eps)

    transformed_chunks = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc=f"Transforming {transform_mode} PCA whitening"):
        batch_paths = image_paths[start : start + batch_size]
        batch = _transform_hog_color_batch(
            batch_paths,
            image_size=image_size,
            hog_orientations=hog_orientations,
            hog_pixels_per_cell=hog_pixels_per_cell,
            hog_cells_per_block=hog_cells_per_block,
            color_space=color_space,
            color_hist_bins=color_hist_bins,
            transform_mode=transform_mode,
            chi2_sample_steps=chi2_sample_steps,
        )
        transformed = ipca.transform(batch).astype(np.float32, copy=False)
        transformed = (transformed / std).astype(np.float32, copy=False)
        transformed_chunks.append(transformed)

    features = np.concatenate(transformed_chunks, axis=0).astype(np.float32, copy=False)
    info = {
        "image_size": int(image_size),
        "hog_orientations": int(hog_orientations),
        "hog_pixels_per_cell": int(hog_pixels_per_cell),
        "hog_cells_per_block": int(hog_cells_per_block),
        "color_space": color_space,
        "color_hist_bins": int(color_hist_bins),
        "transform_mode": str(transform_mode),
        "pca_dim": int(features.shape[1]),
        "whitening_eps": float(whitening_eps),
        "pca_explained_variance": float(
            np.sum(getattr(ipca, "explained_variance_ratio_", np.array([0.0], dtype=np.float32)))
        ),
    }
    if transform_mode == "chi2":
        info["chi2_sample_steps"] = int(chi2_sample_steps)
    return features, info


def extract_hog_color_hellinger_pca_whitening_features(
    image_paths,
    image_size=128,
    hog_orientations=9,
    hog_pixels_per_cell=8,
    hog_cells_per_block=2,
    color_space="rgb",
    color_hist_bins=16,
    pca_dim=256,
    whitening_eps=1e-3,
    batch_size=512,
):
    return _extract_histogram_whitened_features(
        image_paths,
        image_size=image_size,
        hog_orientations=hog_orientations,
        hog_pixels_per_cell=hog_pixels_per_cell,
        hog_cells_per_block=hog_cells_per_block,
        color_space=color_space,
        color_hist_bins=color_hist_bins,
        transform_mode="hellinger",
        pca_dim=pca_dim,
        whitening_eps=whitening_eps,
        batch_size=batch_size,
    )


def extract_hog_color_chi2_pca_whitening_features(
    image_paths,
    image_size=128,
    hog_orientations=9,
    hog_pixels_per_cell=8,
    hog_cells_per_block=2,
    color_space="rgb",
    color_hist_bins=16,
    pca_dim=256,
    whitening_eps=1e-3,
    chi2_sample_steps=2,
    batch_size=512,
):
    return _extract_histogram_whitened_features(
        image_paths,
        image_size=image_size,
        hog_orientations=hog_orientations,
        hog_pixels_per_cell=hog_pixels_per_cell,
        hog_cells_per_block=hog_cells_per_block,
        color_space=color_space,
        color_hist_bins=color_hist_bins,
        transform_mode="chi2",
        pca_dim=pca_dim,
        whitening_eps=whitening_eps,
        chi2_sample_steps=chi2_sample_steps,
        batch_size=batch_size,
    )


def extract_dense_sift_bovw_features(
    image_paths,
    image_size=128,
    bovw_codebook_size=512,
    dense_sift_step=8,
    dense_sift_patch=16,
    random_state=0,
    bovw_max_fit_descriptors=200000,
    bovw_descriptors_per_image=200,
):
    image_paths = [str(item) for item in image_paths]
    descriptor_samples = []
    codebook_size = max(8, int(bovw_codebook_size))
    max_fit_descriptors = max(codebook_size, int(bovw_max_fit_descriptors))

    for image_path in tqdm(image_paths, desc="Sampling dense SIFT descriptors for BoVW"):
        descriptors = extract_dense_sift_descriptors(
            image_path,
            image_size=image_size,
            step=dense_sift_step,
            patch=dense_sift_patch,
        )
        if descriptors.shape[0] == 0:
            continue
        sampled = _sample_descriptors(
            descriptors,
            max_count=max(1, int(bovw_descriptors_per_image)),
            random_state=random_state + len(descriptor_samples),
        )
        descriptor_samples.append(sampled)

    if not descriptor_samples:
        raise RuntimeError("No dense SIFT descriptors were extracted; cannot build BoVW codebook.")

    fit_matrix = np.concatenate(descriptor_samples, axis=0).astype(np.float32, copy=False)
    if fit_matrix.shape[0] > max_fit_descriptors:
        fit_matrix = _sample_descriptors(fit_matrix, max_fit_descriptors, random_state=random_state)

    effective_codebook = min(codebook_size, fit_matrix.shape[0])
    if effective_codebook <= 1:
        raise RuntimeError("Not enough sampled dense SIFT descriptors to build a BoVW codebook.")

    kmeans = MiniBatchKMeans(
        n_clusters=effective_codebook,
        random_state=int(random_state),
        batch_size=min(4096, max(256, effective_codebook * 8)),
        n_init=3,
    )
    kmeans.fit(fit_matrix)

    histograms = []
    for image_path in tqdm(image_paths, desc="Encoding dense SIFT BoVW histograms"):
        descriptors = extract_dense_sift_descriptors(
            image_path,
            image_size=image_size,
            step=dense_sift_step,
            patch=dense_sift_patch,
        )
        hist = np.zeros(effective_codebook, dtype=np.float32)
        if descriptors.shape[0] > 0:
            words = kmeans.predict(descriptors)
            counts = np.bincount(words, minlength=effective_codebook).astype(np.float32)
            hist = counts / max(float(counts.sum()), 1e-12)
        histograms.append(_l2_normalize(hist))

    features = np.stack(histograms, axis=0).astype(np.float32)
    info = {
        "image_size": int(image_size),
        "bovw_codebook_size": int(effective_codebook),
        "dense_sift_step": int(dense_sift_step),
        "dense_sift_patch": int(dense_sift_patch),
        "bovw_max_fit_descriptors": int(max_fit_descriptors),
        "bovw_descriptors_per_image": int(bovw_descriptors_per_image),
        "random_state": int(random_state),
    }
    return features, info


def extract_raw_pca_features(
    image_paths,
    raw_resize_size=32,
    pca_dim=256,
    random_state=0,
    batch_size=512,
):
    image_paths = [str(item) for item in image_paths]
    raw_dim = int(raw_resize_size) * int(raw_resize_size) * 3
    pca_dim = min(int(pca_dim), raw_dim, len(image_paths))
    if pca_dim <= 0:
        raise ValueError("pca_dim must be positive for raw_pca features.")

    ipca = IncrementalPCA(n_components=pca_dim, batch_size=int(batch_size))
    for start in tqdm(range(0, len(image_paths), int(batch_size)), desc="Fitting raw-pixel PCA"):
        batch_paths = image_paths[start : start + int(batch_size)]
        batch = np.stack(
            [_extract_raw_pixel_vector(path, raw_resize_size=raw_resize_size) for path in batch_paths],
            axis=0,
        )
        ipca.partial_fit(batch)

    transformed_chunks = []
    for start in tqdm(range(0, len(image_paths), int(batch_size)), desc="Transforming raw-pixel PCA"):
        batch_paths = image_paths[start : start + int(batch_size)]
        batch = np.stack(
            [_extract_raw_pixel_vector(path, raw_resize_size=raw_resize_size) for path in batch_paths],
            axis=0,
        )
        transformed = ipca.transform(batch).astype(np.float32)
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        transformed_chunks.append((transformed / norms).astype(np.float32))

    features = np.concatenate(transformed_chunks, axis=0).astype(np.float32)
    info = {
        "raw_resize_size": int(raw_resize_size),
        "pca_dim": int(features.shape[1]),
        "pca_explained_variance": float(np.sum(getattr(ipca, "explained_variance_ratio_", np.array([0.0], dtype=np.float32)))),
        "random_state": int(random_state),
    }
    return features, info


def extract_raw_pixels_pca_features(
    image_paths,
    raw_pixel_resize=64,
    raw_pixel_color_mode="rgb",
    raw_pixel_flatten=True,
    raw_pixel_pca_dim=256,
    random_state=0,
    batch_size=512,
):
    image_paths = [str(item) for item in image_paths]
    sample_vector = _extract_raw_pixel_vector_v2(
        image_paths[0],
        raw_pixel_resize=raw_pixel_resize,
        raw_pixel_color_mode=raw_pixel_color_mode,
        raw_pixel_flatten=raw_pixel_flatten,
    )
    raw_dim = int(sample_vector.size)
    target_dim = min(int(raw_pixel_pca_dim), raw_dim, len(image_paths))
    if target_dim <= 0:
        raise ValueError("raw_pixel_pca_dim must be positive for raw_pixels_pca features.")

    ipca = IncrementalPCA(n_components=target_dim, batch_size=int(batch_size))
    for start in tqdm(range(0, len(image_paths), int(batch_size)), desc="Fitting raw-pixels PCA"):
        batch_paths = image_paths[start : start + int(batch_size)]
        batch = np.stack(
            [
                _extract_raw_pixel_vector_v2(
                    path,
                    raw_pixel_resize=raw_pixel_resize,
                    raw_pixel_color_mode=raw_pixel_color_mode,
                    raw_pixel_flatten=raw_pixel_flatten,
                )
                for path in batch_paths
            ],
            axis=0,
        )
        ipca.partial_fit(batch)

    transformed_chunks = []
    for start in tqdm(range(0, len(image_paths), int(batch_size)), desc="Transforming raw-pixels PCA"):
        batch_paths = image_paths[start : start + int(batch_size)]
        batch = np.stack(
            [
                _extract_raw_pixel_vector_v2(
                    path,
                    raw_pixel_resize=raw_pixel_resize,
                    raw_pixel_color_mode=raw_pixel_color_mode,
                    raw_pixel_flatten=raw_pixel_flatten,
                )
                for path in batch_paths
            ],
            axis=0,
        )
        transformed = ipca.transform(batch).astype(np.float32, copy=False)
        norms = np.linalg.norm(transformed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        transformed_chunks.append((transformed / norms).astype(np.float32, copy=False))

    features = np.concatenate(transformed_chunks, axis=0).astype(np.float32, copy=False)
    info = {
        "raw_pixel_resize": int(raw_pixel_resize),
        "raw_pixel_color_mode": str(raw_pixel_color_mode),
        "raw_pixel_flatten": bool(raw_pixel_flatten),
        "raw_pixel_pca_dim": int(features.shape[1]),
        "pca_explained_variance": float(np.sum(getattr(ipca, "explained_variance_ratio_", np.array([0.0], dtype=np.float32)))),
        "random_state": int(random_state),
    }
    return features, info


def extract_fixed_image_features(
    image_paths,
    method="hog_color",
    image_size=128,
    hog_orientations=9,
    hog_pixels_per_cell=8,
    hog_cells_per_block=2,
    color_space="rgb",
    color_hist_bins=16,
    raw_resize_size=32,
    raw_pca_dim=256,
    raw_pixel_resize=64,
    raw_pixel_color_mode="rgb",
    raw_pixel_flatten=True,
    raw_pixel_pca_dim=256,
    bovw_codebook_size=512,
    dense_sift_step=8,
    dense_sift_patch=16,
    bovw_max_fit_descriptors=200000,
    bovw_descriptors_per_image=200,
    histogram_whitening_pca_dim=256,
    pca_whitening_eps=1e-3,
    chi2_sample_steps=2,
    batch_size=None,
    random_state=0,
):
    image_paths = [str(Path(path)) for path in image_paths]
    method = str(method).lower()

    if method == "hog_color":
        features = []
        for image_path in tqdm(image_paths, desc="Extracting fixed image features (hog_color)"):
            feature = extract_hog_color_features(
                image_path,
                image_size=image_size,
                hog_orientations=hog_orientations,
                hog_pixels_per_cell=hog_pixels_per_cell,
                hog_cells_per_block=hog_cells_per_block,
                color_space=color_space,
                color_hist_bins=color_hist_bins,
            )
            features.append(feature)
        return np.stack(features, axis=0).astype(np.float32), {
            "selection_image_repr_method": method,
            "image_size": int(image_size),
            "hog_orientations": int(hog_orientations),
            "hog_pixels_per_cell": int(hog_pixels_per_cell),
            "hog_cells_per_block": int(hog_cells_per_block),
            "color_space": color_space,
            "color_hist_bins": int(color_hist_bins),
        }

    if method == "hog_color_raw":
        features = []
        for image_path in tqdm(image_paths, desc="Extracting fixed image features (hog_color_raw)"):
            feature = _extract_hog_color_base_features(
                image_path,
                image_size=image_size,
                hog_orientations=hog_orientations,
                hog_pixels_per_cell=hog_pixels_per_cell,
                hog_cells_per_block=hog_cells_per_block,
                color_space=color_space,
                color_hist_bins=color_hist_bins,
            )
            features.append(_l2_normalize(feature))
        return np.stack(features, axis=0).astype(np.float32), {
            "selection_image_repr_method": method,
            "image_size": int(image_size),
            "hog_orientations": int(hog_orientations),
            "hog_pixels_per_cell": int(hog_pixels_per_cell),
            "hog_cells_per_block": int(hog_cells_per_block),
            "color_space": color_space,
            "color_hist_bins": int(color_hist_bins),
            "hellinger_transform": False,
        }

    if method == "raw_pca":
        features, info = extract_raw_pca_features(
            image_paths,
            raw_resize_size=raw_resize_size,
            pca_dim=raw_pca_dim,
            random_state=random_state,
            batch_size=batch_size or 512,
        )
        info["selection_image_repr_method"] = method
        return features, info

    if method == "raw_pixels_pca":
        features, info = extract_raw_pixels_pca_features(
            image_paths,
            raw_pixel_resize=raw_pixel_resize,
            raw_pixel_color_mode=raw_pixel_color_mode,
            raw_pixel_flatten=raw_pixel_flatten,
            raw_pixel_pca_dim=raw_pixel_pca_dim,
            random_state=random_state,
            batch_size=batch_size or 512,
        )
        info["selection_image_repr_method"] = method
        return features, info

    if method == "dense_sift_bovw":
        features, info = extract_dense_sift_bovw_features(
            image_paths,
            image_size=image_size,
            bovw_codebook_size=bovw_codebook_size,
            dense_sift_step=dense_sift_step,
            dense_sift_patch=dense_sift_patch,
            random_state=random_state,
            bovw_max_fit_descriptors=bovw_max_fit_descriptors,
            bovw_descriptors_per_image=bovw_descriptors_per_image,
        )
        info["selection_image_repr_method"] = method
        return features, info

    if method == "hog_color_hellinger_pca_whitening":
        features, info = extract_hog_color_hellinger_pca_whitening_features(
            image_paths,
            image_size=image_size,
            hog_orientations=hog_orientations,
            hog_pixels_per_cell=hog_pixels_per_cell,
            hog_cells_per_block=hog_cells_per_block,
            color_space=color_space,
            color_hist_bins=color_hist_bins,
            pca_dim=histogram_whitening_pca_dim,
            whitening_eps=pca_whitening_eps,
            batch_size=batch_size or 512,
        )
        info["selection_image_repr_method"] = method
        return features, info

    if method == "hog_color_chi2_pca_whitening":
        features, info = extract_hog_color_chi2_pca_whitening_features(
            image_paths,
            image_size=image_size,
            hog_orientations=hog_orientations,
            hog_pixels_per_cell=hog_pixels_per_cell,
            hog_cells_per_block=hog_cells_per_block,
            color_space=color_space,
            color_hist_bins=color_hist_bins,
            pca_dim=histogram_whitening_pca_dim,
            whitening_eps=pca_whitening_eps,
            chi2_sample_steps=chi2_sample_steps,
            batch_size=batch_size or 512,
        )
        info["selection_image_repr_method"] = method
        return features, info

    raise ValueError(f"Unsupported fixed image representation method: {method}")
