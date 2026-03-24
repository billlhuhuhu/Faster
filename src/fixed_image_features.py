import math
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import IncrementalPCA
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


def extract_hog_color_features(
    image_path,
    image_size=128,
    hog_orientations=9,
    hog_pixels_per_cell=8,
    hog_cells_per_block=2,
    color_space="rgb",
    color_hist_bins=16,
):
    try:
        from skimage.color import rgb2gray
        from skimage.feature import hog
    except ImportError as exc:
        raise ImportError(
            "extract_hog_color_features requires scikit-image. Install scikit-image in the selection-stage environment."
        ) from exc

    image_array = _open_image_rgb(image_path, image_size=image_size)
    gray = rgb2gray(image_array)
    hog_feat = hog(
        gray,
        orientations=int(hog_orientations),
        pixels_per_cell=(int(hog_pixels_per_cell), int(hog_pixels_per_cell)),
        cells_per_block=(int(hog_cells_per_block), int(hog_cells_per_block)),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
    color_feat = _compute_color_histogram(
        image_array,
        color_space=color_space,
        color_hist_bins=color_hist_bins,
    )
    return _l2_normalize(np.concatenate([hog_feat, color_feat], axis=0))


def _extract_raw_pixel_vector(image_path, raw_resize_size=32):
    image_array = _open_image_rgb(image_path, image_size=raw_resize_size)
    vector = image_array.reshape(-1).astype(np.float32)
    return vector


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

    raise ValueError(f"Unsupported fixed image representation method: {method}")
