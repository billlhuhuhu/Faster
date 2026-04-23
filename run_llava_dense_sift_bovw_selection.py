import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from data.subset_dataset import save_selected_indices
from run_vlm_finetune import extract_llava_turn, load_json_or_jsonl, resolve_image_path
from src.fixed_image_features import extract_fixed_image_features
from src.fixed_text_features import extract_fixed_text_features
from src.proxy_optimization import l2_normalize
from src.subset_match import run_proxy_optimized_selection, sort_selected_indices


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_ratios(value):
    return [float(item) for item in str(value).replace(",", " ").split() if item.strip()]


def build_llava_texts(records):
    texts = []
    for record in records:
        prompt, answer = extract_llava_turn(record)
        texts.append(f"{prompt}\n{answer}".strip())
    return texts


def load_or_build_features(args, records):
    cache_dir = Path(args.cache_dir)
    image_path = cache_dir / "img_features_selection.pt"
    text_path = cache_dir / "txt_features_selection.pt"
    meta_path = cache_dir / "sample_meta.json"
    info_path = cache_dir / "feature_info.json"
    if image_path.exists() and text_path.exists() and meta_path.exists() and info_path.exists() and not args.force_recompute_features:
        print(f"[LLaVA selection] Loading cached features from {cache_dir}", flush=True)
        print(f"[LLaVA selection]   image cache: {image_path}", flush=True)
        img_features = torch.load(image_path, map_location="cpu").numpy()
        print(f"[LLaVA selection]   image features loaded: shape={img_features.shape}", flush=True)
        print(f"[LLaVA selection]   text cache: {text_path}", flush=True)
        txt_features = torch.load(text_path, map_location="cpu").numpy()
        print(f"[LLaVA selection]   text features loaded: shape={txt_features.shape}", flush=True)
        return (
            img_features,
            txt_features,
            json.loads(meta_path.read_text(encoding="utf-8")),
            json.loads(info_path.read_text(encoding="utf-8")),
        )
    missing_cache = [str(path) for path in [image_path, text_path, meta_path, info_path] if not path.exists()]
    if missing_cache:
        print(f"[LLaVA selection] Feature cache incomplete, rebuilding. Missing: {missing_cache}", flush=True)
    elif args.force_recompute_features:
        print("[LLaVA selection] force_recompute_features=True, rebuilding feature cache.", flush=True)

    image_paths = [str(resolve_image_path(record, args.image_root)) for record in records]
    missing = [path for path in image_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images. First missing image: {missing[0]}")
    texts = build_llava_texts(records)

    img_features, image_info = extract_fixed_image_features(
        image_paths,
        method="dense_sift_bovw",
        image_size=args.selection_image_size,
        bovw_codebook_size=args.bovw_codebook_size,
        dense_sift_step=args.dense_sift_step,
        dense_sift_patch=args.dense_sift_patch,
        random_state=args.seed,
        bovw_max_fit_descriptors=args.bovw_max_fit_descriptors,
        bovw_descriptors_per_image=args.bovw_descriptors_per_image,
    )
    txt_features = extract_fixed_text_features(
        texts,
        text_repr_method=args.text_repr_method,
        batch_size=args.selection_text_batch_size,
        device=args.device,
        tfidf_ngram_max=args.tfidf_ngram_max,
        tfidf_stop_words=args.tfidf_stop_words,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_min_df=args.tfidf_min_df,
        tfidf_svd_dim=args.tfidf_svd_dim,
        tfidf_random_state=args.seed,
    )
    sample_meta = [
        {
            "sample_idx": int(idx),
            "id": record.get("id", idx),
            "image": record.get("image", record.get("image_path", record.get("file_name", ""))),
        }
        for idx, record in enumerate(records)
    ]
    feature_info = {
        "selection_image_repr_method": "dense_sift_bovw",
        "selection_text_repr_method": args.text_repr_method,
        "image_info": image_info,
        "tfidf_ngram_max": int(args.tfidf_ngram_max),
        "tfidf_stop_words": args.tfidf_stop_words,
        "tfidf_max_features": int(args.tfidf_max_features),
        "tfidf_min_df": int(args.tfidf_min_df),
        "tfidf_svd_dim": int(args.tfidf_svd_dim),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(img_features, dtype=torch.float32), image_path)
    torch.save(torch.tensor(txt_features, dtype=torch.float32), text_path)
    write_json(meta_path, sample_meta)
    write_json(info_path, feature_info)
    return img_features, txt_features, sample_meta, feature_info


def build_knn_graph(representation, k=15, metric="cosine", desc="Building LLaVA kNN graph"):
    representation = np.asarray(representation, dtype=np.float32)
    n = representation.shape[0]
    k = max(1, min(int(k), n - 1))
    print(
        f"[LLaVA selection] {desc}: fitting/querying kNN "
        f"n={n} dim={representation.shape[1]} k={k} metric={metric}",
        flush=True,
    )
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(representation)
    distances, indices = nn.kneighbors(representation)
    print(f"[LLaVA selection] {desc}: kNN query done, building sparse graph", flush=True)
    rows = []
    cols = []
    vals = []
    for i in tqdm(range(n), desc=desc):
        neigh = indices[i, 1:]
        dist = distances[i, 1:]
        if metric == "cosine":
            weights = 1.0 - dist
        else:
            scale = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
            weights = np.exp(-dist / max(float(scale), 1e-8))
        weights = np.clip(weights.astype(np.float32), 0.0, None)
        rows.extend([i] * len(neigh))
        cols.extend(neigh.tolist())
        vals.extend(weights.tolist())
    graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    graph = graph.maximum(graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def make_selection_args(args, ratio):
    return SimpleNamespace(
        dataset="llava_instruct_150k",
        split="train",
        image_encoder="dense_sift_bovw",
        text_encoder=args.text_repr_method,
        metric=args.metric,
        k=int(args.knn_k),
        alpha=1.0,
        budget_ratio=float(ratio) / 100.0,
        budget_size=None,
        representation_mode="concat",
        reference_embedding_mode="concat",
        spectral_weight=1.0,
        selection_method="proxy_opt",
        cluster_method="minibatch_kmeans",
        degree_weight=0.1,
        geometry_weight=1.0,
        random_state=int(args.seed),
        minibatch_size=int(args.minibatch_size),
        device=args.device,
        proxy_projection_dim=int(args.proxy_projection_dim),
        proxy_init_method=args.proxy_init_method,
        proxy_loss_type=args.proxy_loss_type,
        proxy_objective_mode=None,
        use_pdcfd=False,
        proxy_num_frequencies=64,
        proxy_frequency_scale=1.0,
        proxy_lr=float(args.proxy_lr),
        proxy_num_steps=int(args.proxy_num_steps),
        proxy_reg_weight=float(args.proxy_reg_weight),
        proxy_target_batch_size=int(args.proxy_target_batch_size),
        proxy_batch_size=int(args.proxy_batch_size),
        mmd_kernel="rbf",
        mmd_bandwidth=None,
        mmd_use_median_heuristic=True,
        swd_num_projections=64,
        swd_p=2.0,
        swd_projection_seed=None,
        swd_use_fixed_projections=False,
        use_wavelet_multiscale=True,
        wavelet_scales=args.wavelet_scales,
        wavelet_loss_weight=0.1,
        wavelet_distance_type="swd",
        wavelet_swd_num_projections=None,
        wavelet_swd_p=None,
        wavelet_schedule="coarse_to_fine",
        lambda_main=1.0,
        wavelet_main_scales=args.wavelet_main_scales,
        wavelet_main_scale_weights=None,
        wavelet_main_swd_num_projections=int(args.wavelet_main_swd_num_projections),
        wavelet_cov_weight=float(args.wavelet_cov_weight),
        wavelet_edge_weight=float(args.wavelet_edge_weight),
        wavelet_curriculum_schedule="coarse_to_fine",
        use_pdas=False,
        pdas_num_stages=4,
        pdas_schedule_mode="low_to_high",
        num_freq_pool=256,
        tau_min=0.1,
        tau_max=1.0,
        use_dpp=bool(args.use_dpp),
        lambda_div=0.01,
        lambda_match=0.05,
        lambda_graph=0.05,
        lambda_phase=0.1,
        diversity_sigma=1.0,
        phase_weight_mode="uniform",
        lambda_diff=1.0,
        lambda_ms=None,
        lambda_lsrc=float(args.lambda_lsrc),
        lsrc_mu=1.0,
        lambda_reg=1.0,
        reg_alpha_div=1.0,
        reg_beta_topo=1.0,
        reg_gamma_init=1.0,
        enable_lsrc=True,
        keep_lsrc=True,
        lsrc_k=int(args.lsrc_k),
        lsrc_tau_r=1.0,
        lsrc_tau_c=1.0,
        lsrc_eta=0.5,
        lsrc_beta=0.5,
        lambda_lsrc_cov=0.0,
        lambda_lsrc_rel=0.0,
        lsrc_eps=1e-8,
        lsrc_batch_size=int(args.lsrc_batch_size),
        lsrc_use_global_confidence=False,
        lsrc_coverage_mode="mean",
        lsrc_rel_loss_mode="weight_mean",
        matching_top_k=int(args.matching_top_k),
        matching_candidate_batch_size=128,
        matching_cost_mode=args.matching_cost_mode,
        topology_weight=0.5,
        topology_hop_weight=0.5,
        cost_alpha_diff=0.25,
        cost_beta_wavelet=1.0,
        matching_wavelet_weight=1.0,
        cost_gamma_topo=0.1,
        cost_eta_lsrc=0.1,
        enable_stage2_correction=True,
        enable_stage3_fusion=True,
        enable_stage4_lsrc=True,
        _spectral_embedding=None,
    )


def main():
    parser = argparse.ArgumentParser(description="Select LLaVA instruction subsets with dense_sift_bovw image features and the project's proxy/wavelet sampler.")
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="artifacts/vlm_subset_selection/llava_dense_sift_bovw")
    parser.add_argument("--cache_dir", type=str, default="artifacts/vlm_feature_cache/llava_dense_sift_bovw")
    parser.add_argument("--ratios", type=str, default="1 5 10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_recompute_features", action="store_true", default=False)

    parser.add_argument("--selection_image_size", type=int, default=128)
    parser.add_argument("--bovw_codebook_size", type=int, default=512)
    parser.add_argument("--dense_sift_step", type=int, default=8)
    parser.add_argument("--dense_sift_patch", type=int, default=16)
    parser.add_argument("--bovw_max_fit_descriptors", type=int, default=200000)
    parser.add_argument("--bovw_descriptors_per_image", type=int, default=200)
    parser.add_argument("--text_repr_method", type=str, default="bert", choices=["bert", "tfidf"])
    parser.add_argument("--selection_text_batch_size", type=int, default=256)
    parser.add_argument("--tfidf_ngram_max", type=int, default=2)
    parser.add_argument("--tfidf_stop_words", type=str, default="english")
    parser.add_argument("--tfidf_max_features", type=int, default=20000)
    parser.add_argument("--tfidf_min_df", type=int, default=1)
    parser.add_argument("--tfidf_svd_dim", type=int, default=256)

    parser.add_argument("--knn_k", type=int, default=15)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--proxy_loss_type", type=str, default="wavelet_main")
    parser.add_argument("--proxy_init_method", type=str, default="kmeans")
    parser.add_argument("--proxy_projection_dim", type=int, default=128)
    parser.add_argument("--proxy_lr", type=float, default=0.05)
    parser.add_argument("--proxy_num_steps", type=int, default=200)
    parser.add_argument("--proxy_reg_weight", type=float, default=0.01)
    parser.add_argument("--proxy_batch_size", type=int, default=2048)
    parser.add_argument("--proxy_target_batch_size", type=int, default=2048)
    parser.add_argument("--minibatch_size", type=int, default=2048)
    parser.add_argument("--wavelet_scales", type=str, default="1,2,4")
    parser.add_argument("--wavelet_main_scales", type=str, default="1,2,4")
    parser.add_argument("--wavelet_main_swd_num_projections", type=int, default=64)
    parser.add_argument("--wavelet_cov_weight", type=float, default=0.5)
    parser.add_argument("--wavelet_edge_weight", type=float, default=0.25)
    parser.add_argument("--lambda_lsrc", type=float, default=0.1)
    parser.add_argument("--lsrc_k", type=int, default=32)
    parser.add_argument("--lsrc_batch_size", type=int, default=2048)
    parser.add_argument("--matching_top_k", type=int, default=64)
    parser.add_argument("--matching_cost_mode", type=str, default="candidate_topk")
    parser.add_argument("--use_dpp", action="store_true", default=True)
    parser.add_argument("--disable_dpp", action="store_false", dest="use_dpp")
    args = parser.parse_args()

    records = load_json_or_jsonl(args.annotation_path)
    print(f"[LLaVA selection] Loaded {len(records)} LLaVA records from {args.annotation_path}", flush=True)
    img_features, txt_features, sample_meta, feature_info = load_or_build_features(args, records)
    print("[LLaVA selection] Normalizing image/text features and building unified representation", flush=True)
    img_repr = l2_normalize(img_features.astype(np.float32))
    txt_repr = l2_normalize(txt_features.astype(np.float32))
    representation = np.concatenate([img_repr, txt_repr], axis=1).astype(np.float32)
    unified_graph = build_knn_graph(
        representation,
        k=args.knn_k,
        metric=args.metric,
        desc="Building LLaVA unified kNN graph",
    )
    image_graph = build_knn_graph(
        img_repr,
        k=args.knn_k,
        metric=args.metric,
        desc="Building LLaVA image kNN graph for LSRC",
    )
    text_graph = build_knn_graph(
        txt_repr,
        k=args.knn_k,
        metric=args.metric,
        desc="Building LLaVA text kNN graph for LSRC",
    )

    output_root = Path(args.output_root)
    for ratio in parse_ratios(args.ratios):
        ratio_tag = f"ratio_{int(round(ratio)):02d}"
        output_dir = output_root / ratio_tag / "proxy_opt_lsrc" / f"seed_{int(args.seed)}"
        selected_path = output_dir / "selected_indices.json"
        if selected_path.exists():
            print(f"Skip existing LLaVA dense_sift_bovw selection: {selected_path}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        selection_args = make_selection_args(args, ratio)
        # LSRC needs modality-specific relation graphs. The LLaVA shortcut entry
        # builds them from the same fixed image/text features used for the
        # unified representation, mirroring the main retrieval pipeline.
        selection_args._lsrc_image_graph = image_graph
        selection_args._lsrc_text_graph = text_graph
        selection_args._lsrc_rho_img = 0.5
        selection_args._lsrc_rho_txt = 0.5
        selection_outputs = run_proxy_optimized_selection(selection_args, representation, unified_graph)
        selected_indices = sort_selected_indices(selection_outputs["selected_indices"])
        selected_meta = [sample_meta[int(idx)] for idx in selected_indices]
        save_selected_indices(selected_path, selected_indices)
        write_json(output_dir / "selected_meta.json", selected_meta)
        write_json(
            output_dir / "summary.json",
            {
                "dataset": "llava_instruct_150k",
                "selection_method": "proxy_opt_lsrc",
                "image_feature_mode": "dense_sift_bovw",
                "text_feature_mode": args.text_repr_method,
                "budget_ratio_percent": float(ratio),
                "subset_size": int(len(selected_indices)),
                "num_samples": int(len(records)),
                "seed": int(args.seed),
                "feature_info": feature_info,
                "graph_num_edges": int(unified_graph.nnz),
                "image_graph_num_edges": int(image_graph.nnz),
                "text_graph_num_edges": int(text_graph.nnz),
                "proxy_summary": selection_outputs.get("extra_summary", {}),
            },
        )
        if selection_outputs.get("proxy_bundle") is not None:
            np.save(output_dir / "proxy_points.npy", selection_outputs["proxy_bundle"]["proxy_points"])
        print(f"Saved LLaVA dense_sift_bovw Ours selection: {selected_path}")


if __name__ == "__main__":
    main()
