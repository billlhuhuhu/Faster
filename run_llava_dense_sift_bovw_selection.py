import argparse
import json
import os
import shutil
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
from run_cross_modal_topology import build_parser as build_cross_modal_parser
from run_subset_selection import build_parser as build_subset_parser
from run_topology_graph import build_parser as build_topology_parser
from run_vlm_finetune import extract_llava_turn, load_json_or_jsonl, resolve_image_path
from src.cross_modal_topology import run_cross_modal_topology
from src.fixed_image_features import extract_fixed_image_features
from src.fixed_text_features import extract_fixed_text_features
from src.proxy_optimization import l2_normalize
from src.subset_match import run_proxy_optimized_selection, run_subset_selection, sort_selected_indices
from src.topology_graph import run_topology_graph


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_log(message):
    print(f"[LLaVA selection] {message}", flush=True)


def maybe_init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        if torch.cuda.is_available() and os.environ.get("LOCAL_RANK") is not None:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
    return True


def distributed_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


def is_rank0():
    rank, _ = distributed_info()
    return rank == 0


def distributed_barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def broadcast_skip_flag(skip):
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return bool(skip)
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}") if torch.cuda.is_available() else torch.device("cpu")
    flag = torch.tensor([1 if bool(skip) else 0], dtype=torch.int64, device=device)
    torch.distributed.broadcast(flag, src=0)
    return bool(flag.item())


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


def sanitize_component(value):
    return str(value).replace("\\", "-").replace("/", "-").replace(" ", "_")


def ratio_tag_from_percent(ratio):
    return f"ratio_{int(round(float(ratio))):02d}"


def hardlink_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if src.resolve() == dst.resolve():
                return
        except OSError:
            pass
        if dst.stat().st_size == src.stat().st_size:
            return
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class TeeLogger:
    def __init__(self, path, prefix=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.handle = None

    def __enter__(self):
        import sys

        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.handle = self.path.open("a", encoding="utf-8")
        self.write(f"\n===== stage log start: {self.path} =====\n")
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc, tb):
        import sys

        self.write(f"\n===== stage log end: {self.path} =====\n")
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.handle.close()
        self.handle = None
        return False

    def write(self, data):
        if self.handle is not None:
            self.handle.write(data)
            self.handle.flush()
        self.stdout.write(data)
        self.stdout.flush()

    def flush(self):
        if self.handle is not None:
            self.handle.flush()
        self.stdout.flush()


def run_with_stage_log(log_dir, stage_name, func, *args, **kwargs):
    if not log_dir:
        return func(*args, **kwargs)
    rank, world_size = distributed_info()
    suffix = "" if world_size <= 1 or rank == 0 else f"_rank{rank}"
    log_path = Path(log_dir) / f"{stage_name}{suffix}.log"
    stage_log(f"{stage_name}: log -> {log_path}")
    with TeeLogger(log_path):
        return func(*args, **kwargs)


def ensure_structured_feature_cache(args):
    flat_cache_dir = Path(args.cache_dir)
    model_tag = f"{sanitize_component('dense_sift_bovw')}_{sanitize_component(args.text_repr_method)}"
    structured_dir = Path(args.feature_cache_root) / args.pipeline_dataset_name / "train" / model_tag
    for filename in ["img_features_selection.pt", "txt_features_selection.pt", "sample_meta.json", "feature_info.json"]:
        src = flat_cache_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing flat LLaVA feature cache file: {src}")
        hardlink_or_copy(src, structured_dir / filename)
    print(f"[LLaVA selection] Structured feature cache ready: {structured_dir}", flush=True)
    return structured_dir


def topology_output_dir(args, modality, metric):
    model_tag = f"{sanitize_component('dense_sift_bovw')}_{sanitize_component(args.text_repr_method)}"
    return (
        Path(args.topology_root)
        / args.pipeline_dataset_name
        / "train"
        / model_tag
        / modality
        / f"k{int(args.knn_k)}_{sanitize_component(metric)}"
    )


def cross_modal_output_dir(args):
    model_tag = f"{sanitize_component('dense_sift_bovw')}_{sanitize_component(args.text_repr_method)}"
    return (
        Path(args.cross_modal_root)
        / args.pipeline_dataset_name
        / "train"
        / model_tag
        / f"k{int(args.knn_k)}_{sanitize_component(args.image_metric)}_a1.0"
    )


def legacy_selection_output_dir(args, ratio):
    return Path(args.output_root) / ratio_tag_from_percent(ratio) / "proxy_opt_lsrc" / f"seed_{int(args.seed)}"


def is_current_full_pipeline_selection(output_dir):
    summary_path = Path(output_dir) / "summary.json"
    selected_path = Path(output_dir) / "selected_indices.json"
    if not summary_path.exists() or not selected_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("selection_pipeline") == "full_cross_modal_topology"


def build_topology_args(args, modality, metric):
    parser = build_topology_parser()
    return parser.parse_args(
        [
            "--dataset", args.pipeline_dataset_name,
            "--split", "train",
            "--image_encoder", "dense_sift_bovw",
            "--text_encoder", args.text_repr_method,
            "--modality", modality,
            "--feature_cache_root", args.feature_cache_root,
            "--output_root", args.topology_root,
            "--metric", metric,
            "--knn_k", str(int(args.knn_k)),
            "--graph_reduce_method", args.topology_graph_reduce_method,
            "--graph_feature_dim", str(int(args.topology_graph_feature_dim)),
            "--num_eigs", "32",
            "--spectral_embedding_dim", "32",
            "--n_jobs", str(int(args.topology_n_jobs)),
            "--knn_backend", args.topology_knn_backend,
            "--mst_weight_scale", "1.0",
            "--random_state", str(int(args.seed)),
        ]
    )


def build_cross_modal_args(args):
    parser = build_cross_modal_parser()
    return parser.parse_args(
        [
            "--dataset", args.pipeline_dataset_name,
            "--split", "train",
            "--image_encoder", "dense_sift_bovw",
            "--text_encoder", args.text_repr_method,
            "--topology_root", args.topology_root,
            "--output_root", args.cross_modal_root,
            "--metric", args.image_metric,
            "--image_metric", args.image_metric,
            "--text_metric", args.text_metric,
            "--k", str(int(args.knn_k)),
            "--alpha", "1.0",
            "--correction_mode", "bidirectional",
            "--correction_score_mode", "collapse_score",
            "--collapse_score_mode", "edge_plus_neighborhood",
            "--collapse_neighbor_topk", str(int(args.collapse_neighbor_topk)),
            "--local_relation_alpha", str(float(args.local_relation_alpha)),
            "--asymmetric_correction_lambda", str(float(args.asymmetric_correction_lambda)),
            "--correction_confidence_gap_delta", str(float(args.correction_confidence_gap_delta)),
            "--corrected_image_added_topk", str(int(args.corrected_image_added_topk)),
            "--fusion_domain_mode", "wavelet_latent",
            "--wavelet_fusion_weight_mode", args.wavelet_fusion_weight_mode,
            "--wavelet_fusion_entropy_temperature", str(float(args.wavelet_fusion_entropy_temperature)),
            "--wavelet_fusion_scales", args.wavelet_scales,
            "--wavelet_fusion_probe_dim", str(int(args.wavelet_fusion_probe_dim)),
            "--wavelet_latent_postprocess_topk", str(int(args.wavelet_latent_postprocess_topk)),
            "--num_eigs", "64",
            "--spectral_embedding_dim", "32",
            "--embedding_type", "diffusion",
        ]
    )


def build_subset_args(args, ratio):
    parser = build_subset_parser()
    argv = [
        "--dataset", args.pipeline_dataset_name,
        "--split", "train",
        "--image_encoder", "dense_sift_bovw",
        "--text_encoder", args.text_repr_method,
        "--feature_cache_root", args.feature_cache_root,
        "--cross_modal_root", args.cross_modal_root,
        "--output_root", args.pipeline_selection_output_root,
        "--metric", args.image_metric,
        "--k", str(int(args.knn_k)),
        "--alpha", "1.0",
        "--budget_ratio", f"{float(ratio) / 100.0:.8f}",
        "--selection_method", "proxy_opt",
        "--reference_embedding_mode", "hybrid",
        "--spectral_weight", "1.0",
        "--random_state", str(int(args.seed)),
        "--device", args.device,
        "--proxy_projection_dim", str(int(args.proxy_projection_dim)),
        "--proxy_init_method", args.proxy_init_method,
        "--proxy_loss_type", args.proxy_loss_type,
        "--proxy_lr", str(float(args.proxy_lr)),
        "--proxy_num_steps", str(int(args.proxy_num_steps)),
        "--proxy_reg_weight", str(float(args.proxy_reg_weight)),
        "--proxy_target_batch_size", str(int(args.proxy_target_batch_size)),
        "--proxy_batch_size", str(int(args.proxy_batch_size)),
        "--use_wavelet_multiscale",
        "--wavelet_scales", args.wavelet_scales,
        "--wavelet_distance_type", "swd",
        "--wavelet_schedule", "coarse_to_fine",
        "--lambda_main", "1.0",
        "--wavelet_main_scales", args.wavelet_main_scales,
        "--wavelet_main_swd_num_projections", str(int(args.wavelet_main_swd_num_projections)),
        "--wavelet_cov_weight", str(float(args.wavelet_cov_weight)),
        "--wavelet_edge_weight", str(float(args.wavelet_edge_weight)),
        "--wavelet_curriculum_schedule", "coarse_to_fine",
        "--lambda_lsrc", str(float(args.lambda_lsrc)),
        "--lambda_reg", "1.0",
        "--reg_beta_topo", "1.0",
        "--reg_gamma_init", "1.0",
        "--enable_lsrc",
        "--keep_lsrc",
        "--lsrc_k", str(int(args.lsrc_k)),
        "--lsrc_tau_r", "1.0",
        "--lsrc_tau_c", "1.0",
        "--lsrc_eta", "0.5",
        "--lsrc_beta", "0.5",
        "--lsrc_batch_size", str(int(args.lsrc_batch_size)),
        "--lsrc_coverage_mode", "mean",
        "--lsrc_rel_loss_mode", "weight_mean",
        "--matching_top_k", str(int(args.matching_top_k)),
        "--matching_cost_mode", args.matching_cost_mode,
        "--cost_alpha_diff", "0.25",
        "--cost_beta_wavelet", "1.0",
        "--matching_wavelet_weight", "1.0",
        "--cost_gamma_topo", "0.1",
        "--cost_eta_lsrc", "0.1",
        "--geometry_weight", "1.0",
        "--diversity_sigma", "1.0",
    ]
    if args.use_dpp:
        argv.extend(["--use_dpp", "--lambda_div", "0.01", "--reg_alpha_div", "1.0"])
    else:
        argv.extend(["--lambda_div", "0.0", "--reg_alpha_div", "0.0"])
    return parser.parse_args(argv)


def run_full_cross_modal_pipeline(args, ratio):
    ratio_tag = ratio_tag_from_percent(ratio)
    log_dir = Path(args.log_dir) / ratio_tag if args.log_dir else None
    if is_rank0():
        run_with_stage_log(log_dir, "stage0_structured_feature_cache", ensure_structured_feature_cache, args)
    distributed_barrier()

    image_topology_dir = topology_output_dir(args, "image", args.image_metric)
    if is_rank0() and (args.force_recompute_topology or not (image_topology_dir / "summary.json").exists()):
        stage_log("Stage A1: build image topology graph")
        run_with_stage_log(
            log_dir,
            "stage1_image_topology",
            run_topology_graph,
            build_topology_args(args, "image", args.image_metric),
        )
    elif is_rank0():
        stage_log(f"Skip image topology: {image_topology_dir}")
    distributed_barrier()

    text_topology_dir = topology_output_dir(args, "text", args.text_metric)
    if is_rank0() and (args.force_recompute_topology or not (text_topology_dir / "summary.json").exists()):
        stage_log("Stage A2: build text topology graph")
        run_with_stage_log(
            log_dir,
            "stage2_text_topology",
            run_topology_graph,
            build_topology_args(args, "text", args.text_metric),
        )
    elif is_rank0():
        stage_log(f"Skip text topology: {text_topology_dir}")
    distributed_barrier()

    cross_dir = cross_modal_output_dir(args)
    if is_rank0() and (args.force_recompute_cross_modal or not (cross_dir / "summary.json").exists()):
        stage_log("Stage B: run stage2 correction + stage3 wavelet-latent fusion")
        run_with_stage_log(
            log_dir,
            "stage3_cross_modal_correction_fusion",
            run_cross_modal_topology,
            build_cross_modal_args(args),
        )
    elif is_rank0():
        stage_log(f"Skip cross-modal topology: {cross_dir}")
    distributed_barrier()

    if is_rank0():
        stage_log(f"Stage C: wavelet_main subset selection ratio={ratio}%")
    outputs = run_with_stage_log(
        log_dir,
        "stage4_wavelet_main_selection",
        run_subset_selection,
        build_subset_args(args, ratio),
    )
    distributed_barrier()
    return outputs, cross_dir


def sync_selection_to_legacy_layout(args, ratio, outputs, cross_dir):
    legacy_dir = legacy_selection_output_dir(args, ratio)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    saved = outputs["saved"]
    for key, filename in [
        ("selected_indices", "selected_indices.json"),
        ("selected_meta", "selected_meta.json"),
        ("summary", "summary.json"),
    ]:
        src = Path(saved[key])
        dst = legacy_dir / filename
        shutil.copy2(src, dst)
    summary_path = legacy_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "selection_pipeline": "full_cross_modal_topology",
            "selection_pipeline_reference": "run_wavelet_main_dense_sift_bovw_combo",
            "stage2_correction": True,
            "stage3_wavelet_latent_fusion": True,
            "stage4_wavelet_main_lsrc": True,
            "cross_modal_dir": str(cross_dir),
            "pipeline_selection_output_dir": outputs["output_dir"],
        }
    )
    write_json(summary_path, summary)
    if "proxy_points" in saved:
        shutil.copy2(saved["proxy_points"], legacy_dir / "proxy_points.pt")
    print(f"[LLaVA selection] Synced full-pipeline Ours selection to {legacy_dir}", flush=True)
    return legacy_dir


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
    maybe_init_distributed()
    rank, world_size = distributed_info()
    parser = argparse.ArgumentParser(description="Select LLaVA instruction subsets with dense_sift_bovw image features and the project's proxy/wavelet sampler.")
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="artifacts/vlm_subset_selection/llava_dense_sift_bovw")
    parser.add_argument("--cache_dir", type=str, default="artifacts/vlm_feature_cache/llava_dense_sift_bovw")
    parser.add_argument("--feature_cache_root", type=str, default="artifacts/vlm_feature_cache_llava_dense_sift_bovw_full_pipeline")
    parser.add_argument("--topology_root", type=str, default="artifacts/vlm_topology_graph_dense_sift_bovw")
    parser.add_argument("--cross_modal_root", type=str, default="artifacts/vlm_cross_modal_topology_dense_sift_bovw")
    parser.add_argument("--pipeline_selection_output_root", type=str, default="artifacts/vlm_subset_selection_dense_sift_bovw_full_pipeline")
    parser.add_argument("--pipeline_dataset_name", type=str, default="llava_instruct_150k")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--ratios", type=str, default="1 5 10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_recompute_features", action="store_true", default=False)
    parser.add_argument("--force_recompute_topology", action="store_true", default=False)
    parser.add_argument("--force_recompute_cross_modal", action="store_true", default=False)
    parser.add_argument("--force_recompute_selection", action="store_true", default=False)

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
    parser.add_argument("--metric", type=str, default="euclidean", choices=["cosine", "euclidean"])
    parser.add_argument("--image_metric", type=str, default="euclidean", choices=["cosine", "euclidean"])
    parser.add_argument("--text_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--topology_graph_reduce_method", type=str, default="pca", choices=["none", "pca", "random_projection"])
    parser.add_argument("--topology_graph_feature_dim", type=int, default=256)
    parser.add_argument("--topology_n_jobs", type=int, default=32)
    parser.add_argument("--topology_knn_backend", type=str, default="auto", choices=["auto", "sklearn", "faiss"])
    parser.add_argument("--collapse_neighbor_topk", type=int, default=15)
    parser.add_argument("--local_relation_alpha", type=float, default=0.5)
    parser.add_argument("--asymmetric_correction_lambda", type=float, default=0.3)
    parser.add_argument("--correction_confidence_gap_delta", type=float, default=0.1)
    parser.add_argument("--corrected_image_added_topk", type=int, default=5)
    parser.add_argument("--wavelet_fusion_weight_mode", type=str, default="collapse_aware", choices=["fixed_per_scale", "collapse_aware"])
    parser.add_argument("--wavelet_fusion_entropy_temperature", type=float, default=1.0)
    parser.add_argument("--wavelet_fusion_probe_dim", type=int, default=32)
    parser.add_argument("--wavelet_latent_postprocess_topk", type=int, default=64)
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

    if world_size > 1 and rank == 0:
        stage_log(f"Distributed selection enabled: world_size={world_size}")

    if is_rank0():
        records = load_json_or_jsonl(args.annotation_path)
        print(f"[LLaVA selection] Loaded {len(records)} LLaVA records from {args.annotation_path}", flush=True)
        run_with_stage_log(args.log_dir, "stage0_load_or_build_features", load_or_build_features, args, records)
    distributed_barrier()

    output_root = Path(args.output_root)
    for ratio in parse_ratios(args.ratios):
        ratio_tag = f"ratio_{int(round(ratio)):02d}"
        output_dir = output_root / ratio_tag / "proxy_opt_lsrc" / f"seed_{int(args.seed)}"
        selected_path = output_dir / "selected_indices.json"
        skip_existing = False
        if is_rank0() and selected_path.exists() and not args.force_recompute_selection and is_current_full_pipeline_selection(output_dir):
            print(f"Skip existing full-pipeline LLaVA dense_sift_bovw selection: {selected_path}", flush=True)
            skip_existing = True
        if broadcast_skip_flag(skip_existing):
            continue
        if is_rank0() and selected_path.exists() and not is_current_full_pipeline_selection(output_dir):
            print(
                f"[LLaVA selection] Existing selection is not marked as full pipeline; recomputing: {selected_path}",
                flush=True,
            )
        outputs, cross_dir = run_full_cross_modal_pipeline(args, ratio)
        if is_rank0():
            legacy_dir = sync_selection_to_legacy_layout(args, ratio, outputs, cross_dir)
            print(f"Saved full-pipeline LLaVA dense_sift_bovw Ours selection: {legacy_dir / 'selected_indices.json'}", flush=True)
        distributed_barrier()


if __name__ == "__main__":
    main()
