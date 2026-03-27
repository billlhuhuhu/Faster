import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from src.topology_graph import build_laplacian, build_spectral_embedding, compute_spectrum, parse_multi_scale_ks


def log_cross_modal(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[cross-modal][{timestamp}] {message}", flush=True)


def sanitize_name(name):
    return name.replace("\\", "-").replace("/", "-").replace(" ", "_")


def get_modality_metric(args, modality):
    if modality == "image":
        return getattr(args, "image_metric", None) or args.metric
    if modality == "text":
        return getattr(args, "text_metric", None) or args.metric
    raise ValueError(f"Unsupported modality: {modality}")


def build_graph_dir(args, modality):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    modality_metric = get_modality_metric(args, modality)
    k_list = parse_multi_scale_ks(args.k, getattr(args, "multi_scale_ks", None))
    if len(k_list) == 1:
        graph_tag = f"k{k_list[0]}_{sanitize_name(modality_metric)}"
    else:
        graph_tag = f"ks{'-'.join(str(item) for item in k_list)}_{sanitize_name(modality_metric)}"
    return Path(args.topology_root) / args.dataset / args.split / model_tag / modality / graph_tag


def build_output_dir(args):
    model_tag = f"{sanitize_name(args.image_encoder)}_{sanitize_name(args.text_encoder)}"
    fusion_tag = f"k{args.k}_{sanitize_name(args.metric)}_a{sanitize_name(str(args.alpha))}"
    return Path(args.output_root) / args.dataset / args.split / model_tag / fusion_tag


def load_graph_bundle(graph_dir):
    graph_dir = Path(graph_dir)
    log_cross_modal(f"loading graph bundle from {graph_dir}")
    with open(graph_dir / "summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    graph = sparse.load_npz(graph_dir / "symmetric_graph.npz").tocsr()
    transition = sparse.load_npz(graph_dir / "transition_graph.npz").tocsr()
    sample_meta = json.load(open(graph_dir / "sample_meta.json", "r", encoding="utf-8"))
    return {
        "dir": str(graph_dir),
        "summary": summary,
        "graph": graph,
        "transition": transition,
        "sample_meta": sample_meta,
    }


def validate_modalities(image_bundle, text_bundle):
    image_graph = image_bundle["graph"]
    text_graph = text_bundle["graph"]
    if image_graph.shape != text_graph.shape:
        raise ValueError(f"Image/text graph shape mismatch: {image_graph.shape} vs {text_graph.shape}")
    if len(image_bundle["sample_meta"]) != len(text_bundle["sample_meta"]):
        raise ValueError("Image/text sample_meta length mismatch.")

    image_indices = [item["sample_idx"] for item in image_bundle["sample_meta"]]
    text_indices = [item["sample_idx"] for item in text_bundle["sample_meta"]]
    if image_indices != text_indices:
        raise ValueError("Image/text sample_idx ordering mismatch.")


def choose_healthy_modality(image_summary, text_summary, prefer=None):
    if prefer is not None:
        prefer = prefer.lower()
        if prefer not in {"image", "text"}:
            raise ValueError(f"Invalid preferred modality: {prefer}")
        return prefer

    # Lower collapse_score => healthier. If tied, prefer higher spectral entropy.
    image_score = float(image_summary["collapse_score"])
    text_score = float(text_summary["collapse_score"])
    if image_score < text_score:
        return "image"
    if text_score < image_score:
        return "text"

    image_entropy = float(image_summary["spectral_entropy"])
    text_entropy = float(text_summary["spectral_entropy"])
    if image_entropy >= text_entropy:
        return "image"
    return "text"


def sparse_elementwise_power(matrix, alpha):
    matrix = matrix.tocsr(copy=True)
    matrix.data = np.power(np.clip(matrix.data, 1e-12, None), alpha).astype(np.float32)
    return matrix


def fuzzy_union_symmetrize(graph):
    transpose = graph.transpose().tocsr()
    sym = graph + transpose - graph.multiply(transpose)
    sym = sym.tocsr()
    sym.eliminate_zeros()
    return sym


def row_normalize_graph(graph):
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    degree = np.maximum(degree, 1e-12)
    inv_degree = sparse.diags(1.0 / degree.astype(np.float32))
    transition = inv_degree @ graph
    transition = transition.tocsr()
    transition.eliminate_zeros()
    return transition


def compute_global_confidences(image_score, text_score, tau_g=0.5):
    tau_g = max(float(tau_g), 1e-8)
    logits = np.array([-float(image_score) / tau_g, -float(text_score) / tau_g], dtype=np.float64)
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights), 1e-12)
    return float(weights[0]), float(weights[1])


def scale_sparse_graph(graph, scalar):
    graph = graph.tocsr(copy=True)
    graph.data = (graph.data.astype(np.float32) * float(scalar)).astype(np.float32)
    return graph


def build_bidirectional_correction_weights(image_transition, text_transition, rho_img, rho_txt, eps=1e-8):
    c_img = scale_sparse_graph(image_transition, rho_img)
    c_txt = scale_sparse_graph(text_transition, rho_txt)
    overlap = c_img.multiply(c_txt)

    # Old logic was directional: healthy -> collapsed. New logic uses global confidences
    # to build two edge-wise correction coefficients and updates both modalities.
    numer_txt_to_img = c_txt - overlap
    numer_img_to_txt = c_img - overlap
    numer_txt_to_img.eliminate_zeros()
    numer_img_to_txt.eliminate_zeros()

    alpha_txt_to_img = numer_txt_to_img.tocsr(copy=True)
    alpha_img_to_txt = numer_img_to_txt.tocsr(copy=True)
    alpha_txt_to_img.data = alpha_txt_to_img.data / (alpha_txt_to_img.data + float(eps))
    alpha_img_to_txt.data = alpha_img_to_txt.data / (alpha_img_to_txt.data + float(eps))
    alpha_txt_to_img.eliminate_zeros()
    alpha_img_to_txt.eliminate_zeros()
    return alpha_txt_to_img, alpha_img_to_txt


def apply_bidirectional_correction(image_graph, text_graph, alpha_txt_to_img, alpha_img_to_txt):
    image_graph = image_graph.tocsr()
    text_graph = text_graph.tocsr()

    corrected_image_directed = image_graph + alpha_txt_to_img.multiply(text_graph - image_graph)
    corrected_text_directed = text_graph + alpha_img_to_txt.multiply(image_graph - text_graph)
    corrected_image_directed = corrected_image_directed.tocsr()
    corrected_text_directed = corrected_text_directed.tocsr()
    corrected_image_directed.eliminate_zeros()
    corrected_text_directed.eliminate_zeros()

    corrected_image_symmetric = fuzzy_union_symmetrize(corrected_image_directed)
    corrected_text_symmetric = fuzzy_union_symmetrize(corrected_text_directed)
    corrected_image_symmetric.eliminate_zeros()
    corrected_text_symmetric.eliminate_zeros()
    return (
        corrected_image_directed,
        corrected_image_symmetric,
        corrected_text_directed,
        corrected_text_symmetric,
    )


def unify_topology(image_graph, text_graph, mode="intersection"):
    if mode != "intersection":
        raise ValueError(f"Unsupported fusion mode: {mode}")

    unified = image_graph.multiply(text_graph)
    unified = fuzzy_union_symmetrize(unified)
    unified.eliminate_zeros()
    return unified


def summarize_graph(graph):
    num_nodes = int(graph.shape[0])
    num_edges = int(graph.nnz)
    avg_degree = float(num_edges / max(num_nodes, 1))
    degree = np.asarray(graph.sum(axis=1)).reshape(-1)
    nonzero_degree = degree[degree > 0]
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "min_degree": float(degree.min()) if degree.size else 0.0,
        "max_degree": float(degree.max()) if degree.size else 0.0,
        "mean_nonzero_degree": float(nonzero_degree.mean()) if nonzero_degree.size else 0.0,
        "density": float(num_edges / max(num_nodes * num_nodes, 1)),
    }


def build_unified_spectral_artifacts(
    unified_graph,
    num_eigs=64,
    embedding_dim=32,
    save_eigenvectors=True,
    spectrum_solver_mode="normalized_adjacency_largest",
):
    log_cross_modal(
        f"building unified Laplacian and spectrum: num_nodes={unified_graph.shape[0]}, "
        f"num_edges={unified_graph.nnz}, num_eigs={num_eigs}, embedding_dim={embedding_dim}"
    )
    laplacian = build_laplacian(unified_graph, normalized=True)
    log_cross_modal("computing unified spectrum")
    eigenvalues, eigenvectors = compute_spectrum(
        laplacian,
        num_eigs=num_eigs,
        return_eigenvectors=save_eigenvectors or embedding_dim is not None,
        solver_mode=spectrum_solver_mode,
    )
    log_cross_modal("building unified spectral embedding")
    spectral_embedding = build_spectral_embedding(
        eigenvalues,
        eigenvectors,
        embedding_dim=embedding_dim,
    )
    return {
        "laplacian_sym": laplacian,
        "eigvals": eigenvalues,
        "eigvecs": eigenvectors,
        "spectral_embedding": spectral_embedding,
    }


def build_summary(
    args,
    healthy_modality,
    image_bundle,
    text_bundle,
    rho_img,
    rho_txt,
    corrected_image_summary,
    corrected_text_summary,
    corrected_summary,
    unified_summary,
    unified_spectral_artifacts,
):
    return {
        "dataset": args.dataset,
        "split": args.split,
        "image_encoder": args.image_encoder,
        "text_encoder": args.text_encoder,
        "metric": args.metric,
        "image_metric": get_modality_metric(args, "image"),
        "text_metric": get_modality_metric(args, "text"),
        "k": int(args.k),
        "alpha": float(args.alpha),
        "tau_g": float(getattr(args, "tau_g", 0.5)),
        "correction_eps": float(getattr(args, "correction_eps", 1e-8)),
        "fusion_mode": args.fusion_mode,
        "healthy_modality": healthy_modality,
        "collapsed_modality": "text" if healthy_modality == "image" else "image",
        "rho_img": float(rho_img),
        "rho_txt": float(rho_txt),
        "image_summary": image_bundle["summary"],
        "text_summary": text_bundle["summary"],
        "corrected_image_summary": corrected_image_summary,
        "corrected_text_summary": corrected_text_summary,
        "corrected_summary": corrected_summary,
        "unified_summary": unified_summary,
        "unified_first_eigenvalues": [
            float(x) for x in unified_spectral_artifacts["eigvals"][: min(10, len(unified_spectral_artifacts["eigvals"]))]
        ],
        "unified_embedding_dim": int(unified_spectral_artifacts["spectral_embedding"].shape[1]) if unified_spectral_artifacts["spectral_embedding"] is not None else 0,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_cross_modal_outputs(
    output_dir,
    healthy_modality,
    image_bundle,
    text_bundle,
    corrected_image_directed,
    corrected_image_symmetric,
    corrected_text_directed,
    corrected_text_symmetric,
    unified_graph,
    unified_spectral_artifacts,
    summary,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_cross_modal(f"saving cross-modal artifacts to {output_dir}")

    sparse.save_npz(output_dir / "B_I_or_health.npz", image_bundle["graph"] if healthy_modality == "image" else text_bundle["graph"])
    sparse.save_npz(output_dir / "B_collapsed_raw.npz", text_bundle["graph"] if healthy_modality == "image" else image_bundle["graph"])
    sparse.save_npz(output_dir / "healthy_graph.npz", image_bundle["graph"] if healthy_modality == "image" else text_bundle["graph"])
    sparse.save_npz(output_dir / "healthy_transition.npz", image_bundle["transition"] if healthy_modality == "image" else text_bundle["transition"])
    sparse.save_npz(output_dir / "collapsed_graph.npz", text_bundle["graph"] if healthy_modality == "image" else image_bundle["graph"])

    sparse.save_npz(output_dir / "corrected_image_graph_directed.npz", corrected_image_directed)
    sparse.save_npz(output_dir / "corrected_image_graph_symmetric.npz", corrected_image_symmetric)
    sparse.save_npz(output_dir / "corrected_text_graph_directed.npz", corrected_text_directed)
    sparse.save_npz(output_dir / "corrected_text_graph_symmetric.npz", corrected_text_symmetric)
    sparse.save_npz(
        output_dir / "corrected_graph_directed.npz",
        corrected_text_directed if healthy_modality == "image" else corrected_image_directed,
    )
    sparse.save_npz(
        output_dir / "corrected_graph_symmetric.npz",
        corrected_text_symmetric if healthy_modality == "image" else corrected_image_symmetric,
    )
    sparse.save_npz(output_dir / "unified_graph.npz", unified_graph)
    sparse.save_npz(output_dir / "B_star.npz", unified_graph)
    sparse.save_npz(output_dir / "unified_transition.npz", row_normalize_graph(unified_graph))
    sparse.save_npz(output_dir / "unified_laplacian_sym.npz", unified_spectral_artifacts["laplacian_sym"])
    sparse.save_npz(output_dir / "L_star.npz", unified_spectral_artifacts["laplacian_sym"])

    np.save(output_dir / "unified_first_eigvals.npy", unified_spectral_artifacts["eigvals"])
    if unified_spectral_artifacts["eigvecs"] is not None:
        np.save(output_dir / "unified_eigvecs.npy", unified_spectral_artifacts["eigvecs"])
    if unified_spectral_artifacts["spectral_embedding"] is not None:
        np.save(output_dir / "unified_spectral_embedding.npy", unified_spectral_artifacts["spectral_embedding"])
        np.save(output_dir / "V_full_multi.npy", unified_spectral_artifacts["spectral_embedding"])

    with open(output_dir / "sample_meta.json", "w", encoding="utf-8") as handle:
        json.dump(image_bundle["sample_meta"], handle, ensure_ascii=False, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    selection = {
        "healthy_modality": healthy_modality,
        "collapsed_modality": "text" if healthy_modality == "image" else "image",
        "healthy_graph_dir": image_bundle["dir"] if healthy_modality == "image" else text_bundle["dir"],
        "collapsed_graph_dir": text_bundle["dir"] if healthy_modality == "image" else image_bundle["dir"],
    }
    with open(output_dir / "modality_selection.json", "w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)


def run_cross_modal_topology(args):
    log_cross_modal(
        f"start cross-modal topology: dataset={args.dataset}, split={args.split}, "
        f"image_metric={get_modality_metric(args, 'image')}, text_metric={get_modality_metric(args, 'text')}, "
        f"k={args.k}, alpha={args.alpha}"
    )
    image_bundle = load_graph_bundle(build_graph_dir(args, "image"))
    text_bundle = load_graph_bundle(build_graph_dir(args, "text"))
    log_cross_modal("validating image/text graph compatibility")
    validate_modalities(image_bundle, text_bundle)

    log_cross_modal("choosing healthy modality")
    healthy_modality = choose_healthy_modality(
        image_bundle["summary"],
        text_bundle["summary"],
        prefer=args.prefer_healthy_modality,
    )
    log_cross_modal(f"healthy modality selected for reporting only: {healthy_modality}")

    log_cross_modal("computing global modality confidences")
    rho_img, rho_txt = compute_global_confidences(
        image_bundle["summary"]["collapse_score"],
        text_bundle["summary"]["collapse_score"],
        tau_g=getattr(args, "tau_g", 0.5),
    )
    log_cross_modal(f"global confidences: rho_img={rho_img:.4f}, rho_txt={rho_txt:.4f}")

    log_cross_modal("applying bidirectional graph correction")
    alpha_txt_to_img, alpha_img_to_txt = build_bidirectional_correction_weights(
        image_bundle["transition"],
        text_bundle["transition"],
        rho_img=rho_img,
        rho_txt=rho_txt,
        eps=getattr(args, "correction_eps", 1e-8),
    )
    (
        corrected_image_directed,
        corrected_image_symmetric,
        corrected_text_directed,
        corrected_text_symmetric,
    ) = apply_bidirectional_correction(
        image_bundle["graph"],
        text_bundle["graph"],
        alpha_txt_to_img,
        alpha_img_to_txt,
    )

    log_cross_modal("building unified topology B*")
    unified_graph = unify_topology(
        corrected_image_symmetric,
        corrected_text_symmetric,
        mode=args.fusion_mode,
    )

    log_cross_modal("summarizing corrected and unified graphs")
    corrected_image_summary = summarize_graph(corrected_image_symmetric)
    corrected_text_summary = summarize_graph(corrected_text_symmetric)
    corrected_summary = corrected_text_summary if healthy_modality == "image" else corrected_image_summary
    unified_summary = summarize_graph(unified_graph)
    unified_spectral_artifacts = build_unified_spectral_artifacts(
        unified_graph,
        num_eigs=getattr(args, "num_eigs", 64),
        embedding_dim=getattr(args, "spectral_embedding_dim", 32),
        save_eigenvectors=bool(getattr(args, "save_eigenvectors", False)),
        spectrum_solver_mode=getattr(args, "spectrum_solver_mode", "normalized_adjacency_largest"),
    )
    summary = build_summary(
        args,
        healthy_modality,
        image_bundle,
        text_bundle,
        rho_img,
        rho_txt,
        corrected_image_summary,
        corrected_text_summary,
        corrected_summary,
        unified_summary,
        unified_spectral_artifacts,
    )

    output_dir = build_output_dir(args)
    save_cross_modal_outputs(
        output_dir,
        healthy_modality,
        image_bundle,
        text_bundle,
        corrected_image_directed,
        corrected_image_symmetric,
        corrected_text_directed,
        corrected_text_symmetric,
        unified_graph,
        unified_spectral_artifacts,
        summary,
    )
    log_cross_modal("cross-modal topology completed")

    return {
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "summary": summary,
    }
