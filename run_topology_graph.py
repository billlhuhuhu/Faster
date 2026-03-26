import os
import argparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from src.topology_graph import run_topology_graph


def build_parser():
    parser = argparse.ArgumentParser(description="Build single-modality topology graphs from cached full-train features.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--image_encoder", type=str, required=True)
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--modality", type=str, required=True, choices=["image", "text"])
    parser.add_argument("--feature_cache_root", type=str, default="artifacts/feature_cache")
    parser.add_argument("--output_root", type=str, default="artifacts/topology_graph")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--knn_k", type=int, default=None)
    parser.add_argument("--multi_scale_ks", type=str, default=None, help="Comma-separated k list, e.g. 10,15,30")
    parser.add_argument("--multiscale_ks", type=str, default=None)
    parser.add_argument("--multi_scale_merge_mode", type=str, default="union", choices=["mean", "max", "union"])
    parser.add_argument("--use_mst_connectivity", action="store_true")
    parser.add_argument("--use_mst", action="store_true")
    parser.add_argument("--mst_weight_scale", type=float, default=1.0)
    parser.add_argument("--num_eigs", type=int, default=32)
    parser.add_argument("--spectral_embedding_dim", type=int, default=32)
    parser.add_argument("--spectrum_solver_mode", type=str, default="normalized_adjacency_largest", choices=["normalized_adjacency_largest", "laplacian_smallest"])
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--knn_backend", type=str, default="auto", choices=["auto", "sklearn", "faiss"])
    parser.add_argument("--faiss_use_gpu", action="store_true")
    parser.add_argument("--graph_reduce_method", type=str, default="pca", choices=["none", "pca", "random_projection"])
    parser.add_argument("--graph_feature_dim", type=int, default=256)
    parser.add_argument("--pre_knn_method", type=str, default=None, choices=["none", "pca", "random_projection"])
    parser.add_argument("--pre_knn_dim", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--local_connectivity", type=float, default=1.0)
    parser.add_argument("--bandwidth", type=float, default=None, help="If set, use this target bandwidth instead of log2(k+1).")
    parser.add_argument("--sigma_search_steps", type=int, default=64)
    parser.add_argument("--save_eigenvectors", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional smoke-test cap.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.knn_k is not None:
        args.k = args.knn_k
    if args.multiscale_ks is not None and args.multi_scale_ks is None:
        args.multi_scale_ks = args.multiscale_ks
    if args.use_mst:
        args.use_mst_connectivity = True
    outputs = run_topology_graph(args)
    print("Topology graph saved:")
    print(f"  output_dir: {outputs['output_dir']}")
    print(f"  summary_path: {outputs['summary_path']}")
    summary = outputs["summary"]
    print(f"  num_nodes: {summary['num_nodes']}")
    print(f"  num_edges: {summary['num_edges']}")
    print(f"  avg_degree: {summary['avg_degree']:.4f}")
    print(f"  first_eigenvalues: {summary['first_eigenvalues']}")
    print(f"  collapse_score: {summary['collapse_score']:.6f}")
    print(f"  spectral_entropy: {summary['spectral_entropy']:.6f}")


if __name__ == "__main__":
    main()
