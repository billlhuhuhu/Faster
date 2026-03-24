import argparse

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
    parser.add_argument("--num_eigs", type=int, default=32)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--knn_backend", type=str, default="auto", choices=["auto", "sklearn", "faiss"])
    parser.add_argument("--faiss_use_gpu", action="store_true")
    parser.add_argument("--pre_knn_method", type=str, default="none", choices=["none", "pca", "random_projection"])
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
