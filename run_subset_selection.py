import os
import argparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from src.subset_match import run_subset_selection


def build_parser():
    parser = argparse.ArgumentParser(description="Select a real subset from unified representation/topology.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--image_encoder", type=str, required=True)
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--feature_cache_root", type=str, default="artifacts/feature_cache")
    parser.add_argument("--cross_modal_root", type=str, default="artifacts/cross_modal_topology")
    parser.add_argument("--output_root", type=str, default="artifacts/subset_selection")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=1.0)
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument("--budget_ratio", type=float, default=None)
    budget_group.add_argument("--budget_size", type=int, default=None)
    parser.add_argument("--representation_mode", type=str, default="concat", choices=["concat"])
    parser.add_argument("--reference_embedding_mode", type=str, default="hybrid", choices=["concat", "spectral", "hybrid"])
    parser.add_argument("--spectral_weight", type=float, default=1.0)
    parser.add_argument("--selection_method", type=str, default="proxy_opt", choices=["baseline", "proxy_opt"])
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "minibatch_kmeans"])
    parser.add_argument("--degree_weight", type=float, default=0.1, help="Tie-breaking weight from unified graph degree.")
    parser.add_argument("--geometry_weight", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--minibatch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--proxy_projection_dim", type=int, default=128)
    parser.add_argument("--proxy_init_method", type=str, default="kmeans", choices=["kmeans", "minibatch_kmeans", "sample"])
    parser.add_argument("--proxy_objective_mode", type=str, default="pd_cfd", choices=["cfd", "pd_cfd"])
    parser.add_argument("--use_pdcfd", action="store_true")
    parser.add_argument("--proxy_num_frequencies", type=int, default=64)
    parser.add_argument("--proxy_frequency_scale", type=float, default=1.0)
    parser.add_argument("--proxy_lr", type=float, default=0.05)
    parser.add_argument("--proxy_num_steps", type=int, default=200)
    parser.add_argument("--proxy_reg_weight", type=float, default=0.01)
    parser.add_argument("--proxy_target_batch_size", type=int, default=4096)
    parser.add_argument("--proxy_batch_size", type=int, default=4096)
    parser.add_argument("--use_pdas", action="store_true")
    parser.add_argument("--pdas_num_stages", type=int, default=4)
    parser.add_argument("--pdas_schedule_mode", type=str, default="low_to_high", choices=["low_to_high", "uniform"])
    parser.add_argument("--num_freq_pool", type=int, default=256)
    parser.add_argument("--tau_min", type=float, default=0.1)
    parser.add_argument("--tau_max", type=float, default=1.0)
    parser.add_argument("--use_dpp", action="store_true")
    parser.add_argument("--lambda_div", type=float, default=0.01)
    parser.add_argument("--lambda_match", type=float, default=0.05)
    parser.add_argument("--lambda_graph", type=float, default=0.05)
    parser.add_argument("--lambda_phase", type=float, default=0.1)
    parser.add_argument("--diversity_sigma", type=float, default=1.0)
    parser.add_argument("--phase_weight_mode", type=str, default="uniform", choices=["uniform", "linear"])
    parser.add_argument("--matching_top_k", type=int, default=64)
    parser.add_argument("--matching_candidate_batch_size", type=int, default=128)
    parser.add_argument("--matching_cost_mode", type=str, default="candidate_topk", choices=["candidate_topk", "degree_aware_global"])
    parser.add_argument("--topology_weight", type=float, default=0.5)
    parser.add_argument("--topology_hop_weight", type=float, default=0.5)
    return parser


def main():
    args = build_parser().parse_args()
    if args.use_pdcfd:
        args.proxy_objective_mode = "pd_cfd"
    outputs = run_subset_selection(args)
    print("Subset selection finished:")
    print(f"  output_dir: {outputs['output_dir']}")
    print(f"  subset_size: {outputs['subset_size']}")
    print(f"  selected_indices_path: {outputs['saved']['selected_indices']}")
    print(f"  selected_meta_path: {outputs['saved']['selected_meta']}")
    print(f"  summary_path: {outputs['saved']['summary']}")
    if "proxy_points" in outputs["saved"]:
        print(f"  proxy_points_path: {outputs['saved']['proxy_points']}")
    if "matching_cost" in outputs["saved"]:
        print(f"  matching_cost_path: {outputs['saved']['matching_cost']}")
    print(f"  first_selected_indices: {outputs['selected_indices'][:10]}")


if __name__ == "__main__":
    main()
